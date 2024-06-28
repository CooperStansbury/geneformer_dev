import sys
import os
import argparse
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
import scanpy as sc
import anndata as an
from datasets import Dataset

DEFAULT_NAME_PATH = "/nfs/turbo/umms-indikar/shared/projects/geneformer/gene_names.pkl"
DEFAULT_TOKEN_PATH = "/nfs/turbo/umms-indikar/shared/projects/geneformer/token_dictionary.pkl"
DEFAULT_MEDIAN_PATH = "/nfs/turbo/umms-indikar/shared/projects/geneformer/geneformer/gene_median_dictionary.pkl"
MODEL_INPUT_SIZE = 2048
NUMBER_PROC = 16


def check_counts_column(adata, counts_column):
    """Checks for and calculates a total counts column in AnnData.

    This function examines the AnnData object's observation (`obs`) columns for the specified 
    `counts_column`. If it doesn't exist, the function calculates the sum of each row (cell) 
    across all features in the data matrix (`X`) and stores it as a new column in `obs`.

    Args:
        adata: An AnnData object containing the data to be analyzed.
        counts_column: A string representing the desired name for the total counts column.

    Returns:
        adata: The modified AnnData object, now with the `counts_column` present (either 
               pre-existing or newly calculated).
    """
    obs_columns = adata.obs.columns
    
    if counts_column in obs_columns:
        return adata
    else:
        adata.obs[counts_column] = adata.X.sum(axis=1)
        return adata
    
    
def map_gene_names(adata, gene_id, gene_name_column, gene_names):
    """A function mapping gene names to gene ids """
    var_columns = adata.var.columns
    
    if gene_id in var_columns:
        return adata
    else:
        adata.var[gene_id] = adata.var[gene_name_column].map(gene_names)
        return adata
    
    
def load_gene_names(gene_names_file):
    """
    Loads a gene median dictionary from a pickle file.

    Args:
        gene_names_file (str): Path to the pickle file containing the gene names dictionary.

    Returns:
        dict: A dictionary mapping gene names to IDs
    """

    with open(gene_names_file, "rb") as f:
        gene_names_dict = pickle.load(f)

    return gene_names_dict


def load_gene_median_dict(gene_median_file):
    """
    Loads a gene median dictionary from a pickle file.

    Args:
        gene_median_file (str): Path to the pickle file containing the gene median dictionary.

    Returns:
        dict: A dictionary mapping gene IDs to their median expression values.
    """

    with open(gene_median_file, "rb") as f:
        gene_median_dict = pickle.load(f)

    return gene_median_dict


def load_gene_tokenization(token_dictionary_file):
    """
    Loads gene tokenization data from a pickle file.

    Args:
        token_dictionary_file (str): Path to the pickle file containing the gene-token dictionary.

    Returns:
        dict: Gene-token dictionary (Ensembl ID: token).
        list: List of all gene keys (Ensembl IDs).
        dict: Dictionary mapping gene keys to True (used for selecting genes later).
    """

    with open(token_dictionary_file, "rb") as f:
        gene_token_dict = pickle.load(f)

    gene_keys = list(gene_token_dict.keys())

    # Optimization: Pre-allocate the list for slight performance improvement
    genelist_dict = dict.fromkeys(gene_keys, True)

    return gene_token_dict, gene_keys, genelist_dict


def rank_genes(gene_vector, gene_tokens):
    """Ranks genes based on expression values in descending order.

    Args:
        gene_vector (numpy.ndarray): Array of gene expression values.
        gene_tokens (numpy.ndarray): Array of corresponding gene tokens.

    Returns:
        numpy.ndarray: Array of gene tokens sorted by descending expression value.
    """
    return gene_tokens[np.argsort(-gene_vector)]


def normalize_counts(adata_chunk,  counts_column='n_counts', target_sum=10000):
    """Normalizes gene expression counts within a chunk of AnnData.

    Args:
        adata_chunk (AnnData): A chunk of the AnnData object containing gene expression data.
        counts_column (str): Name of the column in `adata_chunk.obs` containing the total counts per cell.
        target_sum (float): The desired total count per cell after normalization.
        norm_factor_vector (numpy.ndarray): An array of normalization factors for each gene.

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix containing the normalized gene expression counts.

    This function performs the following steps:
        1. Extracts the total counts per cell from the specified column (`counts_column`).
        2. Normalizes the gene expression matrix (`adata_chunk.X`) by dividing by the total counts 
           and multiplying by the `target_sum`.
        3. Further adjusts the normalized values by dividing by the gene-specific normalization 
           factors (`norm_factor_vector`).
        4. Returns the normalized expression matrix as a sparse CSR matrix for efficient storage 
           and computation.
    """
    
    n_counts = adata_chunk.obs[counts_column].values[:, None]  # Cell counts as column vector
    X_norm = adata_chunk.X / n_counts * target_sum / norm_factor_vector
    return sp.csr_matrix(X_norm)  # Efficient sparse representation


def tokenize_anndata(adata, genelist_dict, gene_median_dict, 
                     chunk_size=100000, target_sum=10000, 
                     counts_column='n_counts', gene_id="ensembl_id"):
    """
    Tokenizes and ranks genes within an AnnData object, optimizing for memory efficiency.

    This function processes gene expression data in chunks, applies normalization, and ranks genes
    for each cell based on their expression levels. The resulting tokenized and ranked gene
    representations, along with cell metadata, are returned.

    Args:
        adata (AnnData): The AnnData object containing gene expression data.
        genelist_dict (dict): Dictionary mapping gene IDs to boolean values indicating relevance.
        gene_median_dict (dict): Dictionary mapping gene IDs to their median expression values.
        chunk_size (int, optional): Number of cells to process in each chunk (default: 1000).
        target_sum (int, optional): Target sum for count normalization (default: 10000).
        counts_column (str, optional): The column in `adata.obs` containing cell counts (default: 'n_counts').
        gene_id (str, optional): The column in `adata.var` containing gene IDs (default: 'ensembl_id').

    Returns:
        tuple: 
            - list: List of tokenized and ranked gene lists for each cell.
            - dict: Dictionary containing cell metadata (keys are metadata column names).
    """
    # Filter relevant miRNAs
    coding_miRNA_mask = np.array([genelist_dict.get(i, False) for i in adata.var[gene_id]])
    coding_miRNA_loc = np.where(coding_miRNA_mask)[0]

    # Extract miRNA information
    coding_miRNA_ids = adata.var[gene_id].iloc[coding_miRNA_loc]
    norm_factor_vector = np.array([gene_median_dict[i] for i in coding_miRNA_ids])
    coding_miRNA_tokens = np.array([gene_token_dict[i] for i in coding_miRNA_ids])

    tokenized_cells = []
    file_cell_metadata = {k: [] for k in adata.obs.columns}  # Initialize metadata dict

    # Process in chunks for memory efficiency
    for chunk_start in range(0, adata.shape[0], chunk_size):
        chunk_end = chunk_start + chunk_size
        adata_chunk = adata[chunk_start:chunk_end, coding_miRNA_loc]
        
        # Normalize counts (could be replaced with the untested function above)
        n_counts = adata_chunk.obs[counts_column].values[:, None]
        X_norm = adata_chunk.X / n_counts * target_sum / norm_factor_vector
        X_norm = sp.csr_matrix(X_norm)  

        # Tokenize and rank genes for each cell in chunk
        for i in range(X_norm.shape[0]):
            ranks = rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
            ranks = list(ranks[~np.isnan(ranks)].astype(int))

            tokenized_cells.append(ranks)

        # Update metadata
        for k in adata.obs.columns:
            file_cell_metadata[k].extend(adata_chunk.obs[k].tolist())

    return tokenized_cells, file_cell_metadata


def save_hf_dataset(dataset: Dataset, output_path: str, overwrite=True):
    """
    Saves a Hugging Face Dataset to disk at a specified file path.

    This function serializes a Hugging Face `Dataset` object and saves it to disk in the Arrow format.

    Args:
        dataset (Dataset): The Hugging Face `Dataset` object to be saved.
        output_path (str): The full file path (including the filename) where the dataset will be saved. 
        overwrite (bool, optional): If `True`, an existing dataset at `output_path` will be overwritten. 
                                   If `False` and the file exists, a `FileExistsError` is raised (default: True).

    Raises:
        TypeError: If `dataset` is not a Hugging Face `Dataset` instance.
        FileExistsError: If `output_path` points to an existing file and `overwrite` is False.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("The provided dataset is not a Hugging Face Dataset.")

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Dataset '{output_path}' already exists. Set `overwrite=True` to overwrite."
        )
    dataset.save_to_disk(output_path)

        
if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Data processing script.")
    
    parser.add_argument("-i", "--input_path", help="Path to the input file.")
    
    parser.add_argument("-o", "--output_path", help="Path to save the output file.")
    
    parser.add_argument("-t", "--token_path", nargs='?', 
                        const=DEFAULT_TOKEN_PATH, type=str,
                        default=DEFAULT_TOKEN_PATH,
                        help="Path to the token file (optional).")
    
    parser.add_argument("-m", "--median_path", nargs='?',
                        const=DEFAULT_MEDIAN_PATH, type=str,
                        default=DEFAULT_MEDIAN_PATH,
                        help="Path to the median file (optional).")
    
    parser.add_argument("--n_proc", nargs='?',
                        const=16, type=int,
                        default=16,
                        help="Number of processes to use when tokenizing")
    
    parser.add_argument("--model_size", nargs='?',
                        const=2048, type=int,
                        default=2048,
                        help="Number of genes top-ranked to use before truncating")
    
    parser.add_argument("--target_sum", nargs='?',
                        const=10000, type=float,
                        default=10000,
                        help="Number of genes top-ranked to use before truncating")
    
    parser.add_argument("--gene_id", nargs='?',
                        const='ensembl_id', type=str,
                        default='ensembl_id',
                        help="The column name of ensemble identifiers")
    
    parser.add_argument("--counts_column", nargs='?',
                        const='n_counts', type=str,
                        default='n_counts',
                        help="The column name of ensemble identifiers")
    
    parser.add_argument("--layer", nargs='?',
                        const='X', type=str,
                        default='X',
                        help="The layer of the adata object to use.")
    
    parser.add_argument("--gene_names", nargs='?',
                    const=DEFAULT_NAME_PATH, type=str,
                    default=DEFAULT_NAME_PATH,
                    help="Path to the gene name mapping")
    
        
    parser.add_argument("--gene_name_column", nargs='?',
                        const='gene_name', type=str,
                        default='gene_name',
                        help="Column in adata.var with gene names to be mapped")
    
    parser.add_argument('--map_gene_names', action='store_true',
                       help="Boolean flag controlling gene name --> ensemble id mapping.")
    
    # parse args
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    token_path = args.token_path
    median_path = args.median_path
    n_proc = args.n_proc
    model_size = args.model_size
    target_sum = args.target_sum
    gene_id = args.gene_id
    counts_column = args.counts_column
    layer = args.layer
    gene_names = args.gene_names
    gene_name_column = args.gene_name_column
    map_names = args.map_gene_names
    
    print(args)

    # load the resources
    gene_token_dict, gene_keys, genelist_dict = load_gene_tokenization(token_path)
    gene_median_dict = load_gene_median_dict(median_path)
    gene_names = load_gene_names(gene_names)

    # load the data
    adata = sc.read_h5ad(input_path)
     
    # check gene names
    if map_names:
        adata = map_gene_names(adata, 
                               gene_id, 
                               gene_name_column, 
                               gene_names)
        
    # check the layer
    if not layer == 'X':
        adata.X = adata.layer[layer]
        
    # check the counts column
    adata = check_counts_column(adata, counts_column)
        
    # tokenize raw counts
    tokenized_cells, cell_metadata = tokenize_anndata(adata, 
                                                      genelist_dict, 
                                                      gene_median_dict,
                                                      target_sum=target_sum,
                                                      gene_id=gene_id,
                                                      counts_column=counts_column,
                                                     )
    
    # Merge cell metadata into the dataset dictionary
    dataset_dict = {
        "input_ids": tokenized_cells,
        **cell_metadata 
    }
    
    output_dataset = Dataset.from_dict(dataset_dict)

    def format_cell_features(example):
        example["input_ids"] = example["input_ids"][0 : model_size] # truncate
        example["length"] = len(example["input_ids"])  # Add length for convenience
        return example

    dataset = output_dataset.map(format_cell_features, num_proc=n_proc) 
    
    # store output
    save_hf_dataset(dataset, output_path, overwrite=True)

    
    
    
    
    