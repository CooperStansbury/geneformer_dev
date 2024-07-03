import scanpy as sc
import anndata as an
import pandas as pd
from scipy import sparse
import os
import sys
import pickle

def load_pickle(path):
    """Loads a pickled object from the specified file path.

    Args:
        path (str): The file path to the pickled object.

    Returns:
        The unpickled object.

    Raises:
        FileNotFoundError: If the file does not exist.
        pickle.UnpicklingError: If there's an error unpickling the object.
    """
    
    with open(path, "rb") as f:
        return pickle.load(f)

    
def aggregate_gene_counts(adata):
    """
    Aggregates raw counts from an AnnData object at the gene level.

    Args:
        adata: An AnnData object containing gene expression data in a layer named 'raw_counts'.

    Returns:
        pandas.DataFrame: A DataFrame with gene names as index and summed raw counts as columns.
    """
    # Extract raw counts and transpose for gene-wise analysis
    df = adata.to_df(layer='raw_counts').T
    
    # Reset index to a column named 'gene_name'
    df.reset_index(names='gene_name', inplace=True)

    # Pre-aggregation info for debugging
    print(f"Initial shape: {df.shape}")

    # Extract base gene names (remove potential suffixes)
    df['gene_name'] = df['gene_name'].astype(str).str.split(".", n=1, expand=True)[0]  

    # Group by gene name and sum counts across all cells
    df = df.groupby('gene_name').sum()

    # Post-aggregation info
    print(f"Final shape: {df.shape}")

    return df.T


def create_anndata_from_dataframe(df, adata, gene_ids):
    """Creates an AnnData object from a DataFrame and existing AnnData metadata.

    Args:
        df: DataFrame with rows as genes and columns as samples.
        adata: Existing AnnData for metadata transfer.
        gene_ids: Dictionary mapping gene IDs to gene names.

    Returns:
        AnnData: New AnnData object with data, obs, var, and names.
    """
    pdf = an.AnnData(sparse.csr_matrix(df.to_numpy()))  # Sparse matrix for efficiency
    pdf.obs = adata.obs.copy()  # Copy observation metadata
    
    # Create variable annotations with gene IDs and names
    pdf.var = pd.DataFrame({'ensembl_id': df.columns})  # Assuming columns are Ensembl IDs
    pdf.var['gene_symbol'] = pdf.var['ensembl_id'].map(gene_ids)  # Add gene names
    pdf.var['gene_symbol'] = pdf.var['gene_symbol'].astype(str)

    pdf.var_names = df.columns  # Set variable names
    pdf.obs_names = df.index   # Set observation names

    return pdf



if __name__ == "__main__":
    input_path = sys.argv[1]  
    outpath = sys.argv[2]
    id_path = sys.argv[3]
    
    # load the gene ids 
    gene_ids = load_pickle(id_path)
    
    # load the data
    adata = sc.read_h5ad(input_path)
    adata.var_names = adata.var['ensemblid'].values
    
    # aggregate transcript counts
    df = aggregate_gene_counts(adata)
    
    # build new anndata object
    pdf = create_anndata_from_dataframe(df, adata, gene_ids)

    # save output file 
    pdf.write(outpath)
    
    
    
    
    