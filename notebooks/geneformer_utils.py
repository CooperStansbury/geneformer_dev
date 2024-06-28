import seaborn as sns
import pandas as pd 
import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import torch
import anndata as an
import scanpy as sc
import umap
import gc

from datasets import Dataset, load_from_disk
from datasets import load_dataset
from geneformer import EmbExtractor
import geneformer


def load_and_subset_data(data_path: str, num_cells: int = 100) -> pd.DataFrame:
    """Loads a dataset from disk, selects a subset of cells, and converts it to a Pandas DataFrame.

    Args:
        data_path (str): Path to the dataset file.
        num_cells (int, optional): Number of cells to include in the subset (default: 100).

    Returns:
        pd.DataFrame: The subset of data as a Pandas DataFrame.
    """

    data = load_from_disk(data_path)
    if num_cells > len(data):
        raise ValueError(f"Requested subset size ({num_cells}) exceeds dataset length ({len(data)})")

    data_subset = data.select([i for i in range(num_cells)])
    df = data_subset.to_pandas()

    return df



def embedding_to_adata(df: pd.DataFrame, n_dim: int = None) -> an.AnnData:
    """Converts a Pandas DataFrame with an embedding to an AnnData object.

    Args:
        df: The input DataFrame with numerical embedding columns and optional metadata columns.
        n_dim: The number of dimensions to keep in the embedding. If None, all dimensions are kept.

    Returns:
        The converted AnnData object.

    Raises:
        ValueError: If `n_dim` exceeds the available dimensions in the DataFrame.
    """

    if n_dim is not None and n_dim > df.shape[1]:
        raise ValueError(f"n_dim ({n_dim}) exceeds available dimensions ({df.shape[1]})")

    # Assuming embedding columns are those that are not integers
    is_metadata = df.columns.astype(str).str.isdigit()
    metadata_df = df.loc[:, ~is_metadata]
    embedding_df = df.loc[:, is_metadata]

    cell_index = pd.Index([f"C{x}" for x in range(df.shape[0])], name='obs_names')

    if n_dim is not None:
        embedding_df = embedding_df.iloc[:, :n_dim]

    var_index = pd.Index([f"D{x}" for x in range(embedding_df.shape[1])], name='var_names')

    adata = an.AnnData(embedding_df.to_numpy())
    adata.obs_names = cell_index
    adata.var_names = var_index
    adata.obs = metadata_df
    return adata