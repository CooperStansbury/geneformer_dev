import sys
import seaborn as sns
import pandas as pd 
import numpy as np
from itertools import combinations
from scipy.spatial.distance import squareform, pdist
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch
import anndata as an
import scanpy as sc
import os
import gc
from importlib import reload

from datasets import Dataset, load_from_disk
from datasets import load_dataset
from geneformer import EmbExtractor
import geneformer as gtu

# classifer tools
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# local imports
sys.path.insert(0, '../../scripts/')
import geneformer_utils as gtu

sns.set_style('white')
torch.cuda.empty_cache()

def main():
    if torch.cuda.is_available(): 
        print("CUDA is available! Devices: ", torch.cuda.device_count()) 
        print("Current CUDA device: ", torch.cuda.current_device()) 
        print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device())) 
    else: print("CUDA is not available")

    model_path = "/scratch/indikar_root/indikar1/shared_data/geneformer/fine_tune/240715_geneformer_cellClassifier_no_induced/ksplit1/"
    model = gtu.load_model(model_path)

    token_data_path = "/scratch/indikar_root/indikar1/shared_data/geneformer/resources/token_mapping.csv"
    token_df = pd.read_csv(token_data_path)
    data_path = "/scratch/indikar_root/indikar1/shared_data/geneformer/fine_tune/hsc.dataset"

    # Load from pre-trained data
    raw_data = load_from_disk(data_path)
    
    # Convert to DataFrame for filtering
    df = raw_data.to_pandas()
    print("\nOriginal Dataset:")
    print(f"  - Number of samples: {df.shape[0]:,}")
    print(f"  - Number of columns: {df.shape[1]:,}")
    
    # Cell types to filter on
    cell_types = ['HSC', 'Fibroblast']
    
    # Filtering
    df = df[df['standardized_cell_type'].isin(cell_types)]
    
    # sampling 
    
    ###################   this is a parameter ################################
    sample_size = 10
    ##########################################################################
    df = df.sample(sample_size)
    df = df.reset_index(drop=True)
    
    # add a cell id
    df['cell_id'] = [f"cell_{i+1}" for i in range(len(df))]
    
    print("\nFiltered Dataset:")
    print(f"  - Number of samples: {df.shape[0]:,}")   # Nicer formatting with commas
    print(f"  - Number of columns: {df.shape[1]:,}")
    
    # Value counts with sorting
    print("\nCell Type Distribution (Filtered):")
    print(df['standardized_cell_type'].value_counts().sort_index())  # Sort for readability
    
    # Convert back to Dataset
    data = Dataset.from_pandas(df)
    print(f"\nDataset converted back: {data}")

        ###############################################    this is a parameter #############################
    gene_list = [
        'GATA2', 
        'GFI1B', 
        'FOS', 
        'STAT5A',
        'REL',
        'FOSB',
        'IKZF1',
        'RUNX3',
        'MEF2C',
        'ETV6',
    ]
    ####################################################################################################
    genes = token_df[token_df['gene_name'].isin(gene_list)]
    tf_map = dict(zip(genes['token_id'].values, genes['gene_name'].values))
    # compute all possible combinations of 5 TFs,
    ##################### this, right now, is also a parameter #####################################
    n_tf = 5
    ###############################################################################################
    inputs = list(combinations(genes['token_id'], n_tf))
    print(f'Number of recipes: {len(inputs)}')
    
    def map_tfs(tokens):
        return list(map(tf_map.get, tokens))

    def add_perturbations_to_cell(cell_tokens, perturbation_tokens):
    """
    Modifies a list of cell tokens by adding perturbation tokens and padding.

    Args:
        cell_tokens (list): A list of integers representing gene tokens.
        perturbation_tokens (list): A list of integers representing perturbation tokens.

    Returns:
        list: A new list of tokens with perturbations added, existing perturbations removed,
             and truncated/padded to the original length.
    """

        original_length = len(cell_tokens)
    
        # Remove existing perturbation tokens from the cell
        cell_tokens = [token for token in cell_tokens if token not in perturbation_tokens]
    
        # Add perturbations, then slice or pad to match original length
        final_tokens = (perturbation_tokens + cell_tokens)[:original_length]  # Slice if too long
        final_tokens += [0] * (original_length - len(final_tokens))            # Pad if too short
    
        return final_tokens
    # Filter out just the fibroblasts (initial cells....)

    fb_df = df[df['standardized_cell_type'] == 'Fibroblast'].reset_index(drop=True)
    fb_data = Dataset.from_pandas(fb_df)
    
    reload(gtu)
    torch.cuda.empty_cache()
    fb_embs = gtu.extract_embedding_in_mem(
        model, 
        fb_data, 
        layer_to_quant=-1,
        forward_batch_size=100,
    )
    print(f"{fb_embs.shape=}")
    
    # translate into an anndata object and plot
    fb_adata = gtu.embedding_to_adata(fb_embs)
    fb_adata.obs = fb_df.copy()
    fb_adata.obs.head()

    hsc_df = df[df['standardized_cell_type'] == 'HSC'].reset_index(drop=True)
    hsc_data = Dataset.from_pandas(hsc_df)
    
    reload(gtu)
    torch.cuda.empty_cache()
    hsc_embs = gtu.extract_embedding_in_mem(
        model, 
        hsc_data, 
        layer_to_quant=-1,
        forward_batch_size=100,
    )
    print(f"{hsc_embs.shape=}")
    
    # translate into an anndata object and plot
    hsc_adata = gtu.embedding_to_adata(hsc_embs)
    hsc_adata.obs = hsc_df.copy()
            
        
    #why did we set this again like chill out
    # sample size refers to the number of cells to perturb. this can be cleaned up; we didn't need the first filtering but do need this one. requires that sample_size <= size fb_df, which is at most sample_size as prev set to size of df. SLOPPY 
    sample_size = 5
    
    raw_cells = fb_df.sample(sample_size).reset_index(drop=True)
    print(f"{raw_cells.shape=}")
    raw_cells['recipe'] = 'raw'
    raw_cells['type'] = 'initial'
    
    hsc_sample = hsc_df.sample(sample_size).reset_index(drop=True)
    hsc_sample['recipe'] = 'hsc'
    hsc_sample['type'] = 'target'
    
    reprogramming_df = [
        raw_cells,
        hsc_sample,
    ]

        
    for i, tfs in enumerate(inputs):
        
        if i % 25 == 0:
            print(f"Pertubation {i}/{len(inputs)}...")
        
        # make the dataframe easily useable
        perturb = raw_cells.copy()
        recipe = ";".join(map_tfs(tfs))
        perturb['recipe'] = recipe
        perturb['type'] = 'reprogrammed'
        
        # do the actual perturbation
        perturb['input_ids'] = perturb['input_ids'].apply(lambda x: add_perturbations_to_cell(x, list(tfs)))
        
        # store the updated data
        reprogramming_df.append(perturb)
        
    reprogramming_df = pd.concat(reprogramming_df)
    reprogramming_df = reprogramming_df.reset_index(drop=True)
    print(f"{reprogramming_df.shape=}")

    reload(gtu)
    torch.cuda.empty_cache()
    
    reprogramming_data = Dataset.from_pandas(reprogramming_df)
    
    reprogramming_embs = gtu.extract_embedding_in_mem(
        model, 
        reprogramming_data, 
        layer_to_quant=-1,
        forward_batch_size=100,
    )
    print(f"{reprogramming_embs.shape=}")
    
    # translate into an anndata object and plot
    reprogramming_adata = gtu.embedding_to_adata(reprogramming_embs)
    reprogramming_adata.obs = reprogramming_df.copy()

