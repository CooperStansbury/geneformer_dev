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
    #######################################################################################################################################

    ### Parameters v
    ###

    # cells to filter on
    initial_cell_type = 'Fibroblast'

    # list of genes to perturb with at the front of the list
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

    ############### important! ###############
    # removed sampling, now all cells
    # num_initial_cells = 10
    ################  (1/2)  #################
    
    model_path = "/scratch/indikar_root/indikar1/shared_data/geneformer/fine_tune/240715_geneformer_cellClassifier_no_induced/ksplit1/"

    token_data_path = "/scratch/indikar_root/indikar1/shared_data/geneformer/resources/token_mapping.csv"

    data_path = "/scratch/indikar_root/indikar1/shared_data/geneformer/fine_tune/hsc.dataset"

    ###
    ### Parameters ^

    #######################################################################################################################################

    ### Preliminaries v
    ###

    # Uses token_df to translate from gene_list to tokens_list v
    def get_tokens_list(gene_list):
        # Get a df of the genes we are perturbing with
        genes = token_df[token_df['gene_name'].isin(gene_list)]
    
        tf_map = dict(zip(genes['gene_name'].values, genes['token_id'].values))
    
        # Create tokens_list by looking up each gene_name in the tf_map
        tokens_list = [tf_map.get(gene_name, gene_name) for gene_name in gene_list]

        return tokens_list

    def add_perturbations_to_cell(cell_tokens, perturbation_tokens):


        original_length = len(cell_tokens)

        # Remove existing perturbation tokens from the cell
        cell_tokens = [token for token in cell_tokens if token not in perturbation_tokens]

        # Add perturbations, then slice or pad to match original length
        final_tokens = (perturbation_tokens + cell_tokens)[:original_length]  # Slice if too long
        final_tokens += [0] * (original_length - len(final_tokens))            # Pad if too short

        return final_tokens

    ###

    if torch.cuda.is_available(): 
        print("CUDA is available! Devices: ", torch.cuda.device_count()) 
        print("Current CUDA device: ", torch.cuda.current_device()) 
        print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device())) 
    else: print("CUDA is not available")

    model = gtu.load_model(model_path)
    print('model loaded!')

    token_df = pd.read_csv(token_data_path)
    token_df.head()

    ###
    ### Preliminaries ^

    #######################################################################################################################################

    ### Format raw data, print messages to check v
    ###

    # Load from pre-trained data
    raw_data = load_from_disk(data_path)

    # Convert to DataFrame for filtering
    df = raw_data.to_pandas()
    print("\nOriginal Dataset:")
    print(f"  - Number of samples: {df.shape[0]:,}")
    print(f"  - Number of columns: {df.shape[1]:,}")

    # Filtering
    fb_df = df[df['standardized_cell_type'] == initial_cell_type]

    ############### important! ###############
    # sampling (REMOVED TO TEST ALL CELLS!)
    #fb_df = fb_df.sample(num_initial_cells)
    ################  (2/2)  #################

    fb_df = fb_df.reset_index(drop=True)

    # add a cell id
    fb_df['cell_id'] = [f"cell_{i+1}" for i in range(len(fb_df))]
    fb_df['recipe'] = 'raw'  # as opposed to having a speciofic ;-separated recipe list. other entries will have this.
    fb_df['type'] = 'initial' # this dataframe no longer includes 'target'

    print("\nFiltered Dataset:")
    print(f"  - Number of samples: {fb_df.shape[0]:,}")   # Nicer formatting with commas
    print(f"  - Number of columns: {fb_df.shape[1]:,}")

    # Value counts with sorting
    print("\nCell Type Distribution (Filtered):")
    print(fb_df['standardized_cell_type'].value_counts().sort_index())  # Sort for readability

    # Convert back to Dataset
    fb_data = Dataset.from_pandas(fb_df)
    print(f"\nDataset converted back: {fb_data}")

    ###
    ### Format raw data, print messages to check ^

    #######################################################################################################################################

    ### Get the embeddings into an Anndata object v
    ###

    reload(gtu)
    torch.cuda.empty_cache()

    fb_embs = gtu.extract_embedding_in_mem(
        model, 
        fb_data, 
        layer_to_quant=-1,
        forward_batch_size=100,
    )
    print(f"{fb_embs.shape=}")

    # translate into an anndata object
    fb_adata = gtu.embedding_to_adata(fb_embs)
    fb_adata.obs = fb_df.copy()
    fb_adata.obs.head()


    ###
    ### Get the embeddings into an Anndata object ^

    #######################################################################################################################################

    ### Perform the perturbation v
    ###

    reprogramming_df = [
        fb_df
    ]

    perturb = fb_df.copy()
    recipe = ";".join(gene_list)
    perturb['recipe'] = recipe
    perturb['type'] = 'reprogrammed'
    perturb['input_ids'] = perturb['input_ids'].apply(lambda x: add_perturbations_to_cell(x, get_tokens_list(gene_list)))

    reprogramming_df.append(perturb)

    reprogramming_df = pd.concat(reprogramming_df, ignore_index=True)

    print(f"{reprogramming_df.shape=}")
    reprogramming_df.sample(10)

    ###
    ### Perform the perturbation ^

    #######################################################################################################################################


