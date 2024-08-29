import sys
import os
import pandas as pd 
import numpy as np
import torch
import anndata as an
import scanpy as sc
from datasets import Dataset, load_from_disk

import geneformer_utils as gtu

torch.cuda.empty_cache()





if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Error: Please provide dataset path, model path, and output path as arguments.")
        sys.exit(1)  # Exit with an error code

    dataset_path = sys.argv[1]
    model_path = sys.argv[2]
    outpath = sys.argv[3]
    num_cells = None # all cells, useful for testing 
    
    print(model_path)

    print(f"Loading model from '{model_path}'...")
    model = gtu.load_model(model_path)
    print("Model loaded successfully!")

    print(f"Loading dataset from '{dataset_path}' (up to {num_cells} cells)...")
    try:
        df = gtu.load_data_as_dataframe(dataset_path, num_cells=num_cells)
        data = Dataset.from_pandas(df)
        df = df.drop(columns='input_ids')
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'")
        sys.exit(1)
    except Exception as e:  # Catching other potential errors
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    print("Dataset loaded successfully!")

    print("Extracting embeddings...")
    embs = gtu.extract_embedding_in_mem(model, data)
    adata = gtu.embedding_to_adata(embs)
    adata.obs = df.astype(str).reset_index().copy()
    print("Embeddings extracted successfully!")

    print(f"Writing results to '{outpath}'...")
    try:
        adata.write(outpath)
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    print("Output file written successfully!")
    
    
    
    
    
    
    
    
    