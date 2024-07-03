import scanpy as sc
import pandas as pd
import os
import sys


if __name__ == "__main__":
    input_path = sys.argv[1]  
    outpath = sys.argv[2]
    
    # load the data
    adata = sc.read_h5ad(input_path)
    adata.X = adata.layers['raw_counts']

    # save output file 
    adata.write(outpath)
    
    
    
    
    