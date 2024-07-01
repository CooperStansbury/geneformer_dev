import pandas as pd
import yaml
from pathlib import Path
import re
import os
import sys
from snakemake.utils import Paramspace
from tabulate import tabulate

BASE_DIR = Path(workflow.basedir)
configfile: str(BASE_DIR) + "/config/config.yaml"
OUTPUT = config['output_path']


TS_DATA = [
    'TS_Blood',
    'TS_Bone_Marrow',
    'TS_Fat',
    'TS_Vasculature',
    # 'TS_endothelial',
    # 'TS_stromal',
    # 'TS_epithelial',
    # 'TS_immune',
]


rule all:
    input:
        OUTPUT + "anndata/pellin.h5ad",
        OUTPUT + "resources/tokens.pkl",
        OUTPUT + "resources/gene_names.pkl",
        OUTPUT + "resources/medians.pkl",
        OUTPUT + "resources/token_mapping.csv",
        OUTPUT + "datasets/iHSC.dataset",
        OUTPUT + "datasets/pellin.dataset",
        expand(OUTPUT + "anndata/{ts}.h5ad", ts=TS_DATA),
        expand(OUTPUT + "datasets/{ts}.dataset", ts=TS_DATA),

        
rule get_tokens:
    input:
        config['token_path']
    output:
        OUTPUT + "resources/tokens.pkl"
    shell:
        """cp {input} {output}"""
        
        
rule get_gene_names:
    input:
        config['name_path']
    output:
        OUTPUT + "resources/gene_names.pkl"
    shell:
        """cp {input} {output}"""
        
        
rule get_gene_medians:
    input:
        config['median_path']
    output:
        OUTPUT + "resources/medians.pkl"
    shell:
        """cp {input} {output}"""
        
        
rule get_token_mapping:
    input:
        config['token_mapping_path']
    output:
        OUTPUT + "resources/token_mapping.csv"
    shell:
        """cp {input} {output}"""


rule get_iHSC:
    input:
        config['ihsc_path']
    output:
        OUTPUT + "anndata/iHSC.h5ad"
    shell:
        """cp {input} {output}"""


rule build_iHSC:
    input:
        OUTPUT + "anndata/iHSC.h5ad",
    output:
        directory(OUTPUT + "datasets/iHSC.dataset")
    conda:
        "geneformer"
    shell:
        """python scripts/to_geneformer.py -i {input} -o {output} \
        --counts_column total_counts \
        --layer raw_counts \
        --verbose"""
        
        
rule get_pellin_data:
    input:
        config['pellin_path']
    output:
        OUTPUT + "anndata/pellin.h5ad"
    shell:
        """cp {input} {output}"""
        
         
rule build_pellin_data:
    input:
        OUTPUT + "anndata/pellin.h5ad",
    output:
        directory(OUTPUT + "datasets/pellin.dataset")
    conda:
        "geneformer"
    wildcard_constraints:
        ts='|'.join([re.escape(x) for x in set(TS_DATA)]),
    shell:
        """python scripts/to_geneformer.py -i {input} -o {output} \
        --map_gene_names \
        --verbose"""
        
        
rule get_tablula_data:
    input:
        data=config['ts_dir'] + "{ts}.h5ad",
        id_path=OUTPUT + "resources/gene_names.pkl",
    output:
        OUTPUT + "anndata/{ts}.h5ad",
    conda:
        "geneformer"
    wildcard_constraints:
        ts='|'.join([re.escape(x) for x in set(TS_DATA)]),
    shell:
        """python scripts/get_tabula_data.py {input.data} {output} {input.id_path}"""
        
        
         
rule build_tablula_data:
    input:
        OUTPUT + "anndata/{ts}.h5ad",
    output:
        directory(OUTPUT + "datasets/{ts}.dataset")
    conda:
        "geneformer"
    wildcard_constraints:
        ts='|'.join([re.escape(x) for x in set(TS_DATA)]),
    shell:
        """python scripts/to_geneformer.py -i {input} -o {output} \
        --counts_column n_counts_UMIs \
        --verbose """