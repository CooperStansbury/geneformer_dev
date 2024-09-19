#!/bin/bash

#SBATCH --account=indikar0
#SBATCH --partition=standard
#SBATCH --mail-user=[SOMTHING]@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --time=36:00:00
#SBATCH --job-name=my_array_job
#SBATCH --array=1-10 
#SBATCH --output=my_array_job_%A_%a.out

data_input_path="some_path"
data_ouput_path="some_other_path" + "$SLURM_ARRAY_TASK_ID" + ".csv"

python run_kmeans.py "$data_input_path" "$SLURM_ARRAY_TASK_ID" "$data_ouput_path"