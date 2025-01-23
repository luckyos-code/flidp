#!/bin/bash

#SBATCH --job-name=flidp-dataset_visualisation
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=16G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/%x-%j/stdout.out
#SBATCH --error=logs/%x-%j/stderr.err

singularity exec --nv flidp.sif python3.10 src/notebooks/dataset_visualisation.py