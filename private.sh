#!/bin/bash

#SBATCH --job-name=flidp-private
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --output=logs/%x-%j/stdout.out
#SBATCH --error=logs/%x-%j/stderr.err

CODE_DIR=$HOME/flidp
CONTAINER_FILE=$CODE_DIR/flidp_main.sif

DATASET_CACHE_DIR=$CODE_DIR/dataset-cache

echo "START"

singularity exec --nv $CONTAINER_FILE bash -c \
"cd $CODE_DIR && \
python3 src/private.py \
"

echo "FINISHED"