#!/bin/bash

#SBATCH --job-name=flidp
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%x-%j/stdout.out
#SBATCH --error=logs/%x-%j/stderr.err

CODE_DIR=$HOME/flidp
CONTAINER_FILE=$CODE_DIR/flidp_main.sif

DATASET_CACHE_DIR=$CODE_DIR/dataset-cache

NUM_CLIENTS=10
NUM_CPUS=10

echo "START"

singularity exec --nv $CONTAINER_FILE bash -c \
"cd $CODE_DIR && \
python3 src/main.py \
    --clients $NUM_CLIENTS \
    --cpus $NUM_CPUS \
    --dataset-cache $DATASET_CACHE_DIR
"

echo "FINISHED"