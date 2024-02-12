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
DATASET_CACHE_DIR=$CODE_DIR/dataset-cache

NUM_CLIENTS=10
NUM_CPUS=10


cd $CODE_DIR && python3.9 src/main.py \
    --clients $NUM_CLIENTS \
    --cpus $NUM_CPUS \
    --dataset-cache $DATASET_CACHE_DIR