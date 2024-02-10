#!/bin/bash

#SBATCH --job-name=flidp
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%x-%j/stdout.out
#SBATCH --error=logs/%x-%j/stderr.err

CODE_DIR=$HOME/flidp
which python3.9
cd $CODE_DIR && python3.9 src/main.py \
    --clients 10