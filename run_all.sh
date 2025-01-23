#!/bin/bash

# Script to trigger all the experiments in seperated slurm jobs

TS=$(date '+%Y-%m-%d_%H:%M:%S');

# workspace must be allocated before starting the job
WORK_DIR=$PWD/results/all-$TS
mkdir -p $WORK_DIR

# Define datasets and parameters
datasets=("emnist" "cifar10" "svhn")
privacy=("no-dp" "relaxed" "individual-relaxed" "individual-strict" "strict")

# Loop through datasets
for dataset in "${datasets[@]}"; do
    # Check if the dataset is already IID (SVHN in this case)
    if [ "$dataset" == "svhn" ]; then
        for param in "${privacy[@]}"; do
            sbatch ./${dataset}.sh -d $WORK_DIR -p $param
        done
    else
        # Non-IID runs
        for param in "${privacy[@]}"; do
            sbatch ./${dataset}.sh -d $WORK_DIR -p $param
        done
        # IID runs (add the -r flag) # TODO
        for param in "${privacy[@]}"; do
            sbatch ./${dataset}.sh -d $WORK_DIR -p $param -r
        done
    fi
done

echo "Results will be saved in ${WORK_DIR}"