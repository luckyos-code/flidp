#!/bin/bash

# Script to trigger all the experiments in seperated slurm jobs

TS=$(date '+%Y-%m-%d_%H:%M:%S');

# workspace must be allocated before starting the job
WORK_DIR=$(realpath "/work/$USER-flidp/${TS}-all")
mkdir $WORK_DIR

# MNIST non-iid
sbatch ./emnist.sh -d $WORK_DIR -p no-dp
sbatch ./emnist.sh -d $WORK_DIR -p relaxed
sbatch ./emnist.sh -d $WORK_DIR -p individual-relaxed
sbatch ./emnist.sh -d $WORK_DIR -p individual-strict
sbatch ./emnist.sh -d $WORK_DIR -p strict

# MNIST iid
sbatch ./emnist.sh -d $WORK_DIR -p no-dp -r
sbatch ./emnist.sh -d $WORK_DIR -p relaxed -r
sbatch ./emnist.sh -d $WORK_DIR -p individual-relaxed -r
sbatch ./emnist.sh -d $WORK_DIR -p individual-strict -r
sbatch ./emnist.sh -d $WORK_DIR -p strict -r

# CIFAR10 non-iid
sbatch ./cifar10.sh -d $WORK_DIR -p no-dp
sbatch ./cifar10.sh -d $WORK_DIR -p relaxed
sbatch ./cifar10.sh -d $WORK_DIR -p individual-relaxed
sbatch ./cifar10.sh -d $WORK_DIR -p individual-strict
sbatch ./cifar10.sh -d $WORK_DIR -p strict

# CIFAR10 iid
sbatch ./cifar10.sh -d $WORK_DIR -p no-dp -r
sbatch ./cifar10.sh -d $WORK_DIR -p relaxed -r
sbatch ./cifar10.sh -d $WORK_DIR -p individual-relaxed -r
sbatch ./cifar10.sh -d $WORK_DIR -p individual-strict -r
sbatch ./cifar10.sh -d $WORK_DIR -p strict -r

# SVHN already iid
sbatch ./svhn.sh -d $WORK_DIR -p no-dp
sbatch ./svhn.sh -d $WORK_DIR -p relaxed
sbatch ./svhn.sh -d $WORK_DIR -p individual-relaxed
sbatch ./svhn.sh -d $WORK_DIR -p individual-strict
sbatch ./svhn.sh -d $WORK_DIR -p strict

echo "Results will be saved in $(realpath $WORK_DIR)"