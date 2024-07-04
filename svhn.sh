#!/bin/bash

#SBATCH --job-name=flidp
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --output=/home/sc.uni-leipzig.de/oe152msue/logs/%x-%j/stdout.out
#SBATCH --error=/home/sc.uni-leipzig.de/oe152msue/logs/%x-%j/stderr.err

set -x  # to print all the commands to stderr

CODE_DIR=$HOME/flidp
CONTAINER_FILE=$HOME/flidp_main.sif
DATASET="svhn"
BUDGETS=(3.0 10.0 20.0)  # small budgets lead to very bad results (19% acc vs 76% without DP)

# must be allocated before starting the job
WORK_DIR=/work/$USER-flidp
TS=$(date '+%d.%m.%Y-%H:%M:%S');
RUN_DIR=$WORK_DIR/run-$TS
mkdir $RUN_DIR

echo "START"

echo "Running on ${DATASET}. Privacy budgets are: ${BUDGETS[*]}. The directory where results will be stored is ${RUN_DIR}."

singularity exec --bind /work:/work --nv $CONTAINER_FILE bash -c \
"\
cd $CODE_DIR && \
python3 src/main.py --dir $RUN_DIR/no-dp --dataset $DATASET && \
python3 src/main.py --dir $RUN_DIR/strict --dataset $DATASET --budgets ${BUDGETS[0]} --ratios 1.0 && \
python3 src/main.py --dir $RUN_DIR/inidividual-strict --dataset $DATASET --budgets ${BUDGETS[*]} --ratios 0.54 0.37 0.09 && \
python3 src/main.py --dir $RUN_DIR/individual-relaxed --dataset $DATASET --budgets ${BUDGETS[*]} --ratios 0.34 0.43 0.23 && \
python3 src/main.py --dir $RUN_DIR/relaxed --dataset $DATASET --budgets ${BUDGETS[-1]} --ratios 1.0 \
"

echo "FINISHED"