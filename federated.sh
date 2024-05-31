#!/bin/bash

#SBATCH --job-name=flidp
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --output=/home/sc.uni-leipzig.de/oe152msue/logs/%x-%j/stdout.out
#SBATCH --error=/home/sc.uni-leipzig.de/oe152msue/logs/%x-%j/stderr.err

CODE_DIR=$HOME/flidp
CONTAINER_FILE=$HOME/flidp_main.sif

# must be allocated before starting the job
WORK_DIR=/work/$USER-flidp
TS=$(date '+%d.%m.%Y-%H:%M:%S');
RUN_DIR=$WORK_DIR/run-$TS
mkdir $RUN_DIR

echo "START"

singularity exec --bind /work:/work --nv $CONTAINER_FILE bash -c \
"\
cd $CODE_DIR && \
python3 src/main.py --dir $RUN_DIR/strict --dataset svhn --budgets 1.0 --ratios 1.0 && \
python3 src/main.py --dir $RUN_DIR/inidividual-strict --dataset svhn --budgets 1.0 2.0 3.0 --ratios 0.54 0.37 0.09 && \
python3 src/main.py --dir $RUN_DIR/individual-relaxed --dataset svhn --budgets 1.0 2.0 3.0 --ratios 0.34 0.43 0.23 && \
python3 src/main.py --dir $RUN_DIR/relaxed --dataset svhn --budgets 3.0 --ratios 1.0 \
"

echo "FINISHED"