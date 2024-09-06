#!/bin/bash

#SBATCH --job-name=flidp-cifar10
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/home/sc.uni-leipzig.de/oe152msue/logs/%x-%j/stdout.out
#SBATCH --error=/home/sc.uni-leipzig.de/oe152msue/logs/%x-%j/stderr.err

set -x  # to print all the commands to stderr

CODE_DIR=$HOME/flidp
CONTAINER_FILE=$HOME/flidp_main.sif
DATASET="cifar10"
BUDGETS=(10.0 20.0 30.0)

MODEL="simple-cnn"
ROUNDS=100
CLIENTS_PER_ROUND=30
LOCAL_EPOCHS=10
BATCH_SIZE=512
CLIENT_LR=0.001  # 0.0003
SERVER_LR=1.0

# must be allocated before starting the job
WORK_DIR=/work/$USER-flidp
TS=$(date '+%Y-%m-%d_%H:%M:%S');
RUN_DIR="${WORK_DIR}/${TS}_${DATASET}"
mkdir $RUN_DIR

echo "START"

echo "Running on ${DATASET}. Privacy budgets are: ${BUDGETS[*]}. The directory where results will be stored is ${RUN_DIR}."

singularity exec --bind /work:/work --nv $CONTAINER_FILE bash -x -c \
"\
cd $CODE_DIR && \
python3 src/main.py --save-dir $RUN_DIR/no-dp --dataset $DATASET --model $MODEL --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR && \
python3 src/main.py --save-dir $RUN_DIR/strict --dataset $DATASET --model $MODEL --budgets ${BUDGETS[0]} --ratios 1.0 --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR && \
python3 src/main.py --save-dir $RUN_DIR/inidividual-strict --dataset $DATASET --model $MODEL --budgets ${BUDGETS[*]} --ratios 0.54 0.37 0.09 --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR && \
python3 src/main.py --save-dir $RUN_DIR/individual-relaxed --dataset $DATASET --model $MODEL --budgets ${BUDGETS[*]} --ratios 0.34 0.43 0.23 --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR && \
python3 src/main.py --save-dir $RUN_DIR/relaxed --dataset $DATASET --model $MODEL --budgets ${BUDGETS[-1]} --ratios 1.0 --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR \
"

echo "FINISHED"