#!/bin/bash

#SBATCH --job-name=flidp-cifar10
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/home/sc.uni-leipzig.de/oe152msue/logs/%x-%j/stdout.out
#SBATCH --error=/home/sc.uni-leipzig.de/oe152msue/logs/%x-%j/stderr.err

set -x  # to print all the commands to stderr

MODEL="simple-cnn"
ROUNDS=100
CLIENTS_PER_ROUND=30
LOCAL_EPOCHS=5
BATCH_SIZE=128
CLIENT_LR=0.001  # 0.0003
SERVER_LR=1.0

CODE_DIR=$HOME/flidp
CONTAINER_FILE=$HOME/flidp_main.sif
DATASET="cifar10"
BUDGETS=(1.0 2.0 3.0)
INDIVIDUAL_RELAXED_BUDGET_DISTRIBUTION=(0.34 0.43 0.23)
INDIVIDUAL_STRICT_BUDGET_DISTRIBUTION=(0.54 0.37 0.09)

while getopts d:p:r flag
do
    case "${flag}" in
        d) WORK_DIR=${OPTARG};;
        p) PRIVACY_LEVEL=${OPTARG};;
        r) MAKE_IID=true;;
    esac
done

# check if workdir exists
if ! [ -d $WORK_DIR ]; then
    echo "working directory ${WORK_DIR} does not exist."
    exit 1
fi

# check if privacy level is in defined set of levels
PRIVACY_LEVELS=("no-dp" "relaxed" "individual-relaxed" "strict" "individual-strict")
if ! [[ ${PRIVACY_LEVELS[@]} =~ $PRIVACY_LEVEL ]]; then
    echo "privacy level ${PRIVACY_LEVEL} is not available."
    exit 1
fi

TS=$(date '+%Y-%m-%d_%H:%M:%S');
RUN_DIR="${WORK_DIR}/${TS}_${DATASET}_${PRIVACY_LEVEL}"
if [ "$MAKE_IID" = true ]; then
    RUN_DIR="${RUN_DIR}_iid"
fi

PYTHON_COMMAND=""
case "${PRIVACY_LEVEL}" in
    "no-dp")
    PYTHON_COMMAND="python3 src/main.py --save-dir $RUN_DIR --dataset $DATASET --model $MODEL --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR"
    ;;
    "relaxed")
    PYTHON_COMMAND="python3 src/main.py --save-dir $RUN_DIR --dataset $DATASET --model $MODEL --budgets ${BUDGETS[-1]} --ratios 1.0 --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR"
    ;;
    "individual-relaxed")
    PYTHON_COMMAND="python3 src/main.py --save-dir $RUN_DIR --dataset $DATASET --model $MODEL --budgets ${BUDGETS[*]} --ratios ${INDIVIDUAL_RELAXED_BUDGET_DISTRIBUTION[*]} --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR"
    ;;
    "individual-strict")
    PYTHON_COMMAND="python3 src/main.py --save-dir $RUN_DIR --dataset $DATASET --model $MODEL --budgets ${BUDGETS[*]} --ratios ${INDIVIDUAL_STRICT_BUDGET_DISTRIBUTION[*]} --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR"
    ;;
    "strict")
    PYTHON_COMMAND="python3 src/main.py --save-dir $RUN_DIR --dataset $DATASET --model $MODEL --budgets ${BUDGETS[0]} --ratios 1.0 --rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND --local-epochs $LOCAL_EPOCHS --batch-size $BATCH_SIZE --client-lr $CLIENT_LR --server-lr $SERVER_LR"
    ;;
esac

if [ $PYTHON_COMMAND = "" ]; then
    echo "something went wrong with the command generation."
    exit 1
fi

if [ "$MAKE_IID" = true ]; then
    PYTHON_COMMAND="$PYTHON_COMMAND --make-iid"
fi

echo "START"

mkdir $RUN_DIR
echo $SLURM_JOB_ID > "${RUN_DIR}/slurm-job-id.txt"

echo "Command: ${PYTHON_COMMAND}"

singularity exec --bind /work:/work --nv $CONTAINER_FILE bash -c \
"\
cd $CODE_DIR && $PYTHON_COMMAND\
"

echo "FINISHED"