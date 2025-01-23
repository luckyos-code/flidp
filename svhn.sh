#!/bin/bash

#SBATCH --job-name=flidp-svhn
#SBATCH --partition=clara
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j/stdout.out
#SBATCH --error=logs/%x-%j/stderr.err
#SBATCH --mail-type=FAIL

# CAUTION: no iid version available to -r not working here -- sample slurm run for non-iid and no-dp: sbatch svhn.sh -p no-dp

set -x  # to print all the commands to stderr

# parameters that change depending on dataset
DATASET="svhn"
BUDGETS=(10.0 20.0 30.0)
CLIENTS_PER_ROUND=30

# common paths
CODE_DIR=$PWD
CONTAINER_FILE=$PWD/flidp.sif
FALLBACK_WORK_DIR=$PWD/results

# common training parameters
MODEL="simple-cnn"
ROUNDS=420
LOCAL_EPOCHS=15
BATCH_SIZE=128
CLIENT_LR=0.0005
SERVER_LR=1.0

# privacy distributions
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
if [ -z "$WORK_DIR" ] || ! [ -d "$WORK_DIR" ]; then
    echo "working directory ${WORK_DIR} does not exist. Creating and using: ${FALLBACK_WORK_DIR}"
    mkdir -p $FALLBACK_WORK_DIR
    WORK_DIR=$FALLBACK_WORK_DIR
fi

# check if privacy level is in defined set of levels
PRIVACY_LEVELS=("no-dp" "relaxed" "individual-relaxed" "strict" "individual-strict")
if ! [[ ${PRIVACY_LEVELS[@]} =~ $PRIVACY_LEVEL ]]; then
    echo "privacy level ${PRIVACY_LEVEL} is not available."
    exit 1
fi

TS=$(date '+%Y-%m-%d_%H:%M:%S');
RUN_DIR="${WORK_DIR}/${DATASET}_${PRIVACY_LEVEL}"
if [ "$MAKE_IID" = true ]; then
    RUN_DIR="${RUN_DIR}_iid"
fi
RUN_DIR="${RUN_DIR}_${TS}"

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