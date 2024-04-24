CONTAINER_FILE=/home/sc.uni-leipzig.de/oe152msue/flidp/flidp_main.sif

srun singularity exec --nv $CONTAINER_FILE bash -c \
"cd $PWD && \
$1"