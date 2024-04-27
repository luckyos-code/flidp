CONTAINER_FILE=/home/sc.uni-leipzig.de/oe152msue/flidp/flidp_main.sif

# run command with ticks ('')!
srun singularity exec --nv --bind /work:/work $CONTAINER_FILE bash -c \
"cd $PWD && \
echo 'starting command...' && \
$1 && \
echo 'finished command.'"