# Federated Learning with Individualized Differential Privacy

## Sync files to the cluster

```
rsync -avu . oe152msue@login01.sc.uni-leipzig.de:~/flidp
```

Add `--delete` as option to delete files that do not exist in the source directory.

## Resource Access on Cluster

To interactively allocate resources use `salloc`, e.g.

```
salloc -N 1 -p clara --gres=gpu:rtx2080ti:1 --mem=32G --cpus-per-task=8
```

Possible GPU choices are `gpu:rtx2080ti:1` and `gpu:v100:1`

Then run any command with `srun`. To free the resources type `exit` or cancel the job via `scancel`.

To check the status of your submitted jobs run 

```
squeue -u $USER
```

## Pull docker image

To log into `ghcr` run 

```
apptainer remote login --username <USER> docker://ghcr.io
```

To pull an image run 

```
singularity pull docker://<ghcr-image>
```