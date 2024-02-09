# Federated Learning with Individualized Differential Privacy

## Resource Access on Cluster

To interactively allocate resources use `salloc`, e.g.

```
salloc -N 1 -p clara --gres=gpu:v100:1
```

Then run any command with `srun`. To free the resources type `exit` or cancel the job via `scancel`.