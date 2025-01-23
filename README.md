# Federated Learning with Individualized Differential Privacy
- reference code for paper "Federated Learning with Individualized Differential Privacy" [arxiv preprint](TODO)
- report pdf and results can be found under [/report](./report) TODO

TODO abstract

## Usage Instructions

### Software and Hardware Requirements
* Python 3.10 or comparable (for using `spyhton`)
* our mode of operation as described below requires a Slurm-based computing cluster node with Singularity container support
* alternatively one can change the scripts to run locally or using Docker (or any other equivalent) directly

### Getting Needed Datasets
Get the custom CIFAR-10 and SVHN datasets prepared for our federated learning task.

1. download 'datasets_custom.zip' from our [zenodo repository](TODO)
2. extract contents - there should now be '/datasets_custom/cifar10' and 'datasets_custom/svhn' folders

Optional: in '/src/notebooks' you can find the script to recreate these datasets if needed.

### Running Experiments
1. install sphyton
    `pip install spython`
2. create Singularity file from Dockerfile
    `spython recipe Dockerfile > Singularity`
3. build .sif container from Singularity file
    `singularity build flidp.sif Singularity`
4. test run some code using the .sif container
    `singularity exec --nv flidp.sif python3.10 src/main.py`
5. start single experiment on a slurm cluster using any of the given .sh-scripts or use `./run_all.sh` to start all experiments at once (NOTE: change the #SBATCH configuration for your cluster specifics; for aldaghri experiment uncomment lines in emnist.sh)
6. results will be located in the 'results' folder and any run outputs in the 'logs' folder
