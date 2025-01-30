# Federated Learning with Individualized Differential Privacy
- reference code for preprint paper: [Federated Learning with Individualized Differential Privacy](https://arxiv.org/abs/2501.17634)
- report pdf and results can be found in the [report folder](./report)

## Abstract
With growing concerns about user data collection, individualized privacy has emerged as a promising solution to balance protection and utility by accounting for diverse user privacy preferences. Instead of enforcing a uniform level of anonymization for all users, this approach allows individuals to choose privacy settings that align with their comfort levels. Building on this idea, we propose an adapted method for enabling Individualized Differential Privacy (IDP) in Federated Learning (FL) by handling clients according to their personal privacy preferences. By extending the SAMPLE algorithm from centralized settings to FL, we calculate client-specific sampling rates based on their heterogeneous privacy budgets and integrate them into a modified DP-FedAvg algorithm. We test this method under realistic privacy distributions and multiple datasets. The experimental results demonstrate that our approach achieves clear improvements over uniform DP baselines, reducing the trade-off between privacy and utility. Compared to the alternative SCALE method in related work, which assigns differing noise scales to clients, our method performs notably better. However, challenges remain for complex tasks with non-i.i.d. data, primarily stemming from the constraints of the decentralized setting.

## Usage Instructions

### Software and Hardware Requirements
* Python 3.10 or comparable (for using `spyhton`)
* our mode of operation as described below requires a Slurm-based computing cluster node with Singularity container support
* alternatively one can change the scripts to run locally or using Docker (or any other equivalent) directly

### Getting Required Datasets
Get the custom CIFAR-10 and SVHN datasets prepared for our federated learning task.
FMNIST is directly downloaded in our code as part of the Tensorflow Federated library.

1. download 'datasets_custom.zip' from our zenodo repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14735130.svg)](https://doi.org/10.5281/zenodo.14735130)
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
