import torch
import flwr as fl

from client import client_fn
from data import load_datasets
from server import evaluate_fn
from net import Net, get_parameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_simulation(num_clients: int, num_cpus: int, dataset_cache: str):

    print(f"Running simulation on {DEVICE}.")

    ray_init_args = {"num_cpus": num_cpus}
    client_resources = {"num_gpus": 1} if DEVICE.type == "cuda" else None

    client_sizes = [1.0 / num_clients for _ in range(num_clients)]
    trainloaders, valloaders, testloader = load_datasets(
        dataset_chache=dataset_cache,
        client_sizes=client_sizes,
    )

    params = get_parameters(Net())

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        evaluate_fn=lambda *args: evaluate_fn(*args, valloader=testloader),
    )

    def client_f(cid):
        return client_fn(cid, trainloaders=trainloaders, valloaders=valloaders)

    fl.simulation.start_simulation(
        client_fn=client_f,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )