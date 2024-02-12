import click

from simulation import run_simulation

@click.command()
@click.option('--clients', type=int, required=True, help="number of clients to run the simulation")
@click.option('--cpus', type=int, required=True, help="number of cpu cores that should be used by the simulation")
@click.option('--dataset-cache', type=click.Path(), help="directory where the datasets should be cached")
def main(clients, cpus, dataset_cache):
    run_simulation(
        num_clients=clients,
        num_cpus=cpus,
        dataset_cache=dataset_cache,
    )

if __name__ == "__main__":
    main()