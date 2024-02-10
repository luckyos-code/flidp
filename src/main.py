import click

from simulation import run_simulation

@click.command()
@click.option('--clients', type=int, required=True, help="number of clients to run the simulation")
def main(clients):
    run_simulation(num_clients=clients)

if __name__ == "__main__":
    main()