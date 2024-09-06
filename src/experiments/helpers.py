import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

def create_budgets(num_clients, possible_budgets, budget_ratios):
    assert sum(budget_ratios) == 1, "All budget ratios must sum up to 1."
    return np.random.choice(possible_budgets, p=budget_ratios, size=num_clients)


def get_sampling_rates_per_client(budgets_per_client, budgets, sampling_rates_per_budget):
    sampling_rates_per_client = np.zeros(budgets_per_client.shape)
    for (b, q) in zip(budgets, sampling_rates_per_budget):
        sampling_rates_per_client[budgets_per_client == b] = q
    return sampling_rates_per_client


def make_clientdata_iid(tff_ds: tff.simulation.datasets.ClientData, seed: int=None) -> tff.simulation.datasets.ClientData:
    if seed:
        np.random.seed(seed)
    global_ds = tff_ds.create_tf_dataset_from_all_clients()
    client_ids = tff_ds.client_ids
    client_assignments = dict(zip(tff_ds.client_ids, [{k: list() for k in global_ds.element_spec.keys()} for _ in tff_ds.client_ids]))
    for ex in global_ds:
        client = np.random.choice(client_ids)
        for k in client_assignments[client].keys():
            client_assignments[client][k].append(ex[k])

    client_datasets = {
        client_id: tf.data.Dataset.from_tensor_slices(client_assignments[client_id])
        for client_id in client_ids
    }
    
    def client_fn(client_id):
        return client_datasets[client_id]
    
    return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(client_ids=client_ids, serializable_dataset_fn=client_fn)