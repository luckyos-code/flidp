import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

logical_gpus = 8
logical_gpu_memory = 1024

def create_budgets(num_clients, possible_budgets, budget_ratios):
    assert sum(budget_ratios) == 1, "All budget ratios must sum up to 1."
    return np.random.choice(possible_budgets, p=budget_ratios, size=num_clients)


def get_sampling_rates_per_client(budgets_per_client, budgets, sampling_rates_per_budget):
    sampling_rates_per_client = np.zeros(budgets_per_client.shape)
    for (b, q) in zip(budgets, sampling_rates_per_budget):
        sampling_rates_per_client[budgets_per_client == b] = q
    return sampling_rates_per_client


def setup_tff_runtime():
    tff.backends.native.set_sync_local_cpp_execution_context()
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not gpu_devices:
        print(tf.config.list_logical_devices())
        return
    tf.config.set_logical_device_configuration(
        gpu_devices[0], 
        [
            tf.config.LogicalDeviceConfiguration(memory_limit=logical_gpu_memory) 
            for _ in range(logical_gpus)
        ]
    )
    
    print(tf.config.list_logical_devices())

