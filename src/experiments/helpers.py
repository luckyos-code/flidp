import numpy as np

def create_budgets(num_clients, possible_budgets, budget_ratios):
    assert sum(budget_ratios) == 1, "All budget ratios must sum up to 1."
    return np.random.choice(possible_budgets, p=budget_ratios, size=num_clients)


def get_sampling_rates_per_client(budgets_per_client, budgets, sampling_rates_per_budget):
    sampling_rates_per_client = np.zeros(budgets_per_client.shape)
    for (b, q) in zip(budgets, sampling_rates_per_budget):
        sampling_rates_per_client[budgets_per_client == b] = q
    return sampling_rates_per_client
