import collections
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from tqdm.auto import tqdm

from idputils import get_weights


def sample_clients_without_dp(num_clients, client_ids):
    return np.random.choice(client_ids, num_clients, replace=False)


def sample_clients_with_dp(client_sampling_rate, client_ids):
    assert type(client_sampling_rate) == float
    s = np.random.uniform(0., 1., len(client_ids))
    return client_ids[s < client_sampling_rate]


def sample_clients_with_idp(client_sampling_rates, client_ids):
    assert len(client_ids) == len(client_sampling_rates)
    s = np.random.uniform(0., 1., len(client_ids))
    return client_ids[s < client_sampling_rates]


def save_train_results(save_dir, weights, history):
    save_dir = Path(save_dir)
    history_df = pd.DataFrame.from_records(history)
    history_df.to_csv(save_dir / "history.csv")


def train_without_dp(model_fn, client_optimizer_fn, server_optimizer_fn, train_data, test_data, rounds, clients_per_round, eval_after_rounds=5):
    history = []
    learning_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        use_experimental_simulation_loop=True,
    )
    eval_process = tff.learning.build_federated_evaluation(model_fn)

    state = learning_process.initialize()
    for round in tqdm(range(rounds)):
        if round % eval_after_rounds == 0:
            model_weights = learning_process.get_model_weights(state)
            metrics = eval_process(model_weights, [test_data])['eval']
            if round < 25 or round % 25 == 0:
                print(f'Round {round:3d}: {metrics}')

            history.append({
                'Round': round,
                **metrics
            })

        sampled_clients = sample_clients_without_dp(clients_per_round, train_data.client_ids)
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]

        result = learning_process.next(state, sampled_train_data)
        state = result.state
        metrics = result.metrics
    
    model_weights = learning_process.get_model_weights(state)
    metrics = eval_process(model_weights, [test_data])['eval']
    print(f'Round {rounds:3d}: {metrics}')
    history.append({
        'Round': round,
        **metrics
    })

    return model_weights, history


def train_with_dp(model_fn, client_optimizer_fn, server_optimizer_fn, train_data, test_data, rounds, noise_multiplier, clients_per_round, eval_after_rounds=5):
    history = []
    aggregation_factory = tff.learning.dp_aggregator(noise_multiplier, clients_per_round)
    learning_process = tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        model_aggregator=aggregation_factory,
        use_experimental_simulation_loop=True,
    )
    eval_process = tff.learning.build_federated_evaluation(model_fn)

    client_sampling_rate = clients_per_round / len(train_data.client_ids)

    # Training loop.
    state = learning_process.initialize()
    for round in tqdm(range(rounds)):
        if round % eval_after_rounds == 0:
            model_weights = learning_process.get_model_weights(state)
            metrics = eval_process(model_weights, [test_data])['eval']
            if round < 25 or round % 25 == 0:
                print(f'Round {round:3d}: {metrics}')

            history.append({
                'Round': round,
                'NoiseMultiplier': noise_multiplier,
                **metrics
            })
        
        sampled_clients = sample_clients_with_dp(client_sampling_rate, train_data.client_ids)
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]

        result = learning_process.next(state, sampled_train_data)
        state = result.state
        metrics = result.metrics
    
    model_weights = learning_process.get_model_weights(state)
    metrics = eval_process(model_weights, [test_data])['eval']
    print(f'Round {rounds:3d}: {metrics}')
    history.append({
        'Round': round,
        'NoiseMultiplier': noise_multiplier,
        **metrics
    })

    return model_weights, history


def train_with_idp(model_fn, client_optimizer_fn, server_optimizer_fn, train_data, test_data, client_sampling_rates, rounds, noise_multiplier, clients_per_round, eval_after_rounds=5):
    
    history = []
    aggregation_factory = tff.learning.dp_aggregator(noise_multiplier, clients_per_round)
    learning_process = tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        model_aggregator=aggregation_factory,
        use_experimental_simulation_loop=True,
    )
    eval_process = tff.learning.build_federated_evaluation(model_fn)

    # Training loop.
    state = learning_process.initialize()
    for round in tqdm(range(rounds)):
        if round % eval_after_rounds == 0:
            model_weights = learning_process.get_model_weights(state)
            metrics = eval_process(model_weights, [test_data])['eval']
            if round < 25 or round % 25 == 0:
                print(f'Round {round:3d}: {metrics}')

            history.append({
                'Round': round,
                'NoiseMultiplier': noise_multiplier,
                **metrics
            })

        sampled_clients = sample_clients_with_idp(client_sampling_rates, np.array(train_data.client_ids))
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]

        result = learning_process.next(state, sampled_train_data)
        state = result.state
        metrics = result.metrics

    model_weights = learning_process.get_model_weights(state)
    metrics = eval_process(model_weights, [test_data])['eval']
    print(f'Round {rounds:3d}: {metrics}')
    history.append({
        'Round': round,
        'NoiseMultiplier': noise_multiplier,
        **metrics
    })

    return model_weights, history
