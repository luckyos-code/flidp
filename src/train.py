import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from tqdm.auto import tqdm

from idputils import get_weights
from helpers import get_sampling_rates_per_client, create_budgets
from idputils import get_noise_multiplier, get_weights


def _extract_info_from_dp_state(dp_aggregator_state):
    dp_query_state = dp_aggregator_state["inner_agg"].query_state
    numerator_state = dp_query_state.numerator_state
    sum_state = numerator_state.sum_state

    return {
        "noise_multiplier_after_adaptive_clipping": numerator_state.noise_multiplier,
        "sum_clipping_norm": sum_state.l2_norm_clip,
        "sum_stddev": sum_state.stddev,
    }


def sample_clients_without_dp(num_clients, client_ids):
    return np.random.choice(client_ids, num_clients, replace=False)


def sample_clients_with_dp(client_sampling_rate: float, client_ids: np.array):
    s = np.random.uniform(0., 1., len(client_ids))
    return client_ids[s < client_sampling_rate]


def sample_clients_with_idp(client_sampling_rates: np.array, client_ids: np.array):
    assert len(client_ids) == len(client_sampling_rates)
    s = np.random.uniform(0., 1., len(client_ids))
    return client_ids[s < client_sampling_rates]


def save_train_results(save_dir, weights, history):
    save_dir = Path(save_dir)
    history_df = pd.DataFrame.from_records(history)
    history_df.to_csv(save_dir / "history.csv")


def _training_loop(learning_process, eval_process, client_sample_fn, train_data, test_data, rounds, eval_after_rounds, noise_multiplier=None):
    is_with_dp = noise_multiplier is not None
    history = []
    state = learning_process.initialize()
    num_sampled_clients = 0
    for round in tqdm(range(rounds)):
        metrics = {}
        if noise_multiplier:
            round_info["dp_info"] = {"noise_multiplier": noise_multiplier, **_extract_info_from_dp_state(dp_aggregator_state=state.aggregator)}
        
        if round % eval_after_rounds == 0:
            model_weights = learning_process.get_model_weights(state)
            metrics = eval_process(model_weights, [test_data])['eval']

        sampled_clients = client_sample_fn(train_data.client_ids)
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]

        round_info = {
            "round": round,
            "sampled_clients": num_sampled_clients,
            "metrics": metrics,
            "dp_info": {}
        }
        if is_with_dp:
            round_info["dp_info"] = {"noise_multiplier": noise_multiplier, **_extract_info_from_dp_state(dp_aggregator_state=state.aggregator)}
        
        history.append(round_info)
        print(round_info)

        result = learning_process.next(state, sampled_train_data)
        state = result.state
        metrics = result.metrics
    
    model_weights = learning_process.get_model_weights(state)
    metrics = eval_process(model_weights, [test_data])['eval']
    round_info = {
            "round": round,
            "sampled_clients": num_sampled_clients,
            "metrics": metrics,
            "dp_info": {}
        }
    if is_with_dp:
        round_info["dp_info"] = {"noise_multiplier": noise_multiplier, **_extract_info_from_dp_state(dp_aggregator_state=state.aggregator)}
    print(round_info)
    history.append(round_info)

    return model_weights, history

def train_without_dp(model_fn, client_optimizer_fn, server_optimizer_fn, train_data, test_data, rounds, clients_per_round, eval_after_rounds=5):
    history = []
    learning_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        use_experimental_simulation_loop=True,  # is suggested here https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators
    )
    eval_process = tff.learning.build_federated_evaluation(model_fn)
    print(f"Aggregator: {learning_process.aggregator}")
    
    def client_sample_fn(client_ids):
        return sample_clients_without_dp(clients_per_round, client_ids)

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
    aggregation_factory = tff.learning.dp_aggregator(
        noise_multiplier=noise_multiplier, 
        clients_per_round=clients_per_round
    )
    learning_process = tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        model_aggregator=aggregation_factory,
        use_experimental_simulation_loop=True,
    )
    eval_process = tff.learning.build_federated_evaluation(model_fn)

    client_sampling_rate = clients_per_round / len(train_data.client_ids)
    num_sampled_clients = 0
    # Training loop.
    state = learning_process.initialize()
    for round in tqdm(range(rounds)):
        round_info = {
            "round": round,
            "noise_multiplier": noise_multiplier,
            "sampled_clients": num_sampled_clients,
            **_extract_info_from_dp_state(dp_aggregator_state=state.aggregator)
        }
        print(f"Round Information: {round_info}", flush=True,)  # , flush=True, file=sys.stderr
        if round % eval_after_rounds == 0:
            model_weights = learning_process.get_model_weights(state)
            metrics = eval_process(model_weights, [test_data])['eval']
            if round < 25 or round % 25 == 0:
                print(f'Round {round:3d}: {metrics}')

            history.append({
                **round_info,
                **metrics,
            })
        
        sampled_clients = sample_clients_with_dp(client_sampling_rate, np.array(train_data.client_ids))
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]
        num_sampled_clients = len(sampled_clients)

        result = learning_process.next(state, sampled_train_data)
        state = result.state
        metrics = result.metrics
    
    model_weights = learning_process.get_model_weights(state)
    metrics = eval_process(model_weights, [test_data])['eval']
    print(f'Round {rounds:3d}: {metrics}', flush=True)
    history.append({
        **round_info,
        **metrics,
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

def run_training(
        train_ds, 
        test_ds, 
        client_optimizer_fn, 
        server_optimizer_fn, 
        model_fn, 
        dp_level, 
        rounds, 
        clients_per_round, 
        budgets, 
        budget_ratios, 
        target_delta
):
    trained_weights, train_history = None, None
    num_train_clients = len(train_ds.client_ids)

    if dp_level == 'nodp':
        trained_weights, train_history = train_without_dp(
            model_fn=model_fn,
            client_optimizer_fn=client_optimizer_fn,
            server_optimizer_fn=server_optimizer_fn,
            train_data=train_ds,
            test_data=test_ds,
            rounds=rounds,
            clients_per_round=clients_per_round,    
        )
    elif dp_level == 'dp':
        assert len(budgets) == 1
        budgets = budgets[0]
        assert (len(budget_ratios) == 1) and (budget_ratios[0] == 1.0)
        noise_multiplier = get_noise_multiplier(
            pp_budget=budgets, 
            target_delta=target_delta,
            sample_rate=clients_per_round / len(train_ds.client_ids),
            steps=rounds,
        )
    
        trained_weights, train_history = train_with_dp(
            model_fn=model_fn,
            client_optimizer_fn=client_optimizer_fn,
            server_optimizer_fn=server_optimizer_fn,
            train_data=train_ds,
            test_data=test_ds,
            rounds=rounds,
            noise_multiplier=noise_multiplier,
            clients_per_round=clients_per_round,
        )
    elif dp_level == 'idp':
        budgets_per_client = create_budgets(
            num_clients=len(train_ds.client_ids),
            possible_budgets=budgets,
            budget_ratios=budget_ratios,
        )
        noise_multiplier, qs_per_budget = get_weights(
            pp_budgets=budgets_per_client,
            target_delta=target_delta,
            default_sample_rate=clients_per_round / num_train_clients,
            steps=rounds,
        )

        client_sampling_rates = get_sampling_rates_per_client(
            budgets_per_client=budgets_per_client, 
            budgets=budgets, 
            sampling_rates_per_budget=qs_per_budget
        )

        trained_weights, train_history = train_with_idp(
            model_fn=model_fn,
            client_optimizer_fn=client_optimizer_fn,
            server_optimizer_fn=server_optimizer_fn,
            train_data=train_ds,
            test_data=test_ds,
            client_sampling_rates=client_sampling_rates,
            rounds=rounds,
            noise_multiplier=noise_multiplier,
            clients_per_round=clients_per_round,
        )
    else:
        raise NotImplementedError(f"dp_level {dp_level} is not available")
    
    return trained_weights, train_history