import sys
from pathlib import Path
from dataclasses import dataclass, asdict

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


def save_train_results(save_dir, trained_tff_model, trained_keras_model, history):
    save_dir = Path(save_dir)
    history_df = pd.DataFrame.from_records(map(asdict, history))
    history_df.to_pickle(save_dir / "history.pkl")
    history_df.to_csv(save_dir / "history.csv")
    tff.learning.models.save(trained_tff_model, path=str(save_dir / "trained_tff_model"))
    trained_keras_model.save(str(save_dir / "trained_model.keras"))


@dataclass
class RoundRecord:
    round: int
    round_time: float
    sampled_clients: int
    noise_multiplier: float
    noise_multiplier_after_adaptive_clipping: float
    update_clipping_norm: float
    update_stddev: float
    accuracy: float
    loss: float



def _training_loop(learning_process, eval_process, client_sample_fn, train_data, test_data, rounds, eval_after_rounds, noise_multiplier):
    is_with_dp = noise_multiplier is not None
    history = []
    state = learning_process.initialize()
    for round in tqdm(range(rounds)):
        metrics = {}
        dp_info = {}
        
        if round % eval_after_rounds == 0:
            model_weights = learning_process.get_model_weights(state)
            metrics = eval_process(model_weights, [test_data])['eval']

        sampled_clients = client_sample_fn(np.array(train_data.client_ids))
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]
        num_sampled_clients = len(sampled_clients)
        if is_with_dp:
            dp_info = _extract_info_from_dp_state(dp_aggregator_state=state.aggregator)
        
        round_record = RoundRecord(
            round=round,
            round_time=None,
            sampled_clients=num_sampled_clients,
            noise_multiplier=noise_multiplier,
            noise_multiplier_after_adaptive_clipping=dp_info.get("noise_multiplier_after_adaptive_clipping", None),
            update_clipping_norm=dp_info.get("sum_clipping_norm", None),
            update_stddev=dp_info.get("sum_stddev", None),
            accuracy=metrics.get("sparse_categorical_accuracy", None),
            loss=metrics.get("loss", None)
        )
        history.append(round_record)
        print(f"Round {round}:", metrics, dp_info, flush=True)

        result = learning_process.next(state, sampled_train_data)
        state = result.state
        metrics = result.metrics
    
    model_weights = learning_process.get_model_weights(state)
    metrics = eval_process(model_weights, [test_data])['eval']
    if is_with_dp:
        dp_info = _extract_info_from_dp_state(dp_aggregator_state=state.aggregator)
    round_record = RoundRecord(
        round=round + 1,  # round needs to be incremented once more
        round_time=None,
        sampled_clients=num_sampled_clients,
        noise_multiplier=noise_multiplier,
        noise_multiplier_after_adaptive_clipping=dp_info.get("noise_multiplier_after_adaptive_clipping", None),
        update_clipping_norm=dp_info.get("sum_clipping_norm", None),
        update_stddev=dp_info.get("sum_stddev", None),
        accuracy=metrics.get("sparse_categorical_accuracy", None),
        loss=metrics.get("loss", None)
    )
    print(f"Round {round}:", metrics, dp_info, flush=True)
    history.append(round_record)

    return model_weights, history


def train_without_dp(model_fn, client_optimizer_fn, server_optimizer_fn, train_data, test_data, rounds, clients_per_round, eval_after_rounds=5):
    learning_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        use_experimental_simulation_loop=True,  # is suggested here https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators
    )
    eval_process = tff.learning.build_federated_evaluation(model_fn)
    
    def client_sample_fn(client_ids):
        return sample_clients_without_dp(clients_per_round, client_ids)
    
    return _training_loop(
        learning_process=learning_process,
        eval_process=eval_process,
        client_sample_fn=client_sample_fn,
        train_data=train_data,
        test_data=test_data,
        rounds=rounds,
        eval_after_rounds=eval_after_rounds,
        noise_multiplier=None
    )


def train_with_dp(model_fn, client_optimizer_fn, server_optimizer_fn, train_data, test_data, rounds, noise_multiplier, clients_per_round, eval_after_rounds=5):
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
    def client_sample_fn(client_ids):
        return sample_clients_with_dp(client_sampling_rate, client_ids)

    return _training_loop(
        learning_process=learning_process,
        eval_process=eval_process,
        client_sample_fn=client_sample_fn,
        train_data=train_data,
        test_data=test_data,
        rounds=rounds,
        eval_after_rounds=eval_after_rounds,
        noise_multiplier=noise_multiplier,
    )


def train_with_idp(model_fn, client_optimizer_fn, server_optimizer_fn, train_data, test_data, client_sampling_rates, rounds, noise_multiplier, clients_per_round, eval_after_rounds=5):
    aggregation_factory = tff.learning.dp_aggregator(noise_multiplier, clients_per_round)
    learning_process = tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        model_aggregator=aggregation_factory,
        use_experimental_simulation_loop=True,
    )
    eval_process = tff.learning.build_federated_evaluation(model_fn)

    def client_sample_fn(client_ids):
        return sample_clients_with_idp(client_sampling_rates, client_ids)
    
    return _training_loop(
        learning_process=learning_process,
        eval_process=eval_process,
        client_sample_fn=client_sample_fn,
        train_data=train_data,
        test_data=test_data,
        rounds=rounds,
        eval_after_rounds=eval_after_rounds,
        noise_multiplier=noise_multiplier,
    )


# def run_training(
#         train_ds, 
#         test_ds, 
#         client_optimizer_fn, 
#         server_optimizer_fn, 
#         model_fn, 
#         dp_level, 
#         rounds, 
#         clients_per_round, 
#         budgets, 
#         budget_ratios, 
#         target_delta
# ):
#     train_clients = np.array(train_ds.client_ids)
#     learning_process = tff.learning.algorithms.build_weighted_fed_avg(
#         model_fn=model_fn,
#         client_optimizer_fn=client_optimizer_fn,
#         server_optimizer_fn=server_optimizer_fn,
#         use_experimental_simulation_loop=True,  # is suggested here https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators
#     )
#     eval_process = tff.learning.build_federated_evaluation(model_fn)
#     def training_selection_fn(round):
#         sampled_client_ids = sample_clients_without_dp(clients_per_round, train_clients)
#         return [
#             train_ds.create_tf_dataset_for_client(client)
#             for client in sampled_client_ids
#         ]
    



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