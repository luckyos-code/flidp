import collections

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from tqdm.auto import tqdm

from idputils import get_weights


def create_budgets(num_clients, possible_budgets, budget_ratios):
    assert sum(budget_ratios) == 1, "All budget ratios must sum up to 1."
    return np.random.choice(possible_budgets, p=budget_ratios, size=num_clients)


def get_sampling_rates_per_client(budgets_per_client, budgets, sampling_rates_per_budget):
    sampling_rates_per_client = np.zeros(budgets_per_client.shape)
    for (b, q) in zip(budgets, sampling_rates_per_budget):
        sampling_rates_per_client[budgets_per_client == b] = q
    return sampling_rates_per_client


def sample_clients(client_sampling_rates, client_ids):
    assert len(client_ids) == len(client_sampling_rates)
    s = np.random.uniform(0., 1., len(client_ids))
    return client_ids[s < client_sampling_rates]


def get_emnist_dataset():
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        only_digits=True)

    def element_fn(element):
        return collections.OrderedDict(
            x=tf.expand_dims(element['pixels'], -1), y=element['label'])

    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 418 for Federated EMNIST
        return (dataset.map(element_fn)
                .shuffle(buffer_size=418)
                .repeat(1)
                .batch(32, drop_remainder=False))

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(128, drop_remainder=False)

    emnist_train = emnist_train.preprocess(preprocess_train_dataset)
    emnist_test = preprocess_test_dataset(
        emnist_test.create_tf_dataset_from_all_clients())
    return emnist_train, emnist_test


def model_fn(input_spec):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dense(10)])
    return tff.learning.models.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        input_spec=input_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


def append_metrics(df, round, noise_multiplier, metrics):
    return pd.concat([
        df,
        pd.DataFrame(
            [
                {
                    'Round': round,
                    'NoiseMultiplier': noise_multiplier,
                    **metrics
                }
            ]
        )
    ])


def train(train_data, test_data, client_sampling_rates, rounds, noise_multiplier, clients_per_round, data_frame):
    # need for individual aggregator? -> Updates und Noise abhängig von Sample Rate?
    aggregation_factory = tff.learning.dp_aggregator(
        noise_multiplier, clients_per_round)

    # Build a federated averaging process.
    # Typically a non-adaptive server optimizer is used because the noise in the
    # updates can cause the second moment accumulators to become very large
    # prematurely.
    def model_f():
        return model_fn(test_data.element_spec)

    learning_process = tff.learning.algorithms.build_unweighted_fed_avg(
        model_f,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0, momentum=0.9),
        model_aggregator=aggregation_factory)

    eval_process = tff.learning.build_federated_evaluation(model_f)

    # Training loop.
    state = learning_process.initialize()
    for round in tqdm(range(rounds)):
        if round % 5 == 0:
            model_weights = learning_process.get_model_weights(state)
            metrics = eval_process(model_weights, [test_data])['eval']
            if round < 25 or round % 25 == 0:
                print(f'Round {round:3d}: {metrics}')

            data_frame = append_metrics(data_frame, round, noise_multiplier, metrics)
            # data_frame = data_frame.append({'Round': round,
            #                                 'NoiseMultiplier': noise_multiplier,
            #                                 **metrics}, ignore_index=True)

        sampled_clients = sample_clients(client_sampling_rates, np.array(train_data.client_ids))
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients]

        # Use selected clients for update.
        result = learning_process.next(state, sampled_train_data)
        state = result.state
        metrics = result.metrics

    model_weights = learning_process.get_model_weights(state)
    metrics = eval_process(model_weights, [test_data])['eval']
    print(f'Round {rounds:3d}: {metrics}')
    data_frame = append_metrics(data_frame, rounds, noise_multiplier, metrics)
    # data_frame = data_frame.append({'Round': rounds,
    #                                 'NoiseMultiplier': noise_multiplier,
    #                                 **metrics}, ignore_index=True)

    return data_frame


if __name__ == '__main__':
    print("STARTING PYTON MAIN SCRIPT")
    train_ds, test_ds = get_emnist_dataset()
    budgets = np.array([1.0, 1.0, 1.0])
    budget_ratios = np.array([.34, .43, .23])
    budgets_per_client = create_budgets(len(train_ds.client_ids), budgets, budget_ratios)
    delta = 1e-5
    clients_per_round = 100
    rounds = 100

    print("CALCULATING INDIVIDUAL SAMPLING RATES")
    noise_multiplier, qs_per_budget = get_weights(
        pp_budgets=budgets_per_client,
        target_delta=delta,
        default_sample_rate=clients_per_round / len(train_ds.client_ids),
        steps=rounds,
    )
    qs_per_client = get_sampling_rates_per_client(budgets_per_client, budgets, qs_per_budget)

    print("MEAN")
    print(np.mean([len(sample_clients(qs_per_client, np.array(train_ds.client_ids))) for _ in range(1_000)]))
    train_results = train(
        train_data=train_ds,
        test_data=test_ds,
        client_sampling_rates=qs_per_client,
        rounds=rounds,
        noise_multiplier=noise_multiplier,
        clients_per_round=clients_per_round,
        data_frame=pd.DataFrame(),
    )
