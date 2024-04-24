import collections

import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
import tensorflow_federated as tff


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


def train(train_data, test_data, rounds, noise_multiplier, clients_per_round, data_frame):
    total_clients = len(train_data.client_ids)

    def modelfn():
        return model_fn(input_spec=test_data.element_spec)

    aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
        noise_multiplier, clients_per_round)

    # We use Poisson subsampling which gives slightly tighter privacy guarantees
    # compared to having a fixed number of clients per round. The actual number of
    # clients per round is stochastic with mean clients_per_round.
    sampling_prob = clients_per_round / total_clients

    # Build a federated averaging process.
    # Typically a non-adaptive server optimizer is used because the noise in the
    # updates can cause the second moment accumulators to become very large
    # prematurely.
    learning_process = tff.learning.algorithms.build_unweighted_fed_avg(
        modelfn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0, momentum=0.9),
        model_aggregator=aggregation_factory)

    eval_process = tff.learning.build_federated_evaluation(modelfn)

    # Training loop.
    state = learning_process.initialize()
    for round in range(rounds):
        if round % 5 == 0:
            model_weights = learning_process.get_model_weights(state)
            metrics = eval_process(model_weights, [test_data])['eval']
            if round < 25 or round % 25 == 0:
                print(f'Round {round:3d}: {metrics}')
            data_frame = pd.concat(
                [
                    data_frame,
                    pd.DataFrame.from_dict(
                        data={
                            'Round': [round],
                            'NoiseMultiplier': [noise_multiplier],
                            **metrics
                        }
                    )
                ]
            )

        # Sample clients for a round. Note that if your dataset is large and
        # sampling_prob is small, it would be faster to use gap sampling.
        x = np.random.uniform(size=total_clients)
        sampled_clients = [
            train_data.client_ids[i] for i in range(total_clients)
            if x[i] < sampling_prob]
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
    data_frame = pd.concat(
        [
            data_frame,
            pd.DataFrame.from_dict(
                data={
                    'Round': [rounds],
                    'NoiseMultiplier': [noise_multiplier],
                    **metrics
                }
            )
        ]
    )

    return data_frame


def main():
    emnist_train, emnist_test = get_emnist_dataset()
    data_frame = pd.DataFrame()
    rounds = 20
    clients_per_round = 10

    for noise_multiplier in [0.0, 0.5, 0.75, 1.0]:
        print(f'Starting training with noise multiplier: {noise_multiplier}')
        data_frame = train(emnist_train, emnist_test, rounds, noise_multiplier, clients_per_round, data_frame)
        print()


if __name__ == '__main__':
    main()
