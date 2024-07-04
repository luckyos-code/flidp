from pathlib import Path

from . import *
from .helpers import create_budgets, get_sampling_rates_per_client
from idputils import get_weights
from train import train_without_dp, train_with_idp, save_train_results

CIFAR_DELTA = 1e-4  # tff cifar dataset contains 500 clients
CIFAR_ROUNDS = 200
CIFAR_CLIENTS_PER_ROUND = 100
CIFAR_LOCAL_EPOCHS = 3

def _get_dataset():
    train_ds, test_ds = tff.simulation.datasets.cifar100.load_data()
    def element_fn(element):
        return collections.OrderedDict(
            x=tf.cast(element['image'], dtype=tf.float32) / 255., y=element['coarse_label']
        )
    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 100 for Federated CIFAR100
        return (
            dataset
            .map(element_fn)
            .shuffle(buffer_size=100)
            .repeat(CIFAR_LOCAL_EPOCHS)
            .batch(128, drop_remainder=False)
        )

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(128, drop_remainder=False)

    cifar_train = train_ds.preprocess(preprocess_train_dataset)
    cifar_test = preprocess_test_dataset(
        test_ds.create_tf_dataset_from_all_clients()
    )
    
    return cifar_train, cifar_test

def _get_model(input_spec):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(20),
    ])
    return tff.learning.models.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        input_spec=input_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )


def run_cifar(save_dir, budgets, budget_ratios, dp_level):
    def model_fn():
        return _get_model(test_ds.element_spec)
    
    train_ds, test_ds = _get_dataset()
    client_optimizer_fn = lambda: tf.keras.optimizers.Adam(1e-3)
    server_optimizer_fn = lambda: tf.keras.optimizers.SGD(1.0, momentum=0.9)

    trained_weights, train_history = None, None
    if dp_level == 'idp':
        budgets_per_client = create_budgets(
            num_clients=len(train_ds.client_ids),
            possible_budgets=budgets,
            budget_ratios=budget_ratios,
        )
        noise_multiplier, qs_per_budget = get_weights(
            pp_budgets=budgets_per_client,
            target_delta=CIFAR_DELTA,
            default_sample_rate=CIFAR_CLIENTS_PER_ROUND / len(train_ds.client_ids),
            steps=CIFAR_ROUNDS,
        )

        client_sampling_rates = get_sampling_rates_per_client(
            budgets_per_client=budgets_per_client, 
            budgets=budgets, 
            sampling_rates_per_budget=qs_per_budget
        )

        # print("-------------------------------------------------------------")
        # print(noise_multiplier)
        # print(qs_per_budget)
        # print(budgets_per_client)
        # print(client_sampling_rates)
        # print("-------------------------------------------------------------")

        trained_weights, train_history = train_with_idp(
            model_fn=model_fn,
            client_optimizer_fn=client_optimizer_fn,
            server_optimizer_fn=server_optimizer_fn,
            train_data=train_ds,
            test_data=test_ds,
            client_sampling_rates=client_sampling_rates,
            rounds=CIFAR_ROUNDS,
            noise_multiplier=noise_multiplier,
            clients_per_round=CIFAR_CLIENTS_PER_ROUND,
        )

    elif dp_level == 'nodp':
        trained_weights, train_history = train_without_dp(
            model_fn=model_fn,
            client_optimizer_fn=client_optimizer_fn,
            server_optimizer_fn=server_optimizer_fn,
            train_data=train_ds,
            test_data=test_ds,
            rounds=CIFAR_ROUNDS,
            clients_per_round=CIFAR_CLIENTS_PER_ROUND,    
        )
    else:
        raise NotImplementedError(f"dp_level {dp_level} is not available")
    Path(save_dir).mkdir(parents=True)
    save_train_results(save_dir, trained_weights, train_history)


if __name__ == "__main__":
    print("Biggest Client dataset: ", max(len(list(iter(d))) for d in tff.simulation.datasets.cifar100.load_data()[0].datasets()))