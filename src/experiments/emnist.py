from pathlib import Path

from . import *
from .helpers import create_budgets, get_sampling_rates_per_client
from idputils import get_weights
from train import train, save_train_results

EMNIST_DELTA = 1e-5
EMNIST_ROUNDS = 100
EMNIST_CLIENTS_PER_ROUND = 100


def _get_dataset(only_digits) -> Tuple[tff.simulation.datasets.ClientData, tff.simulation.datasets.ClientData]:
    train_ds, test_ds = tff.simulation.datasets.emnist.load_data(only_digits=only_digits)
    def element_fn(element):
        return collections.OrderedDict(
            x=tf.expand_dims(element['pixels'], -1), y=element['label']
        )

    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 418 for Federated EMNIST
        return (
            dataset
            .map(element_fn)
            .shuffle(buffer_size=418)
            .repeat(1)
            .batch(32, drop_remainder=False)
        )

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(128, drop_remainder=False)

    emnist_train = train_ds.preprocess(preprocess_train_dataset)
    emnist_test = preprocess_test_dataset(
        test_ds.create_tf_dataset_from_all_clients()
    )
    
    return emnist_train, emnist_test
    

def _get_model(input_spec) -> tff.learning.models.VariableModel:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dense(10)
    ])
    return tff.learning.models.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        input_spec=input_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )


def run_emnist(save_dir, budgets, budget_ratios):
    train_ds, test_ds = _get_dataset(only_digits=True)
    budgets_per_client = create_budgets(
        num_clients=len(train_ds.client_ids),
        possible_budgets=budgets,
        budget_ratios=budget_ratios,
    )
    noise_multiplier, qs_per_budget = get_weights(
        pp_budgets=budgets_per_client,
        target_delta=EMNIST_DELTA,
        default_sample_rate=EMNIST_CLIENTS_PER_ROUND / len(train_ds.client_ids),
        steps=EMNIST_ROUNDS,
    )

    client_sampling_rates = get_sampling_rates_per_client(
        budgets_per_client=budgets_per_client, 
        budgets=budgets, 
        sampling_rates_per_budget=qs_per_budget
    )

    def model_fn():
        return _get_model(test_ds.element_spec)

    trained_weights, train_history = train(
        model_fn=model_fn,
        train_data=train_ds,
        test_data=test_ds,
        client_sampling_rates=client_sampling_rates,
        rounds=EMNIST_ROUNDS,
        noise_multiplier=noise_multiplier,
        clients_per_round=EMNIST_CLIENTS_PER_ROUND,
    )

    Path(save_dir).mkdir(parents=True)
    save_train_results(save_dir, trained_weights, train_history)