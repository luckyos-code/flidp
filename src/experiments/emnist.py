from pathlib import Path

from . import *
from .helpers import create_budgets, get_sampling_rates_per_client
from .models import get_model
from idputils import get_weights
from train import train_without_dp, train_with_idp, save_train_results, run_training

IMAGE_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
DELTA = 1e-5
RESCALE_FACTOR = 1/1.


def _get_dataset(local_epochs, batch_size, only_digits) -> Tuple[tff.simulation.datasets.ClientData, tff.simulation.datasets.ClientData]:
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
            .repeat(local_epochs)
            .batch(batch_size, drop_remainder=False)
        )

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(batch_size, drop_remainder=False)

    emnist_train = train_ds.preprocess(preprocess_train_dataset)
    emnist_test = preprocess_test_dataset(
        test_ds.create_tf_dataset_from_all_clients()
    )
    
    return emnist_train, emnist_test
    

def _get_model(model_name, input_spec) -> tff.learning.models.VariableModel:
    # pixels are already scaled to [0,1] (https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)
    model = get_model(model_name, input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, rescale_factor=RESCALE_FACTOR)
    return tff.learning.models.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=input_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )


def run_emnist(save_dir, model, budgets, ratios, dp_level, rounds, clients_per_round, local_epochs, batch_size, client_lr, server_lr):
    def model_fn():
        return _get_model(model, test_ds.element_spec)
    
    train_ds, test_ds = _get_dataset(local_epochs, batch_size, only_digits=True)
    client_optimizer_fn = lambda: tf.keras.optimizers.Adam(client_lr)
    server_optimizer_fn = lambda: tf.keras.optimizers.SGD(server_lr, momentum=0.9)

    trained_weights, train_history = run_training(
        train_ds=train_ds,
        test_ds=test_ds,
        budgets=budgets,
        budget_ratios=ratios,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        model_fn=model_fn,
        dp_level=dp_level, 
        rounds=rounds,
        clients_per_round=clients_per_round,
        target_delta=DELTA,
    )

    Path(save_dir).mkdir(parents=True)
    save_train_results(save_dir, trained_weights, train_history)