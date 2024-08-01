import os
from pathlib import Path

import numpy as np

from . import *
from .helpers import create_budgets, get_sampling_rates_per_client
from .models import get_model
from idputils import get_weights, get_noise_multiplier
from train import save_train_results, run_training


CIFAR10_DIR = os.path.join(os.path.expanduser("~"), ".tff/cifar10")
IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
DELTA = 1e-4  # tff cifar dataset contains 500 clients
RESCALE_FACTOR = 1/255.


def _load_cifar10():
    tensor_spec = {
        'image': tf.TensorSpec((32, 32, 3), dtype=tf.int64),
        'label': tf.TensorSpec((), dtype=tf.int64),
    }
    train_client_data =  tff.simulation.datasets.load_and_parse_sql_client_data(str(Path(CIFAR10_DIR) / 'train.sqlite'), element_spec=tensor_spec, split_name=None)
    test_client_data = tff.simulation.datasets.load_and_parse_sql_client_data(str(Path(CIFAR10_DIR) / 'test.sqlite'), element_spec=tensor_spec, split_name=None)
    return train_client_data, test_client_data


def _get_dataset(local_epochs, batch_size):
    train_ds, test_ds = _load_cifar10()
    def element_fn(element):
        return collections.OrderedDict(
            x=element['image'], y=element['label']
        )
    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 100 for Federated CIFAR100
        return (
            dataset
            .map(element_fn)
            .shuffle(buffer_size=100)
            .repeat(local_epochs)
            .batch(batch_size, drop_remainder=False)
        )

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(batch_size, drop_remainder=False)

    cifar_train = train_ds.preprocess(preprocess_train_dataset)
    cifar_test = preprocess_test_dataset(
        test_ds.create_tf_dataset_from_all_clients()
    )
    
    return cifar_train, cifar_test


def _get_model(model_name, input_spec):
    model = get_model(model_name, input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, rescale_factor=RESCALE_FACTOR)
    return tff.learning.models.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=input_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )


def run_cifar10(save_dir, model, budgets, ratios, dp_level, rounds, clients_per_round, local_epochs, batch_size, client_lr, server_lr):
    def model_fn():
        return _get_model(model, test_ds.element_spec)
    
    train_ds, test_ds = _get_dataset(batch_size=batch_size, local_epochs=local_epochs)
    client_optimizer_fn = lambda: tf.keras.optimizers.Adam(client_lr)
    server_optimizer_fn = lambda: tf.keras.optimizers.Adam(server_lr)

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
