import os
import collections
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from .helpers import create_budgets, get_sampling_rates_per_client
from idputils import get_weights
from train import run_training, save_train_results
from .models import get_model

SVHN_DIR = os.path.join(os.path.expanduser("~"), ".tff/svhn")
DELTA = 1e-4  # I created the dataset with 725 clients
IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
RESCALE_FACTOR = 1/255.


def _load_svhn():
    svhn_spec = {
        'image': tf.TensorSpec(IMAGE_SHAPE, dtype=tf.int64),
        'label': tf.TensorSpec((), dtype=tf.int64),
    }
    train_client_data =  tff.simulation.datasets.load_and_parse_sql_client_data(str(Path(SVHN_DIR) / 'train.sqlite'), element_spec=svhn_spec, split_name=None)
    test_client_data = tff.simulation.datasets.load_and_parse_sql_client_data(str(Path(SVHN_DIR) / 'test.sqlite'), element_spec=svhn_spec, split_name=None)
    return train_client_data, test_client_data


def _get_dataset(local_epochs, batch_size):
    train_ds, test_ds = _load_svhn()
    def element_fn(element):
        return collections.OrderedDict(
            x=element['image'], y=element['label']
        )

    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # currently 138 for my SVHN dataset
        return (
            dataset
            .map(element_fn)
            .shuffle(buffer_size=batch_size)
            .repeat(local_epochs)
            .batch(batch_size, drop_remainder=False)
        )

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(batch_size, drop_remainder=False)

    svhn_train = train_ds.preprocess(preprocess_train_dataset)
    svhn_test = preprocess_test_dataset(
        test_ds.create_tf_dataset_from_all_clients()
    )
    
    return svhn_train, svhn_test


def _get_model(model_name, input_spec):
    model = get_model(model_name, input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, rescale_factor=RESCALE_FACTOR)
    return tff.learning.models.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=input_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )


def run_svhn(save_dir, model, budgets, ratios, dp_level, rounds, clients_per_round, local_epochs, batch_size, client_lr, server_lr, make_iid):
    if make_iid:
        raise NotImplementedError("no iid creation of the dataset available yet")
    def model_fn():
        return _get_model(model, test_ds.element_spec)

    train_ds, test_ds = _get_dataset(local_epochs=local_epochs, batch_size=batch_size)
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
    keras_model = get_model(model, input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, rescale_factor=RESCALE_FACTOR, compile=True)
    trained_weights.assign_weights_to(keras_model)
    tff_model = model_fn()
    trained_weights.assign_weights_to(tff_model)
    save_train_results(
        save_dir=save_dir, 
        trained_tff_model=tff_model,
        trained_keras_model=keras_model,  
        history=train_history
    )
