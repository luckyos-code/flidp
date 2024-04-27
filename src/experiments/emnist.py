from . import *

class EMNISTExperiment(Experiment):
    
    def __init__(self, budgets, budget_ratios, only_digits):
        super().__init__(budgets, budget_ratios)
        self._train, self._test = tff.simulation.datasets.emnist.load_data(only_digits=only_digits)

    def get_dataset(self) -> Tuple[tff.simulation.datasets.ClientData, tff.simulation.datasets.ClientData]:

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

        emnist_train = self._train.preprocess(preprocess_train_dataset)
        emnist_test = preprocess_test_dataset(
            self._test.create_tf_dataset_from_all_clients()
        )
        
        return emnist_train, emnist_test
    
    def get_model(self, input_spec) -> tff.learning.models.VariableModel:
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