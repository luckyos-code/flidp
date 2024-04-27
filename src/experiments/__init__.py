import abc
import collections
from typing import Tuple

import tensorflow as tf
import tensorflow_federated as tff

from .experiment_factory import ExperimentFactory

class Experiment(abc.ABC):
    def __init__(self, budgets, budget_ratios, **kwargs):
        self._budgets = budgets
        self._budget_ratios = budget_ratios

    def get_dataset(self) -> Tuple[tff.simulation.datasets.ClientData, tff.simulation.datasets.ClientData]:
        pass

    def get_model(self, input_spec) -> tff.learning.models.VariableModel:
        pass
