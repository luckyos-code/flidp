import abc
import collections
from typing import Tuple

import tensorflow as tf
import tensorflow_federated as tff

from .emnist import run_emnist
from .svhn import run_svhn
from .cifar100 import run_cifar100
from .cifar10 import run_cifar10