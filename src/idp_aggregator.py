from typing import Any, NamedTuple, Iterable

import tensorflow_privacy as tfp
import tensorflow_federated as tff
from tensorflow_federated.python.aggregators.factory import ValueType


class DPAggregatorState(NamedTuple):
    query_state: Any
    agg_state: Any
    dp_event: Any
    is_init_state: Any


class IndividualDifferentiallyPrivateFactory(tff.aggregators.UnweightedAggregationFactory):

    @classmethod
    def gaussian_fixed(cls, noise_multiplier: float, clients_per_round: Iterable[float], clip: float) -> tff.aggregators.UnweightedAggregationFactory:
        queries = [
            tfp.NormalizedQuery(
                tfp.GaussianSumQuery(l2_norm_clip=clip, stddev=clip * noise_multiplier),
                denominator=cpr,
            )
            for cpr in clients_per_round
        ]

        return cls(queries)

    def __init__(self, queries: Iterable[tfp.DPQuery]):
        super().__init__()
        self._queries = queries

    def create(self, value_type: ValueType) -> tff.templates.AggregationProcess:
        pass