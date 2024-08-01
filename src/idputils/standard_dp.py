from dp_accounting.rdp import RdpAccountant
from dp_accounting.pld import PLDAccountant
from dp_accounting.dp_event import PoissonSampledDpEvent, GaussianDpEvent, SelfComposedDpEvent, DpEvent
from dp_accounting import calibrate_dp_mechanism
from dp_accounting.privacy_accountant import PrivacyAccountant

from .helpers import make_accountant, make_dp_event


def get_noise_multiplier(pp_budget: float, target_delta: float, sample_rate: float, steps: int, accountant_type='rdp') -> float:
    noise_multiplier = calibrate_dp_mechanism(
        make_fresh_accountant=lambda: make_accountant(accountant_type),
        make_event_from_param=lambda x: make_dp_event(noise_multiplier=x, q=sample_rate, steps=steps),
        target_delta=target_delta,
        target_epsilon=pp_budget,
    )

    return noise_multiplier