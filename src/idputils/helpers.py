from typing import Literal

from dp_accounting.rdp import RdpAccountant
from dp_accounting.pld import PLDAccountant
from dp_accounting.privacy_accountant import PrivacyAccountant
from dp_accounting.dp_event import PoissonSampledDpEvent, GaussianDpEvent, SelfComposedDpEvent, DpEvent


def make_accountant(accountant_type: Literal["rdp", "pld"]) -> PrivacyAccountant:
    if accountant_type == "rdp":
        return RdpAccountant()
    elif accountant_type == "pld":
        return PLDAccountant()
    raise NotImplementedError("Unknown accountant")


def make_dp_event(noise_multiplier: float, q: float, steps: int) -> DpEvent:
    g_ev = GaussianDpEvent(noise_multiplier)
    p_ev = PoissonSampledDpEvent(sampling_probability=q, event=g_ev)
    c_ev = SelfComposedDpEvent(p_ev, steps)
    return c_ev