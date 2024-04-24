from typing import Literal
import numpy as np

from dp_accounting.rdp import RdpAccountant
from dp_accounting.pld import PLDAccountant
from dp_accounting.dp_event import PoissonSampledDpEvent, GaussianDpEvent, SelfComposedDpEvent
from dp_accounting import calibrate_dp_mechanism
from dp_accounting.privacy_accountant import PrivacyAccountant

import sample_old

MAX_SIGMA = 1e6
MIN_Q = 1e-9
MAX_Q = 0.1


def make_accountant(accountant_type: Literal["rdp", "pld"]) -> PrivacyAccountant:
    if accountant_type == "rdp":
        return RdpAccountant()
    elif accountant_type == "pld":
        return PLDAccountant()
    else:
        raise NotImplementedError("Unknown accountant")


def make_dp_event(noise_multiplier: float, q: float, steps: int):
    g_ev = GaussianDpEvent(noise_multiplier)
    p_ev = PoissonSampledDpEvent(sampling_probability=q, event=g_ev)
    c_ev = SelfComposedDpEvent(p_ev, steps)
    return c_ev


def get_sample_rate(target_epsilon: float, target_delta: float, noise_multiplier: float, steps: int, accountant_type: Literal["rdp", "pld"] = "rdp"):
    # search interval for q is implicitly set to [0,1] with this call to calibrate_dp_mechanism
    sample_rate = calibrate_dp_mechanism(
        make_fresh_accountant=lambda: make_accountant(accountant_type),
        make_event_from_param=lambda q: make_dp_event(noise_multiplier, q, steps),
        target_epsilon=target_epsilon,
        target_delta=target_delta,
    )
    return sample_rate


def get_epsilon(composed_dp_event: SelfComposedDpEvent, target_delta: float, accountant_type: Literal["rdp"] = "rdp") -> float:
    accountant = make_accountant(accountant_type)
    accountant.compose(composed_dp_event)

    return accountant.get_epsilon(target_delta=target_delta)

def get_sample_rates(
    ratios: list[float],
    target_epsilons: list[float],
    target_delta: float,
    default_sample_rate: float,
    steps: int,
    precision: float = 0.001,
    **kwargs,
) -> (float, np.ndarray):
    r"""
    Computes via nested binary search the sampling frequency q for each privacy
    group to reach a total budget of (target_epsilon, target_delta) at the end
    of epochs, with a given default_sample_rate.
    Args:
        ratios: relative size of each privacy group within the training dataset
        target_epsilons: the privacy budget's epsilon for each privacy group
        target_delta: the privacy budget's delta
        default_sample_rate: sampling frequency to achieve expected_batch_size
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        precision: relation between limits of binary search interval
    Returns:
        The noise level sigma, and each a sampling frequency q for each privacy
        group to ensure privacy budgets of target_epsilons with target_delta
    """
    assert len(ratios) == len(target_epsilons), f"ratios and target_epsilons must have the same length"
    n_groups = len(ratios)
    ratios = np.asarray(ratios)
    sigma_low, sigma_high = 1e-3, 10
    for group, target_epsilon in enumerate(target_epsilons):
        eps_high = float("inf")
        sigma_high_group = 10
        while eps_high > target_epsilon:
            sigma_high_group = 2 * sigma_high_group
            if sigma_high_group > sigma_high:
                sigma_high = sigma_high_group
            ev = make_dp_event(sigma_high_group, default_sample_rate, steps)
            eps_high = get_epsilon(ev, target_delta=target_delta)
            if sigma_high_group > MAX_SIGMA:
                raise ValueError(f"The privacy budget ({target_epsilon}) of"
                                 f"group {group} is too low.")

    q_mean = MAX_Q
    qs = np.array([q_mean] * n_groups)
    while sigma_low / sigma_high < 1 - precision:
        sigma = (sigma_high + sigma_low) / 2
        q_mean = 0
        for group, target_epsilon in enumerate(target_epsilons):
            try:
                q = get_sample_rate(
                    target_epsilon=target_epsilon,
                    target_delta=target_delta,
                    noise_multiplier=sigma,
                    steps=steps,
                    **kwargs,
                )
                qs[group] = q
                q_mean += q * ratios[group]
                if q_mean > default_sample_rate:
                    sigma_high = sigma
                    break
            except ValueError:
                continue
        q_mean = sum(qs * ratios)
        if q_mean > default_sample_rate:
            sigma_high = sigma
        else:
            sigma_low = sigma
    return sigma_high, list(qs)


def get_weights(
    pp_budgets: np.ndarray,
    target_delta: float,
    default_sample_rate: float,
    steps: int,
    precision: float = 0.001,
    **kwargs,
) -> (float, list[float]):
    r"""
    Computes max_grad_norms from given default_max_grad_norm (in case of
    individualize=="clipping") or sample_rates (in case of
    individualize=="sampling") such that all budgets would be exhausted after
    given steps.
    Args:
        pp_budgets: the privacy budget's epsilon for each data point
        target_delta: the privacy budget's delta
        default_max_grad_norm: average clipping threshold over privacy groups
        default_sample_rate: sampling frequency to achieve expected_batch_size
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        precision: relation between limits of binary search interval
    Returns:
        The default noise_multiplier and clipping thresholds or sampling
        frequencies for each privacy group to align with target_epsilons and
        target_delta.
    """
    budgets = list(np.sort(np.unique(pp_budgets)))
    ratios = [sum(pp_budgets == b) / len(pp_budgets) for b in budgets]

    noise_multiplier, qs_per_budget = get_sample_rates(
        ratios=ratios,
        target_epsilons=budgets,
        target_delta=target_delta,
        default_sample_rate=default_sample_rate,
        steps=steps,
        precision=precision,
    )

    return noise_multiplier, qs_per_budget


if __name__ == "__main__":
    target_epsilon = 2.0
    target_delta = 1e-5
    noise_multiplier = 1.5
    steps = 1_000

    print(sample_old.get_sample_rate(target_epsilon, target_delta, noise_multiplier, steps))
    print(get_sample_rate(target_epsilon, target_delta, noise_multiplier, steps))

    target_epsilons = np.array([1.0, 2.0, 3.0])
    ratios = np.array([.34, .43, .23])
    target_delta = 1e-5
    default_sample_rate = 1 / 100
    steps = 1_000

    print(sample_old.get_sample_rates(ratios, target_epsilons, target_delta, default_sample_rate, steps))
    print(get_sample_rates(ratios, target_epsilons, target_delta, default_sample_rate, steps))