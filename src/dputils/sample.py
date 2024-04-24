from typing import Optional, List

import numpy as np
from numpy import ndarray
from opacus.accountants.utils import get_noise_multiplier


MIN_Q = 1e-9
MAX_Q = 0.1


def get_noise_multipliers(
    target_epsilons: List[float],
    target_delta: float,
    sample_rate: float,
    steps: int,
    accountant: str = "rdp",
    precision: float = 0.001,
    **kwargs,
) -> List[float]:
    return [get_noise_multiplier(
        target_epsilon=budget,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant=accountant,
        precision=precision,
        **kwargs,
    ) for budget in target_epsilons]


def get_sample_rate(
    target_epsilon: float,
    target_delta: float,
    noise_multiplier: float,
    steps: int,
    accountant: str = "rdp",
    precision: float = 0.001,
    **kwargs,
) -> float:
    r"""
    Computes via binary search the sampling frequency q to reach a total budget
    of (target_epsilon, target_delta) at the end of epochs, with a given
    noise_multiplier.
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        noise_multiplier: relation between noise std and clipping threshold
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        precision: relation between limits of binary search interval
    Returns:
        The sampling frequency q to ensure privacy budget of
        (target_epsilon, target_delta)
    """
    accountant = create_accountant(mechanism=accountant)
    q_low, q_high = MIN_Q, MAX_Q
    accountant.history = [(noise_multiplier, q_low, steps)]
    eps_low = accountant.get_epsilon(delta=target_delta, **kwargs)
    if eps_low > target_epsilon:
        raise ValueError("The privacy budget is too low.")
    accountant.history = [(noise_multiplier, q_high, steps)]
    eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
    while eps_high < 0:     # decrease q_high whenever a numerical error happens
        q_high *= 0.9
        accountant.history = [(noise_multiplier, q_high, steps)]
        eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
    if eps_high < target_epsilon:
        raise ValueError(f"The given noise_multiplier {noise_multiplier} is "
                         f"too high.")

    while q_low / q_high < 1 - precision:
        q = (q_low + q_high) / 2
        accountant.history = [(noise_multiplier, q, steps)]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)
        if eps < target_epsilon:
            q_low = q
        else:
            q_high = q

    return q_low


def get_sample_rates(
    ratios: List[float],
    target_epsilons: List[float],
    target_delta: float,
    default_sample_rate: float,
    steps: int,
    accountant: str = "rdp",
    precision: float = 0.001,
    **kwargs,
) -> (float, ndarray):
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
    mechanism = accountant
    n_groups = len(ratios)
    ratios = np.asarray(ratios)
    sigma_low, sigma_high = 1e-3, 10
    for group, target_epsilon in enumerate(target_epsilons):
        eps_high = float("inf")
        accountant = create_accountant(mechanism=mechanism)
        sigma_high_group = 10
        while eps_high > target_epsilon:
            sigma_high_group = 2 * sigma_high_group
            if sigma_high_group > sigma_high:
                sigma_high = sigma_high_group
            accountant.history = [
                (sigma_high_group, default_sample_rate, steps)]
            eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
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
                    accountant=mechanism,
                    precision=precision,
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
    pp_budgets: ndarray,
    target_delta: float,
    default_max_grad_norm: float,
    default_sample_rate: float,
    steps: int,
    individualize: str,
    accountant: str = "rdp",
    precision: float = 0.001,
    **kwargs,
) -> (float, List[float]):
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
        individualize: kind of (i)DP-SGD to individualize privacy protection
        accountant: accounting mechanism used to estimate epsilon
        precision: relation between limits of binary search interval
    Returns:
        The default noise_multiplier and clipping thresholds or sampling
        frequencies for each privacy group to align with target_epsilons and
        target_delta.
    """
    budgets = list(np.sort(np.unique(pp_budgets)))
    ratios = [sum(pp_budgets == b) / len(pp_budgets) for b in budgets]
    if individualize == "clipping":
        noise_multipliers = get_noise_multipliers(
            target_epsilons=budgets,
            target_delta=target_delta,
            sample_rate=default_sample_rate,
            steps=steps,
            accountant=accountant,
            precision=precision,
            **kwargs,
        )
        average_noise_multiplier = sum(np.asarray(noise_multipliers)
                                       * np.asarray(ratios))
        clip_scalars = average_noise_multiplier / np.asarray(noise_multipliers)
        max_grad_norms = default_max_grad_norm * clip_scalars
        return average_noise_multiplier, list(max_grad_norms)
    elif individualize == "sampling":
        return get_sample_rates(
            ratios=ratios,
            target_epsilons=budgets,
            target_delta=target_delta,
            default_sample_rate=default_sample_rate,
            steps=steps,
            accountant=accountant,
            precision=precision,
            **kwargs,
        )
    else:
        raise ValueError("individualize must be 'clipping' or 'sampling'!")


def assign_pp_values(
    pp_budgets: ndarray,
    values: List[float],
) -> ndarray:
    r"""
    Assigns a value to each data point according to the given per-point budgets.
    Args:
        pp_budgets: the privacy budget's epsilon for each data point
        values: list of values to be assigned to all data points
    Returns:
        An array of size equal to the training dataset size that contains one
        value for each data point.
    """
    pp_values = np.zeros(len(pp_budgets))
    for i, budget in enumerate(np.sort(np.unique(pp_budgets))):
        pp_values[pp_budgets == budget] = values[i]
    return pp_values
