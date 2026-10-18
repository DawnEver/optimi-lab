"""D116 risk operators and D117 estimators.

Risk operators are functions of a weighted discrete distribution of a LOSS (larger is worse); their
values are checked against hand arithmetic and against the closed-form normal tail. Estimators
return a value with its standard error, checked by coverage of a known truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import (
    Estimate,
    cvar,
    expectation,
    joint_probability,
    mean_estimate,
    mean_plus_std,
    probability,
    replicated,
    satisfied,
    value_at_risk,
    worst_case,
)
from optimi_lab.problem import Chance

#: The standard normal CVaR at 95 %: phi(1.6449) / 0.05 (dimensionless, standard units).
NORMAL_CVAR_95 = 2.062712807507036
#: Sampling error budget for 400 000 normal draws, in the same standard units.
NORMAL_TOLERANCE = 0.02


def test_d116_risk_measures_of_a_weighted_distribution_are_hand_arithmetic() -> None:
    values, weights = np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.3, 0.5])
    assert expectation(values, weights) == 0.2 + 0.6 + 1.5
    assert value_at_risk(values, weights, alpha=0.5) == 2.0
    assert value_at_risk(values, weights, alpha=0.51) == 3.0
    assert cvar(values, weights, alpha=0.5) == 3.0
    assert worst_case(values) == 3.0
    assert mean_plus_std(np.array([0.0, 2.0]), None, k=2.0) == 3.0


def test_d116_cvar_of_a_normal_sample_is_the_closed_form_tail_mean() -> None:
    sample = np.random.default_rng(0).standard_normal(400_000)
    assert abs(cvar(sample, None, alpha=0.95) - NORMAL_CVAR_95) <= NORMAL_TOLERANCE
    assert cvar(sample, None, alpha=0.95) >= value_at_risk(sample, None, alpha=0.95) >= expectation(sample, None)


def test_d116_planted_control_a_distribution_that_is_not_one_is_refused() -> None:
    with pytest.raises(Refusal, match='sum to 1'):
        cvar(np.array([0.0, 1.0]), np.array([0.5, 0.6]), alpha=0.9)
    with pytest.raises(Refusal, match='non-negative'):
        expectation(np.array([0.0, 1.0]), np.array([1.5, -0.5]))
    with pytest.raises(Refusal, match=r'\(0, 1\)'):
        cvar(np.array([0.0, 1.0]), None, alpha=1.0)


def test_d117_a_mean_estimate_covers_the_truth_within_its_standard_error() -> None:
    found = mean_estimate(np.random.default_rng(1).exponential(2.0, 50_000))
    assert found.n == 50_000 and abs(found.value - 2.0) <= 4 * found.std_error
    assert abs(found.std_error - 2.0 / np.sqrt(50_000)) <= 0.05 * found.std_error


def test_d117_a_probability_and_a_chance_constraint_read_the_same_events() -> None:
    values = np.random.default_rng(2).normal(0.0, 1.0, 100_000)
    constraint = Chance('margin', 'm', '<=', 1.0, probability=0.8)
    held = satisfied(values, constraint)
    assert held.value == probability(values <= 1.0).value
    assert abs(held.value - 0.8413447460685429) <= 4 * held.std_error


def test_d117_replicates_pool_into_their_mean_and_spread() -> None:
    pooled = replicated([Estimate(1.0, 0.0, 10), Estimate(3.0, 0.0, 10)])
    assert (pooled.value, pooled.n) == (2.0, 20)
    assert pooled.std_error == np.sqrt(2.0 / 2)  # sample variance 2 over 2 replicates


def test_d117_planted_control_an_estimator_without_spread_is_refused() -> None:
    with pytest.raises(Refusal, match='two'):
        mean_estimate(np.array([1.0]))
    with pytest.raises(Refusal, match='two'):
        replicated([Estimate(1.0, 0.0, 10)])
    with pytest.raises(Refusal, match='boolean'):
        joint_probability(np.array([[0.2, 0.9]]))
