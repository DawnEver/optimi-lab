"""Variance-based sensitivity against the Ishigami function's closed-form indices.

Ishigami, ``f = sin x1 + a sin^2 x2 + b x3^4 sin x1`` on ``[-pi, pi]^3`` with ``a = 7, b = 0.1``,
has analytic first-order and total indices (Sobol' and Levitan 1999), so every reading below is
judged against arithmetic, never against a previous run of the estimator.

TOLERANCES ARE ABSOLUTE ON A DIMENSIONLESS INDEX, AND EACH NAMES ITS SAMPLE SIZE. With a Saltelli
base of ``n = 2**13`` (``n (k + 2) = 40960`` evaluations) the Jansen total and Saltelli first-order
estimators sit within 0.02 of the truth; the binned main effect on 20000 random points within 0.03.
"""

import numpy as np
import pytest

from optimi_lab import Refusal, Variable, VariableSet
from optimi_lab.analysis import SobolIndices, main_effect, sobol_indices

A, B = 7.0, 0.1
SPACE = VariableSet(tuple(Variable(f'x{i}', -np.pi, np.pi) for i in (1, 2, 3)))

_V = A**2 / 8 + B * np.pi**4 / 5 + B**2 * np.pi**8 / 18 + 0.5
_V1 = 0.5 * (1 + B * np.pi**4 / 5) ** 2
_V2 = A**2 / 8
_VT3 = 8 * B**2 * np.pi**8 / 225
FIRST = np.array([_V1, _V2, 0.0]) / _V
TOTAL = np.array([_V1 + _VT3, _V2, _VT3]) / _V


def ishigami(x: np.ndarray) -> np.ndarray:
    return np.sin(x[:, 0]) + A * np.sin(x[:, 1]) ** 2 + B * x[:, 2] ** 4 * np.sin(x[:, 0])


def test_sobol_recovers_the_ishigami_indices_at_n_8192():
    indices = sobol_indices(SPACE, ishigami, 2**13, seed=0)
    assert isinstance(indices, SobolIndices)
    assert indices.names == ('x1', 'x2', 'x3')
    assert indices.n_evaluations == 2**13 * 5
    assert indices.first_order == pytest.approx(FIRST, abs=0.02)
    assert indices.total == pytest.approx(TOTAL, abs=0.02)


def test_the_bootstrap_interval_brackets_its_estimate_and_narrows_with_n():
    small = sobol_indices(SPACE, ishigami, 2**9, seed=1)
    large = sobol_indices(SPACE, ishigami, 2**13, seed=1)
    for indices in (small, large):
        assert indices.first_order_interval.shape == (3, 2)
        assert np.all(indices.first_order_interval[:, 0] <= indices.first_order)
        assert np.all(indices.total_interval[:, 1] >= indices.total)
    width = lambda interval: float(np.mean(interval[:, 1] - interval[:, 0]))  # noqa: E731
    assert width(large.total_interval) < width(small.total_interval)


def test_one_seed_is_one_answer():
    first = sobol_indices(SPACE, ishigami, 2**8, seed=4)
    again = sobol_indices(SPACE, ishigami, 2**8, seed=4)
    assert np.array_equal(first.total, again.total)
    assert np.array_equal(first.total_interval, again.total_interval)


def test_sobol_refuses_a_base_that_is_not_a_power_of_two():
    with pytest.raises(Refusal, match='power of two'):
        sobol_indices(SPACE, ishigami, 1000, seed=0)


def test_sobol_refuses_an_output_that_is_not_one_value_per_point():
    with pytest.raises(Refusal, match='one value per point'):
        sobol_indices(SPACE, lambda x: x, 2**6, seed=0)


def test_sobol_refuses_a_constant_function():
    with pytest.raises(Refusal, match='variance'):
        sobol_indices(SPACE, lambda x: np.ones(len(x)), 2**6, seed=0)


def test_main_effect_reads_the_first_order_indices_from_an_existing_sample():
    points = SPACE.scale(np.random.default_rng(2).uniform(size=(20000, 3)))
    assert main_effect(points, ishigami(points), n_bins=20) == pytest.approx(FIRST, abs=0.03)


def test_main_effect_refuses_misaligned_rows():
    with pytest.raises(Refusal, match='row'):
        main_effect(np.zeros((10, 2)), np.zeros(9))
