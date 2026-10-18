"""P13 gate: failure probabilities against analytic cases (D119-D122, with crude MC as the baseline).

A limit state ``g`` is a function of standard-normal rows ``z (n, d) -> (n,)``; failure is ``g <= 0``.

* LINEAR -- ``g = beta - (z1 + z2) / sqrt 2``: ``pf = Phi(-beta)`` exactly, and FORM is exact.
* PHYSICAL -- ``g = R - S``, ``R ~ N(5, 0.5)``, ``S ~ N(2, 0.6)``: ``beta = 3 / sqrt(0.61)``, exact.
* PARABOLIC -- ``g = beta - z2 + (k / 2) z1^2``: ``pf = int phi(z1) Phi(-(beta + k z1^2 / 2)) dz1``,
  evaluated here by 80-point Gauss-Hermite quadrature (a TEST reference; agreement 1e-12 relative).

Statistical estimators are judged by coverage of the truth within four of their own standard errors;
the deterministic ones carry a stated tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss, hermevander
from scipy.special import ndtr

from optimi_lab.operator import (
    crude_monte_carlo,
    form,
    importance_sampling,
    in_standard_space,
    sorm,
    subset_simulation,
    surrogate_reliability,
)
from optimi_lab.problem import Normal, Uncertainty

SQRT2 = np.sqrt(2.0)
#: FORM on a linear limit state is exact: the reliability index agrees to this (standard units).
FORM_TOLERANCE = 1e-8
#: Hohenbichler-Rackwitz SORM against the quadrature reference, relative (measured 2026-10-18: 0.50 %
#: at beta = 3, k = 0.2, where Breitung's beta-form is 2.26 % off).
SORM_RELATIVE = 0.01


def linear(beta: float):
    return lambda z: beta - (z[:, 0] + z[:, 1]) / SQRT2


def parabolic(beta: float, k: float):
    return lambda z: beta - z[:, 1] + 0.5 * k * z[:, 0] ** 2


def parabolic_truth(beta: float, k: float) -> float:
    nodes, weights = hermegauss(80)
    return float(weights @ ndtr(-(beta + 0.5 * k * nodes**2)) / np.sqrt(2 * np.pi))


def test_crude_monte_carlo_covers_a_linear_failure_probability() -> None:
    found = crude_monte_carlo(linear(2.0), 2, 200_000, np.random.default_rng(0))
    assert abs(found.value - ndtr(-2.0)) <= 4 * found.std_error


def test_d121_form_is_exact_on_a_linear_limit_state_in_physical_units() -> None:
    uncertainty = Uncertainty((Normal('R', 5.0, 0.5, unit='kN'), Normal('S', 2.0, 0.6, unit='kN')))
    g = in_standard_space(lambda x: x['R'] - x['S'], uncertainty)
    point = form(g, 2)
    assert point.converged and abs(point.beta - 3 / np.sqrt(0.61)) <= FORM_TOLERANCE
    assert abs(point.probability - ndtr(-3 / np.sqrt(0.61))) <= FORM_TOLERANCE * point.probability


def test_d121_sorm_corrects_form_on_a_curved_limit_state() -> None:
    g, truth = parabolic(3.0, 0.2), parabolic_truth(3.0, 0.2)
    point = form(g, 2)
    assert abs(point.beta - 3.0) <= FORM_TOLERANCE and np.max(np.abs(point.design_point - [0.0, 3.0])) <= 1e-6
    second = sorm(g, point)
    assert abs(second.probability - truth) <= SORM_RELATIVE * truth
    assert abs(point.probability - truth) > 5 * abs(second.probability - truth)  # the curvature mattered


def test_d119_importance_sampling_at_the_design_point_covers_a_rare_event() -> None:
    g = linear(4.0)
    found = importance_sampling(g, form(g, 2).design_point, 4000, np.random.default_rng(1))
    assert abs(found.value - ndtr(-4.0)) <= 4 * found.std_error
    assert found.std_error <= 0.05 * found.value


def test_d119_planted_control_sampling_at_the_origin_cannot_see_the_rare_event() -> None:
    found = importance_sampling(linear(4.0), np.zeros(2), 4000, np.random.default_rng(1))
    assert found.value == 0.0 or found.std_error > 0.3 * found.value


def test_d120_subset_simulation_reaches_a_rare_event_with_a_few_thousand_samples() -> None:
    """Twenty independent runs: their mean within 10 % of Phi(-4), each run 1000 samples per level."""
    g, truth = linear(4.0), ndtr(-4.0)
    runs = [subset_simulation(g, 2, 1000, np.random.default_rng(seed)) for seed in range(20)]
    values = np.array([r.value for r in runs])
    assert abs(values.mean() - truth) <= 0.1 * truth
    assert all(r.n <= 1000 * 6 for r in runs)


def test_d120_planted_control_a_safe_limit_state_ends_at_the_level_ceiling() -> None:
    found = subset_simulation(lambda z: np.ones(len(z)), 2, 200, np.random.default_rng(0), max_levels=3)
    assert found.value == 0.0 and found.n == 200 + 2 * 180  # level 0, then two levels of 180 new states


def hermite_fit(degree: int):
    """A total-degree probabilists'-Hermite least-squares fit in 2-D -- the TEST stand-in for P10's PCE."""

    def basis(z: np.ndarray) -> np.ndarray:
        a, b = hermevander(z[:, 0], degree), hermevander(z[:, 1], degree)
        return np.column_stack([a[:, i] * b[:, j] for i in range(degree + 1) for j in range(degree + 1 - i)])

    def fit(z: np.ndarray, values: np.ndarray):
        coefficients = np.linalg.lstsq(basis(z), values, rcond=None)[0]
        return lambda points: basis(points) @ coefficients

    return fit


def test_d122_a_chaos_surrogate_of_the_limit_state_gives_its_failure_probability() -> None:
    limit, truth = parabolic(2.5, 0.5), parabolic_truth(2.5, 0.5)
    rows: list[int] = []

    def g(z: np.ndarray) -> np.ndarray:
        rows.append(len(z))
        return limit(z)

    found = surrogate_reliability(g, 2, hermite_fit(2), 50, n_mc=400_000, rng=np.random.default_rng(2))
    assert sum(rows) == 50 and abs(found.value - truth) <= 4 * found.std_error


def test_d122_planted_control_a_model_that_drops_the_curvature_is_caught() -> None:
    """The tangent plane at the design point (FORM's model) in place of the fitted chaos."""
    g, truth = parabolic(2.5, 0.5), parabolic_truth(2.5, 0.5)
    tangent = lambda _z, _values: lambda points: 2.5 - points[:, 1]  # noqa: E731
    found = surrogate_reliability(g, 2, tangent, 50, n_mc=400_000, rng=np.random.default_rng(2))
    assert abs(found.value - truth) > 4 * found.std_error


@pytest.mark.parametrize('n', [0, 1])
def test_the_estimators_refuse_a_sample_too_small_to_carry_an_error(n: int) -> None:
    with pytest.raises(ValueError, match='at least two'):
        crude_monte_carlo(linear(2.0), 2, n, np.random.default_rng(0))
