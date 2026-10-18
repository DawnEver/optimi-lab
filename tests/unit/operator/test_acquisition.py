"""D59-D60 -- acquisitions against their closed forms, Monte Carlo and asymptotic series.

Units: means, stds and improvements are in the objective's unit (dimensionless here), so an absolute
tolerance below is in that unit; a log-score tolerance is relative; a Monte Carlo comparison is held
to ``Z_GATE`` standard errors of its own estimate.
"""

import numpy as np
import pytest
from scipy.special import ndtr

from optimi_lab.core import Refusal
from optimi_lab.metric import hypervolume
from optimi_lab.operator import tchebycheff
from optimi_lab.operator.acquisition import (
    expected_hypervolume_improvement,
    expected_improvement,
    hypervolume_improvement,
    knowledge_gradient,
    log_expected_improvement,
    log_h,
    log_probability_of_feasibility,
    log_probability_of_improvement,
    noisy_expected_hypervolume_improvement,
    parego_scores,
    probability_of_improvement,
    upper_confidence_bound,
)

#: Standard errors a Monte Carlo comparison may sit apart (two-sided p ~ 6e-7 per comparison).
Z_GATE = 5.0
#: Relative, dimensionless: a log score against the log of the same closed form.
LOG_REL = 1e-10
#: Absolute, objective-unit volume: two exact hypervolume computations of one set.
VOLUME = 1e-12


def _series_log_h(z: float, terms: int = 7) -> float:
    """``log h(z)`` from ``h ~ phi(z) / z^2 sum_k (-1)^k (2k + 1)!! / z^(2k)`` (z -> -inf)."""
    total, term = 0.0, 1.0
    for k in range(terms):
        total += term
        term *= -(2 * k + 3) / z**2
    return -0.5 * z**2 - 0.5 * np.log(2 * np.pi) - 2 * np.log(-z) + np.log(total)


def test_expected_and_probability_of_improvement_match_monte_carlo():
    rng = np.random.default_rng(0)
    mean, std, best = np.array([0.3, -0.2, 1.4]), np.array([0.5, 0.1, 0.8]), 0.1
    draws = mean + std * rng.standard_normal((2_000_000, 3))
    gain = np.maximum(best - draws, 0.0)
    assert np.all(np.abs(expected_improvement(mean, std, best) - gain.mean(0)) <= Z_GATE * gain.std(0) / 1414.2)
    hit = (draws < best).astype(float)
    assert np.all(np.abs(probability_of_improvement(mean, std, best) - hit.mean(0)) <= Z_GATE * hit.std(0) / 1414.2)


def test_a_zero_std_is_the_limit_never_a_division():
    assert expected_improvement([0.5, -0.5], [0.0, 0.0], 0.0).tolist() == [0.0, 0.5]
    assert probability_of_improvement([0.5, -0.5], [0.0, 0.0], 0.0).tolist() == [0.0, 1.0]
    assert log_expected_improvement([0.5], [0.0], 0.0)[0] == -np.inf


def test_log_ei_is_the_log_of_ei_where_ei_is_representable():
    z = np.linspace(-30.0, 6.0, 400)
    std = 0.7
    mean = -z * std
    assert np.allclose(
        log_expected_improvement(mean, np.full_like(z, std), 0.0),
        np.log(expected_improvement(mean, np.full_like(z, std), 0.0)),
        rtol=LOG_REL,
        atol=0.0,
    )


@pytest.mark.parametrize('z', [-40.0, -300.0, -5e3, -1e6])
def test_log_h_follows_the_asymptotic_series_where_ei_underflows(z):
    assert log_h(np.array([z]))[0] == pytest.approx(_series_log_h(z), rel=LOG_REL, abs=0.0)


def test_planted_naive_log_of_ei_underflows_where_log_ei_does_not():
    """The control: the comparison above is not vacuous -- the naive form is -inf there."""
    with np.errstate(divide='ignore'):
        assert np.log(expected_improvement([40.0], [1.0], 0.0))[0] == -np.inf
    assert np.isfinite(log_expected_improvement([40.0], [1.0], 0.0)[0])


def test_log_ei_is_strictly_increasing_through_every_branch():
    z = -np.logspace(-3, 7, 2000)[::-1]
    assert np.all(np.diff(log_h(z)) > 0.0)


def test_log_pi_and_ucb_closed_forms():
    mean, std = np.array([0.0, 3.0, 50.0]), np.array([1.0, 0.5, 1.0])
    assert np.allclose(
        log_probability_of_improvement(mean, std, 1.0)[:2], np.log(ndtr((1.0 - mean[:2]) / std[:2])), rtol=LOG_REL
    )
    assert np.isfinite(log_probability_of_improvement(mean, std, 1.0)[2])
    assert np.allclose(upper_confidence_bound(mean, std, 4.0), -mean + 2.0 * std, rtol=0.0, atol=0.0)
    with pytest.raises(Refusal, match='beta'):
        upper_confidence_bound(mean, std, -1.0)


def test_knowledge_gradient_of_two_mirrored_lines_is_e_abs_z():
    assert knowledge_gradient([0.0, 0.0], [-1.0, 1.0]) == pytest.approx(np.sqrt(2.0 / np.pi), rel=1e-14, abs=0.0)


def test_knowledge_gradient_matches_monte_carlo_over_a_random_set():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=12), rng.normal(size=12)
    z = rng.standard_normal(1_000_000)
    gains = np.max(a[None] + b[None] * z[:, None], axis=1) - a.max()
    assert abs(knowledge_gradient(a, b) - gains.mean()) <= Z_GATE * gains.std() / 1000.0


def test_planted_equal_slopes_carry_no_knowledge():
    """The control: a set whose values all move together learns nothing from the observation."""
    assert knowledge_gradient([0.3, -1.0, 2.0], [0.5, 0.5, 0.5]) == 0.0


def test_probability_of_feasibility_is_the_product_of_margins():
    mean, std = np.array([[-1.0, 0.5]]), np.array([[1.0, 2.0]])
    assert np.exp(log_probability_of_feasibility(mean, std))[0] == pytest.approx(
        ndtr(1.0) * ndtr(-0.25), rel=1e-14, abs=0.0
    )


FRONT2 = np.array([[0.1, 0.9], [0.4, 0.5], [0.8, 0.2]])
FRONT3 = np.array([[0.1, 0.8, 0.5], [0.5, 0.3, 0.6], [0.7, 0.6, 0.1], [0.3, 0.4, 0.9]])


@pytest.mark.parametrize(
    ('front', 'mean', 'std'), [(FRONT2, [0.35, 0.45], [0.2, 0.3]), (FRONT3, [0.4, 0.5, 0.4], [0.2, 0.1, 0.3])]
)
def test_ehvi_is_the_monte_carlo_mean_of_the_exact_improvement(front, mean, std):
    rng = np.random.default_rng(7)
    reference = np.ones(front.shape[1])
    draws = np.array(mean) + np.array(std) * rng.standard_normal((200_000, front.shape[1]))
    gains = hypervolume_improvement(draws, front, reference)
    ehvi = expected_hypervolume_improvement(np.array([mean]), np.array([std]), front, reference)[0]
    assert abs(ehvi - gains.mean()) <= Z_GATE * gains.std() / np.sqrt(len(gains))


def test_the_improvement_is_the_hypervolume_difference_and_ehvi_at_zero_std():
    rng = np.random.default_rng(1)
    reference = np.ones(3)
    points = rng.random((20, 3))
    gains = hypervolume_improvement(points, FRONT3, reference)
    truth = [hypervolume(np.vstack([FRONT3, p]), reference) - hypervolume(FRONT3, reference) for p in points]
    assert np.allclose(gains, truth, rtol=0.0, atol=VOLUME)
    assert np.allclose(
        expected_hypervolume_improvement(points, np.zeros_like(points), FRONT3, reference), gains, atol=VOLUME
    )


def test_planted_a_dominated_mean_with_no_spread_improves_nothing():
    assert expected_hypervolume_improvement([[0.9, 0.95]], [[0.0, 0.0]], FRONT2, np.ones(2))[0] == 0.0


def test_nehvi_over_identical_draws_is_the_plain_improvement_with_no_error():
    baseline = np.repeat(FRONT2[None], 8, axis=0)
    candidate = np.repeat([[0.3, 0.3]], 8, axis=0)
    estimate = noisy_expected_hypervolume_improvement(baseline, candidate, np.ones(2))
    assert estimate.value == pytest.approx(hypervolume_improvement([[0.3, 0.3]], FRONT2, np.ones(2))[0], abs=VOLUME)
    assert estimate.std_error == 0.0 and estimate.n == 8


def test_parego_scores_are_knowles_augmented_tchebycheff_free_of_the_objective_units():
    rng = np.random.default_rng(2)
    values, weights = rng.random((30, 2)), np.array([0.0, 1.0])
    scaled = (values - values.min(0)) / (values.max(0) - values.min(0))
    knowles = np.max(weights * scaled, axis=1) + 0.05 * np.sum(weights * scaled, axis=1)
    assert np.allclose(parego_scores(values, weights), knowles, rtol=1e-12, atol=0.0)
    rescaled = values * np.array([1.0, 1e3]) + np.array([0.0, 5.0])
    assert np.allclose(parego_scores(values, weights), parego_scores(rescaled, weights), rtol=1e-12, atol=0.0)
    # the control: without the normalisation the score moves with the unit
    assert not np.allclose(tchebycheff(values, weights, np.zeros(2)), tchebycheff(rescaled, weights, np.zeros(2)))
