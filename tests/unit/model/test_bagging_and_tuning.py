"""Bagging and hyper-parameter tuning against an analytic quadratic.

``y = 1 + x1 - 2 x2 + x1 x2`` is reproduced EXACTLY by a degree-2 polynomial fitted on any resample
holding enough distinct points, so a bagged degree-2 polynomial predicts it to round-off with zero
spread; a degree-1 member cannot, and its bag shows a positive spread. Tuning the degree over
``(1, 2, 3)`` must choose 2 (2 and 3 are both exact; a tie keeps grid order) and score degree 1 lower.
"""

import numpy as np
import pytest

from optimi_lab.core.errors import Refusal
from optimi_lab.model import Bagging, Give, Polynomial, RegressionTree, tune

RNG = np.random.default_rng(0)
X = RNG.uniform(-1.0, 1.0, size=(60, 2))
QUERY = RNG.uniform(-1.0, 1.0, size=(15, 2))


def quadratic(x: np.ndarray) -> np.ndarray:
    return (1.0 + x[:, 0] - 2.0 * x[:, 1] + x[:, 0] * x[:, 1])[:, None]


def test_a_bag_of_exact_members_is_exact_with_zero_spread():
    fitted = Bagging(Polynomial(degree=2), n_members=8, seed=1).fit(X, quadratic(X))
    p = fitted.predict(QUERY, variance=True, gradient=True)
    assert p.mean == pytest.approx(quadratic(QUERY), abs=1e-9)
    assert p.variance == pytest.approx(np.zeros_like(p.variance), abs=1e-12)
    expected_slope = np.column_stack([1.0 + QUERY[:, 1], -2.0 + QUERY[:, 0]])
    assert p.gradient[:, 0, :] == pytest.approx(expected_slope, abs=1e-8)


def test_a_bag_of_biased_members_shows_spread_and_is_reproducible():
    bag = Bagging(Polynomial(degree=1), n_members=8, seed=1)
    first = bag.fit(X, quadratic(X)).predict(QUERY, variance=True)
    again = bag.fit(X, quadratic(X)).predict(QUERY, variance=True)
    assert np.all(first.variance > 0)
    assert np.array_equal(first.mean, again.mean)


def test_a_bag_gives_variance_always_and_gradient_only_through_its_member():
    assert Bagging(Polynomial()).gives == {Give.VARIANCE, Give.GRADIENT}
    assert Bagging(RegressionTree()).gives == {Give.VARIANCE}


def test_a_bag_refuses_degenerate_settings():
    with pytest.raises(Refusal, match='at least two'):
        Bagging(Polynomial(), n_members=1)
    with pytest.raises(Refusal, match='share of the rows'):
        Bagging(Polynomial(), max_samples=0.0)


def test_tuning_chooses_the_degree_that_reproduces_the_quadratic():
    tuning = tune('polynomial', {'degree': (1, 2, 3)}, X, quadratic(X), k=5, seed=2)
    assert tuning.best.settings == {'degree': 2}
    assert tuning.best.score == pytest.approx(1.0, abs=1e-9)
    assert [t.settings['degree'] for t in tuning.trials] == [1, 2, 3]
    assert tuning.trials[0].score < 0.99
    assert tuning.surrogate == Polynomial(degree=2)


def test_tuning_minimises_an_error_score_over_a_two_setting_grid():
    grid = {'degree': (1, 2), 'penalty': (None,)}
    tuning = tune('polynomial', grid, X, quadratic(X), metric='rmse', seed=2)
    assert len(tuning.trials) == 2
    assert tuning.best.settings == {'degree': 2, 'penalty': None}
    assert tuning.best.score == pytest.approx(0.0, abs=1e-9)


def test_tuning_refuses_strangers():
    with pytest.raises(Refusal, match='score is one of'):
        tune('polynomial', {'degree': (1,)}, X, quadratic(X), metric='nope')
    with pytest.raises(Refusal, match='at least one candidate'):
        tune('polynomial', {'degree': ()}, X, quadratic(X))
    with pytest.raises(Refusal, match='surrogate is one of'):
        tune('nope', {}, X, quadratic(X))
