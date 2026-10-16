"""Decision support on synthetic fronts whose answers are known analytically, minimization frame."""

import numpy as np
import pytest

from optimi_lab import Refusal
from optimi_lab.analysis import PowerLaw, effective_dimension, knee_index, monotone_links, power_law_fit, topsis


def _box(n=60, seed=1):
    return np.random.default_rng(seed).uniform([35, 2, 12], [50, 4.5, 20], (n, 3))


def test_monotone_links_pair_a_column_with_its_monotone_transforms_either_way():
    x = _box()
    values = np.column_stack([x[:, 0], np.exp(x[:, 0] / 10), -(x[:, 0] ** 3), x[:, 1]])
    links = monotone_links(('a', 'b', 'c', 'd'), values)
    assert {(p, q) for p, q, _ in links} == {('a', 'b'), ('a', 'c'), ('b', 'c')}
    assert dict(((p, q), rho) for p, q, rho in links)[('a', 'c')] == pytest.approx(-1.0)


def test_monotone_links_refuse_a_name_count_that_is_not_the_column_count():
    with pytest.raises(Refusal, match='name'):
        monotone_links(('a',), np.zeros((4, 2)))


def test_effective_dimension_counts_independent_directions():
    x = _box(200)
    assert effective_dimension(np.column_stack([x[:, 0], 2 * x[:, 0], np.log(x[:, 0])])) == 1
    assert effective_dimension(np.column_stack([x[:, 0], x[:, 0] ** 2, x[:, 1], x[:, 2]])) == 3
    assert effective_dimension(np.ones((5, 2))) == 0


def test_power_law_recovers_an_analytic_product():
    x = _box()
    fit = power_law_fit(x, 7.5e-6 * x[:, 0] * x[:, 1] * x[:, 2])
    assert isinstance(fit, PowerLaw)
    assert fit.exponents == pytest.approx((1.0, 1.0, 1.0))
    assert fit.coefficient == pytest.approx(7.5e-6)
    assert fit.r2 == pytest.approx(1.0)
    assert power_law_fit(x, x[:, 0] - 40.0) is None


def test_knee_of_a_convex_front_is_its_corner():
    """A super-ellipse front hugging the axes plus one point near the ideal: that point is the knee."""
    t = np.linspace(0, np.pi / 2, 21)
    values = np.vstack([np.column_stack([1 - np.cos(t) ** 8, 1 - np.sin(t) ** 8]), [0.1, 0.1]])
    assert knee_index(values) == len(values) - 1
    assert knee_index(values * [1.0, 1000.0]) == len(values) - 1


def test_knee_refuses_anything_but_two_objectives():
    with pytest.raises(Refusal, match='two objectives'):
        knee_index(np.zeros((4, 3)))


def test_topsis_ranks_ideal_first_anti_ideal_last_and_weights_steer():
    """Minimization frame: column 0 is a maximised objective already negated by the caller."""
    values = np.array([[-10.0, 1.0], [0.0, 9.0], [-5.0, 5.0], [-10.0, 9.0]])
    closeness = topsis(values)
    assert closeness[0] == pytest.approx(1.0)
    assert closeness[1] == pytest.approx(0.0)
    assert topsis(values, [1.0, 0.0])[3] == pytest.approx(1.0)
    assert np.argmax(topsis(values[1:], [0.0, 1.0])) == 1
    with pytest.raises(Refusal, match='non-negative'):
        topsis(values, [0.0, 0.0])
