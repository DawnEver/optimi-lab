"""D102-D106 -- hypervolume (exact, Monte Carlo, contributions), IGD+, spread, attainment, rank tests.

Oracles: closed-form boxes for the volumes; the definitions of IGD+ and of Deb's spread on fronts
whose value is arithmetic; scipy.stats (TEST ONLY, R1) for the rank-sum and signed-rank p-values and
the average ranks. Objective values are dimensionless here, so an absolute tolerance on a volume is
in (objective unit)^m.
"""

import numpy as np
import pytest
from scipy import stats

from optimi_lab.core import Refusal
from optimi_lab.metric import (
    attained,
    bootstrap_interval,
    hypervolume,
    hypervolume_contributions,
    hypervolume_mc,
    igd,
    igd_plus,
    rank,
    rank_sum_test,
    signed_rank_test,
    spearman,
    spread,
)

REF3 = np.array([1.0, 1.0, 1.0])


def test_exact_hypervolume_of_three_boxes_in_three_objectives_is_inclusion_exclusion():
    """Boxes [a, 1]^3: |A|+|B|+|C| - pairwise + triple, done by hand."""
    points = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    single, pair, triple = 0.25, 0.125, 0.125
    assert hypervolume(points, REF3) == pytest.approx(3 * single - 3 * pair + triple)


def test_monte_carlo_hypervolume_agrees_with_the_exact_one_within_its_standard_error():
    points = np.random.default_rng(1).random((12, 3)) * 0.8
    exact = hypervolume(points, REF3)
    estimate, error = hypervolume_mc(points, REF3, n_samples=100_000, rng=np.random.default_rng(2))
    assert abs(estimate - exact) <= 4.0 * error and error < 0.01


def test_each_contribution_is_the_volume_lost_without_that_point():
    points = np.array([[0.0, 3.0], [1.0, 1.2], [3.0, 0.0], [3.5, 3.5]])
    reference = np.array([4.0, 4.0])
    assert hypervolume_contributions(points, reference).tolist() == pytest.approx([1.0, 3.6, 1.2, 0.0])
    cube = np.random.default_rng(3).random((8, 3))
    full = hypervolume(cube, REF3 * 1.1)
    for i, c in enumerate(hypervolume_contributions(cube, REF3 * 1.1)):
        assert c == pytest.approx(full - hypervolume(np.delete(cube, i, axis=0), REF3 * 1.1))


def test_igd_plus_counts_only_the_part_of_a_gap_that_is_worse():
    """IGD+ (Ishibuchi et al., 2015): d+ = |max(a - z, 0)|; a point better than the reference costs 0."""
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert igd_plus(np.array([[-0.5, 0.5], [0.5, -0.5]]), reference) == 0.0
    assert igd(np.array([[-0.5, 0.5], [0.5, -0.5]]), reference) > 0.0, 'plain IGD charges the better gap'
    assert igd_plus(np.array([[0.0, 1.5], [1.5, 0.0]]), reference) == pytest.approx(0.5)


def test_spread_is_zero_for_an_even_front_touching_its_extremes():
    """Deb's Delta = (d_f + d_l + sum|d_i - mean d|) / (d_f + d_l + (N - 1) mean d)."""
    even = np.column_stack([np.linspace(0, 1, 11), 1 - np.linspace(0, 1, 11)])
    assert spread(even, extremes=np.array([[0.0, 1.0], [1.0, 0.0]])) == pytest.approx(0.0, abs=1e-12)
    bunched = np.array([[0.0, 1.0], [0.05, 0.95], [0.1, 0.9], [1.0, 0.0]])
    gaps = np.array([np.hypot(0.05, 0.05)] * 2 + [np.hypot(0.9, 0.9)])
    expected = np.abs(gaps - gaps.mean()).sum() / (3 * gaps.mean())
    assert spread(bunched, extremes=np.array([[0.0, 1.0], [1.0, 0.0]])) == pytest.approx(expected)


def test_a_point_is_attained_at_level_k_when_k_of_the_runs_weakly_dominate_it():
    runs = [np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[0.5, 0.5]]), np.array([[2.0, 2.0]])]
    probes = np.array([[0.6, 0.6], [1.0, 1.0], [0.0, 0.0]])
    assert attained(runs, probes).tolist() == pytest.approx([1 / 3, 2 / 3, 0.0])


def test_rank_averages_ties_like_the_textbook():
    data = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert rank(data).tolist() == stats.rankdata(data).tolist()
    m = np.random.default_rng(0).integers(0, 5, (30, 3)).astype(float)
    assert spearman(m) == pytest.approx(stats.spearmanr(m).statistic)


@pytest.mark.parametrize('shift', [0.0, 0.3, 1.0])
def test_the_rank_sum_p_value_matches_the_reference_within_its_normal_approximation(shift):
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=40), rng.normal(size=35) + shift
    ours = rank_sum_test(a, b)
    theirs = stats.mannwhitneyu(a, b, method='asymptotic', use_continuity=True)
    assert ours.statistic == pytest.approx(theirs.statistic)
    assert ours.p_value == pytest.approx(theirs.pvalue, abs=1e-12)


def test_the_signed_rank_p_value_matches_the_reference():
    rng = np.random.default_rng(6)
    a = rng.normal(size=30)
    b = a + rng.normal(0.2, 1.0, size=30)
    ours = signed_rank_test(a, b)
    theirs = stats.wilcoxon(a, b, method='approx', correction=True)
    assert ours.statistic == pytest.approx(theirs.statistic)
    assert ours.p_value == pytest.approx(theirs.pvalue, abs=1e-12)


def test_a_percentile_bootstrap_interval_brackets_the_sample_mean_at_the_normal_width():
    sample = np.random.default_rng(7).normal(10.0, 1.0, 200)
    low, high = bootstrap_interval(sample, np.mean, confidence=0.95, n_resamples=2000, rng=np.random.default_rng(8))
    assert low < sample.mean() < high and high - low == pytest.approx(2 * 1.96 / np.sqrt(200), rel=0.2, abs=0.0)


def test_a_confidence_outside_the_open_unit_interval_is_refused():
    with pytest.raises(Refusal, match='confidence'):
        bootstrap_interval(np.ones(3), np.mean, confidence=1.0, n_resamples=10, rng=np.random.default_rng(0))
