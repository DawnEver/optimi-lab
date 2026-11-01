"""Kendall's tau-b, the Wilson score interval and the cluster bootstrap.

Oracles: scipy.stats.kendalltau (TEST ONLY, R1) for tau-b, ties included; the textbook Wilson
interval of 5 successes in 10; and, for the bootstrap, a verbatim copy of the cluster resampling
loop it replaces, so the RNG stream is pinned draw for draw rather than in distribution.
"""

from collections import defaultdict

import numpy as np
import pytest
from scipy import stats

from optimi_lab.core import Refusal
from optimi_lab.metric import cluster_bootstrap, kendall_tau_a, kendall_tau_b, wilson


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_tau_b_matches_scipy_with_ties(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, 25).astype(float)
    y = x + rng.integers(-2, 3, 25)
    assert kendall_tau_b(x, y) == pytest.approx(stats.kendalltau(x, y).statistic, abs=1e-14)


def test_tau_b_matches_scipy_without_ties():
    rng = np.random.default_rng(7)
    x, y = rng.random(40), rng.random(40)
    assert kendall_tau_b(x, y) == pytest.approx(stats.kendalltau(x, y).statistic, abs=1e-14)


def test_tau_b_closed_forms():
    assert kendall_tau_b([1, 2, 3], [10, 20, 30]) == 1.0
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert np.isnan(kendall_tau_b([1, 1, 1], [1, 2, 3]))
    assert np.isnan(kendall_tau_b([1.0], [2.0]))
    assert np.isnan(kendall_tau_b([1.0, np.nan, 3.0], [1.0, 2.0, 3.0]))
    with pytest.raises(Refusal):
        kendall_tau_b([1, 2], [1, 2, 3])


def test_wilson_interval_of_five_in_ten_is_the_textbook_one():
    p, lo, hi = wilson(5, 10)
    assert p == 0.5
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)


def test_wilson_edges():
    p, lo, hi = wilson(0, 20)
    assert (p, lo) == (0.0, 0.0) and 0.0 < hi < 0.2
    p, lo, hi = wilson(0, 0)
    assert np.isnan(p) and (lo, hi) == (0.0, 1.0)
    with pytest.raises(Refusal):
        wilson(3, 2)


def _reference_bootstrap(clusters, statistic, *, draws, seed):
    """The cluster resampling loop being replaced, verbatim in its RNG use."""
    keys = list(clusters)
    rng = np.random.default_rng(seed)

    def summed(chosen):
        total = defaultdict(lambda: defaultdict(float))
        for key in chosen:
            for policy, counts in clusters[key].items():
                for name, value in counts.items():
                    total[policy][name] += value
        return total

    point = statistic(summed(keys))
    values = [statistic(summed([keys[i] for i in rng.integers(0, len(keys), len(keys))])) for _ in range(draws)]
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return point, float('nan'), float('nan')
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


@pytest.mark.parametrize('seed', [0, 5, 11])
def test_cluster_bootstrap_reproduces_the_replaced_stream_exactly(seed):
    rng = np.random.default_rng(100 + seed)
    clusters = {
        (case, s): {'a': {'tp': int(rng.integers(0, 5)), 'n': 5}, 'b': {'tp': int(rng.integers(0, 5)), 'n': 5}}
        for case in ('p', 'q')
        for s in range(6)
    }

    def ratio(total):
        return total['a']['tp'] / total['a']['n'] - total['b']['tp'] / total['b']['n']

    keys = list(clusters)

    def by_index(chosen):
        total = defaultdict(lambda: defaultdict(float))
        for i in chosen:
            for policy, counts in clusters[keys[i]].items():
                for name, value in counts.items():
                    total[policy][name] += value
        return ratio(total)

    expected = _reference_bootstrap(clusters, ratio, draws=300, seed=seed)
    assert cluster_bootstrap(by_index, len(keys), draws=300, seed=seed) == expected


def test_cluster_bootstrap_drops_non_finite_draws_and_reports_nan_when_none_remain():
    point, lo, hi = cluster_bootstrap(lambda _idx: float('nan'), 4, draws=10, seed=0)
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)
    point, lo, hi = cluster_bootstrap(lambda idx: float(len(idx)), 3, draws=10, seed=0)
    assert (point, lo, hi) == (3.0, 3.0, 3.0)


def test_kendall_tau_a_closed_forms() -> None:
    assert kendall_tau_a([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert kendall_tau_a([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    # (a, b) = ([1,1,2,3], [1,2,2,3]): 6 pairs, (0,1) tied in a, (1,2) tied in b, the other 4 concordant
    assert kendall_tau_a([1, 1, 2, 3], [1, 2, 2, 3]) == 4 / 6
    assert np.isnan(kendall_tau_a([1.0], [2.0]))


def test_kendall_tau_a_differs_from_tau_b_under_ties() -> None:
    a, b = [1, 1, 2, 3], [1, 2, 2, 3]
    assert kendall_tau_b(a, b) == pytest.approx(4 / 5)
    assert kendall_tau_a(a, b) != kendall_tau_b(a, b)


def test_kendall_tau_a_reproduces_the_pairwise_sign_sum_bit_for_bit() -> None:
    rng = np.random.default_rng(7)
    a, b = rng.integers(0, 4, 23).astype(float), rng.normal(size=23)
    n, score = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            score += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    assert kendall_tau_a(list(a), list(b)) == float(score / (n * (n - 1) / 2))
