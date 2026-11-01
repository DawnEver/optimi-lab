"""The offline surrogate bench: nested selection, derived entries and the plain metrics.

Agreement checks (the copies being replaced must agree numerically before they are deleted):
IGD+ and the Gaussian feasibility probability against verbatim copies of the replaced code, rank
correlation against scipy.stats (TEST ONLY, R1). The rest is closed form.
"""

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtr

from optimi_lab.metric import igd_plus
from optimi_lab.model import cross_validate, score, surrogate
from optimi_lab.model.bench import (
    DERIVED,
    FAMILIES,
    POOL,
    ColumnBound,
    Fit,
    balanced_accuracy,
    brier,
    feasibility_probability,
    fit_bench,
    front_recovery,
    interval_coverage,
    naive_select_score,
    nested_select,
    normalized_errors,
    rank_correlation,
)


def _copied_igd_plus(front, reference):
    if len(front) == 0:
        return float('inf')
    gap = np.maximum(front[None, :, :] - reference[:, None, :], 0.0)
    return float(np.sqrt((gap**2).sum(axis=-1)).min(axis=1).mean())


def _copied_feasibility(mean, sd, constraints):
    p = np.ones(len(mean))
    for c in constraints:
        s = np.maximum(sd[:, c.column], 1e-12)
        margin = (mean[:, c.column] - c.bound) / s
        p *= ndtr(margin if c.lower else -margin)
    return p


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_library_igd_plus_agrees_with_the_replaced_copy(seed):
    rng = np.random.default_rng(seed)
    front, reference = rng.random((15, 3)), rng.random((40, 3))
    assert igd_plus(front, reference) == _copied_igd_plus(front, reference)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_feasibility_probability_agrees_with_the_replaced_copy(seed):
    rng = np.random.default_rng(seed)
    mean, sd = rng.normal(size=(50, 3)), rng.random((50, 3))
    sd[0, 1] = 0.0
    bounds = [ColumnBound(0, True, 0.2), ColumnBound(2, False, -0.1)]
    np.testing.assert_allclose(
        feasibility_probability(mean, sd, bounds), _copied_feasibility(mean, sd, bounds), rtol=1e-12, atol=1e-300
    )
    assert feasibility_probability(mean, sd, []).tolist() == [1.0] * 50


def test_rank_correlation_agrees_with_scipy_and_scores_a_constant_column_zero():
    rng = np.random.default_rng(3)
    truth = rng.random((30, 2))
    mean = truth + 0.3 * rng.random((30, 2))
    mean[:, 1] = 1.0
    rho, tau = rank_correlation(truth, mean)
    assert rho[0] == pytest.approx(stats.spearmanr(truth[:, 0], mean[:, 0]).statistic, abs=1e-14)
    assert tau[0] == pytest.approx(stats.kendalltau(truth[:, 0], mean[:, 0]).statistic, abs=1e-14)
    assert (rho[1], tau[1]) == (0.0, 0.0)


def test_plain_metrics_by_hand():
    rmse, mae = normalized_errors(np.zeros((2, 1)), np.array([[3.0], [-1.0]]), np.array([2.0]))
    assert rmse[0] == pytest.approx(np.sqrt(5.0) / 2) and mae[0] == 1.0
    truth, mean, sd = np.zeros((4, 1)), np.array([[0.5], [1.5], [-0.9], [3.0]]), np.ones((4, 1))
    assert interval_coverage(truth, mean, sd, 1.0) == 0.5
    assert balanced_accuracy([True, True, False, False], [True, False, False, False]) == 0.75
    assert np.isnan(balanced_accuracy([True, True], [True, False]))
    assert brier([1, 0], [0.5, 0.5]) == 0.25
    assert naive_select_score({'a': np.zeros(2), 'b': np.ones(2)}, np.ones(2), lambda t, m: -np.abs(t - m).sum()) == 0


def test_front_recovery_by_hand():
    """True front {0, 1}; predicted front {1, 2}: one hit of two each way; IGD+ reads row 1 and the infeasible row 2."""
    truth = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [2.0, 2.0]])
    ok = np.array([True, True, False, True])
    predicted = np.array([[3.0, 3.0], [1.0, 0.0], [0.0, 1.0], [4.0, 4.0]])
    out = front_recovery(truth, ok, predicted, np.array([True, True, True, True]), np.ones(2))
    assert out['precision'] == 0.5 and out['recall'] == 0.5
    assert out['igd_plus'] == pytest.approx(0.5)  # (|(1,0)-(0,1)|+ = 1, then 0) / 2 over the true front


def _data(n=30, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 2))
    y = np.column_stack([1 + x[:, 0] ** 2 + x[:, 0] * x[:, 1], 3 * x[:, 1] - x[:, 0]])
    return x, y, rng.random((10, 2))


def test_the_bench_fits_every_family_and_derives_its_entries():
    x, y, test = _data()
    fits = fit_bench(x, y, test)
    assert set(fits) == set(FAMILIES) | set(DERIVED)
    assert all(f.mean.shape == (10, 2) and f.sd.shape == (10, 2) for f in fits.values())
    truth = np.column_stack([1 + test[:, 0] ** 2 + test[:, 0] * test[:, 1], 3 * test[:, 1] - test[:, 0]])
    np.testing.assert_allclose(fits['polynomial'].mean, truth, atol=1e-8)  # a quadratic fits a quadratic
    members = np.stack([fits[n].mean for n in ('gp', 'rbf', 'ridge')])
    np.testing.assert_allclose(fits['ensemble'].mean, members.mean(axis=0), rtol=1e-10)
    inner = np.stack([fits[n].inner_rmse for n in FAMILIES])
    assert fits['selected'].choice == tuple(list(FAMILIES)[i] for i in inner.argmin(axis=0))


def test_the_pool_entry_is_the_member_with_the_best_cross_validated_r2():
    x, y, test = _data(seed=4)
    z = (y - y.mean(axis=0)) / y.std(axis=0)
    r2 = {key: score('r2', z, cross_validate(surrogate(key), x, z, k=5, seed=3).mean) for key in POOL}
    assert fit_bench(x, y, test, families=('polynomial',))['pool'].choice == (max(r2, key=r2.get),)


def test_nested_select_picks_per_column_by_inner_score():
    a = Fit(np.zeros((2, 2)), np.ones((2, 2)), np.array([1.0, 3.0]), 1.0)
    b = Fit(np.ones((2, 2)), 2 * np.ones((2, 2)), np.array([2.0, 2.0]), 2.0, native_sd=True)
    s = nested_select({'a': a, 'b': b})
    assert s.choice == ('a', 'b') and s.inner_rmse.tolist() == [1.0, 2.0] and s.seconds == 3.0
    assert s.mean[:, 0].tolist() == [0.0, 0.0] and s.mean[:, 1].tolist() == [1.0, 1.0] and not s.native_sd
