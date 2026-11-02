"""An offline bench of surrogate families on solved rows: which model is worth steering with, measured.

Every family is a registry model (:mod:`optimi_lab.model.registry`); this module declares
configurations, combines predictions and scores them -- it implements no regressor of its own.

* A :class:`Family` is a registry key and the configurations INNER cross-validation chooses among, on
  the training rows alone, so a tuned family is still scored only on rows it never saw (nested).
* :func:`fit_bench` fits every family on one training set and returns its test mean and a predictive
  standard deviation: the GP's own (observation variance included), else the homoscedastic inner-CV
  RMSE -- the baseline that makes calibration comparable at all.
* Derived entries, :data:`DERIVED`: ``pool`` (the best cross-validated ``r2`` of :data:`POOL`,
  refitted), ``ensemble`` (mean of the GP, the RBF and the ridge; variance the GP's plus the members'
  spread) and ``selected`` (per column the lowest inner score -- the inner score travels with it, so
  the optimism of a winner read on its own holdout stays visible, :func:`naive_select_score`).

Metrics are plain functions of (truth, mean, sd): range-normalised errors, Spearman / Kendall tau-b,
central-interval coverage, Deb-front precision / recall and IGD+, and a Gaussian feasibility
probability with its balanced accuracy and Brier score.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from optimi_lab.core.pareto import constrained_non_dominated_sorting
from optimi_lab.metric.indicators import igd_plus
from optimi_lab.metric.statistics import kendall_tau_b, spearman
from optimi_lab.model.polynomial import Ridge
from optimi_lab.model.registry import surrogate
from optimi_lab.model.validation import cross_validate, scored_fit
from optimi_lab.operator.acquisition import log_probability_of_feasibility

__all__ = [
    'DERIVED',
    'FAMILIES',
    'POOL',
    'ColumnBound',
    'Family',
    'Fit',
    'balanced_accuracy',
    'brier',
    'feasibility_probability',
    'fit_bench',
    'front_recovery',
    'interval_coverage',
    'naive_select_score',
    'nested_select',
    'normalized_errors',
    'rank_correlation',
]

#: Inner folds of every configuration choice and homoscedastic standard deviation, and their split seed.
INNER_FOLDS: Final[int] = 5
_INNER_SEED: Final[int] = 3


@dataclass(frozen=True)
class Family:
    """A named family: registry key and the configurations inner CV chooses among."""

    key: str
    configs: tuple[dict, ...]
    role: str


#: The families on the bench; the hyper-parameters stated here are the whole search.
FAMILIES: Final[dict[str, Family]] = {
    'polynomial': Family('polynomial', ({'degree': 2},), 'quadratic response surface, OLS'),
    'radial_basis': Family('radial_basis', ({},), 'thin-plate spline, interpolating'),
    'nearest_neighbours': Family('nearest_neighbours', ({'k': 5},), '5-NN'),
    'ridge': Family(
        'polynomial',
        tuple({'degree': 2, 'penalty': Ridge(alpha)} for alpha in (1e-3, 1e-2, 1e-1, 1.0)),
        'quadratic response surface, ridge penalty by inner CV',
    ),
    'forest': Family('forest', ({'n_trees': 30, 'min_samples_leaf': 2},), 'bagged trees'),
    'gp': Family('gaussian_process', ({'family': 'matern52', 'restarts': 1},), 'GP, Matern 5/2, ARD'),
    'rbf': Family(
        'radial_basis', tuple({'smoothing': s} for s in (0.0, 1e-3, 1e-2, 1e-1)), 'thin-plate spline, smoothing by CV'
    ),
}
#: The models the ``pool`` entry chooses among, each fitted at its registry defaults.
POOL: Final[tuple[str, ...]] = ('polynomial', 'radial_basis', 'nearest_neighbours')
#: Entries computed from the fitted families, not fitted themselves.
DERIVED: Final[tuple[str, ...]] = ('pool', 'ensemble', 'selected')
_ENSEMBLE: Final[tuple[str, ...]] = ('gp', 'rbf', 'ridge')


@dataclass(frozen=True)
class Fit:
    """One entry's test predictions: ``mean``/``sd`` ``(n_test, k)``, its inner score and its fit seconds."""

    mean: np.ndarray
    sd: np.ndarray
    inner_rmse: np.ndarray
    seconds: float
    native_sd: bool = False
    choice: tuple[str, ...] = field(default=())


def _inner_rmse(key: str, config: dict, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    k = min(INNER_FOLDS, len(x))
    mean = np.asarray(cross_validate(surrogate(key, **config), x, z, k=k, seed=_INNER_SEED).mean)
    return np.sqrt(np.mean((mean - z) ** 2, axis=0))


def _fit_family(family: Family, x: np.ndarray, z: np.ndarray, test: np.ndarray) -> Fit:
    started = time.perf_counter()
    if family.key == 'gaussian_process':
        # Dubrule's closed-form leave-one-out at the fitted theta: one fit, and the observation variance.
        fitted = surrogate(family.key, **family.configs[0]).fit(x, z)
        inner = np.sqrt(np.mean((np.asarray(fitted.leave_one_out().mean) - z) ** 2, axis=0))
        predicted = fitted.predict(test, variance=True)
        sd = np.sqrt(np.asarray(predicted.variance) + np.asarray(fitted.noise)[None, :])
        return Fit(np.asarray(predicted.mean), sd, inner, time.perf_counter() - started, native_sd=True)
    scores = [_inner_rmse(family.key, config, x, z) for config in family.configs]
    best = int(np.argmin([s.mean() for s in scores]))  # one configuration for every column
    mean = np.asarray(surrogate(family.key, **family.configs[best]).fit(x, z).predict(test).mean)
    return Fit(mean, np.broadcast_to(scores[best], mean.shape).copy(), scores[best], time.perf_counter() - started)


def _pool(x: np.ndarray, z: np.ndarray, test: np.ndarray) -> Fit:
    started = time.perf_counter()
    best = None
    for key in POOL:
        fit = scored_fit(surrogate(key), x, z)  # each pool member at its defaults
        if best is None or fit.scores['r2'] > best[1].scores['r2']:
            best = (key, fit)
    key, fit = best
    mean = np.asarray(fit.fitted.predict(test).mean)
    inner = _inner_rmse(key, FAMILIES[key].configs[0], x, z)
    return Fit(mean, np.broadcast_to(inner, mean.shape).copy(), inner, time.perf_counter() - started, choice=(key,))


def fit_bench(
    x: np.ndarray, y: np.ndarray, test: np.ndarray, *, families: Sequence[str] = tuple(FAMILIES)
) -> dict[str, Fit]:
    """Every family plus the derived entries, fitted on ``(x, y)`` and predicted at ``test``, in ``y``'s units."""
    x, y, test = (np.asarray(a, dtype=float) for a in (x, y, test))
    level, scale = y.mean(axis=0), y.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (y - level) / scale
    fits = {name: _fit_family(FAMILIES[name], x, z, test) for name in families}
    fits['pool'] = _pool(x, z, test)
    if all(name in fits for name in _ENSEMBLE):
        members = np.stack([fits[name].mean for name in _ENSEMBLE])
        inner = np.mean([fits[name].inner_rmse for name in _ENSEMBLE], axis=0)
        sd = np.sqrt(fits['gp'].sd ** 2 + members.var(axis=0))
        seconds = sum(fits[name].seconds for name in _ENSEMBLE)
        fits['ensemble'] = Fit(members.mean(axis=0), sd, inner, seconds, native_sd=True)
    chosen = [name for name in FAMILIES if name in fits]
    if chosen:
        fits['selected'] = nested_select({name: fits[name] for name in chosen})
    return {
        name: Fit(f.mean * scale + level, f.sd * scale, f.inner_rmse, f.seconds, f.native_sd, f.choice)
        for name, f in fits.items()
    }


def nested_select(candidates: Mapping[str, Fit]) -> Fit:
    """Per column, the candidate with the lowest INNER RMSE; the test is untouched by the choice."""
    names = list(candidates)
    inner = np.stack([candidates[n].inner_rmse for n in names])
    pick = inner.argmin(axis=0)
    mean = np.stack([candidates[names[p]].mean[:, j] for j, p in enumerate(pick)], axis=1)
    sd = np.stack([candidates[names[p]].sd[:, j] for j, p in enumerate(pick)], axis=1)
    return Fit(
        mean,
        sd,
        inner.min(axis=0),
        sum(c.seconds for c in candidates.values()),
        native_sd=all(candidates[names[p]].native_sd for p in pick),
        choice=tuple(names[p] for p in pick),
    )


def normalized_errors(truth: np.ndarray, mean: np.ndarray, span: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column RMSE and MAE divided by ``span`` (the column's range over the whole case)."""
    error = np.asarray(mean, dtype=float) - np.asarray(truth, dtype=float)
    return np.sqrt(np.mean(error**2, axis=0)) / span, np.mean(np.abs(error), axis=0) / span


def rank_correlation(truth: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column Spearman rho and Kendall tau-b; a constant column scores 0, never NaN."""
    rho, tau = [], []
    for j in range(truth.shape[1]):
        if np.ptp(truth[:, j]) == 0 or np.ptp(mean[:, j]) == 0:
            rho.append(0.0)
            tau.append(0.0)
            continue
        rho.append(float(spearman(np.column_stack([truth[:, j], mean[:, j]]))))
        tau.append(kendall_tau_b(truth[:, j], mean[:, j]))
    return np.array(rho), np.array(tau)


def interval_coverage(truth: np.ndarray, mean: np.ndarray, sd: np.ndarray, z: float) -> float:
    """Share of test values inside ``mean +- z sd`` (pooled over columns)."""
    return float(np.mean(np.abs(truth - mean) <= z * np.maximum(sd, 1e-300)))


@dataclass(frozen=True)
class ColumnBound:
    """``column >= bound`` (``lower``) or ``column <= bound`` on a predicted column."""

    column: int
    lower: bool
    bound: float


def feasibility_probability(mean: np.ndarray, sd: np.ndarray, bounds: Sequence[ColumnBound]) -> np.ndarray:
    """``P(feasible)`` under independent Gaussian predictions (sd floored at 1e-12); 1 with no bound."""
    if not bounds:
        return np.ones(len(mean))
    margin = np.column_stack([
        (b.bound - mean[:, b.column]) if b.lower else (mean[:, b.column] - b.bound) for b in bounds
    ])
    std = np.column_stack([np.maximum(sd[:, b.column], 1e-12) for b in bounds])
    return np.exp(log_probability_of_feasibility(margin, std))


def balanced_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Mean of the two class recalls; NaN when a class is absent from the truth."""
    truth, predicted = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    if truth.all() or not truth.any():
        return float('nan')
    return float((np.mean(predicted[truth]) + np.mean(~predicted[~truth])) / 2)


def brier(truth: np.ndarray, probability: np.ndarray) -> float:
    """Mean squared error of a probability against a 0/1 truth."""
    return float(np.mean((np.asarray(probability, dtype=float) - np.asarray(truth, dtype=float)) ** 2))


def _first_front(values: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    idx = np.asarray(constrained_non_dominated_sorting(values, (~feasible).astype(float))).ravel()
    member = np.zeros(len(values), dtype=bool)
    member[idx] = True
    return member & feasible if feasible.any() else member


def front_recovery(
    truth_min: np.ndarray, truth_ok: np.ndarray, predicted_min: np.ndarray, predicted_ok: np.ndarray, span: np.ndarray
) -> dict[str, float]:
    """Deb-front membership precision / recall, and IGD+ (in ``span`` units) of the predicted front's truly
    feasible TRUE values against the true front.
    """
    true_front = _first_front(truth_min, truth_ok)
    pred_front = _first_front(predicted_min, predicted_ok)
    hit = np.sum(true_front & pred_front)
    chosen = truth_min[pred_front & truth_ok] / span
    return {
        'precision': float(hit / pred_front.sum()) if pred_front.any() else 0.0,
        'recall': float(hit / true_front.sum()) if true_front.any() else float('nan'),
        'igd_plus': igd_plus(chosen, truth_min[true_front] / span) if true_front.any() else float('nan'),
    }


def naive_select_score(candidates: Mapping[str, np.ndarray], truth: np.ndarray, metric: Callable) -> float:
    """The OPTIMISTIC reading nested validation refuses: the best candidate chosen ON the holdout, scored there."""
    return max(metric(truth, mean) for mean in candidates.values())
