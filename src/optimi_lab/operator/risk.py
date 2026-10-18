"""D116 risk operators: one number from a distribution of a LOSS (larger is worse, as everywhere here).

Each takes ``values (n,)`` and ``weights (n,) | None`` -- a discrete distribution, ``None`` meaning
equally weighted samples -- so a scenario set, a quadrature rule and a Monte Carlo sample are the
same input.

* :func:`expectation`, :func:`mean_plus_std` (``mean + k std``, the population std of the weights);
* :func:`value_at_risk` -- the lower ``alpha``-quantile ``min{t : F(t) >= alpha}``;
* :func:`cvar` -- Rockafellar-Uryasev ``VaR + E[(X - VaR)+] / (1 - alpha)``, which splits an atom
  that straddles the quantile, so the CVaR of a 10 % atom at 90 % is that atom (B0-14) and never
  the mean of "the values at or above VaR";
* :func:`worst_case` -- the largest value.
"""

from __future__ import annotations

import numpy as np

from optimi_lab.core.errors import refuse

__all__ = ['cvar', 'expectation', 'mean_plus_std', 'value_at_risk', 'worst_case']

#: How far the weights' sum may sit from 1 (dimensionless; rounding of a few hundred additions).
_SUM_TOLERANCE = 1e-9


def _distribution(values, weights) -> tuple[np.ndarray, np.ndarray]:
    values = np.ravel(np.asarray(values, dtype=float))
    if values.size == 0 or not np.all(np.isfinite(values)):
        refuse(f'a risk measure needs finite values, got {values.size} values')
    if weights is None:
        return values, np.full(values.size, 1.0 / values.size)
    weights = np.ravel(np.asarray(weights, dtype=float))
    if weights.shape != values.shape:
        refuse(f'weights {weights.shape} do not match values {values.shape}')
    if np.any(weights < 0):
        refuse('weights must be non-negative')
    if abs(weights.sum() - 1.0) > _SUM_TOLERANCE:
        refuse(f'weights must sum to 1, got {weights.sum()!r}')
    return values, weights


def _alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        refuse(f'alpha must lie in (0, 1), got {alpha!r}')
    return float(alpha)


def expectation(values, weights) -> float:
    values, weights = _distribution(values, weights)
    return float(weights @ values)


def mean_plus_std(values, weights, *, k: float) -> float:
    values, weights = _distribution(values, weights)
    mean = float(weights @ values)
    return mean + k * float(np.sqrt(weights @ (values - mean) ** 2))


def value_at_risk(values, weights, *, alpha: float) -> float:
    values, weights = _distribution(values, weights)
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, _alpha(alpha) - _SUM_TOLERANCE)), values.size - 1)
    return float(values[order][index])


def cvar(values, weights, *, alpha: float) -> float:
    values, weights = _distribution(values, weights)
    var = value_at_risk(values, weights, alpha=alpha)
    above = values > var
    straddle = max(float(weights[~above].sum()) - _alpha(alpha), 0.0)  # the quantile atom's share of the tail
    tail = float(weights[above].sum()) + straddle  # == 1 - alpha, read off the weights so atoms stay exact
    return (float(weights[above] @ values[above]) + straddle * var) / tail


def worst_case(values) -> float:
    values, _ = _distribution(values, None)
    return float(values.max())
