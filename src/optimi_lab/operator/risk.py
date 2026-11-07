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

:class:`Risk` is the DECLARED choice of one of them with its parameter, the one value a robust
solver selects on (:mod:`optimi_lab.solver.population`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import refuse

__all__ = ['Measure', 'Risk', 'cvar', 'expectation', 'mean_plus_std', 'value_at_risk', 'worst_case']

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


class Measure(StrEnum):
    """The risk measures a :class:`Risk` may declare."""

    MEAN = 'mean'
    MEAN_PLUS_STD = 'mean_plus_std'
    VAR = 'var'
    CVAR = 'cvar'
    WORST_CASE = 'worst_case'


_ALPHA: frozenset[Measure] = frozenset({Measure.VAR, Measure.CVAR})


@dataclass(frozen=True, slots=True)
class _RiskFields:
    measure: Measure
    alpha: float | None
    k: float | None


@dataclass(frozen=True, slots=True, init=False)
class Risk(_RiskFields):
    """A declared risk *measure* with its ``alpha`` (VaR, CVaR) or ``k`` (mean + k std)."""

    def __init__(self, measure: str, *, alpha: float | None = None, k: float | None = None) -> None:
        if measure not in Measure.__members__.values():
            refuse(f'no risk measure {measure!r}; the measures are {[m.value for m in Measure]}')
        measure = Measure(measure)
        if (alpha is not None) != (measure in _ALPHA):
            refuse(
                f'alpha= is given exactly for {sorted(m.value for m in _ALPHA)}, got {measure.value} alpha={alpha!r}'
            )
        if (k is not None) != (measure is Measure.MEAN_PLUS_STD):
            refuse(f'k= is given exactly for mean_plus_std, got {measure.value} with k={k!r}')
        _RiskFields.__init__(self, measure, None if alpha is None else _alpha(alpha), None if k is None else float(k))

    @property
    def label(self) -> str:
        """The statistic a value of this risk IS, as a report prints it: ``CVaR0.9``, ``VaR0.95``, ``mean+2std``."""
        if self.alpha is not None:
            return f'{"CVaR" if self.measure is Measure.CVAR else "VaR"}{self.alpha:g}'
        if self.k is not None:
            return f'mean+{self.k:g}std'
        return self.measure.value

    def of(self, values, weights=None) -> float:
        """The declared measure of the loss distribution *values* (equally weighted when *weights* is None)."""
        if self.measure is Measure.MEAN:
            return expectation(values, weights)
        if self.measure is Measure.MEAN_PLUS_STD:
            return mean_plus_std(values, weights, k=self.k)
        if self.measure is Measure.VAR:
            return value_at_risk(values, weights, alpha=self.alpha)
        if self.measure is Measure.CVAR:
            return cvar(values, weights, alpha=self.alpha)
        return worst_case(values)
