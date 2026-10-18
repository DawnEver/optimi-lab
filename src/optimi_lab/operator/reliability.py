"""Sampling estimators of a failure probability ``P(g(Z) <= 0)``, ``Z`` standard normal (D119, D120, D122).

A limit state ``g`` maps standard-normal rows ``(n, d)`` to ``(n,)``; failure is ``g <= 0``. A limit
state written in physical parameters is carried into that space by :func:`in_standard_space`
(the independent-parameter Rosenblatt map of :meth:`Uncertainty.at_standard`). Every estimator
returns an :class:`~optimi_lab.operator.estimators.Estimate` whose ``n`` is the number of limit-state
evaluations it spent.

* :func:`crude_monte_carlo` -- the frequency of failure; the baseline every other one is judged by.
* :func:`importance_sampling` -- draws from ``N(center, I)`` (``center`` = a design point, e.g. from
  :func:`~optimi_lab.operator.form`) weighted by ``phi(z) / phi(z - center)``.
* :func:`subset_simulation` -- Au and Beck (2001): intermediate thresholds at the ``p0``-quantile,
  each conditional level grown from its seeds by component-wise (modified) Metropolis chains. The
  standard error is Au-Beck's independent-chain approximation, ``pf sqrt(sum (1 - p_i) / (p_i N))``,
  a LOWER bound: correlation along a chain only adds to it.
* :func:`surrogate_reliability` -- *n_train* limit-state evaluations fit a model by the injected
  *fit* (a polynomial chaos in production; the model layer owns it), then crude Monte Carlo on the
  model. Its standard error is that of the Monte Carlo on the MODEL; the model's own error is not in
  it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.operator.estimators import Estimate, mean_estimate, probability
from optimi_lab.problem import Uncertainty

__all__ = [
    'LimitState',
    'crude_monte_carlo',
    'importance_sampling',
    'in_standard_space',
    'subset_simulation',
    'surrogate_reliability',
]

LimitState = Callable[[np.ndarray], np.ndarray]

#: The Metropolis proposal's standard deviation per component, in standard-normal units (Au-Beck: ~1).
_PROPOSAL = 1.0


def in_standard_space(g: Callable[[dict[str, np.ndarray]], np.ndarray], uncertainty: Uncertainty) -> LimitState:
    """A physical limit state ``g(sample) -> (n,)`` as a function of standard-normal rows."""
    return lambda z: np.asarray(g(uncertainty.at_standard(z)), dtype=float)


def _rows(g: LimitState, z: np.ndarray) -> np.ndarray:
    values = np.ravel(np.asarray(g(z), dtype=float))
    if values.shape != (len(z),):
        refuse(f'a limit state returns one value per row: {len(z)} rows gave shape {values.shape}')
    return values


def crude_monte_carlo(g: LimitState, width: int, n: int, rng: np.random.Generator) -> Estimate:
    return probability(_rows(g, rng.standard_normal((n, width))) <= 0.0)


def importance_sampling(g: LimitState, center, n: int, rng: np.random.Generator) -> Estimate:
    center = np.asarray(center, dtype=float)
    z = center + rng.standard_normal((n, center.size))
    weights = np.exp(-z @ center + 0.5 * float(center @ center))
    return mean_estimate((_rows(g, z) <= 0.0) * weights)


def _metropolis(g: LimitState, seeds: np.ndarray, values: np.ndarray, threshold: float, steps: int, *, rng):
    """*steps* states of each chain (the seed first) inside ``g <= threshold``; their rows and values."""
    states, outputs = [seeds], [values]
    current, current_g = seeds, values
    for _ in range(steps - 1):
        candidate = current + _PROPOSAL * rng.standard_normal(current.shape)
        keep = rng.random(current.shape) >= np.exp(0.5 * (current**2 - candidate**2))
        candidate = np.where(keep, current, candidate)
        candidate_g = _rows(g, candidate)
        accept = candidate_g <= threshold
        current = np.where(accept[:, None], candidate, current)
        current_g = np.where(accept, candidate_g, current_g)
        states.append(current)
        outputs.append(current_g)
    return np.concatenate(states), np.concatenate(outputs)


def subset_simulation(
    g: LimitState, width: int, n: int, rng: np.random.Generator, *, p0: float = 0.1, max_levels: int = 20
) -> Estimate:
    """Au-Beck subset simulation with *n* samples per level; see the module docstring."""
    seeds_per_level = round(p0 * n)
    if max_levels < 1:
        refuse(f'max_levels must be at least 1, got {max_levels}')
    if not (0.0 < p0 < 1.0 and seeds_per_level >= 1 and n % seeds_per_level == 0):
        refuse(f'p0 * n must be a whole number of seeds dividing n; got p0={p0!r}, n={n}')
    z = rng.standard_normal((n, width))
    values = _rows(g, z)
    evaluations, scale, deltas = n, 1.0, 0.0
    for _ in range(max_levels - 1):
        order = np.argsort(values, kind='stable')
        threshold = float(values[order[seeds_per_level - 1]])
        if threshold <= 0.0:
            break
        scale *= p0
        deltas += (1.0 - p0) / (p0 * n)
        seeds = order[:seeds_per_level]
        z, values = _metropolis(g, z[seeds], values[seeds], threshold, n // seeds_per_level, rng=rng)
        evaluations += n - seeds_per_level
    share = float(np.mean(values <= 0.0))
    deltas += (1.0 - share) / (share * n) if share > 0 else 0.0
    value = scale * share
    return Estimate(value, value * float(np.sqrt(deltas)), evaluations)


def surrogate_reliability(
    g: LimitState,
    width: int,
    fit: Callable[[np.ndarray, np.ndarray], LimitState],
    n_train: int,
    *,
    n_mc: int,
    rng: np.random.Generator,
) -> Estimate:
    """Fit a model of *g* on *n_train* standard-normal draws, then crude Monte Carlo on the model."""
    z = rng.standard_normal((n_train, width))
    model = fit(z, _rows(g, z))
    return crude_monte_carlo(model, width, n_mc, rng)
