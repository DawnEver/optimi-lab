"""Scalarisations: one objective vector to one number, each declaring how much of a front it can REACH.

``scalarisation(values, weights, reference, **parameters)`` scores every row of ``values`` (minimised
frame) under a weight row (or one row per value row) relative to a reference point -- the ideal for
most, the aspiration for ASF. The ``reach`` (B0-02) says which Pareto points some setting of the
parameter can make optimal:

* ``CONVEX_HULL`` -- only points on the front's convex hull: a weighted sum cannot reach the middle
  of a concave front at any weight.
* ``WHOLE_FRONT`` -- every Pareto point for some setting (Tchebycheff, ASF, epsilon-constraint,
  NBI; NBI may also return a non-Pareto boundary point).
* ``PARAMETER_DEPENDENT`` -- PBI: its penalty ``theta`` decides how concave a region it reaches.

:func:`decomposition_weights` and :func:`nearest_neighbors` build MOEA/D's subproblems.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from optimi_lab.core import partition_count, refuse, require_axis
from optimi_lab.operator.reference import das_dennis

__all__ = [
    'SCALARISATIONS',
    'Reach',
    'Scalarisation',
    'asf',
    'decomposition_weights',
    'epsilon_constraint',
    'nbi',
    'nearest_neighbors',
    'pbi',
    'tchebycheff',
    'weighted_sum',
]


class Reach(StrEnum):
    """Which Pareto points a scalarisation can make optimal."""

    CONVEX_HULL = 'convex_hull'
    WHOLE_FRONT = 'whole_front'
    PARAMETER_DEPENDENT = 'parameter_dependent'


@dataclass(frozen=True, slots=True)
class Scalarisation:
    """A named scalarisation and its declared reach; calling it scores rows."""

    name: str
    reach: Reach
    function: Callable[..., np.ndarray]

    def __call__(self, values: np.ndarray, weights: np.ndarray, reference: np.ndarray, **parameters) -> np.ndarray:
        weights = np.atleast_2d(np.asarray(weights, dtype=float))
        values = require_axis(np.atleast_2d(np.asarray(values, dtype=float)), weights.shape[1], 1, 'objective rows')
        if values.shape[0] != weights.shape[0] and 1 not in (values.shape[0], weights.shape[0]):
            refuse(f'values {values.shape} and weights {weights.shape} pair row by row, or one side is a single row')
        if np.any(weights < 0.0):
            refuse(f'{self.name} weights are importances and must be non-negative')
        return self.function(values, weights, np.asarray(reference, dtype=float), **parameters)


def _weighted_sum(values: np.ndarray, weights: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.sum(weights * (values - reference), axis=1)


def _tchebycheff(values: np.ndarray, weights: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.max(weights * np.abs(values - reference), axis=1)


def _asf(values: np.ndarray, weights: np.ndarray, reference: np.ndarray, *, rho: float = 1e-6) -> np.ndarray:
    scaled = (values - reference) / np.maximum(weights, np.finfo(float).eps)
    return scaled.max(axis=1) + rho * scaled.sum(axis=1)


def _pbi(values: np.ndarray, weights: np.ndarray, reference: np.ndarray, *, theta: float = 5.0) -> np.ndarray:
    unit = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    d1 = np.sum((values - reference) * unit, axis=1)
    d2 = np.linalg.norm(values - reference - d1[:, None] * unit, axis=1)
    return d1 + theta * d2


def _epsilon_constraint(
    values: np.ndarray, weights: np.ndarray, reference: np.ndarray, *, bounds: np.ndarray, penalty: float = 1e3
) -> np.ndarray:
    """The objective of largest weight, plus ``penalty`` times the excess of every other over its bound."""
    primary = np.broadcast_to(np.argmax(weights, axis=1), (len(values),))
    excess = np.maximum(values - np.asarray(bounds, dtype=float), 0.0)
    excess[np.arange(len(values)), primary] = 0.0
    return values[np.arange(len(values)), primary] - reference[primary] + penalty * excess.sum(axis=1)


def _nbi(
    values: np.ndarray,
    weights: np.ndarray,
    reference: np.ndarray,
    *,
    anchors: np.ndarray | None = None,
    theta: float = 1e3,
) -> np.ndarray:
    """Normal-boundary intersection (Das and Dennis, 1998) as a penalty: travel ``t`` along the
    quasi-normal from the hull point ``reference + Phi beta``, penalised by the distance off that line.

    ``anchors`` are the individual minima (one row per objective); None takes them from ``values``.
    """
    anchors = values[np.argmin(values, axis=0)] if anchors is None else np.asarray(anchors, dtype=float)
    shell = anchors - reference
    normal = -shell.sum(axis=0)
    normal /= np.linalg.norm(normal)
    foot = reference + (weights / weights.sum(axis=1, keepdims=True)) @ shell
    offset = values - foot
    t = offset @ normal
    return -t + theta * np.linalg.norm(offset - t[:, None] * normal, axis=1)


weighted_sum = Scalarisation('weighted_sum', Reach.CONVEX_HULL, _weighted_sum)
tchebycheff = Scalarisation('tchebycheff', Reach.WHOLE_FRONT, _tchebycheff)
asf = Scalarisation('asf', Reach.WHOLE_FRONT, _asf)
pbi = Scalarisation('pbi', Reach.PARAMETER_DEPENDENT, _pbi)
epsilon_constraint = Scalarisation('epsilon_constraint', Reach.WHOLE_FRONT, _epsilon_constraint)
nbi = Scalarisation('nbi', Reach.WHOLE_FRONT, _nbi)

#: Every scalarisation this package ships, so a test holds each one to its declared reach.
SCALARISATIONS = (weighted_sum, tchebycheff, asf, pbi, epsilon_constraint, nbi)


def decomposition_weights(n_obj: int, n_points: int, *, rng: np.random.Generator) -> np.ndarray:
    """``n_points`` weight vectors on the unit simplex: an even sweep, a thinned Das-Dennis grid, or Dirichlet.

    One objective is refused rather than one subproblem repeated.

    Examples:
        >>> import numpy as np
        >>> decomposition_weights(2, 3, rng=np.random.default_rng(0)).tolist()
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]

    """
    if n_obj < 2:
        refuse(f'a decomposition needs at least 2 objectives, got {n_obj}; one objective is one subproblem')
    if n_obj == 2:
        sweep = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
        return np.hstack([sweep, 1.0 - sweep])
    if n_obj == 3:
        grid = das_dennis(n_obj, partition_count(n_obj, n_points))
        return grid if len(grid) == n_points else grid[rng.choice(len(grid), n_points, replace=False)]
    return rng.dirichlet(np.ones(n_obj), n_points)


def nearest_neighbors(weights: np.ndarray, size: int) -> np.ndarray:
    """The ``size`` closest weight vectors to each one, itself included, as index rows."""
    if size < 1:
        refuse(f'size must be at least 1, got {size!r}')
    if size > len(weights):
        refuse(f'a neighbourhood of {size} weight vector(s) exceeds the {len(weights)} there are')
    distances = np.linalg.norm(weights[:, np.newaxis, :] - weights[np.newaxis, :, :], axis=2)
    return np.argsort(distances, axis=1)[:, :size]
