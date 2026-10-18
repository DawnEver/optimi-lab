"""Decision support over a finished front: redundancy, a knee, a TOPSIS ranking and a power law.

Every objective matrix here is in the MINIMIZATION frame of :mod:`optimi_lab.core.pareto` -- the
caller signs and normalises each column once, through :meth:`~optimi_lab.problem.Objective.minimized`
-- so no function takes a direction and none reads a name. Feasibility is
:meth:`~optimi_lab.oracle.Result.violation` and non-domination is
:func:`~optimi_lab.core.pareto.non_dominated_sorting`; this module reuses both rather than holding
a copy. The caller declares thresholds and weights; this module only applies them.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numerics_lab.linear.dense import least_squares

from optimi_lab.core.errors import refuse, require_finite
from optimi_lab.core.pareto import non_dominated_sorting
from optimi_lab.metric import rank, spearman

__all__ = ['PowerLaw', 'effective_dimension', 'knee_index', 'monotone_links', 'power_law_fit', 'topsis']


def _matrix(values: np.ndarray, what: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        refuse(f'{what} must be a 2-D matrix of at least two rows and one column, got shape {values.shape}')
    return require_finite(values, what)


def monotone_links(
    names: Sequence[str], values: np.ndarray, *, threshold: float = 0.99
) -> list[tuple[str, str, float]]:
    """Every column pair whose Spearman ``|rho| >= threshold``: one is (near) monotone in the other.

    A constant column correlates with nothing (its rho is NaN) and is never linked.
    """
    values = _matrix(values, 'a matrix to correlate')
    if len(names) != values.shape[1]:
        refuse(f'one name per column: got {len(names)} name(s) for {values.shape[1]} column(s)')
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = np.atleast_2d(spearman(values))
    return [
        (names[i], names[j], float(rho[i, j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if np.isfinite(rho[i, j]) and abs(rho[i, j]) >= threshold
    ]


def effective_dimension(values: np.ndarray, *, share: float = 0.95) -> int:
    """How many principal components of the standardised column RANKS carry ``share`` of the variance.

    Ranks make the count blind to monotone transforms, so ``x`` and ``log x`` are one direction. A
    matrix whose every column is constant has no direction at all and reads 0.
    """
    ranks = rank(_matrix(values, 'a matrix to reduce'), axis=0)
    ranks -= ranks.mean(axis=0)
    spread = ranks.std(axis=0)
    ranks = ranks[:, spread > 0] / spread[spread > 0]
    if ranks.shape[1] == 0:
        return 0
    energy = np.linalg.svd(ranks, compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(energy) / energy.sum(), share - 1e-12) + 1)


@dataclass(frozen=True, slots=True)
class PowerLaw:
    """``y = coefficient * prod(x_k ** exponents_k)`` fitted in log space; ``r2`` near 1 says it is one."""

    exponents: tuple[float, ...]
    coefficient: float
    r2: float


def power_law_fit(inputs: np.ndarray, outputs: np.ndarray) -> PowerLaw | None:
    """Least squares of ``log y`` on ``log x``; ``None`` when any value is not strictly positive.

    ``None`` is the answer "this is not a power law", not a failure: a non-positive value has no
    logarithm, so no product of powers reaches it.
    """
    inputs = _matrix(inputs, 'the power-law inputs')
    outputs = np.asarray(outputs, dtype=float)
    if outputs.shape != (inputs.shape[0],):
        refuse(f'one output per input row: got outputs {outputs.shape} for inputs {inputs.shape}')
    require_finite(outputs, 'the power-law outputs')
    if (inputs <= 0).any() or (outputs <= 0).any():
        return None
    design = np.column_stack([np.ones(len(outputs)), np.log(inputs)])
    log_out = np.log(outputs)
    coef = least_squares(design, log_out)
    residual = float(((log_out - design @ coef) ** 2).sum())
    spread = float(((log_out - log_out.mean()) ** 2).sum())
    r2 = 1.0 - residual / spread if spread > 0 else 1.0
    return PowerLaw(tuple(float(p) for p in coef[1:]), float(np.exp(coef[0])), r2)


def knee_index(values: np.ndarray) -> int:
    """The knee of a two-objective set: its front point farthest beyond the extremes' chord.

    Both columns are scaled to ``[0, 1]`` first, so neither unit dominates; the front is the set's
    own first front, and "beyond" is the side of the chord facing the ideal ``(0, 0)``.
    """
    values = _matrix(values, 'a set to find the knee of')
    if values.shape[1] != 2:
        refuse(f'a knee is read on two objectives, got {values.shape[1]}; project the set first')
    low, high = values.min(axis=0), values.max(axis=0)
    cost = np.where(high > low, (values - low) / np.where(high > low, high - low, 1.0), 0.0)
    front = non_dominated_sorting(cost)
    first, second = front[np.argmin(cost[front, 0])], front[np.argmin(cost[front, 1])]
    chord = cost[second] - cost[first]
    if not chord.any():
        return int(first)
    normal = np.array([-chord[1], chord[0]])
    if normal @ (-cost[first]) < 0:
        normal = -normal
    return int(front[np.argmax((cost[front] - cost[first]) @ normal)])


def topsis(values: np.ndarray, weights: Sequence[float] | None = None) -> np.ndarray:
    """TOPSIS closeness in ``[0, 1]`` per row, 1 at the ideal: vector-normalised, weights summed to 1."""
    values = _matrix(values, 'a set to rank')
    w = np.ones(values.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (values.shape[1],) or (w < 0).any() or w.sum() == 0:
        refuse(f'weights {w.tolist()}: need {values.shape[1]} non-negative values, not all zero')
    norm = np.sqrt((values * values).sum(axis=0))
    scaled = (w / w.sum()) * values / np.where(norm > 0, norm, 1.0)
    to_ideal = np.sqrt(((scaled - scaled.min(axis=0)) ** 2).sum(axis=1))
    to_anti = np.sqrt(((scaled - scaled.max(axis=0)) ** 2).sum(axis=1))
    total = to_ideal + to_anti
    return np.where(total > 0, to_anti / np.where(total > 0, total, 1.0), 1.0)
