"""Selection by reference direction: NSGA-III and its unified and reference-point forms, and RVEA.

* :func:`reference_select` -- NSGA-III's survivor selection (Deb and Jain, 2014): whole fronts,
  then the overflowing front filled by niching over the Das-Dennis :func:`das_dennis` directions,
  after the hyperplane normalisation of :func:`normalise`. The niche counts are taken over EVERY
  survivor already chosen; ties go to the lowest direction index and the nearest member, so the
  operator is deterministic and a run replays.
* :func:`niching_tournament` -- U-NSGA-III's mating selection (Seada and Deb, 2016): two rows of
  one niche compete on rank, then distance; two of different niches are a coin toss.
* :func:`aspiration_directions` -- R-NSGA-III's directions: a shrunken simplex around each
  aspiration point's projection onto the unit simplex.
* :func:`angle_penalized_select` and :func:`adapt_directions` -- RVEA (Cheng et al., 2016).

Minimised frame throughout; values are the normalised objectives of
:meth:`optimi_lab.problem.Objective.minimized`.
"""

from math import comb

import numpy as np

from optimi_lab.core import constrained_non_dominated_sorting, partition_count, refuse
from optimi_lab.operator.selection import require_rankable, require_selectable, violation_of

__all__ = [
    'adapt_directions',
    'angle_penalized_select',
    'aspiration_directions',
    'associate',
    'das_dennis',
    'niching_tournament',
    'normalise',
    'reference_select',
]

_AXIS_WEIGHT = 1e-6
"""The off-axis weight of the achievement function that finds an extreme point."""


def das_dennis(n_obj: int, partitions: int) -> np.ndarray:
    """Every split of ``partitions`` into ``n_obj`` parts, as unit-simplex directions, first part ascending."""
    if n_obj < 1 or partitions < 1:
        refuse(f'a Das-Dennis grid needs n_obj >= 1 and partitions >= 1, got {n_obj} and {partitions}')
    rows: list[list[int]] = [[]]
    for _ in range(n_obj - 1):
        rows = [[*row, k] for row in rows for k in range(partitions - sum(row) + 1)]
    grid = np.array([[*row, partitions - sum(row)] for row in rows], dtype=float) / partitions
    if len(grid) != comb(partitions + n_obj - 1, n_obj - 1):
        refuse('the Das-Dennis enumeration lost a composition')
    return grid


def normalise(values: np.ndarray, first_front: np.ndarray) -> np.ndarray:
    """Deb-Jain normalisation: translate by the ideal, divide by the extreme points' hyperplane intercepts.

    A degenerate hyperplane (singular, or a non-positive intercept) falls back to the first front's
    nadir, and a zero span to one, so every column is divided by a positive number.
    """
    ideal = values.min(axis=0)
    shifted = values - ideal
    n_obj = values.shape[1]
    weights = np.full((n_obj, n_obj), _AXIS_WEIGHT) + (1.0 - _AXIS_WEIGHT) * np.eye(n_obj)
    extremes = shifted[[int(np.argmin(np.max(shifted / w, axis=1))) for w in weights]]
    try:
        plane = np.linalg.solve(extremes, np.ones(n_obj))
        intercepts = 1.0 / plane
        valid = np.all(np.isfinite(intercepts)) and np.all(intercepts > _AXIS_WEIGHT)
    except np.linalg.LinAlgError:
        valid = False
    if not valid:
        intercepts = shifted[first_front].max(axis=0)
    intercepts = np.where(intercepts > _AXIS_WEIGHT, intercepts, shifted.max(axis=0))
    return shifted / np.where(intercepts > 0.0, intercepts, 1.0)


def associate(normalised: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's nearest direction by perpendicular distance, and that distance."""
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    projection = normalised @ unit.T
    distance = np.linalg.norm(normalised[:, None, :] - projection[:, :, None] * unit[None, :, :], axis=2)
    nearest = np.argmin(distance, axis=1)
    return nearest, distance[np.arange(len(normalised)), nearest]


def _directions_for(values: np.ndarray, n_selected: int, directions: np.ndarray | None, partitions: int | None):
    if directions is not None:
        if partitions is not None:
            refuse('pass directions= or partitions=, not both: the directions ARE a partition grid')
        directions = np.asarray(directions, dtype=float)
        if directions.ndim != 2 or directions.shape[1] != values.shape[1]:
            refuse(f'directions must be (n, {values.shape[1]}), got {directions.shape}')
        return directions
    if partitions is None:
        partitions = partition_count(values.shape[1], n_selected)
    return das_dennis(values.shape[1], partitions)


def reference_select(
    values: np.ndarray,
    n_selected: int,
    *,
    directions: np.ndarray | None = None,
    partitions: int | None = None,
    violation: np.ndarray | None = None,
) -> np.ndarray:
    """NSGA-III survivors: whole constrained fronts, the overflowing one filled by niching.

    ``directions`` defaults to the Das-Dennis grid of :func:`optimi_lab.core.pareto.partition_count`
    partitions, which refuses a single objective -- one objective has one direction at every count.
    """
    values = require_rankable(values)
    require_selectable(n_selected, values.shape[0])
    directions = _directions_for(values, n_selected, directions, partitions)
    fronts = constrained_non_dominated_sorting(values, violation_of(violation, len(values)), only_first_front=False)
    chosen: list[int] = []
    for front in fronts:
        if len(chosen) + len(front) > n_selected:
            survivors = np.array(chosen + [int(i) for i in front], dtype=np.intp)
            first = np.flatnonzero(np.isin(survivors, fronts[0]))
            nearest, distance = associate(normalise(values[survivors], first), directions)
            counts = np.bincount(nearest[: len(chosen)], minlength=len(directions))
            members = {
                j: sorted(np.flatnonzero(nearest[len(chosen) :] == j).tolist(), key=lambda p: distance[len(chosen) + p])
                for j in range(len(directions))
            }
            while len(chosen) < n_selected:
                j = min((j for j in members if members[j]), key=lambda j: (counts[j], j))
                chosen.append(int(front[members[j].pop(0)]))
                counts[j] += 1
            break
        chosen.extend(int(i) for i in front)
    return np.array(chosen, dtype=np.intp)


def niching_tournament(
    values: np.ndarray,
    n_selected: int,
    rng: np.random.Generator,
    *,
    directions: np.ndarray | None = None,
    violation: np.ndarray | None = None,
) -> np.ndarray:
    """U-NSGA-III mating: Deb's rule on violation, then rank and distance inside a niche, else a coin."""
    values = require_rankable(values)
    if len(values) < 2:
        refuse(f'a tournament needs two rows, got {len(values)}')
    violation = violation_of(violation, len(values))
    directions = _directions_for(values, len(values), directions, None)
    fronts = constrained_non_dominated_sorting(values, violation, only_first_front=False)
    rank = np.empty(len(values), dtype=int)
    for depth, front in enumerate(fronts):
        rank[front] = depth
    niche, distance = associate(normalise(values, fronts[0]), directions)
    winners = np.empty(n_selected, dtype=np.intp)
    for slot in range(n_selected):
        a, b = rng.choice(len(values), 2, replace=False)
        if violation[a] != violation[b]:
            winners[slot] = a if violation[a] < violation[b] else b
        elif niche[a] == niche[b]:
            winners[slot] = min((a, b), key=lambda i: (rank[i], distance[i]))
        else:
            winners[slot] = a if rng.random() < 0.5 else b
    return winners


def aspiration_directions(points: np.ndarray, *, partitions: int, spread: float) -> tuple[np.ndarray, np.ndarray]:
    """R-NSGA-III directions: ``(1 - spread) p + spread g`` for each projected point ``p`` and grid direction ``g``.

    Returns the directions and, per direction, the index of the aspiration point that owns it. Each
    direction lies on the unit simplex within ``spread`` (max-norm) of its point's projection.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or np.any(points < 0.0) or np.any(points.sum(axis=1) <= 0.0):
        refuse('aspiration points are rows of non-negative normalised objectives with a positive sum')
    if not 0.0 < spread <= 1.0:
        refuse(f'spread is the shrink factor of the simplex around each point, in (0, 1], got {spread!r}')
    projected = points / points.sum(axis=1, keepdims=True)
    grid = das_dennis(points.shape[1], partitions)
    directions = (1.0 - spread) * projected[:, None, :] + spread * grid[None, :, :]
    owner = np.repeat(np.arange(len(points)), len(grid))
    return directions.reshape(-1, points.shape[1]), owner


def angle_penalized_select(
    values: np.ndarray, *, directions: np.ndarray, progress: float, alpha: float = 2.0
) -> np.ndarray:
    """RVEA: per niche (largest cosine), the member of least angle-penalised distance.

    ``APD = (1 + M (progress ** alpha) theta / gamma) |f - z|``, ``z`` the population's ideal,
    ``theta`` the angle to the niche's direction, ``gamma`` that direction's smallest angle to
    another. ``progress`` is the run's fraction elapsed, in ``[0, 1]``: early on distance rules,
    late the angle does.
    """
    values = require_rankable(values)
    if not 0.0 <= progress <= 1.0:
        refuse(f'progress is the fraction of the run elapsed, in [0, 1], got {progress!r}')
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    shifted = values - values.min(axis=0)
    norm = np.linalg.norm(shifted, axis=1)
    cosine = np.clip(shifted @ unit.T / np.maximum(norm, np.finfo(float).tiny)[:, None], -1.0, 1.0)
    niche = np.argmax(cosine, axis=1)
    between = np.clip(unit @ unit.T, -1.0, 1.0)
    np.fill_diagonal(between, -1.0)
    gamma = np.arccos(between.max(axis=1))
    theta = np.arccos(cosine[np.arange(len(values)), niche])
    apd = (1.0 + values.shape[1] * progress**alpha * theta / gamma[niche]) * norm
    return np.array(
        [int(np.flatnonzero(niche == j)[np.argmin(apd[niche == j])]) for j in np.unique(niche)], dtype=np.intp
    )


def adapt_directions(directions: np.ndarray, values: np.ndarray) -> np.ndarray:
    """RVEA's adaptation: scale the initial directions by the population's per-objective span."""
    span = values.max(axis=0) - values.min(axis=0)
    scaled = directions * np.where(span > 0.0, span, 1.0)
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
