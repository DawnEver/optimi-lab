"""L4 metric: front indicators -- hypervolume (exact, Monte Carlo, per-point contribution), IGD, IGD+, spread.

The analytic benchmark problems are :mod:`optimi_lab.benchmark`. Minimisation frame throughout; a
volume is in the product of the objectives' units, a distance in their common unit.

Many-objective reading of one run against a reference set (both in ONE fixed frame,
:func:`fixed_normalization`, so nothing is rescaled per run): Schott's :func:`spacing`, Zhou et al.'s
:func:`generalized_spread`, the attained share :func:`coverage`, and :func:`front_metrics` -- every
one of them plus IGD+ and hypervolume at several reference levels. :func:`within_budget` is the
equal-budget cut: how many leading rows a solve budget pays for.
"""

from collections.abc import Sequence

import numpy as np

from optimi_lab.core.errors import refuse, require_finite
from optimi_lab.core.pareto import non_dominated_sorting
from optimi_lab.metric.attainment import attained

__all__ = [
    'FRONT_REFERENCES',
    'coverage',
    'fixed_normalization',
    'front_metrics',
    'generalized_spread',
    'hypervolume',
    'hypervolume_2d',
    'hypervolume_contributions',
    'hypervolume_mc',
    'igd',
    'igd_plus',
    'spacing',
    'spread',
    'within_budget',
]

#: Hypervolume reference levels per normalized axis (ideal 0, nadir 1): the interior and knee (1.01),
#: the common convention (1.1), the extremes (2.0).
FRONT_REFERENCES: tuple[float, ...] = (1.01, 1.1, 2.0)


def igd(front: np.ndarray, reference: np.ndarray) -> float:
    """Inverted Generational Distance from a reference front to ``front``.

    Mean Euclidean distance from each reference point to its nearest solution in ``front``, so
    lower is better and 0 means the reference is perfectly covered. An EMPTY ``front`` is
    ``inf`` rather than 0, because a run that found nothing is the worst result and not the best.
    """
    if front.size == 0:
        return float('inf')
    d = np.sqrt(((reference[:, None, :] - front[None, :, :]) ** 2).sum(axis=-1))
    return float(d.min(axis=1).mean())


def igd_plus(front: np.ndarray, reference: np.ndarray) -> float:
    """IGD+ (Ishibuchi et al., 2015): IGD over ``d+(a, z) = |max(a - z, 0)|``, Pareto-compliant.

    Only the part of a gap in which the solution is WORSE than the reference point counts.
    """
    if front.size == 0:
        return float('inf')
    gap = np.maximum(front[None, :, :] - reference[:, None, :], 0.0)
    return float(np.sqrt((gap**2).sum(axis=-1)).min(axis=1).mean())


def spread(front: np.ndarray, *, extremes: np.ndarray) -> float:
    """Deb's spread Delta of a two-objective front: 0 for even gaps reaching both *extremes*.

    ``(d_f + d_l + sum |d_i - mean d|) / (d_f + d_l + (N - 1) mean d)``, the gaps taken along the
    front sorted by its first objective, ``d_f`` and ``d_l`` the distances to the true extremes.
    """
    front = np.asarray(front, dtype=float)
    if front.ndim != 2 or front.shape[1] != 2 or len(front) < 2:
        refuse(f'spread reads a two-objective front of at least two points, got shape {front.shape}')
    front = front[np.argsort(front[:, 0])]
    gaps = np.linalg.norm(np.diff(front, axis=0), axis=1)
    first, last = np.asarray(extremes, dtype=float)[np.argsort(np.asarray(extremes)[:, 0])]
    d_f, d_l = np.linalg.norm(front[0] - first), np.linalg.norm(front[-1] - last)
    return float((d_f + d_l + np.abs(gaps - gaps.mean()).sum()) / (d_f + d_l + len(gaps) * gaps.mean()))


def hypervolume_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Dominated hypervolume of a 2-objective ``front`` w.r.t. ``ref_point``.

    Only points that dominate ``ref_point`` contribute, and larger is better. The front is swept in
    ascending ``f1``; each step adds the rectangle it dominates. Refuses more than two objectives
    rather than silently reading the first two, because a 3-objective hypervolume is a different
    algorithm and a plausible wrong number is worse than a refusal.
    """
    if front.shape[1] != 2:
        refuse(f'hypervolume_2d sweeps 2 objectives, got {front.shape[1]}; use hypervolume for more')
    pts = front[np.all(front < ref_point, axis=1)]
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_f2 = ref_point[1]
    for f1, f2 in pts:
        if f2 < prev_f2:
            hv += (ref_point[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact dominated hypervolume of an ``m``-objective ``front`` w.r.t. ``ref_point``, ``m >= 2``.

    Hypervolume by slicing objectives (HSO): the points are swept in ascending last objective and
    each slab up to the next value adds the ``(m - 1)``-objective volume of the points already
    swept times its depth, recursing down to :func:`hypervolume_2d`. Exact; its cost grows like
    ``n ** (m - 1)``, which is cheap for ``m <= 4`` and a few hundred points. Minimization frame;
    only points strictly dominating ``ref_point`` contribute.
    """
    front = np.asarray(front, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    if front.ndim != 2 or front.shape[1] < 2 or ref_point.shape != (front.shape[1],):
        refuse(
            f'a hypervolume needs a (n_points, m >= 2) front and a reference of width m; '
            f'got front {front.shape} and reference {ref_point.shape}'
        )
    require_finite(front, 'a front to measure')
    require_finite(ref_point, 'a hypervolume reference point')
    return _sliced_volume(front[np.all(front < ref_point, axis=1)], ref_point)


def _sliced_volume(points: np.ndarray, ref_point: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    if points.shape[1] == 2:
        return hypervolume_2d(points, ref_point)
    points = points[non_dominated_sorting(points)]
    points = points[np.argsort(points[:, -1])]
    depths = np.diff(np.append(points[:, -1], ref_point[-1]))
    return float(
        sum(
            _sliced_volume(points[: index + 1, :-1], ref_point[:-1]) * depth
            for index, depth in enumerate(depths)
            if depth > 0.0
        )
    )


def hypervolume_mc(
    front: np.ndarray, ref_point: np.ndarray, *, n_samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Monte Carlo hypervolume and its standard error: the dominated fraction of the box to the reference.

    The box spans the front's ideal to ``ref_point``; for many objectives this is the estimator the
    exact sweep's ``n ** (m - 1)`` cost leaves.
    """
    front = np.asarray(front, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    front = front[np.all(front < ref_point, axis=1)]
    if len(front) == 0:
        return 0.0, 0.0
    low = front.min(axis=0)
    box = float(np.prod(ref_point - low))
    probes = low + rng.random((n_samples, front.shape[1])) * (ref_point - low)
    hit = np.zeros(n_samples, dtype=bool)
    for point in front:
        hit |= np.all(point <= probes, axis=1)
    share = hit.mean()
    return box * share, box * float(np.sqrt(share * (1.0 - share) / n_samples))


def hypervolume_contributions(front: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    """Each point's EXCLUSIVE contribution: the volume the set loses without it (0 for a dominated point)."""
    front = np.asarray(front, dtype=float)
    total = hypervolume(front, ref_point)
    return np.array([total - hypervolume(np.delete(front, i, axis=0), ref_point) for i in range(len(front))])


def _first_front(f: np.ndarray) -> np.ndarray:
    return f[np.sort(np.asarray(non_dominated_sorting(f)).ravel())] if len(f) else f


def fixed_normalization(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(ideal, nadir)`` of minimized rows: the minimum, and the maximum of their front; a flat axis spans 1."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not len(points):
        refuse('a fixed normalization needs at least one row')
    ideal, nadir = points.min(axis=0), _first_front(points).max(axis=0)
    return ideal, np.where(nadir <= ideal, ideal + 1.0, nadir)


def spacing(front: np.ndarray) -> float:
    """Schott's spacing: the standard deviation of each point's L1 distance to its nearest neighbour (0 = even)."""
    if len(front) < 2:
        return float('nan')
    d = np.abs(front[:, None, :] - front[None, :, :]).sum(axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    return float(np.sqrt(((nearest - nearest.mean()) ** 2).sum() / (len(front) - 1)))


def generalized_spread(front: np.ndarray, reference: np.ndarray) -> float:
    """Zhou et al.'s Delta*: the reference's per-axis extremes reached and nearest gaps even (0 ideal)."""
    if len(front) < 2:
        return float('nan')
    extremes = reference[np.argmin(reference, axis=0)]
    d_extreme = np.array([np.linalg.norm(front - e, axis=1).min() for e in extremes])
    d = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    mean = nearest.mean()
    return float((d_extreme.sum() + np.abs(nearest - mean).sum()) / (d_extreme.sum() + len(front) * mean))


def coverage(front: np.ndarray, reference: np.ndarray) -> float:
    """The share of the reference points *front* weakly dominates (0 for an empty front)."""
    if not len(front):
        return 0.0
    return float(attained([front], reference).mean())


def front_metrics(
    feasible: np.ndarray, reference: np.ndarray, *, references: Sequence[float] = FRONT_REFERENCES
) -> dict[str, float]:
    """``HV(r)`` per reference level, ``IGD+``, ``coverage``, ``spacing`` and ``spread`` of normalized rows."""
    front = _first_front(feasible) if len(feasible) else np.empty((0, reference.shape[1]))
    out = {}
    for r in references:
        inside = front[np.all(front < r, axis=1)] if len(front) else front
        out[f'HV({r})'] = float(hypervolume(inside, np.full(reference.shape[1], r))) if len(inside) else 0.0
    out['IGD+'] = float(igd_plus(front, reference)) if len(front) else float('inf')
    out['coverage'] = coverage(front, reference)
    out['spacing'] = spacing(front)
    out['spread'] = generalized_spread(front, reference)
    return out


def within_budget(costs: Sequence[float], budget: float) -> int:
    """How many LEADING rows *budget* pays for, each costing its entry of *costs* (a 1e-12 relative slack)."""
    spent = np.cumsum(np.asarray(costs, dtype=float))
    return int(np.searchsorted(spent, budget * (1.0 + 1e-12), side='right'))
