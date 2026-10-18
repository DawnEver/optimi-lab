"""L4 metric: the convergence indicators kept from 0.3 -- IGD and exact hypervolume.

Moved here from the retired root benchmark module (D30); the benchmark PROBLEMS went to the test
tree (``tests/benchmarks/catalogue.py``), because a problem with a known front is a test fixture
and an indicator is library numerics. P12 rebuilds these (D102 HV, D104 IGD+) and adds the rest.
Minimisation frame throughout.
"""

import numpy as np

from optimi_lab.core.errors import refuse, require_finite
from optimi_lab.core.pareto import non_dominated_sorting

__all__ = ['hypervolume', 'hypervolume_2d', 'igd']


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


def hypervolume_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Dominated hypervolume of a 2-objective ``front`` w.r.t. ``ref_point``.

    Only points that dominate ``ref_point`` contribute, and larger is better. The front is swept in
    ascending ``f1``; each step adds the rectangle it dominates. Refuses more than two objectives
    rather than silently reading the first two, because a 3-objective hypervolume is a different
    algorithm and a plausible wrong number is worse than a refusal.
    """
    if front.shape[1] != 2:
        msg = 'hypervolume_2d only supports 2 objectives.'
        raise ValueError(msg)
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
