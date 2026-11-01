"""Repair: bring a raw child back into the unit cube every design lives in.

Three rules for a coordinate outside ``[0, 1]`` (coordinates inside are never touched):

* ``CLIP`` -- onto the violated bound (piles points on the faces);
* ``REFLECT`` -- mirrored at the violated bound, repeatedly, so the move's length is kept;
* ``RANDOM`` -- redrawn uniformly (loses the move, keeps the distribution uniform).

Discrete coordinates are repaired after this by :func:`optimi_lab.operator.discrete.snap`.

Two repairs outside the cube rule: :func:`round_to_level` puts a RELAXED level index (a real stand-in
for a categorical) on its nearest level and says whether that moved it; :func:`bisect_to_admissible`
moves an inadmissible point toward an admissible target only as far as a bisection on the segment
says it must.
"""

from collections.abc import Callable
from enum import StrEnum

import numpy as np

from optimi_lab.core import coerce_enum, refuse

__all__ = ['Repair', 'bisect_to_admissible', 'repair', 'round_to_level']


class Repair(StrEnum):
    """How an out-of-bounds coordinate is brought back."""

    CLIP = 'clip'
    REFLECT = 'reflect'
    RANDOM = 'random'


def repair(points: np.ndarray, how: Repair | str, *, rng: np.random.Generator) -> np.ndarray:
    """A copy of *points* inside the unit cube by rule *how*."""
    how = coerce_enum(Repair, how, 'repair rule')
    points = np.array(points, dtype=float)
    outside = (points < 0.0) | (points > 1.0)
    if how is Repair.CLIP:
        fixed = np.clip(points, 0.0, 1.0)
    elif how is Repair.REFLECT:
        folded = np.mod(points, 2.0)
        fixed = np.where(folded > 1.0, 2.0 - folded, folded)
    else:
        fixed = rng.random(points.shape)
    return np.where(outside, fixed, points)


def round_to_level(raw: float, n_levels: int) -> tuple[int, bool]:
    """The level index ``rint(raw)`` clipped to ``[0, n_levels - 1]``, and whether it differs from *raw*."""
    if n_levels < 1:
        refuse(f'a level index needs at least one level, got {n_levels}')
    index = int(np.clip(np.rint(float(raw)), 0, n_levels - 1))
    return index, float(raw) != float(index)


def bisect_to_admissible(
    point: np.ndarray, target: np.ndarray, admissible: Callable[[np.ndarray], bool], *, steps: int
) -> tuple[np.ndarray, float]:
    """``(point + t (target - point), t)`` with ``t`` the admissible end of *steps* bisections on ``[0, 1]``.

    *target* is taken as admissible (``t = 1`` is never re-checked); the returned point is the last one
    the bisection found admissible, or *target* itself.
    """
    point, target = np.asarray(point, dtype=float), np.asarray(target, dtype=float)
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        if admissible(point + mid * (target - point)):
            hi = mid
        else:
            lo = mid
    return point + hi * (target - point), hi
