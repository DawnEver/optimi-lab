"""Repair: bring a raw child back into the unit cube every design lives in.

Three rules for a coordinate outside ``[0, 1]`` (coordinates inside are never touched):

* ``CLIP`` -- onto the violated bound (piles points on the faces);
* ``REFLECT`` -- mirrored at the violated bound, repeatedly, so the move's length is kept;
* ``RANDOM`` -- redrawn uniformly (loses the move, keeps the distribution uniform).

Discrete coordinates are repaired after this by :func:`optimi_lab.operator.discrete.snap`.
"""

from enum import StrEnum

import numpy as np

from optimi_lab.core import coerce_enum

__all__ = ['Repair', 'repair']


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
