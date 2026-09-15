"""Dominance, non-dominated sorting, and the size of a Das-Dennis reference grid.

Everything here works in the MINIMIZATION frame — every column smaller-is-better. A caller with
a maximizing objective negates that column first (:meth:`optimi_lab.core.space.ObjectiveSet.
to_minimization`); nothing in this module knows about directions, and nothing here reads a name.

Non-finite input is refused rather than sorted. ``NaN <= x`` is False for every comparison, so a
point holding a NaN is dominated by nothing and dominates nothing — it lands on the first front
by arithmetic accident. The only rows that reach here are the ones whose every objective
produced a value (:meth:`optimi_lab.core.outcomes.Evaluation.complete`).
"""

from math import comb

import numpy as np

from optimi_lab.core.errors import refuse, require_finite, require_positive

__all__ = ['dominates', 'non_dominated_sorting', 'partition_count']

_MAX_PARTITIONS = 10_000
"""Ceiling on the partition search, so a grid that cannot be reached terminates."""


def dominates(obj_1: np.ndarray, obj_2: np.ndarray) -> np.ndarray | bool:
    """Whether ``obj_1`` dominates ``obj_2`` in a minimization problem.

    Dominance is "no worse everywhere AND strictly better somewhere", broadcast over the last
    axis, so ``dominates(a[:, None, :], b[None, :, :])`` answers every pair at once.
    """
    return np.all(obj_1 <= obj_2, axis=-1) & np.any(obj_1 < obj_2, axis=-1)


def non_dominated_sorting(values: np.ndarray, only_first_front: bool = True) -> np.ndarray | list[np.ndarray]:
    """Group a ``(n_points, n_obj)`` matrix into Pareto fronts, in the minimization frame.

    Returns the first front's indices, or with ``only_first_front=False`` every front as a list
    of index arrays in dominance order. Refuses a matrix that is not 2-D with at least one
    column, or that holds a non-finite entry — see the module docstring for why.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        refuse(
            f'an objective matrix must be 2-D with one column per objective, got shape {values.shape}; '
            f'a front is read over at least one objective'
        )
    require_finite(values, 'an objective matrix to sort')
    n_points = values.shape[0]
    if n_points == 0:
        return np.array([], dtype=np.intp)

    dominated_by = dominates(values[:, None, :], values[None, :, :])
    domination_count = np.sum(dominated_by.T, axis=1)
    current_front = np.where(domination_count == 0)[0]
    if only_first_front:
        return current_front

    fronts = [current_front]
    dominated_points = [np.where(dominated_by[index])[0].tolist() for index in range(n_points)]
    remaining = domination_count.copy()
    while current_front.size:
        next_front = []
        for index in current_front:
            for dominated_index in dominated_points[index]:
                remaining[dominated_index] -= 1
                if remaining[dominated_index] == 0:
                    next_front.append(dominated_index)
        if next_front:
            fronts.append(np.array(next_front))
        current_front = np.array(next_front)
    return fronts


def partition_count(n_obj: int, n_points: int) -> int:
    """Smallest Das-Dennis partition count whose reference grid covers ``n_points``.

    ``h`` partitions over ``n_obj`` objectives generate ``comb(h + n_obj - 1, n_obj - 1)``
    directions, which is 1 for every ``h`` when ``n_obj`` is 1 — the old loop asked for
    ``comb(h, 0) >= 2`` and incremented ``h`` forever, so ``das_dennis_partition_count(1, 2)``
    never returned. Refining a grid needs two objectives to refine over, so ``n_obj < 2`` is
    refused, and the search carries a ceiling of its own.
    """
    if n_obj < 2:
        refuse(
            f'a Das-Dennis reference grid needs at least 2 objectives to refine over, got n_obj={n_obj}; '
            f'with one objective the grid holds comb(h, 0) = 1 direction for every h'
        )
    require_positive(n_points, 'n_points')
    partitions = 1
    while comb(partitions + n_obj - 1, n_obj - 1) < n_points:
        partitions += 1
        if partitions > _MAX_PARTITIONS:
            refuse(
                f'no Das-Dennis grid over {n_obj} objectives holds {n_points} points within {_MAX_PARTITIONS} '
                f'partitions; the largest it holds is {comb(_MAX_PARTITIONS + n_obj - 1, n_obj - 1)}'
            )
    return partitions
