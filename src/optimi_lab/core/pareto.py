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

__all__ = [
    'constrained_dominates',
    'constrained_non_dominated_sorting',
    'dominates',
    'non_dominated_sorting',
    'partition_count',
    'total_violation',
]

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


def total_violation(shortfalls: np.ndarray) -> np.ndarray:
    """Per row, the sum of a ``(n_points, n_con)`` matrix of normalised shortfalls.

    A row holding a NaN sums to NaN: a constraint that did not compute leaves its point's
    feasibility UNKNOWN, and an unknown is never read as satisfied. Zero constraint columns sum to
    zero, which is what makes an unconstrained run the feasible special case of a constrained one.
    """
    shortfalls = np.asarray(shortfalls, dtype=float)
    if shortfalls.ndim != 2:
        refuse(f'a shortfall matrix must be 2-D with one column per constraint, got shape {shortfalls.shape}')
    return shortfalls.sum(axis=1)


def constrained_dominates(
    obj_1: np.ndarray, obj_2: np.ndarray, violation_1: np.ndarray | float, violation_2: np.ndarray | float
) -> np.ndarray | bool:
    """Deb's constrained dominance of ``obj_1`` over ``obj_2``, in the minimization frame.

    A feasible point (violation 0) beats an infeasible one whatever their objectives; of two
    infeasible points the smaller violation wins; two feasible points compare by
    :func:`dominates`. Broadcast like :func:`dominates`, the violations over the leading axes.
    """
    violation_1 = np.asarray(violation_1, dtype=float)
    violation_2 = np.asarray(violation_2, dtype=float)
    both_feasible = (violation_1 == 0.0) & (violation_2 == 0.0)
    return (violation_1 < violation_2) | (both_feasible & dominates(obj_1, obj_2))


def constrained_non_dominated_sorting(
    values: np.ndarray, violation: np.ndarray, only_first_front: bool = True
) -> np.ndarray | list[np.ndarray]:
    """Group rows into fronts under Deb's rule: the feasible fronts first, then by violation.

    The feasible rows are sorted by :func:`non_dominated_sorting`; every infeasible row follows,
    one front per distinct violation in ascending order, because two infeasible rows of equal
    violation dominate neither way. With every row feasible this IS :func:`non_dominated_sorting`
    -- the same call, so an unconstrained run cannot be moved by the constrained path. A violation
    that is NaN (unknown), negative or misaligned with ``values`` is refused.
    """
    values = np.asarray(values, dtype=float)
    violation = np.asarray(violation, dtype=float)
    if violation.ndim != 1 or violation.shape[0] != values.shape[0]:
        refuse(
            f'a constrained sort needs one violation per row, got shape {violation.shape} for {values.shape[0]} row(s)'
        )
    require_finite(violation, 'a violation vector to sort')
    if np.any(violation < 0.0):
        refuse('a violation is a sum of shortfalls and is never negative; got a negative entry')
    feasible = np.flatnonzero(violation == 0.0)
    if feasible.size == violation.size:
        return non_dominated_sorting(values, only_first_front=only_first_front)
    fronts: list[np.ndarray] = []
    if feasible.size:
        sorted_feasible = non_dominated_sorting(values[feasible], only_first_front=False)
        fronts.extend(feasible[front] for front in sorted_feasible)
    else:
        require_finite(values, 'an objective matrix to sort')
    fronts.extend(np.flatnonzero(violation == level) for level in np.unique(violation[violation > 0.0]))
    return fronts[0] if only_first_front else fronts
