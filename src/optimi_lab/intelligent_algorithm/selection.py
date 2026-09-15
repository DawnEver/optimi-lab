"""How points are ranked and chosen: Pareto fronts, crowding, reference directions, Tchebycheff.

``n_selected`` is an ARGUMENT here, not a stored field. The old selectors held it, and the two
that did disagreed about being asked for more individuals than they had rows: ``RandomSelection``
raised numpy's "Cannot take a larger sample than population" from inside ``np.random.choice``
while ``TournamentSelection`` silently duplicated indices, so one request was a crash and a lie
depending only on the selector installed. :func:`_require_selectable` is the one rule every
selection goes through: more rows than exist is REFUSED, naming both counts, because a duplicated
index is a population smaller than the one that was asked for.
"""

from collections.abc import Callable, Iterator
from functools import partial

import numpy as np

from optimi_lab.core import non_dominated_sorting, partition_count, refuse, require_axis, require_finite

__all__ = [
    'crowding_distance',
    'crowding_select',
    'decomposition_weights',
    'nearest_neighbors',
    'random_select',
    'reference_select',
    'tchebycheff',
    'tournament_select',
]


def _require_selectable(n_selected: int, n_rows: int) -> None:
    """Refuse a selection no population can serve, naming the count asked and the count there is."""
    if not n_selected >= 1:
        refuse(f'n_selected must be at least 1, got {n_selected!r}')
    if n_selected > n_rows:
        refuse(
            f'n_selected {n_selected} exceeds the {n_rows} row(s) available; the valid n_selected here is 1 to {n_rows}'
        )


def _require_rankable(values: np.ndarray, what: str = 'a fitness matrix') -> np.ndarray:
    """Return a finite 2-D objective matrix, or refuse naming what the caller passed instead."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        refuse(f'{what} must be 2-D with one column per objective, got shape {values.shape}')
    return require_finite(values, what)


def random_select(values: np.ndarray, n_selected: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_selected`` distinct rows uniformly, without replacement.

    ``values`` is read for its row count alone, but it is the argument both parent selectors take,
    so one of them being indifferent to fitness is a property of this strategy.
    """
    values = _require_rankable(values)
    _require_selectable(n_selected, values.shape[0])
    return rng.choice(values.shape[0], n_selected, replace=False)


def tournament_select(
    values: np.ndarray, n_selected: int, rng: np.random.Generator, *, tournament_size: int = 2
) -> np.ndarray:
    """Run ``n_selected`` tournaments and keep each winner, best front then crowding distance first.

    A tournament of one row would select that row whatever its fitness, which is why the size has
    a floor of two and a ceiling of the population.
    """
    values = _require_rankable(values)
    _require_selectable(n_selected, values.shape[0])
    if tournament_size < 2:
        refuse(f'tournament_size must be at least 2, got {tournament_size!r}')
    if tournament_size > values.shape[0]:
        refuse(
            f'tournament_size {tournament_size} exceeds the {values.shape[0]} row(s) a tournament may draw from; '
            f'the valid tournament sizes for this set are 2 to {values.shape[0]}'
        )
    winners = np.empty(n_selected, dtype=np.intp)
    for slot in range(n_selected):
        contenders = rng.choice(values.shape[0], tournament_size, replace=False)
        front = non_dominated_sorting(values[contenders])
        if len(front) == 1:
            winners[slot] = contenders[front[0]]
        else:
            winners[slot] = contenders[front[np.argmax(crowding_distance(values[contenders][front]))]]
    return winners


def crowding_distance(values: np.ndarray) -> np.ndarray:
    """Per-row crowding distance, infinite at each objective's two extreme rows.

    Examples:
        >>> import numpy as np
        >>> crowding_distance(np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])).tolist()
        [inf, inf, 2.0]

    """
    values = _require_rankable(values)
    n_points, n_objectives = values.shape
    if n_points <= 2:
        return np.full(n_points, np.inf)
    distance = np.zeros(n_points)
    for column in range(n_objectives):
        order = np.argsort(values[:, column])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = values[order[-1], column] - values[order[0], column]
        if span > 0.0:
            distance[order[1:-1]] += (values[order[2:], column] - values[order[:-2], column]) / span
    return distance


def crowding_select(values: np.ndarray, n_selected: int) -> np.ndarray:
    """The best ``n_selected`` rows by Pareto front, the overflow front broken by crowding distance."""
    return _fronts_then_fill(_require_rankable(values), n_selected, _crowding_fill)


def reference_select(values: np.ndarray, n_selected: int, *, partitions: int | None = None) -> np.ndarray:
    """The same fronts, the overflow front filled by proximity to a Das-Dennis reference direction.

    ``partitions`` defaults to :func:`optimi_lab.core.pareto.partition_count`, which refuses a
    single-objective problem -- one objective holds one direction at every partition count, so
    there is nothing to refine and the old search for a count never returned.
    """
    values = _require_rankable(values)
    _require_selectable(n_selected, values.shape[0])
    if partitions is None:
        partitions = partition_count(values.shape[1], n_selected)
    directions = _reference_directions(values.shape[1], partitions)
    return _fronts_then_fill(values, n_selected, partial(_reference_fill, directions=directions))


def _fronts_then_fill(values: np.ndarray, n_selected: int, fill: Callable[[np.ndarray, int], np.ndarray]) -> np.ndarray:
    """Whole fronts in order, the one that overflows filled by ``fill(front, n_fill)`` positions."""
    _require_selectable(n_selected, values.shape[0])
    chosen: list[int] = []
    for front in non_dominated_sorting(values, only_first_front=False):
        if len(chosen) + len(front) <= n_selected:
            chosen.extend(int(index) for index in front)
            continue
        chosen.extend(int(front[position]) for position in fill(values[front], n_selected - len(chosen)))
        break
    return np.array(chosen, dtype=np.intp)


def _crowding_fill(front: np.ndarray, n_fill: int) -> np.ndarray:
    """The positions of the ``n_fill`` most isolated rows of one front."""
    return np.argsort(-crowding_distance(front))[:n_fill]


def _reference_directions(n_obj: int, partitions: int) -> np.ndarray:
    """Every way to split ``partitions`` into ``n_obj`` parts, as unit-simplex directions.

    Examples:
        >>> _reference_directions(2, 2).tolist()
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]

    """
    if partitions < 1:
        refuse(f'partitions must be at least 1, got {partitions!r}')
    return np.array(list(_compositions(partitions, n_obj)), dtype=float) / partitions


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Every way to write ``total`` as ``parts`` non-negative integers, first part ascending."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _reference_fill(front: np.ndarray, n_fill: int, *, directions: np.ndarray) -> np.ndarray:
    """Positions of the rows a niching step keeps: the emptiest direction first, its nearest row first.

    The front reaching here has more rows than ``n_fill``, so a direction with a row left always
    exists and the loop has no fallback to fall through to -- the old version had one, and it
    could never be reached.
    """
    distances = _distance_to_directions(_to_unit_box(front), directions)
    nearest = np.argmin(distances, axis=1)
    members = [np.flatnonzero(nearest == direction).tolist() for direction in range(directions.shape[0])]
    chosen: list[int] = []
    for _ in range(n_fill):
        direction = min(
            (index for index, rows in enumerate(members) if rows), key=lambda index: (len(members[index]), index)
        )
        row = min(members[direction], key=lambda position: distances[position, direction])
        members[direction].remove(row)
        chosen.append(row)
    return np.array(chosen, dtype=np.intp)


def _to_unit_box(values: np.ndarray) -> np.ndarray:
    """Map a front onto ``[0, 1]`` per objective, by its own ideal and nadir points."""
    ideal = values.min(axis=0)
    span = values.max(axis=0) - ideal
    return (values - ideal) / np.where(span > 0.0, span, 1.0)


def _distance_to_directions(values: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Perpendicular distance from every row to every direction, shape ``(n_rows, n_directions)``."""
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    projection = values @ unit.T
    return np.linalg.norm(values[:, np.newaxis, :] - projection[:, :, np.newaxis] * unit[np.newaxis, :, :], axis=2)


def tchebycheff(values: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """``max_m(weights[m] * |values[m] - ideal[m]|)`` per row: a decomposition's rank of a point.

    The worst column decides, so a row is never preferred for being good at one objective alone.
    ``values`` and ``weights`` pair row by row, or one side holds a single row -- scoring one
    offspring against a whole neighbourhood of weight vectors is the common case.
    """
    values = require_axis(values, weights.shape[1], 1, 'an objective matrix to decompose')
    if values.shape[0] != weights.shape[0] and 1 not in (values.shape[0], weights.shape[0]):
        refuse(
            f'values {values.shape} and weights {weights.shape} must pair row by row, or hold a single row on one side'
        )
    if not np.all(weights >= 0.0):
        refuse('weights must be non-negative; a Tchebycheff weight is the importance of its objective')
    return np.max(weights * np.abs(values - ideal), axis=1)


def decomposition_weights(n_obj: int, n_points: int, *, rng: np.random.Generator) -> np.ndarray:
    """``n_points`` weight vectors on the unit simplex: an even sweep, a grid, or a Dirichlet draw.

    Two objectives get an even sweep, three get a Das-Dennis grid thinned without replacement, and
    more get a Dirichlet draw. The partition count comes from
    :func:`~optimi_lab.core.pareto.partition_count`, so the grid always covers the population it is
    thinned to, and one objective is refused rather than silently one subproblem repeated.

    Examples:
        >>> import numpy as np
        >>> decomposition_weights(2, 3, rng=np.random.default_rng(0)).tolist()
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]

    """
    if n_obj < 2:
        refuse(
            f'a decomposition needs at least 2 objectives, got {n_obj}; with one objective every weight vector is the same subproblem'
        )
    if n_obj == 2:
        sweep = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
        return np.hstack([sweep, 1.0 - sweep])
    if n_obj == 3:
        grid = _reference_directions(n_obj, partition_count(n_obj, n_points))
        return grid if len(grid) == n_points else grid[rng.choice(len(grid), n_points, replace=False)]
    return rng.dirichlet(np.ones(n_obj), n_points)


def nearest_neighbors(weights: np.ndarray, size: int) -> np.ndarray:
    """The ``size`` closest weight vectors to each one, itself included, as index rows."""
    if size < 1:
        refuse(f'size must be at least 1, got {size!r}')
    if size > len(weights):
        refuse(
            f'a neighbourhood of {size} weight vector(s) exceeds the {len(weights)} there are; the valid sizes are 1 to {len(weights)}'
        )
    distances = np.linalg.norm(weights[:, np.newaxis, :] - weights[np.newaxis, :, :], axis=2)
    return np.argsort(distances, axis=1)[:, :size]
