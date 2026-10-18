"""How points are ranked and chosen: random, tournament, and Pareto fronts broken by crowding distance.

The reference-direction selectors live in :mod:`~optimi_lab.operator.reference`, the indicator
selectors (SPEA2 strength, hypervolume contribution) in :mod:`~optimi_lab.operator.indicator`; all
of them fill the overflowing front through :func:`fronts_then_fill`.

``n_selected`` is an ARGUMENT here, not a stored field. The old selectors held it, and the two
that did disagreed about being asked for more individuals than they had rows: ``RandomSelection``
raised numpy's "Cannot take a larger sample than population" from inside ``np.random.choice``
while ``TournamentSelection`` silently duplicated indices, so one request was a crash and a lie
depending only on the selector installed. :func:`require_selectable` is the one rule every
selection goes through: more rows than exist is REFUSED, naming both counts, because a duplicated
index is a population smaller than the one that was asked for.

EVERY SELECTOR TAKES ``violation``, one total constraint violation per row
(:func:`optimi_lab.core.pareto.total_violation`), and ranks by Deb's rule through
:func:`~optimi_lab.core.pareto.constrained_non_dominated_sorting`. ``None`` is every row
feasible, which is the plain Pareto ranking index for index -- an unconstrained run cannot be
moved by the constrained path, because with zero violation it IS the unconstrained path.
"""

from collections.abc import Callable

import numpy as np

from optimi_lab.core import constrained_non_dominated_sorting, refuse, require_finite

__all__ = [
    'crowding_distance',
    'crowding_select',
    'fronts_then_fill',
    'random_select',
    'require_rankable',
    'require_selectable',
    'tournament_select',
    'violation_of',
]


def require_selectable(n_selected: int, n_rows: int) -> None:
    """Refuse a selection no population can serve, naming the count asked and the count there is."""
    if not n_selected >= 1:
        refuse(f'n_selected must be at least 1, got {n_selected!r}')
    if n_selected > n_rows:
        refuse(
            f'n_selected {n_selected} exceeds the {n_rows} row(s) available; the valid n_selected here is 1 to {n_rows}'
        )


def require_rankable(values: np.ndarray, what: str = 'a fitness matrix') -> np.ndarray:
    """Return a finite 2-D objective matrix, or refuse naming what the caller passed instead."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        refuse(f'{what} must be 2-D with one column per objective, got shape {values.shape}')
    return require_finite(values, what)


def violation_of(violation: np.ndarray | None, n_rows: int) -> np.ndarray:
    """The per-row violation, every row feasible when ``None``."""
    return np.zeros(n_rows) if violation is None else np.asarray(violation, dtype=float)


def random_select(
    values: np.ndarray,
    n_selected: int,
    rng: np.random.Generator,
    *,
    violation: np.ndarray | None = None,
) -> np.ndarray:
    """Draw ``n_selected`` distinct rows uniformly, without replacement.

    ``values`` is read for its row count alone, but it is the argument both parent selectors take,
    so one of them being indifferent to fitness is a property of this strategy -- and so is its
    indifference to ``violation``, which it takes for the shared signature and does not read.
    """
    del violation
    values = require_rankable(values)
    require_selectable(n_selected, values.shape[0])
    return rng.choice(values.shape[0], n_selected, replace=False)


def tournament_select(
    values: np.ndarray,
    n_selected: int,
    rng: np.random.Generator,
    *,
    tournament_size: int = 2,
    violation: np.ndarray | None = None,
) -> np.ndarray:
    """Run ``n_selected`` tournaments and keep each winner, best front then crowding distance first.

    A tournament of one row would select that row whatever its fitness, which is why the size has
    a floor of two and a ceiling of the population.
    """
    values = require_rankable(values)
    require_selectable(n_selected, values.shape[0])
    if tournament_size < 2:
        refuse(f'tournament_size must be at least 2, got {tournament_size!r}')
    if tournament_size > values.shape[0]:
        refuse(
            f'tournament_size {tournament_size} exceeds the {values.shape[0]} row(s) a tournament may draw from; '
            f'the valid tournament sizes for this set are 2 to {values.shape[0]}'
        )
    violation = violation_of(violation, values.shape[0])
    winners = np.empty(n_selected, dtype=np.intp)
    for slot in range(n_selected):
        contenders = rng.choice(values.shape[0], tournament_size, replace=False)
        front = constrained_non_dominated_sorting(values[contenders], violation[contenders])
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
    values = require_rankable(values)
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


def crowding_select(values: np.ndarray, n_selected: int, *, violation: np.ndarray | None = None) -> np.ndarray:
    """The best ``n_selected`` rows by constrained front, the overflow front broken by crowding distance."""
    return fronts_then_fill(require_rankable(values), n_selected, _crowding_fill, violation)


def fronts_then_fill(
    values: np.ndarray,
    n_selected: int,
    fill: Callable[[np.ndarray, int], np.ndarray],
    violation: np.ndarray | None,
) -> np.ndarray:
    """Whole constrained fronts in order, the overflowing one filled by ``fill(front, n_fill)`` positions."""
    require_selectable(n_selected, values.shape[0])
    chosen: list[int] = []
    violation = violation_of(violation, values.shape[0])
    for front in constrained_non_dominated_sorting(values, violation, only_first_front=False):
        if len(chosen) + len(front) <= n_selected:
            chosen.extend(int(index) for index in front)
            continue
        chosen.extend(int(front[position]) for position in fill(values[front], n_selected - len(chosen)))
        break
    return np.array(chosen, dtype=np.intp)


def _crowding_fill(front: np.ndarray, n_fill: int) -> np.ndarray:
    """The positions of the ``n_fill`` most isolated rows of one front."""
    return np.argsort(-crowding_distance(front))[:n_fill]
