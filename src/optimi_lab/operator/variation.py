"""How one generation is made from the last: crossover, mutation and the swarm updates.

Each operator is a function of the arrays it acts on, the unit-cube WIDTH they live in and the
generator it draws from, so nothing here reads module state. Every operator works in the unit cube a
:class:`~optimi_lab.problem.Space` encodes into, so a bound is enforced in one place (:func:`_clip`)
and a velocity limit is a fraction of a range by construction -- the old absolute ``v_max`` of 0.1
was, on a range 5000 wide, a particle that cannot move.
"""

from enum import StrEnum

import numpy as np

from optimi_lab.core import coerce_enum, refuse, require_axis

__all__ = [
    'DEStrategy',
    'binomial_crossover',
    'differential_mutation',
    'exponential_crossover',
    'polynomial_mutation',
    'pso_position',
    'pso_velocity',
    'simulated_binary_crossover',
]


class DEStrategy(StrEnum):
    """The differential-evolution mutation strategies (Storn and Price, 1997; Das and Suganthan, 2011)."""

    RAND_1 = 'rand/1'
    RAND_2 = 'rand/2'
    BEST_1 = 'best/1'
    BEST_2 = 'best/2'
    CURRENT_TO_BEST_1 = 'current-to-best/1'
    CURRENT_TO_RAND_1 = 'current-to-rand/1'


#: How many distinct OTHER rows each strategy draws; the population must exceed it.
_DONORS = {
    DEStrategy.RAND_1: 3,
    DEStrategy.RAND_2: 5,
    DEStrategy.BEST_1: 2,
    DEStrategy.BEST_2: 4,
    DEStrategy.CURRENT_TO_BEST_1: 2,
    DEStrategy.CURRENT_TO_RAND_1: 3,
}
_NEEDS_BEST = frozenset({DEStrategy.BEST_1, DEStrategy.BEST_2, DEStrategy.CURRENT_TO_BEST_1})


def _clip(points: np.ndarray) -> np.ndarray:
    """A copy of *points* inside the unit cube, where every operator works."""
    return np.clip(points, 0.0, 1.0)


def _require_rate(rate: float, what: str) -> float:
    """Return a probability in ``[0, 1]``, or refuse naming the interval it must lie in."""
    if not 0.0 <= rate <= 1.0:
        refuse(f'{what} must be in [0, 1], got {rate!r}')
    return rate


def _require_pair(parent_1: np.ndarray, parent_2: np.ndarray, width: int):
    """Return both parents as matching ``(n_points, n_var)`` arrays, or refuse naming the shapes."""
    first = require_axis(parent_1, width, 1, 'parent_1')
    second = require_axis(parent_2, width, 1, 'parent_2')
    if first.shape != second.shape:
        refuse(f'parent_1 and parent_2 must hold one row per pair, got {first.shape} and {second.shape}')
    return first, second


def binomial_crossover(
    parent_1: np.ndarray,
    parent_2: np.ndarray,
    width: int,
    *,
    crossover_rate: float = 0.5,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take each gene from ``parent_2`` with probability ``crossover_rate``, forcing one per row.

    The forced gene keeps the operator total: an all-False mask would return a copy of ``parent_1``.
    """
    parent_1, parent_2 = _require_pair(parent_1, parent_2, width)
    _require_rate(crossover_rate, 'crossover_rate')
    n_points, n_var = parent_1.shape
    donor = rng.random((n_points, n_var)) < crossover_rate
    donor[np.arange(n_points), rng.integers(0, n_var, n_points)] = True
    return _clip(np.where(donor, parent_2, parent_1))


def simulated_binary_crossover(
    parent_1: np.ndarray,
    parent_2: np.ndarray,
    width: int,
    *,
    crossover_rate: float = 0.9,
    eta_crossover: float = 20.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """SBX: one child per pair, spread around the parents by the index ``eta_crossover``.

    Each gene takes one of SBX's two mirrored children at random, so the operator is symmetric
    about the parents' midpoint; a gene whose two parents are equal is copied through.
    """
    parent_1, parent_2 = _require_pair(parent_1, parent_2, width)
    _require_rate(crossover_rate, 'crossover_rate')
    if not eta_crossover >= 0:
        refuse(f'eta_crossover must be non-negative, got {eta_crossover!r}')
    lower = np.minimum(parent_1, parent_2)
    upper = np.maximum(parent_1, parent_2)
    crossing = (rng.random(parent_1.shape) < crossover_rate) & (upper > lower)
    draw = rng.random(parent_1.shape)
    exponent = 1.0 / (eta_crossover + 1.0)
    beta = np.where(draw <= 0.5, np.power(2.0 * draw, exponent), np.power(1.0 / (2.0 * (1.0 - draw)), exponent))
    first = 0.5 * ((1.0 + beta) * lower + (1.0 - beta) * upper)
    second = 0.5 * ((1.0 - beta) * lower + (1.0 + beta) * upper)
    child = np.where(rng.random(parent_1.shape) < 0.5, first, second)
    return _clip(np.where(crossing, child, parent_1))


def polynomial_mutation(
    offspring: np.ndarray,
    width: int,
    *,
    mutation_rate: float = 0.1,
    eta_mutation: float = 20.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb each gene with probability ``mutation_rate``, near its value for a large ``eta``.

    The gene is mapped onto its own variable's ``[0, 1]`` interval first, so one setting is the same
    relative move on a 5000-wide variable and on a 1-wide one.
    """
    offspring = require_axis(offspring, width, 1, 'the offspring to mutate')
    _require_rate(mutation_rate, 'mutation_rate')
    if not eta_mutation > 0:
        refuse(f'eta_mutation must be greater than 0, got {eta_mutation!r}')
    span = 1.0
    position = np.clip(offspring, 0.0, 1.0)
    mutate = rng.random(offspring.shape) < mutation_rate
    draw = rng.random(offspring.shape)
    exponent = 1.0 / (eta_mutation + 1.0)
    delta = np.where(
        draw <= 0.5,
        np.power(2.0 * draw + (1.0 - 2.0 * draw) * np.power(1.0 - position, eta_mutation + 1.0), exponent) - 1.0,
        1.0 - np.power(2.0 * (1.0 - draw) + 2.0 * (draw - 0.5) * np.power(position, eta_mutation + 1.0), exponent),
    )
    return _clip(offspring + delta * span * mutate)


def exponential_crossover(
    parent_1: np.ndarray,
    parent_2: np.ndarray,
    width: int,
    *,
    crossover_rate: float = 0.5,
    rng: np.random.Generator,
) -> np.ndarray:
    """DE's exponential crossover: one cyclic run of genes from ``parent_2``, from a random start.

    The run continues while a draw stays under ``crossover_rate``, so its length is geometric and at
    least one gene is always taken.
    """
    parent_1, parent_2 = _require_pair(parent_1, parent_2, width)
    _require_rate(crossover_rate, 'crossover_rate')
    n_points, n_var = parent_1.shape
    donor = np.zeros((n_points, n_var), dtype=bool)
    for row in range(n_points):
        start, length = rng.integers(n_var), 1
        while length < n_var and rng.random() < crossover_rate:
            length += 1
        donor[row, (start + np.arange(length)) % n_var] = True
    return _clip(np.where(donor, parent_2, parent_1))


def _trial(strategy: DEStrategy, x: np.ndarray, *, i: int, b: int | None, r: np.ndarray, f: float) -> np.ndarray:
    """One strategy's published formula for row *i*, best row *b*, donors *r*."""
    match strategy:
        case DEStrategy.RAND_1:
            return x[r[0]] + f * (x[r[1]] - x[r[2]])
        case DEStrategy.RAND_2:
            return x[r[0]] + f * (x[r[1]] - x[r[2]]) + f * (x[r[3]] - x[r[4]])
        case DEStrategy.BEST_1:
            return x[b] + f * (x[r[0]] - x[r[1]])
        case DEStrategy.BEST_2:
            return x[b] + f * (x[r[0]] - x[r[1]]) + f * (x[r[2]] - x[r[3]])
        case DEStrategy.CURRENT_TO_BEST_1:
            return x[i] + f * (x[b] - x[i]) + f * (x[r[0]] - x[r[1]])
        case _:
            return x[i] + f * (x[r[0]] - x[i]) + f * (x[r[1]] - x[r[2]])


def differential_mutation(
    population: np.ndarray,
    width: int,
    *,
    strategy: DEStrategy | str = DEStrategy.RAND_1,
    sizing_factor: float = 0.5,
    best: np.ndarray | None = None,
    rng: np.random.Generator,
    clip: bool = True,
) -> np.ndarray:
    """One trial vector per row by *strategy*; ``best`` is the best row's index, one per row.

    Donors are distinct rows other than the row itself, so a population no larger than the donor
    count is REFUSED rather than mutated by copying (the 0.3 body copied the individual through).
    ``clip=False`` returns the raw trial, so a repair other than the clip can be applied.
    """
    strategy = coerce_enum(DEStrategy, strategy, 'DE strategy')
    population = require_axis(population, width, 1, 'the population to mutate')
    n_points, count = population.shape[0], _DONORS[strategy]
    if n_points < count + 1:
        refuse(
            f'DE {strategy.value} draws {count} rows other than the target, so it needs at least {count + 1} rows, '
            f'got {n_points}; the valid population sizes start there'
        )
    if not 0.0 < sizing_factor <= 2.0:
        refuse(f'sizing_factor must be in (0, 2], got {sizing_factor!r}')
    if strategy in _NEEDS_BEST:
        if best is None:
            refuse(f'DE {strategy.value} moves toward the best row; pass best= one row index per row')
        best = np.broadcast_to(np.asarray(best, dtype=np.intp), (n_points,))
    x, f = population, sizing_factor
    trial = np.empty_like(x)
    for i in range(n_points):
        r = rng.choice(np.delete(np.arange(n_points), i), count, replace=False)
        trial[i] = _trial(strategy, x, i=i, b=None if best is None else best[i], r=r, f=f)
    return _clip(trial) if clip else trial


def pso_velocity(
    velocities: np.ndarray,
    positions: np.ndarray,
    personal_best: np.ndarray,
    global_best: np.ndarray | None,
    width: int,
    *,
    inertia: float = 0.4,
    cognitive: float = 2.0,
    social: float = 2.0,
    max_velocity_fraction: float = 0.5,
    rng: np.random.Generator,
) -> np.ndarray:
    """``v <- inertia*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)``, held inside a fraction of the range.

    ``max_velocity_fraction`` multiplies each variable's OWN span, so one setting is one behaviour
    everywhere. An empty archive means there is no leader and the social term is zero -- the swarm
    is pulled by its own bests alone until a point has been ranked onto the archive.
    """
    positions = require_axis(positions, width, 1, 'positions')
    velocities = require_axis(velocities, width, 1, 'velocities')
    personal_best = require_axis(personal_best, width, 1, 'personal_best_positions')
    if velocities.shape != positions.shape or personal_best.shape != positions.shape:
        refuse(
            f'velocities {velocities.shape} and personal_best_positions {personal_best.shape} must each hold one row '
            f'per particle, as positions {positions.shape} does'
        )
    if not 0.0 < max_velocity_fraction <= 1.0:
        refuse(
            f'max_velocity_fraction is the fraction of a variable range a particle may cross in one step, so it must '
            f'be in (0, 1], got {max_velocity_fraction!r}'
        )
    if global_best is None or len(global_best) == 0:
        leader = positions
    else:
        archive = require_axis(global_best, width, 1, 'global_best_positions')
        leader = archive[rng.integers(0, archive.shape[0], positions.shape[0])]
    updated = (
        inertia * velocities
        + cognitive * rng.random(positions.shape) * (personal_best - positions)
        + social * rng.random(positions.shape) * (leader - positions)
    )
    limit = max_velocity_fraction
    return np.clip(updated, -limit, limit)


def pso_position(positions: np.ndarray, velocities: np.ndarray, width: int) -> np.ndarray:
    """``x <- x + v``, clipped onto the unit cube.

    The clip is the only thing holding a particle inside the search interval.
    """
    positions = require_axis(positions, width, 1, 'positions')
    velocities = require_axis(velocities, width, 1, 'velocities')
    if velocities.shape != positions.shape:
        refuse(f'velocities {velocities.shape} must hold one row per particle, as positions {positions.shape} does')
    return _clip(positions + velocities)
