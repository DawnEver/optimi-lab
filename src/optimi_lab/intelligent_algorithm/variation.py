"""How one generation is made from the last: crossover, mutation and the swarm updates.

Each operator is a function of the arrays it acts on, the space it must stay inside and the
generator it draws from, so nothing here reads module state. Measures this module removes:
``best_inputs``, a mutation argument documented "not used; present for API compatibility"; the
``None`` bounds that made ``SimulatedBinaryCrossover`` raise ``TypeError`` on ``>=`` -- a
:class:`~optimi_lab.core.space.VariableSet` refuses ``lower >= upper``, so no bound-less space can
arrive and :meth:`VariableSet.clip` is the one place a bound is enforced; and
``PSOVelocityUpdate.v_max``, documented "relative to variable range" and clipped as an ABSOLUTE
0.1, which on a range 5000 wide is a particle that cannot move.
"""

import numpy as np

from optimi_lab.core import VariableSet, refuse, require_axis

__all__ = [
    'binomial_crossover',
    'differential_mutation',
    'polynomial_mutation',
    'pso_position',
    'pso_velocity',
    'simulated_binary_crossover',
]

_MINIMUM_DIFFERENTIAL_POPULATION = 4
"""DE/rand/1 draws a base and three other rows, so it is defined from four rows up."""


def _require_rate(rate: float, what: str) -> float:
    """Return a probability in ``[0, 1]``, or refuse naming the interval it must lie in."""
    if not 0.0 <= rate <= 1.0:
        refuse(f'{what} must be in [0, 1], got {rate!r}')
    return rate


def _require_pair(parent_1: np.ndarray, parent_2: np.ndarray, space: VariableSet):
    """Return both parents as matching ``(n_points, n_var)`` arrays, or refuse naming the shapes."""
    first = require_axis(parent_1, space.n_var, 1, 'parent_1')
    second = require_axis(parent_2, space.n_var, 1, 'parent_2')
    if first.shape != second.shape:
        refuse(f'parent_1 and parent_2 must hold one row per pair, got {first.shape} and {second.shape}')
    return first, second


def binomial_crossover(
    parent_1: np.ndarray,
    parent_2: np.ndarray,
    space: VariableSet,
    *,
    crossover_rate: float = 0.5,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take each gene from ``parent_2`` with probability ``crossover_rate``, forcing one per row.

    The forced gene keeps the operator total: an all-False mask would return a copy of ``parent_1``.
    """
    parent_1, parent_2 = _require_pair(parent_1, parent_2, space)
    _require_rate(crossover_rate, 'crossover_rate')
    n_points, n_var = parent_1.shape
    donor = rng.random((n_points, n_var)) < crossover_rate
    donor[np.arange(n_points), rng.integers(0, n_var, n_points)] = True
    return space.clip(np.where(donor, parent_2, parent_1))


def simulated_binary_crossover(
    parent_1: np.ndarray,
    parent_2: np.ndarray,
    space: VariableSet,
    *,
    crossover_rate: float = 0.9,
    eta_crossover: float = 20.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """SBX: one child per pair, spread around the parents by the index ``eta_crossover``.

    The parents are sorted per gene first, so neither argument is privileged, and a gene whose two
    parents are equal is copied through: there is no span to draw a spread from.
    """
    parent_1, parent_2 = _require_pair(parent_1, parent_2, space)
    _require_rate(crossover_rate, 'crossover_rate')
    if not eta_crossover >= 0:
        refuse(f'eta_crossover must be non-negative, got {eta_crossover!r}')
    lower = np.minimum(parent_1, parent_2)
    upper = np.maximum(parent_1, parent_2)
    crossing = (rng.random(parent_1.shape) < crossover_rate) & (upper > lower)
    draw = rng.random(parent_1.shape)
    exponent = 1.0 / (eta_crossover + 1.0)
    beta = np.where(draw <= 0.5, np.power(2.0 * draw, exponent), np.power(1.0 / (2.0 * (1.0 - draw)), exponent))
    child = 0.5 * ((1.0 + beta) * lower + (1.0 - beta) * upper)
    return space.clip(np.where(crossing, child, parent_1))


def polynomial_mutation(
    offspring: np.ndarray,
    space: VariableSet,
    *,
    mutation_rate: float = 0.1,
    eta_mutation: float = 20.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb each gene with probability ``mutation_rate``, near its value for a large ``eta``.

    The gene is mapped onto its own variable's ``[0, 1]`` interval first, so one setting is the same
    relative move on a 5000-wide variable and on a 1-wide one.
    """
    offspring = require_axis(offspring, space.n_var, 1, 'the offspring to mutate')
    _require_rate(mutation_rate, 'mutation_rate')
    if not eta_mutation > 0:
        refuse(f'eta_mutation must be greater than 0, got {eta_mutation!r}')
    span = space.upper_bounds - space.lower_bounds
    position = np.clip((offspring - space.lower_bounds) / span, 0.0, 1.0)
    mutate = rng.random(offspring.shape) < mutation_rate
    draw = rng.random(offspring.shape)
    exponent = 1.0 / (eta_mutation + 1.0)
    delta = np.where(
        draw <= 0.5,
        np.power(2.0 * draw + (1.0 - 2.0 * draw) * np.power(1.0 - position, eta_mutation + 1.0), exponent) - 1.0,
        1.0 - np.power(2.0 * (1.0 - draw) + 2.0 * (draw - 0.5) * np.power(position, eta_mutation + 1.0), exponent),
    )
    return space.clip(offspring + delta * span * mutate)


def differential_mutation(
    population: np.ndarray, space: VariableSet, *, sizing_factor: float = 0.5, rng: np.random.Generator
) -> np.ndarray:
    """DE/rand/1: ``x[r1] + sizing_factor * (x[r2] - x[r3])``, three other rows drawn per row.

    Fewer than four rows is REFUSED: the old body copied the individual through, a mutation that
    mutates nothing while reporting success.
    """
    population = require_axis(population, space.n_var, 1, 'the population to mutate')
    n_points = population.shape[0]
    if n_points < _MINIMUM_DIFFERENTIAL_POPULATION:
        refuse(
            f'differential mutation draws x[r1], x[r2] and x[r3] from other rows, so it needs at least '
            f'{_MINIMUM_DIFFERENTIAL_POPULATION} rows, got {n_points}; the valid population sizes start there'
        )
    if not 0.0 < sizing_factor <= 2.0:
        refuse(f'sizing_factor must be in (0, 2], got {sizing_factor!r}')
    donors = np.empty((n_points, 3), dtype=np.intp)
    for row in range(n_points):
        donors[row] = rng.choice(np.delete(np.arange(n_points), row), 3, replace=False)
    trial = population[donors[:, 0]] + sizing_factor * (population[donors[:, 1]] - population[donors[:, 2]])
    return space.clip(trial)


def pso_velocity(
    velocities: np.ndarray,
    positions: np.ndarray,
    personal_best: np.ndarray,
    global_best: np.ndarray | None,
    space: VariableSet,
    *,
    inertia: float = 0.4,
    cognitive: float = 2.0,
    social: float = 2.0,
    max_velocity_fraction: float = 0.1,
    rng: np.random.Generator,
) -> np.ndarray:
    """``v <- inertia*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)``, held inside a fraction of the range.

    ``max_velocity_fraction`` multiplies each variable's OWN span, so one setting is one behaviour
    everywhere. An empty archive means there is no leader and the social term is zero -- the swarm
    is pulled by its own bests alone until a point has been ranked onto the archive.
    """
    positions = require_axis(positions, space.n_var, 1, 'positions')
    velocities = require_axis(velocities, space.n_var, 1, 'velocities')
    personal_best = require_axis(personal_best, space.n_var, 1, 'personal_best_positions')
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
        archive = require_axis(global_best, space.n_var, 1, 'global_best_positions')
        leader = archive[rng.integers(0, archive.shape[0], positions.shape[0])]
    updated = (
        inertia * velocities
        + cognitive * rng.random(positions.shape) * (personal_best - positions)
        + social * rng.random(positions.shape) * (leader - positions)
    )
    limit = max_velocity_fraction * (space.upper_bounds - space.lower_bounds)
    return np.clip(updated, -limit, limit)


def pso_position(positions: np.ndarray, velocities: np.ndarray, space: VariableSet) -> np.ndarray:
    """``x <- x + v``, clipped onto the bounds ``space`` declares.

    ``space`` is read, not decorative: it is the only thing holding a particle inside the search
    interval, and a range of zero cannot exist here to make the clip a no-op.
    """
    positions = require_axis(positions, space.n_var, 1, 'positions')
    velocities = require_axis(velocities, space.n_var, 1, 'velocities')
    if velocities.shape != positions.shape:
        refuse(f'velocities {velocities.shape} must hold one row per particle, as positions {positions.shape} does')
    return space.clip(positions + velocities)
