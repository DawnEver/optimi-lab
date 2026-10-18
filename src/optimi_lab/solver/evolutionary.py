"""The (mu + lambda) evolutionary algorithms, as factories of :class:`EvolutionarySolver`.

Each factory takes the problem and the oracle capability and returns a solver for
:func:`optimi_lab.run.run`. They are PARAMETERISATIONS of one loop -- a variation, a survivor
selection, a mating selection -- so each is a few lines naming its operators:

==========  ==================================  ===================================  ==================
factory     survivors                           mating                               offspring
==========  ==================================  ===================================  ==================
ga          constrained rank on ONE score       tournament                           SBX + PM
nsga2       fronts + crowding                   tournament                           SBX + PM
nsga3       fronts + reference niching          tournament                           SBX + PM
unsga3      fronts + reference niching          niching tournament                   SBX + PM
rnsga3      niching on aspiration directions    tournament                           SBX + PM
spea2       strength + density truncation       binary tournament on SPEA2 fitness   SBX + PM
sms_emoa    fronts + least HV contribution      random                               ONE child (SBX+PM)
mode        fronts + crowding                   none (every row is a target)         DE strategy + crossover
==========  ==================================  ===================================  ==================

``**settings`` are the base solver's (``spec``, ``initial``, ``scalarise``, ``seed``,
``discrete_mutation_rate``) and the variation's rates; a setting neither takes is refused naming the
valid set, read off the variation's own signature.
"""

import inspect
from collections.abc import Callable
from functools import partial

import numpy as np

from optimi_lab.core import constrained_non_dominated_sorting, refuse
from optimi_lab.operator import (
    DEStrategy,
    aspiration_directions,
    binomial_crossover,
    crowding_select,
    differential_mutation,
    exponential_crossover,
    hypervolume_select,
    niching_tournament,
    polynomial_mutation,
    random_select,
    reference_select,
    simulated_binary_crossover,
    spea2_fitness,
    spea2_select,
    tournament_select,
)
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.population import LOOP_ARGUMENTS, EvolutionarySolver, require_one_score

__all__ = [
    'differential',
    'evolutionary',
    'ga',
    'mode',
    'nsga2',
    'nsga3',
    'rnsga3',
    'sbx_polynomial',
    'sms_emoa',
    'spea2',
    'unsga3',
]

#: Settings that belong to the base solver rather than to a variation.
_BASE = frozenset({'spec', 'initial', 'scalarise', 'seed', 'discrete_mutation_rate'})


def sbx_polynomial(
    population: np.ndarray,
    width: int,
    rng: np.random.Generator,
    *,
    values: np.ndarray,
    violation: np.ndarray,
    crossover_rate: float = 0.9,
    eta_crossover: float = 20.0,
    mutation_rate: float = 0.1,
    eta_mutation: float = 20.0,
) -> np.ndarray:
    """One SBX generation: recombine each individual with the one selected before it, then mutate."""
    del values, violation
    partner = np.roll(population, 1, axis=0)
    children = simulated_binary_crossover(
        population, partner, width, crossover_rate=crossover_rate, eta_crossover=eta_crossover, rng=rng
    )
    return polynomial_mutation(children, width, mutation_rate=mutation_rate, eta_mutation=eta_mutation, rng=rng)


def differential(
    population: np.ndarray,
    width: int,
    rng: np.random.Generator,
    *,
    values: np.ndarray,
    violation: np.ndarray,
    strategy: str = DEStrategy.RAND_1,
    sizing_factor: float = 0.5,
    crossover_rate: float = 0.5,
    crossover: str = 'binomial',
) -> np.ndarray:
    """One DE generation: a trial per row by *strategy*, crossed with its target (binomial or exponential).

    A "best" strategy's best row is drawn per target from the constrained first front, so on several
    objectives each target is pulled toward SOME non-dominated row rather than one privileged one.
    """
    if crossover not in {'binomial', 'exponential'}:
        refuse(f"crossover is 'binomial' or 'exponential', got {crossover!r}")
    first = constrained_non_dominated_sorting(values, violation)
    best = rng.choice(first, len(population))
    trial = differential_mutation(population, width, strategy=strategy, sizing_factor=sizing_factor, best=best, rng=rng)
    cross = binomial_crossover if crossover == 'binomial' else exponential_crossover
    return cross(population, trial, width, crossover_rate=crossover_rate, rng=rng)


def _spea2_mating(
    values: np.ndarray, n_selected: int, rng: np.random.Generator, *, violation: np.ndarray
) -> np.ndarray:
    """SPEA2's binary tournament: the lower SPEA2 fitness of two random rows wins."""
    fitness = spea2_fitness(values, violation=violation)
    pairs = rng.integers(0, len(values), (n_selected, 2))
    return np.where(fitness[pairs[:, 0]] <= fitness[pairs[:, 1]], pairs[:, 0], pairs[:, 1])


def evolutionary(
    problem: Problem,
    capability: Capability,
    what: str,
    *,
    variation: Callable[..., np.ndarray],
    selection: Callable[..., np.ndarray],
    parent_selection: Callable[..., np.ndarray] | None,
    settings: dict[str, object],
    n_offspring: int | None = None,
    owner: type[EvolutionarySolver] = EvolutionarySolver,
    own: frozenset[str] = frozenset(),
) -> EvolutionarySolver:
    """*owner* over *variation*/*selection*; a setting is a variation rate, a base setting or one of *own*."""
    rates = set(inspect.signature(variation).parameters) - LOOP_ARGUMENTS
    unknown = sorted(set(settings) - rates - _BASE - own)
    if unknown:
        refuse(f'{unknown} are not settings of {what}; the valid settings are {sorted(rates | _BASE | own)}')
    return owner(
        problem,
        capability,
        variation=variation,
        variation_settings={k: v for k, v in settings.items() if k in rates},
        selection=selection,
        parent_selection=parent_selection,
        n_offspring=n_offspring,
        **{k: v for k, v in settings.items() if k in _BASE | own},
    )


def ga(problem: Problem, capability: Capability, **settings: object) -> EvolutionarySolver:
    """A real-coded elitist genetic algorithm on ONE score (Deb's constrained ranking)."""
    require_one_score(problem, settings.get('scalarise'), 'ga')
    return evolutionary(
        problem,
        capability,
        'ga',
        variation=sbx_polynomial,
        selection=crowding_select,
        parent_selection=tournament_select,
        settings=settings,
    )


def nsga2(
    problem: Problem,
    capability: Capability,
    *,
    selection_operator: Callable[..., np.ndarray] = crowding_select,
    parent_selection_operator: Callable[..., np.ndarray] = tournament_select,
    **settings: object,
) -> EvolutionarySolver:
    """NSGA-II (Deb et al., 2002): fronts and crowding distance, SBX and polynomial mutation."""
    return evolutionary(
        problem,
        capability,
        'nsga2',
        variation=sbx_polynomial,
        selection=selection_operator,
        parent_selection=parent_selection_operator,
        settings=settings,
    )


def nsga3(
    problem: Problem,
    capability: Capability,
    *,
    partitions: int | None = None,
    selection_operator: Callable[..., np.ndarray] = reference_select,
    parent_selection_operator: Callable[..., np.ndarray] = tournament_select,
    **settings: object,
) -> EvolutionarySolver:
    """NSGA-III (Deb and Jain, 2014): NSGA-II's generation, the overflow filled by reference niching."""
    if partitions is not None:
        if selection_operator is not reference_select:
            refuse('partitions is a setting of reference_select; leave it unset with another selection_operator')
        selection_operator = partial(reference_select, partitions=partitions)
    return nsga2(
        problem,
        capability,
        selection_operator=selection_operator,
        parent_selection_operator=parent_selection_operator,
        **settings,
    )


def unsga3(problem: Problem, capability: Capability, **settings: object) -> EvolutionarySolver:
    """U-NSGA-III (Seada and Deb, 2016): NSGA-III with a niching tournament for mating."""
    return nsga3(problem, capability, parent_selection_operator=niching_tournament, **settings)


def rnsga3(
    problem: Problem,
    capability: Capability,
    *,
    aspiration: np.ndarray,
    partitions: int = 4,
    spread: float = 0.2,
    **settings: object,
) -> EvolutionarySolver:
    """R-NSGA-III: reference niching over directions clustered around *aspiration* points.

    *aspiration* rows are in the minimised normalised objective frame; ``spread`` is how far (on
    the unit simplex) the directions around one point reach.
    """
    directions, _ = aspiration_directions(np.atleast_2d(aspiration), partitions=partitions, spread=spread)
    return nsga2(problem, capability, selection_operator=partial(reference_select, directions=directions), **settings)


def spea2(problem: Problem, capability: Capability, **settings: object) -> EvolutionarySolver:
    """SPEA2 (Zitzler, Laumanns and Thiele, 2001): the population IS the archive, truncated by density."""
    return evolutionary(
        problem,
        capability,
        'spea2',
        variation=sbx_polynomial,
        selection=spea2_select,
        parent_selection=_spea2_mating,
        settings=settings,
    )


def sms_emoa(
    problem: Problem, capability: Capability, *, reference: np.ndarray | None = None, **settings: object
) -> EvolutionarySolver:
    """SMS-EMOA (Beume et al., 2007): steady state, one child, the least hypervolume contributor leaves.

    ``reference`` is the hypervolume reference point in the minimised normalised frame (None: the
    overflowing front's nadir plus one).
    """
    return evolutionary(
        problem,
        capability,
        'sms_emoa',
        variation=sbx_polynomial,
        selection=partial(hypervolume_select, reference=reference),
        parent_selection=random_select,
        settings=settings,
        n_offspring=1,
    )


def mode(
    problem: Problem,
    capability: Capability,
    *,
    selection_operator: Callable[..., np.ndarray] = crowding_select,
    **settings: object,
) -> EvolutionarySolver:
    """Multi-objective differential evolution (Xue et al., 2005), any :class:`DEStrategy`, no mating pool."""
    return evolutionary(
        problem,
        capability,
        'mode',
        variation=differential,
        selection=selection_operator,
        parent_selection=None,
        settings=settings,
    )
