"""The five published algorithms, as factories of :class:`optimi_lab.solver.Solver` over three loops.

Each function takes the problem and the oracle capability and returns a solver for
:func:`optimi_lab.run.run`: ``run(problem, oracle, nsga2(problem, oracle.capability), record, budget)``.
The population size is the row count of ``spec``'s first sample. The algorithms are
parameterizations: NSGA-II and
NSGA-III are the same generation with a different fill of the front that overflows, MODE is that
generation with a differential variation and no mating pool, and MOEA/D and MOPSO are loops of
their own.

Every selection operator is called as ``operator(values, n_selected, [rng,] violation=...)``,
one total constraint violation per row, so a custom operator ranks by the same rule.

A setting a factory does not take is refused NAMING the ones it does, read off the operator's own
signature, so a misspelled rate is an answer rather than a ``TypeError`` from inside a call.
"""

import inspect
from collections.abc import Callable
from functools import partial

import numpy as np

from optimi_lab.core import SampleSpec, refuse
from optimi_lab.intelligent_algorithm.decomposition import DecompositionSolver
from optimi_lab.intelligent_algorithm.selection import crowding_select, reference_select, tournament_select
from optimi_lab.intelligent_algorithm.solvers import EvolutionarySolver, PopulationSolver
from optimi_lab.intelligent_algorithm.swarm import SwarmSolver
from optimi_lab.intelligent_algorithm.variation import (
    binomial_crossover,
    differential_mutation,
    polynomial_mutation,
    simulated_binary_crossover,
)
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem

__all__ = ['mode', 'moead', 'mopso', 'nsga2', 'nsga3']


def _require_settings(owner: Callable[..., object], settings: dict, what: str) -> None:
    """Refuse a setting ``owner`` does not take, naming the ones that it does.

    The accepted names are read off ``owner``'s signature, so the valid set a refusal names is the
    operator's own and cannot drift from it.
    """
    accepted = set(inspect.signature(owner).parameters) - {
        'self',
        'problem',
        'capability',
        'spec',
        'seed',
        'population',
        'width',
        'rng',
    }
    unknown = sorted(set(settings) - accepted)
    if unknown:
        refuse(f'{unknown} are not settings of {what}; the valid settings are {sorted(accepted)}')


def _settings_of(variation: Callable[..., np.ndarray], settings: dict, what: str) -> dict:
    """Return ``settings``, refusing one ``variation`` does not take."""
    _require_settings(variation, settings, what)
    return settings


def _solver(
    owner: type[PopulationSolver],
    problem: Problem,
    *,
    capability: Capability,
    spec: SampleSpec | None,
    seed: int,
    settings: dict,
    what: str,
) -> PopulationSolver:
    """``owner`` as ``what``, refusing a setting it does not take."""
    _require_settings(owner, settings, what)
    return owner(problem, capability, spec=spec, seed=seed, **settings)


def _sbx_polynomial(
    population: np.ndarray,
    width: int,
    rng: np.random.Generator,
    *,
    crossover_rate: float = 0.9,
    eta_crossover: float = 20.0,
    mutation_rate: float = 0.1,
    eta_mutation: float = 20.0,
) -> np.ndarray:
    """One SBX generation: recombine each individual with the one selected before it, then mutate."""
    partner = np.roll(population, 1, axis=0)
    children = simulated_binary_crossover(
        population, partner, width, crossover_rate=crossover_rate, eta_crossover=eta_crossover, rng=rng
    )
    return polynomial_mutation(children, width, mutation_rate=mutation_rate, eta_mutation=eta_mutation, rng=rng)


def _differential(
    population: np.ndarray,
    width: int,
    rng: np.random.Generator,
    *,
    sizing_factor: float = 0.5,
    crossover_rate: float = 0.5,
) -> np.ndarray:
    """One differential-evolution generation: mutate by difference vectors, then cross over."""
    trial = differential_mutation(population, width, sizing_factor=sizing_factor, rng=rng)
    return binomial_crossover(population, trial, width, crossover_rate=crossover_rate, rng=rng)


def nsga2(
    problem: Problem,
    capability: Capability,
    *,
    spec: SampleSpec | None = None,
    selection_operator: Callable[..., np.ndarray] = crowding_select,
    parent_selection_operator: Callable[..., np.ndarray] = tournament_select,
    seed: int = 0,
    **variation: float,
) -> PopulationSolver:
    """Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002): SBX, crowding distance.

    Parents by tournament, offspring by SBX and polynomial mutation, survivors by Pareto front and
    crowding distance. ``**variation`` are the rates of
    :func:`~optimi_lab.intelligent_algorithm.variation.simulated_binary_crossover` and
    :func:`~optimi_lab.intelligent_algorithm.variation.polynomial_mutation`.
    """
    settings = {
        'variation': _sbx_polynomial,
        'variation_settings': _settings_of(_sbx_polynomial, variation, 'nsga2'),
        'selection': selection_operator,
        'parent_selection': parent_selection_operator,
    }
    return _solver(
        EvolutionarySolver, problem, capability=capability, spec=spec, seed=seed, settings=settings, what='nsga2'
    )


def nsga3(
    problem: Problem,
    capability: Capability,
    *,
    spec: SampleSpec | None = None,
    partitions: int | None = None,
    selection_operator: Callable[..., np.ndarray] = reference_select,
    parent_selection_operator: Callable[..., np.ndarray] = tournament_select,
    seed: int = 0,
    **variation: float,
) -> PopulationSolver:
    """Non-dominated Sorting Genetic Algorithm III (Deb and Jain, 2014): reference directions.

    The generation is :func:`nsga2`'s -- SBX then polynomial mutation, with the same ``**variation``
    settings -- and only the fill of the front that overflows differs. ``partitions`` is that
    direction grid's partition count, and it belongs to
    :func:`~optimi_lab.intelligent_algorithm.selection.reference_select` alone.
    """
    if partitions is not None:
        if selection_operator is not reference_select:
            refuse(
                f'partitions is a setting of reference_select, and selection_operator here is {selection_operator!r}; '
                f'leave partitions unset, or pass selection_operator=reference_select'
            )
        selection_operator = partial(reference_select, partitions=partitions)
    return nsga2(
        problem,
        capability,
        spec=spec,
        selection_operator=selection_operator,
        parent_selection_operator=parent_selection_operator,
        seed=seed,
        **variation,
    )


def mode(
    problem: Problem,
    capability: Capability,
    *,
    spec: SampleSpec | None = None,
    selection_operator: Callable[..., np.ndarray] = crowding_select,
    seed: int = 0,
    **variation: float,
) -> PopulationSolver:
    """Multi-Objective Differential Evolution (Xue et al., 2005): difference vectors, no mating pool.

    Every individual is a parent -- a trial vector is built for each one from three others and
    recombined with it -- so there is no parent selector to install, and passing one is not
    possible rather than ignored. ``**variation`` are ``sizing_factor`` and the binomial
    ``crossover_rate``.
    """
    settings = {
        'variation': _differential,
        'variation_settings': _settings_of(_differential, variation, 'mode'),
        'selection': selection_operator,
    }
    return _solver(
        EvolutionarySolver, problem, capability=capability, spec=spec, seed=seed, settings=settings, what='mode'
    )


def moead(
    problem: Problem, capability: Capability, *, spec: SampleSpec | None = None, seed: int = 0, **settings: object
) -> PopulationSolver:
    """Multi-Objective Evolutionary Algorithm based on Decomposition (Zhang and Li, 2007).

    ``**settings`` are :class:`~optimi_lab.intelligent_algorithm.decomposition.DecompositionSolver`'s:
    the neighbourhood (``neighborhood_size``, ``neighbor_rate``, ``max_replace``) and the
    SBX/mutation rates of a child. One subproblem per weight vector is ranked by Tchebycheff
    distance to the ideal point, so two objectives or more are required: one objective has nothing
    to decompose.
    """
    return _solver(
        DecompositionSolver, problem, capability=capability, spec=spec, seed=seed, settings=settings, what='moead'
    )


def mopso(
    problem: Problem,
    capability: Capability,
    *,
    spec: SampleSpec | None = None,
    selection_operator: Callable[..., np.ndarray] = crowding_select,
    seed: int = 0,
    **settings: object,
) -> PopulationSolver:
    """Multi-Objective Particle Swarm Optimization (Coello Coello and Lechuga, 2002).

    ``selection_operator`` thins the archive of non-dominated points the swarm draws its leaders
    from; ``**settings`` are :class:`~optimi_lab.intelligent_algorithm.swarm.SwarmSolver`'s
    ``inertia``, ``cognitive``, ``social``, ``max_velocity_fraction`` and ``archive_size``.
    """
    return _solver(
        SwarmSolver,
        problem,
        capability=capability,
        spec=spec,
        seed=seed,
        settings={'selection': selection_operator, **settings},
        what='mopso',
    )
