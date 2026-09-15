"""The five published algorithms, as ``build_proposer`` factories over the three loops.

:func:`optimi_lab.core.optimize.optimize` takes ``build_proposer: Callable[[int], Proposer]`` --
the population size, one proposer -- so each function here returns that callable and a run reads
``optimize(..., build_proposer=nsga2(space))``. The algorithms are parameterizations: NSGA-II and
NSGA-III are the same generation with a different fill of the front that overflows, MODE is that
generation with a differential variation and no mating pool, and MOEA/D and MOPSO are loops of
their own.

A setting a factory does not take is refused NAMING the ones it does, read off the operator's own
signature, so a misspelled rate is an answer rather than a ``TypeError`` from inside a call.
"""

import inspect
from collections.abc import Callable
from functools import partial

import numpy as np

from optimi_lab.core import Proposer, VariableSet, refuse
from optimi_lab.intelligent_algorithm.decomposition import DecompositionProposer
from optimi_lab.intelligent_algorithm.proposers import EvolutionaryProposer, PopulationProposer
from optimi_lab.intelligent_algorithm.selection import crowding_select, reference_select, tournament_select
from optimi_lab.intelligent_algorithm.swarm import SwarmProposer
from optimi_lab.intelligent_algorithm.variation import (
    binomial_crossover,
    differential_mutation,
    polynomial_mutation,
    simulated_binary_crossover,
)

__all__ = ['mode', 'moead', 'mopso', 'nsga2', 'nsga3']


def _require_settings(owner: Callable[..., object], settings: dict, what: str) -> None:
    """Refuse a setting ``owner`` does not take, naming the ones that it does.

    The accepted names are read off ``owner``'s signature, so the valid set a refusal names is the
    operator's own and cannot drift from it.
    """
    accepted = set(inspect.signature(owner).parameters) - {'self', 'space', 'pop_size', 'population', 'rng'}
    unknown = sorted(set(settings) - accepted)
    if unknown:
        refuse(f'{unknown} are not settings of {what}; the valid settings are {sorted(accepted)}')


def _proposer(
    owner: type[PopulationProposer], space: VariableSet, seed: int, settings: dict, what: str
) -> Callable[[int], Proposer]:
    """Return the ``build_proposer`` of ``owner`` as ``what``, refusing a setting it does not take."""
    _require_settings(owner, settings, what)

    def build(pop_size: int) -> Proposer:
        return owner(space, pop_size, seed=seed, **settings)

    return build


def _vary(variation: Callable[..., np.ndarray], settings: dict, what: str) -> Callable[..., np.ndarray]:
    """Return ``variation`` with ``settings`` bound, refusing a setting it does not take."""
    _require_settings(variation, settings, what)
    return partial(variation, **settings)


def _sbx_polynomial(
    population: np.ndarray,
    space: VariableSet,
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
        population, partner, space, crossover_rate=crossover_rate, eta_crossover=eta_crossover, rng=rng
    )
    return polynomial_mutation(children, space, mutation_rate=mutation_rate, eta_mutation=eta_mutation, rng=rng)


def _differential(
    population: np.ndarray,
    space: VariableSet,
    rng: np.random.Generator,
    *,
    sizing_factor: float = 0.5,
    crossover_rate: float = 0.5,
) -> np.ndarray:
    """One differential-evolution generation: mutate by difference vectors, then cross over."""
    trial = differential_mutation(population, space, sizing_factor=sizing_factor, rng=rng)
    return binomial_crossover(population, trial, space, crossover_rate=crossover_rate, rng=rng)


def nsga2(
    space: VariableSet,
    *,
    selection_operator: Callable[[np.ndarray, int], np.ndarray] = crowding_select,
    parent_selection_operator: Callable[..., np.ndarray] = tournament_select,
    seed: int = 0,
    **variation: float,
) -> Callable[[int], Proposer]:
    """Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002): SBX, crowding distance.

    Parents by tournament, offspring by SBX and polynomial mutation, survivors by Pareto front and
    crowding distance. ``**variation`` are the rates of
    :func:`~optimi_lab.intelligent_algorithm.variation.simulated_binary_crossover` and
    :func:`~optimi_lab.intelligent_algorithm.variation.polynomial_mutation`.
    """
    settings = {
        'variation': _vary(_sbx_polynomial, variation, 'nsga2'),
        'selection': selection_operator,
        'parent_selection': parent_selection_operator,
    }
    return _proposer(EvolutionaryProposer, space, seed, settings, 'nsga2')


def nsga3(
    space: VariableSet,
    *,
    partitions: int | None = None,
    selection_operator: Callable[[np.ndarray, int], np.ndarray] = reference_select,
    parent_selection_operator: Callable[..., np.ndarray] = tournament_select,
    seed: int = 0,
    **variation: float,
) -> Callable[[int], Proposer]:
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
        space,
        selection_operator=selection_operator,
        parent_selection_operator=parent_selection_operator,
        seed=seed,
        **variation,
    )


def mode(
    space: VariableSet,
    *,
    selection_operator: Callable[[np.ndarray, int], np.ndarray] = crowding_select,
    seed: int = 0,
    **variation: float,
) -> Callable[[int], Proposer]:
    """Multi-Objective Differential Evolution (Xue et al., 2005): difference vectors, no mating pool.

    Every individual is a parent -- a trial vector is built for each one from three others and
    recombined with it -- so there is no parent selector to install, and passing one is not
    possible rather than ignored. ``**variation`` are ``sizing_factor`` and the binomial
    ``crossover_rate``.
    """
    settings = {'variation': _vary(_differential, variation, 'mode'), 'selection': selection_operator}
    return _proposer(EvolutionaryProposer, space, seed, settings, 'mode')


def moead(space: VariableSet, *, seed: int = 0, **settings: object) -> Callable[[int], Proposer]:
    """Multi-Objective Evolutionary Algorithm based on Decomposition (Zhang and Li, 2007).

    ``**settings`` are :class:`~optimi_lab.intelligent_algorithm.decomposition.DecompositionProposer`'s:
    the neighbourhood (``neighborhood_size``, ``neighbor_rate``, ``max_replace``) and the
    SBX/mutation rates of a child. One subproblem per weight vector is ranked by Tchebycheff
    distance to the ideal point, so two objectives or more are required: one objective has nothing
    to decompose.
    """
    return _proposer(DecompositionProposer, space, seed, settings, 'moead')


def mopso(
    space: VariableSet,
    *,
    selection_operator: Callable[[np.ndarray, int], np.ndarray] = crowding_select,
    seed: int = 0,
    **settings: object,
) -> Callable[[int], Proposer]:
    """Multi-Objective Particle Swarm Optimization (Coello Coello and Lechuga, 2002).

    ``selection_operator`` thins the archive of non-dominated points the swarm draws its leaders
    from; ``**settings`` are :class:`~optimi_lab.intelligent_algorithm.swarm.SwarmProposer`'s
    ``inertia``, ``cognitive``, ``social``, ``max_velocity_fraction`` and ``archive_size``.
    """
    return _proposer(SwarmProposer, space, seed, {'selection': selection_operator, **settings}, 'mopso')
