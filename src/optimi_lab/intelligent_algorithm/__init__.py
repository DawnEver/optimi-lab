"""The proposers: five published algorithms, three loops, one kit of operators.

Every proposer here implements :class:`optimi_lab.core.protocols.Proposer`: :meth:`ask` returns a
batch of shape ``(n_points, n_var)`` and :meth:`tell` accepts an
:class:`~optimi_lab.core.outcomes.Evaluation` whose values are already in the MINIMIZATION frame.
A run is :func:`optimi_lab.core.optimize.optimize`, which draws the first batch with
:func:`optimi_lab.core.sampling.sample`, hands that row count to one of the factories below, and
stacks every batch into a :class:`~optimi_lab.core.optimize.Record`.

    >>> import numpy as np
    >>> from optimi_lab import Evaluation, Objective, ObjectiveSet, SampleSpec, Variable, VariableSet, optimize
    >>> from optimi_lab.intelligent_algorithm import nsga2
    >>> space = VariableSet([Variable('x', 0.0, 1.0), Variable('y', 0.0, 1.0)])
    >>> objectives = ObjectiveSet([Objective('to_origin'), Objective('to_one')])
    >>> def evaluate(points):
    ...     return Evaluation(
    ...         inputs=points,
    ...         values=np.column_stack([np.sum(points**2, axis=1), np.sum((points - 1.0) ** 2, axis=1)]),
    ...     )
    >>> record = optimize(
    ...     space=space,
    ...     objectives=objectives,
    ...     evaluate=evaluate,
    ...     build_proposer=nsga2(space),
    ...     spec=SampleSpec(n_samples=8, seed=1),
    ...     n_batches=3,
    ... )
    >>> len(record), len(record.pareto()) > 0
    (24, True)

The five algorithms are parameterizations of the three loops:
:mod:`~optimi_lab.intelligent_algorithm.algorithms` binds the operators the
:mod:`~optimi_lab.intelligent_algorithm.proposers`, :mod:`~optimi_lab.intelligent_algorithm.
decomposition` and :mod:`~optimi_lab.intelligent_algorithm.swarm` loops run. :func:`nsga2` and
:func:`nsga3` differ only in the fill of the front that overflows, :func:`mode` is the same
generation with a differential variation and no mating pool. The operators live in
:mod:`~optimi_lab.intelligent_algorithm.variation` and
:mod:`~optimi_lab.intelligent_algorithm.selection`, and any of them can be passed in place of a
default.
"""

from optimi_lab.intelligent_algorithm.algorithms import mode, moead, mopso, nsga2, nsga3
from optimi_lab.intelligent_algorithm.decomposition import DecompositionProposer
from optimi_lab.intelligent_algorithm.proposers import EvolutionaryProposer, PopulationProposer
from optimi_lab.intelligent_algorithm.selection import (
    crowding_distance,
    crowding_select,
    decomposition_weights,
    nearest_neighbors,
    random_select,
    reference_select,
    tchebycheff,
    tournament_select,
)
from optimi_lab.intelligent_algorithm.swarm import SwarmProposer
from optimi_lab.intelligent_algorithm.variation import (
    binomial_crossover,
    differential_mutation,
    polynomial_mutation,
    pso_position,
    pso_velocity,
    simulated_binary_crossover,
)

__all__ = [
    'DecompositionProposer',
    'EvolutionaryProposer',
    'PopulationProposer',
    'SwarmProposer',
    'binomial_crossover',
    'crowding_distance',
    'crowding_select',
    'decomposition_weights',
    'differential_mutation',
    'mode',
    'moead',
    'mopso',
    'nearest_neighbors',
    'nsga2',
    'nsga3',
    'polynomial_mutation',
    'pso_position',
    'pso_velocity',
    'random_select',
    'reference_select',
    'simulated_binary_crossover',
    'tchebycheff',
    'tournament_select',
]
