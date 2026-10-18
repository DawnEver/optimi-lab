"""The population solvers: five published algorithms, three loops, one kit of operators.

Every solver here implements :class:`optimi_lab.solver.Solver` and is driven by
:func:`optimi_lab.run.run`, which files every result in an ``outcomes.Record``:

    >>> import tempfile
    >>> from pathlib import Path
    >>> from lab_commons.outcomes import Record
    >>> from optimi_lab.core import SampleSpec
    >>> from optimi_lab.intelligent_algorithm import nsga2
    >>> from optimi_lab.oracle import Capability, from_function
    >>> from optimi_lab.problem import Objective, Problem, Real, Space
    >>> from optimi_lab.run import Budget, run
    >>> problem = Problem(Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))), (Objective('f'), Objective('g')))
    >>> def bowls(action):
    ...     x, y = action.design['x'], action.design['y']
    ...     return {'f': x**2 + y**2, 'g': (x - 1) ** 2 + (y - 1) ** 2}
    >>> oracle = from_function(bowls, Capability(outputs=('f', 'g')))
    >>> solver = nsga2(problem, oracle.capability, spec=SampleSpec(n_samples=8, seed=1))
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     outcome = run(problem, oracle, solver, Record(Path(tmp) / 'run.jsonl'), Budget(evaluations=24))
    >>> outcome.ledger.evaluations, len(outcome.archive.front()) > 0
    (24, True)

The five algorithms are parameterizations of the three loops:
:mod:`~optimi_lab.intelligent_algorithm.algorithms` binds the operators the
:mod:`~optimi_lab.intelligent_algorithm.solvers`, :mod:`~optimi_lab.intelligent_algorithm.
decomposition` and :mod:`~optimi_lab.intelligent_algorithm.swarm` loops run. :func:`nsga2` and
:func:`nsga3` differ only in the fill of the front that overflows, :func:`mode` is the same
generation with a differential variation and no mating pool. The operators live in
:mod:`~optimi_lab.intelligent_algorithm.variation` and
:mod:`~optimi_lab.intelligent_algorithm.selection`, and any of them can be passed in place of a
default.
"""

from optimi_lab.intelligent_algorithm.algorithms import mode, moead, mopso, nsga2, nsga3
from optimi_lab.intelligent_algorithm.decomposition import DecompositionSolver
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
from optimi_lab.intelligent_algorithm.solvers import EvolutionarySolver, PopulationSolver
from optimi_lab.intelligent_algorithm.swarm import SwarmSolver
from optimi_lab.intelligent_algorithm.variation import (
    binomial_crossover,
    differential_mutation,
    polynomial_mutation,
    pso_position,
    pso_velocity,
    simulated_binary_crossover,
)

__all__ = [
    'DecompositionSolver',
    'EvolutionarySolver',
    'PopulationSolver',
    'SwarmSolver',
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
