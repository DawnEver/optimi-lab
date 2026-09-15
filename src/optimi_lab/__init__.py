"""optimi_lab — multi-objective optimization and surrogate models, as pure numerics.

The package answers three questions and nothing else:

1. **What is optimized** — :class:`VariableSet` and :class:`ObjectiveSet`, where direction and
   operating point are FIELDS. There is no ``key@oc<N>@max|min`` name grammar to parse, so no
   position in a list decides what a column means.
2. **What an evaluation produced** — :class:`Evaluation`, which carries an :class:`Outcome` per
   cell. A point that did not compute is never a number: ``OK`` is the one outcome that
   produced a value, and every other outcome has its number replaced by NaN at construction.
3. **How a run proceeds** — :func:`optimize`, an ask/tell loop over a :class:`Proposer`. The
   proposer is a Protocol, so an algorithm keeps its own constructor and its own state.

Dependencies are numpy, scipy and scikit-learn. No plotting, no logging framework, no path
resolution, no configuration file, and no import-time side effect: importing this package
reads nothing, writes nothing, and leaves the host process's globals untouched.

    >>> from optimi_lab import Evaluation, Objective, ObjectiveSet, Outcome, VariableSet, Variable
    >>> space = VariableSet([Variable('radius', 1.0, 2.0, 'mm'), Variable('turns', 10, 20)])
    >>> objectives = ObjectiveSet([Objective('torque', 'maximize', 'rated'), Objective('mass', 'minimize')])
    >>> evaluation = Evaluation(
    ...     inputs=[[1.5, 15], [1.8, 18]],
    ...     values=[[3.2, 4.0], [float('nan'), 5.1]],
    ...     outcomes=[[Outcome.OK, Outcome.OK], [Outcome.ERROR, Outcome.OK]],
    ... )
    >>> [outcome.value for outcome in evaluation.point_outcomes()]
    ['ok', 'error']
    >>> objectives.to_minimization(evaluation.values).tolist()[0]
    [-3.2, 4.0]

The subpackages ``intelligent_algorithm`` (the proposers) and ``surrogate_model`` (the fits)
are built against :mod:`optimi_lab.core.protocols` and export their own names on top of these.
:mod:`optimi_lab.benchmarks` adds what a caller needs to JUDGE a run rather than merely make one --
the ZDT problems with known Pareto fronts, the IGD and hypervolume metrics, and a one-call
:func:`~optimi_lab.benchmarks.solve`.
"""

from optimi_lab.benchmarks import Problem, hypervolume_2d, igd, solve, zdt1, zdt2
from optimi_lab.core import (
    SAMPLE_KINDS,
    Direction,
    Evaluation,
    Evaluator,
    Objective,
    ObjectiveSet,
    Outcome,
    Proposer,
    Record,
    Refusal,
    SampleKind,
    SampleSpec,
    Surrogate,
    Variable,
    VariableSet,
    dominates,
    non_dominated_sorting,
    optimize,
    partition_count,
    sample,
)

__all__ = [
    'SAMPLE_KINDS',
    'Direction',
    'Evaluation',
    'Evaluator',
    'Objective',
    'ObjectiveSet',
    'Outcome',
    'Problem',
    'Proposer',
    'Record',
    'Refusal',
    'SampleKind',
    'SampleSpec',
    'Surrogate',
    'Variable',
    'VariableSet',
    'dominates',
    'hypervolume_2d',
    'igd',
    'non_dominated_sorting',
    'optimize',
    'partition_count',
    'sample',
    'solve',
    'zdt1',
    'zdt2',
]
