"""optimi_lab -- a self-implemented decision library, layered as plan II.7 lays it out.

    problem (L1) -> oracle (L2) -> archive (L3) -> metric (L4) -> ... -> solver (L7) -> run (L9)

1. **What is decided** -- :mod:`optimi_lab.problem`: a :class:`Space` of Real, Integer, Categorical
   and Field variables, objectives with a fixed normalisation, five constraint kinds, and an
   ``outcomes.Identity`` per problem and per design.
2. **What an evaluator answers** -- :mod:`optimi_lab.oracle`: a :class:`Result` is a
   ``lab_commons.outcomes.Cell`` (status, values, reason, evidence, cost) plus derivatives. A point
   that did not compute carries a status and a reason, never a number.
3. **What was learned** -- :mod:`optimi_lab.archive`: views over one ``outcomes.Record``.
4. **How a run proceeds** -- :func:`run`: ask, dispatch, file, tell, until a named reason stops it;
   an interrupted run resumes by replaying its record.

The runtime is numpy, scipy, scikit-learn and ``lab_commons.outcomes``; importing this package reads
nothing, writes nothing and leaves the host process's globals untouched.
"""

from optimi_lab.archive import Archive
from optimi_lab.core import Refusal, SampleSpec
from optimi_lab.oracle import Action, Capability, Information, Oracle, Result, from_function
from optimi_lab.problem import (
    BlackBox,
    Categorical,
    Chance,
    Explicit,
    Fidelity,
    Field,
    Integer,
    Linear,
    Objective,
    Path,
    Problem,
    Real,
    Relation,
    Sense,
    Space,
    When,
)
from optimi_lab.run import Budget, Termination, run
from optimi_lab.solver import Solver, match

__all__ = [
    'Action',
    'Archive',
    'BlackBox',
    'Budget',
    'Capability',
    'Categorical',
    'Chance',
    'Explicit',
    'Fidelity',
    'Field',
    'Information',
    'Integer',
    'Linear',
    'Objective',
    'Oracle',
    'Path',
    'Problem',
    'Real',
    'Refusal',
    'Relation',
    'Result',
    'SampleSpec',
    'Sense',
    'Solver',
    'Space',
    'Termination',
    'When',
    'from_function',
    'match',
    'run',
]
