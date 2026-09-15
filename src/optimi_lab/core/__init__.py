"""The core: what a run is described by, what it produced, and the loop between them.

Pure numerics. numpy, scipy and scikit-learn are the only dependencies, no module here mutates
process-global state when imported, and nothing here reads or writes a file, resolves a path, or
attaches a log handler. The removed ``optimi_lab.utils.visualization`` set three matplotlib
rcParams at import time, and a bare ``import optimi_lab.utils.<anything>`` used to create a
per-run output directory and rebind the process's active logger: a library that is merely
imported must not do that to its host.

The modules are grouped by the question each answers — :mod:`~optimi_lab.core.space` (what is
optimized), :mod:`~optimi_lab.core.sampling` (the first batch), :mod:`~optimi_lab.core.outcomes`
(what an evaluation produced), :mod:`~optimi_lab.core.pareto` (dominance),
:mod:`~optimi_lab.core.protocols` (the interfaces) and :mod:`~optimi_lab.core.optimize` (the
driver and its record).
"""

from optimi_lab.core.errors import (
    Refusal,
    coerce_enum,
    refuse,
    require_axis,
    require_finite,
    require_positive,
)
from optimi_lab.core.optimize import Record, optimize
from optimi_lab.core.outcomes import Evaluation, Outcome
from optimi_lab.core.pareto import dominates, non_dominated_sorting, partition_count
from optimi_lab.core.protocols import Evaluator, Proposer, Surrogate
from optimi_lab.core.sampling import SAMPLE_KINDS, SampleKind, SampleSpec, sample
from optimi_lab.core.space import Direction, Objective, ObjectiveSet, Variable, VariableSet

# `__all__` is written out rather than aliased from the submodules: the guard helpers are
# reachable as `optimi_lab.core.refuse` for a package that wants one refusal shape, but the
# surface a user reads is the one below.
__all__ = [
    'SAMPLE_KINDS',
    'Direction',
    'Evaluation',
    'Evaluator',
    'Objective',
    'ObjectiveSet',
    'Outcome',
    'Proposer',
    'Record',
    'Refusal',
    'SampleKind',
    'SampleSpec',
    'Surrogate',
    'Variable',
    'VariableSet',
    'coerce_enum',
    'dominates',
    'non_dominated_sorting',
    'optimize',
    'partition_count',
    'refuse',
    'require_axis',
    'require_finite',
    'require_positive',
    'sample',
]
