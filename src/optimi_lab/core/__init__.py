"""The core: what every layer stands on -- refusals, dominance, and the unit-cube sampler.

Pure numerics, no import-time side effect. Problems, results and runs live in the II.7 layers above
(:mod:`optimi_lab.problem`, :mod:`optimi_lab.oracle`, :mod:`optimi_lab.archive`,
:mod:`optimi_lab.run`); the 0.3 evaluation, record and driver this package held are retired and
named in ``tests/architecture/test_a_retired_spelling_stays_retired.py``.
"""

from optimi_lab.core.errors import (
    Refusal,
    coerce_enum,
    refuse,
    require_axis,
    require_finite,
    require_positive,
)
from optimi_lab.core.pareto import (
    constrained_dominates,
    constrained_non_dominated_sorting,
    dominates,
    epsilon_dominates,
    incremental_sort,
    non_dominated_sorting,
    partition_count,
    tolerance_dominates,
    total_violation,
)
from optimi_lab.core.sampling import SAMPLE_KINDS, SampleKind, SampleSpec, sample

__all__ = [
    'SAMPLE_KINDS',
    'Refusal',
    'SampleKind',
    'SampleSpec',
    'coerce_enum',
    'constrained_dominates',
    'constrained_non_dominated_sorting',
    'dominates',
    'epsilon_dominates',
    'incremental_sort',
    'non_dominated_sorting',
    'partition_count',
    'refuse',
    'require_axis',
    'require_finite',
    'require_positive',
    'sample',
    'tolerance_dominates',
    'total_violation',
]
