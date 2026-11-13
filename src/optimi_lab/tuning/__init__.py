"""L10 tuning: hyper-parameter SEARCH over a named space -- the caller supplies only the objective.

:func:`minimise` runs one :data:`TUNING_METHODS` entry over a ``Mapping[str, Dimension]`` of learn-lab's
:mod:`learn_lab.space` dimensions (``Real``, ``Integer``, ``Choice``) and returns a :class:`Tuning` of every
trial and the best. Detail: :mod:`optimi_lab.tuning.search`.
"""

from optimi_lab.tuning.search import TUNING_METHODS, Method, Trial, Tuning, minimise
from optimi_lab.tuning.space import Kind, kind_of, problem_space

__all__ = [
    'TUNING_METHODS',
    'Kind',
    'Method',
    'Trial',
    'Tuning',
    'kind_of',
    'minimise',
    'problem_space',
]
