"""L2 oracle: what an evaluator can tell, what it is asked, what it answers, and how it is called."""

from optimi_lab.oracle.action import Action
from optimi_lab.oracle.capability import Capability, Information
from optimi_lab.oracle.oracle import Infeasible, Invalid, NotConverged, Oracle, Unavailable, from_function
from optimi_lab.oracle.result import Result
from optimi_lab.oracle.transforms import (
    Penalty,
    Scheme,
    check_gradient,
    embed,
    finite_difference,
    with_finite_differences,
)

__all__ = [
    'Action',
    'Capability',
    'Infeasible',
    'Information',
    'Invalid',
    'NotConverged',
    'Oracle',
    'Penalty',
    'Result',
    'Scheme',
    'Unavailable',
    'check_gradient',
    'embed',
    'finite_difference',
    'from_function',
    'with_finite_differences',
]
