"""L7 solver: the protocol every solver implements, and the registry that matches one to a problem.

The solvers themselves are rebuilt with P9-P15; the population solvers of P12 are registered here.
"""

from optimi_lab.solver.protocol import Solver
from optimi_lab.solver.requires import (
    REGISTRY,
    Eligible,
    Ineligible,
    Match,
    MultiObjective,
    NeedsTransform,
    Requires,
    match,
)

__all__ = [
    'REGISTRY',
    'Eligible',
    'Ineligible',
    'Match',
    'MultiObjective',
    'NeedsTransform',
    'Requires',
    'Solver',
    'match',
]
