"""L7 solver: the protocol every solver implements, the registry that matches one to a problem, and the
gradient methods (P9).

The population solvers of P12 are registered here. A gradient method is a pure function over
``evaluate(x)`` (:mod:`optimi_lab.solver.smooth`), driven by the run loop through :class:`GradientSolver`.
"""

from optimi_lab.solver.bounded import lbfgsb, newton_cg, projected_gradient
from optimi_lab.solver.constrained import augmented_lagrangian, interior_point, sqp
from optimi_lab.solver.gradient import METHODS, GradientSolver
from optimi_lab.solver.moving_asymptotes import mma
from optimi_lab.solver.multiobjective import mgda, pareto_continuation
from optimi_lab.solver.protocol import Solver
from optimi_lab.solver.requires import (
    SOLVERS,
    Eligible,
    Ineligible,
    Match,
    MultiObjective,
    NeedsTransform,
    Requires,
    match,
)
from optimi_lab.solver.smooth import Evaluate, Halt, SmoothView, Solution, Unevaluable

__all__ = [
    'METHODS',
    'SOLVERS',
    'Eligible',
    'Evaluate',
    'GradientSolver',
    'Halt',
    'Ineligible',
    'Match',
    'MultiObjective',
    'NeedsTransform',
    'Requires',
    'SmoothView',
    'Solution',
    'Solver',
    'Unevaluable',
    'augmented_lagrangian',
    'interior_point',
    'lbfgsb',
    'match',
    'mgda',
    'mma',
    'newton_cg',
    'pareto_continuation',
    'projected_gradient',
    'sqp',
]
