"""L5 operator: the building blocks solvers compose -- QPs, KKT assembly and merit functions.

Every linear solve, line search, trust-region step and quasi-Newton update is numerics-lab's; this
layer assembles what those primitives are applied to.
"""

from optimi_lab.operator.kkt import KKTResidual, Linearization, damped_secant, kkt_residual, lagrangian_gradient
from optimi_lab.operator.merit import augmented_lagrangian_merit, l1_merit, l1_penalty, l1_slope
from optimi_lab.operator.qp import QPSolution, active_set_qp, bound_qp

__all__ = [
    'KKTResidual',
    'Linearization',
    'QPSolution',
    'active_set_qp',
    'augmented_lagrangian_merit',
    'bound_qp',
    'damped_secant',
    'kkt_residual',
    'l1_merit',
    'l1_penalty',
    'l1_slope',
    'lagrangian_gradient',
]
