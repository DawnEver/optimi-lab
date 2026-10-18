"""L5 operator: the pieces a solver composes.

* population -- variation, discrete moves, repair, selection, reference directions, indicators,
  scalarisation, stratified sampling: each a function of the arrays it acts on and of an explicit
  generator, reading no module state;
* stochastic -- risk measures, estimators with standard errors, MC/QMC robust evaluation,
  reliability (crude MC, importance sampling, subset simulation, FORM/SORM, surrogate), multilevel
  Monte Carlo and OCBA replicate allocation;
* gradient -- the generalised Cauchy point, QPs, KKT assembly and merit functions. Every linear solve, line search, trust-region
  step and quasi-Newton update is numerics-lab's; this layer assembles what they are applied to.

Solvers (L7, :mod:`optimi_lab.solver`) bind them.
"""

from optimi_lab.operator.allocation import ocba
from optimi_lab.operator.cauchy import cauchy_point
from optimi_lab.operator.discrete import (
    Block,
    Lattice,
    discrete_mutation,
    inversion_mutation,
    lattice,
    order_crossover,
    snap,
    swap_mutation,
)
from optimi_lab.operator.estimators import (
    Estimate,
    joint_probability,
    mean_estimate,
    probability,
    replicated,
    satisfied,
)
from optimi_lab.operator.form import CURVATURE_STEP, STEP, DesignPoint, SecondOrder, form, sorm
from optimi_lab.operator.indicator import hypervolume_select, spea2_fitness, spea2_select
from optimi_lab.operator.kkt import KKTResidual, Linearization, damped_secant, kkt_residual, lagrangian_gradient
from optimi_lab.operator.merit import augmented_lagrangian_merit, l1_merit, l1_penalty, l1_slope
from optimi_lab.operator.multilevel import Multilevel, mlmc
from optimi_lab.operator.qp import QPSolution, active_set_qp, bound_qp
from optimi_lab.operator.reference import (
    adapt_directions,
    angle_penalized_select,
    aspiration_directions,
    associate,
    das_dennis,
    niching_tournament,
    normalise,
    reference_select,
)
from optimi_lab.operator.reliability import (
    LimitState,
    crude_monte_carlo,
    importance_sampling,
    in_standard_space,
    subset_simulation,
    surrogate_reliability,
)
from optimi_lab.operator.repair import Repair, repair
from optimi_lab.operator.risk import cvar, expectation, mean_plus_std, value_at_risk, worst_case
from optimi_lab.operator.robust import Response, Rule, draw, robust_mean
from optimi_lab.operator.sampling import stratified
from optimi_lab.operator.scalarisation import (
    SCALARISATIONS,
    Reach,
    Scalarisation,
    asf,
    decomposition_weights,
    epsilon_constraint,
    nbi,
    nearest_neighbors,
    pbi,
    tchebycheff,
    weighted_sum,
)
from optimi_lab.operator.selection import (
    crowding_distance,
    crowding_select,
    fronts_then_fill,
    random_select,
    require_rankable,
    require_selectable,
    tournament_select,
    violation_of,
)
from optimi_lab.operator.variation import (
    DEStrategy,
    binomial_crossover,
    differential_mutation,
    exponential_crossover,
    polynomial_mutation,
    pso_position,
    pso_velocity,
    simulated_binary_crossover,
)

__all__ = [
    'CURVATURE_STEP',
    'SCALARISATIONS',
    'STEP',
    'Block',
    'DEStrategy',
    'DesignPoint',
    'Estimate',
    'KKTResidual',
    'Lattice',
    'LimitState',
    'Linearization',
    'Multilevel',
    'QPSolution',
    'Reach',
    'Repair',
    'Response',
    'Rule',
    'Scalarisation',
    'SecondOrder',
    'active_set_qp',
    'adapt_directions',
    'angle_penalized_select',
    'asf',
    'aspiration_directions',
    'associate',
    'augmented_lagrangian_merit',
    'binomial_crossover',
    'bound_qp',
    'cauchy_point',
    'crowding_distance',
    'crowding_select',
    'crude_monte_carlo',
    'cvar',
    'damped_secant',
    'das_dennis',
    'decomposition_weights',
    'differential_mutation',
    'discrete_mutation',
    'draw',
    'epsilon_constraint',
    'expectation',
    'exponential_crossover',
    'form',
    'fronts_then_fill',
    'hypervolume_select',
    'importance_sampling',
    'in_standard_space',
    'inversion_mutation',
    'joint_probability',
    'kkt_residual',
    'l1_merit',
    'l1_penalty',
    'l1_slope',
    'lagrangian_gradient',
    'lattice',
    'mean_estimate',
    'mean_plus_std',
    'mlmc',
    'nbi',
    'nearest_neighbors',
    'niching_tournament',
    'normalise',
    'ocba',
    'order_crossover',
    'pbi',
    'polynomial_mutation',
    'probability',
    'pso_position',
    'pso_velocity',
    'random_select',
    'reference_select',
    'repair',
    'replicated',
    'require_rankable',
    'require_selectable',
    'robust_mean',
    'satisfied',
    'simulated_binary_crossover',
    'snap',
    'sorm',
    'spea2_fitness',
    'spea2_select',
    'stratified',
    'subset_simulation',
    'surrogate_reliability',
    'swap_mutation',
    'tchebycheff',
    'tournament_select',
    'value_at_risk',
    'violation_of',
    'weighted_sum',
    'worst_case',
]
