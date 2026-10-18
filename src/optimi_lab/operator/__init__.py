"""L5 operator: the pieces a solver composes -- variation, repair, selection, scalarisation, stratified sampling.

Every operator is a function of the arrays it acts on (unit-cube points, minimised normalised
objective rows, one total violation per row) and of an explicit generator; none reads module state
or knows which solver calls it. Solvers (L7, :mod:`optimi_lab.solver`) bind them.
"""

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
from optimi_lab.operator.indicator import hypervolume_select, spea2_fitness, spea2_select
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
from optimi_lab.operator.repair import Repair, repair
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
    'SCALARISATIONS',
    'Block',
    'DEStrategy',
    'Lattice',
    'Reach',
    'Repair',
    'Scalarisation',
    'adapt_directions',
    'angle_penalized_select',
    'asf',
    'aspiration_directions',
    'associate',
    'binomial_crossover',
    'crowding_distance',
    'crowding_select',
    'das_dennis',
    'decomposition_weights',
    'differential_mutation',
    'discrete_mutation',
    'epsilon_constraint',
    'exponential_crossover',
    'fronts_then_fill',
    'hypervolume_select',
    'inversion_mutation',
    'lattice',
    'nbi',
    'nearest_neighbors',
    'niching_tournament',
    'normalise',
    'order_crossover',
    'pbi',
    'polynomial_mutation',
    'pso_position',
    'pso_velocity',
    'random_select',
    'reference_select',
    'repair',
    'require_rankable',
    'require_selectable',
    'simulated_binary_crossover',
    'snap',
    'spea2_fitness',
    'spea2_select',
    'stratified',
    'swap_mutation',
    'tchebycheff',
    'tournament_select',
    'violation_of',
    'weighted_sum',
]
