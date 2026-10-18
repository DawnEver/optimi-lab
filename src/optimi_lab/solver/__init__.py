"""L7 solver: the protocol every solver implements, the registry that matches one to a problem, and the solvers.

Every factory takes ``(problem, capability, **settings)`` and returns a :class:`Solver` for
:func:`optimi_lab.run.run`; a setting it does not take is refused naming the ones it does.

* population, Pareto -- :func:`nsga2`, :func:`nsga3`, :func:`unsga3`, :func:`rnsga3`, :func:`spea2`,
  :func:`sms_emoa`, :func:`rvea`, :func:`mode`, :func:`moead`, :func:`mopso`;
* population, one score -- :func:`ga`, :func:`cmaes`, :func:`map_elites`;
* gradient (P9) -- :func:`lbfgsb`, :func:`projected_gradient`, :func:`newton_cg`, :func:`sqp`,
  :func:`interior_point`, :func:`augmented_lagrangian`, :func:`mma`, :func:`mgda`,
  :func:`pareto_continuation`: pure functions over ``evaluate(x)`` (:mod:`optimi_lab.solver.smooth`),
  driven by :class:`GradientSolver`;
* local derivative-free -- :func:`nelder_mead`, :func:`mads`, :func:`bobyqa`;
* gradient (P9) -- pure functions over ``evaluate(x)`` (:mod:`optimi_lab.solver.smooth`) driven by
  :class:`GradientSolver`: :func:`projected_gradient`, :func:`lbfgsb`, :func:`newton_cg`, :func:`sqp`,
  :func:`interior_point`, :func:`augmented_lagrangian`, :func:`mma`, :func:`mgda`,
  :func:`pareto_continuation`;
* model-based (P11) -- :func:`bo` (log-EI/EI/PI/UCB/KG, constrained EI), :func:`parego`,
  :func:`ehvi`, :func:`nehvi`, each batching and running asynchronously through Kriging-believer
  fantasies (:mod:`optimi_lab.solver.model_based`); :func:`mfbo` (augmented EI on AR1 co-Kriging),
  :func:`turbo` (TuRBO-1), :func:`prescreened` (surrogate-prescreened NSGA-II/GA),
  :func:`model_trust_region` (trust-region model management);
* structured (P14) -- :func:`lp` (revised simplex, Mehrotra interior point), :func:`qp` (a route
  onto P9's QPs), :func:`milp` (branch and bound with Gomory cuts) over a :class:`LinearProgram`,
  each reporting a :class:`StructuredSolution`; :func:`dynamic_programming` (backward induction)
  and :func:`cp` (finite-domain search with forward checking); :func:`benders` (two-stage
  decomposition) and :func:`admm` (consensus ADMM over proximal operators);
* compositions -- :class:`Chain`, :func:`scalarisation_scan`, :func:`multistart`, :func:`restarts`,
  :func:`ipop_cmaes`, :func:`warm_start`; multi-fidelity -- :class:`SuccessiveHalving`,
  :func:`hyperband`, :class:`ASHA`.

Operators live one layer down, in :mod:`optimi_lab.operator`.
"""

from optimi_lab.solver.admm import Prox, admm, box_prox, l1_prox, quadratic_prox
from optimi_lab.solver.bayesian import ACQUISITIONS, BayesianOptimisation, bo
from optimi_lab.solver.benders import benders
from optimi_lab.solver.bounded import lbfgsb, newton_cg, projected_gradient
from optimi_lab.solver.branch_and_bound import milp
from optimi_lab.solver.cmaes import CMAES, cmaes
from optimi_lab.solver.constrained import augmented_lagrangian, interior_point, sqp
from optimi_lab.solver.constraint_programming import CPSolution, Predicate, all_different, cp
from optimi_lab.solver.decomposition import DecompositionSolver, moead
from optimi_lab.solver.dynamic_programming import DPSolution, dynamic_programming
from optimi_lab.solver.evolutionary import (
    differential,
    ga,
    mode,
    nsga2,
    nsga3,
    rnsga3,
    sbx_polynomial,
    sms_emoa,
    spea2,
    unsga3,
)
from optimi_lab.solver.fidelity import ASHA, SuccessiveHalving, asha, hyperband
from optimi_lab.solver.fidelity_bo import MultiFidelityBO, mfbo
from optimi_lab.solver.gradient import METHODS, GradientSolver
from optimi_lab.solver.linear import LP_METHODS, lp
from optimi_lab.solver.linear_program import LinearProgram, StructuredSolution
from optimi_lab.solver.mads import MADS, mads
from optimi_lab.solver.map_elites import Descriptor, MapElites, map_elites
from optimi_lab.solver.meta import Chain, ipop_cmaes, multistart, restarts, scalarisation_scan, warm_start
from optimi_lab.solver.model_based import ModelBased, maximise
from optimi_lab.solver.model_trust_region import ModelTrustRegion, model_trust_region
from optimi_lab.solver.moving_asymptotes import gcmma, mma
from optimi_lab.solver.multiobjective import mgda, pareto_continuation
from optimi_lab.solver.pareto_bo import HypervolumeImprovement, ParEGO, ehvi, nehvi, parego
from optimi_lab.solver.population import EvolutionarySolver, PopulationSolver, Scalarise
from optimi_lab.solver.prescreened import Prescreened, prescreened
from optimi_lab.solver.protocol import Solver
from optimi_lab.solver.quadratic import qp
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
from optimi_lab.solver.rvea import ReferenceVectorSolver, rvea
from optimi_lab.solver.sequential import NelderMead, SequentialSolver, nelder_mead
from optimi_lab.solver.smooth import ATOL, RTOL, Evaluate, Halt, SmoothView, Solution, Unevaluable, kkt_bar
from optimi_lab.solver.stochastic import SampleAverage, saa, sample_average
from optimi_lab.solver.swarm import SwarmSolver, mopso
from optimi_lab.solver.trust_region import TrustRegion, bobyqa
from optimi_lab.solver.turbo import TuRBO, turbo

__all__ = [
    'ACQUISITIONS',
    'ASHA',
    'ATOL',
    'CMAES',
    'LP_METHODS',
    'MADS',
    'METHODS',
    'RTOL',
    'SOLVERS',
    'BayesianOptimisation',
    'CPSolution',
    'Chain',
    'DPSolution',
    'DecompositionSolver',
    'Descriptor',
    'Eligible',
    'Evaluate',
    'EvolutionarySolver',
    'GradientSolver',
    'Halt',
    'HypervolumeImprovement',
    'Ineligible',
    'LinearProgram',
    'MapElites',
    'Match',
    'ModelBased',
    'ModelTrustRegion',
    'MultiFidelityBO',
    'MultiObjective',
    'NeedsTransform',
    'NelderMead',
    'ParEGO',
    'PopulationSolver',
    'Predicate',
    'Prescreened',
    'Prox',
    'ReferenceVectorSolver',
    'Requires',
    'SampleAverage',
    'Scalarise',
    'SequentialSolver',
    'SmoothView',
    'Solution',
    'Solver',
    'StructuredSolution',
    'SuccessiveHalving',
    'SwarmSolver',
    'TrustRegion',
    'TuRBO',
    'Unevaluable',
    'admm',
    'all_different',
    'asha',
    'augmented_lagrangian',
    'benders',
    'bo',
    'bobyqa',
    'box_prox',
    'cmaes',
    'cp',
    'differential',
    'dynamic_programming',
    'ehvi',
    'ga',
    'gcmma',
    'hyperband',
    'interior_point',
    'ipop_cmaes',
    'kkt_bar',
    'l1_prox',
    'lbfgsb',
    'lp',
    'mads',
    'map_elites',
    'match',
    'maximise',
    'mfbo',
    'mgda',
    'milp',
    'mma',
    'mode',
    'model_trust_region',
    'moead',
    'mopso',
    'multistart',
    'nehvi',
    'nelder_mead',
    'newton_cg',
    'nsga2',
    'nsga3',
    'parego',
    'pareto_continuation',
    'prescreened',
    'projected_gradient',
    'qp',
    'quadratic_prox',
    'restarts',
    'rnsga3',
    'rvea',
    'saa',
    'sample_average',
    'sbx_polynomial',
    'scalarisation_scan',
    'sms_emoa',
    'spea2',
    'sqp',
    'turbo',
    'unsga3',
    'warm_start',
]
