"""L7 solver: the protocol every solver implements, the registry that matches one to a problem, and the solvers.

Every factory takes ``(problem, capability, **settings)`` and returns a :class:`Solver` for
:func:`optimi_lab.run.run`; a setting it does not take is refused naming the ones it does.

* population, Pareto -- :func:`nsga2`, :func:`nsga3`, :func:`unsga3`, :func:`rnsga3`, :func:`spea2`,
  :func:`sms_emoa`, :func:`rvea`, :func:`mode`, :func:`moead`, :func:`mopso`;
* population, one score -- :func:`ga`, :func:`cmaes`, :func:`map_elites`;
* local derivative-free -- :func:`nelder_mead`, :func:`mads`, :func:`bobyqa`;
* compositions -- :class:`Chain`, :func:`scalarisation_scan`, :func:`multistart`, :func:`restarts`,
  :func:`ipop_cmaes`, :func:`warm_start`; multi-fidelity -- :class:`SuccessiveHalving`,
  :func:`hyperband`, :class:`ASHA`.

Operators live one layer down, in :mod:`optimi_lab.operator`.
"""

from optimi_lab.solver.cmaes import CMAES, cmaes
from optimi_lab.solver.decomposition import DecompositionSolver, moead
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
from optimi_lab.solver.mads import MADS, mads
from optimi_lab.solver.map_elites import Descriptor, MapElites, map_elites
from optimi_lab.solver.meta import Chain, ipop_cmaes, multistart, restarts, scalarisation_scan, warm_start
from optimi_lab.solver.population import EvolutionarySolver, PopulationSolver, Scalarise
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
from optimi_lab.solver.rvea import ReferenceVectorSolver, rvea
from optimi_lab.solver.sequential import NelderMead, SequentialSolver, nelder_mead
from optimi_lab.solver.swarm import SwarmSolver, mopso
from optimi_lab.solver.trust_region import TrustRegion, bobyqa

__all__ = [
    'ASHA',
    'CMAES',
    'MADS',
    'SOLVERS',
    'Chain',
    'DecompositionSolver',
    'Descriptor',
    'Eligible',
    'EvolutionarySolver',
    'Ineligible',
    'MapElites',
    'Match',
    'MultiObjective',
    'NeedsTransform',
    'NelderMead',
    'PopulationSolver',
    'ReferenceVectorSolver',
    'Requires',
    'Scalarise',
    'SequentialSolver',
    'Solver',
    'SuccessiveHalving',
    'SwarmSolver',
    'TrustRegion',
    'asha',
    'bobyqa',
    'cmaes',
    'differential',
    'ga',
    'hyperband',
    'ipop_cmaes',
    'mads',
    'map_elites',
    'match',
    'mode',
    'moead',
    'mopso',
    'multistart',
    'nelder_mead',
    'nsga2',
    'nsga3',
    'restarts',
    'rnsga3',
    'rvea',
    'sbx_polynomial',
    'scalarisation_scan',
    'sms_emoa',
    'spea2',
    'unsga3',
    'warm_start',
]
