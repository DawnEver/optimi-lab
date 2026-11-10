"""Build a registered solver BY NAME: the library's match first, then the solver seeded by its row.

:func:`build_solver` asks :func:`~optimi_lab.solver.requires.match` whether the named solver can run the
problem against the oracle's capability and raises its reason; an admitted solver is seeded by its
:data:`~optimi_lab.solver.seeding.SEEDING` row. ``lp`` and ``qp`` are built as the
:class:`~optimi_lab.solver.structured.StructuredSolver` that reads their coefficients off the oracle; the
other algebraic entries take their own data, not an oracle, and are refused naming that (:data:`UNDRIVEN`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import numpy as np

from optimi_lab.core import SampleSpec, refuse
from optimi_lab.operator import Risk
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.bayesian import bo
from optimi_lab.solver.cmaes import cmaes
from optimi_lab.solver.decomposition import moead
from optimi_lab.solver.evolutionary import ga, mode, nsga2, nsga3, rnsga3, sms_emoa, spea2, unsga3
from optimi_lab.solver.fidelity import asha, hyperband
from optimi_lab.solver.fidelity_bo import mfbo
from optimi_lab.solver.gradient import GradientSolver
from optimi_lab.solver.mads import mads
from optimi_lab.solver.map_elites import descriptors_of, map_elites
from optimi_lab.solver.meta import ipop_cmaes, multistart, scalarisation_scan
from optimi_lab.solver.model_trust_region import model_trust_region
from optimi_lab.solver.pareto_bo import ehvi, nehvi, parego
from optimi_lab.solver.prescreened import prescreened
from optimi_lab.solver.protocol import Solver
from optimi_lab.solver.requires import Ineligible, NeedsTransform, Requires, match
from optimi_lab.solver.rvea import rvea
from optimi_lab.solver.seeding import SEEDING, Seeding, search_embedding
from optimi_lab.solver.sequential import nelder_mead
from optimi_lab.solver.structured import STRUCTURED, StructuredSolver
from optimi_lab.solver.swarm import mopso
from optimi_lab.solver.trust_region import bobyqa
from optimi_lab.solver.turbo import turbo

__all__ = ['UNDRIVEN', 'build_solver']


#: Every seeded solver that is not a gradient method -> its factory ``(problem, capability, **settings)``.
_FACTORIES: Final[Mapping[str, Callable[..., Solver]]] = MappingProxyType({
    'asha': asha,
    'bo': bo,
    'bobyqa': bobyqa,
    'cmaes': cmaes,
    'ehvi': ehvi,
    'ga': ga,
    'hyperband': hyperband,
    'ipop_cmaes': ipop_cmaes,
    'mads': mads,
    'map_elites': map_elites,
    'mfbo': mfbo,
    'mode': mode,
    'model_trust_region': model_trust_region,
    'moead': moead,
    'mopso': mopso,
    'multistart': multistart,
    'nehvi': nehvi,
    'nelder_mead': nelder_mead,
    'nsga2': nsga2,
    'nsga3': nsga3,
    'parego': parego,
    'prescreened': prescreened,
    'rnsga3': rnsga3,
    'rvea': rvea,
    'scalarisation_scan': scalarisation_scan,
    'sms_emoa': sms_emoa,
    'spea2': spea2,
    'turbo': turbo,
    'unsga3': unsga3,
})

#: Settings a solver cannot run without and no default can stand in for.
_REQUIRED: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    'map_elites': ('descriptors',),
    'multistart': ('local',),
    'rnsga3': ('aspiration',),
    'scalarisation_scan': ('local', 'weights'),
})

#: Registered solvers no oracle drives, each with why: they read their own algebraic data.
UNDRIVEN: Final[Mapping[str, str]] = MappingProxyType({
    'admm': 'takes proximal operators (prox_f, prox_g), not a Problem against an oracle; call solver.admm',
    'benders': 'takes a master LinearProgram and its recourse data, not an oracle; call solver.benders',
    'cp': 'enumerates finite domains under predicate rules (domains=, rules=), not an oracle; call solver.cp',
    'dynamic_programming': 'takes stages, states and transitions, not an oracle; call solver.dynamic_programming',
    'milp': 'takes a LinearProgram and its integrality, not an oracle; call solver.milp',
})

#: Declarations the builder owns; a caller's settings may not restate them.
_BUILDER_OWNED: Final = frozenset({
    'problem',
    'capability',
    'spec',
    'initial',
    'seed',
    'risk',
    'scenarios',
    'embedding',
    'population',
    'start',
    'starts',
    'max_evaluations',
    'generations',
})


def build_solver(
    name: str,
    problem: Problem,
    capability: Capability,
    *,
    seed: int,
    population: int,
    initial: np.ndarray | None,
    generations: int,
    settings: Mapping[str, object] | None = None,
    risk: Risk | None = None,
    scenarios: int | None = None,
) -> Solver:
    """The registered solver *name* over *problem*, once :func:`~optimi_lab.solver.requires.match` admits the two.

    *initial* is unit-cube rows: a ``POPULATION`` solver takes them all (None draws *population* by LHS), a
    solver drawing its own first batch takes none, a ``START``/``SLOPES`` one (and ``lp``/``qp``) one row, its
    start. *generations* is the search's length: ``rvea`` adapts over it, and a sequential solver's evaluation
    budget is ``population * (generations + 1)``. *risk* and *scenarios* are the robust declaration, handed
    on as declared; a conditional space is read through its embedding (:func:`search_embedding`).
    """
    if name in UNDRIVEN:
        refuse(f'{name} is not driven through an oracle: it {UNDRIVEN[name]}')
    extra = dict(settings or {})
    if overlap := set(extra) & _BUILDER_OWNED:
        refuse(f'settings repeat builder-owned declarations {sorted(overlap)}')
    verdict = match(Requires.of(name), problem, capability)
    if isinstance(verdict, Ineligible):
        refuse(f'{name} cannot run this problem: {verdict.reason}')
    if isinstance(verdict, NeedsTransform) and set(verdict.transforms) - {'embed'}:
        refuse(f'{name} runs this problem only after {list(verdict.transforms)}, which the builder does not apply')
    if name in STRUCTURED:
        return StructuredSolver(problem, capability, name, **_design_start(name, problem, initial), **extra)
    if missing := [key for key in _REQUIRED.get(name, ()) if key not in extra]:
        refuse(f'{name} needs settings.{missing[0]}')
    if 'aspiration' in extra:
        extra['aspiration'] = np.asarray(extra['aspiration'], dtype=float)
    if 'descriptors' in extra:
        extra['descriptors'] = descriptors_of(problem, extra['descriptors'])
    if name == 'rvea':
        extra['generations'] = generations
    robust = {'risk': risk, 'scenarios': scenarios, 'embedding': search_embedding(problem)}
    extra |= {key: value for key, value in robust.items() if value is not None}
    seeding = SEEDING[name]
    if seeding is Seeding.SLOPES:
        return GradientSolver(problem, capability, name, **_design_start(name, problem, initial), **extra)
    seeded = _seeded(name, seeding, extra, seed=seed, population=population, generations=generations, initial=initial)
    return _FACTORIES[name](problem, capability, seed=seed, **seeded, **extra)


def _seeded(
    name: str,
    seeding: Seeding,
    extra: dict[str, object],
    *,
    seed: int,
    population: int,
    generations: int,
    initial: np.ndarray | None,
) -> dict[str, object]:
    """The first-batch and budget settings *seeding* hands the factory (*extra* loses what they consume)."""
    evaluations = population * (generations + 1)
    if seeding is Seeding.START:
        return {'max_evaluations': evaluations, **({} if initial is None else {'start': _one_row(name, initial)})}
    if seeding.draws_own and initial is not None:
        refuse(f'{name} draws its own first batch ({seeding.value}); it was handed {len(initial)} row(s) to ignore')
    if seeding is Seeding.STARTS:
        return {'local': _local(name, extra.pop('local')), 'starts': population, 'max_evaluations': generations + 1}
    if seeding is Seeding.SCAN:
        weights = np.atleast_2d(np.asarray(extra.pop('weights'), dtype=float))
        share = max(1, evaluations // len(weights))
        return {'local': _local(name, extra.pop('local')), 'weights': weights, 'max_evaluations': share}
    own = {Seeding.LEVELS: {}, Seeding.MODEL: {'initial': population}, Seeding.SELF: {'population': population}}
    if seeding in own:
        return own[seeding]
    return {'spec': SampleSpec(n_samples=population, seed=seed) if initial is None else None, 'initial': initial}


def _one_row(name: str, initial: np.ndarray) -> np.ndarray:
    """The one unit-cube row a one-start solver walks from."""
    if len(initial) != 1:
        refuse(f'{name} walks from ONE start point; it was handed {len(initial)} row(s)')
    return np.asarray(initial[0], dtype=float)


def _design_start(name: str, problem: Problem, initial: np.ndarray | None) -> dict[str, object]:
    """A design-reading one-start solver's ``start=``: the one row decoded, else none (the cube's centre)."""
    return {} if initial is None else {'start': problem.space.decode(_one_row(name, initial))}


def _local(name: str, local: object) -> Callable[..., Solver]:
    """The local search ``settings.local`` names: a ``START`` row of :data:`SEEDING`."""
    starts = sorted(key for key, seeding in SEEDING.items() if seeding is Seeding.START)
    if not isinstance(local, str) or local not in starts:
        refuse(f'{name} chains a local search: settings.local must name one of {starts}, got {local!r}')
    return _FACTORIES[local]
