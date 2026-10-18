"""What a solver needs, and the one match that says whether a problem and an oracle supply it.

The information an evaluator offers decides what an algorithm can be used for (plan I.2.3). A
:class:`Requires` declares a solver's needs; :func:`match` answers:

* :class:`Eligible` -- usable as is;
* :class:`NeedsTransform` -- usable after the NAMED transforms, at a stated evaluation cost per
  action (finite differences multiply it); the transforms used are part of what a result reports
  (B0-20);
* :class:`Ineligible` -- not usable, with the reason and what would make it usable.

:data:`SOLVERS` names every solver this package ships, so a stranger is refused naming the set.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from optimi_lab.core.errors import refuse
from optimi_lab.oracle import Capability, Information
from optimi_lab.problem import BlackBox, Categorical, Chance, Explicit, Field, Form, Linear, Path, Problem, Real

__all__ = ['SOLVERS', 'Eligible', 'Ineligible', 'Match', 'MultiObjective', 'NeedsTransform', 'Requires', 'match']


class MultiObjective(StrEnum):
    """How a solver treats several objectives (B0-20): they are not interchangeable claims."""

    SINGLE = 'single'
    NATIVE_PARETO = 'native_pareto'
    WEIGHTED = 'weighted'
    LEXICOGRAPHIC = 'lexicographic'
    EXTERNAL_SCAN = 'external_scan'


@dataclass(frozen=True, slots=True)
class Requires:
    """A solver's needs: information, many-objective handling, metric, constraint kinds, uncertainty, forms."""

    objectives: MultiObjective
    information: frozenset[Information] = frozenset({Information.VALUES})
    metric: bool = False
    constraints: frozenset[type] = frozenset()
    uncertainty: bool = False
    forms: frozenset[Form] = frozenset()
    fidelity: bool = False

    @staticmethod
    def of(name: str) -> Requires:
        """The registered needs of solver *name*."""
        if name not in SOLVERS:
            refuse(f'no solver {name!r} is registered; the registered solvers are {sorted(SOLVERS)}')
        return SOLVERS[name]


@dataclass(frozen=True, slots=True)
class Eligible:
    """Usable as is."""


@dataclass(frozen=True, slots=True)
class NeedsTransform:
    """Usable after *transforms*, costing *evaluations_per_action* oracle calls per action."""

    transforms: tuple[str, ...]
    evaluations_per_action: float


@dataclass(frozen=True, slots=True)
class Ineligible:
    """Not usable, and why."""

    reason: str


Match = Eligible | NeedsTransform | Ineligible

_GRADIENTS: Final = frozenset({Information.JVP, Information.VJP})


def _refusals(requires: Requires, problem: Problem, capability: Capability) -> Iterator[str]:
    """Every reason the solver cannot run, in the order a reader should fix them."""
    if len(problem.objectives) > 1 and requires.objectives is MultiObjective.SINGLE:
        yield f'single-objective solver on {len(problem.objectives)} objectives; it needs a scalarisation first'
    chance = [c.name for c in problem.constraints if isinstance(c, Chance) and Chance not in requires.constraints]
    if chance:
        yield f'Chance constraints {chance} need an estimator over replicates; the solver handles none'
    if requires.fidelity and problem.fidelity is None:
        yield 'the solver promotes through fidelity levels and the problem declares no Fidelity'
    if problem.uncertainty is not None and not requires.uncertainty:
        yield 'the problem declares uncertainty and the solver has no risk estimator'
    for form in sorted(requires.forms):
        lacking = [o for o in problem.outputs if problem.structure is None or not problem.structure.has(o, form)]
        if lacking:
            yield f'the solver needs {form.name} outputs; {lacking} are not declared {form.name} in Structure'
    missing = requires.information - capability.provides
    if missing - _GRADIENTS:
        yield f'the solver needs {sorted(missing - _GRADIENTS)}, which no transform supplies from values'
    if missing and Information.STOCHASTIC in capability.provides:
        yield 'finite differences of a stochastic oracle difference its noise, not its slope'
    discrete = [v.name for v in problem.space if not isinstance(v, (Real, Field))]
    if missing and discrete:
        yield f'finite differences need continuous variables; {discrete} are discrete'


def match(requires: Requires, problem: Problem, capability: Capability) -> Match:
    """Whether a solver needing *requires* can run on *problem* against an oracle of *capability*."""
    reason = next(_refusals(requires, problem, capability), None)
    if reason is not None:
        return Ineligible(reason)
    transforms: list[str] = []
    per_action = 1.0
    if requires.information - capability.provides:
        transforms.append('finite_differences')
        per_action = 1.0 + 2.0 * problem.space.width
    if requires.metric and any(isinstance(v, Categorical) and v.embedding is None for v in problem.space):
        transforms.append('embed')
    unhandled = (BlackBox, Path, Linear, Explicit)
    if any(isinstance(c, unhandled) and type(c) not in requires.constraints for c in problem.constraints):
        transforms.append('penalise')
    return NeedsTransform(tuple(transforms), per_action) if transforms else Eligible()


_SLOPES: Final = frozenset({Information.VALUES, Information.VJP})
_SMOOTH: Final = frozenset({BlackBox, Linear, Explicit})
_PARETO: Final = Requires(MultiObjective.NATIVE_PARETO, metric=True, constraints=frozenset({BlackBox}))
_SINGLE: Final = Requires(MultiObjective.SINGLE, metric=True, constraints=frozenset({BlackBox}))
_LINEAR: Final = Requires(MultiObjective.SINGLE, constraints=frozenset({Linear}), forms=frozenset({Form.LINEAR}))

#: Every solver this package ships, by name. Population and local solvers rank by Deb's rule (or a
#: barrier on the total violation), so they take black-box constraints natively; MOEA/D decomposes
#: by Tchebycheff weights; a scan returns one point per scalarisation; the multi-fidelity schedules
#: need Fidelity levels. The model-based solvers (P11) model each BlackBox constraint by its own GP;
#: ParEGO scalarises by a fresh weight per pick (WEIGHTED), EHVI/NEHVI keep the front (NATIVE_PARETO);
#: multi-fidelity BO and TuRBO model no constraint (penalise first); the prescreened EA ranks by
#: Deb's rule on predicted rows and model management is an extreme barrier, like BOBYQA. The
#: structured solvers (P14) read algebra, not samples: each needs its outputs DECLARED in that form. The gradient methods (``optimi_lab.solver.gradient.METHODS``) need slopes;
#: the box methods take no other constraint, the constrained ones every constraint that has a gradient.
SOLVERS: Final[dict[str, Requires]] = {
    **dict.fromkeys(('nsga2', 'nsga3', 'unsga3', 'rnsga3', 'spea2', 'sms_emoa', 'rvea', 'mode', 'mopso'), _PARETO),
    'projected_gradient': Requires(MultiObjective.SINGLE, information=_SLOPES),
    'lbfgsb': Requires(MultiObjective.SINGLE, information=_SLOPES),
    'newton_cg': Requires(MultiObjective.SINGLE, information=_SLOPES),
    'sqp': Requires(MultiObjective.SINGLE, information=_SLOPES, constraints=_SMOOTH),
    'interior_point': Requires(MultiObjective.SINGLE, information=_SLOPES, constraints=_SMOOTH),
    'augmented_lagrangian': Requires(MultiObjective.SINGLE, information=_SLOPES, constraints=_SMOOTH),
    'mma': Requires(MultiObjective.SINGLE, information=_SLOPES, constraints=_SMOOTH),
    'gcmma': Requires(MultiObjective.SINGLE, information=_SLOPES, constraints=_SMOOTH),
    'mgda': Requires(MultiObjective.NATIVE_PARETO, information=_SLOPES),
    'pareto_continuation': Requires(MultiObjective.NATIVE_PARETO, information=_SLOPES, constraints=_SMOOTH),
    **dict.fromkeys(('nsga2', 'nsga3', 'unsga3', 'rnsga3', 'spea2', 'sms_emoa', 'rvea', 'mode', 'mopso'), _PARETO),
    'moead': Requires(MultiObjective.WEIGHTED, metric=True, constraints=frozenset({BlackBox})),
    **dict.fromkeys(
        ('ga', 'cmaes', 'ipop_cmaes', 'map_elites', 'nelder_mead', 'mads', 'bobyqa', 'multistart'), _SINGLE
    ),
    'scalarisation_scan': Requires(MultiObjective.EXTERNAL_SCAN, metric=True, constraints=frozenset({BlackBox})),
    **dict.fromkeys(('hyperband', 'asha'), Requires(MultiObjective.SINGLE, metric=True, fidelity=True)),
    'bo': _SINGLE,
    'parego': Requires(MultiObjective.WEIGHTED, metric=True, constraints=frozenset({BlackBox})),
    **dict.fromkeys(('ehvi', 'nehvi'), _PARETO),
    'mfbo': Requires(MultiObjective.SINGLE, metric=True, fidelity=True),
    'turbo': Requires(MultiObjective.SINGLE, metric=True),
    'prescreened': _PARETO,
    'model_trust_region': _SINGLE,
    'lp': _LINEAR,
    'milp': _LINEAR,
    'qp': Requires(
        MultiObjective.SINGLE, constraints=frozenset({Linear}), forms=frozenset({Form.QUADRATIC, Form.CONVEX})
    ),
}
