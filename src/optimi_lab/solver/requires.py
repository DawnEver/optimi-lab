"""What a solver needs, and the one match that says whether a problem and an oracle supply it.

The information an evaluator offers decides what an algorithm can be used for (plan I.2.3). A
:class:`Requires` declares a solver's needs; :func:`match` answers:

* :class:`Eligible` -- usable as is;
* :class:`NeedsTransform` -- usable after the NAMED transforms, at a stated evaluation cost per
  action (finite differences multiply it); the transforms used are part of what a result reports
  (B0-20);
* :class:`Ineligible` -- not usable, with the reason and what would make it usable.

:data:`REGISTRY` names every solver this package ships, so a stranger is refused naming the set.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from optimi_lab.core.errors import refuse
from optimi_lab.oracle import Capability, Information
from optimi_lab.problem import BlackBox, Categorical, Chance, Explicit, Field, Form, Linear, Path, Problem, Real

__all__ = ['REGISTRY', 'Eligible', 'Ineligible', 'Match', 'MultiObjective', 'NeedsTransform', 'Requires', 'match']


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

    @staticmethod
    def of(name: str) -> Requires:
        """The registered needs of solver *name*."""
        if name not in REGISTRY:
            refuse(f'no solver {name!r} is registered; the registered solvers are {sorted(REGISTRY)}')
        return REGISTRY[name]


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


#: Every solver this package ships, by name. The population solvers rank by Deb's rule, so they take
#: black-box constraints natively; MOEA/D decomposes by Tchebycheff weights.
REGISTRY: Final[dict[str, Requires]] = {
    'nsga2': Requires(MultiObjective.NATIVE_PARETO, metric=True, constraints=frozenset({BlackBox})),
    'nsga3': Requires(MultiObjective.NATIVE_PARETO, metric=True, constraints=frozenset({BlackBox})),
    'mode': Requires(MultiObjective.NATIVE_PARETO, metric=True, constraints=frozenset({BlackBox})),
    'mopso': Requires(MultiObjective.NATIVE_PARETO, metric=True, constraints=frozenset({BlackBox})),
    'moead': Requires(MultiObjective.WEIGHTED, metric=True, constraints=frozenset({BlackBox})),
}
