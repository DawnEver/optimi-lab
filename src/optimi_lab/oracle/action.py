"""What a solver asks an oracle to do: evaluate one design, at one fidelity, for some information.

The identity of an action is an :class:`lab_commons.outcomes.Identity` over the canonical design,
the problem, the fidelity, the information wanted, the replicate, the scenario and the oracle version (B0-12): two
actions with one identity are the same task, and an archive may answer one with the other's result.
A *scenario* is one draw of the problem's :class:`~optimi_lab.problem.Uncertainty` (parameter name
-> value) the oracle evaluates the design under; a robust solver asks one action per scenario.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from lab_commons.outcomes import Identity

from optimi_lab.core.errors import refuse
from optimi_lab.oracle.capability import Capability, Information
from optimi_lab.problem import Problem

__all__ = ['Action']


@dataclass(frozen=True, slots=True)
class Action:
    """A canonical *design*, its *fidelity*, the *want*-ed information, a *replicate*, and its *identity*.

    Build with :meth:`of`, which checks every part against the problem and the oracle.
    """

    design: Mapping[str, object]
    fidelity: str | None
    want: frozenset[Information]
    replicate: int | None
    identity: Identity = field(compare=False)
    #: The unit-cube direction ``v`` of an asked Hessian-vector product ``H v``; set iff HVP is wanted.
    direction: tuple[float, ...] | None = None
    #: One draw of the problem's uncertainty (parameter -> value); set iff a robust solver asked it.
    scenario: Mapping[str, float] | None = None

    @classmethod
    def of(
        cls,
        problem: Problem,
        capability: Capability,
        design: Mapping[str, object],
        *,
        fidelity: str | None = None,
        want: frozenset[Information] = frozenset({Information.VALUES}),
        replicate: int | None = None,
        direction: tuple[float, ...] | None = None,
        scenario: Mapping[str, float] | None = None,
    ) -> Action:
        want = frozenset(Information(i) for i in want)
        beyond = sorted(want - capability.provides)
        if beyond:
            refuse(
                f'the action wants {[i.name for i in beyond]}, which the oracle does not provide ({sorted(capability.provides)})'
            )
        if fidelity is not None and fidelity not in capability.fidelities:
            refuse(f'the oracle does not run fidelity {fidelity!r}; it runs {list(capability.fidelities)}')
        if (Information.HVP in want) != (direction is not None):
            refuse('a Hessian-vector product is asked WITH its direction: want HVP iff a direction is given')
        if direction is not None:
            direction = tuple(float(v) for v in direction)
            if len(direction) != problem.space.width:
                refuse(f'a direction has the space width {problem.space.width}, got {len(direction)}')
        if scenario is not None:
            if problem.uncertainty is None:
                refuse('a scenario is a draw of the problem uncertainty, and the problem declares no uncertainty')
            if set(scenario) != set(problem.uncertainty.names):
                refuse(
                    f'a scenario names every uncertain parameter {problem.uncertainty.names}, got {sorted(scenario)}'
                )
            scenario = MappingProxyType({name: float(scenario[name]) for name in problem.uncertainty.names})
        canonical = problem.space.canonical(design)
        identity = problem.identify(
            canonical,
            fidelity,
            extra={'want': sorted(want), 'replicate': replicate, 'oracle': capability.version, 'direction': direction}
            | ({} if scenario is None else {'scenario': dict(scenario)}),
        )
        return cls(MappingProxyType(dict(canonical)), fidelity, want, replicate, identity, direction, scenario)

    def to_plain(self) -> dict[str, object]:
        """The JSON-ready form an archive files with the submission."""
        return {
            'design': dict(self.design),
            'fidelity': self.fidelity,
            'want': sorted(self.want),
            'replicate': self.replicate,
            'direction': self.direction,
            'scenario': None if self.scenario is None else dict(self.scenario),
        }
