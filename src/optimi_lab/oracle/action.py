"""What a solver asks an oracle to do: evaluate one design, at one fidelity, for some information.

The identity of an action is an :class:`lab_commons.outcomes.Identity` over the canonical design,
the problem, the fidelity, the information wanted, the replicate and the oracle version (B0-12): two
actions with one identity are the same task, and an archive may answer one with the other's result.
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
    ) -> Action:
        want = frozenset(Information(i) for i in want)
        beyond = sorted(want - capability.provides)
        if beyond:
            refuse(
                f'the action wants {[i.name for i in beyond]}, which the oracle does not provide ({sorted(capability.provides)})'
            )
        if fidelity is not None and fidelity not in capability.fidelities:
            refuse(f'the oracle does not run fidelity {fidelity!r}; it runs {list(capability.fidelities)}')
        canonical = problem.space.canonical(design)
        identity = problem.identify(
            canonical,
            fidelity,
            extra={'want': sorted(want), 'replicate': replicate, 'oracle': capability.version},
        )
        return cls(MappingProxyType(dict(canonical)), fidelity, want, replicate, identity)

    def to_plain(self) -> dict[str, object]:
        """The JSON-ready form an archive files with the submission."""
        return {
            'design': dict(self.design),
            'fidelity': self.fidelity,
            'want': sorted(self.want),
            'replicate': self.replicate,
        }
