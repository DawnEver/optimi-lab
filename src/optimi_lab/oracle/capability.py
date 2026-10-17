"""What an oracle can tell: which information classes, which outputs, at which fidelities, how wide.

The information an evaluator offers decides what an algorithm may be used for (plan I.2.3), so it
is declared here once and matched against a solver's ``Requires`` by the capability registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from optimi_lab.core.errors import refuse
from optimi_lab.problem import Problem

__all__ = ['Capability', 'Information']


class Information(StrEnum):
    """One class of information an oracle can return about a design."""

    VALUES = 'values'
    JVP = 'jvp'
    VJP = 'vjp'
    HVP = 'hvp'
    STRUCTURE = 'structure'
    STOCHASTIC = 'stochastic'


@dataclass(frozen=True, slots=True)
class Capability:
    """*provides* (always VALUES), *outputs* produced together, *fidelities*, *nodes* in parallel, *version*.

    *version* names the evaluator build; it enters every action's identity (B0-12), so a result from
    one solver version is never served for another.
    """

    outputs: tuple[str, ...]
    provides: frozenset[Information] = frozenset({Information.VALUES})
    fidelities: tuple[str, ...] = ()
    nodes: int = 1
    version: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'outputs', tuple(self.outputs))
        object.__setattr__(self, 'fidelities', tuple(self.fidelities))
        object.__setattr__(self, 'provides', frozenset(Information(i) for i in self.provides))
        if Information.VALUES not in self.provides:
            refuse(f'an oracle provides VALUES at least, got {sorted(self.provides)}')
        if not isinstance(self.nodes, int) or self.nodes < 1:
            refuse(f'an oracle runs on nodes >= 1, got {self.nodes!r}')
        if not self.outputs:
            refuse('an oracle produces at least one output')

    def require(self, problem: Problem) -> None:
        """Refuse, naming what is missing, unless this oracle can evaluate *problem*."""
        missing = [name for name in problem.outputs if name not in self.outputs]
        if missing:
            refuse(f'the oracle does not produce outputs {missing}; add them to Capability(outputs=...)')
        if problem.fidelity is not None:
            levels = [level.name for level in problem.fidelity.levels]
            missing = [name for name in levels if name not in self.fidelities]
            if missing:
                refuse(f'the oracle does not run fidelities {missing}; add them to Capability(fidelities=...)')
