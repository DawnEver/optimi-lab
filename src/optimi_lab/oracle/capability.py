"""What an oracle can tell: which information classes, which outputs, at which fidelities, how wide.

The information an evaluator offers decides what an algorithm may be used for (plan I.2.3), so it
is declared here once and matched against a solver's ``Requires`` by the capability registry.

A declared :class:`~optimi_lab.problem.Form` says an output IS linear (quadratic); only the oracle can
say WHAT its coefficients are (P14 ruling 2026-10-19). :data:`COEFFICIENTS_FOR` is the one table from
a form to the information class that supplies them; an action wanting one is answered with each
output's coefficients over the natural design vector (:class:`~optimi_lab.oracle.Result`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from optimi_lab.core.errors import refuse
from optimi_lab.problem import Form, Problem

__all__ = ['COEFFICIENTS_FOR', 'Capability', 'Information']


class Information(StrEnum):
    """One class of information an oracle can return about a design."""

    VALUES = 'values'
    JVP = 'jvp'
    VJP = 'vjp'
    HVP = 'hvp'
    STRUCTURE = 'structure'
    STOCHASTIC = 'stochastic'
    #: Each output's ``a`` in ``value = a.x + k`` over the natural design vector.
    LINEAR_COEFFICIENTS = 'linear_coefficients'
    #: Each output's ``a`` and ``Q`` in ``value = 1/2 x.Q x + a.x + k``.
    QUADRATIC_COEFFICIENTS = 'quadratic_coefficients'


#: The information that supplies the coefficients of a declared form -- the one table.
COEFFICIENTS_FOR: Final = MappingProxyType({
    Form.LINEAR: Information.LINEAR_COEFFICIENTS,
    Form.QUADRATIC: Information.QUADRATIC_COEFFICIENTS,
})


@dataclass(frozen=True, slots=True)
class _CapabilityFields:
    outputs: tuple[str, ...]
    provides: frozenset[Information] = frozenset({Information.VALUES})
    fidelities: tuple[str, ...] = ()
    nodes: int = 1
    version: str = ''


@dataclass(frozen=True, slots=True, init=False)
class Capability(_CapabilityFields):
    """*provides* (always VALUES), *outputs* produced together, *fidelities*, *nodes* in parallel, *version*.

    *version* names the evaluator build; it enters every action's identity (B0-12), so a result from
    one solver version is never served for another.
    """

    def __init__(
        self,
        outputs: tuple[str, ...],
        provides: frozenset[Information] = frozenset({Information.VALUES}),
        fidelities: tuple[str, ...] = (),
        nodes: int = 1,
        version: str = '',
    ) -> None:
        outputs, fidelities = tuple(outputs), tuple(fidelities)
        provides = frozenset(Information(i) for i in provides)
        if Information.VALUES not in provides:
            refuse(f'an oracle provides VALUES at least, got {sorted(provides)}')
        if not isinstance(nodes, int) or nodes < 1:
            refuse(f'an oracle runs on nodes >= 1, got {nodes!r}')
        if not outputs:
            refuse('an oracle produces at least one output')
        _CapabilityFields.__init__(self, outputs, provides, fidelities, nodes, version)

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
