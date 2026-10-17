"""The oracle protocol, and the adapter that makes a plain function one.

An :class:`Oracle` is asynchronous: ``await oracle(action)`` yields one :class:`Result`. How many
run at once, and where, is the run layer's ``dispatch``, not the oracle's.

:func:`from_function` wraps a synchronous ``function(action) -> values`` and applies the II.3
optimisation mapping at the one place an exception becomes a status:

========================  ===================================
raised                    cell
========================  ===================================
nothing                   ``OK``
:class:`Invalid`          ``REJECTED`` / ``invalid``
:class:`Infeasible`       ``REJECTED`` / ``infeasible``
:class:`NotConverged`     ``NONCONVERGED`` / ``nonconverged``
:class:`Unavailable`      ``UNAVAILABLE`` / ``unavailable``
any other exception       ``ERROR`` / ``error``
========================  ===================================

A licence miss or a crash is RETRYABLE, never a verdict on the design (B0-06).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from lab_commons.outcomes import Cell, Cost, Reason, Status

from optimi_lab.oracle.action import Action
from optimi_lab.oracle.capability import Capability
from optimi_lab.oracle.result import Result

__all__ = ['Infeasible', 'Invalid', 'NotConverged', 'Oracle', 'Unavailable', 'from_function']


class Invalid(Exception):
    """The design cannot be built (``REJECTED`` / ``invalid``)."""


class Infeasible(Exception):
    """The design is proven to violate a hard requirement (``REJECTED`` / ``infeasible``)."""


class NotConverged(Exception):
    """The evaluator's own iteration did not converge (``NONCONVERGED``)."""


class Unavailable(Exception):
    """A resource the evaluator needs is missing, e.g. a licence (``UNAVAILABLE``)."""


#: Exception type -> status and reason kind; anything else is ``ERROR``.
_MAPPING: tuple[tuple[type[Exception], Status, str], ...] = (
    (Invalid, Status.REJECTED, 'invalid'),
    (Infeasible, Status.REJECTED, 'infeasible'),
    (NotConverged, Status.NONCONVERGED, 'nonconverged'),
    (Unavailable, Status.UNAVAILABLE, 'unavailable'),
)


@runtime_checkable
class Oracle(Protocol):
    """An evaluator: its :class:`Capability`, and ``await oracle(action) -> Result``."""

    capability: Capability

    async def __call__(self, action: Action) -> Result: ...


@dataclass(frozen=True, slots=True)
class _FunctionOracle:
    function: Callable[[Action], Mapping[str, float] | Result]
    capability: Capability

    async def __call__(self, action: Action) -> Result:
        start = time.perf_counter()
        try:
            produced = self.function(action)
        except Exception as error:  # noqa: BLE001 -- every exception is mapped to a status, none is lost
            status, kind = next(((s, k) for t, s, k in _MAPPING if isinstance(error, t)), (Status.ERROR, 'error'))
            cost = Cost(wall_s=time.perf_counter() - start)
            return Result(action, Cell(status, reason=Reason(kind, f'{type(error).__name__}: {error}'), cost=cost))
        if isinstance(produced, Result):
            return produced
        return Result(action, Cell(Status.OK, values=dict(produced), cost=Cost(wall_s=time.perf_counter() - start)))


def from_function(function: Callable[[Action], Mapping[str, float] | Result], capability: Capability) -> Oracle:
    """An :class:`Oracle` over a synchronous *function* of the action, under the mapping above."""
    return _FunctionOracle(function, capability)
