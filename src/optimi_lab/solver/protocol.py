"""The solver protocol the run loop drives: ``ask(capacity)``, ``tell(results)``, ``done``.

A solver builds its own :class:`~optimi_lab.oracle.Action` objects (it holds the problem and the
oracle capability it was made for), is told results in ANY order and possibly a part of a batch at a
time, and says when it has nothing more to do. It never sees a file, a clock or a budget: those are
the run layer's.

DETERMINISM IS WHAT MAKES RESUME A REPLAY. A solver whose asks depend only on its seed and on what
it was told re-asks the same actions when re-driven, and the run answers them from the record.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from optimi_lab.oracle import Action, Result

__all__ = ['Solver']


@runtime_checkable
class Solver(Protocol):
    """See the module docstring."""

    def ask(self, capacity: int) -> Sequence[Action]:
        """At most *capacity* actions to evaluate next; none means the solver has nothing to ask."""
        ...

    def tell(self, results: Sequence[Result]) -> None:
        """Results of earlier asks, in any order."""
        ...

    @property
    def done(self) -> bool:
        """True when the solver has concluded and asks nothing more."""
        ...
