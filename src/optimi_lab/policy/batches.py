"""A generation budget around a solver: it counts told batches and never varies or selects.

The run's evaluation budget charges FRESH results, not reused identities, so a population that
repeats decided points spends nothing. Counting completed ask/tell batches as well is what keeps a
finite generation request from becoming an unbounded run.
"""

from __future__ import annotations

from collections.abc import Sequence

from lab_commons.outcomes import Status

from optimi_lab.oracle import Action, Result
from optimi_lab.solver.protocol import Solver

__all__ = ['Batches']


class Batches:
    """*solver*, done after *batches* tells; ``last_batch_complete`` says whether the last told batch held an OK."""

    def __init__(self, solver: Solver, batches: int) -> None:
        self.solver, self.remaining = solver, batches
        self.last_batch_complete: bool | None = None

    def ask(self, capacity: int) -> Sequence[Action]:
        return self.solver.ask(capacity)

    def tell(self, results: Sequence[Result]) -> None:
        self.last_batch_complete = any(result.cell.status is Status.OK for result in results)
        self.solver.tell(results)
        self.remaining -= 1

    @property
    def done(self) -> bool:
        return self.remaining <= 0 or self.solver.done
