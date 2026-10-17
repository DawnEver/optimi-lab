"""What a run may spend, what it has spent, and the named reason it stopped.

The ledger charges ONE evaluation per result, however many outputs it carries (B0-16: one solver
call that yields torque and copper loss is one call), plus the result's fidelity cost and its wall
time. An unknown wall time is counted as unknown -- :attr:`Ledger.unmeasured` -- never as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from optimi_lab.core.errors import refuse
from optimi_lab.oracle import Result
from optimi_lab.problem import Problem

__all__ = ['Budget', 'Ledger', 'Stop', 'Termination']


class Termination(StrEnum):
    """Why a run stopped."""

    EVALUATIONS = 'evaluations'
    WALL = 'wall'
    COST = 'cost'
    SOLVER_DONE = 'solver_done'
    NOTHING_ASKED = 'nothing_asked'


@dataclass(frozen=True, slots=True)
class Stop:
    """The *reason* a run stopped and a sentence a reader can act on."""

    reason: Termination
    detail: str


@dataclass(frozen=True, slots=True)
class Budget:
    """Limits on evaluations, wall seconds and fidelity cost units; at least one is set."""

    evaluations: int | None = None
    wall_s: float | None = None
    cost: float | None = None

    def __post_init__(self) -> None:
        limits = [v for v in (self.evaluations, self.wall_s, self.cost) if v is not None]
        if not limits:
            refuse('a budget needs at least one limit (evaluations, wall_s or cost); an unbounded run never stops')
        if not all(v > 0 for v in limits):
            refuse(f'every budget limit must be > 0, got {self}')


@dataclass(slots=True)
class Ledger:
    """What a run on *problem* has spent so far."""

    problem: Problem
    evaluations: int = 0
    wall_s: float = 0.0
    cost: float = 0.0
    unmeasured: int = 0

    def charge(self, result: Result) -> None:
        """Charge one evaluation of *result*."""
        self.evaluations += 1
        fidelity = self.problem.fidelity
        level = result.action.fidelity
        self.cost += 1.0 if fidelity is None or level is None else fidelity.levels[fidelity.index(level)].cost
        wall = None if result.cell.cost is None else result.cell.cost.wall_s
        if wall is None:
            self.unmeasured += 1
        else:
            self.wall_s += wall

    def remaining(self, budget: Budget) -> int | None:
        """Evaluations left under *budget*, or None when it sets no evaluation limit."""
        return None if budget.evaluations is None else max(budget.evaluations - self.evaluations, 0)

    def exhausted(self, budget: Budget) -> Stop | None:
        """The stop *budget* calls for now, or None."""
        spent = f'after {self.evaluations} evaluation(s), {self.wall_s:g} s wall, {self.cost:g} cost unit(s)'
        if budget.evaluations is not None and self.evaluations >= budget.evaluations:
            return Stop(Termination.EVALUATIONS, f'evaluation limit {budget.evaluations} reached {spent}')
        if budget.wall_s is not None and self.wall_s >= budget.wall_s:
            return Stop(Termination.WALL, f'wall limit {budget.wall_s:g} s reached {spent}')
        if budget.cost is not None and self.cost >= budget.cost:
            return Stop(Termination.COST, f'cost limit {budget.cost:g} reached {spent}')
        return None
