"""The scenarios a robust population solver shares across designs, and the risk it selects on.

See :mod:`optimi_lab.solver.population` for how a generation is asked and scored through them.
"""

from collections.abc import Sequence

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.operator import Risk, draw
from optimi_lab.oracle import Result
from optimi_lab.problem import Chance, Problem

__all__ = ['DEFAULT_SCENARIOS', 'Scenarios']

#: Scenarios per design when a problem declares uncertainty and the caller names no count.
DEFAULT_SCENARIOS = 16


class Scenarios:
    """The fixed scenarios, the declared risk, and the problem a single scenario is scored against."""

    def __init__(self, problem: Problem, risk: Risk, scenarios: tuple[dict[str, float], ...]) -> None:
        self.problem, self.risk, self.scenarios = problem, risk, scenarios
        self.chances = tuple(c for c in problem.constraints if isinstance(c, Chance))
        self.sampled = Problem(
            problem.space,
            problem.objectives,
            constraints=tuple(c for c in problem.constraints if not isinstance(c, Chance)),
            structure=problem.structure,
            uncertainty=problem.uncertainty,
            fidelity=problem.fidelity,
        )

    @classmethod
    def of(
        cls, problem: Problem, risk: Risk | None, scenarios: int | None, rng: np.random.Generator
    ) -> 'Scenarios | None':
        if problem.uncertainty is None:
            if risk is not None or scenarios is not None:
                refuse('risk= and scenarios= estimate over the problem uncertainty, and the problem declares none')
            return None
        if risk is not None and not isinstance(risk, Risk):
            refuse(f'risk= is an optimi_lab.operator.Risk, got {type(risk).__name__}')
        n = DEFAULT_SCENARIOS if scenarios is None else scenarios
        if not isinstance(n, int) or n < 1:
            refuse(f'scenarios must be a positive integer, got {scenarios!r}')
        drawn = draw(problem.uncertainty, n, rng)
        table = tuple({name: float(column[i]) for name, column in drawn.items()} for i in range(n))
        return cls(problem, Risk('mean') if risk is None else risk, table)

    def row(self, values: np.ndarray, violation: np.ndarray, group: Sequence[Result]) -> tuple[np.ndarray, float]:
        """One design's row and violation from its per-scenario *values* and *violation* (scored against
        :attr:`sampled`): the risk of each over the scenarios, plus each chance constraint's shortfall.
        """
        row = np.array([self.risk.of(column) for column in values.T])
        total = self.risk.of(violation)
        for chance in self.chances:
            holds = np.mean([_satisfied(chance, r.cell.values[chance.output]) for r in group])
            total += max(chance.probability - float(holds), 0.0) / chance.probability
        return row, total


def _satisfied(chance: Chance, value: float) -> bool:
    return value <= chance.bound if chance.relation == '<=' else value >= chance.bound
