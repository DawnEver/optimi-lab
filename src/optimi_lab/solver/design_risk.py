"""A robust run read per DESIGN: each nominal design's scenario results and the declared risk over them.

A robust solver files one result per (design, scenario) and selects on each design's risk row.
:func:`design_risk` reads those rows back through the solver's OWN estimator
(:meth:`~optimi_lab.solver.scenarios.Scenarios.row` over :func:`~optimi_lab.solver.population.score_rows`),
never a second implementation, so a report and the solver cannot disagree about a design's risk or
feasibility. :func:`design_front` is the per-design front a robust run's report ranks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core import non_dominated_sorting, refuse
from optimi_lab.operator import Risk
from optimi_lab.oracle import Result
from optimi_lab.problem import Objective, Problem, Sense
from optimi_lab.solver.population import score_rows
from optimi_lab.solver.scenarios import Scenarios

__all__ = ['DesignRisk', 'design_front', 'design_risk']


@dataclass(frozen=True)
class DesignRisk:
    """One nominal design of a robust run: its scenario results and the declared risk over them.

    ``objectives`` is each objective's risk in its OWN sense and unit (the minimised risk mapped back
    through the objective's normalization), ``loss`` the minimised row the solver ranks on, and
    ``violation`` the solver's: the risk of the per-scenario violation plus each chance constraint's
    shortfall from its probability. All three are None for an INCOMPLETE design -- fewer results than
    *scenarios*, or any not OK -- which, as in the solver, is never ranked.
    """

    design: Mapping[str, object]
    results: tuple[Result, ...]
    risk: Risk
    scenarios: int
    objectives: Mapping[str, float] | None
    loss: tuple[float, ...] | None
    violation: float | None

    @property
    def complete(self) -> bool:
        return self.objectives is not None

    @property
    def rankable(self) -> bool:
        """The reader's name for :attr:`complete`, shared with a single-result view."""
        return self.complete

    @property
    def feasible(self) -> bool:
        return self.complete and self.violation == 0.0

    @property
    def statistic(self) -> str:
        """What every number of this design is: ``CVaR0.9 over 16 scenarios``."""
        return f'{self.risk.label} over {self.scenarios} scenarios'

    def scores(self) -> tuple[float, ...]:
        """The risk vector in objective order and natural sense; an incomplete design has none."""
        if self.objectives is None:
            refuse(
                f'design {dict(self.design)} is incomplete: {len(self.results)} scenario result(s) of '
                f'{self.scenarios} declared, all OK required; no risk stands in for scenarios never computed'
            )
        return tuple(self.objectives.values())


def design_risk(problem: Problem, results: Sequence[Result], *, risk: Risk, scenarios: int) -> tuple[DesignRisk, ...]:
    """Group a robust run's *results* by nominal design, in first-seen order, and read each through the estimator.

    A design is complete when it holds *scenarios* results and every one is OK -- the solver's rule
    for dropping a design whole.
    """
    groups: dict[tuple, list[Result]] = {}
    for result in results:
        groups.setdefault(tuple(sorted(result.action.design.items())), []).append(result)
    estimator = Scenarios(problem, risk, ())
    designs = []
    for key, group in groups.items():
        objectives = loss = violation = None
        if len(group) == scenarios and all(r.cell.status is Status.OK for r in group):
            row, total = estimator.row(*score_rows(estimator.sampled, None, group), group)
            loss = tuple(float(value) for value in row)
            objectives = {o.name: _natural(o, v) for o, v in zip(problem.objectives, loss, strict=True)}
            violation = float(total)
        designs.append(DesignRisk(dict(key), tuple(group), risk, scenarios, objectives, loss, violation))
    return tuple(designs)


def design_front(designs: Sequence[DesignRisk]) -> tuple[DesignRisk, ...]:
    """The non-dominated FEASIBLE designs, ranked on their minimised risk rows, in the given order."""
    ranked = [design for design in designs if design.feasible]
    if not ranked:
        return ()
    kept = {int(index) for index in non_dominated_sorting(np.array([design.loss for design in ranked]))}
    return tuple(design for index, design in enumerate(ranked) if index in kept)


def _natural(objective: Objective, minimized: float) -> float:
    """Undo :meth:`Objective.minimized`: an affine map, which every declared risk measure commutes with."""
    signed = -minimized if objective.sense is Sense.MAXIMIZE else minimized
    return signed * objective.normalization.scale + objective.normalization.reference
