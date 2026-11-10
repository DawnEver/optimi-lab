"""A study names any driven solver: a gradient method walks from its one start, a robust search pays every scenario."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Reason, Record, Status
from study_toys import PROBLEM, SPACE, Toy

from optimi_lab.core import SampleKind, SampleSpec
from optimi_lab.core.errors import Refusal
from optimi_lab.operator import Risk
from optimi_lab.oracle import Action, Capability, Information, Result
from optimi_lab.problem import Objective, Problem
from optimi_lab.problem.declarations import Uncertainty
from optimi_lab.solver import design_front, design_risk
from optimi_lab.study import FixedDesign, SampleDesign, SearchStrategy, search

BOWL = Problem(SPACE, (Objective('y'),))
SPANS = np.array([1.0, 2.0])  # hi - lo of a and b: d y / d u = (d y / d p) * span


class Bowl:
    """y = (a - 0.3)^2 + (b - 3)^2, answered with its unit-cube slope when the action asks for it."""

    capability = Capability(('y',), version='bowl', provides=frozenset({Information.VALUES, Information.VJP}))
    evaluators = 1

    def __init__(self) -> None:
        self.asked: list[Action] = []

    def answer(self, action: Action) -> Result:
        a, b = float(action.design['a']), float(action.design['b'])
        slope = np.array([2.0 * (a - 0.3), 2.0 * (b - 3.0)]) * SPANS
        gradient = {'y': slope} if Information.VJP in action.want else None
        return Result(action, Cell(Status.OK, values={'y': (a - 0.3) ** 2 + (b - 3.0) ** 2}), gradient=gradient)

    def screened(self, action: Action) -> Result:
        return Result(action, Cell(Status.SCREENED, reason=Reason('screened', 'the policy spent no solve')))

    def routed(self, action: Action, index: int) -> Action:
        return action

    def publish_batch(self, actions, on_result) -> None:
        for action in actions:
            self.asked.append(action)
            on_result(self.answer(action))

    async def __call__(self, action: Action) -> Result:
        return self.answer(action)


def test_a_search_reaches_the_gradient_solver_lbfgsb_by_name(tmp_path: Path) -> None:
    bowl = Bowl()
    searched = search(
        BOWL, bowl, Record(tmp_path / 'r.jsonl'), algorithm='lbfgsb', generations=20, initial=np.array([[0.9, 2.2]])
    )
    assert bowl.asked and all(Information.VJP in action.want for action in bowl.asked)
    assert bowl.asked[0].design == {'a': pytest.approx(0.9), 'b': pytest.approx(2.2)}
    best = min(result.cell.values['y'] for result in searched.ordered_results())
    assert best < 1e-6


def test_a_strategy_names_a_driven_solver_and_refuses_a_first_batch_it_would_replace() -> None:
    assert not SearchStrategy(FixedDesign([(0.5, 3.0)]), algorithm='lbfgsb').draws_own
    assert SearchStrategy(
        SampleDesign(SampleSpec(n_samples=4, kind=SampleKind.LATIN_HYPERCUBE)), algorithm='bo'
    ).draws_own
    with pytest.raises(Refusal, match='draws its own first batch'):
        SearchStrategy(FixedDesign([(0.1, 2.5), (0.4, 3.0)]), algorithm='bo')
    with pytest.raises(Refusal, match='unknown algorithm'):
        SearchStrategy(FixedDesign([(0.1, 2.5)]), algorithm='milp')


def test_a_robust_search_pays_every_scenario_and_reads_back_per_design(tmp_path: Path) -> None:
    robust = Problem(SPACE, PROBLEM.objectives, uncertainty=Uncertainty.declared([('t', 'normal', 0.0, 1.0, None)]))
    toy, risk = Toy(), Risk('cvar', alpha=0.9)
    searched = search(robust, toy, Record(tmp_path / 'r.jsonl'), population=4, generations=2, risk=risk, scenarios=3)
    assert searched.search_refusal is None
    # Three batches of four designs at three scenarios each: the budget counts every scenario.
    assert searched.batch_ends[0] == 4 * 3 and len(searched.batch_ends) == 3
    designs = design_risk(robust, searched.ordered_results(), risk=risk, scenarios=3)
    assert designs and all(design.complete for design in designs)
    assert designs[0].statistic == 'CVaR0.9 over 3 scenarios'
    assert all(any(front is design for design in designs) for front in design_front(designs))
