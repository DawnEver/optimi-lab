"""A study search over a problem that declares Uncertainty asks each design once per shared scenario.

The robust population solvers already estimate a declared ``Risk`` over ``scenarios``
(``tests/unit/solver/test_robust_population.py``); these tests pin that the STUDY engine hands
them both, that its evaluation budget counts one evaluation per scenario, and that a strategy's
risk declaration is part of its canonical slot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Record, Status
from study_toys import AXES, READINGS, SPACE, Codec, Publisher, Toy

from optimi_lab.core.errors import Refusal
from optimi_lab.operator import Risk
from optimi_lab.oracle import Action, Result
from optimi_lab.problem import Objective, Problem, Sense, Uncertainty, Uniform
from optimi_lab.study import FixedDesign, Request, SearchBinding, SearchStrategy, build_solver, search, study

XI = Uncertainty((Uniform('xi', -0.5, 0.5),))
ROBUST = Problem(SPACE, (Objective('y'), Objective('z', Sense.MAXIMIZE)), uncertainty=XI)
START = np.array([[0.1, 2.5], [0.4, 3.0], [0.7, 3.5]])


class Perturbed(Toy):
    """The toy whose answer moves with its scenario: y = a + b + xi."""

    def answer(self, action: Action) -> Result:
        a, b = float(action.design['a']), float(action.design['b'])
        xi = float(action.scenario['xi'])
        return Result(action, Cell(Status.OK, values={'y': a + b + xi, 'z': a - b / 4}))


class RobustCodec(Codec):
    def searching(self, filing, record: Record) -> SearchBinding:
        return SearchBinding(ROBUST, self.oracle, record)


def test_a_robust_search_asks_every_design_under_every_shared_scenario(tmp_path: Path) -> None:
    oracle = Perturbed()
    searched = search(
        ROBUST,
        oracle,
        Record(tmp_path / 'r.jsonl'),
        algorithm='nsga2',
        population=3,
        generations=1,
        initial=START,
        risk=Risk('worst_case'),
        scenarios=4,
    )
    results = searched.ordered_results()
    assert searched.search_refusal is None
    assert all(result.action.scenario is not None for result in results)
    per_design: dict[tuple, set[float]] = {}
    for result in results:
        per_design.setdefault(tuple(sorted(result.action.design.items())), set()).add(result.action.scenario['xi'])
    # Every design is asked under ALL FOUR shared draws: a scenario is one evaluation, so the generations stop
    # (population * scenarios per batch) does not cut the second batch short. An offspring repeating a parent
    # is served from the archive, so the second batch adds at most population new designs.
    assert len(set(map(frozenset, per_design.values()))) == 1
    assert {len(draws) for draws in per_design.values()} == {4}
    assert 3 < len(per_design) <= 3 * 2
    assert len(results) == 4 * len(per_design)


def test_risk_on_a_problem_without_uncertainty_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Refusal, match='uncertainty'):
        search(
            Problem(SPACE, (Objective('y'),)),
            Toy(),
            Record(tmp_path / 'r.jsonl'),
            population=3,
            initial=START,
            scenarios=4,
        )


def test_build_solver_hands_the_declared_risk_to_the_solver() -> None:
    solver = build_solver('nsga2', ROBUST, Perturbed().capability, seed=0, population=3, generations=1, scenarios=5)
    assert len(solver.ask(100)) == 3 * 5


def test_a_strategy_declares_risk_and_scenarios_and_they_open_their_own_slot(tmp_path: Path) -> None:
    path = tmp_path / 'r.jsonl'
    design = FixedDesign(tuple(map(tuple, START)))
    for slot, strategy in enumerate((
        SearchStrategy(design, algorithm='nsga2', generations=1, risk=Risk('mean'), scenarios=2),
        SearchStrategy(design, algorithm='nsga2', generations=1, risk=Risk('worst_case'), scenarios=2),
    )):
        oracle = Perturbed()
        study(
            Request(
                SPACE,
                READINGS,
                Publisher(),
                strategy,
                {'case': 'c'},
                axes=AXES,
                codec=RobustCodec(oracle),
                record_path=path,
            )
        )
        searches = [e for e in Record(path).replay() if e.kind == 'study_search']
        assert len(searches) == slot + 1
        assert searches[-1].data['risk'] == {
            'measure': strategy.risk.measure.value,
            'alpha': None,
            'k': None,
            'scenarios': 2,
        }
        assert slot == 1 or oracle.published  # the first slot solved its scenario rows; the second reuses them


def test_a_scenario_count_below_one_is_refused() -> None:
    with pytest.raises(Refusal, match='scenarios'):
        SearchStrategy(FixedDesign(tuple(map(tuple, START))), scenarios=0)
