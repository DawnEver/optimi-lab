"""A declared solve budget stops a study after that many oracle evaluations; each scenario is one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Record
from study_toys import AXES, PROBLEM, READINGS, SPACE, Codec, Publisher, Toy

from optimi_lab.core.errors import Refusal
from optimi_lab.operator import Risk
from optimi_lab.problem import Problem
from optimi_lab.problem.declarations import Uncertainty
from optimi_lab.run import Termination
from optimi_lab.study import FixedDesign, Request, SearchStrategy, SurrogateStrategy, search, study

START = np.array([[0.1, 2.5], [0.4, 3.0], [0.7, 3.5], [0.9, 2.2]])


def test_a_search_stops_once_it_has_made_its_declared_solves(tmp_path: Path) -> None:
    toy = Toy()
    searched = search(PROBLEM, toy, Record(tmp_path / 'r.jsonl'), population=4, generations=2, initial=START, solves=6)
    assert searched.execution is not None
    assert searched.execution.ledger.evaluations == len(toy.published) == 6
    assert searched.execution.stop.reason is Termination.EVALUATIONS


def test_a_budget_above_the_generation_bound_leaves_the_generations_to_stop_it(tmp_path: Path) -> None:
    free, capped = Toy(), Toy()
    search(PROBLEM, free, Record(tmp_path / 'free.jsonl'), population=4, generations=2, initial=START)
    searched = search(
        PROBLEM, capped, Record(tmp_path / 'r.jsonl'), population=4, generations=2, initial=START, solves=99
    )
    assert searched.execution is not None
    assert capped.published == free.published
    assert searched.execution.stop.reason is Termination.SOLVER_DONE


def test_every_scenario_counts_as_one_solve(tmp_path: Path) -> None:
    robust = Problem(SPACE, PROBLEM.objectives, uncertainty=Uncertainty.declared([('t', 'normal', 0.0, 1.0, None)]))
    toy = Toy()
    searched = search(
        robust,
        toy,
        Record(tmp_path / 'r.jsonl'),
        population=4,
        generations=2,
        risk=Risk('cvar', alpha=0.9),
        scenarios=3,
        solves=15,
    )
    assert searched.execution is not None
    # Four designs at three scenarios is twelve; the budget then admits three more scenario evaluations.
    assert searched.execution.ledger.evaluations == 15
    assert searched.execution.stop.reason is Termination.EVALUATIONS


def test_a_nonpositive_budget_and_a_budget_beside_surrogate_feedback_are_refused(tmp_path: Path) -> None:
    with pytest.raises(Refusal, match='solves'):
        search(PROBLEM, Toy(), Record(tmp_path / 'r.jsonl'), population=4, initial=START, solves=0)
    with pytest.raises(Refusal, match='solves'):
        SearchStrategy(FixedDesign([(0.1, 2.5)]), solves=0)
    with pytest.raises(Refusal, match='surrogate'):
        SearchStrategy(FixedDesign([(0.1, 2.5)]), surrogate=SurrogateStrategy(), solves=4)


def test_a_study_applies_the_strategy_budget_and_files_it_with_the_search(tmp_path: Path) -> None:
    path, toy = tmp_path / 'r.jsonl', Toy()
    strategy = SearchStrategy(FixedDesign(tuple(map(tuple, START))), generations=2, seed=3, solves=5)
    outcome = study(
        Request(SPACE, READINGS, Publisher(), strategy, {'case': 'c'}, axes=AXES, codec=Codec(toy), record_path=path)
    )
    assert len(toy.published) == 5
    assert outcome.result.execution.stop.reason is Termination.EVALUATIONS
    searches = [event.data for event in Record(path).replay() if event.kind == 'study_search']
    assert searches and searches[-1]['solves'] == 5
