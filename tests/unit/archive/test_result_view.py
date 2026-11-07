"""The archive read as views: rankable, violation, feasibility, scores, first-submitted order, the record door."""

from __future__ import annotations

from pathlib import Path

import pytest
from lab_commons.outcomes import Cell, Reason, Record, Status

from optimi_lab.archive import Archive, ResultView, open_record, views
from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import BlackBox, Objective, Problem, Real, Sense, Space

PROBLEM = Problem(
    Space((Real('x', 0.0, 1.0),)),
    (Objective('cost'), Objective('gain', Sense.MAXIMIZE)),
    constraints=(BlackBox('temp', '<=', 100.0, scale=10.0),),
)
CAPABILITY = Capability(outputs=('cost', 'gain', 'temp'), version='toy')


def _result(x: float, temp: float = 50.0, status: Status = Status.OK) -> Result:
    action = Action.of(PROBLEM, CAPABILITY, {'x': x})
    if status is Status.OK:
        return Result(action, Cell(Status.OK, values={'cost': x, 'gain': 1.0 - x, 'temp': temp}))
    return Result(action, Cell(status, reason=Reason('infeasible')))


def test_a_view_reads_feasibility_and_scores_in_natural_sense() -> None:
    view = ResultView(PROBLEM, _result(0.25))
    assert view.rankable and view.feasible and view.violation == 0.0
    assert view.scores() == (0.25, 0.75)


def test_a_violating_view_is_rankable_but_not_feasible() -> None:
    view = ResultView(PROBLEM, _result(0.25, temp=120.0))
    assert view.rankable and not view.feasible
    assert view.violation == pytest.approx(2.0)


def test_a_failed_view_has_no_scores() -> None:
    view = ResultView(PROBLEM, _result(0.5, status=Status.REJECTED))
    assert not view.rankable and view.uncomputed == ('cost', 'gain')
    with pytest.raises(Refusal, match='did not compute'):
        view.scores()


def test_views_of_a_reopened_record_equal_the_live_ones(tmp_path: Path) -> None:
    record = Record(tmp_path / 'archive.jsonl')
    archive = Archive(PROBLEM, record)
    first, second = _result(0.75), _result(0.25)
    archive.submit(first.action)
    archive.submit(second.action)
    archive.tell(second)  # completes before the first-submitted one
    archive.tell(first)
    live = views(archive)
    assert [view.result.action.design['x'] for view in live] == [0.75, 0.25]
    reopened = views(Archive(PROBLEM, Record(tmp_path / 'archive.jsonl')))
    assert reopened == live
    assert len({*live, *reopened}) == 2


def test_open_record_passes_a_record_opens_a_path_and_needs_a_root_for_a_fresh_one(tmp_path: Path) -> None:
    record = Record(tmp_path / 'given.jsonl')
    assert open_record(record) is record
    assert open_record(tmp_path / 'at.jsonl').path == tmp_path / 'at.jsonl'
    fresh = open_record(root=tmp_path)
    assert fresh.path.parent.parent == tmp_path and fresh.path.name == 'archive.jsonl'
    with pytest.raises(Refusal, match='root='):
        open_record()
