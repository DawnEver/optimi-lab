"""D19-D21: the archive over ``outcomes.Record``, its views, and predictions kept apart."""

from __future__ import annotations

from pathlib import Path

import pytest
from lab_commons.outcomes import Cell, Evidence, FinalityError, Reason, Record, Status

from optimi_lab.archive import Archive
from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import BlackBox, Fidelity, Level, Objective, Problem, Real, Sense, Space

PROBLEM = Problem(
    Space((Real('x', 0.0, 1.0),)),
    (Objective('cost'), Objective('gain', Sense.MAXIMIZE)),
    constraints=(BlackBox('temp', '<=', 100.0, scale=10.0),),
    fidelity=Fidelity((Level('mec', 1.0), Level('fea', 10.0))),
)
CAPABILITY = Capability(outputs=('cost', 'gain', 'temp'), fidelities=('mec', 'fea'), version='toy')


def _action(x: float, fidelity: str = 'fea') -> Action:
    return Action.of(PROBLEM, CAPABILITY, {'x': x}, fidelity=fidelity)


def _ok(
    x: float,
    cost: float,
    gain: float,
    temp: float = 50.0,
    *,
    fidelity: str = 'fea',
    evidence: Evidence = Evidence.OBSERVED,
) -> Result:
    return Result(
        _action(x, fidelity), Cell(Status.OK, values={'cost': cost, 'gain': gain, 'temp': temp}, evidence=evidence)
    )


def _archive(tmp_path: Path) -> Archive:
    return Archive(PROBLEM, Record(tmp_path / 'run.jsonl'))


def test_d19_the_archive_files_submissions_and_outcomes_in_the_record(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    action = _action(0.1)
    archive.submit(action)
    assert [a.identity for a in archive.pending()] == [action.identity]
    archive.tell(_ok(0.1, 1.0, 2.0))
    assert archive.pending() == ()
    reopened = _archive(tmp_path)
    assert reopened.result(action.identity).cell.values['cost'] == 1.0
    assert dict(reopened.result(action.identity).action.design) == {'x': 0.1}


def test_d19_planted_control_a_conflicting_second_decision_is_refused(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.submit(_action(0.1))
    archive.tell(_ok(0.1, 1.0, 2.0))
    with pytest.raises(FinalityError):
        archive.tell(Result(_action(0.1), Cell(Status.REJECTED, reason=Reason('infeasible'))))


def test_d19_b0_06_a_timeout_then_a_success_is_one_success_and_never_infeasible(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.submit(_action(0.2))
    archive.tell(Result(_action(0.2), Cell(Status.TIMEOUT, reason=Reason('timeout', 'wall limit 600 s'))))
    assert [r.action.identity for r in archive.retryable()] == [_action(0.2).identity]
    assert archive.complete() == ()
    archive.tell(_ok(0.2, 1.0, 1.0))
    assert len(archive.complete()) == 1
    assert archive.retryable() == ()


def test_d20_views_complete_feasible_front_and_fidelity(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    for result in (
        _ok(0.1, 1.0, 1.0),
        _ok(0.2, 2.0, 3.0),
        _ok(0.3, 3.0, 2.0),  # dominated by 0.2
        _ok(0.4, 0.5, 5.0, temp=104.0),  # best, but infeasible by 0.4 scaled units
        _ok(0.5, 0.1, 0.1, fidelity='mec'),
    ):
        archive.submit(result.action)
        archive.tell(result)
    archive.submit(_action(0.6))
    archive.tell(Result(_action(0.6), Cell(Status.REJECTED, reason=Reason('invalid'))))
    assert len(archive.complete()) == 5
    assert len(archive.feasible()) == 4
    assert len(archive.feasible(tol=0.5)) == 5
    assert {r.action.design['x'] for r in archive.front(fidelity='fea')} == {0.1, 0.2}
    assert [r.action.design['x'] for r in archive.at_fidelity('mec')] == [0.5]


def test_d20_planted_control_the_front_refuses_to_mix_fidelities(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    with pytest.raises(Refusal, match='fidelity'):
        archive.front()


def test_d21_predictions_are_kept_apart_b0_08(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.submit(_action(0.1))
    archive.tell(_ok(0.1, 2.0, 2.0))
    archive.predict(_ok(0.9, 0.0, 9.0, evidence=Evidence.PREDICTED))
    assert {r.action.design['x'] for r in archive.front(fidelity='fea')} == {0.1}
    predicted_front = archive.front(Evidence.PREDICTED, fidelity='fea')
    assert {r.action.design['x'] for r in predicted_front} == {0.9}
    assert all(r.cell.evidence is Evidence.PREDICTED for r in archive.predictions())
    assert len(list(Record(tmp_path / 'run.jsonl').replay())) == 2  # a prediction never reaches the record


def test_d21_planted_control_a_prediction_cannot_be_told_as_an_observation(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    with pytest.raises(Refusal, match='PREDICTED'):
        archive.tell(_ok(0.1, 0.0, 9.0, evidence=Evidence.PREDICTED))
    with pytest.raises(Refusal, match='OBSERVED'):
        archive.predict(_ok(0.1, 0.0, 9.0))
