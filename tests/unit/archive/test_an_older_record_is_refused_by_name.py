"""FORMATS-MIGRATE-ONCE (user ruling D51, 2026-11-11): the archive reads only the current submission shape.

A submission filed before actions carried a ``scenario`` is refused BY NAME with its migration instruction,
never upgraded on read.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lab_commons.outcomes import Record

from optimi_lab.archive import Archive
from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability
from optimi_lab.problem import Objective, Problem, Real, Space

PROBLEM = Problem(Space((Real('x', 0.0, 1.0),)), (Objective('cost'),))
CAPABILITY = Capability(outputs=('cost',), version='toy')


def _file(path: Path, *, drop_scenario: bool) -> Action:
    action = Action.of(PROBLEM, CAPABILITY, {'x': 0.5})
    data = action.to_plain() | {'identity': action.identity.canonical, 'key': action.identity.key}
    if drop_scenario:
        del data['scenario']
    Record(path).submit(action.identity, data)
    return action


def test_planted_control_a_current_record_reopens(tmp_path: Path) -> None:
    """The refusal below is about the missing key, not about reopening a record at all."""
    action = _file(tmp_path / 'r.jsonl', drop_scenario=False)
    assert [a.identity for a in Archive(PROBLEM, Record(tmp_path / 'r.jsonl')).pending()] == [action.identity]


def test_a_submission_without_a_scenario_is_refused_by_name(tmp_path: Path) -> None:
    action = _file(tmp_path / 'r.jsonl', drop_scenario=True)
    with pytest.raises(Refusal, match='older record shape') as refusal:
        Archive(PROBLEM, Record(tmp_path / 'r.jsonl'))
    assert repr(action.identity.key) in str(refusal.value)
    assert 're-running the study into a fresh record' in str(refusal.value)
