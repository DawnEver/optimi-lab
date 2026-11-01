"""The study record events are BYTE-IDENTICAL to the ones motronics wrote before the engine moved.

The golden lines were produced by motronics ``pareto.study.study`` (4320f5e438) on the same request --
two axes as motronics' ``AxisSpec`` spells them, one reading, a two-batch feedback design -- and are
copied here verbatim; this test imports nothing from motronics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lab_commons.outcomes import Record
from study_toys import AXES, READINGS, SPACE, Codec, Publisher

from optimi_lab.study import FeedbackDesign, FixedDesign, Request, SampleDesign, study, study_writer

GOLDEN = [
    '{"data":{"evaluation_identity":{"case":"c","n":1},"physical_record_schema":2,"space":[{"addresses":[["x",0]],"hi":1.0,"kind":"design","lo":0.0,"name":"a","samples":null,"unit":"m"},{"addresses":[["y"]],"hi":4.0,"kind":"design","lo":2.0,"name":"b","samples":null,"unit":"1"}]},"key":"11ba6d15ccd5e72f5a472a5eb7b60c7e17a718064a751b4c0fbe63e6d58ae057","kind":"study_context","schema":3}',
    '{"data":{"name":"y","unit":"V"},"key":"y","kind":"study_reading","schema":3}',
    '{"data":{"who":"test"},"key":"ce1cdd781e0c669b11ec7bc14a129d3538b316d22428478e6b8a383b7668e81c","kind":"study_provenance","schema":3}',
    '{"data":{"kind":"FeedbackDesign","slot":0},"key":"strategy:0","kind":"study_strategy","schema":3}',
    '{"data":{"groups":["g","h"],"order":"C","points":[{"a":0.25,"b":3.0},{"a":0.5,"b":2.5}],"shape":[2]},"key":"strategy:0:batch:0","kind":"study_design","schema":3}',
    '{"data":{"groups":[],"order":"C","points":[{"a":0.75,"b":4.0}],"shape":[]},"key":"strategy:0:batch:1","kind":"study_design","schema":3}',
]


def _request(path: Path, design, publisher: Publisher | None = None, **kw) -> Request:
    return Request(
        SPACE,
        AXES,
        READINGS,
        publisher or Publisher(),
        Codec(),
        design,
        {'case': 'c', 'n': 1},
        path,
        **kw,
    )


def _feedback() -> FeedbackDesign:
    initial = FixedDesign(((0.25, 3.0), (0.5, 2.5)), shape=(2,), groups=('g', 'h'))
    return FeedbackDesign(initial, lambda _rows: FixedDesign(((0.75, 4.0),)), 2)


def test_a_feedback_study_writes_the_motronics_study_events_byte_for_byte(tmp_path: Path) -> None:
    path = tmp_path / 'r.jsonl'
    publisher = Publisher()
    outcome = study(_request(path, _feedback(), publisher, provenance={'who': 'test'}))
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if '"kind":"study_' in line]
    assert lines == GOLDEN
    assert publisher.asked == [(0.25, 3.0), (0.5, 2.5), (0.75, 4.0)]
    assert [row.y for row in outcome.result.rows] == [3.25, 3.0, 4.75]


def test_a_repeated_study_reuses_its_slot_and_a_changed_layout_is_refused(tmp_path: Path) -> None:
    path = tmp_path / 'r.jsonl'
    study(_request(path, _feedback()))
    study(_request(path, _feedback()))
    strategies = [e.key for e in Record(path).replay() if e.kind == 'study_strategy']
    assert strategies == ['strategy:0']
    with pytest.raises(ValueError, match='layout'):
        study(_request(path, FeedbackDesign(FixedDesign(((0.25, 3.0),)), lambda _rows: None, 1)))
    with pytest.raises(ValueError, match='identity'):
        study(Request(SPACE, AXES, READINGS, Publisher(), Codec(), _feedback(), {'case': 'd'}, path))


def test_a_sampled_grid_keeps_its_shape_and_a_fixed_design_refuses_a_stranger() -> None:
    assert len(FixedDesign(((0.1, 2.0), (0.2, 3.0))).points) == 2
    with pytest.raises(ValueError, match='duplicate'):
        FixedDesign(((0.1, 2.0), (0.1, 2.0)))
    with pytest.raises(ValueError, match='shape'):
        FixedDesign(((0.1, 2.0),), shape=(2,))
    assert isinstance(SampleDesign, type)


def test_a_second_writer_on_one_record_is_refused(tmp_path: Path) -> None:
    path = tmp_path / 'r.jsonl'
    with study_writer(path), pytest.raises(OSError, match='another study'), study_writer(path):
        pass


def test_the_study_record_names_the_request_layout_as_json(tmp_path: Path) -> None:
    path = tmp_path / 'r.jsonl'
    study(_request(path, FixedDesign(((0.5, 3.0),))))
    design = next(json.loads(line) for line in path.read_text().splitlines() if '"study_design"' in line)
    assert design['data'] == {'groups': [], 'order': 'C', 'points': [{'a': 0.5, 'b': 3.0}], 'shape': []}
