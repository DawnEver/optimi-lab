"""Run a declared study under one writer, and file what it is in the record before any answer.

The record carries, ahead of the caller's own physical events: one ``study_context`` (the evaluation
identity, the axes as the caller spells them, the record schema), one ``study_reading`` per reading,
one ``study_evaluator`` per extra fidelity, ``study_provenance``, one ``study_strategy`` per strategy
slot, and per batch of an independent-point design a ``study_design`` keyed
``strategy:{slot}:batch:{i}``. A rerun on the same record re-reads them and refuses a different
study rather than mixing two.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from lab_commons.outcomes import Event, Evidence, Identity, Record

from optimi_lab.core import SampleKind
from optimi_lab.core.errors import refuse
from optimi_lab.problem import Space
from optimi_lab.study._lock import study_writer
from optimi_lab.study.engine import Searched
from optimi_lab.study.protocols import Answer, Codec, Filing, Publisher
from optimi_lab.study.rounds import searched_study
from optimi_lab.study.strategy import (
    EvaluationPolicy,
    FeedbackDesign,
    FixedDesign,
    SampleDesign,
    SearchStrategy,
    SurrogateRound,
    canonical_subject,
)

__all__ = ['RECORD_SCHEMA', 'FixedStudyResult', 'Request', 'StudyOutcome', 'fixed_design', 'study']

#: The ``physical_record_schema`` a study context declares.
RECORD_SCHEMA = 2


@dataclass(frozen=True)
class Request:
    """One study: the *space*, how the record names its *axes* and *readings*, who answers, and what is run.

    *axes* and *readings* are plain descriptors as the caller's codec spells them (each reading has a
    ``name``); they enter the record context verbatim. *publisher* answers independent points, *codec*
    files them and binds a search. *policy* applies to a :class:`SearchStrategy` only.
    """

    space: Space
    axes: tuple[Mapping[str, Any], ...]
    readings: tuple[Mapping[str, Any], ...]
    publisher: Publisher
    codec: Codec
    design: FixedDesign | SampleDesign | FeedbackDesign | SearchStrategy
    evaluation_identity: Mapping[str, object]
    record_path: Path
    policy: EvaluationPolicy | None = None
    evidence: Evidence = Evidence.OBSERVED
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evaluation_identity:
            refuse('a study requires an explicit nonempty evaluation_identity')
        object.__setattr__(self, 'evaluation_identity', canonical_subject(self.evaluation_identity))
        object.__setattr__(self, 'provenance', canonical_subject(self.provenance))
        object.__setattr__(self, 'axes', tuple(dict(axis) for axis in self.axes))
        object.__setattr__(self, 'readings', tuple(dict(reading) for reading in self.readings))
        names = [reading.get('name') for reading in self.readings]
        if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
            refuse('study readings must be nonempty and uniquely named')
        if self.policy is not None and not isinstance(self.design, SearchStrategy):
            refuse('a fixed/DoE publisher has no optimizer variation or fidelity policy')


@dataclass(frozen=True)
class FixedStudyResult:
    """A reading-only view of the filed rows; deliberately no front and no optimization scores."""

    request: Request
    record: Record
    designs: tuple[FixedDesign, ...]
    filing: Filing

    @property
    def rows(self) -> tuple[object, ...]:
        return tuple(
            row for design in self.designs for point in design.points if (row := self.filing.row(point)) is not None
        )


@dataclass(frozen=True)
class StudyOutcome:
    """The study's *result*, and for a search its last surrogate *fit*, *steering* and *rounds*."""

    result: FixedStudyResult | Searched
    fit: object = None
    steering: object = None
    rounds: tuple[SurrogateRound, ...] = ()


def fixed_design(space: Space, design: FixedDesign | SampleDesign) -> FixedDesign:
    """*design* as fixed natural points; a sampled grid keeps its shape when the sample fills it."""
    if isinstance(design, FixedDesign):
        if any(len(point) != len(space.names) for point in design.points):
            refuse('a fixed point belongs to a different design space')
        return design
    if not isinstance(design, SampleDesign):
        msg = 'a study accepts independent fixed points or a sampling operator, not a stateful march'
        raise TypeError(msg)
    matrix = space.natural_sample(design.spec)
    shape: tuple[int, ...] = ()
    if design.spec.kind is SampleKind.GRID and design.spec.anchor is None:
        # Layout from the actual sample, never from a second sampling rule.
        counts = tuple(len(np.unique(matrix[:, index])) for index in range(len(space.names)))
        if prod(counts) == len(matrix):
            shape = counts
    return FixedDesign(tuple(tuple(row) for row in matrix), shape=shape)


def _evaluate(request: Request, design: FixedDesign, filing: Filing) -> None:
    fresh = tuple(point for point in design.points if not filing.reusable(point))
    pending = set(fresh)
    lock = Lock()

    def completed(answer: Answer) -> None:
        with lock:
            if answer.point not in pending:
                refuse('a publisher answered an unrequested or already completed point')
            filing.file(answer)
            pending.remove(answer.point)

    if fresh:
        for point in fresh:
            filing.submit(point)
        request.publisher.publish(fresh, completed)
    if pending:
        refuse('a publisher returned without completing every requested point')


def _context(request: Request, record: Record) -> None:
    context = json.loads(
        json.dumps({
            'evaluation_identity': json.loads(Identity.of(request.evaluation_identity).canonical)['subject'],
            'space': list(request.axes),
            'physical_record_schema': RECORD_SCHEMA,
        })
    )
    previous = [event for event in record.replay() if event.kind == 'study_context']
    if previous and any(event.data != context for event in previous):
        refuse('the record belongs to another study evaluation identity or reading declaration')
    if not previous and any(record.replay()):
        refuse('the record has no study identity context; it cannot be reused implicitly')
    if not previous:
        record.append(Event('study_context', Identity.of(context).key, data=context))
    declared = {event.key: event.data for event in record.replay() if event.kind == 'study_reading'}
    for reading in request.readings:
        if reading['name'] in declared and declared[reading['name']] != reading:
            refuse(f'physical reading {reading["name"]!r} already declares another unit; conversion is not implicit')
        record.append(Event('study_reading', reading['name'], data=reading))


def study(request: Request) -> StudyOutcome:
    """Execute a declared study; every completed answer is filed before this returns."""
    with study_writer(request.record_path):
        return _study(request, Record(request.record_path))


def _study(request: Request, record: Record) -> StudyOutcome:
    _context(request, record)
    for fidelity in request.policy.fidelities if request.policy is not None else ():
        identity = Identity.of(fidelity.evaluation_identity)
        record.append(Event('study_evaluator', identity.key, data=json.loads(identity.canonical)['subject']))
    if request.provenance:
        provenance = json.loads(Identity.of(request.provenance).canonical)['subject']
        record.append(Event('study_provenance', Identity.of(provenance).key, data=provenance))
    filing = request.codec.filing(record)
    kind = type(request.design).__name__
    strategies = [event for event in record.replay() if event.kind == 'study_strategy']
    slot = strategies[-1].data['slot'] if strategies and strategies[-1].data['kind'] == kind else len(strategies)
    record.append(Event('study_strategy', f'strategy:{slot}', data={'kind': kind, 'slot': slot}))
    if isinstance(request.design, SearchStrategy):
        bound = request.codec.searching(filing, record)
        initial = fixed_design(request.space, request.design.initial)
        searched, fit, steering, rounds = searched_study(
            bound, request.design, request.policy or EvaluationPolicy(), np.array(initial.points), request.evidence
        )
        return StudyOutcome(searched, fit, steering, rounds)
    feedback = request.design if isinstance(request.design, FeedbackDesign) else None
    initial = request.design.initial if feedback is not None else request.design
    if not isinstance(initial, (FixedDesign, SampleDesign)):
        msg = 'an independent-point study needs a fixed/sample design; a stateful march belongs to its solver'
        raise TypeError(msg)
    design = fixed_design(request.space, initial)
    designs: list[FixedDesign] = []
    for index in range(feedback.max_batches if feedback is not None else 1):
        layout = {
            'points': [dict(zip(request.space.names, point, strict=True)) for point in design.points],
            'shape': list(design.shape),
            'groups': list(design.groups),
            'order': design.order,
        }
        key = f'strategy:{slot}:batch:{index}'
        previous = [event for event in record.replay() if event.kind == 'study_design' and event.key == key]
        if previous and any(event.data != layout for event in previous):
            refuse('the recorded sweep layout/order differs from the requested design')
        record.append(Event('study_design', key, data=layout))
        _evaluate(request, design, filing)
        designs.append(design)
        if feedback is None or index + 1 == feedback.max_batches:
            break
        next_design = feedback.propose(FixedStudyResult(request, record, tuple(designs), filing).rows)
        if next_design is None:
            break
        design = fixed_design(request.space, next_design)
    return StudyOutcome(FixedStudyResult(request, record, tuple(designs), filing))
