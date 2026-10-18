"""Normalization enters frozen records through construction, including dataclass replacement."""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from optimi_lab.core.errors import Refusal
from optimi_lab.core.sampling import SampleKind, SampleSpec
from optimi_lab.problem import BlackBox, Objective, Real, Relation, Scale, Sense


@pytest.mark.parametrize(
    ('record', 'field', 'given', 'expected'),
    [
        (Objective('torque'), 'sense', 'maximize', Sense.MAXIMIZE),
        (Real('f', 1.0, 10.0), 'scale', 'log', Scale.LOG),
        (BlackBox('t', '<=', 1.0), 'relation', '>=', Relation.AT_LEAST),
        (SampleSpec(), 'kind', 'poisson_disk', SampleKind.POISSON_DISK),
    ],
)
def test_replacement_reenters_normalized_frozen_construction(record, field, given, expected):
    updated = replace(record, **{field: given})
    assert asdict(updated)[field] == expected
    assert updated == replace(updated)


def test_a_declaration_cannot_be_assigned_after_construction():
    real = Real('f', 1.0, 10.0)
    with pytest.raises(FrozenInstanceError):
        real.lo = 2.0


def test_replacement_is_refused_where_construction_is():
    with pytest.raises(Refusal, match='lo > 0'):
        replace(Real('f', 1.0, 10.0), lo=0.0, scale='log')
