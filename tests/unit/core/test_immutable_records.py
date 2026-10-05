"""Normalization enters frozen records through construction, including dataclass replacement."""

from dataclasses import MISSING, FrozenInstanceError, asdict, fields, is_dataclass, replace
from inspect import Parameter, signature

import numpy as np
import pytest

from optimi_lab import Evaluation, Objective, ObjectiveSet, SampleSpec, Variable, VariableSet
from optimi_lab.core.outcomes import Outcome
from optimi_lab.core.sampling import SampleKind
from optimi_lab.core.space import Direction


@pytest.mark.parametrize(
    ('record', 'field', 'given', 'expected'),
    [
        (Objective('torque'), 'direction', 'maximize', Direction.MAXIMIZE),
        (VariableSet([Variable('x', 0, 1)]), 'variables', [Variable('y', 0, 2)], (Variable('y', 0, 2),)),
        (ObjectiveSet([Objective('torque')]), 'objectives', [Objective('loss')], (Objective('loss'),)),
        (SampleSpec(kind='uniform'), 'steps', [2, 3], (2, 3)),
        (SampleSpec(), 'kind', 'poisson_disk', SampleKind.POISSON_DISK),
    ],
)
def test_replacement_reenters_normalized_frozen_construction(record, field, given, expected):
    updated = replace(record, **{field: given})
    assert is_dataclass(updated)
    assert isinstance(asdict(updated)[field], type(expected))
    assert asdict(updated) == asdict(replace(record, **{field: expected}))
    assert updated == replace(updated)
    assert hash(updated) == hash(replace(updated))
    with pytest.raises(FrozenInstanceError):
        if isinstance(updated, Objective):
            updated.direction = Direction.MINIMIZE
        elif isinstance(updated, VariableSet):
            updated.variables = ()
        elif isinstance(updated, ObjectiveSet):
            updated.objectives = ()
        else:
            updated.seed = 99


def test_evaluation_replacement_preserves_cell_outcomes_and_identity_equality():
    original = Evaluation([[1]], [[2]])
    updated = replace(original, values=[[7]], outcomes=[['error']])
    assert updated.outcomes[0, 0] is Outcome.ERROR
    assert np.isnan(updated.values[0, 0])
    assert original.values[0, 0] == 2
    assert updated is not replace(updated)
    assert updated != replace(updated)
    assert [field.name for field in fields(updated)] == ['inputs', 'values', 'outcomes']
    with pytest.raises(FrozenInstanceError):
        updated.values = np.array([[0]])


@pytest.mark.parametrize(
    ('record_type', 'parameters'),
    [
        (Evaluation, ('inputs', 'values', 'outcomes')),
        (Objective, ('name', 'direction', 'operating_point')),
        (VariableSet, ('variables',)),
        (ObjectiveSet, ('objectives',)),
        (SampleSpec, ('kind', 'n_samples', 'seed', 'steps', 'radius', 'matrix')),
    ],
)
def test_public_constructor_signatures_and_dataclass_fields_stay_declared(record_type, parameters):
    assert tuple(signature(record_type).parameters) == parameters
    assert all(
        parameter.kind is Parameter.POSITIONAL_OR_KEYWORD for parameter in signature(record_type).parameters.values()
    )
    assert tuple(field.name for field in fields(record_type)) == parameters
    for field in fields(record_type):
        expected = Parameter.empty if field.default is MISSING else field.default
        assert signature(record_type).parameters[field.name].default == expected
