"""The declared embedding of a conditional, categorical space: closed-form rows, and every decode legal.

The space is a switch ``kind in (flat, bowl, ramp)``, an always-active ``x`` and a radius ``r`` that
exists only for ``bowl``. Every expected row below is written out by hand from the declaration.
"""

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.problem import Categorical, Embedding, Encoding, Inactive, Real, Space, When, needs_embedding

SPACE = Space((
    Categorical('kind', ('flat', 'bowl', 'ramp')),
    Real('x', 0.0, 1.0),
    Real('r', 0.0, 1.0, active_if=When('kind', ('bowl',))),
))
FLAT = {'kind': 'flat', 'x': 0.25}
BOWL = {'kind': 'bowl', 'x': 0.25, 'r': 0.75}


def test_one_hot_and_neutral_fill_an_inactive_variable_with_the_middle_of_the_cube():
    embedding = Embedding(SPACE)
    assert embedding.width == 5
    np.testing.assert_array_equal(embedding.encode(FLAT), [1.0, 0.0, 0.0, 0.25, 0.5])
    np.testing.assert_array_equal(embedding.encode(BOWL), [0.0, 1.0, 0.0, 0.25, 0.75])


def test_the_indicator_embedding_appends_one_activity_column_per_conditional_variable():
    embedding = Embedding(SPACE, inactive=Inactive.INDICATOR)
    assert embedding.width == 6
    np.testing.assert_array_equal(embedding.encode(FLAT), [1.0, 0.0, 0.0, 0.25, 0.5, 0.0])
    np.testing.assert_array_equal(embedding.encode(BOWL), [0.0, 1.0, 0.0, 0.25, 0.75, 1.0])


def test_the_ordinal_encoding_is_one_column_in_the_declared_order_and_neutral_is_declared_per_name():
    embedding = Embedding(SPACE, categorical=Encoding.ORDINAL, neutral={'r': 0.0})
    assert embedding.width == 3
    np.testing.assert_array_equal(embedding.encode(FLAT), [0.0, 0.25, 0.0])
    np.testing.assert_array_equal(embedding.encode({'kind': 'ramp', 'x': 1.0}), [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(embedding.encode(BOWL), [0.5, 0.25, 0.75])


def test_a_declared_categorical_embedding_is_kept():
    space = Space((Categorical('c', ('a', 'b'), embedding=((0.0,), (3.0,))), Real('x', 0.0, 1.0)))
    assert Embedding(space).space == space


@pytest.mark.parametrize('inactive', list(Inactive))
@pytest.mark.parametrize('categorical', list(Encoding))
def test_every_row_decodes_to_a_legal_design(categorical, inactive):
    embedding = Embedding(SPACE, categorical=categorical, inactive=inactive)
    for row in np.random.default_rng(0).random((200, embedding.width)):
        design = embedding.decode(row)
        assert SPACE.canonical(design) == design  # active variables only, every value declared
        assert ('r' in design) == (design['kind'] == 'bowl')


@pytest.mark.parametrize('inactive', list(Inactive))
def test_a_design_round_trips_through_its_row(inactive):
    embedding = Embedding(SPACE, inactive=inactive)
    for design in (FLAT, BOWL, {'kind': 'ramp', 'x': 0.0}):
        assert embedding.decode(embedding.encode(design)) == design


def test_a_neutral_value_or_a_switched_off_column_is_never_read_back():
    embedding = Embedding(SPACE, inactive=Inactive.INDICATOR)
    row = embedding.encode(FLAT)
    row[4], row[5] = 0.9, 1.0  # a variation moved the switched-off radius and its indicator
    assert embedding.decode(row) == FLAT


def test_what_needs_an_embedding_is_named():
    assert needs_embedding(SPACE) == ['kind', 'r']
    assert needs_embedding(Space((Real('x', 0.0, 1.0),))) == []


def test_a_space_that_needs_an_embedding_is_refused_without_one_naming_the_setting():
    with pytest.raises(Refusal, match='embedding='):
        Embedding.declared(SPACE, None)
    plain = Space((Real('x', 0.0, 1.0),))
    assert Embedding.declared(plain, None).width == 1


def test_an_embedding_of_another_space_is_refused():
    with pytest.raises(Refusal, match='another space'):
        Embedding.declared(SPACE, Embedding(Space((Real('x', 0.0, 1.0),))))


@pytest.mark.parametrize(
    ('settings', 'match'),
    [
        ({'neutral': {'x': 0.5}}, 'not conditional'),
        ({'neutral': 1.5}, r'\[0, 1\]'),
        ({'categorical': 'binary'}, 'one_hot'),
        ({'inactive': 'drop'}, 'indicator'),
    ],
)
def test_an_undeclared_choice_is_refused(settings, match):
    with pytest.raises(Refusal, match=match):
        Embedding(SPACE, **settings)


def test_a_row_of_the_wrong_width_is_refused():
    with pytest.raises(Refusal, match='shape'):
        Embedding(SPACE).decode(np.zeros(4))
