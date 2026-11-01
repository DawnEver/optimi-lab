"""D1-D5: the four variable kinds, their unit-cube maps, and ``active_if``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optimi_lab.core import SampleKind, SampleSpec
from optimi_lab.core.errors import Refusal
from optimi_lab.problem import Categorical, Field, Integer, Real, Scale, Space, When


def test_d1_real_maps_linearly_and_logarithmically() -> None:
    linear = Real('r', 2.0, 4.0, unit='mm')
    assert linear.encode(3.0).tolist() == [0.5]
    assert linear.decode(np.array([0.25])) == 2.5
    log = Real('f', 1.0, 100.0, unit='Hz', scale='log')
    assert log.scale is Scale.LOG
    assert log.encode(10.0).tolist() == pytest.approx([0.5])
    assert log.decode(np.array([0.5])) == pytest.approx(10.0)
    assert log.decode(np.array([1.0])) == 100.0


def test_d1_planted_control_a_log_real_through_zero_is_refused() -> None:
    with pytest.raises(Refusal, match='lo > 0'):
        Real('f', 0.0, 1.0, scale=Scale.LOG)
    with pytest.raises(Refusal, match='lo < hi'):
        Real('r', 1.0, 1.0)
    with pytest.raises(Refusal, match='scale'):
        Real('r', 0.0, 1.0, scale='cubic')


def test_d2_integer_rounds_onto_its_step_grid() -> None:
    turns = Integer('turns', 10, 20, step=2)
    assert turns.levels == (10, 12, 14, 16, 18, 20)
    assert turns.decode(np.array([0.33])) == 14
    assert turns.contains(12)
    assert not turns.contains(13)
    assert not turns.contains(True)


def test_d2_planted_control_a_step_that_misses_hi_is_refused() -> None:
    with pytest.raises(Refusal, match='divides'):
        Integer('turns', 10, 20, step=3)
    with pytest.raises(Refusal, match='integer'):
        Integer('turns', 1.5, 3)


def test_d3_categorical_has_no_metric_until_embedded() -> None:
    plain = Categorical('steel', ('M19', 'M270', 'NO20'))
    assert plain.width == 0
    with pytest.raises(Refusal, match='embedding='):
        plain.encode('M19')
    embedded = Categorical('steel', ('M19', 'M270', 'NO20'), embedding=((0.0, 1.0), (1.0, 1.0), (0.5, 0.0)))
    assert embedded.width == 2
    assert embedded.encode('NO20').tolist() == [0.5, 0.0]
    assert embedded.decode(np.array([0.9, 0.8])) == 'M270'


def test_d3_planted_control_an_embedding_that_merges_two_levels_is_refused() -> None:
    with pytest.raises(Refusal, match='two levels to one point'):
        Categorical('steel', ('a', 'b'), embedding=((1.0,), (1.0,)))
    with pytest.raises(Refusal, match='coordinate row per level'):
        Categorical('steel', ('a', 'b', 'c'), embedding=((1.0,), (2.0,)))
    with pytest.raises(Refusal, match='distinct'):
        Categorical('steel', ('a', 'a'))


def test_d4_field_carries_cells_and_a_neighbourhood() -> None:
    density = Field('rho', 4, 0.0, 1.0, neighbors=((0, 1), (1, 2), (2, 3)))
    assert density.width == 4
    assert density.contains([0.0, 0.5, 1.0, 0.2])
    assert not density.contains([0.0, 0.5, 1.5, 0.2])
    assert density.decode(np.array([0.0, 0.5, 2.0, -1.0])) == (0.0, 0.5, 1.0, 0.0)


def test_d4_planted_control_a_neighbour_outside_the_field_is_refused() -> None:
    with pytest.raises(Refusal, match='outside cells'):
        Field('rho', 3, 0.0, 1.0, neighbors=((0, 3),))
    with pytest.raises(Refusal, match='outside cells'):
        Field('rho', 3, 0.0, 1.0, neighbors=((1, 1),))


def _switched() -> Space:
    return Space((
        Categorical('magnet', ('none', 'bar')),
        Real('magnet_width', 1.0, 5.0, unit='mm', active_if=When('magnet', ('bar',))),
        Real('gap', 0.5, 1.0, unit='mm'),
    ))


def test_d5_an_inactive_variable_is_not_part_of_the_design() -> None:
    space = _switched()
    without = space.canonical({'magnet': 'none', 'gap': 0.7, 'magnet_width': 3.0})
    assert without == {'gap': 0.7, 'magnet': 'none'}
    assert space.canonical({'magnet': 'none', 'gap': 0.7}) == without
    assert space.active({'magnet': 'bar', 'gap': 0.7, 'magnet_width': 2.0}) == ('magnet', 'magnet_width', 'gap')


def test_d5_planted_control_a_missing_active_value_and_an_unknown_parent_are_refused() -> None:
    with pytest.raises(Refusal, match="'magnet_width'"):
        _switched().canonical({'magnet': 'bar', 'gap': 0.7})
    with pytest.raises(Refusal, match='declared earlier'):
        Space((Real('w', 1.0, 2.0, active_if=When('magnet', ('bar',))), Categorical('magnet', ('none', 'bar'))))
    with pytest.raises(Refusal, match='not a level'):
        Space((Categorical('magnet', ('none', 'bar')), Real('w', 1.0, 2.0, active_if=When('magnet', ('rod',)))))
    with pytest.raises(Refusal, match='discrete'):
        Space((Real('g', 0.0, 1.0), Real('w', 1.0, 2.0, active_if=When('g', (0.5,)))))


@pytest.mark.parametrize('kind', list(SampleKind))
def test_d6_an_empty_space_is_the_empty_product_and_samples_zero_width_rows(kind: SampleKind) -> None:
    empty = Space(())
    assert (len(empty), empty.width, empty.names) == (0, 0, ())
    assert empty.natural_sample(SampleSpec(kind=kind, n_samples=5)).shape == (5, 0)
    assert empty.canonical({}) == {}
    assert empty.decode(np.zeros(0)) == {}


def test_d6_planted_control_an_empty_space_keeps_every_other_refusal() -> None:
    with pytest.raises(Refusal, match='unique'):
        Space((Real('x', 0.0, 1.0), Real('x', 0.0, 2.0)))
    with pytest.raises(Refusal, match='Real, Integer'):
        Space(('x',))
    with pytest.raises(Refusal):
        Real('x', 1.0, 0.0)
    with pytest.raises(Refusal, match='no variable'):
        Space(()).canonical({'x': 1.0})


def test_d6_space_round_trips_through_the_unit_cube() -> None:
    space = Space((
        Real('r', 1.0, 100.0, scale='log'),
        Integer('n', 0, 10),
        Categorical('c', ('a', 'b'), embedding=((0.0,), (1.0,))),
        Field('rho', 2, 0.0, 2.0),
    ))
    assert space.width == 5
    design = {'r': 10.0, 'n': 4, 'c': 'b', 'rho': (0.5, 2.0)}
    unit = space.encode(design)
    assert unit.tolist() == pytest.approx([0.5, 0.4, 1.0, 0.25, 1.0])
    decoded = space.decode(unit)
    assert decoded['n'] == 4 and decoded['c'] == 'b' and decoded['rho'] == (0.5, 2.0)
    assert math.isclose(decoded['r'], 10.0)


def test_d6_inactive_dimensions_encode_as_nan_and_decode_ignores_them() -> None:
    space = _switched()
    with pytest.raises(Refusal, match='no metric'):
        space.encode({'magnet': 'none', 'gap': 0.7})
    embedded = Space((
        Categorical('magnet', ('none', 'bar'), embedding=((0.0,), (1.0,))),
        Real('w', 1.0, 5.0, active_if=When('magnet', ('bar',))),
    ))
    unit = embedded.encode({'magnet': 'none'})
    assert unit[0] == 0.0 and np.isnan(unit[1])
    assert embedded.decode(unit) == {'magnet': 'none'}


def test_d6_planted_control_a_stranger_and_an_out_of_range_value_are_refused() -> None:
    space = Space((Real('x', 0.0, 1.0),))
    with pytest.raises(Refusal, match="no variable named 'y'"):
        space.canonical({'x': 0.5, 'y': 1.0})
    with pytest.raises(Refusal, match='outside'):
        space.canonical({'x': 1.5})
    with pytest.raises(Refusal, match='unique'):
        Space((Real('x', 0.0, 1.0), Real('x', 0.0, 2.0)))
    with pytest.raises(Refusal, match='Real, Integer, Categorical or Field'):
        Space(('x',))  # type: ignore[arg-type]
