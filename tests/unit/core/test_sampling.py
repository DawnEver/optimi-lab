"""Every sample kind fills exactly its batch, the grid is a full factorial, and the anchor is row 0."""

import numpy as np
import pytest

from optimi_lab import Refusal, SampleKind, SampleSpec, Variable, VariableSet, sample
from optimi_lab.core.sampling import MAX_GRID_POINTS, grid_counts

SPACE = VariableSet((Variable('a', 0.0, 1.0), Variable('b', -2.0, 2.0)))


@pytest.mark.parametrize('kind', [SampleKind.LATIN_HYPERCUBE, SampleKind.POISSON_DISK])
def test_a_drawn_kind_fills_its_batch_inside_the_bounds_and_repeats_under_one_seed(kind):
    spec = SampleSpec(kind=kind, n_samples=12, seed=4)
    points = sample(SPACE, spec)
    assert points.shape == (12, 2)
    assert np.all(points >= SPACE.lower_bounds)
    assert np.all(points <= SPACE.upper_bounds)
    assert np.array_equal(points, sample(SPACE, spec))


def test_the_default_spec_draws_the_default_count():
    assert sample(SPACE).shape == (32, 2)


def test_the_grid_is_the_full_factorial_of_its_levels_in_declared_order():
    points = sample(SPACE, SampleSpec(kind='grid', levels=(2, 3)))
    expected = [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0], [1.0, -2.0], [1.0, 0.0], [1.0, 2.0]]
    assert points.tolist() == expected


def test_a_grid_without_levels_factorizes_its_count():
    assert grid_counts(2, 12) == (4, 3)
    assert sample(SPACE, SampleSpec(kind='grid', n_samples=12)).shape == (12, 2)


def test_a_grid_whose_count_disagrees_with_its_levels_is_refused():
    with pytest.raises(Refusal, match='describe a grid of 6'):
        sample(SPACE, SampleSpec(kind='grid', n_samples=7, levels=(2, 3)))


def test_a_grid_needs_one_level_count_per_variable_of_at_least_two():
    with pytest.raises(Refusal, match='one count per variable'):
        sample(SPACE, SampleSpec(kind='grid', levels=(2, 3, 4)))
    with pytest.raises(Refusal, match='at least 2'):
        SampleSpec(kind='grid', levels=(1, 3))


def test_a_grid_over_the_ceiling_is_refused_before_a_point_is_built():
    levels = (MAX_GRID_POINTS, 2)
    with pytest.raises(Refusal, match=f'ceiling of {MAX_GRID_POINTS}'):
        sample(SPACE, SampleSpec(kind='grid', levels=levels))


def test_levels_belong_to_the_grid_alone():
    with pytest.raises(Refusal, match="belongs to sample kind 'grid'"):
        SampleSpec(kind='latin_hypercube', levels=(2, 2))


def test_the_anchor_is_row_zero_of_a_grid_and_counts_in_the_batch():
    points = sample(SPACE, SampleSpec(kind='grid', levels=(2, 2), anchor=(0.5, 0.25)))
    assert points.shape == (5, 2)
    assert points[0].tolist() == [0.5, 0.25]
    assert points[1:].tolist() == [[0.0, -2.0], [0.0, 2.0], [1.0, -2.0], [1.0, 2.0]]


@pytest.mark.parametrize('kind', [SampleKind.LATIN_HYPERCUBE, SampleKind.POISSON_DISK])
def test_the_anchor_is_row_zero_of_a_drawn_batch_of_the_requested_size(kind):
    points = sample(SPACE, SampleSpec(kind=kind, n_samples=9, anchor=(0.5, 0.25)))
    assert points.shape == (9, 2)
    assert points[0].tolist() == [0.5, 0.25]


def test_an_anchor_outside_the_bounds_is_refused_naming_the_variable():
    with pytest.raises(Refusal, match="variable 'b'"):
        sample(SPACE, SampleSpec(kind='latin_hypercube', n_samples=4, anchor=(0.5, 3.0)))


def test_an_anchor_of_the_wrong_width_is_refused():
    with pytest.raises(Refusal, match='one coordinate per variable'):
        sample(SPACE, SampleSpec(n_samples=4, anchor=(0.5,)))


def test_an_anchor_leaves_room_for_at_least_one_drawn_point():
    with pytest.raises(Refusal, match='n_samples'):
        SampleSpec(n_samples=1, anchor=(0.5, 0.0))


def test_an_imported_matrix_is_the_whole_batch_so_it_takes_no_anchor():
    with pytest.raises(Refusal, match='anchor'):
        SampleSpec(kind='import', n_samples=1, matrix=np.zeros((1, 2)), anchor=(0.5, 0.0))


def test_an_imported_matrix_is_returned_as_a_copy():
    matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    points = sample(SPACE, SampleSpec(kind='import', n_samples=2, matrix=matrix))
    assert points.tolist() == matrix.tolist()
    points[0, 0] = 9.0
    assert matrix[0, 0] == 0.1


def test_an_imported_matrix_of_the_wrong_shape_or_with_a_nan_is_refused():
    with pytest.raises(Refusal, match='shape'):
        sample(SPACE, SampleSpec(kind='import', n_samples=3, matrix=np.zeros((2, 2))))
    with pytest.raises(Refusal, match='finite'):
        sample(SPACE, SampleSpec(kind='import', n_samples=1, matrix=np.array([[np.nan, 0.0]])))
