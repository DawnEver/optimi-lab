"""Every sample kind fills exactly its batch, the grid is a full factorial, and the anchor is row 0."""

from pathlib import Path

import numpy as np
import pytest

from optimi_lab.core import sampling
from optimi_lab.core.errors import Refusal
from optimi_lab.core.sampling import MAX_GRID_POINTS, SampleKind, SampleSpec, grid_counts, sample

SPACE = 2  # unit-cube width
DRAWN = [kind for kind in SampleKind if kind is not SampleKind.GRID]


@pytest.mark.parametrize('kind', DRAWN)
def test_a_drawn_kind_fills_its_batch_inside_the_bounds_and_repeats_under_one_seed(kind):
    spec = SampleSpec(kind=kind, n_samples=12, seed=4)
    points = sample(SPACE, spec)
    assert points.shape == (12, 2)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)
    assert np.array_equal(points, sample(SPACE, spec))


def test_the_default_spec_draws_the_default_count():
    assert sample(SPACE).shape == (32, 2)


def test_the_grid_is_the_full_factorial_of_its_levels_in_declared_order():
    points = sample(SPACE, SampleSpec(kind='grid', levels=(2, 3)))
    expected = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]
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
    assert points[1:].tolist() == [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


@pytest.mark.parametrize('kind', DRAWN)
def test_the_anchor_is_row_zero_of_a_drawn_batch_of_the_requested_size(kind):
    points = sample(SPACE, SampleSpec(kind=kind, n_samples=9, anchor=(0.5, 0.25)))
    assert points.shape == (9, 2)
    assert points[0].tolist() == [0.5, 0.25]


def test_an_anchor_outside_the_unit_cube_is_refused_naming_the_coordinate():
    with pytest.raises(Refusal, match=r'coordinates \[1\]'):
        sample(SPACE, SampleSpec(kind='latin_hypercube', n_samples=4, anchor=(0.5, 3.0)))


def test_an_anchor_of_the_wrong_width_is_refused():
    with pytest.raises(Refusal, match='one coordinate per unit-cube dimension'):
        sample(SPACE, SampleSpec(n_samples=4, anchor=(0.5,)))


def test_an_anchor_leaves_room_for_at_least_one_drawn_point():
    with pytest.raises(Refusal, match='n_samples'):
        SampleSpec(n_samples=1, anchor=(0.5, 0.0))


def test_the_import_kind_is_retired():
    with pytest.raises(Refusal, match='latin_hypercube'):
        SampleSpec(kind='import')


def test_every_kind_is_drawn_without_scipy_stats():
    """R1: the samplers are self-implemented; scipy.stats.qmc is a test oracle only."""
    source = Path(sampling.__file__).read_text(encoding='utf-8')
    assert 'scipy' not in source


def test_a_poisson_disk_set_keeps_its_radius_between_every_pair():
    """D69: the defining property, in unit-cube units: no two points closer than the radius."""
    points = sample(SPACE, SampleSpec(kind='poisson_disk', n_samples=20, radius=0.1, seed=2))
    gaps = np.linalg.norm(points[:, None] - points[None], axis=-1)[np.triu_indices(20, 1)]
    assert gaps.min() >= 0.1


def test_a_poisson_radius_no_batch_fits_is_refused_naming_the_count_reached():
    with pytest.raises(Refusal, match='admitted'):
        sample(SPACE, SampleSpec(kind='poisson_disk', n_samples=50, radius=0.5, seed=0))


@pytest.mark.parametrize('kind', [SampleKind.SOBOL, SampleKind.HALTON])
def test_a_sequence_kind_is_a_scrambled_low_discrepancy_set(kind):
    from scipy.stats import qmc  # noqa: PLC0415 -- the test oracle

    drawn = sample(4, SampleSpec(kind=kind, n_samples=64, seed=1))
    assert qmc.discrepancy(drawn) < qmc.discrepancy(np.random.default_rng(1).random((64, 4)))
