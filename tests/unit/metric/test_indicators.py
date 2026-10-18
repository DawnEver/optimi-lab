"""The kept L4 indicators against closed-form values (moved from the retired benchmark tests)."""

import numpy as np
import pytest

from optimi_lab.core.errors import Refusal
from optimi_lab.metric import hypervolume, hypervolume_2d, igd


def test_igd_is_zero_when_the_front_covers_the_reference():
    reference = np.column_stack([np.linspace(0, 1, 20), 1 - np.sqrt(np.linspace(0, 1, 20))])
    assert igd(reference, reference) == pytest.approx(0.0)


def test_an_empty_front_scores_inf_rather_than_zero():
    """A run that found nothing is the worst result, and by arithmetic it would read as the best."""
    assert igd(np.zeros((0, 2)), np.array([[0.0, 1.0]])) == float('inf')


def test_igd_grows_with_the_distance_to_the_reference():
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    near = np.array([[0.1, 1.0], [1.0, 0.1]])
    far = np.array([[0.5, 1.0], [1.0, 0.5]])
    assert 0.0 < igd(near, reference) < igd(far, reference)


def test_hypervolume_of_hand_built_staircases():
    unit = np.array([1.0, 1.0])
    # One point at the origin dominates the whole unit square.
    assert hypervolume_2d(np.array([[0.0, 0.0]]), unit) == pytest.approx(1.0)
    # Two steps: (0, 0.5) covers 1.0 x 0.5, then (0.5, 0) adds 0.5 x 0.5.
    assert hypervolume_2d(np.array([[0.0, 0.5], [0.5, 0.0]]), unit) == pytest.approx(0.75)


def test_a_point_that_does_not_dominate_the_reference_contributes_nothing():
    unit = np.array([1.0, 1.0])
    assert hypervolume_2d(np.array([[1.5, 1.5]]), unit) == 0.0


def test_hypervolume_refuses_a_third_objective_rather_than_reading_the_first_two():
    """A 3-objective hypervolume is a different algorithm, and a plausible wrong number is worse."""
    with pytest.raises(ValueError, match='2 objectives'):
        hypervolume_2d(np.zeros((3, 3)), np.array([1.0, 1.0, 1.0]))


def test_hypervolume_agrees_with_the_two_objective_sweep():
    """On two objectives the general recursion and the staircase sweep are one number."""
    rng = np.random.default_rng(3)
    ref = np.array([1.1, 1.1])
    for _ in range(5):
        front = rng.uniform(0.0, 1.2, (40, 2))
        assert hypervolume(front, ref) == pytest.approx(hypervolume_2d(front, ref), abs=1e-12)


def test_hypervolume_of_a_hand_built_three_objective_set():
    """Three unit-box corners at 0.5: inclusion-exclusion gives 3(0.5) - 3(0.25) + 0.125 = 0.875."""
    ref = np.ones(3)
    corners = np.array([[0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    assert hypervolume(corners, ref) == pytest.approx(0.875)
    assert hypervolume(np.vstack([corners, [0.6, 0.6, 0.6], [2.0, 0.0, 0.0]]), ref) == pytest.approx(0.875)
    assert hypervolume(np.array([[1.5, 0.0, 0.0]]), ref) == 0.0


def test_hypervolume_of_a_four_objective_lattice_matches_its_cell_count():
    """Points on a quarter lattice dominate whole cells, so the exact volume is a count of cells."""
    rng = np.random.default_rng(7)
    front = rng.integers(0, 4, (30, 4)) / 4.0
    cells = np.stack(np.meshgrid(*[np.arange(4) / 4.0] * 4, indexing='ij'), axis=-1).reshape(-1, 4)
    covered = (front[None, :, :] <= cells[:, None, :]).all(axis=-1).any(axis=1)
    assert hypervolume(front, np.ones(4)) == pytest.approx(covered.sum() / 4.0**4)


def test_hypervolume_refuses_a_reference_of_another_width():
    with pytest.raises(Refusal, match='reference'):
        hypervolume(np.zeros((2, 3)), np.ones(2))
