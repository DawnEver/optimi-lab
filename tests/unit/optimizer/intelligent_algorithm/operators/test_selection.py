"""Selection operators test module

Tests functionality, parameter validation and edge cases for selection operators
"""

import numpy as np
import pytest
from pydantic import ValidationError

from optimi_lab.intelligent_algorithm.operators.selection import (
    ParetoCrowdingSelection,
    ParetoRefSelection,
    ParetoSelection,
    RandomSelection,
    SelectionBase,
    TournamentSelection,
    associate_to_reference_points,
    calc_crowding_distance,
    generate_das_dennis_reference_directions,
    normalize_objectives,
)
from optimi_lab.utils.exceptions import ParameterException


class MockSelection(SelectionBase):
    """Mock selection operator for testing"""

    def do(self, fitness: np.ndarray) -> np.ndarray:
        return np.arange(min(self.n_selected, len(fitness)))


class TestSelectionBase:
    """Selection base tests"""

    def test_validation_and_abstract(self):
        """Test parameter validation and abstract methods"""
        # Valid values
        assert MockSelection(n_selected=2).n_selected == 2
        assert MockSelection(n_selected=100).n_selected == 100

        # Invalid values
        for invalid_n in [0, 1, -1]:
            with pytest.raises((ValidationError, ParameterException)):
                MockSelection(n_selected=invalid_n)

        # Abstract class cannot be instantiated directly
        with pytest.raises(TypeError):
            SelectionBase(n_selected=5)


class TestRandomSelection:
    """Random selection tests"""

    def test_random_selection(self):
        """Test random selection functionality"""
        selection = RandomSelection(n_selected=3)
        fitness = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [0.5, 0.5], [1.5, 1.5]])

        # Basic selection
        selected = selection.do(fitness)
        assert len(selected) == 3
        assert len(set(selected)) == 3  # no duplicates
        assert all(0 <= idx < len(fitness) for idx in selected)

        # Select all individuals
        selection_all = RandomSelection(n_selected=2)
        fitness_small = np.array([[1.0, 2.0], [2.0, 1.0]])
        selected_all = selection_all.do(fitness_small)
        assert len(selected_all) == 2
        assert set(selected_all) == {0, 1}

        # Insufficient population raises error
        with pytest.raises(ValueError, match='Cannot take a larger sample than population'):
            RandomSelection(n_selected=3).do(np.array([[1.0, 2.0]]))

        # Reproducibility
        np.random.seed(42)
        selected_a = selection.do(fitness)
        np.random.seed(42)
        selected_b = selection.do(fitness)
        np.testing.assert_array_equal(selected_a, selected_b)


class TestTournamentSelection:
    """Tournament selection tests"""

    def test_tournament_selection(self):
        """Test tournament selection"""
        # Parameter validation
        selection = TournamentSelection(n_selected=3, tournament_size=2)
        assert selection.tournament_size == 2

        for invalid_size in [1, 0]:
            with pytest.raises((ValidationError, ParameterException)):
                TournamentSelection(n_selected=3, tournament_size=invalid_size)

        # Basic tournament selection
        fitness = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        selected = selection.do(fitness)
        assert len(selected) == 3
        assert all(0 <= idx < len(fitness) for idx in selected)

        # Single-objective
        single_obj_selection = TournamentSelection(n_selected=2, tournament_size=2)
        single_fitness = np.array([[1.0], [0.5], [2.0], [0.1]])
        selected_single = single_obj_selection.do(single_fitness)
        assert len(selected_single) == 2

        # Test else branch: when fronts is empty
        # Create special case where non_dominated_sorting returns empty fronts
        special_fitness = np.array([[np.inf, np.inf], [1.0, 1.0], [2.0, 2.0]])
        selection_large = TournamentSelection(n_selected=2, tournament_size=3)
        selected_special = selection_large.do(special_fitness)
        assert len(selected_special) == 2

        # Test multiple non-dominated solutions covering crowding-distance selection branch
        balanced_fitness = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        balanced_selection = TournamentSelection(n_selected=3, tournament_size=4)
        selected_balanced = balanced_selection.do(balanced_fitness)
        assert len(selected_balanced) == 3


class TestParetoSelection:
    """Pareto selection tests"""

    def test_pareto_selection(self):
        """Test Pareto selection"""
        selection = ParetoSelection(n_selected=3)

        # Basic Pareto selection
        fitness = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0]])
        selected = selection.do(fitness)
        assert len(selected) == 3
        assert set(selected) == {0, 1, 2}  # first three are Pareto-optimal

        # Pareto front insufficient; need to select from other fronts
        selection_more = ParetoSelection(n_selected=4)
        selected_more = selection_more.do(fitness)
        assert len(selected_more) == 4

        # Single-objective Pareto selection
        single_fitness = np.array([[3.0], [1.0], [2.0], [4.0]])
        selected_single = selection.do(single_fitness)
        assert len(selected_single) == 3

        selection_more.do(np.array([]))


class TestParetoCrowdingSelection:
    """Pareto crowding-distance selection tests"""

    def test_crowding_selection(self):
        """Test crowding-distance selection"""
        selection = ParetoCrowdingSelection(n_selected=2)

        # Multiple points on the same front
        fitness = np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]])
        selected = selection.do(fitness)
        assert len(selected) == 2
        # Boundary points should have higher selection probability
        assert 0 in selected or 4 in selected

        # Test _select_from_front branch where n_select >= len(front_indices)
        front_indices = [0, 1, 2]
        result = selection._select_from_front(fitness, front_indices, 5)
        assert result == front_indices


class TestParetoRefSelection:
    """Pareto reference-point selection tests"""

    def test_pareto_ref_selection(self):
        """Test Pareto reference-point selection"""
        # Parameter validation
        selection = ParetoRefSelection(n_selected=3, n_objectives=2, n_partitions=4)
        assert selection.n_objectives == 2
        assert selection.n_partitions == 4

        # Invalid parameter tests
        for invalid_obj in [1, 0]:
            with pytest.raises((ValidationError, ParameterException)):
                ParetoRefSelection(n_selected=3, n_objectives=invalid_obj)

        with pytest.raises((ValidationError, ParameterException)):
            ParetoRefSelection(n_selected=3, n_objectives=2, n_partitions=0)

        # Reference direction generation
        assert selection.reference_directions is not None
        assert selection.reference_directions.shape[1] == selection.n_objectives

        # Custom reference directions
        custom_dirs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        custom_selection = ParetoRefSelection(n_selected=2, n_objectives=2, reference_directions=custom_dirs)
        np.testing.assert_array_equal(custom_selection.reference_directions, custom_dirs)

        # Wrong reference direction dimensions
        wrong_dirs = np.array([[1.0, 0.0, 0.0]])  # 3D directions used for 2D problem
        with pytest.raises(ParameterException):
            ParetoRefSelection(n_selected=2, n_objectives=2, reference_directions=wrong_dirs)

        # Basic selection test
        fitness = np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]])
        selected = selection.do(fitness)
        assert len(selected) == 3

        # Test boundary case in _select_from_front
        front_indices = [0, 1, 2]
        # Case where n_select >= len(front_indices)
        result = selection._select_from_front(fitness, front_indices, 5)
        assert result == front_indices

        # Test case with no valid candidates (else branch)
        # Create a scenario where no individuals associate to reference points
        special_fitness = np.array([[10.0, 10.0], [11.0, 11.0]])  # points far from reference directions
        special_front = [0, 1]
        result_special = selection._select_from_front(special_fitness, special_front, 1)
        assert len(result_special) == 1

        # Test case when available candidates are exhausted (break condition)
        small_fitness = np.array([[0.1, 0.9], [0.9, 0.1]])
        small_front = [0, 1]
        result_small = selection._select_from_front(small_fitness, small_front, 3)  # request exceeds available
        assert len(result_small) == 2  # should return all available candidates


class TestUtilityFunctions:
    """Utility functions tests"""

    def test_calc_crowding_distance(self):
        """Test crowding distance calculation"""
        # Basic case
        outputs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        distances = calc_crowding_distance(outputs)
        assert distances[0] == np.inf  # boundary point
        assert distances[2] == np.inf  # boundary point
        assert np.isfinite(distances[1])  # middle point

        # Edge cases
        assert len(calc_crowding_distance(np.array([]).reshape(0, 2))) == 0  # empty array
        assert calc_crowding_distance(np.array([[1.0, 2.0]]))[0] == np.inf  # single point

        # both points are infinite
        two_points = calc_crowding_distance(np.array([[0.0, 1.0], [1.0, 0.0]]))
        assert all(dist == np.inf for dist in two_points)

        # identical points case
        same_points = calc_crowding_distance(np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
        assert same_points[0] == np.inf and same_points[2] == np.inf

    def test_generate_das_dennis_reference_directions(self):
        """Test Das-Dennis reference direction generation"""
        # 2D and 3D cases
        for dims, partitions in [(2, 4), (3, 3)]:
            dirs = generate_das_dennis_reference_directions(dims, partitions)
            assert dirs.shape[1] == dims
            assert dirs.shape[0] > 0
            assert np.allclose(np.sum(dirs, axis=1), 1.0)  # unit simplex
            assert np.all(dirs >= 0)  # non-negativity

    def test_normalize_objectives(self):
        """Test objective normalization"""
        objectives = np.array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]])

        # basic normalization
        normalized = normalize_objectives(objectives)
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
        assert np.allclose(np.min(normalized, axis=0), [0.0, 0.0])
        assert np.allclose(np.max(normalized, axis=0), [1.0, 1.0])

        # specify ideal and nadir points
        ideal = np.array([0.0, 0.0])
        nadir = np.array([10.0, 10.0])
        normalized_custom = normalize_objectives(objectives, ideal, nadir)
        expected = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(normalized_custom, expected)

        # identical values boundary case
        same_objectives = np.array([[5.0, 5.0], [5.0, 5.0]])
        normalized_same = normalize_objectives(same_objectives)
        assert normalized_same.shape == same_objectives.shape
        assert np.all(np.isfinite(normalized_same))

    def test_associate_to_reference_points(self):
        """Test association of solutions to reference points"""
        objectives = np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]])
        ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        ref_indices, distances = associate_to_reference_points(objectives, ref_dirs)

        assert len(ref_indices) == len(objectives)
        assert len(distances) == len(objectives)
        assert all(0 <= idx < len(ref_dirs) for idx in ref_indices)
        assert all(dist >= 0 for dist in distances)


if __name__ == '__main__':
    pytest.main([__file__])
