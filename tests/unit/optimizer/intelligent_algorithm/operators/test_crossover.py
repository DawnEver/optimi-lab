"""Crossover operator test module"""

import numpy as np
import pytest
from pydantic import ValidationError

from optimi_lab.intelligent_algorithm.operators.crossover import (
    BinomialCrossover,
    CrossoverBase,
    SimulatedBinaryCrossover,
)
from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.variable_space import VariableSpace


class MockCrossover(CrossoverBase):
    """Mock crossover operator for testing"""

    def do(self, parent1: np.ndarray, parent2: np.ndarray, variable_space: VariableSpace) -> np.ndarray:
        return parent1.copy()


class TestCrossoverBase:
    """Crossover base tests"""

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid values and defaults
        assert MockCrossover().crossover_rate == 0.9
        assert MockCrossover(crossover_rate=0.5).crossover_rate == 0.5
        assert MockCrossover(crossover_rate=0.0).crossover_rate == 0.0
        assert MockCrossover(crossover_rate=1.0).crossover_rate == 1.0

        # Invalid values
        with pytest.raises((ValidationError, ParameterException)):
            MockCrossover(crossover_rate=-0.1)
        with pytest.raises((ValidationError, ParameterException)):
            MockCrossover(crossover_rate=1.1)


class TestBinomialCrossover:
    """Binomial crossover tests"""

    @pytest.fixture
    def crossover(self):
        return BinomialCrossover(crossover_rate=0.7)

    @pytest.fixture
    def variable_space(self):
        return VariableSpace(var_name_list=['x1', 'x2'], lower_bounds=[0.0, -1.0], upper_bounds=[1.0, 1.0])

    def test_basic_functionality(self, crossover, variable_space):
        """Test basic functionality and boundary cases"""
        # Basic crossover
        parent1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        parent2 = np.array([[0.5, 0.6], [0.7, 0.8]])
        offspring = crossover.do(parent1, parent2, variable_space)
        assert offspring.shape == parent1.shape

        # Single individual
        parent1_single = np.array([[0.1, 0.2]])
        parent2_single = np.array([[0.5, 0.6]])
        offspring_single = crossover.do(parent1_single, parent2_single, variable_space)
        assert offspring_single.shape == (1, 2)

        # Empty arrays
        empty1 = np.array([]).reshape(0, 2)
        empty2 = np.array([]).reshape(0, 2)
        offspring_empty = crossover.do(empty1, empty2, variable_space)
        assert offspring_empty.shape == (0, 2)

    def test_guaranteed_crossover(self, crossover, variable_space):
        """Test each individual has at least one dimension crossed"""
        parent1 = np.array([[1.0, 1.0]])
        parent2 = np.array([[3.0, 3.0]])

        # Run multiple times to verify randomness
        for _ in range(5):
            offspring = crossover.do(parent1, parent2, variable_space)
            # At least one dimension differs from parent1
            assert not np.array_equal(offspring[0], parent1[0]) or np.array_equal(offspring[0], parent2[0])


class TestSimulatedBinaryCrossover:
    """Simulated binary crossover tests"""

    @pytest.fixture
    def crossover(self):
        return SimulatedBinaryCrossover(crossover_rate=0.9, eta_crossover=20.0)

    @pytest.fixture
    def variable_space(self):
        return VariableSpace(
            var_name_list=['x1', 'x2', 'x3'], lower_bounds=[0.0, -2.0, -1.0], upper_bounds=[1.0, 2.0, 3.0]
        )

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid values
        assert SimulatedBinaryCrossover(eta_crossover=0.0).eta_crossover == 0.0
        assert SimulatedBinaryCrossover(eta_crossover=50.0).eta_crossover == 50.0

        # Invalid values
        with pytest.raises((ValidationError, ParameterException)):
            SimulatedBinaryCrossover(eta_crossover=-1.0)

    def test_basic_crossover(self, crossover, variable_space):
        """Test basic SBX crossover"""
        parent1 = np.array([[0.2, 0.0, 1.0], [0.8, 1.5, -0.5]])
        parent2 = np.array([[0.6, -0.5, 2.0], [0.4, -1.0, 2.5]])

        offspring1, offspring2 = crossover.do(parent1, parent2, variable_space)

        assert offspring1.shape == parent1.shape
        assert offspring2.shape == parent2.shape

    def test_boundary_constraints(self, crossover, variable_space):
        """Test boundary constraints"""
        parent1 = np.array([[0.0, -2.0, -1.0]])  # lower bound
        parent2 = np.array([[1.0, 2.0, 3.0]])  # upper bound

        offspring1, offspring2 = crossover.do(parent1, parent2, variable_space)

        lb = np.array(variable_space.lower_bounds)
        ub = np.array(variable_space.upper_bounds)

        assert np.all(offspring1 >= lb) and np.all(offspring1 <= ub)
        assert np.all(offspring2 >= lb) and np.all(offspring2 <= ub)

    def test_edge_cases(self, variable_space):
        """Test edge cases"""
        # No crossover when rate is 0
        crossover_zero = SimulatedBinaryCrossover(crossover_rate=0.0)
        parent1 = np.array([[0.2, 0.0, 1.0]])
        parent2 = np.array([[0.6, -0.5, 2.0]])

        offspring1, offspring2 = crossover_zero.do(parent1, parent2, variable_space)
        np.testing.assert_array_equal(offspring1, parent1)
        np.testing.assert_array_equal(offspring2, parent2)

        # Identical parents
        crossover_normal = SimulatedBinaryCrossover(crossover_rate=1.0)
        offspring1, offspring2 = crossover_normal.do(parent1, parent1, variable_space)
        np.testing.assert_array_almost_equal(offspring1, offspring2)

    def test_reproducibility(self, crossover, variable_space):
        """Test reproducibility of random seeds"""
        parent1 = np.array([[0.2, 0.0, 1.0]])
        parent2 = np.array([[0.6, -0.5, 2.0]])

        np.random.seed(42)
        offspring1_a, offspring2_a = crossover.do(parent1, parent2, variable_space)

        np.random.seed(42)
        offspring1_b, offspring2_b = crossover.do(parent1, parent2, variable_space)

        np.testing.assert_array_equal(offspring1_a, offspring1_b)
        np.testing.assert_array_equal(offspring2_a, offspring2_b)


if __name__ == '__main__':
    pytest.main([__file__])
