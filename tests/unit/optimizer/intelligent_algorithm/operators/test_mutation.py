"""Mutation operator tests module

Concise, efficient tests aiming for high coverage
"""

import numpy as np
import pytest
from pydantic import ValidationError

from optimi_lab.intelligent_algorithm.operators.mutation import (
    DifferentialMutation,
    MutationBase,
    PolynomialMutation,
)
from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.variable_space import VariableSpace


class MockMutation(MutationBase):
    """Mock mutation operator for testing"""

    def do(self, current_inputs: np.ndarray, best_inputs: np.ndarray, variable_space: VariableSpace) -> np.ndarray:
        return current_inputs.copy()


class TestMutationBase:
    """Mutation base tests"""

    def test_abstract_class(self):
        """Test abstract class and concrete implementation"""
        # Abstract class cannot be instantiated directly
        with pytest.raises(TypeError):
            MutationBase()

        # Concrete implementation can be instantiated and called
        mutation = MockMutation()
        vs = VariableSpace(var_name_list=['x'], lower_bounds=[0.0], upper_bounds=[1.0])
        inputs = np.array([[0.5]])
        result = mutation.do(inputs, np.array([]), vs)
        np.testing.assert_array_equal(result, inputs)


class TestPolynomialMutation:
    """Polynomial mutation tests"""

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters
        pm = PolynomialMutation(mutation_rate=0.5, eta_mutation=10.0)
        assert pm.mutation_rate == 0.5
        assert pm.eta_mutation == 10.0

        # boundary values
        assert PolynomialMutation(mutation_rate=0.0).mutation_rate == 0.0
        assert PolynomialMutation(mutation_rate=1.0).mutation_rate == 1.0
        assert PolynomialMutation(eta_mutation=0.1).eta_mutation == 0.1

        # Invalid mutation_rate
        with pytest.raises((ValidationError, ParameterException)):
            PolynomialMutation(mutation_rate=-0.1)
        with pytest.raises((ValidationError, ParameterException)):
            PolynomialMutation(mutation_rate=1.1)

        # invalid eta_mutation
        with pytest.raises((ValidationError, ParameterException)):
            PolynomialMutation(eta_mutation=0.0)
        with pytest.raises((ValidationError, ParameterException)):
            PolynomialMutation(eta_mutation=-1.0)

    def test_mutation_operations(self):
        """Test mutation operations through all branches"""
        vs = VariableSpace(var_name_list=['x1', 'x2'], lower_bounds=[0.0, -1.0], upper_bounds=[1.0, 1.0])

        # Test mutation_rate=0 (no mutation)
        pm_zero = PolynomialMutation(mutation_rate=0.0)
        inputs = np.array([[0.5, 0.0]])
        result = pm_zero.do(inputs, np.array([]), vs)
        np.testing.assert_array_equal(result, inputs)

        # Test mutation_rate=1 (always mutate) - cover mask1 branch (r <= 0.5)
        pm_full = PolynomialMutation(mutation_rate=1.0, eta_mutation=1.0)

        # Force r <= 0.5
        np.random.seed(1)  # this seed produces random numbers < 0.5
        result1 = pm_full.do(inputs, np.array([]), vs)
        assert np.all(result1 >= np.array(vs.lower_bounds))
        assert np.all(result1 <= np.array(vs.upper_bounds))

        # Force r > 0.5 - cover mask2 branch
        np.random.seed(10)  # this seed produces random numbers > 0.5
        result2 = pm_full.do(inputs, np.array([]), vs)
        assert np.all(result2 >= np.array(vs.lower_bounds))
        assert np.all(result2 <= np.array(vs.upper_bounds))

        # Test boundary cases
        boundary_inputs = np.array([[0.0, -1.0], [1.0, 1.0]])  # lower and upper bounds
        result_boundary = pm_full.do(boundary_inputs, np.array([]), vs)
        assert np.all(result_boundary >= np.array(vs.lower_bounds))
        assert np.all(result_boundary <= np.array(vs.upper_bounds))

        # Test empty input
        empty_inputs = np.array([]).reshape(0, 2)
        result_empty = pm_full.do(empty_inputs, np.array([]), vs)
        assert result_empty.shape == (0, 2)


class TestDifferentialMutation:
    """Differential mutation tests"""

    def test_parameter_validation(self):
        """Test parameter validation"""
        # valid parameters
        dm = DifferentialMutation(sizing_factor=0.5)
        assert dm.sizing_factor == 0.5

        # boundary values
        assert DifferentialMutation(sizing_factor=0.1).sizing_factor == 0.1
        assert DifferentialMutation(sizing_factor=2.0).sizing_factor == 2.0

        # Invalid values
        with pytest.raises((ValidationError, ParameterException)):
            DifferentialMutation(sizing_factor=0.0)
        with pytest.raises((ValidationError, ParameterException)):
            DifferentialMutation(sizing_factor=-0.1)
        with pytest.raises((ValidationError, ParameterException)):
            DifferentialMutation(sizing_factor=2.1)

    def test_differential_mutation(self):
        """Test differential mutation operation"""
        vs = VariableSpace(var_name_list=['x1', 'x2'], lower_bounds=[0.0, -1.0], upper_bounds=[1.0, 1.0])
        dm = DifferentialMutation(sizing_factor=0.5)

        # Normal population (>= 3 individuals)
        inputs = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        result = dm.do(inputs, np.array([]), vs)

        assert result.shape == inputs.shape
        assert np.all(result >= np.array(vs.lower_bounds))
        assert np.all(result <= np.array(vs.upper_bounds))

        # Population too small (<3 individuals) - cover insufficient-population branch
        small_inputs = np.array([[0.1, 0.2], [0.3, 0.4]])
        result_small = dm.do(small_inputs, np.array([]), vs)
        assert result_small.shape == small_inputs.shape
        # When population is too small should return original individuals
        np.testing.assert_array_equal(result_small, small_inputs)

        # Single individual
        single_input = np.array([[0.5, 0.0]])
        result_single = dm.do(single_input, np.array([]), vs)
        assert result_single.shape == (1, 2)
        np.testing.assert_array_equal(result_single, single_input)

        # Test boundary constraints
        boundary_inputs = np.array([[0.0, -1.0], [1.0, 1.0], [0.5, 0.0]])
        result_boundary = dm.do(boundary_inputs, np.array([]), vs)
        assert np.all(result_boundary >= np.array(vs.lower_bounds))
        assert np.all(result_boundary <= np.array(vs.upper_bounds))


if __name__ == '__main__':
    pytest.main([__file__])
