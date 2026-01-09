"""Tests for PSO (Particle Swarm Optimization) operators."""

import numpy as np
import pytest
from pydantic import ValidationError

from optimi_lab.intelligent_algorithm.operators.pso_operators import (
    PSOOperatorBase,
    PSOPositionUpdate,
    PSOVelocityUpdate,
)
from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.variable_space import VariableSpace


class MockPSOOperator(PSOOperatorBase):
    """Mock PSO operator for testing abstract base class."""

    def do(self, *args, **kwargs):
        return np.array([[1.0, 2.0]])


class TestPSOOperatorBase:
    """Test PSO operator base class."""

    def test_abstract_instantiation_raises_error(self):
        """Test abstract class cannot be instantiated"""
        with pytest.raises(TypeError):
            PSOOperatorBase()

    def test_mock_implementation_works(self):
        """Test mock implementation works"""
        operator = MockPSOOperator()
        result = operator.do()
        np.testing.assert_array_equal(result, [[1.0, 2.0]])


class TestPSOPositionUpdate:
    """Test PSO position update operator."""

    @pytest.fixture
    def updater(self):
        return PSOPositionUpdate()

    @pytest.fixture
    def var_space(self):
        return VariableSpace(var_name_list=['x1', 'x2'], lower_bounds=[0.0, -1.0], upper_bounds=[1.0, 1.0])

    def test_basic_position_update(self, updater, var_space):
        """Test basic position update"""
        positions = np.array([[0.3, 0.2], [0.7, -0.5]])
        velocities = np.array([[0.1, -0.1], [-0.2, 0.3]])
        new_positions = updater.do(positions, velocities, var_space)
        np.testing.assert_array_equal(new_positions, positions + velocities)

    def test_boundary_clipping(self, updater, var_space):
        """Test boundary clipping"""
        positions = np.array([[0.9, 0.8]])
        velocities = np.array([[0.5, 0.5]])  # will exceed bounds
        new_positions = updater.do(positions, velocities, var_space)

        assert np.all(new_positions >= np.array(var_space.lower_bounds))
        assert np.all(new_positions <= np.array(var_space.upper_bounds))

    @pytest.mark.parametrize('pop_size', [1, 10, 100])
    def test_different_population_sizes(self, updater, var_space, pop_size):
        """Test different population sizes"""
        n_var = len(var_space.var_name_list)
        positions = np.random.rand(pop_size, n_var)
        velocities = 0.1 * (np.random.rand(pop_size, n_var) - 0.5)

        new_positions = updater.do(positions, velocities, var_space)

        assert new_positions.shape == (pop_size, n_var)
        assert np.all(new_positions >= np.array(var_space.lower_bounds))
        assert np.all(new_positions <= np.array(var_space.upper_bounds))

    def test_empty_input(self, updater, var_space):
        """Test empty input"""
        positions = np.array([]).reshape(0, 2)
        velocities = np.array([]).reshape(0, 2)
        new_positions = updater.do(positions, velocities, var_space)
        assert new_positions.shape == (0, 2)


class TestPSOVelocityUpdate:
    """Test PSO velocity update operator."""

    @pytest.fixture
    def updater(self):
        return PSOVelocityUpdate(w=0.4, c1=2.0, c2=2.0, v_max=0.1)

    @pytest.fixture
    def var_space(self):
        return VariableSpace(var_name_list=['x1', 'x2'], lower_bounds=[0.0, -1.0], upper_bounds=[1.0, 1.0])

    @pytest.mark.parametrize(
        ('param', 'value', 'should_raise'),
        [
            ('w', 0.5, False),
            ('w', 0.0, False),
            ('w', 1.0, False),
            ('w', -0.1, True),
            ('w', 1.1, True),
            ('c1', 2.0, False),
            ('c1', 0.0, False),
            ('c1', -0.1, True),
            ('c2', 2.0, False),
            ('c2', 0.0, False),
            ('c2', -0.1, True),
            ('v_max', 0.1, False),
            ('v_max', 1.0, False),
            ('v_max', 0.0, True),
            ('v_max', -0.1, True),
        ],
    )
    def test_parameter_validation(self, param, value, should_raise):
        """Test parameter validation"""
        kwargs = {'w': 0.5, 'c1': 2.0, 'c2': 2.0, 'v_max': 0.1}
        kwargs[param] = value

        if should_raise:
            with pytest.raises((ValidationError, ParameterException)):
                PSOVelocityUpdate(**kwargs)
        else:
            updater = PSOVelocityUpdate(**kwargs)
            assert getattr(updater, param) == value

    def test_basic_velocity_update(self, updater, var_space):
        """Test basic velocity update"""
        velocities = np.array([[0.05, -0.03]])
        positions = np.array([[0.3, 0.2]])
        personal_best = np.array([[0.5, 0.4]])
        global_best = np.array([[0.8, 0.6]])

        new_velocities = updater.do(velocities, positions, personal_best, global_best, var_space)

        assert new_velocities.shape == velocities.shape
        assert np.all(np.abs(new_velocities) <= updater.v_max)

    def test_velocity_clipping(self, var_space):
        """Test velocity clipping"""
        updater = PSOVelocityUpdate(w=0.9, c1=5.0, c2=5.0, v_max=0.1)
        velocities = np.array([[0.08, -0.08]])
        positions = np.array([[0.1, 0.1]])
        personal_best = np.array([[0.9, 0.9]])
        global_best = np.array([[0.9, 0.9]])

        new_velocities = updater.do(velocities, positions, personal_best, global_best, var_space)
        assert np.all(np.abs(new_velocities) <= updater.v_max)

    def test_global_best_scenarios(self, updater, var_space):
        """Test different global-best scenarios"""
        velocities = np.array([[0.05, -0.03]])
        positions = np.array([[0.3, 0.2]])
        personal_best = np.array([[0.5, 0.4]])

        # No global best
        new_v1 = updater.do(velocities, positions, personal_best, np.array([]), var_space)
        assert new_v1.shape == velocities.shape

        # Multiple global bests
        global_best = np.array([[0.8, 0.6], [0.9, 0.7]])
        new_v2 = updater.do(velocities, positions, personal_best, global_best, var_space)
        assert new_v2.shape == velocities.shape

    def test_reproducibility(self, updater, var_space):
        """Test reproducibility of random seeds"""
        test_data = (np.array([[0.05, -0.03]]), np.array([[0.3, 0.2]]), np.array([[0.5, 0.4]]), np.array([[0.8, 0.6]]))

        np.random.seed(42)
        result1 = updater.do(*test_data, var_space)
        np.random.seed(42)
        result2 = updater.do(*test_data, var_space)

        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.parametrize(
        ('w', 'c1', 'c2'),
        [
            (0.8, 0.0, 0.0),  # pure inertia
            (0.0, 2.0, 0.0),  # pure cognitive
            (0.0, 0.0, 2.0),  # pure social
        ],
    )
    def test_component_effects(self, var_space, w, c1, c2):
        """Test effects of PSO components"""
        updater = PSOVelocityUpdate(w=w, c1=c1, c2=c2, v_max=1.0)
        velocities = np.array([[0.1, -0.1]])
        positions = np.array([[0.3, 0.2]])
        personal_best = np.array([[0.5, 0.4]])
        global_best = np.array([[0.9, 0.6]])

        np.random.seed(42)
        new_velocities = updater.do(velocities, positions, personal_best, global_best, var_space)

        if w > 0 and c1 == 0 and c2 == 0:
            # pure inertia term
            np.testing.assert_array_almost_equal(new_velocities, w * velocities)
        else:
            assert new_velocities.shape == velocities.shape

    def test_convergence_case(self, updater, var_space):
        """Test convergence case"""
        optimal = np.array([[0.7, 0.3]])
        velocities = np.array([[0.05, -0.03]])

        new_velocities = updater.do(velocities, optimal, optimal, optimal, var_space)
        assert np.all(np.abs(new_velocities) <= np.abs(velocities))

    def test_parameter_impact_comparison(self, var_space):
        """Test comparison of parameter impacts"""
        test_data = (np.array([[0.05, -0.03]]), np.array([[0.3, 0.2]]), np.array([[0.8, 0.7]]), np.array([[0.9, 0.8]]))

        # Compare high/low inertia weights
        high_w = PSOVelocityUpdate(w=0.9, c1=0.5, c2=0.5, v_max=1.0)
        low_w = PSOVelocityUpdate(w=0.1, c1=0.5, c2=0.5, v_max=1.0)

        np.random.seed(123)
        result_high = high_w.do(*test_data, var_space)
        np.random.seed(123)
        result_low = low_w.do(*test_data, var_space)

        assert not np.array_equal(result_high, result_low)

    @pytest.mark.parametrize('pop_size', [1, 10, 100])
    def test_population_scalability(self, updater, var_space, pop_size):
        """Test population scalability"""
        n_var = len(var_space.var_name_list)
        velocities = 0.05 * (np.random.rand(pop_size, n_var) - 0.5)
        positions = np.random.rand(pop_size, n_var)
        personal_best = np.random.rand(pop_size, n_var)
        global_best = np.random.rand(1, n_var)

        new_velocities = updater.do(velocities, positions, personal_best, global_best, var_space)

        assert new_velocities.shape == (pop_size, n_var)
        assert np.all(np.abs(new_velocities) <= updater.v_max)
