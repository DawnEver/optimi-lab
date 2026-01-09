from unittest.mock import patch

import numpy as np
import pytest

from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.variable_space import SAMPLE_TYPES, VariableSpace


class TestVariableSpace:
    """Comprehensive test suite for VariableSpace class - 100% coverage."""

    # Basic initialization tests
    def test_import_type_minimal(self):
        """Test import type with minimal data."""
        vs = VariableSpace(var_name_list=['x'], sample_type='import')
        assert vs.var_name_list == ['x']
        assert vs.sample_type == 'import'
        assert vs.var_space_matrix is None

    def test_uniform_sampling_basic(self):
        """Test uniform sampling with default step."""
        vs = VariableSpace(var_name_list=['x'], lower_bounds=[0.0], upper_bounds=[2.0], sample_type='uniform')
        expected = np.array([[0.0], [1.0], [2.0]])
        np.testing.assert_array_equal(vs.var_space_matrix, expected)

    def test_uniform_sampling_custom_step(self):
        """Test uniform sampling with custom step."""
        vs = VariableSpace(
            var_name_list=['x'], lower_bounds=[0.0], upper_bounds=[4.0], sample_type='uniform', step_list=[2.0]
        )
        expected = np.array([[0.0], [2.0], [4.0]])
        np.testing.assert_array_equal(vs.var_space_matrix, expected)

    @patch('optimi_lab.utils.variable_space.LatinHypercube')
    def test_latin_hypercube_sampling(self, mock_lhc):
        """Test Latin Hypercube sampling."""
        mock_sampler = mock_lhc.return_value
        mock_sampler.random.return_value = np.array([[0.5]])

        vs = VariableSpace(
            var_name_list=['x'],
            lower_bounds=[0.0],
            upper_bounds=[10.0],
            sample_type='latin_hypercube',
            n_count=1,
            random_seed=42,
        )

        mock_lhc.assert_called_once_with(d=1, seed=42)
        mock_sampler.random.assert_called_once_with(n=1)
        expected = np.array([[5.0]])
        np.testing.assert_array_equal(vs.var_space_matrix, expected)

    @patch('optimi_lab.utils.variable_space.PoissonDisk')
    def test_poisson_disk_sampling(self, mock_pd):
        """Test Poisson Disk sampling with custom radius."""
        mock_sampler = mock_pd.return_value
        mock_sampler.random.return_value = np.array([[0.5]])

        vs = VariableSpace(
            var_name_list=['x'],
            lower_bounds=[0.0],
            upper_bounds=[10.0],
            sample_type='poisson_disk',
            disk_radius=0.5,
            n_count=1,
            random_seed=123,
        )

        mock_pd.assert_called_once_with(d=1, radius=0.5, seed=123)
        expected = np.array([[5.0]])
        np.testing.assert_array_equal(vs.var_space_matrix, expected)
        vs = VariableSpace(
            var_name_list=['x'],
            lower_bounds=[0.0],
            upper_bounds=[10.0],
            sample_type='poisson_disk',
            n_count=1,
            random_seed=123,
        )
        vs.disk_radius = (1 / vs.n_count) ** (-len(vs.var_name_list))

    # Validation error tests
    def test_invalid_sample_type(self):
        """Test invalid sample type raises error."""
        with pytest.raises(ParameterException, match='sample_type must be one of'):
            VariableSpace(var_name_list=['x'], sample_type='invalid')

    def test_missing_bounds_for_sampling(self):
        """Test missing bounds for non-import types."""
        with pytest.raises(ParameterException, match='lower_bounds and upper_bounds must be provided'):
            VariableSpace(var_name_list=['x'], sample_type='uniform')

    def test_mismatched_lengths(self):
        """Test mismatched array lengths."""
        with pytest.raises(ParameterException, match='must have the same length and >0'):
            VariableSpace(var_name_list=['x', 'y'], lower_bounds=[0.0], upper_bounds=[10.0], sample_type='uniform')

    def test_empty_var_name_list(self):
        """Test empty variable list."""
        with pytest.raises(ParameterException, match='must have the same length and >0'):
            VariableSpace(var_name_list=[], lower_bounds=[], upper_bounds=[], sample_type='uniform')

    def test_invalid_bounds_order(self):
        """Test lower bounds >= upper bounds to cover lines 76-77."""
        with pytest.raises(ParameterException, match='lower_bounds must be less than upper_bounds'):
            VariableSpace(var_name_list=['x'], lower_bounds=[10.0], upper_bounds=[5.0], sample_type='uniform')

    def test_equal_bounds(self):
        """Test equal lower and upper bounds."""
        with pytest.raises(ParameterException, match='lower_bounds must be less than upper_bounds'):
            VariableSpace(var_name_list=['x'], lower_bounds=[5.0], upper_bounds=[5.0], sample_type='uniform')

    def test_step_list_length_mismatch(self):
        """Test step list length mismatch."""
        with pytest.raises(ParameterException, match='step_list must have the same length'):
            VariableSpace(
                var_name_list=['x', 'y'],
                lower_bounds=[0.0, 0.0],
                upper_bounds=[10.0, 10.0],
                sample_type='uniform',
                step_list=[1.0],
            )

    # Edge cases and comprehensive coverage
    @pytest.mark.parametrize('sample_type', SAMPLE_TYPES)
    def test_all_sample_types(self, sample_type):
        """Test all sample types work correctly."""
        if sample_type == 'import':
            vs = VariableSpace(var_name_list=['x'], sample_type=sample_type)
            assert vs.var_space_matrix is None
        else:
            vs = VariableSpace(var_name_list=['x'], lower_bounds=[0.0], upper_bounds=[1.0], sample_type=sample_type)
            assert vs.var_space_matrix is not None

    def test_case_insensitive_sample_type(self):
        """Test sample type is case insensitive."""
        vs = VariableSpace(var_name_list=['x'], lower_bounds=[0.0], upper_bounds=[1.0], sample_type='UNIFORM')
        assert vs.sample_type == 'uniform'

    def test_multiple_variables_uniform(self):
        """Test uniform sampling with multiple variables."""
        vs = VariableSpace(
            var_name_list=['x', 'y'], lower_bounds=[0.0, 10.0], upper_bounds=[1.0, 11.0], sample_type='uniform'
        )
        assert vs.var_space_matrix.shape == (4, 2)  # 2x2 combinations

    def test_array_conversion(self):
        """Test that arrays are properly converted."""
        vs = VariableSpace(
            var_name_list=['x'], lower_bounds=np.array([0.0]), upper_bounds=np.array([1.0]), sample_type='uniform'
        )
        assert isinstance(vs.lower_bounds, np.ndarray)
        assert isinstance(vs.upper_bounds, np.ndarray)

    def test_reproducibility_with_seed(self):
        """Test reproducibility with random seed."""
        vs1 = VariableSpace(
            var_name_list=['x'], lower_bounds=[0.0], upper_bounds=[1.0], sample_type='latin_hypercube', random_seed=42
        )
        vs2 = VariableSpace(
            var_name_list=['x'], lower_bounds=[0.0], upper_bounds=[1.0], sample_type='latin_hypercube', random_seed=42
        )
        np.testing.assert_array_equal(vs1.var_space_matrix, vs2.var_space_matrix)


if __name__ == '__main__':
    pytest.main([__file__])
