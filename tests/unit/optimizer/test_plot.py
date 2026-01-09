"""Final optimized comprehensive test suite for plot.py module
Achieving maximum coverage with minimal, efficient test cases
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optimi_lab.plot import parse_cols_to_draw, plot, plot_contour, plot_moo
from optimi_lab.utils.exceptions import ParameterException


@pytest.fixture
def sample_data():
    """Unified test data fixture"""
    return {
        'vars': ['x1', 'x2', 'x3'],
        'objs': ['f1', 'f2'],
        'df': pd.DataFrame({
            'x1': [1, 2, 3, 4],
            'x2': [5, 6, 7, 8],
            'x3': [9, 10, 11, 12],
            'f1': [0.1, 0.2, 0.3, 0.4],
            'f2': [0.5, 0.6, 0.7, 0.8],
        }),
        'df_pareto': pd.DataFrame({'x1': [1, 3], 'x2': [4, 6], 'f1': [1, 3], 'f2': [3, 1]}),
        'df_contour': pd.DataFrame({'w': [4, 3, 2, 1], 'x': [1, 1, 2, 2], 'y': [1, 2, 1, 2], 'z': [1, 2, 3, 4]}),
    }


class TestParseColsToDraw:
    """Test parse_cols_to_draw function with parameterization"""

    @pytest.mark.parametrize(
        ('args', 'expected_vars', 'expected_objs'),
        [
            (('var#0', 'obj#1'), ['x1'], ['f2']),
            (('var#0', 'var#1', 'obj#0'), ['x1', 'x2'], ['f1']),
            (('obj#0', 'obj#1'), [], ['f1', 'f2']),
            ((None, 'var#2'), ['x3'], []),
            ((), [], []),
            (('var#0', None, 'obj#1'), ['x1'], ['f2']),
        ],
    )
    def test_valid_parsing(self, args, expected_vars, expected_objs, sample_data):
        """Test valid axis parsing scenarios"""
        vars_result, objs_result = parse_cols_to_draw(
            *args, var_name_list=sample_data['vars'], obj_name_list=sample_data['objs']
        )
        assert vars_result == expected_vars
        assert objs_result == expected_objs

    @pytest.mark.parametrize(
        ('args', 'error_type'),
        [
            (('obj#5',), IndexError),  # Invalid index
            (('var#10',), IndexError),  # Invalid index
            (('invalid#0',), ParameterException),  # Invalid type
            (('invalid',), ParameterException),  # Invalid format
        ],
    )
    def test_invalid_parsing(self, args, error_type, sample_data):
        """Test invalid axis parsing scenarios"""
        with pytest.raises(error_type):
            parse_cols_to_draw(*args, var_name_list=sample_data['vars'], obj_name_list=sample_data['objs'])


class TestPlot:
    """Test plot function with comprehensive scenarios"""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplot')
    def test_plot_comprehensive(self, mock_subplot, mock_savefig, sample_data):
        """Single comprehensive test covering all plot scenarios"""
        mock_ax = MagicMock()
        mock_subplot.return_value = mock_ax

        # Test scenarios: no axes, 1D, 2D, 3D, colorful, save
        scenarios = [
            {},  # No axes
            {'x_axis': 'var#0'},  # 1D
            {'x_axis': 'var#0', 'y_axis': 'obj#1', 'plot': False, 'scatter': False},  # 2D
            {
                'x_axis': 'var#0',
                'y_axis': 'var#1',
                'z_axis': 'obj#1',
                'plot': False,
                'scatter': False,
                'surface': True,
            },  # 3D
            {'x_axis': 'var#0', 'colorful': True},  # Colorful
            {
                'x_axis': 'var#0',
                'savefig': True,
                'xlabel': 'X-axis',
                'ylabel': 'Y-axis',
                'zlabel': 'Z-axis',
            },  # Save figure,
            {
                'axises': ['obj#0', 'obj#1'],
                'style_plot': {'color': 'blue'},
                'style_surface': {'cmap': 'rainbow'},
                'style_scatter': {'color': 'red'},
            },
            {
                'axises': ['var#0', 'obj#0', 'obj#1'],
                'xlabel': 'X-axis',
                'ylabel': 'Y-axis',
                'zlabel': 'Z-axis',
            },  # Custom style
        ]

        for scenario in scenarios:
            mock_ax.reset_mock()
            plot(
                var_name_list=sample_data['vars'],
                obj_name_list=sample_data['objs'],
                data_frame=sample_data['df'],
                **scenario,
            )

    @patch('matplotlib.pyplot.subplot')
    def test_plot_errors(self, mock_subplot, sample_data):
        """Test error conditions"""
        mock_ax = MagicMock()
        mock_subplot.return_value = mock_ax

        # Invalid axis format
        with pytest.raises(ParameterException, match='Invalid axis type'):
            plot(
                var_name_list=sample_data['vars'],
                obj_name_list=sample_data['objs'],
                data_frame=sample_data['df'],
                x_axis='invalid_axis',
            )

        with pytest.raises(ParameterException, match='The number of valid var/obj is'):
            plot(
                var_name_list=sample_data['vars'],
                obj_name_list=sample_data['objs'],
                data_frame=sample_data['df'],
                axises=['var#0', 'var#1', 'obj#0', 'obj#1'],
            )

        with pytest.raises(ParameterException, match='Cannot draw surface with two axises'):
            plot(
                var_name_list=sample_data['vars'],
                obj_name_list=sample_data['objs'],
                data_frame=sample_data['df'],
                axises=['var#0', 'var#1'],
                surface=True,
            )


class TestPlotMoo:
    """Test plot_moo function"""

    @patch('matplotlib.pyplot.subplot')
    def test_moo_comprehensive(self, mock_subplot, sample_data):
        """Comprehensive plot_moo test"""
        mock_ax = MagicMock()
        mock_subplot.return_value = mock_ax

        # Test scenarios
        scenarios = [
            {},  # No axes
            {'axises': ['obj#0'], 'draw_pareto_front': True},  # Basic 1D
            {'axises': ['obj#0'], 'draw_pareto_point': True, 'draw_pareto_front': False},  # Basic 1D
            {
                'axises': ['var#0'],
                'xlabel': 'X-axis',
                'ylabel': 'Y-axis',
                'zlabel': 'Z-axis',
                'draw_pareto_front': False,
            },
            {'x_axis': 'obj#0', 'y_axis': 'obj#1', 'draw_pareto_front': True},
            {'x_axis': 'obj#0', 'y_axis': 'obj#1', 'draw_pareto_point': True, 'draw_pareto_front': False},
            {
                'x_axis': 'obj#0',
                'y_axis': 'obj#1',
                'xlabel': 'X-axis',
                'ylabel': 'Y-axis',
                'zlabel': 'Z-axis',
                'draw_pareto_front': False,
            },
            {'x_axis': 'obj#0', 'y_axis': 'obj#1', 'z_axis': 'var#0', 'draw_pareto_front': True, 'savefig': True},  # 3D
            {'x_axis': 'obj#0', 'y_axis': 'obj#1', 'z_axis': 'var#0', 'draw_pareto_front': False},  # 3D
            {
                'axises': ['var#0', 'obj#0', 'obj#1'],
                'style_scatter': {'color': 'red'},
                'style_scatter_pareto': {'color': 'red'},
                'xlabel': 'X-axis',
                'ylabel': 'Y-axis',
                'zlabel': 'Z-axis',
            },
        ]

        for scenario in scenarios:
            mock_ax.reset_mock()
            plot_moo(
                var_name_list=sample_data['vars'],
                obj_name_list=sample_data['objs'],
                df_all=sample_data['df'],
                df_pareto=sample_data['df_pareto'],
                **scenario,
            )

        # Make ax.plot_trisurf(x, y, z) raise RuntimeError
        plot_moo(
            var_name_list=sample_data['vars'],
            obj_name_list=sample_data['objs'],
            df_all=sample_data['df'],
            df_pareto=sample_data['df'],
            axises=['var#0', 'obj#0', 'obj#1'],
            draw_pareto_front=True,
        )

    @patch('matplotlib.pyplot.subplot')
    def test_moo_trisurf_error(self, mock_subplot, sample_data):
        """Test trisurf error handling"""
        mock_ax = MagicMock()
        mock_subplot.return_value = mock_ax
        # Make plot_trisurf raise an exception
        mock_ax.plot_trisurf.side_effect = Exception('Trisurf error')

        plot_moo(
            var_name_list=sample_data['vars'],
            obj_name_list=sample_data['objs'],
            df_all=sample_data['df'],
            df_pareto=sample_data['df_pareto'],
            x_axis='obj#0',
            y_axis='obj#1',
            z_axis='var#0',
        )

        with pytest.raises(ParameterException, match='The number of valid var/obj is'):
            plot_moo(
                var_name_list=sample_data['vars'],
                obj_name_list=sample_data['objs'],
                df_all=sample_data['df'],
                df_pareto=sample_data['df_pareto'],
                axises=['var#0', 'var#1', 'obj#0', 'obj#1'],
            )


class TestPlotContour:
    """Test plot_contour function"""

    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.subplot')
    def test_contour_comprehensive(self, mock_subplot, mock_colorbar, sample_data):
        """Comprehensive contour plot test"""
        mock_ax = MagicMock()
        mock_subplot.return_value = mock_ax
        mock_colorbar.return_value = MagicMock()

        # Test scenarios
        scenarios = [
            {'x_axis': 'var#0', 'y_axis': 'var#1', 'z_axis': 'obj#0', 'savefig': True},  # Basic contour
            {'x_axis': 'var#0', 'y_axis': 'var#1', 'z_axis': 'obj#0', 'fill': False},  # Contour lines
            {
                'x_axis': 'var#0',
                'y_axis': 'var#1',
                'z_axis': 'obj#0',
                'styles_contour': {'levels': 10},
                'styles_contourf': {'levels': 10},
            },  # Custom levels
            {
                'axises': ['var#0', 'var#1', 'obj#0'],
                'xlabel': 'X-axis',
                'ylabel': 'Y-axis',
                'zlabel': 'Z-axis',
            },  # Default labels
        ]

        for scenario in scenarios:
            mock_ax.reset_mock()
            plot_contour(
                var_name_list=['x', 'y'], obj_name_list=['z'], data_frame=sample_data['df_contour'], **scenario
            )

    @patch('matplotlib.pyplot.subplot')
    def test_contour_errors(self, mock_subplot, sample_data):
        """Test contour error conditions"""
        mock_ax = MagicMock()
        mock_subplot.return_value = mock_ax

        # Too few columns
        with pytest.raises(ParameterException, match='Plot failed'):
            plot_contour(
                var_name_list=['x', 'y'], obj_name_list=['z'], data_frame=sample_data['df_contour'], x_axis='var#0'
            )

        # Insufficient ticks
        test_df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1]})
        with pytest.raises(ParameterException, match='ticks number must be larger than 2'):
            plot_contour(
                var_name_list=['x', 'y'],
                obj_name_list=['z'],
                data_frame=test_df,
                x_axis='var#0',
                y_axis='var#1',
                z_axis='obj#0',
            )

        with pytest.raises(NotImplementedError, match=''):
            plot_contour(
                var_name_list=['w'],
                obj_name_list=['x', 'y', 'z'],
                data_frame=sample_data['df_contour'],
                axises=['var#0', 'obj#0', 'obj#1', 'obj#2'],
            )
        with pytest.raises(ParameterException, match='not enough variables and objectives'):
            plot_contour(
                var_name_list=['w', 'x'],
                obj_name_list=['x', 'y', 'z'],
                data_frame=sample_data['df_contour'],
                axises=['var#0', 'var#1', 'obj#0', 'obj#1'],
            )

    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.subplot')
    def test_contour_data_validation(self, mock_subplot, mock_colorbar, sample_data):
        """Test contour data validation warnings"""
        mock_ax = MagicMock()
        mock_subplot.return_value = mock_ax
        mock_colorbar.return_value = MagicMock()

        # Create DataFrame with duplicate x,y combinations (should warn but not fail)
        df_excess = pd.DataFrame({'x': [1, 1, 2, 2, 1, 1, 2, 2], 'y': [1, 2, 1, 2, 1, 2, 1, 2], 'z': list(range(8))})

        # Should complete successfully despite warning
        plot_contour(
            var_name_list=['x', 'y'],
            obj_name_list=['z'],
            data_frame=df_excess,
            x_axis='var#0',
            y_axis='obj#0',
            z_axis='var#1',
        )

        df_excess = pd.DataFrame({'x': list(range(8)), 'y': list(range(8)), 'z': [1, 2, 1, 2, 1, 2, 1, 2]})

        # Should complete successfully despite warning
        plot_contour(
            var_name_list=['x', 'y'],
            obj_name_list=['z'],
            data_frame=df_excess,
            x_axis='var#0',
            y_axis='obj#0',
            z_axis='var#1',
        )

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
