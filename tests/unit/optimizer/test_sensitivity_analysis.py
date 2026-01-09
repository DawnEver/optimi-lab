from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optimi_lab.sensitivity_analysis import correlation_analysis, pair_plot, plot_matrix
from optimi_lab.utils.exceptions import ParameterException


def test_pair_plot_invalid_diagonal():
    """Test pair_plot with an invalid diagonal argument."""
    data = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)})
    with pytest.raises(ParameterException, match="Expected 'hist', 'kde', or 'density'"):
        pair_plot(data, diagonal='invalid')


def test_pair_plot(tmp_path: Path):
    """Test pair_plot with savefig option."""
    data = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100), 'C': np.random.rand(100)})
    pair_plot(data, density_kwds={}, hist_kwds={}, scatter_kwds={'alpha': 0.1, 'marker': '.'}, diagonal='density')
    fig_path = tmp_path / 'pair_plot.png'
    pair_plot(data, savefig=True, fig_path=fig_path, diagonal='density')
    assert fig_path.exists()
    pair_plot(data, savefig=False, fig_path=fig_path, diagonal='hist')
    data_ = data[['A']]
    with pytest.raises(AssertionError, match='DataFrame must have at least two numeric columns'):
        pair_plot(data_)


def test_pair_plot_empty_dataframe():
    """Test pair_plot with an empty DataFrame."""
    data = pd.DataFrame()
    with pytest.raises(AssertionError, match='DataFrame must have at least two numeric columns'):
        pair_plot(data)


def test_correlation_analysis():
    """Test correlation_analysis with different methods."""
    data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10], 'C': [5, 4, 3, 2, 1]})

    # Test pearson method (default)
    corr = correlation_analysis(data)
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (3, 3)
    assert abs(corr.loc['A', 'B'] - 1.0) < 1e-10  # Perfect positive correlation

    # Test other methods
    for method in ['kendall', 'spearman']:
        corr = correlation_analysis(data, method=method)
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)

    # Test invalid method
    with pytest.raises(ValueError, match='Method must be one of'):
        correlation_analysis(data, method='invalid')


def test_plot_matrix(tmp_path: Path):
    """Test plot_matrix with different parameters."""
    # Create test correlation matrix
    data = pd.DataFrame({'X': [0.8, -0.3], 'Y': [-0.3, 0.9]}, index=['X', 'Y'])

    # Test default parameters
    plt_obj = plot_matrix(data)
    assert plt_obj is not None

    # Test with absolute values (default)
    plot_matrix(data, use_abs=True)

    # Test without absolute values
    plot_matrix(data, use_abs=False)

    # Test with custom ticks
    plot_matrix(data, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    plot_matrix(data, use_abs=False, ticks=[0, 0.25, 0.5, 0.75, 1.0])

    # Test with custom fontsize
    plot_matrix(data, fontsize=12)

    # Test save functionality
    fig_path = tmp_path / 'correlation_matrix.png'
    plot_matrix(data, savefig=True, fig_path=fig_path)
    assert fig_path.exists()


if __name__ == '__main__':
    pytest.main([__file__])
