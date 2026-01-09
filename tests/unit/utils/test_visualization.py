from pathlib import Path

import numpy as np
import pytest

from optimi_lab.utils import visualization
from optimi_lab.utils.exceptions import ParameterException


def test_config_plt_params():
    visualization.config_plt_params()
    param_dict = {'figure.figsize': (5, 4), 'figure.dpi': 199, 'font.family': ['Times New Roman']}
    visualization.config_plt_params(**param_dict)


def test_config_plt_params_invalid_key():
    """Test invalid key handling in config_plt_params."""
    with pytest.raises(ParameterException):
        visualization.config_plt_params(nonexistent_key=123)


def test_visualize_2d_invalid_input():
    """Test visualize_2d with invalid input types."""
    with pytest.raises(ParameterException, match='X or Y type .* is not supported'):
        visualization.visualize_2d('invalid', [1, 2, 3])
    with pytest.raises(ParameterException):
        visualization.visualize_2d(np.array([1, 2, 3]), np.array([1, 2]))


def test_visualize_2d_savefig(tmp_path: Path):
    """Test visualize_2d with savefig option."""
    X = np.random.rand(100)
    Y = np.random.rand(100)
    fig_path = tmp_path / 'visualize_2d.png'
    style_dict_list = [
        {
            'scatter': {'color': 'red'},
            'plot': {'color': 'blue'},
            'bar': {'color': 'green'},
        },
    ]
    legend_dict = {'title': 'Test Legend', 'fontsize': 10}
    visualization.visualize_2d(
        X,
        Y,
        fig_types=['bar'],
        savefig=True,
        fig_path=fig_path,
        style_dict_list=style_dict_list,
        legend_dict=legend_dict,
    )
    assert fig_path.exists()
    X = X.reshape(50, 2)
    Y = Y.reshape(50, 2)
    visualization.visualize_2d(X, Y, fig_types=['plot'], equal_aspect=True)
    visualization.visualize_2d(X, Y, fig_types=['scatter'], xlim=(0, 1), ylim=(0, 1))
    visualization.visualize_2d(X, Y, fig_types=['invalid'])


def test_visualize_2d():
    X_ = np.random.rand(100)
    Y_ = np.random.rand(100)

    X = [X_] * 3
    Y = [Y_] * 3
    style_dict_list = {
        'scatter': {'color': 'red'},
        'plot': {'color': 'blue'},
        'bar': {'color': 'green'},
    }
    visualization.visualize_2d(X, Y, fig_types=['plot'], style_dict_list=style_dict_list)
    visualization.visualize_2d(X, Y, fig_types=[])
