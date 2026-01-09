import random
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from .exceptions import ParameterException

__all__ = ['config_plt_params', 'visualize_2d']


def config_plt_params(**kwargs) -> None:
    """Configure matplotlib parameters.
    See matplotlib.rcParams:
    https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams

    Args:
        **kwargs: Parameters to update in plt.rcParams.

    Example:
        ```python
        config_plt_params(**{'figure.figsize': (8, 6), 'figure.dpi': 100, 'font.family': ['Times New Roman']})
        ```

    """
    param_dict = {
        'axes.unicode_minus': True,
        'figure.constrained_layout.use': True,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'font.family': [
            'Times New Roman',
            'STFangsong',
        ],
    }
    param_dict.update(kwargs)

    try:
        plt.rcParams.update(param_dict)
    except KeyError as e:
        msg = f'One or more keys are not valid plt.rcParams keys. Please check the input keys: {e}'
        raise ParameterException(msg) from e


config_plt_params()

"""
Visualization utilities
"""


def visualize_2d(
    X: np.ndarray | list[np.ndarray],
    Y: np.ndarray | list[np.ndarray],
    xlabel: str = '',
    ylabel: str = '',
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    xsticks: dict | None = None,
    ysticks: dict | None = None,
    fig_types: list[str] | None = None,
    style_dict_list: list[dict] | None = None,
    savefig: bool = False,
    fig_path: Path = './visualize_2d.png',
    legend_dict: dict | None = None,
    equal_aspect: bool = False,
    title: str = '',
):
    """2-variable visualization.

    Args:
        X (np.ndarray | list[np.array]): First variable. Can be a numpy.ndarray or a list of np.ndarray.
        Y (np.ndarray | list[np.array]): Second variable. Same type as X.
        xlabel (str): Label for x-axis. Default is empty.
        ylabel (str): Label for y-axis. Default is empty.
        xlim (tuple | None): Range for x-axis as (xmin, xmax). Default is None.
        ylim (tuple | None): Range for y-axis as (ymin, ymax). Default is None.
        xsticks (dict | None): x-axis ticks settings as a dict. Default is None.
        ysticks (dict | None): y-axis ticks settings as a dict. Default is None.
        fig_types (list[str] | None): List of plot types to draw. Default is ['scatter'].
            - 'scatter': scatter plot
            - 'plot': line plot
            - 'bar': bar chart
        style_dict_list (list[dict] | None): List of style dictionaries for each plot. If None, a random color is generated.
        savefig (bool): Whether to save the figure. Default is False.
        fig_path (Path): Path to save the figure. Default is './visualize_2d.png'.
        legend_dict (dict | None): Legend settings as a dict. Default is None.
        equal_aspect (bool): Whether to set equal aspect ratio. Default is False.
        title (str): Plot title. Default is empty.

    Example:
        ```python
        X = np.random.rand(100)
        Y = np.random.rand(100)
        plt = visualize_2d(X, Y, xlabel='X-axis', ylabel='Y-axis', fig_types=['scatter', 'plot'])
        plt.show()
        ```

    """
    if fig_types is None:
        fig_types = ['scatter']
    is_ndarray = True
    if isinstance(X, np.ndarray):
        x_shape = X.shape
        y_shape = Y.shape
        if x_shape != y_shape:
            msg = f'x.shape {x_shape} != y.shape {y_shape}'
            raise ParameterException(msg)
    elif isinstance(X, list):
        assert isinstance(X[0], np.ndarray)
        assert isinstance(Y, list)
        assert isinstance(Y[0], np.ndarray)
        is_ndarray = False
    else:
        msg = f'X or Y type {type(X)} {type(Y)} is not supported'
        raise ParameterException(msg)

    if 1:
        # generate random RGB color values
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        # convert to hexadecimal format
        hex_color = f'#{r:02x}{g:02x}{b:02x}'

    if style_dict_list is None:
        style_dict_list = [
            {
                'scatter': {'color': hex_color},
                'plot': {'color': hex_color},
                'bar': {'color': hex_color},
            },
        ]
    if not isinstance(style_dict_list, list):
        style_dict_list = [style_dict_list]
    n_styles = len(style_dict_list)

    plt.clf()
    ax = plt.subplot()

    def draw(x, y, index=0) -> None:
        for fig_type in fig_types:
            fig_type_ = fig_type.lower()
            if fig_type_ == 'bar':
                ax.bar(x, y, **style_dict_list[index][fig_type_])
            elif fig_type_ == 'plot':
                ax.plot(x, y, **style_dict_list[index][fig_type_])
            elif fig_type_ == 'scatter':
                ax.scatter(x, y, **style_dict_list[index][fig_type_])

    if is_ndarray:  # X is numpy.ndarray
        if X.ndim == 1:
            draw(X, Y)
        else:
            for i in range(len(X)):
                x = X[i, :]
                y = Y[i, :]
                draw(x, y, index=i % n_styles)
    else:  # X is list[np.array]
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            draw(x, y, index=i % n_styles)
    if legend_dict is not None:
        ax.legend(loc='best', **legend_dict)
    if equal_aspect:
        ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xsticks is not None:
        ax.set_xticks(**xsticks)
    if ysticks is not None:
        ax.set_yticks(**ysticks)

    if savefig:
        plt.savefig(fig_path)
    return plt
