import itertools
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from optimi_lab.utils.config import PathData
from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.logger import log


def parse_cols_to_draw(*axises, var_name_list: list[str], obj_name_list: list[str]) -> tuple[list, list]:
    """Parse variables and objectives from axis strings.

    Args:
        axises(list[str]): list of axis strings, e.g. ['var#0', 'obj#1']
        var_name_list(list[str]): list of variable names
        obj_name_list(list[str]): list of objective names

    Returns:
        var_to_draw_list: list of variable names to draw
        obj_to_draw_list: list of objective names to draw

    """
    var_to_draw_list = []
    obj_to_draw_list = []
    for axis_str in axises:
        if axis_str is None:
            continue
        # axis_str=axis_str.lower()
        axis_list = axis_str.split('#')
        if len(axis_list) != 2:
            msg = f'Invalid axis type {axis_str}. It should be in form of var#index or obj#index.'
            raise ParameterException(msg)
        axis_type = axis_list[0]
        axis_index = int(axis_list[1])

        if axis_type in ['var', 'variable']:
            n_var = len(var_name_list)
            if axis_index > n_var - 1:
                msg = f'Invalid axis index {axis_str}'
                log(msg, level='ERROR')
                raise IndexError(msg)
            var_to_draw_list.append(var_name_list[axis_index])
        elif axis_type in ['obj', 'object']:
            n_obj = len(obj_name_list)
            if axis_index > n_obj - 1:
                msg = f'Invalid axis index {axis_str}'
                log(msg, level='ERROR')
                raise IndexError(msg)
            obj_to_draw_list.append(obj_name_list[axis_index])
        else:
            msg = f'Invalid axis type {axis_str}'
            raise ParameterException(msg)
    return var_to_draw_list, obj_to_draw_list


def plot(
    var_name_list: list[str],
    obj_name_list: list[str],
    data_frame: pd.DataFrame,
    x_axis: str | None = None,
    y_axis: str | None = None,
    z_axis: str | None = None,
    axises: list[str] | None = None,
    savefig: bool = False,
    fig_path: str = PathData.default_fig_path,
    scatter: bool = True,
    plot: bool = True,
    surface: bool = False,
    colorful: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    fontsize: int = 16,
    style_plot: dict | None = None,
    style_surface: dict | None = None,
    style_scatter: dict | None = None,
) -> plt:
    """Figure function compatible with HMDFigure.

    Allows selecting variables and objectives to plot. x_axis, y_axis, z_axis should be
    in the form var#index or obj#index.
    """
    if axises is None:
        axises = [x_axis, y_axis, z_axis]
    if style_plot is None:
        style_plot = {
            'c': '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]),
        }
    if style_surface is None:
        style_surface = {
            'cmap': 'rainbow',
        }
    if style_scatter is None:
        style_scatter = {
            'marker': 'o',
            'c': '#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]),
        }

    var_to_draw_list, obj_to_draw_list = parse_cols_to_draw(
        *axises, var_name_list=var_name_list, obj_name_list=obj_name_list
    )
    column_to_draw_list = var_to_draw_list + obj_to_draw_list
    len_cols = len(column_to_draw_list)
    if len_cols == 0:
        return plt

    if len_cols in [1, 2]:
        ax = plt.subplot()
    elif len_cols == 3:
        ax = plt.axes(projection='3d')
    else:
        msg = f'The number of valid var/obj is {len_cols}'
        raise ParameterException(msg)

    def draw_points(*args):
        if plot:
            ax.plot(*args, **style_plot)
        if surface:
            if len(args) <= 2:
                msg = 'Cannot draw surface with two axises'
                raise ParameterException(msg)
            try:
                ax.plot_trisurf(*args, **style_surface)
            except RuntimeError as e:
                msg = f'Unsolved matplotlib exception: {e}'
                log(msg, level='WARNING')
        if scatter:
            ax.scatter(*args, **style_scatter)

    def draw(df_to_draw, index: int = 0, xlabel=None, ylabel=None, zlabel=None):
        x = df_to_draw[column_to_draw_list[0]]
        if xlabel is None:
            xlabel = column_to_draw_list[0]
        ax.set_xlabel(xlabel, fontsize=fontsize)

        if len_cols == 1:
            y = np.zeros(len(x)) + index  # index used to distinguish different groups when `colorful` is True
        else:
            y = df_to_draw[column_to_draw_list[1]]
            if ylabel is None:
                ylabel = column_to_draw_list[1]
            ax.set_ylabel(ylabel, fontsize=fontsize)
            if len_cols > 2:
                z = df_to_draw[column_to_draw_list[2]]
                if zlabel is None:
                    zlabel = column_to_draw_list[2]
                ax.set_zlabel(zlabel, fontsize=fontsize)
                draw_points(x, y, z)
            else:
                draw_points(x, y)

    if colorful:
        # Build combinations of values for variables that are not plotted
        var_values_not_to_draw = []
        var_name_not_to_draw = [var_name for var_name in var_name_list if var_name not in column_to_draw_list]
        for var_name in var_name_not_to_draw:
            values = data_frame[var_name].unique().tolist()
            var_values_not_to_draw.append(values)

        var_values_not_to_draw_matrix = list(itertools.product(*var_values_not_to_draw))
        for index, var_values_not_to_draw_list in enumerate(var_values_not_to_draw_matrix):
            mask = data_frame[var_name_not_to_draw].isin(var_values_not_to_draw_list).all(axis=1)
            # Use boolean indexing to get rows that match the condition
            draw(data_frame[mask], index, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    else:
        draw(data_frame, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    log(msg='Plot is done', level='DEBUG')

    if savefig:
        plt.savefig(fig_path)
    return plt


def plot_moo(
    var_name_list: list[str],
    obj_name_list: list[str],
    df_all: pd.DataFrame,
    df_pareto: pd.DataFrame,
    x_axis: str | None = None,
    y_axis: str | None = None,
    z_axis: str | None = None,
    axises: list[str] | None = None,
    savefig: bool = False,
    fig_path: str = PathData.default_fig_path,
    draw_pareto_point: bool = False,
    draw_pareto_front: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    fontsize: int | None = None,
    style_scatter: dict | None = None,
    style_scatter_pareto: dict | None = None,
) -> plt:
    if axises is None:
        axises = [x_axis, y_axis, z_axis]
    draw_pareto_point = draw_pareto_point or draw_pareto_front

    # Get column_to_draw_list
    var_to_draw_list, obj_to_draw_list = parse_cols_to_draw(
        *axises, var_name_list=var_name_list, obj_name_list=obj_name_list
    )
    column_to_draw_list = var_to_draw_list + obj_to_draw_list
    len_cols = len(column_to_draw_list)
    if len_cols == 0:
        return plt

    mask = df_all.isin(df_pareto)
    df_all = df_all[~mask].dropna()

    if style_scatter is None:
        style_scatter = {
            'marker': 'o',
            'color': 'blue',
            'alpha': 0.8,
        }

    if style_scatter_pareto is None:
        style_scatter_pareto = {
            'marker': '*',
            'color': 'red',
        }

    if len_cols in [1, 2]:
        ax = plt.subplot()
    elif len_cols == 3:
        ax = plt.axes(projection='3d')
    else:
        msg = f'The number of valid var/obj is {len_cols}.'
        raise ParameterException(msg)

    if len_cols == 1:
        x = df_all[column_to_draw_list[0]].to_numpy()
        y = np.zeros(len(x))
        ax.scatter(x, y, **style_scatter)

        if draw_pareto_point:
            x = df_pareto[column_to_draw_list[0]].to_numpy()
            ax.scatter(x, y, **style_scatter_pareto)
            if draw_pareto_front:
                ax.plot(x, y)
        if xlabel is None:
            xlabel = column_to_draw_list[0]
        ax.set_xlabel(xlabel, fontsize=fontsize)
    elif len_cols == 2:
        x = df_all[column_to_draw_list[0]].to_numpy()
        y = df_all[column_to_draw_list[1]].to_numpy()
        ax.scatter(x, y, **style_scatter)

        if draw_pareto_point:
            x = df_pareto[column_to_draw_list[0]].to_numpy()
            y = df_pareto[column_to_draw_list[1]].to_numpy()
            ax.scatter(x, y, **style_scatter_pareto)
            if draw_pareto_front:
                ax.plot(x, y)
        if xlabel is None:
            xlabel = column_to_draw_list[0]
        ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel is None:
            ylabel = column_to_draw_list[1]
        ax.set_ylabel(ylabel, fontsize=fontsize)
    else:  # len_cols == 3:
        x = df_all[column_to_draw_list[0]].to_numpy()
        y = df_all[column_to_draw_list[1]].to_numpy()
        z = df_all[column_to_draw_list[2]].to_numpy()
        ax.scatter(x, y, z, **style_scatter)

        if draw_pareto_point:
            x = df_pareto[column_to_draw_list[0]].to_numpy()
            y = df_pareto[column_to_draw_list[1]].to_numpy()
            z = df_pareto[column_to_draw_list[2]].to_numpy()
            ax.scatter(xs=x, ys=y, zs=z, **style_scatter_pareto)
            len_z = len(z)
            if draw_pareto_front and len_z > 2:
                try:
                    ax.plot_trisurf(x, y, z)
                except RuntimeError as e:
                    msg = f'Unsolved matplotlib exception: {e}'
                    log(msg, level='ERROR')
                """
                Exception has occurred: RuntimeError
                Error in qhull Delaunay triangulation calculation:
                singular input data (exitcode=2); use python verbose
                option (-v) to see original qhull error.
                """
        if xlabel is None:
            xlabel = column_to_draw_list[0]
        ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel is None:
            ylabel = column_to_draw_list[1]
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if zlabel is None:
            zlabel = column_to_draw_list[2]
        ax.set_zlabel(zlabel, fontsize=fontsize)

    ax.legend()
    log(msg='Plot is done', level='DEBUG')
    if savefig:
        plt.savefig(fig_path)
    return plt


def plot_contour(
    var_name_list: list[str],
    obj_name_list: list[str],
    data_frame: pd.DataFrame,
    x_axis: str | None = None,
    y_axis: str | None = None,
    z_axis: str | None = None,
    axises: list[str] | None = None,
    savefig: bool = False,
    fig_path: str = PathData.default_fig_path,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    fontsize: int = 16,
    fill: bool = True,
    styles_contourf: dict | None = None,
    styles_contour: dict | None = None,
) -> plt:
    """Plot contour (filled or lines).

    Figure compatible with HMDFigure. Allows selection of variables and objectives.
    """
    if axises is None:
        axises = [x_axis, y_axis, z_axis]

    if styles_contourf is None:
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
        styles_contourf = {
            'levels': 10,
        }
    if styles_contour is None:
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html
        styles_contour = {
            'levels': 10,
        }
    var_to_draw_list, obj_to_draw_list = parse_cols_to_draw(
        *axises, var_name_list=var_name_list, obj_name_list=obj_name_list
    )
    column_to_draw_list = var_to_draw_list + obj_to_draw_list
    len_cols = len(column_to_draw_list)
    if len_cols == 0:
        return plt

    if len_cols < 3:
        msg = 'Plot failed: at least three variables are required to plot.'
        raise ParameterException(msg)
    if len_cols > 3:
        """
        # TODO Feature
        # Consider drawing contour plots for each objective when there are multiple objectives.
        """
        if len(obj_to_draw_list) > 2 or len(var_to_draw_list) > 2:
            msg = ''
            raise NotImplementedError(msg)
        msg = f'Plot failed: variable number {len_cols} is larger than 3; not enough variables and objectives: {len(var_to_draw_list)},{len(obj_to_draw_list)}'
        raise ParameterException(msg)

    ax = plt.subplot()

    x = data_frame[column_to_draw_list[0]]
    if xlabel is None:
        xlabel = column_to_draw_list[0]
    ax.set_xlabel(xlabel, fontsize=fontsize)
    y = data_frame[column_to_draw_list[1]]
    if ylabel is None:
        ylabel = column_to_draw_list[1]
    ax.set_ylabel(ylabel, fontsize=fontsize)
    z = data_frame[column_to_draw_list[2]].to_numpy()
    if zlabel is None:
        zlabel = column_to_draw_list[2]

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    x, y = np.meshgrid(unique_y, unique_x)
    l_unique_x = len(unique_x)
    l_unique_y = len(unique_y)
    if l_unique_x < 2 or l_unique_y < 2:
        msg = f'Plot failed: x/y ticks number must be larger than 2. (x,y) = ({l_unique_x},{l_unique_y})'
        raise ParameterException(msg)

    l_z = len(z)

    if l_z > l_unique_x * l_unique_y:
        # Often z contains multiple x*y groups
        # Here we only take the first group
        msg = f'Multiple x*y combinations detected in z. x ticks={len(unique_x)}, y ticks={len(unique_y)}, z length={len(z)}'
        log(msg=msg, level='WARNING')
        # There should be an option to select the N-th group
        z = z[: l_unique_x * l_unique_y]
    elif l_z < l_unique_x * l_unique_y:
        # z length is insufficient — this is invalid
        # Pad with zeros to match expected length
        msg = f'Data length not matched! x ticks={len(unique_x)}, y ticks={len(unique_y)}, z length={len(z)}'
        log(msg=msg, level='ERROR')
        z = np.pad(array=z, pad_width=(0, l_unique_x * l_unique_y - l_z), mode='minimum')

    z = z.reshape((l_unique_x, l_unique_y))

    if fill:
        conter_plot = ax.contourf(x, y, z, **styles_contourf)
    else:
        conter_plot = ax.contour(x, y, z, **styles_contour)
    ax.set_title(zlabel, fontsize=fontsize)
    ax.legend()
    log(msg='Plot is done', level='DEBUG')

    plt.colorbar(mappable=conter_plot)
    if savefig:
        plt.savefig(fig_path)
    return plt
