from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from optimi_lab.utils.config import PathData
from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.logger import log

CORRELATION_METHODS = ['pearson', 'kendall', 'spearman']


def correlation_analysis(data_frame: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """Perform correlation analysis on a DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame to analyze.
        method (str): The method of correlation to use. Options are 'pearson', 'kendall', 'spearman'.

    Returns:
        pd.DataFrame: A DataFrame containing the correlation coefficients.

    """
    if method not in CORRELATION_METHODS:
        msg = f"Method must be one of {CORRELATION_METHODS}, got '{method}' instead."
        raise ValueError(msg)
    return data_frame.corr(method=method)


def plot_matrix(
    data_frame: pd.DataFrame = None,
    savefig: bool = False,
    fig_path: Path = PathData.default_fig_path,
    use_abs: bool = True,
    ticks: list | None = None,
    fontsize: int = 16,
) -> plt:
    """Plot a matrix from a DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame to plot.
        savefig (bool): Whether to save the figure.
        fig_path (Path): Path to save the figure if `savefig` is True.
        use_abs (bool): Whether to use absolute values in the plot.
        ticks (list | None): Custom ticks for the colorbar. If None, default ticks are used.
        fontsize (int): Font size for the labels.

    Returns:
        plt: The matplotlib plot object.

    Notes:
        The input `data_frame` should be in a rectangular form where rows are indices
        and columns are numeric variables, e.g.:
        ```
                    column_0,  column_1    ...
        index_0,   ...         ...
        index_1,   ...
        ...
        ```

    """
    ax = plt.subplot()

    if use_abs:
        data_frame = data_frame.abs()
        if ticks is None:
            ticks = [0, 0.5, 1]
    elif ticks is None:
        ticks = [-1, -0.5, 0, 0.5, 1]

    length_y = data_frame.shape[0]
    length_x = data_frame.shape[1]

    matrix = data_frame.to_numpy()
    matrix_map = ax.imshow(matrix, cmap='Wistia')
    for i in range(length_y):
        for j in range(length_x):
            ax.text(j, i, round(matrix[i, j], 3), fontsize=fontsize, ha='center', va='center')
    ax.set_xticks(range(len(data_frame.columns)), data_frame.columns, rotation=30, fontsize=fontsize)
    ax.set_yticks(range(len(data_frame.index)), data_frame.index, rotation=30, fontsize=fontsize)
    ax.legend()
    log(msg='Plot is done', level='DEBUG')

    plt.colorbar(mappable=matrix_map, ticks=ticks)
    if savefig:
        plt.savefig(fig_path)
    return plt


def pair_plot(
    data_frame: pd.DataFrame,
    diagonal: str = 'hist',
    scatter_kwds: dict | None = None,
    hist_kwds: dict | None = None,
    density_kwds: dict | None = None,
    savefig: bool = False,
    fig_path: Path = PathData.default_fig_path,
) -> plt:
    """Plot a scatterplot matrix.

    Args:
        data_frame (pd.DataFrame): DataFrame to plot.
        diagonal (str): How to draw diagonal plots; one of 'hist', 'kde', or 'density'.
        scatter_kwds (dict): Keyword args for scatter plots.
        hist_kwds (dict): Keyword args for histograms.
        density_kwds (dict): Keyword args for density plots.
        savefig (bool): Whether to save the figure.
        fig_path (str): Path to save the figure.

    Examples:
        ```python
        data_frame = pd.DataFrame({'a': np.random.rand(100), 'b': np.random.rand(100), 'c': np.random.rand(100)})
        plt = pair_plot(data_frame, diagonal='hist', scatter_kwds={'alpha': 0.5})
        plt.show()
        ```

    """
    if density_kwds is None:
        density_kwds = {'alpha': 0.1, 'marker': '.'}
    if hist_kwds is None:
        hist_kwds = {}
    if scatter_kwds is None:
        scatter_kwds = {}
    df_ = data_frame._get_numeric_data()
    n = df_.columns.size
    assert n > 1, 'DataFrame must have at least two numeric columns'

    plt.clf()
    _, axs = plt.subplots(n, n)

    boundaries_list = []
    range_padding = 0.05
    for a in df_.columns:
        values = df_[a].to_numpy()
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2
        boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))

    for i, a in enumerate(df_.columns):
        for j, b in enumerate(df_.columns):
            ax = axs[i, j]

            if i == j:
                values = df_[a].to_numpy()
                # Deal with the diagonal by drawing a histogram there.
                if diagonal == 'hist':
                    ax.hist(values, **hist_kwds)

                elif diagonal in {'kde', 'density'}:
                    y = values
                    gkde = gaussian_kde(y)
                    ind = np.linspace(y.min(), y.max(), 1000)
                    ax.plot(ind, gkde.evaluate(ind), **density_kwds)
                else:
                    msg = f"Invalid diagonal argument '{diagonal}'. Expected 'hist', 'kde', or 'density'."
                    raise ParameterException(msg)
                ax.set_xlim(boundaries_list[i])

            else:
                ax.scatter(df_[b], df_[a], **scatter_kwds)

                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])

            ax.set_xlabel(b)
            ax.set_ylabel(a)

            if j != 0:
                ax.yaxis.set_visible(False)
            if i != n - 1:
                ax.xaxis.set_visible(False)

    lim1 = boundaries_list[0]
    locs = axs[0][1].yaxis.get_majorticklocs()
    locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
    adj = (locs - lim1[0]) / (lim1[1] - lim1[0])

    lim0 = axs[0][0].get_ylim()
    adj = adj * (lim0[1] - lim0[0]) + lim0[0]
    axs[0][0].yaxis.set_ticks(adj)

    axs[0][0].yaxis.set_ticklabels(locs)

    if savefig:
        plt.savefig(fig_path)
    return plt
