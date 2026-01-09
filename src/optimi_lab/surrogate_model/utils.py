import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from optimi_lab.utils.exceptions import ParameterException

# score methods
# https://scikit-learn.org/stable/api/sklearn.metrics.html

# Score method
# 1. Mean Squared Error
# 2. KL(Kullback-Leibler) divergence

SCORE_METHODS = ['r2', 'mse', 'rmse', 'mae', 'mad']


def score_regression(y_true: np.ndarray, y_pred: np.ndarray, score_methods: list[str] | None = None) -> dict:
    """Calculate regression scores for given true and predicted values.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        score_methods (list[str]): List of score methods to calculate. Defaults to SCORE_METHODS.

    Returns:
        dict: Dictionary with score method names as keys and calculated scores as values.

    """
    if score_methods is None:
        score_methods = SCORE_METHODS

    score_dict = {}
    for score_method in score_methods:
        if score_method == 'mse':
            score = mean_squared_error(y_true, y_pred)
        elif score_method == 'r2':
            score = r2_score(y_true, y_pred)
        elif score_method == 'rmse':
            score = root_mean_squared_error(y_true, y_pred)
        elif score_method == 'mae':
            score = mean_absolute_error(y_true, y_pred)
        elif score_method == 'mad':
            # median absolute deviation
            score = median_absolute_error(y_true, y_pred)
        else:
            msg = f'score method {score_method} is not supported, should be in {SCORE_METHODS}'
            raise KeyError(msg)
        score_dict[score_method] = score
    return score_dict


def float_int_1(v: float) -> float | int:
    if v <= 1 and v > 0:
        v = float(v)
    elif v >= 1:
        v = int(v)
    else:
        msg = f'Value should be at least 0, got {v}'
        raise ParameterException(msg)
    return v


def float_int_2(v: float) -> float | int:
    if v <= 1 and v > 0:
        v = float(v)
    elif v >= 2:
        v = int(v)
    else:
        msg = f'Value should be at least 0, got {v}'
        raise ParameterException(msg)
    return v
