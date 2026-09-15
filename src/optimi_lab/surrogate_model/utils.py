"""Scoring a fit, always against the values that were measured.

``score_regression(y_true, y_pred)`` was called with its pair SWAPPED by its only caller, so the R2
the mixture weighted its members by was measured against the mean of the PREDICTIONS. The
absolute-error statistics are symmetric in the pair and hid the swap; R2 is not, and R2 is what the
weighting reads. The count-and-fraction reading that lived here as two near-identical helpers is one
function in :mod:`~optimi_lab.surrogate_model.registry`, beside the table that applies it.
"""

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from optimi_lab.core.errors import refuse

__all__ = ['SCORE_METHODS', 'score_regression']

SCORE_METHODS = ('r2', 'mse', 'rmse', 'mae', 'mad')
"""The declared score methods, in one place — a refusal names this tuple rather than a constant
in another module."""

_SCORERS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    'r2': r2_score,
    'mse': mean_squared_error,
    'rmse': root_mean_squared_error,
    'mae': mean_absolute_error,
    'mad': median_absolute_error,  # scikit-learn's median of |y_true - y_pred|, not a deviation from the median
}


def score_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_methods: Sequence[str] = SCORE_METHODS,
) -> dict[str, float]:
    """Score a prediction against the measured values, one entry per method.

    The measured values come FIRST: ``r2_score`` is not symmetric in its pair, and the caller of
    this function is the fit that produced the prediction.
    """
    measured = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    unsupported = [method for method in score_methods if method not in _SCORERS]
    if unsupported:
        refuse(f'score method(s) {unsupported} are not supported; the declared methods are {list(SCORE_METHODS)}')
    return {method: float(_SCORERS[method](measured, predicted)) for method in score_methods}
