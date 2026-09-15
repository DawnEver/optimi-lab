"""Surrogate models: one fit over a registered estimator, and a mixture over several fits.

:class:`~optimi_lab.core.protocols.Surrogate` says what a surrogate IS — ``fit`` returns the fitted
model, ``predict`` reads it back — and a surrogate is a fit, never a mode. A prediction is not an
:class:`~optimi_lab.core.outcomes.Outcome`, so nothing here can hand one to ``Proposer.tell`` and
have it scored as if the points had run. No module here reads a variable name list, a configuration
file or a validity flag.

    >>> import numpy as np
    >>> from optimi_lab.surrogate_model import MixtureSurrogate, SurrogateModel
    >>> x = np.linspace(0.0, 1.0, 40).reshape(-1, 2)
    >>> y = np.column_stack([np.sin(2 * np.pi * x[:, 0]), x[:, 1] ** 2])
    >>> SurrogateModel('rid', n_splits=3).fit(x, y).predict(x[:3]).shape
    (3, 2)
    >>> mixture = MixtureSurrogate([{'model_type': 'rid'}, {'model_type': 'knn', 'n_neighbors': 3}])
    >>> mixture.fit(x, y).predict(x[:3]).shape
    (3, 2)

The keys a pool entry may name are declared in :mod:`optimi_lab.surrogate_model.registry`, and a
refusal names every one of them.
"""

from optimi_lab.surrogate_model.base import SurrogateModel
from optimi_lab.surrogate_model.mixture import WEIGHT_METHODS, MixtureSurrogate, WeightMethod
from optimi_lab.surrogate_model.registry import (
    REGISTRY,
    SURROGATE_KEYS,
    Regressor,
    fraction_or_count,
    resolve,
)
from optimi_lab.surrogate_model.utils import SCORE_METHODS, score_regression

__all__ = [
    'REGISTRY',
    'SCORE_METHODS',
    'SURROGATE_KEYS',
    'WEIGHT_METHODS',
    'MixtureSurrogate',
    'Regressor',
    'SurrogateModel',
    'WeightMethod',
    'fraction_or_count',
    'resolve',
    'score_regression',
]
