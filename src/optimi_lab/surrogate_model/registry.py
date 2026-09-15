"""The declared regressors: one table, keyed by the name a pool entry writes.

The old package spread these over ten near-identical modules, each a pydantic model whose
``@model_validator(mode='after')`` built its estimator on the side. Three measured defects die with
that shape: nine of the ten validators returned ``None`` where ``Bagging`` returned ``self``
(pydantic warned on every construction); ``KNearestNeighbors.n_neighbors`` read 3 while the
estimator was given ``min(len(var_name_list), n_neighbors)`` — measured 1/2/3/3 for 1/2/3/5 declared
variables — so the field and the estimator disagreed, and no name list exists here for a count to be
derived from; and an unregistered key was refused by a message that did not list the valid keys,
while a constant holding exactly those keys sat one module away.

A parameter a row does not name keeps scikit-learn's own default; each row declares the parameters
whose value departs from it, plus the ones read as "a fraction of the whole or a count of items".
"""

import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from optimi_lab.core.errors import refuse

__all__ = ['REGISTRY', 'SURROGATE_KEYS', 'Regressor', 'fraction_or_count', 'resolve']


def fraction_or_count(value: object, floor: int, what: str) -> object:
    """Return a parameter scikit-learn reads as "a fraction of the whole or a count of items".

    The reading is the TYPE's, as scikit-learn reads it: an int is a count of at least ``floor``, a
    float is a fraction within ``(0, 1]`` — 1.0 is the whole set — and a value that is neither is
    refused, naming both readings and the floor it refused. The pair of helpers this replaces
    answered "Value should be at least 0" for floors of 1 and 2 and for a value of 1.5, which is
    above zero; measured, they disagreed across the whole interval between their floors, one
    truncating 1.5 to the count 1 and the other refusing it.
    """
    if isinstance(value, (str, type(None))):
        return value  # scikit-learn's own non-numeric readings ('sqrt', 'log2', None) are not this rule's business
    numeric = not isinstance(value, bool) and isinstance(value, numbers.Real)
    is_count = numeric and isinstance(value, numbers.Integral) and value >= floor
    is_fraction = numeric and not isinstance(value, numbers.Integral) and 0.0 < value <= 1.0
    if not (is_count or is_fraction):
        refuse(f'{what} must be a fraction within (0, 1] or a count of at least {floor}, got {value!r}')
    return int(value) if is_count else float(value)


@dataclass(frozen=True, slots=True)
class Regressor:
    """One registered estimator: how to build it, its departing defaults, its count parameters.

    ``counts`` maps a parameter to the smallest count scikit-learn accepts for it, which is the
    floor :func:`fraction_or_count` is given.
    """

    key: str
    estimator: Callable[..., Any]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)

    def build(self, params: Mapping[str, object]) -> object:
        """Return a fresh estimator: the row's defaults, overridden by ``params``."""
        settings = {**self.defaults, **params}
        for name, floor in self.counts.items():
            if name in settings:
                settings[name] = fraction_or_count(settings[name], floor, f'{name} of the {self.key!r} regressor')
        return self.estimator(**settings)


def _polynomial_regression(**settings: object) -> Pipeline:
    """A polynomial expansion of ``degree`` feeding a least-squares fit, as one estimator."""
    degree = settings.pop('degree', 3)
    expansion = PolynomialFeatures(degree=degree, include_bias=False)
    return make_pipeline(expansion, LinearRegression(**settings))


_TREE_DEFAULTS = {'min_samples_leaf': 2, 'max_leaf_nodes': 100, 'max_features': 1.0}
_TREE_COUNTS = {'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 1}

REGISTRY: dict[str, Regressor] = {
    'pol': Regressor('pol', _polynomial_regression, {'degree': 3}),
    'knn': Regressor('knn', KNeighborsRegressor, {'n_neighbors': 3}),
    'kri': Regressor('kri', GaussianProcessRegressor),
    'rid': Regressor('rid', Ridge),
    'las': Regressor('las', Lasso),
    'mlp': Regressor('mlp', MLPRegressor),
    'dec': Regressor('dec', DecisionTreeRegressor, _TREE_DEFAULTS, _TREE_COUNTS),
    'extra': Regressor('extra', ExtraTreeRegressor, _TREE_DEFAULTS, _TREE_COUNTS),
    'forest': Regressor('forest', RandomForestRegressor, _TREE_DEFAULTS, _TREE_COUNTS),
    'bag': Regressor('bag', BaggingRegressor, {'max_samples': 1.0}, {'max_samples': 1, 'max_features': 1}),
}

SURROGATE_KEYS = tuple(REGISTRY)
"""The registered keys — a refusal names this tuple rather than a constant in another module."""


def resolve(key: object, what: str) -> Regressor:
    """Return the regressor ``key`` names, or refuse naming every registered key."""
    if not isinstance(key, str) or key not in REGISTRY:
        refuse(f'{what} must be one of {list(SURROGATE_KEYS)}, got {key!r}')
    return REGISTRY[key]
