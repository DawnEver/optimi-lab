"""The fit: one parameterised surrogate over the registered estimators.

The old base carried three things this one has nowhere to put. ``train``/``_train`` were two
training paths, and the mixture used the one that deep-copied its members, trained the COPY and
dropped it — so the members stayed unfitted and the default configuration raised ``AttributeError:
'NoneType' object has no attribute 'predict'``. ``check_valid``/``_valid`` was a flag consulted by
nothing, True before and after training. ``do_calc_score`` decided whether a member could be SCORED,
and therefore whether the mixture had a weight: with it off every weight fell back to a default and
the mixture became an unweighted mean while its docstring said "weight is r2 alone". Here the only
fit is the one a caller calls, scoring is what a fit produces, and the value shape is ONE shape —
``PolinomialRegression`` and ``KNearestNeighbors`` returned ``(n, 1)`` where ``RidgeRegression``
returned ``(n,)``, so the mixture could not stack them and a search died on one broadcast error.
"""

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from optimi_lab.core.errors import refuse, require_axis, require_finite
from optimi_lab.surrogate_model.registry import Regressor, resolve
from optimi_lab.surrogate_model.utils import SCORE_METHODS, score_regression

__all__ = ['SurrogateModel']

_FOLD_SEED = 0
"""The folds are seeded: the scores decide the mixture's weights, and weights that move between two
runs of the same data are not a measurement."""


def _require_points(values: np.ndarray, what: str) -> np.ndarray:
    """Return a ``(n_points, n_columns)`` matrix of finite numbers, or refuse."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        refuse(f'{what} must be a 2-D array with one row per point, got shape {array.shape}')
    return require_finite(array, what)


class SurrogateModel:
    """A registered estimator, fitted on measured data, carrying its cross-validated scores.

    The model is declared once — ``SurrogateModel('bag', n_estimators=5)`` — and every parameter is
    scikit-learn's own, passed through to it. Nothing here reads a variable or objective name: the
    columns are the caller's, in the order their ``VariableSet`` declared them.
    """

    __slots__ = ('_estimator', '_n_obj', '_n_splits', '_params', '_regressor', '_score_methods', '_scores')

    def __init__(
        self, key: str, *, score_methods: Sequence[str] = SCORE_METHODS, n_splits: int = 5, **params: object
    ) -> None:
        self._regressor: Regressor = resolve(key, 'the surrogate model type')
        self._params = dict(params)
        self._score_methods = tuple(score_methods)
        unsupported = [method for method in self._score_methods if method not in SCORE_METHODS]
        if unsupported:
            refuse(f'score method(s) {unsupported} are not supported; the declared methods are {list(SCORE_METHODS)}')
        if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
            refuse(f'the number of scoring folds must be an integer of at least 2, got {n_splits!r}')
        self._n_splits = n_splits
        self._estimator: object = None
        self._scores: dict[str, float] | None = None
        self._n_obj = 0

    @classmethod
    def from_declaration(cls, declaration: object) -> Self:
        """Return the model a pool entry names: a model itself, or a mapping keyed by ``model_type``.

        Every other entry of the mapping is a parameter, so ``{'model_type': 'knn', 'n_neighbors':
        3}`` and ``SurrogateModel('knn', n_neighbors=3)`` are the same model.
        """
        if isinstance(declaration, cls):
            return declaration
        if not isinstance(declaration, Mapping):
            refuse(
                f'every member of a pool must be a {cls.__name__} or a mapping with a model_type key, '
                f'got {type(declaration).__name__}'
            )
        settings = dict(declaration)
        return cls(settings.pop('model_type', None), **settings)

    @property
    def key(self) -> str:
        """The registered key this model was built from."""
        return self._regressor.key

    @property
    def estimator(self) -> object:
        """The fitted scikit-learn estimator, or a refusal naming the key if there is none yet."""
        return self._require_fitted()

    @property
    def scores(self) -> dict[str, float]:
        """The fitted model's cross-validated scores, one entry per declared method."""
        self._require_fitted()
        return dict(self._scores or {})

    def fit(self, inputs: np.ndarray, values: np.ndarray) -> Self:
        """Fit on ``(n_points, n_var)`` inputs and ``(n_points, n_obj)`` MEASURED values.

        A row of ``values`` is a measured one — what ``Evaluation.complete`` hands out. One model
        is kept, fitted on every row; the folds that score it never decide what it is fitted on.
        """
        inputs = _require_points(inputs, 'the inputs a surrogate is fitted on')
        values = _require_points(values, 'the values a surrogate is fitted on')
        if inputs.shape[0] != values.shape[0]:
            refuse(
                f'the values must hold one row per point, got {values.shape[0]} row(s) for {inputs.shape[0]} point(s)'
            )
        template = self._regressor.build(self._params)
        estimator = clone(template).fit(inputs, values)
        scores = self._cross_validated_scores(template, inputs, values)
        # Assigned together and last: a fit that raised above leaves the model UNFITTED, not
        # half-fitted, which is the state `_require_fitted` exists to name.
        self._n_obj, self._estimator, self._scores = values.shape[1], estimator, scores
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return predictions of shape ``(n_points, n_obj)``, refusing any other shape.

        A prediction is a number the caller may use and must label as one; it is not an
        ``Evaluation`` and cannot be told to a proposer as if the points had run.
        """
        estimator = self._require_fitted()
        points = _require_points(inputs, 'the inputs a surrogate predicts on')
        predicted = np.asarray(estimator.predict(points), dtype=float)
        if predicted.ndim == 1:
            predicted = predicted[:, np.newaxis]  # one objective is a column, never a flat vector
        return require_finite(
            require_axis(predicted, self._n_obj, 1, f'the prediction of the {self.key!r} surrogate'),
            f'the prediction of the {self.key!r} surrogate',
        )

    def _require_fitted(self) -> object:
        """Return the fitted estimator, or refuse naming the key and what to call instead."""
        if self._estimator is None:
            refuse(
                f'the {self._regressor.key!r} surrogate has not been fitted; call fit(inputs, values) and '
                f'predict with what it returns'
            )
        return self._estimator

    def _cross_validated_scores(self, template: object, inputs: np.ndarray, values: np.ndarray) -> dict[str, float]:
        """Score held-out rows: the folds that score are never the rows that were fitted."""
        n_splits = min(self._n_splits, len(inputs))
        predictions, measured = [], []
        for train_index, test_index in KFold(n_splits=n_splits, shuffle=True, random_state=_FOLD_SEED).split(inputs):
            fold = clone(template).fit(inputs[train_index], values[train_index])
            predictions.append(fold.predict(inputs[test_index]))
            measured.append(values[test_index])
        return score_regression(np.concatenate(measured), np.concatenate(predictions), self._score_methods)
