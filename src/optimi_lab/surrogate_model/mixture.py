"""The mixture: a pool of fitted surrogates, combined by weight.

The pool is REQUIRED and non-empty at construction. It used to default to ``[]`` and be filled by
``build_surrogate_model_pool``, a method with no call site anywhere in the library, so the headline
model's default configuration failed as ``TypeError: 'NoneType' object is not subscriptable`` out
of ``_predict``, naming neither the pool nor the keys. The constructor resolves the keys, and the
refusal for an empty pool names the registered types.

The weight is the member's cross-validated R2 and nothing else. ``_train`` used to deep-copy the
pool, train the copy and drop it, leaving the members unfitted; :meth:`MixtureSurrogate.fit` fits the
members it holds and returns the mixture, so there is no copy to lose. A pool whose every member
scored ``r2 <= 0`` is REFUSED rather than silently averaged — the old code answered with a uniform
weight array, which is a mean and not a weight. And the ``max`` rule, which indexed a
``(n_model, 1, 1)`` weight array with ``np.argmax`` of itself, selects by the member axis.
"""

from collections.abc import Iterable
from enum import StrEnum
from typing import Self

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse
from optimi_lab.surrogate_model.base import SurrogateModel
from optimi_lab.surrogate_model.registry import SURROGATE_KEYS

__all__ = ['WEIGHT_METHODS', 'MixtureSurrogate', 'WeightMethod']


class WeightMethod(StrEnum):
    """How the pool's predictions become one prediction."""

    LINEAR = 'linear'
    MAX = 'max'


WEIGHT_METHODS = tuple(method.value for method in WeightMethod)
"""The declared weight methods — an unknown one is refused naming this tuple."""


class MixtureSurrogate:
    """A pool of surrogates fitted on the same data and combined by their R2."""

    __slots__ = ('_fitted', '_pool', '_weight_method')

    def __init__(self, pool: Iterable[object] | None, weight_method: str = 'linear') -> None:
        entries = () if pool is None else tuple(pool)
        self._pool = tuple(SurrogateModel.from_declaration(entry) for entry in entries)
        if not self._pool:
            refuse(
                f'a mixture combines a pool of at least one surrogate, got none; name the members as the '
                f'registered types, e.g. pool=[{{"model_type": "pol"}}], from {list(SURROGATE_KEYS)}'
            )
        self._weight_method = coerce_enum(WeightMethod, weight_method, 'the mixture weight_method')
        self._fitted = False

    @property
    def pool(self) -> tuple[SurrogateModel, ...]:
        """The members, in the order they were declared."""
        return self._pool

    def fit(self, inputs: np.ndarray, values: np.ndarray) -> Self:
        """Fit every member on the same measured data, and return the mixture."""
        for member in self._pool:
            member.fit(inputs, values)
        self._fitted = True
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return the combined prediction, shape ``(n_points, n_obj)``."""
        if not self._fitted:
            refuse('this mixture has not been fitted; call fit(inputs, values) and predict with what it returns')
        predictions = np.stack([member.predict(inputs) for member in self._pool])
        weights = self._weights()
        if self._weight_method is WeightMethod.MAX:
            return predictions[int(np.argmax(weights))]
        return np.tensordot(weights, predictions, axes=1)

    def _weights(self) -> np.ndarray:
        """The members' R2, clamped at zero and normalised, or a refusal if no member has one."""
        measured = np.array([member.scores['r2'] for member in self._pool], dtype=float)
        weights = np.maximum(np.where(np.isfinite(measured), measured, 0.0), 0.0)
        if not np.any(weights > 0.0):
            scored = ', '.join(f'{member.key}: {r2:g}' for member, r2 in zip(self._pool, measured, strict=True))
            refuse(
                f'every member of the mixture scored r2 <= 0 in its own cross-validation ({scored}), so the '
                f'weights would be a uniform mean of the members and not a weight'
            )
        return weights / weights.sum()
