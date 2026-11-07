"""A bagging surrogate: one declared surrogate fitted on bootstrap resamples, the fits averaged.

:class:`Bagging` fits ``n_members`` copies of *member*, each on ``round(max_samples * n)`` rows drawn
with replacement by ``default_rng(seed)``, and predicts their mean. It always gives a variance: the
spread of the member means (law of total variance, plus each member's own variance when the member
gives one). It gives a gradient when the member does -- the mean of the member gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.model.protocol import Fitted, Give, Prediction, Surrogate, require_gives, training_data

__all__ = ['Bagging', 'FittedBagging']


@dataclass(frozen=True, slots=True)
class Bagging:
    """*n_members* fits of *member* on resamples of ``max_samples`` of the rows, drawn by *seed*."""

    member: Surrogate
    n_members: int = 10
    max_samples: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_members < 2:
            refuse(f'bagging averages at least two members, got {self.n_members}')
        if not 0.0 < self.max_samples <= 1.0:
            refuse(f'max_samples is a share of the rows in (0, 1], got {self.max_samples}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.VARIANCE}) | (self.member.gives & {Give.GRADIENT})

    def fit(self, x, y) -> FittedBagging:
        x, y = training_data(x, y)
        rng = np.random.default_rng(self.seed)
        size = max(1, round(self.max_samples * len(x)))
        draws = [rng.integers(0, len(x), size) for _ in range(self.n_members)]
        return FittedBagging(tuple(self.member.fit(x[d], y[d]) for d in draws), self.gives, self.member.gives)


@dataclass(frozen=True, slots=True, eq=False)
class FittedBagging:
    """The fitted members, averaged with equal weight."""

    members: tuple[Fitted, ...]
    gives: frozenset[Give]
    member_gives: frozenset[Give]

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='this bagging surrogate')
        own = variance and Give.VARIANCE in self.member_gives
        parts = [m.predict(x, variance=own, gradient=gradient) for m in self.members]
        means = np.stack([p.mean for p in parts])
        var = None
        if variance:
            var = means.var(axis=0) + (np.mean([p.variance for p in parts], axis=0) if own else 0.0)
        slope = np.mean([p.gradient for p in parts], axis=0) if gradient else None
        return Prediction(means.mean(axis=0), var, slope)
