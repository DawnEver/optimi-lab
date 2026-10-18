"""D123 multilevel Monte Carlo (Giles, 2008): ``E[P_L] = E[P_0] + sum_l E[P_l - P_{l-1}]``.

A *sampler* ``(level, n, rng) -> (corrections (n,), cost per sample)`` returns ``P_0`` at level 0 and
``P_l - P_{l-1}`` above it, the two sharing their randomness (the coupling that makes the
corrections' variance shrink -- MEC as the coarse level of a field solve is the same shape).

The algorithm is Giles' adaptive one, to a root-mean-square accuracy *epsilon*:

1. *n_initial* samples on each new level estimate its variance ``V_l`` and cost ``C_l``;
2. the sample counts ``N_l = ceil(2 / eps^2 sqrt(V_l / C_l) sum_k sqrt(V_k C_k))`` put the variance at
   ``eps^2 / 2``, and the missing samples are drawn;
3. the bias is estimated from the last two corrections, ``max(|Y_L|, |Y_{L-1}| / 2^a) / (2^a - 1)``,
   with the weak order *a* given or regressed from the corrections' means; while it exceeds
   ``eps / sqrt 2`` a level is added, up to *max_levels* (then it stops and says so, ``converged=False``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.operator.estimators import Estimate

__all__ = ['Multilevel', 'mlmc']

Sampler = Callable[[int, int, np.random.Generator], tuple[np.ndarray, float]]


@dataclass(frozen=True, slots=True, eq=False)
class Multilevel:
    """The pooled *estimate*, per-level *samples*, *variances* and *means*, the total *cost*, and whether it converged."""

    estimate: Estimate
    samples: tuple[int, ...]
    variances: tuple[float, ...]
    means: tuple[float, ...]
    cost: float
    converged: bool


@dataclass(slots=True)
class _Level:
    n: int = 0
    total: float = 0.0
    squares: float = 0.0
    cost: float = 1.0

    def add(self, sample: tuple[np.ndarray, float]) -> None:
        values, cost = sample
        values = np.ravel(np.asarray(values, dtype=float))
        self.n, self.total, self.squares = (
            self.n + values.size,
            self.total + values.sum(),
            self.squares + values @ values,
        )
        self.cost = float(cost)

    @property
    def mean(self) -> float:
        return self.total / self.n

    @property
    def variance(self) -> float:
        return max(self.squares / self.n - self.mean**2, 0.0)


def _order(levels: list[_Level], weak_order: float | None) -> float:
    if weak_order is not None:
        return weak_order
    rungs = np.arange(1, len(levels))
    means = np.abs([level.mean for level in levels[1:]])
    slope = np.polyfit(rungs, np.log2(np.maximum(means, 1e-300)), 1)[0]
    return max(0.5, float(-slope))


def mlmc(
    sampler: Sampler,
    epsilon: float,
    rng: np.random.Generator,
    *,
    n_initial: int = 1000,
    min_levels: int = 3,
    max_levels: int = 12,
    weak_order: float | None = None,
) -> Multilevel:
    """Giles' adaptive MLMC to accuracy *epsilon* (the unit of the sampled quantity)."""
    if not (epsilon > 0 and 2 <= min_levels <= max_levels and n_initial >= 2):
        refuse(
            f'need epsilon > 0, 2 <= min_levels <= max_levels and n_initial >= 2; got {epsilon!r}, '
            f'{min_levels}, {max_levels}, {n_initial}'
        )
    levels: list[_Level] = []
    for _ in range(min_levels):
        levels.append(_Level())
        levels[-1].add(sampler(len(levels) - 1, n_initial, rng))
    converged = False
    while True:
        while True:
            root = np.sqrt([level.variance * level.cost for level in levels]).sum()
            wanted = [int(np.ceil(2 / epsilon**2 * np.sqrt(lv.variance / lv.cost) * root)) for lv in levels]
            missing = [max(w - lv.n, 0) for w, lv in zip(wanted, levels, strict=True)]
            if not any(missing):
                break
            for index, extra in enumerate(missing):
                if extra:
                    levels[index].add(sampler(index, extra, rng))
        a = _order(levels, weak_order)
        bias = max(abs(levels[-1].mean), abs(levels[-2].mean) / 2**a) / (2**a - 1)
        if bias <= epsilon / np.sqrt(2):
            converged = True
            break
        if len(levels) == max_levels:
            break
        levels.append(_Level())
        levels[-1].add(sampler(len(levels) - 1, n_initial, rng))
    variance = sum(level.variance / level.n for level in levels)
    return Multilevel(
        Estimate(
            float(sum(level.mean for level in levels)), float(np.sqrt(variance)), sum(level.n for level in levels)
        ),
        tuple(level.n for level in levels),
        tuple(float(level.variance) for level in levels),
        tuple(float(level.mean) for level in levels),
        float(sum(level.n * level.cost for level in levels)),
        converged,
    )
