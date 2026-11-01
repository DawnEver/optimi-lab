"""Steered variation: a policy sets the SBX/PM rates, the engine's own operator makes the children."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from optimi_lab.oracle import Result
from optimi_lab.solver.evolutionary import sbx_polynomial

__all__ = ['VariationPolicy']


class VariationPolicy:
    """:func:`sbx_polynomial` whose rates *variation* may update before each of the first *generations* calls.

    *read* returns the results the variation is shown (the routing policy's told results); *settings* are
    the starting rates. A variation answering None leaves the rates as they are.
    """

    def __init__(
        self,
        read: Callable[[], tuple[Result, ...]],
        variation: Callable[[tuple[Result, ...]], Mapping[str, float] | None],
        generations: int,
        settings: Mapping[str, object],
    ) -> None:
        self.read, self.variation = read, variation
        self.remaining = generations
        self.settings = dict(settings)

    def __call__(
        self, population: np.ndarray, width: int, rng: np.random.Generator, *, values: np.ndarray, violation: np.ndarray
    ) -> np.ndarray:
        if self.remaining > 0:
            updates = self.variation(self.read())
            if updates is not None:
                self.settings.update(updates)
            self.remaining -= 1
        return sbx_polynomial(population, width, rng, values=values, violation=violation, **self.settings)
