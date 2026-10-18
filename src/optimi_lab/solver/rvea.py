"""RVEA (Cheng, Jin, Olhofer and Sendhoff, 2016): reference vectors and the angle-penalised distance.

The generation is NSGA-II's variation over randomly paired parents; survival keeps, per reference
vector, the member of least angle-penalised distance
(:func:`optimi_lab.operator.angle_penalized_select`). The penalty grows with the fraction of the
run elapsed, so RVEA must be told how many generations it has (``generations``) -- the solver never
sees the budget. Every ``adapt_every`` of that span the vectors are rescaled to the population's
objective spans (:func:`optimi_lab.operator.adapt_directions`).

Constraints: when the union holds a feasible row, survival runs over the feasible rows only;
otherwise the least-violating rows survive. The population is one row per OCCUPIED vector, so it
may be smaller than the vector count; ``pop_size`` offspring are still made each generation.
"""

import numpy as np

from optimi_lab.core import partition_count, refuse
from optimi_lab.operator import adapt_directions, angle_penalized_select, das_dennis
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.evolutionary import sbx_polynomial
from optimi_lab.solver.population import EvolutionarySolver, build

__all__ = ['ReferenceVectorSolver', 'rvea']


class ReferenceVectorSolver(EvolutionarySolver):
    """RVEA's loop; ``alpha`` is the penalty's growth exponent, ``variation_settings`` the SBX/PM rates."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        generations: int,
        alpha: float = 2.0,
        adapt_every: float = 0.1,
        variation_settings: dict[str, float] | None = None,
        **base: object,
    ) -> None:
        if generations < 1:
            refuse(f'generations is the number of generations RVEA plans for, at least 1, got {generations!r}')
        if not 0.0 < adapt_every <= 1.0:
            refuse(f'adapt_every is a fraction of the planned generations, in (0, 1], got {adapt_every!r}')
        super().__init__(
            problem,
            capability,
            variation=sbx_polynomial,
            variation_settings=variation_settings,
            selection=self._survive,
            parent_selection=self._mate,
            **base,
        )
        self._generations, self._alpha = generations, alpha
        self._period = max(1, round(adapt_every * generations))
        self._generation_count = 0
        self._initial: np.ndarray | None = None
        self._directions: np.ndarray | None = None

    def _mate(self, values: np.ndarray, n: int, rng: np.random.Generator, *, violation: np.ndarray) -> np.ndarray:
        """RVEA pairs parents uniformly at random, ``pop_size`` of them whatever the survivors number."""
        del n, violation
        return rng.integers(0, len(values), self._pop_size)

    def _survive(self, values: np.ndarray, n: int, violation: np.ndarray) -> np.ndarray:
        del n
        if self._initial is None:
            self._initial = das_dennis(values.shape[1], partition_count(values.shape[1], self._pop_size))
            self._directions = self._initial
        self._generation_count += 1
        progress = min(1.0, self._generation_count / self._generations)
        feasible = np.flatnonzero(violation == 0.0)
        if len(feasible) == 0:
            return np.argsort(violation, kind='stable')[: len(self._initial)]
        kept = feasible[
            angle_penalized_select(values[feasible], directions=self._directions, progress=progress, alpha=self._alpha)
        ]
        if self._generation_count % self._period == 0:
            self._directions = adapt_directions(self._initial, values[kept])
        return kept


def rvea(problem: Problem, capability: Capability, *, generations: int, **settings: object) -> ReferenceVectorSolver:
    """RVEA planned for *generations* generations; ``**settings`` are :class:`ReferenceVectorSolver`'s."""
    return build(ReferenceVectorSolver, problem, capability, 'rvea', generations=generations, **settings)
