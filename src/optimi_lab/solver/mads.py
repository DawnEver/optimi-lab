"""MADS with the progressive barrier (Audet and Dennis, 2006 and 2009), OrthoMADS polling.

The mesh has size ``4 ** -l`` and the poll frame ``2 ** -l`` (unit-cube units). Each iteration polls
``x +- d_j`` for the ``2n`` columns of a Householder matrix of a fresh random direction (OrthoMADS,
Abramson et al., 2009), rounded onto the mesh; a poll point outside the unit cube is not evaluated
(the cube is an extreme barrier).

CONSTRAINTS ARE A PROGRESSIVE BARRIER on the total violation ``h`` (:meth:`Result.violation`):

* the feasible incumbent is the best feasible point; the infeasible incumbent is the point of
  least ``h`` among the non-dominated ``(h, f)`` filter with ``h <= h_max``;
* both are polled; an iteration is DOMINATING (a better feasible point, or an infeasible point
  dominating the infeasible incumbent) and the frame grows, IMPROVING (an infeasible point of
  smaller ``h``) and ``h_max`` drops to the largest filter ``h`` below the old incumbent's, or it
  FAILS, the frame shrinks and ``h_max`` drops to the infeasible incumbent's ``h``.

The search concludes when the frame is below ``tol`` (unit-cube units).
"""

import numpy as np

from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.population import build
from optimi_lab.solver.sequential import Search, SequentialSolver

__all__ = ['MADS', 'mads']


class MADS(SequentialSolver):
    """OrthoMADS from ``start`` with the progressive barrier; ``level`` is the initial mesh index."""

    def __init__(
        self, problem: Problem, capability: Capability, *, level: int = 2, tol: float = 1e-6, **base: object
    ) -> None:
        self._level0, self._tol = level, tol
        super().__init__(problem, capability, **base)

    def _directions(self, level: int) -> np.ndarray:
        n = self._width
        v = self._rng.standard_normal(n)
        v /= np.linalg.norm(v)
        householder = np.eye(n) - 2.0 * np.outer(v, v)
        mesh, frame = 4.0**-level, 2.0**-level
        columns = householder / np.abs(householder).max(axis=0)
        steps = mesh * np.round(frame / mesh * columns)
        return np.vstack([steps.T, -steps.T])

    def _poll(self, centre: np.ndarray, level: int) -> np.ndarray:
        trial = centre + self._directions(level)
        return trial[np.all((trial >= 0.0) & (trial <= 1.0), axis=1)]

    def _search(self) -> Search:
        level = self._level0
        scores, violation = yield self._start[None]
        feasible: tuple[np.ndarray, float] | None = None
        filter_: list[tuple[float, float, np.ndarray]] = []
        h_max = np.inf

        def admit(point: np.ndarray, f: float, h: float) -> None:
            nonlocal feasible
            if np.isnan(f) or np.isnan(h):
                return
            if h == 0.0:
                if feasible is None or f < feasible[1]:
                    feasible = (point, f)
            elif h <= h_max and not any(hh <= h and ff <= f for hh, ff, _ in filter_):
                filter_[:] = [(hh, ff, x) for hh, ff, x in filter_ if not (h <= hh and f <= ff)]
                filter_.append((h, f, point))

        admit(self._start, scores[0], violation[0])
        while 2.0**-level >= self._tol:
            infeasible = min(filter_, key=lambda e: (e[0], e[1])) if filter_ else None
            centres = [c for c in (feasible and feasible[0], infeasible and infeasible[2]) if c is not None]
            points = np.vstack([self._poll(c, level) for c in centres]) if centres else np.empty((0, self._width))
            if len(points) == 0:
                level += 1
                continue
            before_f = feasible[1] if feasible else np.inf
            scores, violation = yield points
            for point, f, h in zip(points, scores, violation, strict=True):
                admit(point, f, h)
            new_infeasible = min(filter_, key=lambda e: (e[0], e[1])) if filter_ else None
            dominating = (feasible is not None and feasible[1] < before_f) or (
                infeasible is not None
                and new_infeasible is not None
                and new_infeasible[0] <= infeasible[0]
                and new_infeasible[1] <= infeasible[1]
                and (new_infeasible[0], new_infeasible[1]) != (infeasible[0], infeasible[1])
            )
            improving = infeasible is not None and new_infeasible is not None and new_infeasible[0] < infeasible[0]
            if dominating:
                level = max(0, level - 1)
            elif improving:
                below = [hh for hh, _, _ in filter_ if hh < infeasible[0]]
                h_max = max(below) if below else infeasible[0]
            else:
                level += 1
                if new_infeasible is not None:
                    h_max = new_infeasible[0]
            filter_[:] = [e for e in filter_ if e[0] <= h_max]


def mads(problem: Problem, capability: Capability, **settings: object) -> MADS:
    """MADS with the progressive barrier; ``**settings`` are :class:`MADS`'s and the sequential base's."""
    return build(MADS, problem, capability, 'mads', **settings)
