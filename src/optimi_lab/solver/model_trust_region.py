"""D68 -- trust-region model management (Alexandrov, Dennis, Lewis and Torczon, 1998) over any L6 surrogate.

The expensive function is only ever evaluated at points a CORRECTED model proposes, and the model is
trusted only as far as it has predicted well:

* MODEL -- the surrogate fitted on the told feasible points NEAR the centre (inside twice the
  trust box, at least the ``2d + 1`` nearest, distinct, spanning the space affinely) in coordinates
  ``(x - x_c) / radius`` -- a radial basis is not scale-free, and at a small radius the raw
  interpolation matrix underflows toward singular -- plus the zero-order correction ``f(x_c) - s(x_c)`` that makes it agree with the
  truth at the centre ``x_c`` (consistency the convergence theory asks of the model; a values-only
  oracle gives no first-order term to match). The default surrogate is ORBIT's (Wild, Regis and
  Shoemaker, 2008): a cubic radial basis interpolant with a linear tail -- a GP over every point
  was measured on 4-D Rosenbrock (2026-10-18) to mis-predict until the radius collapsed at
  f = 3.0 after 20 evaluations.
* STEP -- the corrected model minimised inside the trust box ``|x - x_c|_inf <= radius`` intersected
  with the unit cube, by :func:`~optimi_lab.solver.bayesian.maximise` (Sobol candidates, then
  numerics-lab's L-BFGS polish).
* RATIO -- ``rho = (f(x_c) - f(x)) / (m(x_c) - m(x))``; ``rho > 0`` moves the centre, ``rho > 0.75``
  with the step on the box boundary doubles the radius (to ``radius_max``).
* A FAILED STEP (``rho < 0.25``, or no predicted decrease) halves the radius only when the model is
  CERTIFIED -- at least ``2d + 1`` told points inside the trust box. Otherwise the next ask is a
  model-improvement point ``x_c +- radius e_j`` (directions in turn), as in ORBIT: shrinking on a
  model that has not seen the region is how the radius collapses away from a minimum.

The first ask is the centre and its ``2d`` axis neighbours at ``radius``. It concludes when the
radius falls below ``radius_end`` (unit-cube units). A constraint is an extreme barrier, as in
:func:`~optimi_lab.solver.bobyqa`: an infeasible or failed point scores NaN and never enters the model.
"""

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.model import RadialBasis, Surrogate
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.bayesian import maximise
from optimi_lab.solver.population import build
from optimi_lab.solver.sequential import Search, SequentialSolver

__all__ = ['ModelTrustRegion', 'model_trust_region']


class ModelTrustRegion(SequentialSolver):
    """Trust-region model management from ``start`` with a corrected *surrogate*."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        surrogate: Surrogate | None = None,
        radius: float = 0.2,
        radius_end: float = 1e-4,
        radius_max: float = 0.5,
        candidates: int | None = None,
        polish: int = 2,
        **base: object,
    ) -> None:
        if not 0.0 < radius_end < radius <= radius_max:
            refuse(f'need 0 < radius_end < radius <= radius_max, got {radius_end}, {radius}, {radius_max}')
        self._surrogate = surrogate or RadialBasis(kernel='cubic', degree=1)
        self._radius0, self._radius_end, self._radius_max = radius, radius_end, radius_max
        self._candidate_count, self._polish = candidates, polish
        super().__init__(problem, capability, **base)

    def _search(self) -> Search:
        n, radius = self._width, self._radius0
        candidates = self._candidate_count or min(128 * n, 2048)
        centre = self._start.copy()
        first = np.vstack([centre, np.clip(centre + np.vstack([np.eye(n), -np.eye(n)]) * radius, 0.0, 1.0)])
        scores, violation = yield first
        points: list[np.ndarray] = []
        values: list[float] = []
        for point, score, shortfall in zip(first, scores, violation, strict=True):
            _keep(points, values, point, score, shortfall)
        if not values:
            return
        best = int(np.argmin(values))
        centre, f_centre = points[best].copy(), values[best]
        direction, improve = 0, False
        while radius >= self._radius_end:
            stored, told = np.array(points), np.array(values)
            near = np.abs(stored - centre).max(axis=1)
            certified = np.count_nonzero(near <= radius * (1 + 1e-9)) >= 2 * n + 1
            if improve and not certified:
                trial = _geometry(centre, radius, direction, n)
                direction = (direction + 1) % (2 * n)
                scores, violation = yield trial[None]
                _keep(points, values, trial, scores[0], violation[0])
                if violation[0] == 0.0 and scores[0] < f_centre:
                    centre, f_centre = trial, float(scores[0])
                improve = False
                continue
            local = _local(stored, near, radius, n)
            fitted = self._surrogate.fit((stored[local] - centre) / radius, told[local][:, None])
            shift = f_centre - fitted.predict(np.zeros((1, n))).mean[0, 0]

            def model(x: np.ndarray, fitted=fitted, shift=shift, centre=centre, radius=radius) -> np.ndarray:
                return fitted.predict((x - centre) / radius).mean[:, 0] + shift

            lo, hi = np.maximum(centre - radius, 0.0), np.minimum(centre + radius, 1.0)
            step = maximise(
                lambda x, model=model: -model(x),
                n,
                rng=self._rng,
                candidates=candidates,
                polish=self._polish,
                lo=lo,
                hi=hi,
            )
            predicted = f_centre - model(step[None])[0]
            if not predicted > 0.0:
                radius, improve = (radius * 0.5, False) if certified else (radius, True)
                continue
            scores, violation = yield step[None]
            _keep(points, values, step, scores[0], violation[0])
            f_step = scores[0] if violation[0] == 0.0 else np.nan
            ratio = (f_centre - f_step) / predicted if not np.isnan(f_step) else -np.inf
            if ratio < 0.25:
                radius, improve = (radius * 0.5, False) if certified else (radius, True)
            elif ratio > 0.75 and np.any(np.isclose(np.abs(step - centre), radius, rtol=1e-3)):
                radius = min(2.0 * radius, self._radius_max)
            if ratio > 0.0:
                centre, f_centre = step.copy(), float(f_step)


def _local(stored: np.ndarray, near: np.ndarray, radius: float, n: int) -> np.ndarray:
    """The nearest told points: all inside twice the trust box, at least ``2d + 1``, distinct, AFFINELY SPANNING.

    Nearest first, a point within ``1e-3`` radii of one already taken is passed over (two such
    rows make the interpolation matrix singular at this scale), and the set grows until ``[1, x]``
    has full column rank ``d + 1`` -- a linear tail is singular on a lower-dimensional plane.
    """
    wanted = max(np.count_nonzero(near <= 2.0 * radius), 2 * n + 1)
    taken: list[int] = []
    for i in np.argsort(near, kind='stable'):
        if any(np.abs(stored[i] - stored[j]).max() < 1e-3 * radius for j in taken):
            continue
        taken.append(int(i))
        spanning = np.linalg.matrix_rank(np.column_stack([np.ones(len(taken)), stored[taken]])) > n
        if len(taken) >= wanted and spanning:
            break
    return np.array(taken)


def _keep(points: list[np.ndarray], values: list[float], point: np.ndarray, score: float, shortfall: float) -> None:
    """File a feasible computed point once: a repeated point would make an interpolant singular."""
    if shortfall == 0.0 and not np.isnan(score) and not any(np.array_equal(point, p) for p in points):
        points.append(point.copy())
        values.append(float(score))


def _geometry(centre: np.ndarray, radius: float, direction: int, n: int) -> np.ndarray:
    """The model-improvement point ``centre +- radius e_j``, turned back into the unit cube when it would leave it."""
    j, sign = direction % n, 1.0 if direction < n else -1.0
    if not 0.0 <= centre[j] + sign * radius <= 1.0:
        sign = -sign
    point = centre.copy()
    point[j] = np.clip(centre[j] + sign * radius, 0.0, 1.0)
    return point


def model_trust_region(problem: Problem, capability: Capability, **settings: object) -> ModelTrustRegion:
    """Trust-region model management; ``**settings`` are :class:`ModelTrustRegion`'s and the sequential base's."""
    return build(ModelTrustRegion, problem, capability, 'model_trust_region', **settings)
