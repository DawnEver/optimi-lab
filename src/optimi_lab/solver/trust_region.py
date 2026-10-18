"""A BOBYQA-like derivative-free trust region (after Powell, 2009): quadratic models inside a box.

Not Powell's code, the same idea: an interpolation quadratic built from evaluated points, minimised
inside the trust box ``|s|_inf <= radius`` intersected with the unit cube, and a radius driven by
the ratio of actual to predicted decrease.

* MODEL -- the minimum-norm least-squares quadratic (``numerics_lab.linear.dense.least_squares`` over the monomials
  ``1, s_i, s_i s_j / 2``) through the evaluated points nearest the incumbent; with fewer points
  than coefficients it is the minimum-norm interpolant, which is how BOBYQA's minimum-Frobenius
  model starts from ``2n + 1`` points.
* STEP -- projected gradient on the model inside the box, step ``1 / L`` with ``L`` the Hessian's
  spectral norm.
* RADIUS -- doubled (to at most 0.5) on ratio >= 0.7 at the boundary, halved below 0.1, when a
  geometry point ``x +- radius e_j`` (coordinates in turn) is also asked to repair the model.

It concludes when the radius falls below ``radius_end`` (unit-cube units). A constraint is an
extreme barrier: an infeasible point scores NaN and never enters the model.
"""

import numpy as np
from numerics_lab.linear.dense import least_squares

from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.population import build
from optimi_lab.solver.sequential import Search, SequentialSolver

__all__ = ['TrustRegion', 'bobyqa']

_STEPS = 60
"""Projected-gradient iterations on the model per trust-region step."""


def _features(steps: np.ndarray) -> np.ndarray:
    n = steps.shape[1]
    upper = np.triu_indices(n)
    scale = np.where(upper[0] == upper[1], 0.5, 1.0)
    return np.hstack([np.ones((len(steps), 1)), steps, steps[:, upper[0]] * steps[:, upper[1]] * scale])


def _model(steps: np.ndarray, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian at the incumbent of the min-norm quadratic through *steps* -> *values*."""
    coefficients = least_squares(_features(steps), values)
    gradient = coefficients[1 : n + 1]
    hessian = np.zeros((n, n))
    hessian[np.triu_indices(n)] = coefficients[n + 1 :]
    return gradient, hessian + np.triu(hessian, 1).T


def _minimise(gradient: np.ndarray, hessian: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    lipschitz = max(np.linalg.norm(hessian, 2), 1e-12)
    step = np.zeros_like(gradient)
    for _ in range(_STEPS):
        step = np.clip(step - (gradient + hessian @ step) / lipschitz, lo, hi)
    return step


class TrustRegion(SequentialSolver):
    """The trust region from ``start`` with initial ``radius`` and final ``radius_end`` (unit-cube units)."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        radius: float = 0.1,
        radius_end: float = 1e-6,
        **base: object,
    ) -> None:
        self._radius0, self._radius_end = radius, radius_end
        super().__init__(problem, capability, **base)

    def _search(self) -> Search:
        n, radius = self._width, self._radius0
        x = self._start.copy()
        offsets = np.vstack([np.eye(n), -np.eye(n)]) * radius
        first = np.vstack([x, np.clip(x + offsets, 0.0, 1.0)])
        scores, violation = yield first
        points = [p for p, f, h in zip(first, scores, violation, strict=True) if h == 0.0 and not np.isnan(f)]
        values = [f for f, h in zip(scores, violation, strict=True) if h == 0.0 and not np.isnan(f)]
        if not values:
            return
        best = int(np.argmin(values))
        x, fx = points[best], values[best]
        coordinate = 0
        size = (n + 1) * (n + 2) // 2
        while radius >= self._radius_end:
            stored = np.array(points)
            nearest = np.argsort(np.linalg.norm(stored - x, axis=1))[: size + n]
            gradient, hessian = _model(stored[nearest] - x, np.array(values)[nearest] - fx, n)
            lo, hi = np.maximum(-radius, -x), np.minimum(radius, 1.0 - x)
            step = _minimise(gradient, hessian, lo, hi)
            predicted = -(gradient @ step + 0.5 * step @ hessian @ step)
            if predicted <= 0.0 or np.abs(step).max() < 1e-3 * radius:
                radius *= 0.5
                continue
            sign = 1.0 if x[coordinate] + radius <= 1.0 else -1.0
            geometry = x.copy()
            geometry[coordinate] += sign * radius
            trial = np.vstack([x + step, geometry])
            scores, violation = yield trial
            for p, f, h in zip(trial, scores, violation, strict=True):
                if h == 0.0 and not np.isnan(f):
                    points.append(p)
                    values.append(float(f))
            actual = fx - scores[0] if violation[0] == 0.0 else -np.inf
            ratio = actual / predicted
            if ratio >= 0.7 and np.abs(step).max() >= 0.99 * radius:
                radius = min(2.0 * radius, 0.5)
            elif not ratio >= 0.1:
                radius *= 0.5
            coordinate = (coordinate + 1) % n
            best = int(np.argmin(values))
            x, fx = points[best].copy(), values[best]


def bobyqa(problem: Problem, capability: Capability, **settings: object) -> TrustRegion:
    """The BOBYQA-like trust region; ``**settings`` are :class:`TrustRegion`'s and the sequential base's."""
    return build(TrustRegion, problem, capability, 'bobyqa', **settings)
