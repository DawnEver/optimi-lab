"""D121 first- and second-order reliability: the design point and the curvature correction.

A limit state ``g`` maps standard-normal rows ``(n, d) -> (n,)``; failure is ``g <= 0`` (see
:mod:`optimi_lab.operator.reliability`).

* :func:`form` -- the design point ``u* = argmin |u| s.t. g(u) = 0`` by the improved HL-RF iteration
  (Zhang and Der Kiureghian, 1995): the HL-RF direction, its length chosen by an Armijo search
  (numerics-lab's) on the merit ``|u|^2 / 2 + c |g(u)|``. ``beta = sign(g(0)) |u*|`` and
  ``pf ~ Phi(-beta)``, exact for a linear ``g``. Without an analytic *gradient*, central
  differences of step :data:`STEP` (standard units) are taken, one batched call per point.
* :func:`sorm` -- Hohenbichler and Rackwitz's form of Breitung's asymptotic formula,
  ``pf ~ Phi(-beta) prod (1 + psi kappa_i)^-1/2`` with ``psi = phi(beta) / Phi(-beta)`` (Breitung's
  ``beta`` is its large-``beta`` limit; measured 0.5 % against 2.3 % off at ``beta = 3, k = 0.2``),
  the principal curvatures ``kappa_i`` of the limit-state surface at ``u*`` read off the Hessian of
  ``g`` (central second differences of step :data:`CURVATURE_STEP`) projected on the tangent plane
  and divided by ``|grad g|``. Positive curvature bends the failure set away from the origin.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numerics_lab.line_search import armijo
from scipy.special import ndtr

from optimi_lab.core.errors import refuse

__all__ = ['CURVATURE_STEP', 'STEP', 'DesignPoint', 'SecondOrder', 'form', 'sorm']

#: Central-difference step of a FORM gradient, in standard-normal units.
STEP = 1e-6
#: Central second-difference step of a SORM Hessian, in standard-normal units.
CURVATURE_STEP = 1e-4


@dataclass(frozen=True, slots=True, eq=False)
class DesignPoint:
    """The most probable failure point *design_point*, its index *beta*, ``Phi(-beta)`` and the cost."""

    design_point: np.ndarray
    beta: float
    probability: float
    gradient: np.ndarray
    evaluations: int
    converged: bool


@dataclass(frozen=True, slots=True, eq=False)
class SecondOrder:
    """The second-order *probability* from the principal *curvatures* at the design point."""

    probability: float
    curvatures: np.ndarray
    evaluations: int


def _scalar(g: Callable, rows: np.ndarray) -> np.ndarray:
    return np.ravel(np.asarray(g(np.atleast_2d(rows)), dtype=float))


def form(
    g: Callable[[np.ndarray], np.ndarray],
    width: int,
    *,
    gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    start=None,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> DesignPoint:
    """The improved HL-RF iteration; see the module docstring. *tolerance* is in standard units."""
    u = np.zeros(width) if start is None else np.asarray(start, dtype=float).copy()
    evaluations = 0

    def value_and_gradient(at: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        if gradient is not None:
            evaluations += 1
            return float(_scalar(g, at)[0]), np.asarray(gradient(at), dtype=float)
        shifts = np.vstack([at, at + STEP * np.eye(width), at - STEP * np.eye(width)])
        values = _scalar(g, shifts)
        evaluations += len(shifts)
        return float(values[0]), (values[1 : width + 1] - values[width + 1 :]) / (2 * STEP)

    value, slope = value_and_gradient(u)
    scale = max(abs(value), 1.0)
    for _ in range(max_iterations):
        norm = float(np.linalg.norm(slope))
        if norm == 0.0:
            refuse('the limit state has a zero gradient: no design point direction exists here')
        alpha = -slope / norm
        if abs(value) <= tolerance * scale and np.linalg.norm(u - (alpha @ u) * alpha) <= tolerance:
            return _point(u, slope, evaluations, converged=True)
        d = (float(slope @ u) - value) / norm**2 * slope - u
        c = 2.0 * max(float(np.linalg.norm(u)) / norm, 1.0)

        def merit(t: float, d: np.ndarray = d, c: float = c, base: np.ndarray = u) -> float:
            nonlocal evaluations
            evaluations += 1
            trial = base + t * d
            return 0.5 * float(trial @ trial) + c * abs(float(_scalar(g, trial)[0]))

        merit_slope = float((u + c * np.sign(value) * slope) @ d)
        step = 1.0
        if merit_slope < 0:
            search = armijo(merit, 0.5 * float(u @ u) + c * abs(value), merit_slope)
            if not search.accepted:
                return _point(u, slope, evaluations, converged=False)
            step = search.step
        u = u + step * d
        value, slope = value_and_gradient(u)
    return _point(u, slope, evaluations, converged=False)


def _point(u: np.ndarray, slope: np.ndarray, evaluations: int, *, converged: bool) -> DesignPoint:
    beta = float(np.linalg.norm(u)) * (1.0 if float(slope @ u) <= 0 else -1.0)
    return DesignPoint(u, beta, float(ndtr(-beta)), slope, evaluations, converged)


def sorm(g: Callable[[np.ndarray], np.ndarray], point: DesignPoint) -> SecondOrder:
    """The Hohenbichler-Rackwitz second-order probability at a FORM *point*; see the module docstring."""
    u, h = point.design_point, CURVATURE_STEP
    width = u.size
    eye = h * np.eye(width)
    pairs = [(i, j) for i in range(width) for j in range(i + 1, width)]
    rows = [u, *(u + e for e in eye), *(u - e for e in eye)]
    for i, j in pairs:
        rows += [u + eye[i] + eye[j], u + eye[i] - eye[j], u - eye[i] + eye[j], u - eye[i] - eye[j]]
    values = _scalar(g, np.vstack(rows))
    hessian = np.diag((values[1 : width + 1] - 2 * values[0] + values[width + 1 : 2 * width + 1]) / h**2)
    for k, (i, j) in enumerate(pairs):
        pp, pm, mp, mm = values[2 * width + 1 + 4 * k : 2 * width + 5 + 4 * k]
        hessian[i, j] = hessian[j, i] = (pp - pm - mp + mm) / (4 * h * h)
    alpha = -point.gradient / np.linalg.norm(point.gradient)
    basis = np.linalg.qr(np.column_stack([alpha, np.eye(width)]))[0][:, :width]  # first column is +-alpha
    tangent = basis[:, 1:]
    curvatures = np.linalg.eigvalsh(tangent.T @ hessian @ tangent / np.linalg.norm(point.gradient))
    psi = float(np.exp(-0.5 * point.beta**2) / np.sqrt(2 * np.pi) / ndtr(-point.beta))
    factors = 1.0 + psi * curvatures
    if np.any(factors <= 0.0):
        refuse(f'the asymptotic formula needs 1 + psi kappa > 0; psi={psi!r}, curvatures={curvatures.tolist()}')
    probability = float(ndtr(-point.beta) / np.sqrt(np.prod(factors)))
    return SecondOrder(probability, curvatures, len(rows))
