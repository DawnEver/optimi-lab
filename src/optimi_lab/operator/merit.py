"""Merit functions: one number that says whether a trial point is better, objective and constraints together.

* The EXACT l1 PENALTY ``f + nu (||h||_1 + ||max(c, 0)||_1)``: for ``nu`` above the largest
  multiplier its local minimisers are the constrained ones. :func:`l1_slope` bounds its directional
  derivative along ANY step through the violation the step removes from the linearised constraints
  (:func:`l1_reduction`) -- the slope an Armijo search (numerics-lab) is handed. A least-violating
  step from an infeasible linearisation removes only PART of the violation; a slope that assumed all
  of it promised a decrease no step length had (2026-11-13: an SQP refine on exact gradients
  stalled). :func:`l1_penalty` raises ``nu`` until a step that removes violation descends.
* The AUGMENTED LAGRANGIAN (Rockafellar's form for inequalities), smooth in ``x`` for fixed
  multipliers and penalty, which is what lets a bound-constrained gradient method minimise it.
"""

from __future__ import annotations

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.operator.kkt import Linearization

__all__ = ['augmented_lagrangian_merit', 'l1_merit', 'l1_penalty', 'l1_reduction', 'l1_slope']

#: How far ``nu`` is kept above the largest multiplier, relatively.
_MARGIN = 1.5
#: The share of ``nu * reduction`` the penalty keeps as the step's guaranteed l1 descent.
_DESCENT = 0.1


def l1_merit(point: Linearization, nu: float) -> float:
    """``f + nu * violation``."""
    return point.objective + nu * point.violation


def l1_reduction(point: Linearization, direction: np.ndarray) -> float:
    """The violation *direction* removes from the linearised constraints: ``violation - ||h + dh d||_1 -
    ||max(c + dc d, 0)||_1``. All of it for a step that satisfies the linearisation, part of it for a least-violating
    one, and negative for a step that breaks it.
    """
    h = point.h + point.dh @ direction
    c = point.c + point.dc @ direction
    return point.violation - float(np.sum(np.abs(h)) + np.sum(np.maximum(c, 0.0)))


def l1_slope(point: Linearization, direction: np.ndarray, nu: float) -> float:
    """``grad f . d - nu * l1_reduction``: an upper bound on the l1 merit's derivative along ANY step *direction*
    (the linearised violation is convex along it), equal to it where no linearised row crosses zero at the point.
    """
    return float(point.gradient @ direction) - nu * l1_reduction(point, direction)


def l1_penalty(multipliers: np.ndarray, previous: float, *, descent: float = 0.0, reduction: float = 0.0) -> float:
    """``nu`` for the l1 merit: never below *previous*, at least ``1.5 * max |multiplier|``, and, when a step removes
    *reduction* > 0 of the linearised violation while the objective changes at *descent* along it, at least
    ``descent / ((1 - _DESCENT) * reduction)`` -- so its :func:`l1_slope` is ``<= -_DESCENT * nu * reduction``, a
    descent (Nocedal and Wright, eq. 18.36). The multipliers alone say nothing at a step whose QP was infeasible.
    """
    largest = float(np.max(np.abs(multipliers), initial=0.0))
    nu = max(previous, _MARGIN * largest)
    if reduction > 0.0:
        nu = max(nu, descent / ((1.0 - _DESCENT) * reduction))
    return nu


def augmented_lagrangian_merit(
    point: Linearization, y: np.ndarray, z: np.ndarray, rho: float
) -> tuple[float, np.ndarray]:
    """``(value, gradient)`` of ``f + y.h + rho/2 |h|^2 + sum psi(c_i, z_i)``, ``psi = (max(0, z + rho c)^2 - z^2) / 2 rho``."""
    if not rho > 0:
        refuse(f'an augmented Lagrangian needs a penalty rho > 0, got {rho!r}')
    shifted = np.maximum(0.0, z + rho * point.c)
    value = (
        point.objective
        + float(y @ point.h)
        + 0.5 * rho * float(point.h @ point.h)
        + float(np.sum(shifted**2 - z**2)) / (2.0 * rho)
    )
    gradient = point.gradient + point.dh.T @ (y + rho * point.h) + point.dc.T @ shifted
    return value, gradient
