"""Merit functions: one number that says whether a trial point is better, objective and constraints together.

* The EXACT l1 PENALTY ``f + nu (||h||_1 + ||max(c, 0)||_1)``: for ``nu`` above the largest
  multiplier its local minimisers are the constrained ones. :func:`l1_slope` is the bound on its
  directional derivative along a step that satisfies the linearised constraints, which is the slope
  an Armijo search (numerics-lab) is handed.
* The AUGMENTED LAGRANGIAN (Rockafellar's form for inequalities), smooth in ``x`` for fixed
  multipliers and penalty, which is what lets a bound-constrained gradient method minimise it.
"""

from __future__ import annotations

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.operator.kkt import Linearization

__all__ = ['augmented_lagrangian_merit', 'l1_merit', 'l1_penalty', 'l1_slope']

#: How far ``nu`` is kept above the largest multiplier, relatively.
_MARGIN = 1.5


def l1_merit(point: Linearization, nu: float) -> float:
    """``f + nu * violation``."""
    return point.objective + nu * point.violation


def l1_slope(point: Linearization, direction: np.ndarray, nu: float) -> float:
    """``grad f . d - nu * violation``: an upper bound on the l1 merit's derivative along a linearised step."""
    return float(point.gradient @ direction) - nu * point.violation


def l1_penalty(multipliers: np.ndarray, previous: float) -> float:
    """``nu`` for the l1 merit: never below *previous*, and at least ``1.5 * max |multiplier|``."""
    largest = float(np.max(np.abs(multipliers), initial=0.0))
    return max(previous, _MARGIN * largest)


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
