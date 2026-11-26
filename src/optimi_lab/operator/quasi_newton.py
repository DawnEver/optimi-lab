"""Dense quasi-Newton models for trust-region and SQP strategies: BFGS and SR1 Hessian updates.

:func:`bfgs_update`, :func:`bfgs_inverse_update` and :func:`sr1_update` update a small dense model.

Every update keeps its secant equation exactly in exact arithmetic (``H y = s`` or ``B s = y``), and
an update whose curvature vanishes is SKIPPED (the model returned unchanged), never divided through.
The secant pair, its two-loop recursion, the L-BFGS pair memory and Broyden's root-finding updates
are numerics-lab's (``numerics_lab.quasi_newton``): its nonlinear solves and its basic minimizer run on them.
"""

from __future__ import annotations

import numpy as np
from numerics_lab.quasi_newton import SECANT_FLOOR
from numpy.typing import NDArray

__all__ = [
    'bfgs_inverse_update',
    'bfgs_update',
    'sr1_update',
]

Vector = NDArray[np.float64]


def bfgs_inverse_update(h: NDArray[np.float64], s: Vector, y: Vector) -> NDArray[np.float64]:
    """``(I - rho s y^T) H (I - rho y s^T) + rho s s^T``; ``H`` unchanged when ``y . s <= 0``.

    A non-positive curvature would destroy positive definiteness, so that pair is skipped.
    """
    ys = float(y @ s)
    if not ys > SECANT_FLOOR:
        return h
    rho = 1.0 / ys
    hy = h @ y
    return h - rho * (np.outer(s, hy) + np.outer(hy, s)) + (rho * rho * float(y @ hy) + rho) * np.outer(s, s)


def bfgs_update(b: NDArray[np.float64], s: Vector, y: Vector) -> NDArray[np.float64]:
    """The direct BFGS Hessian update ``B - B s s^T B / (s^T B s) + y y^T / (y^T s)``; skipped when ``y . s <= 0``."""
    ys = float(y @ s)
    bs = b @ s
    sbs = float(s @ bs)
    if not ys > SECANT_FLOOR or not sbs > SECANT_FLOOR:
        return b
    return b - np.outer(bs, bs) / sbs + np.outer(y, y) / ys


def sr1_update(b: NDArray[np.float64], s: Vector, y: Vector, *, skip: float = 1e-8) -> NDArray[np.float64]:
    """The symmetric rank-1 update ``B + (y - B s)(y - B s)^T / ((y - B s)^T s)``.

    Skipped (``B`` returned) when ``|(y - B s) . s| < skip ||s|| ||y - B s||`` (Nocedal and Wright,
    eq. 6.26): the denominator is then too small to trust. SR1 does not keep positive definiteness.
    """
    v = y - b @ s
    denominator = float(v @ s)
    if abs(denominator) < skip * float(np.linalg.norm(s)) * float(np.linalg.norm(v)) or denominator == 0.0:
        return b
    return b + np.outer(v, v) / denominator
