"""Quasi-Newton models for MINIMIZATION: dense BFGS and SR1 Hessian updates and the L-BFGS pair memory.

- DENSE, for small models (a trust-region or SQP Hessian): :func:`bfgs_update`,
  :func:`bfgs_inverse_update`, :func:`sr1_update`.
- LIMITED MEMORY: :class:`LbfgsMemory`, one model in two forms (the two-loop ``H q`` and the compact
  direct ``B``).

Every update keeps its secant equation exactly in exact arithmetic (``H y = s`` or ``B s = y``), and
an update whose curvature vanishes is SKIPPED (the model returned unchanged), never divided through.
The secant pair, its two-loop recursion and Broyden's root-finding updates are numerics-lab's
(``numerics_lab.quasi_newton``): its nonlinear solves run on them.
"""

from __future__ import annotations

import numpy as np
from numerics_lab.quasi_newton import SECANT_FLOOR, SecantPair, two_loop
from numpy.typing import NDArray

__all__ = [
    'CURVATURE_EPS',
    'LbfgsMemory',
    'SingularCompactModel',
    'bfgs_inverse_update',
    'bfgs_update',
    'sr1_update',
]

#: An L-BFGS pair is kept only if ``s . y > CURVATURE_EPS * y . y`` (Byrd, Lu, Nocedal and Zhu 1995):
#: a curvature that is roundoff against ``|y|^2`` would scale ``theta = y.y / s.y`` past float range.
CURVATURE_EPS = float(np.finfo(float).eps)

Vector = NDArray[np.float64]


class SingularCompactModel(np.linalg.LinAlgError):
    """The compact L-BFGS middle matrix is singular in floating point; reset the memory."""


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


class LbfgsMemory:
    """The L-BFGS pair memory: the ONE bookkeeping behind every limited-memory BFGS model here.

    Keeps the last ``size`` secant pairs (oldest first), SKIPS a pair whose curvature ``y . s`` is not above ``CURVATURE_EPS * y . y`` or
    above :data:`SECANT_FLOOR`, and scales by ``gamma = s.y / y.y`` of the newest kept pair (``1``
    when empty). One model, two forms: :meth:`inverse_apply` is the two-loop ``H q`` on ``gamma I``;
    :meth:`hessian` is the compact direct form ``B = theta I - W M W^T`` with ``theta = 1 / gamma``
    (Byrd, Nocedal and Schnabel 1994, eq. 3.5), which a bound-constrained (Cauchy point) step needs.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            msg = f'an L-BFGS memory holds at least one secant pair, got {size!r}'
            raise ValueError(msg)
        self.size = size
        self.pairs: tuple[SecantPair, ...] = ()
        self.gamma = 1.0
        self._dimension: int | None = None

    def push(self, s: Vector, y: Vector) -> bool:
        """Keep ``(s, y)`` if ``s.y`` clears both :data:`SECANT_FLOOR` and ``CURVATURE_EPS * y.y``; True if kept."""
        s, y = np.asarray(s, dtype=float), np.asarray(y, dtype=float)
        self._dimension = s.size
        ys = float(y @ s)
        if not (ys > SECANT_FLOOR and ys > CURVATURE_EPS * float(y @ y)):
            return False
        self.pairs = (*self.pairs[max(0, len(self.pairs) + 1 - self.size) :], SecantPair(s.copy(), y.copy(), 1.0 / ys))
        self.gamma = ys / float(y @ y)
        return True

    def clear(self) -> None:
        """Forget every pair (a restart): the model is the identity again."""
        self.pairs, self.gamma = (), 1.0

    def inverse_apply(self, q: Vector) -> Vector:
        """``H q``: the two-loop recursion over the kept pairs on ``H0 = gamma I``."""
        return two_loop(np.asarray(q, dtype=float), self.pairs, lambda v: self.gamma * v)

    def compact_factors(self) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """``(theta, W, M)`` with ``B = theta I - W M W^T``; ``W`` is ``n x 2m`` (``[Y, theta S]``)."""
        theta = 1.0 / self.gamma
        if not self.pairs:
            return theta, np.zeros((self._dimension or 0, 0)), np.zeros((0, 0))
        s = np.column_stack([pair.s for pair in self.pairs])
        y = np.column_stack([pair.y for pair in self.pairs])
        sy = s.T @ y
        lower = np.tril(sy, -1)
        middle = np.block([[-np.diag(np.diag(sy)), lower.T], [lower, theta * (s.T @ s)]])
        msg = f'the compact L-BFGS middle matrix ({middle.shape[0]}x{middle.shape[0]}) is singular'
        # Singular IN FLOATING POINT, as LAPACK's gesv judges it: a reciprocal 1-norm condition below eps.
        with np.errstate(divide='ignore', invalid='ignore'):
            rcond = 1.0 / float(np.linalg.cond(middle, 1))
        if not rcond >= np.finfo(float).eps:
            raise SingularCompactModel(msg)
        try:
            m = np.linalg.solve(middle, np.eye(middle.shape[0]))
        except np.linalg.LinAlgError as exc:
            raise SingularCompactModel(msg) from exc
        return theta, np.hstack([y, theta * s]), m

    def hessian(self) -> NDArray[np.float64]:
        """The dense direct model ``B`` (the compact form expanded); for small ``n`` and for tests."""
        if self._dimension is None:
            msg = 'an L-BFGS memory that never saw a pair has no dimension'
            raise ValueError(msg)
        theta, w, m = self.compact_factors()
        return theta * np.eye(self._dimension) - w @ m @ w.T
