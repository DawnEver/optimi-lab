"""D132 the alternating direction method of multipliers in consensus form (Boyd et al. 2011, sec. 3).

    minimise f(x) + g(z)   subject to   x - z = 0

in scaled form: ``x <- prox_f(z - u)``, ``z <- prox_g(x + u)``, ``u <- u + x - z``, where
``prox_phi(v, rho) = argmin_x phi(x) + rho/2 |x - v|^2``. Splitting a problem this way is how a
smooth part (a quadratic, a fitted model) and a simple non-smooth part (a box, an l1 penalty, a
projection) are each handled by the operation that is cheap for it.

It stops on Boyd's residuals (sec. 3.3.1): primal ``|x - z|`` and dual ``rho |z - z_prev|`` below
``sqrt(n) atol + rtol max(|x|, |z|)`` and ``sqrt(n) atol + rtol |rho u|``; *atol* is in the units of
``x`` and *rtol* is dimensionless. The reported point is ``z`` -- the one ``g`` (the constraint
side) accepted.

Proximal operators provided: :func:`quadratic_prox` (its linear solve is numerics-lab's Cholesky
factor, formed once per ``rho``), :func:`box_prox`, :func:`l1_prox`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numerics_lab.linear.cholesky import jittered_cholesky
from scipy.linalg import cho_solve

from optimi_lab.solver.linear_program import StructuredSolution
from optimi_lab.solver.smooth import Halt

__all__ = ['Prox', 'admm', 'box_prox', 'l1_prox', 'quadratic_prox']

#: ``prox(v, rho) -> argmin phi(x) + rho/2 |x - v|^2``.
Prox = Callable[[np.ndarray, float], np.ndarray]


def quadratic_prox(h, g) -> Prox:
    """The prox of ``1/2 x.H x + g.x``: ``(H + rho I) x = rho v - g``, ``H`` positive semidefinite."""
    h, g = np.asarray(h, dtype=float), np.asarray(g, dtype=float)
    factors: dict[float, tuple[np.ndarray, bool]] = {}

    def prox(v: np.ndarray, rho: float) -> np.ndarray:
        if rho not in factors:
            factors[rho] = (jittered_cholesky(h + rho * np.eye(g.size)).lower, True)
        return cho_solve(factors[rho], rho * v - g)

    return prox


def box_prox(lo, hi) -> Prox:
    """The prox of the indicator of ``[lo, hi]``: the projection."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return lambda v, _rho: np.clip(v, lo, hi)


def l1_prox(weight) -> Prox:
    """The prox of ``weight |x|_1``: soft thresholding at ``weight / rho``."""
    weight = np.asarray(weight, dtype=float)
    return lambda v, rho: np.sign(v) * np.maximum(np.abs(v) - weight / rho, 0.0)


def admm(
    prox_f: Prox,
    prox_g: Prox,
    n: int,
    *,
    rho: float = 1.0,
    x0=None,
    objective: Callable[[np.ndarray], float] | None = None,
    atol: float = 1e-9,
    rtol: float = 1e-7,
    max_iterations: int = 10_000,
) -> StructuredSolution:
    """Run scaled consensus ADMM; see the module docstring."""
    z = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    u = np.zeros(n)
    halt, iteration = Halt.ITERATION_LIMIT, max_iterations
    for k in range(1, max_iterations + 1):
        x = prox_f(z - u, rho)
        previous, z = z, prox_g(x + u, rho)
        u = u + x - z
        primal, dual = np.linalg.norm(x - z), rho * np.linalg.norm(z - previous)
        primal_bar = np.sqrt(n) * atol + rtol * max(np.linalg.norm(x), np.linalg.norm(z))
        dual_bar = np.sqrt(n) * atol + rtol * rho * np.linalg.norm(u)
        if primal <= primal_bar and dual <= dual_bar:
            halt, iteration = Halt.CONVERGED, k
            break
    value = float(objective(z)) if objective is not None else np.nan
    reason = '' if halt is Halt.CONVERGED else f'the residuals did not settle within {max_iterations} iterations'
    return StructuredSolution(halt, z, value, iteration, bound=np.nan, reason=reason, evidence=(f'admm:rho:{rho:g}',))
