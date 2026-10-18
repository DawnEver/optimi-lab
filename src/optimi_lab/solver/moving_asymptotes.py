"""The method of moving asymptotes (Svanberg 1987) and its globally convergent form, GCMMA (Svanberg 2002).

For ``min f0  s.t.  f_i <= 0, lo <= x <= hi`` each outer iteration replaces every ``f_i`` by the
convex, separable approximation

    f~_i(x) = r_i + sum_j p_ij / (U_j - x_j) + q_ij / (x_j - L_j)

around the current point, with moving asymptotes ``L < x < U`` that contract on oscillation and
expand on monotone progress, and solves the approximate problem through its DUAL: for multipliers
``lam >= 0`` the primal minimiser is closed-form per coordinate, so the dual is a smooth concave
function of ``lam`` alone, maximised here by :func:`~optimi_lab.solver.bounded.lbfgsb` on
``[0, inf)``. Artificial variables ``y_i`` (cost ``1000 y + y^2 / 2``) keep every subproblem feasible.

GCMMA (``conservative=True``) adds inner iterations: a candidate whose approximations are not
CONSERVATIVE (``f~_i < f_i`` for some ``i``) is re-solved with that approximation's curvature
``rho_i`` raised -- the step that makes the method globally convergent, at one extra evaluation per
inner iteration. Equalities are refused: the method has no form for them.
"""

from __future__ import annotations

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.operator import Linearization, kkt_residual
from optimi_lab.solver.bounded import lbfgsb
from optimi_lab.solver.smooth import Evaluate, Halt, Solution, kkt_bar

__all__ = ['mma']

#: Svanberg's artificial-variable costs ``c_i`` and ``d_i``.
_COST, _CURVATURE = 1000.0, 1.0
_MOVE = 0.5
_INITIAL, _GROW, _SHRINK = 0.5, 1.2, 0.7


def _approximation(point: Linearization, lower, upper, rho, span):
    """``(p, q, r)`` for the objective (row 0) and each constraint, around *point*."""
    values = np.concatenate([point.f[:1], point.c])
    slopes = np.vstack([point.df[:1], point.dc])
    plus, minus = np.maximum(slopes, 0.0), np.maximum(-slopes, 0.0)
    extra = rho[:, None] / span
    p = (upper - point.x) ** 2 * (1.001 * plus + 0.001 * minus + extra)
    q = (point.x - lower) ** 2 * (0.001 * plus + 1.001 * minus + extra)
    r = values - np.sum(p / (upper - point.x) + q / (point.x - lower), axis=1)
    return p, q, r


def _approximate(x, p, q, r, *, lower, upper) -> np.ndarray:
    return r + np.sum(p / (upper - x) + q / (x - lower), axis=1)


def _subproblem(p, q, r, *, lower, upper, alpha, beta, lam0):
    """The approximate problem's primal minimiser and multipliers, through its dual."""
    m = r.size - 1

    def primal(lam: np.ndarray) -> np.ndarray:
        big_p = p[0] + lam @ p[1:]
        big_q = q[0] + lam @ q[1:]
        sp, sq = np.sqrt(big_p), np.sqrt(big_q)
        return np.clip((sp * lower + sq * upper) / np.maximum(sp + sq, 1e-300), alpha, beta)

    def negative_dual(lam: np.ndarray) -> Linearization:
        x = primal(lam)
        y = np.maximum(0.0, (lam - _COST) / _CURVATURE)
        f = _approximate(x, p, q, r, lower=lower, upper=upper)
        value = f[0] + lam @ (f[1:] - y) + np.sum(_COST * y + 0.5 * _CURVATURE * y**2)
        return Linearization(x=lam, f=np.array([-value]), df=-(f[1:] - y)[None, :])

    # Below ~1e-9 the dual gradient is rounding: the search stagnates there (measured 2026-10-18).
    dual = lbfgsb(negative_dual, lam0, np.zeros(m), np.full(m, np.inf), tolerance=1e-8, max_iterations=500)
    return primal(dual.x), dual.x


def mma(
    evaluate: Evaluate,
    x0,
    lo,
    hi,
    *,
    conservative: bool = False,
    tolerance: float = 1e-6,
    max_iterations: int = 300,
    max_inner: int = 15,
) -> Solution:
    """MMA, or GCMMA with ``conservative=True``; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    span = hi - lo
    point = evaluate(np.clip(np.asarray(x0, dtype=float), lo, hi))
    bar = kkt_bar(point, tolerance)
    if point.h.size:
        refuse(f'MMA has no form for equality constraints; this problem has {point.h.size} -- use sqp')
    m = point.c.size
    lam = np.ones(m)
    history: list[np.ndarray] = []
    lower = upper = None
    for iteration in range(max_iterations):
        if iteration and kkt_residual(point, lo, hi, z=lam).worst <= bar:
            return Solution(Halt.CONVERGED, point.x, point, iteration, kkt_residual(point, lo, hi, z=lam).worst)
        x = point.x
        if len(history) < 2:
            lower, upper = x - _INITIAL * span, x + _INITIAL * span
        else:
            trend = (x - history[-1]) * (history[-1] - history[-2])
            gamma = np.where(trend < 0, _SHRINK, np.where(trend > 0, _GROW, 1.0))
            lower = np.clip(x - gamma * (history[-1] - lower), x - 10 * span, x - 0.01 * span)
            upper = np.clip(x + gamma * (upper - history[-1]), x + 0.01 * span, x + 10 * span)
        alpha = np.maximum.reduce([lo, lower + 0.1 * (x - lower), x - _MOVE * span])
        beta = np.minimum.reduce([hi, upper - 0.1 * (upper - x), x + _MOVE * span])
        slopes = np.vstack([point.df[:1], point.dc])
        rho = (
            np.maximum(0.1 / x.size * np.sum(np.abs(slopes) * span, axis=1), 1e-5)
            if conservative
            else np.full(m + 1, 1e-5)
        )
        for _ in range(max_inner):
            p, q, r = _approximation(point, lower, upper, rho, span)
            candidate_x, lam = _subproblem(p, q, r, lower=lower, upper=upper, alpha=alpha, beta=beta, lam0=lam)
            candidate = evaluate(candidate_x)
            if not conservative:
                break
            approximated = _approximate(candidate_x, p, q, r, lower=lower, upper=upper)
            actual = np.concatenate([candidate.f[:1], candidate.c])
            short = actual - approximated
            if np.all(short <= 1e-12 * (1.0 + np.abs(actual))):
                break
            dx = candidate_x - x
            distance = float(np.sum((upper - lower) * dx**2 / ((upper - candidate_x) * (candidate_x - lower) * span)))
            delta = np.where(short > 0, short / max(distance, 1e-300), 0.0)
            rho = np.where(short > 0, np.minimum(1.1 * (rho + delta), 10.0 * rho), rho)
        history = [*history, x][-2:]
        point = candidate
    return Solution(Halt.ITERATION_LIMIT, point.x, point, max_iterations, kkt_residual(point, lo, hi, z=lam).worst)
