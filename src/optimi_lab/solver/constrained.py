"""Constrained gradient methods: SQP, a primal-dual interior point, and the augmented Lagrangian.

All three solve ``min f  s.t.  h = 0, c <= 0, lo <= x <= hi`` and halt CONVERGED when the KKT
residual (:func:`~optimi_lab.operator.kkt_residual`) falls to ``kkt_bar`` (``rtol`` x start, floored by ``atol``). The Lagrangian Hessian is
a damped BFGS model (Powell damping here, the update numerics-lab's).

* :func:`sqp` -- each step solves the QP of the quadratic model under the linearised constraints and
  the box (:func:`~optimi_lab.operator.active_set_qp`); the step is searched on the exact l1 merit by
  Armijo backtracking. An infeasible linearisation takes the QP's least-violating step.
* :func:`interior_point` -- a primal-dual barrier method: the bounds become inequality rows, every
  inequality gets a slack, the Newton system is condensed onto ``(dx, dy)`` and solved as a KKT
  system -- DENSE or SPARSE (``sparse=True``) through numerics-lab's ``solve_kkt``. Fraction to the
  boundary on slacks and duals; Armijo on the l1 barrier merit; the barrier falls superlinearly
  (``mu <- min(0.2 mu, mu^1.5)``) once its own subproblem is solved to ``10 mu``.
* :func:`augmented_lagrangian` -- outer multiplier updates around an inner :func:`lbfgsb` on the box;
  the penalty grows tenfold when the infeasibility did not fall to a quarter.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numerics_lab.line_search import armijo
from numerics_lab.linear.kkt import solve_kkt
from numerics_lab.quasi_newton import bfgs_update
from scipy.linalg import LinAlgError

from optimi_lab.operator import (
    Linearization,
    active_set_qp,
    augmented_lagrangian_merit,
    damped_secant,
    kkt_residual,
    l1_merit,
    l1_penalty,
    l1_slope,
    lagrangian_gradient,
)
from optimi_lab.solver.bounded import lbfgsb
from optimi_lab.solver.smooth import ATOL, RTOL, Evaluate, Halt, Solution, kkt_bar, trial

__all__ = ['augmented_lagrangian', 'interior_point', 'sqp']

_ARMIJO = 1e-4


def _solution(
    point: Linearization, box: tuple, multipliers: tuple, iterations: int, halt: Halt, *, reason: str = ''
) -> Solution:
    (lo, hi), (y, z) = box, multipliers
    residual = kkt_residual(point, lo, hi, y=y, z=z)
    return Solution(halt, point.x, point, iterations, residual.worst, reason=reason)


def _secant(model: np.ndarray, old: Linearization, new: Linearization, y, z) -> np.ndarray:
    s = new.x - old.x
    change = lagrangian_gradient(new, y, z) - lagrangian_gradient(old, y, z)
    return bfgs_update(model, s, damped_secant(model, s, change)) if np.any(s) else model


def sqp(
    evaluate: Evaluate, x0, lo, hi, *, rtol: float = RTOL, atol: float = ATOL, max_iterations: int = 200
) -> Solution:
    """Line-search SQP on the l1 merit; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    point = evaluate(np.clip(np.asarray(x0, dtype=float), lo, hi))
    bar = kkt_bar(point.df[0], rtol=rtol, atol=atol)
    n = point.x.size
    model, nu = np.eye(n), 1.0
    y, z = np.zeros(point.h.size), np.zeros(point.c.size)
    for iteration in range(max_iterations):
        if iteration and kkt_residual(point, lo, hi, y=y, z=z).worst <= bar:
            return _solution(point, (lo, hi), (y, z), iteration, Halt.CONVERGED)
        qp = active_set_qp(
            model,
            point.gradient,
            a_eq=point.dh,
            b_eq=-point.h,
            a_in=np.vstack([point.dc, np.eye(n), -np.eye(n)]),
            b_in=np.concatenate([-point.c, hi - point.x, point.x - lo]),
        )
        d = qp.x
        if qp.feasible:
            y, z = qp.multipliers_eq, qp.multipliers_in[: point.c.size]
            if kkt_residual(point, lo, hi, y=y, z=z).worst <= bar:
                return _solution(point, (lo, hi), (y, z), iteration, Halt.CONVERGED)
        nu = l1_penalty(np.concatenate([y, z]), nu)
        slope = l1_slope(point, d, nu)
        if not slope < 0.0:
            halt = Halt.INFEASIBLE if not qp.feasible else Halt.STALLED
            return _solution(
                point, (lo, hi), (y, z), iteration, halt, reason='the QP step does not descend the l1 merit'
            )
        seen: dict[float, Linearization | None] = {}

        def merit(
            t: float, d: np.ndarray = d, seen: dict = seen, nu: float = nu, x: np.ndarray = point.x
        ) -> float | None:
            seen[t] = trial(evaluate, np.clip(x + t * d, lo, hi))
            return None if seen[t] is None else l1_merit(seen[t], nu)

        search = armijo(merit, l1_merit(point, nu), slope, c1=_ARMIJO)
        if not search.accepted:
            return _solution(point, (lo, hi), (y, z), iteration, Halt.STALLED, reason='no step decreased the l1 merit')
        accepted = seen[search.step]
        model = _secant(model, point, accepted, y, z)
        point = accepted
    return _solution(point, (lo, hi), (y, z), max_iterations, Halt.ITERATION_LIMIT)


def _rows(point: Linearization, lo, hi, sparse: bool):
    """``C(x)`` and its Jacobian: the inequalities, then ``lo - x <= 0`` and ``x - hi <= 0`` for finite bounds."""
    n = point.x.size
    low, high = np.isfinite(lo), np.isfinite(hi)
    values = np.concatenate([point.c, (lo - point.x)[low], (point.x - hi)[high]])
    eye = sp.identity(n, format='csr') if sparse else np.eye(n)
    parts = [sp.csr_matrix(point.dc) if sparse else point.dc, -eye[low], eye[high]]
    return values, (sp.vstack(parts, format='csr') if sparse else np.vstack(parts))


def interior_point(
    evaluate: Evaluate,
    x0,
    lo,
    hi,
    *,
    sparse: bool = False,
    rtol: float = RTOL,
    atol: float = ATOL,
    max_iterations: int = 300,
) -> Solution:
    """A primal-dual interior-point method; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    width = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
    inner = np.clip(np.asarray(x0, dtype=float), lo + 0.01 * width, hi - 0.01 * width)
    point = evaluate(inner)
    bar = kkt_bar(point.df[0], rtol=rtol, atol=atol)
    n = point.x.size
    values, jacobian = _rows(point, lo, hi, sparse)
    s = np.maximum(-values, 1e-2)
    s[point.c.size :] = -values[point.c.size :]  # x starts strictly inside the box
    mu = 0.1
    z = mu / s
    y = np.zeros(point.h.size)
    model, nu = np.eye(n), 1.0
    for iteration in range(max_iterations):
        r_d = point.gradient + point.dh.T @ y + jacobian.T @ z
        r_p = values + s
        primal_dual = max(
            float(np.max(np.abs(r_d))),
            float(np.max(np.abs(r_p), initial=0.0)),
            float(np.max(np.abs(point.h), initial=0.0)),
        )
        if max(primal_dual, float(np.max(s * z, initial=0.0))) <= bar:
            return _solution(point, (lo, hi), (y, z[: point.c.size]), iteration, Halt.CONVERGED)
        if max(primal_dual, float(np.max(np.abs(s * z - mu), initial=0.0))) <= 10 * mu:
            mu = max(bar / 10, min(0.2 * mu, mu**1.5))
        sigma = z / s
        weights = sp.diags(sigma) if sparse else np.diag(sigma)
        condensed = (sp.csr_matrix(model) if sparse else model) + jacobian.T @ weights @ jacobian
        rhs = -r_d - jacobian.T @ (sigma * r_p + (mu - s * z) / s)
        dh = sp.csr_matrix(point.dh) if sparse else point.dh
        try:
            step = solve_kkt(condensed, dh, rhs, -point.h)
        except LinAlgError as singular:
            box, pair = (lo, hi), (y, z[: point.c.size])
            return _solution(point, box, pair, iteration, Halt.STALLED, reason=f'the barrier KKT system: {singular}')
        dx, dy = step.x, step.multipliers
        ds = -r_p - jacobian @ dx
        dz = sigma * (jacobian @ dx + r_p) + (mu - s * z) / s
        tau = max(0.99, 1.0 - mu)
        alpha_s = float(np.min(tau * -s[ds < 0] / ds[ds < 0], initial=1.0))
        alpha_z = float(np.min(tau * -z[dz < 0] / dz[dz < 0], initial=1.0))
        multipliers = np.concatenate([y + dy, z + dz])
        nu = l1_penalty(multipliers, nu)
        infeasibility = float(np.sum(np.abs(point.h)) + np.sum(np.abs(r_p)))
        descent = float(point.gradient @ dx) - mu * float(np.sum(ds / s))
        if infeasibility > 0 and descent - nu * infeasibility >= 0:
            nu = 2.0 * descent / infeasibility
        slope = alpha_s * (descent - nu * infeasibility)

        def barrier(candidate: Linearization, slack: np.ndarray, nu: float = nu, mu: float = mu) -> float:
            rows, _ = _rows(candidate, lo, hi, sparse)
            return (
                candidate.objective
                - mu * float(np.sum(np.log(slack)))
                + nu * float(np.sum(np.abs(candidate.h)) + np.sum(np.abs(rows + slack)))
            )

        seen: dict[float, Linearization | None] = {}

        def merit(t: float, *, dx=dx, ds=ds, a: float = alpha_s, seen: dict = seen, x=point.x, s=s, barrier=barrier):
            seen[t] = trial(evaluate, x + t * a * dx)
            return None if seen[t] is None else barrier(seen[t], s + t * a * ds)

        if not slope < 0.0:
            return _solution(
                point, (lo, hi), (y, z[: point.c.size]), iteration, Halt.STALLED, reason='no barrier descent'
            )
        search = armijo(merit, barrier(point, s), slope, c1=_ARMIJO)
        if not search.accepted:
            return _solution(point, (lo, hi), (y, z[: point.c.size]), iteration, Halt.STALLED, reason='no barrier step')
        t = search.step
        accepted = seen[t]
        s = s + t * alpha_s * ds
        y, z = y + t * alpha_s * dy, z + t * alpha_z * dz
        model = _secant(model, point, accepted, y, z[: point.c.size])
        point = accepted
        values, jacobian = _rows(point, lo, hi, sparse)
        bounds = slice(point.c.size, None)
        s[bounds] = -values[bounds]  # linear rows: the slack IS the distance to the bound, kept exact
    return _solution(point, (lo, hi), (y, z[: point.c.size]), max_iterations, Halt.ITERATION_LIMIT)


def augmented_lagrangian(
    evaluate: Evaluate, x0, lo, hi, *, rtol: float = RTOL, atol: float = ATOL, rho: float = 10.0, max_outer: int = 40
) -> Solution:
    """The bound-constrained augmented Lagrangian method; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    point = evaluate(np.clip(np.asarray(x0, dtype=float), lo, hi))
    y, z = np.zeros(point.h.size), np.zeros(point.c.size)
    previous, bar = np.inf, kkt_bar(point.df[0], rtol=rtol, atol=atol)
    for outer in range(max_outer):
        if outer and kkt_residual(point, lo, hi, y=y, z=z).worst <= bar:
            return _solution(point, (lo, hi), (y, z), outer, Halt.CONVERGED)

        def merit(x: np.ndarray, y: np.ndarray = y, z: np.ndarray = z, rho: float = rho) -> Linearization:
            value, gradient = augmented_lagrangian_merit(evaluate(x), y, z, rho)
            return Linearization(x=x, f=np.array([value]), df=gradient[None, :])

        inner = lbfgsb(merit, point.x, lo, hi, rtol=0.0, atol=max(0.1 * bar, 10.0 ** -(outer + 2)))
        point = evaluate(inner.x)
        infeasibility = max(
            float(np.max(np.abs(point.h), initial=0.0)),
            float(np.max(np.abs(np.minimum(-point.c, z / rho)), initial=0.0)),
        )
        y, z = y + rho * point.h, np.maximum(0.0, z + rho * point.c)
        if infeasibility > 0.25 * previous:
            rho *= 10.0
        previous = infeasibility
    return _solution(point, (lo, hi), (y, z), max_outer, Halt.ITERATION_LIMIT)
