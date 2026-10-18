"""Convex quadratic programs: a general primal active-set method, and a projected Newton method for bounds.

    minimise 1/2 x.H x + g.x   subject to   A_eq x = b_eq,   A_in x <= b_in        (active_set_qp)
    minimise 1/2 x.H x + g.x   subject to   lo <= x <= hi                           (bound_qp)

Multiplier signs: ``H x + g + A_eq^T y + A_in^T z = 0`` with ``z >= 0``.

* :func:`active_set_qp` is Nocedal and Wright's Algorithm 16.3. Each iterate solves the
  equality-constrained QP on the working set through numerics-lab's KKT solve; a blocking constraint
  is added (it is independent of the working set because the step lies in that set's null space), a
  constraint with a negative multiplier is dropped. An infeasible start is first moved to a feasible
  one by a phase-1 QP in ``(x, t)``; when the constraints admit no point the least-violating point
  is returned with ``feasible=False`` -- never presented as a solution.
* :func:`bound_qp` is Bertsekas' projected Newton method: a Newton step on the variables not held at a
  bound (conjugate gradients, numerics-lab), searched along the projected arc by backtracking
  (numerics-lab). It needs ``H`` positive definite and refuses otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numerics_lab.line_search import backtracking
from numerics_lab.linear import LinearSolveError
from numerics_lab.linear.cholesky import jittered_cholesky
from numerics_lab.linear.kkt import solve_kkt
from numerics_lab.linear.krylov import conjugate_gradient

from optimi_lab.core.errors import refuse

__all__ = ['QPSolution', 'active_set_qp', 'bound_qp']

#: Phase-1 regularisation: small enough that a feasible point is always preferred to any violation.
_PHASE_ONE_CURVATURE = 1e-6
#: The Armijo constant of the projected search.
_ARMIJO = 1e-4


@dataclass(frozen=True, slots=True, eq=False)
class QPSolution:
    """*x*, the multipliers of the equality and inequality rows, whether *x* is *feasible*, and the *iterations*."""

    x: np.ndarray
    multipliers_eq: np.ndarray
    multipliers_in: np.ndarray
    feasible: bool
    iterations: int


def _matrix(a: np.ndarray | None, n: int) -> np.ndarray:
    return np.zeros((0, n)) if a is None else np.asarray(a, dtype=float).reshape(-1, n)


def _vector(b: np.ndarray | None, m: int) -> np.ndarray:
    return np.zeros(m) if b is None else np.asarray(b, dtype=float).reshape(m)


def _working_set(
    h, g, a_eq, a_in, b_in, *, x, tolerance, max_iterations
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Algorithm 16.3 from a FEASIBLE *x*: ``(x, y, z, iterations)``."""
    working: list[int] = []
    n_eq = a_eq.shape[0]
    scale = 1.0 + float(np.max(np.abs(x), initial=0.0))
    for iteration in range(1, max_iterations + 1):
        rows = np.vstack([a_eq, a_in[working]])
        gradient = h @ x + g
        try:
            step = solve_kkt(h, rows, -gradient, np.zeros(rows.shape[0]))
        except LinearSolveError:
            refuse('the QP Hessian is not positive definite on the null space of the active constraints')
        p, multipliers = step.x, step.multipliers
        if float(np.max(np.abs(p), initial=0.0)) <= tolerance * scale:
            z_working = multipliers[n_eq:]
            if not working or float(np.min(z_working)) >= -tolerance:
                z = np.zeros(a_in.shape[0])
                z[working] = np.maximum(z_working, 0.0)
                return x, multipliers[:n_eq], z, iteration
            working.pop(int(np.argmin(z_working)))
            continue
        alpha, blocking = 1.0, None
        for i in range(a_in.shape[0]):
            ap = float(a_in[i] @ p)
            if i not in working and ap > tolerance:
                limit = max(float(b_in[i] - a_in[i] @ x) / ap, 0.0)
                if limit < alpha:
                    alpha, blocking = limit, i
        x = x + alpha * p
        if blocking is not None:
            working.append(blocking)
    refuse(f'the active-set QP did not settle within {max_iterations} iterations (cycling on a degenerate vertex)')
    return None


def active_set_qp(
    h: np.ndarray,
    g: np.ndarray,
    *,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    a_in: np.ndarray | None = None,
    b_in: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
) -> QPSolution:
    """Solve the QP from *x0* (zeros by default; it need not be feasible). See the module docstring."""
    h, g = np.asarray(h, dtype=float), np.asarray(g, dtype=float)
    n = g.size
    a_eq, a_in = _matrix(a_eq, n), _matrix(a_in, n)
    b_eq, b_in = _vector(b_eq, a_eq.shape[0]), _vector(b_in, a_in.shape[0])
    if a_eq.shape[0] and np.linalg.matrix_rank(a_eq) < a_eq.shape[0]:
        refuse(f'the {a_eq.shape[0]} equality rows have rank {np.linalg.matrix_rank(a_eq)}; drop the dependent ones')
    limit = 10 * (n + a_eq.shape[0] + a_in.shape[0]) + 10 if max_iterations is None else max_iterations
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if a_eq.shape[0]:
        x = solve_kkt(np.eye(n), a_eq, x, b_eq).x
    excess = float(np.max(a_in @ x - b_in, initial=0.0))
    spent = 0
    if excess > tolerance:
        # Phase 1 in (x, t): minimise t (+ a vanishing proximal term) with A_in x - t <= b_in, t >= 0.
        h1 = _PHASE_ONE_CURVATURE * np.eye(n + 1)
        g1 = np.concatenate([-_PHASE_ONE_CURVATURE * x, [1.0]])
        a1_eq = np.hstack([a_eq, np.zeros((a_eq.shape[0], 1))])
        a1_in = np.vstack([np.hstack([a_in, -np.ones((a_in.shape[0], 1))]), np.eye(1, n + 1, n) * -1.0])
        b1_in = np.concatenate([b_in, [0.0]])
        start = np.concatenate([x, [excess]])
        found, _, _, spent = _working_set(
            h1, g1, a1_eq, a1_in, b1_in, x=start, tolerance=tolerance, max_iterations=limit
        )
        x = found[:n]
        if found[n] > np.sqrt(tolerance):
            return QPSolution(x, np.zeros(a_eq.shape[0]), np.zeros(a_in.shape[0]), False, spent)
    x, y, z, iterations = _working_set(h, g, a_eq, a_in, b_in, x=x, tolerance=tolerance, max_iterations=limit)
    return QPSolution(x, y, z, True, spent + iterations)


def bound_qp(
    h: np.ndarray,
    g: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> QPSolution:
    """Solve the bound-constrained QP by projected Newton; ``multipliers_in`` is ``[lower..., upper...]``."""
    h, g = np.asarray(h, dtype=float), np.asarray(g, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    try:
        jittered_cholesky(h, max_tries=0)
    except LinearSolveError:
        refuse('bound_qp needs a positive definite H; this one is not')
    x = np.clip(0.5 * (lo + hi) if x0 is None else np.asarray(x0, dtype=float), lo, hi)

    def model(v: np.ndarray) -> float:
        return 0.5 * float(v @ h @ v) + float(g @ v)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        gradient = h @ x + g
        projected = np.clip(x - gradient, lo, hi) - x
        gap = float(np.max(np.abs(projected), initial=0.0))
        if gap <= tolerance:
            break
        band = min(1e-3, gap)
        held = ((x <= lo + band) & (gradient > 0)) | ((x >= hi - band) & (gradient < 0))
        free = ~held
        d = np.zeros_like(x)
        if free.any():
            d[free] = conjugate_gradient(h[np.ix_(free, free)], -gradient[free], rtol=1e-14).x
        d[held] = -gradient[held]
        q0 = model(x)

        def trial(step: float, d: np.ndarray = d, x: np.ndarray = x) -> np.ndarray:
            return np.clip(x + step * d, lo, hi)

        def accept(
            _: float, xt: np.ndarray, gradient: np.ndarray = gradient, q0: float = q0, x: np.ndarray = x
        ) -> bool:
            return model(xt) <= q0 + _ARMIJO * float(gradient @ (xt - x))

        search = backtracking(trial, accept, tries=60)
        if not search.accepted:
            break
        x = search.trial
    gradient = h @ x + g
    lower = np.where(x <= lo, np.maximum(gradient, 0.0), 0.0)
    upper = np.where(x >= hi, np.maximum(-gradient, 0.0), 0.0)
    return QPSolution(x, np.zeros(0), np.concatenate([lower, upper]), True, iteration)
