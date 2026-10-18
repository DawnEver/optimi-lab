"""Gradient methods for several objectives: MGDA, and Pareto continuation for two.

* :func:`mgda` -- the multiple-gradient descent of Fliege and Svaiter on a box: the common descent
  direction ``d`` solves ``min t + |d|^2 / 2  s.t.  grad f_k . d <= t,  lo <= x + d <= hi`` (an
  :func:`~optimi_lab.operator.active_set_qp` in ``(d, t)``; without the box it is Desideri's
  minimum-norm element of the gradients' convex hull), and a step is accepted when EVERY objective
  meets its own Armijo condition. It halts at a Pareto-critical point (``d = 0``). Several starting
  points (rows of *x0*) give several critical points -- the front is the set of them.
* :func:`pareto_continuation` -- traces a bi-objective front by the epsilon-constraint homotopy: the
  two anchors (each objective minimised alone), then ``min f0  s.t.  f1 <= eps`` for *points* evenly
  spaced levels between them, each solved by :func:`~optimi_lab.solver.constrained.sqp` from a
  SECANT PREDICTOR (the last two solutions extrapolated) -- the corrector. Unlike a weighted sum it
  reaches the non-convex parts of a front. More than two objectives are refused.
"""

from __future__ import annotations

import numpy as np
from numerics_lab.line_search import backtracking

from optimi_lab.core.errors import refuse
from optimi_lab.operator import Linearization, active_set_qp
from optimi_lab.solver.constrained import sqp
from optimi_lab.solver.smooth import Evaluate, Halt, Solution, trial

__all__ = ['mgda', 'pareto_continuation']

_ARMIJO = 1e-4
#: The curvature on ``t`` that makes the direction QP strictly convex; negligible beside ``|d|^2``.
_T_CURVATURE = 1e-6


def _direction(point: Linearization, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    n, k = point.x.size, point.f.size
    h = np.diag(np.concatenate([np.ones(n), [_T_CURVATURE]]))
    g = np.concatenate([np.zeros(n), [1.0]])
    a_in = np.vstack([
        np.hstack([point.df, -np.ones((k, 1))]),
        np.hstack([np.eye(n), np.zeros((n, 1))]),
        np.hstack([-np.eye(n), np.zeros((n, 1))]),
    ])
    b_in = np.concatenate([np.zeros(k), hi - point.x, point.x - lo])
    return active_set_qp(h, g, a_in=a_in, b_in=b_in).x[:n]


def _descend(evaluate: Evaluate, x0, lo, hi, *, tolerance: float, max_iterations: int) -> Solution:
    point = evaluate(np.clip(np.asarray(x0, dtype=float), lo, hi))
    if point.c.size or point.h.size:
        refuse('mgda takes no constraints beyond the box; penalise them, or use pareto_continuation')
    for iteration in range(max_iterations):
        d = _direction(point, lo, hi)
        size = float(np.max(np.abs(d), initial=0.0))
        if size <= tolerance:
            return Solution(Halt.CONVERGED, point.x, point, iteration, size)
        slopes = point.df @ d

        def step(t: float, d: np.ndarray = d, x: np.ndarray = point.x) -> Linearization | None:
            return trial(evaluate, np.clip(x + t * d, lo, hi))

        def accept(t: float, candidate: Linearization, slopes: np.ndarray = slopes, f: np.ndarray = point.f) -> bool:
            return bool(np.all(candidate.f <= f + _ARMIJO * t * slopes))

        search = backtracking(step, accept, tries=40)
        if not search.accepted:
            return Solution(Halt.STALLED, point.x, point, iteration, size, reason='no step decreased every objective')
        point = search.trial
    return Solution(
        Halt.ITERATION_LIMIT, point.x, point, max_iterations, float(np.max(np.abs(_direction(point, lo, hi))))
    )


def mgda(evaluate: Evaluate, x0, lo, hi, *, tolerance: float = 1e-8, max_iterations: int = 500) -> Solution:
    """MGDA from each row of *x0*; the solution's *front* holds every critical point reached."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    starts = np.atleast_2d(np.asarray(x0, dtype=float))
    found = [_descend(evaluate, start, lo, hi, tolerance=tolerance, max_iterations=max_iterations) for start in starts]
    worst = max(found, key=lambda s: s.residual)
    halt = Halt.CONVERGED if all(s.halt is Halt.CONVERGED for s in found) else worst.halt
    return Solution(
        halt,
        worst.x,
        worst.point,
        sum(s.iterations for s in found),
        worst.residual,
        front=tuple(s.point for s in found if s.point is not None),
        reason=worst.reason,
    )


def _restricted(evaluate: Evaluate, keep: int, level: float | None = None, scale: float = 1.0) -> Evaluate:
    """*evaluate* with only objective *keep*; the other one becomes ``(f_other - level) / scale <= 0`` when *level* is set."""
    other = 1 - keep

    def restricted(x: np.ndarray) -> Linearization:
        p = evaluate(x)
        c, dc = p.c, p.dc
        if level is not None:
            c = np.concatenate([c, [(p.f[other] - level) / scale]])
            dc = np.vstack([dc, p.df[other] / scale])
        return Linearization(x=p.x, f=p.f[keep : keep + 1], df=p.df[keep : keep + 1], c=c, dc=dc, h=p.h, dh=p.dh)

    return restricted


def pareto_continuation(
    evaluate: Evaluate, x0, lo, hi, *, points: int = 11, tolerance: float = 1e-8, max_iterations: int = 200
) -> Solution:
    """The epsilon-constraint continuation of a bi-objective front; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    start = evaluate(np.clip(np.asarray(x0, dtype=float), lo, hi))
    if start.f.size != 2:
        refuse(f'pareto_continuation traces a front of two objectives; this problem has {start.f.size}')
    if points < 2:
        refuse(f'a front needs points >= 2, got {points}')
    settings = {'tolerance': tolerance, 'max_iterations': max_iterations}
    first = sqp(_restricted(evaluate, 0), start.x, lo, hi, **settings)
    last = sqp(_restricted(evaluate, 1), start.x, lo, hi, **settings)
    a, b = evaluate(first.x), evaluate(last.x)
    scale = max(abs(a.f[1] - b.f[1]), 1e-12)
    solutions, path = [first], [first.x]
    for level in np.linspace(a.f[1], b.f[1], points)[1:-1]:
        guess = path[-1] if len(path) < 2 else np.clip(2 * path[-1] - path[-2], lo, hi)
        solved = sqp(_restricted(evaluate, 0, level, scale), guess, lo, hi, **settings)
        solutions.append(solved)
        path.append(solved.x)
    solutions.append(last)
    front = tuple(evaluate(s.x) for s in solutions)
    worst = max(solutions, key=lambda s: s.residual)
    halt = Halt.CONVERGED if all(s.halt is Halt.CONVERGED for s in solutions) else worst.halt
    return Solution(
        halt, worst.x, front[0], sum(s.iterations for s in solutions), worst.residual, front=front, reason=worst.reason
    )
