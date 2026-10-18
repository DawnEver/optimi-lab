"""D131 Benders decomposition of a two-stage linear program (Benders 1962; Conejo et al. 2006, ch. 3).

    minimise c.x + q.y   subject to   master rows and bounds on x,   T x + W y <= h,   y >= 0

with ``x`` possibly mixed-integer (*integrality*). The master carries ``theta``, a lower model of
the recourse ``Q(x) = min { q.y : W y <= h - T x, y >= 0 }``, built from cuts:

* OPTIMALITY: the subproblem's sensitivities ``lambda <= 0`` (``d Q / d rhs``) are dual feasible
  for every right-hand side, so ``theta >= lambda.(h - T x)`` everywhere and is tight at ``x_k``;
* FEASIBILITY: when the subproblem is infeasible, its phase-1 program
  ``min 1.s : W y - s <= h - T x_k`` has a positive value and sensitivities ``mu``, and every
  feasible ``x`` satisfies ``mu.(h - T x) <= 0``.

The master value is a proven lower bound and the best ``c.x + Q(x)`` an upper bound; it stops when
they agree within *tolerance* relative to ``1 + |upper|``. The master is solved by
:func:`~optimi_lab.solver.milp` (or :func:`~optimi_lab.solver.lp` when nothing is integer).
``theta`` needs a floor: zero when ``q >= 0`` (``y >= 0`` makes ``Q >= 0``), else *theta_lower*.
"""

from __future__ import annotations

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.solver.branch_and_bound import milp
from optimi_lab.solver.linear import lp
from optimi_lab.solver.linear_program import LinearProgram, StructuredSolution
from optimi_lab.solver.smooth import Halt

__all__ = ['benders']


def _master(master: LinearProgram, theta_lower: float) -> LinearProgram:
    def pad(a: np.ndarray) -> np.ndarray:
        return np.hstack([a, np.zeros((a.shape[0], 1))])

    return LinearProgram(
        np.append(master.c, 1.0),
        a_ub=pad(master.a_ub),
        b_ub=master.b_ub,
        a_eq=pad(master.a_eq),
        b_eq=master.b_eq,
        lo=np.append(master.lo, theta_lower),
        hi=np.append(master.hi, np.inf),
    )


def _recourse(q, t, w, h, x: np.ndarray) -> tuple[StructuredSolution, np.ndarray | None]:
    """The subproblem at *x*, and the feasibility-cut sensitivities ``mu`` when it is infeasible."""
    rhs = h - t @ x
    sub = lp(LinearProgram(q, a_ub=w, b_ub=rhs))
    if sub.halt is not Halt.INFEASIBLE:
        return sub, None
    m, k = w.shape
    phase_one = lp(LinearProgram(np.append(np.zeros(k), np.ones(m)), a_ub=np.hstack([w, -np.eye(m)]), b_ub=rhs))
    return sub, phase_one.duals_ub


def benders(
    master: LinearProgram,
    q,
    t,
    w,
    h,
    *,
    integrality=None,
    theta_lower: float | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 100,
) -> StructuredSolution:
    """Solve the two-stage program; ``x`` of the report is ``[x, y]``. See the module docstring."""
    q, h = np.asarray(q, dtype=float), np.asarray(h, dtype=float)
    t = np.asarray(t, dtype=float).reshape(h.size, master.width)
    w = np.asarray(w, dtype=float).reshape(h.size, q.size)
    if theta_lower is None:
        if np.any(q < 0):
            refuse('q has negative entries, so the recourse has no zero floor; pass theta_lower')
        theta_lower = 0.0
    integer = np.zeros(master.width, dtype=bool) if integrality is None else np.asarray(integrality, dtype=bool)
    problem = _master(master, theta_lower)
    upper, best, lower, cuts = np.inf, None, -np.inf, 0
    for iteration in range(1, max_iterations + 1):
        found = milp(problem, np.append(integer, False)) if integer.any() else lp(problem)
        if found.halt is not Halt.CONVERGED:
            return StructuredSolution(found.halt, found.x, np.nan, iteration, reason=f'master: {found.reason}')
        x, lower = found.x[:-1], found.value
        sub, mu = _recourse(q, t, w, h, x)
        if sub.halt is Halt.UNBOUNDED:
            return StructuredSolution(Halt.UNBOUNDED, x, np.nan, iteration, reason='the recourse is unbounded')
        if mu is not None:
            row, side = np.append(-(mu @ t), 0.0), -(mu @ h)
        else:
            total = float(master.c @ x) + sub.value
            if total < upper:
                upper, best = total, np.concatenate([x, sub.x])
            row, side = np.append(-(sub.duals_ub @ t), -1.0), -(sub.duals_ub @ h)
        if best is not None and upper - lower <= tolerance * (1.0 + abs(upper)):
            evidence = (f'benders:cuts:{cuts}', 'master:milp' if integer.any() else 'master:lp')
            return StructuredSolution(Halt.CONVERGED, best, upper, iteration, bound=lower, evidence=evidence)
        problem = problem.with_rows(row[None, :], np.array([side]))
        cuts += 1
    return StructuredSolution(
        Halt.ITERATION_LIMIT,
        best if best is not None else np.zeros(master.width + q.size),
        upper,
        max_iterations,
        bound=lower,
        reason=f'the bounds still differ by {upper - lower} after {max_iterations} master solves',
        evidence=(f'benders:cuts:{cuts}',),
    )
