"""D127 convex quadratic programs over a :class:`~optimi_lab.solver.linear_program.LinearProgram`'s rows.

    minimise 1/2 x.H x + c.x   subject to   A_ub x <= b_ub,   A_eq x = b_eq,   lo <= x <= hi

This is a ROUTE, not a third QP method: the two methods are P9's, in :mod:`optimi_lab.operator.qp`.

* bounds only and ``H`` positive definite -> :func:`~optimi_lab.operator.bound_qp` (projected Newton);
* otherwise -> :func:`~optimi_lab.operator.active_set_qp`, with each finite bound folded in as a row.

Multipliers are returned as SENSITIVITIES (``d value / d b``, the LP convention), so the active-set
multipliers ``y, z`` of ``H x + c + A_eq^T y + A_ub^T z = 0`` are reported negated. An infeasible
program is ``INFEASIBLE`` with the least-violating point, never presented as a solution.
"""

from __future__ import annotations

import numpy as np
from numerics_lab.linear import LinearSolveError
from numerics_lab.linear.cholesky import jittered_cholesky

from optimi_lab.operator import active_set_qp, bound_qp
from optimi_lab.solver.linear_program import LinearProgram, StructuredSolution
from optimi_lab.solver.smooth import Halt

__all__ = ['qp']


def _positive_definite(h: np.ndarray) -> bool:
    try:
        jittered_cholesky(h, max_tries=0)
    except LinearSolveError:
        return False
    return True


def qp(problem: LinearProgram, h, *, tolerance: float = 1e-10) -> StructuredSolution:
    """Solve the QP whose linear term and rows are *problem*'s and whose Hessian is *h* (see module docstring)."""
    h = np.asarray(h, dtype=float).reshape(problem.width, problem.width)
    n, m_ub = problem.width, problem.a_ub.shape[0]
    x0 = np.clip(np.zeros(n), problem.lo, problem.hi)
    if not m_ub and not problem.a_eq.shape[0] and _positive_definite(h):
        found = bound_qp(h, problem.c, problem.lo, problem.hi, x0=x0, tolerance=tolerance)
        route, duals_ub, duals_eq = 'qp:bound_qp', np.zeros(0), np.zeros(0)
    else:
        lower, upper = np.flatnonzero(np.isfinite(problem.lo)), np.flatnonzero(np.isfinite(problem.hi))
        eye = np.eye(n)
        a_in = np.vstack([problem.a_ub, -eye[lower], eye[upper]])
        b_in = np.concatenate([problem.b_ub, -problem.lo[lower], problem.hi[upper]])
        found = active_set_qp(
            h, problem.c, a_eq=problem.a_eq, b_eq=problem.b_eq, a_in=a_in, b_in=b_in, x0=x0, tolerance=tolerance
        )
        route, duals_ub, duals_eq = 'qp:active_set_qp', -found.multipliers_in[:m_ub], -found.multipliers_eq
    x = found.x
    value = 0.5 * float(x @ h @ x) + float(problem.c @ x)
    halt = Halt.CONVERGED if found.feasible else Halt.INFEASIBLE
    reason = '' if found.feasible else 'no point satisfies the rows and bounds; x is the least-violating point'
    return StructuredSolution(
        halt,
        x,
        value if found.feasible else np.nan,
        found.iterations,
        duals_ub,
        duals_eq,
        bound=value if found.feasible else np.nan,
        reason=reason,
        evidence=(route,),
    )
