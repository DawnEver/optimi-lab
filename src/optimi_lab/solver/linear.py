"""D126 :func:`lp`: one entry for both linear-programming methods, reporting on the original variables.

``method='simplex'`` (default) is the two-phase revised simplex
(:mod:`optimi_lab.solver.simplex`) and names INFEASIBLE and UNBOUNDED programs; ``method='ipm'`` is
Mehrotra's predictor-corrector (:mod:`optimi_lab.solver.interior_lp`), which reports a run that
did not converge as ``ITERATION_LIMIT`` instead of certifying either case.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.solver.interior_lp import mehrotra
from optimi_lab.solver.linear_program import LinearProgram, StructuredSolution, standard_form
from optimi_lab.solver.simplex import Vertex, simplex
from optimi_lab.solver.smooth import Halt

__all__ = ['LP_METHODS', 'lp', 'report']

#: Every linear-programming method by name.
LP_METHODS: Final = {'simplex': simplex, 'ipm': mehrotra}


def report(problem: LinearProgram, vertex: Vertex, *, evidence: tuple[str, ...] = ()) -> StructuredSolution:
    """The standard-form *vertex* of *problem* as a :class:`StructuredSolution` on its own variables."""
    form = standard_form(problem)
    x = form.p @ vertex.z[: form.p.shape[1]] + form.q
    duals_ub, duals_eq = form.rows_of(vertex.y)
    value = float(problem.c @ x) if vertex.halt is Halt.CONVERGED else np.nan
    reasons = {
        Halt.INFEASIBLE: 'phase 1 left a positive artificial sum: no point satisfies the rows and bounds',
        Halt.UNBOUNDED: 'an improving direction has no blocking row: the value decreases without limit',
        Halt.ITERATION_LIMIT: 'not converged; the interior point certifies neither infeasibility nor unboundedness',
    }
    return StructuredSolution(
        vertex.halt,
        x,
        value,
        vertex.iterations,
        duals_ub,
        duals_eq,
        bound=value,
        reason=reasons.get(vertex.halt, ''),
        evidence=evidence,
    )


def lp(problem: LinearProgram, *, method: str = 'simplex', tolerance: float = 1e-9) -> StructuredSolution:
    """Solve *problem* by *method*; *tolerance* is dimensionless (relative to the problem's scale)."""
    if method not in LP_METHODS:
        refuse(f'no LP method {method!r}; the methods are {sorted(LP_METHODS)}')
    vertex = LP_METHODS[method](standard_form(problem), tolerance=tolerance)
    return report(problem, vertex, evidence=(f'lp:{method}',))
