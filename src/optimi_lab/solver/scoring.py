"""Score told results in the minimised objective frame, optionally scalarised to one column.

Every algorithm works on the MINIMISED, normalised objective rows (:meth:`optimi_lab.problem.Objective.minimized`)
and the total scaled violation (:meth:`optimi_lab.oracle.Result.violation`); ``scalarise`` maps those rows to one
score per row, which is how a single-objective algorithm is pointed at several objectives.
"""

from collections.abc import Callable, Sequence

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.oracle import Result
from optimi_lab.problem import Problem

__all__ = ['Scalarise', 'require_one_score', 'score_rows']

#: Minimised objective rows ``(n, m)`` -> one score per row ``(n,)``.
Scalarise = Callable[[np.ndarray], np.ndarray]


def require_one_score(problem: Problem, scalarise: Scalarise | None, what: str) -> None:
    """Refuse a single-score algorithm on several objectives unless a scalarisation is supplied."""
    if scalarise is None and len(problem.objectives) > 1:
        refuse(
            f'{what} minimises one score and the problem has {len(problem.objectives)} objectives; '
            f'pass scalarise= (an optimi_lab.operator scalarisation bound to its weights)'
        )


def score_rows(
    problem: Problem, scalarise: Scalarise | None, results: Sequence[Result]
) -> tuple[np.ndarray, np.ndarray]:
    """``(values, violation)`` of OK *results*: minimised rows (or one scalarised column), violation per row."""
    values = np.array([[o.minimized(r.cell.values[o.name]) for o in problem.objectives] for r in results])
    values = values.reshape(len(results), len(problem.objectives))
    if scalarise is not None:
        values = np.asarray(scalarise(values), dtype=float).reshape(-1, 1)
    return values, np.array([r.violation(problem) for r in results])
