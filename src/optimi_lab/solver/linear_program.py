"""D126 linear programs: the data, the one standard form both methods solve, and what a structured solver reports.

    minimise c.x   subject to   A_ub x <= b_ub,   A_eq x = b_eq,   lo <= x <= hi

(``lo`` defaults to 0 and ``hi`` to +inf, the convention of every LP code). :func:`standard_form`
rewrites it as ``min c_s.z + offset  s.t.  A_s z = b_s,  z >= 0`` with ``x = P z + q``:

* a variable with a finite ``lo`` is shifted, ``x = lo + z``; a finite ``hi`` then adds the row
  ``z + s = hi - lo``;
* a variable bounded above only is reflected, ``x = hi - z``;
* a free variable is split, ``x = z+ - z-``;
* each ``A_ub`` row takes its own slack.

The rows are ``[ub..., upper bounds..., eq...]``. When no variable is split every standard column is
an AFFINE function of ``x`` (``z = W x + w``), which is what lets a cut derived on the standard form be
written back on ``x`` (:mod:`optimi_lab.solver.branch_and_bound`).

:class:`StructuredSolution` is the report of every structured solver (P14): the
:class:`~optimi_lab.solver.smooth.Halt` vocabulary is the gradient methods', so ``UNBOUNDED`` and
``INFEASIBLE`` mean the same thing on either side. Duals are SENSITIVITIES, ``d value / d b`` (the
``marginals`` convention), so an active ``<=`` row of a minimisation has a non-positive dual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.solver.smooth import Halt

__all__ = ['LinearProgram', 'StandardForm', 'StructuredSolution', 'standard_form']


@dataclass(frozen=True, slots=True, eq=False)
class StructuredSolution:
    """Why it *halt*-ed, *x*, its *value*, *iterations*, the row *duals*, a proven *bound*, and *evidence*."""

    halt: Halt
    x: np.ndarray
    value: float
    iterations: int
    duals_ub: np.ndarray = field(default_factory=lambda: np.zeros(0))
    duals_eq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: A proven lower bound on the optimal value (branch and bound, Benders); the value itself otherwise.
    bound: float = np.nan
    reason: str = ''
    #: Named choices that shaped the answer, e.g. ``'cuts:gomory_mixed_integer:3'`` -- never silent.
    evidence: tuple[str, ...] = ()


def _rows(a, n: int, what: str) -> np.ndarray:
    rows = np.zeros((0, n)) if a is None else np.atleast_2d(np.asarray(a, dtype=float))
    if rows.size == 0:
        return np.zeros((0, n))
    if rows.shape[1] != n:
        refuse(f'{what} has {rows.shape[1]} column(s) for {n} variable(s)')
    return rows


def _side(b, m: int, what: str) -> np.ndarray:
    side = np.zeros(0) if b is None else np.atleast_1d(np.asarray(b, dtype=float))
    if side.size != m:
        refuse(f'{what} has {side.size} entr(ies) for {m} row(s)')
    return side


@dataclass(frozen=True, slots=True, eq=False)
class _LinearProgramFields:
    c: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray
    a_eq: np.ndarray
    b_eq: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


@dataclass(frozen=True, slots=True, eq=False, init=False)
class LinearProgram(_LinearProgramFields):
    """See the module docstring; every array is read-only once built."""

    def __init__(self, c, *, a_ub=None, b_ub=None, a_eq=None, b_eq=None, lo=None, hi=None) -> None:
        c = np.atleast_1d(np.asarray(c, dtype=float))
        n = c.size
        a_ub, a_eq = _rows(a_ub, n, 'a_ub'), _rows(a_eq, n, 'a_eq')
        parts = {
            'c': c,
            'a_ub': a_ub,
            'b_ub': _side(b_ub, a_ub.shape[0], 'b_ub'),
            'a_eq': a_eq,
            'b_eq': _side(b_eq, a_eq.shape[0], 'b_eq'),
            'lo': np.zeros(n) if lo is None else _side(lo, n, 'lo'),
            'hi': np.full(n, np.inf) if hi is None else _side(hi, n, 'hi'),
        }
        if np.any(parts['lo'] > parts['hi']):
            refuse(f'lo exceeds hi at variable(s) {np.flatnonzero(parts["lo"] > parts["hi"]).tolist()}')
        for part in parts.values():
            part.setflags(write=False)
        _LinearProgramFields.__init__(self, **parts)

    @property
    def width(self) -> int:
        return self.c.size

    @property
    def bounds(self) -> tuple[tuple[float | None, float | None], ...]:
        """``(lo, hi)`` per variable with ``None`` for an infinite side."""
        return tuple(
            (None if np.isinf(a) else float(a), None if np.isinf(b) else float(b))
            for a, b in zip(self.lo, self.hi, strict=True)
        )

    def with_bounds(self, lo: np.ndarray, hi: np.ndarray) -> LinearProgram:
        """The same program over new bounds (a branch-and-bound node)."""
        return LinearProgram(self.c, a_ub=self.a_ub, b_ub=self.b_ub, a_eq=self.a_eq, b_eq=self.b_eq, lo=lo, hi=hi)

    def with_rows(self, a_ub: np.ndarray, b_ub: np.ndarray) -> LinearProgram:
        """The same program with extra ``<=`` rows appended (a cut, a Benders cut)."""
        return LinearProgram(
            self.c,
            a_ub=np.vstack([self.a_ub, a_ub]),
            b_ub=np.concatenate([self.b_ub, b_ub]),
            a_eq=self.a_eq,
            b_eq=self.b_eq,
            lo=self.lo,
            hi=self.hi,
        )


@dataclass(frozen=True, slots=True, eq=False)
class StandardForm:
    """``min c.z + offset, a z = b, z >= 0`` with ``x = p z + q``; *affine* is ``(W, w)`` with ``z = W x + w`` or None."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    offset: float
    p: np.ndarray
    q: np.ndarray
    n_ub: int
    n_upper: int
    affine: tuple[np.ndarray, np.ndarray] | None

    def rows_of(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The standard-row duals *y* split into the original ``(ub, eq)`` rows."""
        return y[: self.n_ub], y[self.n_ub + self.n_upper :]


def standard_form(problem: LinearProgram) -> StandardForm:
    """See the module docstring."""
    n = problem.width
    columns: list[np.ndarray] = []  # each a column of P
    q = np.zeros(n)
    inverse: list[tuple[np.ndarray, float]] = []  # z_j = row . x + const
    upper: list[tuple[int, float]] = []  # (std column, hi - lo)
    split = False
    for j, (lo, hi) in enumerate(zip(problem.lo, problem.hi, strict=True)):
        unit = np.eye(n)[j]
        if np.isfinite(lo):
            q[j] = lo
            columns.append(unit)
            inverse.append((unit, -lo))
            if np.isfinite(hi):
                upper.append((len(columns) - 1, hi - lo))
        elif np.isfinite(hi):
            q[j] = hi
            columns.append(-unit)
            inverse.append((-unit, hi))
        else:
            split = True
            columns += [unit, -unit]
    p = np.array(columns).T.reshape(n, -1)
    n_x, n_ub, n_upper, n_eq = p.shape[1], problem.a_ub.shape[0], len(upper), problem.a_eq.shape[0]
    a = np.zeros((n_ub + n_upper + n_eq, n_x + n_ub + n_upper))
    a[:n_ub, :n_x] = problem.a_ub @ p
    a[:n_ub, n_x : n_x + n_ub] = np.eye(n_ub)
    for k, (column, _) in enumerate(upper):
        a[n_ub + k, column] = 1.0
        a[n_ub + k, n_x + n_ub + k] = 1.0
    a[n_ub + n_upper :, :n_x] = problem.a_eq @ p
    b = np.concatenate([
        problem.b_ub - problem.a_ub @ q,
        [width for _, width in upper],
        problem.b_eq - problem.a_eq @ q,
    ])
    c = np.concatenate([problem.c @ p, np.zeros(n_ub + n_upper)])
    affine = None
    if not split:
        slack_ub = [(-row, b_i) for row, b_i in zip(problem.a_ub, problem.b_ub, strict=True)]
        slack_upper = [(-inverse[column][0], width - inverse[column][1]) for column, width in upper]
        pairs = inverse + slack_ub + slack_upper
        affine = (np.array([r for r, _ in pairs]).reshape(-1, n), np.array([k for _, k in pairs], dtype=float))
    return StandardForm(a, b, c, float(problem.c @ q), p, q, n_ub, n_upper, affine)
