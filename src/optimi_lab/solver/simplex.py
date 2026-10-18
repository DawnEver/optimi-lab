"""D126 the revised simplex method on a :class:`~optimi_lab.solver.linear_program.StandardForm`.

Two phases (Dantzig; Chvatal 1983, ch. 8): phase 1 minimises the sum of one artificial per row from
the all-artificial basis; a positive optimum is INFEASIBLE, never a point. Artificials left basic at
zero are pivoted out, and a row whose artificial cannot leave is linearly dependent and is dropped
(its dual is zero). Phase 2 minimises the cost from that basis.

Pricing and the ratio test follow BLAND'S RULE (smallest entering index, smallest leaving basic
index among ties), which cannot cycle on a degenerate vertex. Every basis solve -- ``B x_B = b``,
``B^T y = c_B``, ``B d = a_q`` -- goes through numerics-lab (``numerics_lab.linear``).

The reduced-cost tolerance is relative to ``1 + max |c|`` and the pivot tolerance is absolute on the
scale of the (dimensionless) basis-solve column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numerics_lab.linear import solve_linear_system

from optimi_lab.solver.linear_program import StandardForm
from optimi_lab.solver.smooth import Halt

__all__ = ['Vertex', 'basis_solve', 'simplex']

#: Pivot elements below this (in the units of ``B^-1 a_q``, dimensionless) are treated as zero.
_PIVOT = 1e-9


@dataclass(frozen=True, slots=True, eq=False)
class Vertex:
    """The standard-form *z*, row duals *y* (zero on dropped rows), the final *basis* and kept *rows*."""

    halt: Halt
    z: np.ndarray
    y: np.ndarray
    basis: tuple[int, ...]
    rows: np.ndarray
    iterations: int


def basis_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``matrix^-1 rhs`` for a square basis, by numerics-lab's direct solve."""
    return np.asarray(solve_linear_system(sp.csc_matrix(matrix), np.asarray(rhs, dtype=float)), dtype=float)


def _iterate(a, b, c, basis: list[int], *, tolerance: float, limit: int) -> tuple[Halt, int]:
    """Bland-rule simplex from a feasible *basis* (updated in place): ``(halt, iterations)``."""
    reduced_bar = tolerance * (1.0 + float(np.max(np.abs(c), initial=0.0)))
    for iteration in range(1, limit + 1):
        matrix = a[:, basis]
        x_b = basis_solve(matrix, b)
        y = basis_solve(matrix.T, c[basis])
        reduced = c - a.T @ y
        reduced[basis] = 0.0
        entering = np.flatnonzero(reduced < -reduced_bar)
        if entering.size == 0:
            return Halt.CONVERGED, iteration
        q = int(entering[0])
        d = basis_solve(matrix, a[:, q])
        rising = np.flatnonzero(d > _PIVOT)
        if rising.size == 0:
            return Halt.UNBOUNDED, iteration
        ratios = np.maximum(x_b[rising], 0.0) / d[rising]
        ties = rising[ratios <= ratios.min() + _PIVOT * (1.0 + ratios.min())]
        leaving = int(min(ties, key=lambda r: basis[r]))
        basis[leaving] = q
    return Halt.ITERATION_LIMIT, limit


def simplex(form: StandardForm, *, tolerance: float = 1e-9, max_iterations: int | None = None) -> Vertex:
    """Two-phase revised simplex; see the module docstring."""
    m, n = form.a.shape
    limit = 50 * (m + n) + 50 if max_iterations is None else max_iterations
    sign = np.where(form.b < 0, -1.0, 1.0)
    a, b = form.a * sign[:, None], form.b * sign
    phase_one = np.hstack([a, np.eye(m)])
    basis = list(range(n, n + m))
    halt, spent = _iterate(
        phase_one, b, np.concatenate([np.zeros(n), np.ones(m)]), basis, tolerance=tolerance, limit=limit
    )
    infeasibility = float(np.sum(basis_solve(phase_one[:, basis], b)[np.array(basis) >= n])) if m else 0.0
    if halt is not Halt.CONVERGED or infeasibility > tolerance * (1.0 + float(np.max(np.abs(b), initial=0.0))):
        z = np.zeros(n)
        return Vertex(Halt.INFEASIBLE, z, np.zeros(m), tuple(basis), np.arange(m), spent)
    keep = np.ones(m, dtype=bool)
    for position in range(m):
        if basis[position] < n:
            continue
        row = basis_solve(phase_one[:, basis].T, np.eye(m)[position]) @ a
        row[[j for j in basis if j < n]] = 0.0
        candidate = int(np.argmax(np.abs(row)))
        if abs(row[candidate]) > _PIVOT:
            basis[position] = candidate
        else:
            keep[position] = False
    basis = [j for j, kept in zip(basis, keep, strict=True) if kept]
    a, b = a[keep], b[keep]
    if not a.shape[0]:
        halt, more = (Halt.UNBOUNDED if np.any(form.c < 0) else Halt.CONVERGED), 0
    else:
        halt, more = _iterate(a, b, form.c, basis, tolerance=tolerance, limit=limit)
    z = np.zeros(n)
    y = np.zeros(m)
    if basis:
        matrix = a[:, basis]
        z[basis] = np.maximum(basis_solve(matrix, b), 0.0)
        y[keep] = basis_solve(matrix.T, form.c[basis])
    return Vertex(halt, z, y * sign, tuple(basis), np.flatnonzero(keep), spent + more)
