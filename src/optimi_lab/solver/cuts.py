"""D128 Gomory mixed-integer cuts read off an optimal simplex basis, written back on the original variables.

For a basic integer standard column with tableau row ``z_B + sum_N abar_j z_j = bbar`` and fractional
part ``f0 = frac(bbar)``, the GMI cut (Gomory 1960; Cornuejols 2008, eq. 5) is

    sum_{j int, f_j <= f0} f_j / f0 z_j + sum_{j int, f_j > f0} (1 - f_j) / (1 - f0) z_j
        + sum_{j cont, abar_j > 0} abar_j / f0 z_j + sum_{j cont, abar_j < 0} -abar_j / (1 - f0) z_j  >=  1

It is valid for every mixed-integer point and cuts off the vertex (where every ``z_N = 0``). A
standard column is integer when it is a shifted or reflected integer variable with an integral
bound; slacks are taken as continuous, which keeps the cut valid without assuming integral rows.
Through ``z = W x + w`` (:func:`~optimi_lab.solver.linear_program.standard_form`) the cut becomes
one ``<=`` row on ``x``; a program with a split (free) variable has no such map and gets no cut.
"""

from __future__ import annotations

import numpy as np

from optimi_lab.solver.linear_program import StandardForm
from optimi_lab.solver.simplex import Vertex, basis_solve

__all__ = ['gomory_cuts', 'integer_columns']

#: A basic value this close to an integer (in the variable's own unit) is not cut on.
_AWAY = 1e-3
#: A cut whose largest to smallest nonzero coefficient ratio exceeds this is numerically unsafe.
_DYNAMISM = 1e6


def integer_columns(form: StandardForm, integrality: np.ndarray) -> np.ndarray:
    """Which standard columns are integer-valued at every integer-feasible point."""
    integer = np.zeros(form.a.shape[1], dtype=bool)
    for j in range(form.p.shape[1]):
        k = int(np.flatnonzero(form.p[:, j])[0])
        integer[j] = bool(integrality[k]) and float(form.q[k]).is_integer()
    return integer


def _fraction(v: np.ndarray) -> np.ndarray:
    return v - np.floor(v)


def gomory_cuts(form: StandardForm, vertex: Vertex, integer: np.ndarray, *, limit: int = 10):
    """Up to *limit* GMI cuts ``(A, b)`` with ``A x <= b``, most fractional rows first; empty when none apply."""
    n = form.p.shape[0]
    if form.affine is None or not vertex.basis:
        return np.zeros((0, n)), np.zeros(0)
    w_map, w_shift = form.affine
    a = form.a[vertex.rows]
    basis = list(vertex.basis)
    values = vertex.z[basis]
    fractions = _fraction(values)
    distance = np.minimum(fractions, 1.0 - fractions)
    order = [r for r in np.argsort(-distance) if integer[basis[r]] and distance[r] > _AWAY][:limit]
    nonbasic = np.setdiff1d(np.arange(a.shape[1]), basis)
    rows, sides = [], []
    for r in order:
        tableau = basis_solve(a[:, basis].T, np.eye(len(basis))[r]) @ a
        f0 = fractions[r]
        coefficients = np.zeros(a.shape[1])
        for j in nonbasic:
            abar = tableau[j]
            if integer[j]:
                f = _fraction(np.array([abar]))[0]
                coefficients[j] = f / f0 if f <= f0 else (1.0 - f) / (1.0 - f0)
            else:
                coefficients[j] = abar / f0 if abar > 0 else -abar / (1.0 - f0)
        row = -(coefficients @ w_map)
        side = float(coefficients @ w_shift) - 1.0
        magnitudes = np.abs(row[np.abs(row) > 1e-12])
        if magnitudes.size and magnitudes.max() / magnitudes.min() <= _DYNAMISM:
            rows.append(row)
            sides.append(side)
    return (np.array(rows).reshape(-1, n), np.array(sides))
