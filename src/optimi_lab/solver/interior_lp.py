"""D126 Mehrotra's predictor-corrector interior point on a :class:`~optimi_lab.solver.linear_program.StandardForm`.

``min c.z, A z = b, z >= 0`` and its dual ``A^T y + s = c, s >= 0`` (Mehrotra 1992; Nocedal and
Wright, Algorithm 14.3). Each iterate solves the normal equations ``A (Z/S) A^T dy = r`` through
numerics-lab's Cholesky factor (``numerics_lab.linear.cholesky``), once for the affine predictor and
once, on the same factor, for the centred corrector with ``sigma = (mu_aff / mu)^3``.

It stops when the primal and dual residuals, relative to ``1 + |b|`` and ``1 + |c|``, and the duality
gap, relative to ``1 + |c.z|``, are all below *tolerance* (dimensionless). This basic form does NOT
certify infeasibility or unboundedness: an iterate that diverges, or a limit reached, is reported
as ``ITERATION_LIMIT`` with that reason, and the simplex is the method that names the two cases.
Dependent equality rows are refused (the normal equations would be singular).
"""

from __future__ import annotations

import numpy as np
from numerics_lab.linear.cholesky import jittered_cholesky
from scipy.linalg import cho_solve

from optimi_lab.core.errors import refuse
from optimi_lab.solver.linear_program import StandardForm
from optimi_lab.solver.simplex import Vertex
from optimi_lab.solver.smooth import Halt

__all__ = ['mehrotra']

#: The fraction of the step to the boundary taken each iteration.
_STEP_FRACTION = 0.99
#: An iterate this large (in the problem's own units) is declared divergent.
_DIVERGED = 1e12


def _step_to_boundary(v: np.ndarray, dv: np.ndarray) -> float:
    falling = dv < 0
    return min(1.0, float(np.min(-v[falling] / dv[falling]))) if falling.any() else 1.0


def _start(a, b, c, factor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mehrotra's starting point: least-norm ``z``, least-squares ``y``, both shifted positive."""
    z = a.T @ cho_solve(factor, b)
    y = cho_solve(factor, a @ c)
    s = c - a.T @ y
    z = z + max(-1.5 * float(np.min(z, initial=0.0)), 0.0)
    s = s + max(-1.5 * float(np.min(s, initial=0.0)), 0.0)
    zs = float(z @ s)
    z = z + 0.5 * zs / max(float(np.sum(s)), 1e-300) + 1e-8
    s = s + 0.5 * zs / max(float(np.sum(z)), 1e-300) + 1e-8
    return z, y, s


def mehrotra(form: StandardForm, *, tolerance: float = 1e-9, max_iterations: int = 200) -> Vertex:
    """See the module docstring."""
    a, b, c = form.a, form.b, form.c
    m, n = a.shape
    if m and np.linalg.matrix_rank(a) < m:
        refuse(f'the {m} standard rows have rank {np.linalg.matrix_rank(a)}; drop the dependent equality rows')
    rows = np.arange(m)
    z, y, s = _start(a, b, c, (jittered_cholesky(a @ a.T).lower, True))
    for iteration in range(1, max_iterations + 1):
        r_p, r_d = b - a @ z, c - a.T @ y - s
        mu = float(z @ s) / n
        primal, dual = float(c @ z), float(b @ y)
        if (
            np.linalg.norm(r_p) <= tolerance * (1.0 + np.linalg.norm(b))
            and np.linalg.norm(r_d) <= tolerance * (1.0 + np.linalg.norm(c))
            and abs(primal - dual) <= tolerance * (1.0 + abs(primal))
        ):
            return Vertex(Halt.CONVERGED, z, y, (), rows, iteration)
        if max(np.max(np.abs(z)), np.max(np.abs(y), initial=0.0)) > _DIVERGED:
            break
        d = z / s
        factor = (jittered_cholesky(a @ (d[:, None] * a.T)).lower, True)

        def direction(r_zs: np.ndarray, *, rhs=(z, s, d, factor, r_p, r_d)) -> tuple[np.ndarray, ...]:
            z, s, d, factor, r_p, r_d = rhs
            dy = cho_solve(factor, r_p - a @ (r_zs / s - d * r_d))
            ds = r_d - a.T @ dy
            return (r_zs - z * ds) / s, dy, ds

        dz, dy, ds = direction(-z * s)
        alpha_p, alpha_d = _step_to_boundary(z, dz), _step_to_boundary(s, ds)
        mu_affine = float((z + alpha_p * dz) @ (s + alpha_d * ds)) / n
        sigma = (mu_affine / mu) ** 3
        dz, dy, ds = direction(-z * s - dz * ds + sigma * mu)
        alpha_p = min(1.0, _STEP_FRACTION * _step_to_boundary(z, dz))
        alpha_d = min(1.0, _STEP_FRACTION * _step_to_boundary(s, ds))
        z, y, s = z + alpha_p * dz, y + alpha_d * dy, s + alpha_d * ds
    return Vertex(Halt.ITERATION_LIMIT, z, y, (), rows, max_iterations)
