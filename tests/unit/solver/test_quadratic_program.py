"""D127 quadratic programs: a route onto P9's active-set and bound QPs, checked by closed forms and KKT certificates.

* Euclidean projection onto the probability simplex (Duchi et al. 2008): ``x = max(v - tau, 0)``
  with ``tau`` from the sorted ``v`` -- the closed form for ``min 1/2 |x - v|^2, sum x = 1, x >= 0``.
* Random strictly convex QPs: the returned point is certified by its own KKT conditions (stationarity,
  primal feasibility, sign and complementarity of the sensitivities) -- a certificate, not an oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.solver import Halt, LinearProgram, qp

#: Agreement with the closed form / KKT residual, in the variables' (dimensionless) units.
TOLERANCE = 1e-8


def simplex_projection(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u) - 1.0
    rho = np.flatnonzero(u - cumulative / np.arange(1, v.size + 1) > 0)[-1]
    return np.maximum(v - cumulative[rho] / (rho + 1), 0.0)


@pytest.mark.parametrize('seed', range(4))
def test_d127_projection_onto_the_simplex_matches_the_closed_form(seed: int) -> None:
    v = np.random.default_rng(seed).normal(size=6)
    found = qp(LinearProgram(-v, a_eq=np.ones((1, 6)), b_eq=[1.0]), np.eye(6))
    assert found.halt is Halt.CONVERGED
    assert found.evidence == ('qp:active_set_qp',)
    assert np.allclose(found.x, simplex_projection(v), atol=TOLERANCE)


def test_d127_bounds_only_route_to_the_projected_newton_method() -> None:
    v = np.array([2.0, -3.0, 0.4])
    found = qp(LinearProgram(-v, lo=-np.ones(3), hi=np.ones(3)), np.eye(3))
    assert found.evidence == ('qp:bound_qp',)
    assert np.allclose(found.x, np.clip(v, -1, 1), atol=TOLERANCE)


@pytest.mark.parametrize('seed', range(6))
def test_d127_random_convex_qps_satisfy_their_kkt_certificate(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, m = 5, 4
    root = rng.normal(size=(n, n))
    h = root @ root.T + 0.1 * np.eye(n)
    a_ub, a_eq = rng.normal(size=(m, n)), rng.normal(size=(1, n))
    x_feasible = rng.normal(size=n)
    problem = LinearProgram(
        rng.normal(size=n) * 5,
        a_ub=a_ub,
        b_ub=a_ub @ x_feasible + 0.1,
        a_eq=a_eq,
        b_eq=a_eq @ x_feasible,
        lo=np.full(n, -np.inf),
    )
    found = qp(problem, h)
    x, u, w = found.x, found.duals_ub, found.duals_eq
    assert found.halt is Halt.CONVERGED
    assert np.max(np.abs(h @ x + problem.c - a_ub.T @ u - a_eq.T @ w)) <= 1e-6  # stationarity, d value / d b signs
    assert np.max(a_ub @ x - problem.b_ub) <= TOLERANCE
    assert np.max(u) <= TOLERANCE
    assert np.max(np.abs(u * (a_ub @ x - problem.b_ub))) <= 1e-6


def test_d127_an_infeasible_qp_is_named_with_its_least_violating_point() -> None:
    found = qp(LinearProgram([0.0], a_ub=[[1.0], [-1.0]], b_ub=[1.0, -2.0], lo=[-np.inf]), np.eye(1))
    assert found.halt is Halt.INFEASIBLE
    assert np.isnan(found.value)


def test_d127_planted_control_dropping_the_equality_row_misses_the_simplex_projection() -> None:
    v = np.random.default_rng(0).normal(size=6)
    found = qp(LinearProgram(-v), np.eye(6))
    assert not np.allclose(found.x, simplex_projection(v), atol=1e-3)
