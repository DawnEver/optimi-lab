"""D31: the active-set QP and the bound QP, judged against scipy.optimize (a REFERENCE ORACLE, tests only).

Every comparison goes through :func:`agrees`, and the planted control calls the same function on a
displaced answer: a check that cannot fail on a wrong point is no evidence about a right one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from optimi_lab.core.errors import Refusal
from optimi_lab.operator import active_set_qp, bound_qp

#: Agreement in the variables' own (dimensionless, unit-cube-like) scale.
TOLERANCE = 1e-6


def agrees(found: np.ndarray, reference: np.ndarray, tolerance: float = TOLERANCE) -> bool:
    """Pointwise agreement within *tolerance* -- the one check every comparison here uses."""
    return bool(np.max(np.abs(np.asarray(found) - np.asarray(reference))) <= tolerance)


def _convex(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    m = rng.normal(size=(n, n))
    return m @ m.T + n * np.eye(n), rng.normal(size=n)


def _reference(h, g, *, a_eq=None, b_eq=None, a_in=None, b_in=None, lo=None, hi=None) -> np.ndarray:
    constraints = []
    if a_eq is not None and len(a_eq):
        constraints.append({'type': 'eq', 'fun': lambda x: a_eq @ x - b_eq, 'jac': lambda _: a_eq})
    if a_in is not None and len(a_in):
        constraints.append({'type': 'ineq', 'fun': lambda x: b_in - a_in @ x, 'jac': lambda _: -a_in})
    bounds = None if lo is None else list(zip(lo, hi, strict=True))
    found = minimize(
        lambda x: 0.5 * x @ h @ x + g @ x,
        np.zeros(g.size) if lo is None else 0.5 * (lo + hi),
        jac=lambda x: h @ x + g,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-14, 'maxiter': 500},
    )
    assert found.success, found.message
    return found.x


@pytest.mark.parametrize('seed', range(8))
def test_d31_active_set_qp_agrees_with_slsqp_on_random_convex_qps(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 5
    h, g = _convex(rng, n)
    a_eq, b_eq = rng.normal(size=(1, n)), rng.normal(size=1)
    a_in, b_in = rng.normal(size=(4, n)), rng.uniform(0.1, 1.0, size=4)
    found = active_set_qp(h, g, a_eq=a_eq, b_eq=b_eq, a_in=a_in, b_in=b_in)
    assert found.feasible
    assert agrees(found.x, _reference(h, g, a_eq=a_eq, b_eq=b_eq, a_in=a_in, b_in=b_in))
    stationarity = h @ found.x + g + a_eq.T @ found.multipliers_eq + a_in.T @ found.multipliers_in
    assert np.max(np.abs(stationarity)) <= 1e-9
    assert np.all(found.multipliers_in >= -1e-12)
    assert np.max(np.abs(found.multipliers_in * (a_in @ found.x - b_in))) <= 1e-9


def test_d31_active_set_qp_starts_from_an_infeasible_point() -> None:
    h, g = np.eye(2), np.zeros(2)
    a_in, b_in = np.array([[-1.0, 0.0], [0.0, -1.0]]), np.array([-1.0, -2.0])  # x >= 1, y >= 2
    found = active_set_qp(h, g, a_in=a_in, b_in=b_in, x0=np.zeros(2))
    assert found.feasible and agrees(found.x, np.array([1.0, 2.0]))
    assert agrees(found.multipliers_in, np.array([1.0, 2.0]))


def test_d31_an_infeasible_qp_reports_its_least_violating_point_not_a_solution() -> None:
    a_in, b_in = np.array([[1.0], [-1.0]]), np.array([0.0, -1.0])  # x <= 0 and x >= 1
    found = active_set_qp(np.eye(1), np.zeros(1), a_in=a_in, b_in=b_in)
    assert not found.feasible
    assert 0.0 <= found.x[0] <= 1.0


def test_d31_a_rank_deficient_equality_set_is_refused() -> None:
    with pytest.raises(Refusal, match='rank'):
        active_set_qp(np.eye(2), np.zeros(2), a_eq=np.ones((2, 2)), b_eq=np.ones(2))


@pytest.mark.parametrize('seed', range(8))
def test_d31_bound_qp_agrees_with_slsqp(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    n = 6
    h, g = _convex(rng, n)
    g = 5 * g  # push the unconstrained minimiser outside the box so bounds bind
    lo, hi = -np.ones(n), np.ones(n)
    found = bound_qp(h, g, lo, hi)
    assert agrees(found.x, _reference(h, g, lo=lo, hi=hi))
    assert np.all(found.x >= lo) and np.all(found.x <= hi)


def test_d31_bound_qp_refuses_an_indefinite_model() -> None:
    with pytest.raises(Refusal, match='positive definite'):
        bound_qp(np.diag([1.0, -1.0]), np.ones(2), -np.ones(2), np.ones(2))


def test_d31_planted_control_a_displaced_answer_disagrees() -> None:
    h, g = _convex(np.random.default_rng(0), 3)
    reference = _reference(h, g, lo=-np.ones(3), hi=np.ones(3))
    assert agrees(bound_qp(h, g, -np.ones(3), np.ones(3)).x, reference)
    assert not agrees(reference + 10 * TOLERANCE, reference)
