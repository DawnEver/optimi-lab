"""D131 Benders decomposition and D132 ADMM, against the monolithic program and closed forms.

* Benders: random two-stage programs (continuous and mixed-integer first stage, with a first stage
  that can make the recourse infeasible) agree with the SAME program solved whole by
  ``scipy.optimize.milp`` (HiGHS, TEST-ONLY).
* ADMM: a separable lasso ``1/2 |x - v|^2 + lam |x|_1`` has the soft-threshold closed form; a box QP
  agrees with :func:`~optimi_lab.solver.qp` (P9's projected Newton through the D127 route).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint
from scipy.optimize import milp as highs_milp

from optimi_lab.solver import Halt, LinearProgram, admm, benders, box_prox, l1_prox, qp, quadratic_prox

#: Agreement of the optimal value with HiGHS, relative to ``1 + |value|`` (dimensionless).
VALUE_TOLERANCE = 1e-6
#: ADMM point agreement with the closed form, in the units of x (atol 1e-9, rtol 1e-7 stopping).
POINT_TOLERANCE = 1e-5


def two_stage(seed: int):
    rng = np.random.default_rng(seed)
    n, k, m = 3, 4, 5
    master = LinearProgram(rng.uniform(1, 3, n), a_ub=np.ones((1, n)), b_ub=[6.0], hi=np.full(n, 4.0))
    q = rng.uniform(1, 4, k)
    t = rng.uniform(0, 2, (m, n))
    w = -rng.uniform(0.5, 2, (m, k))  # W y <= h - T x reads: y must cover demand T x - h
    h = -rng.uniform(1, 4, m)
    return master, q, t, w, h


def monolithic(master, q, t, w, h, *, integer):
    n, k = master.width, q.size
    a = np.vstack([np.hstack([master.a_ub, np.zeros((1, k))]), np.hstack([t, w])])
    return highs_milp(
        np.concatenate([master.c, q]),
        integrality=np.concatenate([integer, np.zeros(k)]).astype(int),
        bounds=Bounds(np.zeros(n + k), np.concatenate([master.hi, np.full(k, np.inf)])),
        constraints=LinearConstraint(a, -np.inf, np.concatenate([master.b_ub, h])),
    )


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('integer', [False, True])
def test_d131_gate_benders_agrees_with_the_whole_program(seed: int, integer: bool) -> None:
    master, q, t, w, h = two_stage(seed)
    mask = np.full(master.width, integer)
    found = benders(master, q, t, w, h, integrality=mask)
    reference = monolithic(master, q, t, w, h, integer=mask)
    assert found.halt is Halt.CONVERGED
    assert abs(found.value - reference.fun) <= VALUE_TOLERANCE * (1 + abs(reference.fun))
    assert found.bound <= found.value + VALUE_TOLERANCE


def test_d131_a_first_stage_that_starves_the_recourse_draws_a_feasibility_cut() -> None:
    # y <= x - 1 and y >= 0.5: x = 0 (the cheap master start) leaves the recourse infeasible.
    master = LinearProgram([1.0], hi=[5.0])
    found = benders(master, [1.0], [[-1.0], [0.0]], [[1.0], [-1.0]], [-1.0, -0.5])
    assert found.halt is Halt.CONVERGED
    assert abs(found.value - 2.0) <= VALUE_TOLERANCE  # x = 1.5, y = 0.5


def test_d131_planted_control_one_master_solve_leaves_the_bounds_apart() -> None:
    found = benders(*two_stage(0), max_iterations=1)
    assert found.halt is Halt.ITERATION_LIMIT
    assert found.bound < found.value or np.isinf(found.value)


def test_d132_admm_reproduces_the_soft_threshold_of_a_separable_lasso() -> None:
    v = np.array([3.0, -0.2, 0.5, -2.0])
    found = admm(quadratic_prox(np.eye(4), -v), l1_prox(1.0), 4)
    assert found.halt is Halt.CONVERGED
    assert np.allclose(found.x, np.sign(v) * np.maximum(np.abs(v) - 1.0, 0.0), atol=POINT_TOLERANCE)


def test_d132_admm_agrees_with_the_box_qp_route() -> None:
    rng = np.random.default_rng(1)
    root = rng.normal(size=(5, 5))
    h, g = root @ root.T + 0.5 * np.eye(5), rng.normal(size=5) * 3
    found = admm(
        quadratic_prox(h, g), box_prox(-np.ones(5), np.ones(5)), 5, objective=lambda x: 0.5 * x @ h @ x + g @ x
    )
    reference = qp(LinearProgram(g, lo=-np.ones(5), hi=np.ones(5)), h)
    assert found.halt is Halt.CONVERGED
    assert np.allclose(found.x, reference.x, atol=POINT_TOLERANCE)
    assert abs(found.value - reference.value) <= POINT_TOLERANCE


def test_d132_planted_control_too_few_iterations_are_reported_unsettled() -> None:
    v = np.array([3.0, -0.2, 0.5, -2.0])
    found = admm(quadratic_prox(np.eye(4), -v), l1_prox(1.0), 4, rho=1e-3, max_iterations=5)
    assert found.halt is Halt.ITERATION_LIMIT
