"""D126 linear programming: the revised simplex and Mehrotra's interior point on one standard form.

Closed forms: ``max x + y`` over ``x + 2y <= 4, 3x + y <= 6, x, y >= 0`` has its vertex at
``(8/5, 6/5)``, value ``-14/5`` as a minimisation, with shadow prices ``(-2/5, -1/5)`` (the dual
solves ``[1 3; 2 1] u = (1, 1)``). Random instances are checked against ``scipy.optimize.linprog``
(HiGHS), a TEST-ONLY oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from optimi_lab.core.errors import Refusal
from optimi_lab.problem import Form, Linear
from optimi_lab.solver import Halt, LinearProgram, Requires, lp

#: Agreement of the optimal value and the point with the closed form / HiGHS, in the problem's own
#: (dimensionless) units, relative to ``1 + |value|``.
VALUE_TOLERANCE = 1e-7
#: The interior point stops on a relative gap, so its point agrees to this, in the variables' units.
IPM_POINT_TOLERANCE = 1e-5

VERTEX = LinearProgram(c=[-1.0, -1.0], a_ub=[[1.0, 2.0], [3.0, 1.0]], b_ub=[4.0, 6.0])


@pytest.mark.parametrize('method', ['simplex', 'ipm'])
def test_d126_the_vertex_value_and_shadow_prices_match_the_closed_form(method: str) -> None:
    found = lp(VERTEX, method=method)
    assert found.halt is Halt.CONVERGED
    assert np.allclose(found.x, [1.6, 1.2], atol=IPM_POINT_TOLERANCE)
    assert abs(found.value + 2.8) <= VALUE_TOLERANCE * 3.8
    assert np.allclose(found.duals_ub, [-0.4, -0.2], atol=IPM_POINT_TOLERANCE)


@pytest.mark.parametrize('method', ['simplex', 'ipm'])
def test_d126_bounds_free_variables_and_equalities_are_folded_into_the_standard_form(method: str) -> None:
    # min x0 - x1 + x2  s.t.  x0 + x1 + x2 = 2,  x0 - x2 <= 1,  -1 <= x0 <= 3,  x1 <= 1.5 (free below), x2 free
    problem = LinearProgram(
        c=[1.0, -1.0, 1.0],
        a_ub=[[1.0, 0.0, -1.0]],
        b_ub=[1.0],
        a_eq=[[1.0, 1.0, 1.0]],
        b_eq=[2.0],
        lo=[-1.0, -np.inf, -np.inf],
        hi=[3.0, 1.5, np.inf],
    )
    reference = linprog(problem.c, problem.a_ub, problem.b_ub, problem.a_eq, problem.b_eq, list(problem.bounds))
    found = lp(problem, method=method)
    assert found.halt is Halt.CONVERGED
    assert abs(found.value - reference.fun) <= VALUE_TOLERANCE * (1 + abs(reference.fun))


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('method', ['simplex', 'ipm'])
def test_d126_gate_random_feasible_bounded_programs_agree_with_highs(method: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, m_ub, m_eq = 6, 5, 2
    x_feasible = rng.uniform(0, 1, n)
    a_ub, a_eq = rng.normal(size=(m_ub, n)), rng.normal(size=(m_eq, n))
    problem = LinearProgram(
        c=rng.normal(size=n),
        a_ub=a_ub,
        b_ub=a_ub @ x_feasible + rng.uniform(0.1, 1, m_ub),
        a_eq=a_eq,
        b_eq=a_eq @ x_feasible,
        lo=np.zeros(n),
        hi=np.full(n, 2.0),
    )
    reference = linprog(problem.c, problem.a_ub, problem.b_ub, problem.a_eq, problem.b_eq, list(problem.bounds))
    found = lp(problem, method=method)
    assert found.halt is Halt.CONVERGED
    assert abs(found.value - reference.fun) <= VALUE_TOLERANCE * (1 + abs(reference.fun))
    assert np.allclose(found.duals_ub, reference.ineqlin.marginals, atol=IPM_POINT_TOLERANCE * 10)


def test_d126_infeasible_and_unbounded_programs_are_named_never_answered() -> None:
    infeasible = LinearProgram(c=[1.0], a_ub=[[1.0], [-1.0]], b_ub=[1.0, -2.0])
    unbounded = LinearProgram(c=[-1.0, 0.0], a_ub=[[-1.0, 1.0]], b_ub=[1.0])
    assert lp(infeasible).halt is Halt.INFEASIBLE
    assert lp(unbounded).halt is Halt.UNBOUNDED


def test_d126_a_degenerate_vertex_does_not_cycle_under_blands_rule() -> None:
    # Beale's cycling example (Chvatal 1983, ch. 3): Dantzig's rule cycles here; the optimum is -1/20.
    problem = LinearProgram(
        c=[-0.75, 150.0, -0.02, 6.0],
        a_ub=[[0.25, -60.0, -0.04, 9.0], [0.5, -90.0, -0.02, 3.0], [0.0, 0.0, 1.0, 0.0]],
        b_ub=[0.0, 0.0, 1.0],
    )
    found = lp(problem)
    assert found.halt is Halt.CONVERGED
    assert abs(found.value + 0.05) <= VALUE_TOLERANCE


def test_d126_planted_control_a_wrong_cost_moves_the_vertex() -> None:
    found = lp(LinearProgram(c=[-1.0, -3.0], a_ub=VERTEX.a_ub, b_ub=VERTEX.b_ub))
    assert not np.allclose(found.x, [1.6, 1.2], atol=0.1)


def test_d126_shape_disagreement_and_an_unknown_method_are_refused() -> None:
    with pytest.raises(Refusal, match='a_ub'):
        LinearProgram(c=[1.0, 1.0], a_ub=[[1.0]], b_ub=[1.0])
    with pytest.raises(Refusal, match='simplex'):
        lp(VERTEX, method='ellipsoid')


def test_d126_lp_is_registered_for_declared_linear_outputs_and_rows() -> None:
    needs = Requires.of('lp')
    assert needs.forms == {Form.LINEAR}
    assert needs.constraints == {Linear}
