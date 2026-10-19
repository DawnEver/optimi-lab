"""P14 ruling 2026-10-19: a declared LINEAR/QUADRATIC form is solvable only through the oracle's coefficient capability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Record
from scipy.optimize import linprog, minimize

from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import COEFFICIENTS_FOR, Capability, Information, analytic
from optimi_lab.problem import BlackBox, Form, Linear, Objective, Problem, Real, Sense, Space, Structure
from optimi_lab.run import Budget, Termination, run
from optimi_lab.solver import Halt, Ineligible, Requires, StructuredSolver, match

SPACE = Space((Real('x', 0.0, 4.0), Real('y', -1.0, 3.0)))
LINEAR = frozenset({Form.LINEAR})
# maximise 3x + 2y + 1  s.t.  x + y <= 4 (oracle output 'load'),  x - y >= -2 (a Linear row),  the box
LP = Problem(
    SPACE,
    (Objective('profit', Sense.MAXIMIZE),),
    constraints=(BlackBox('load', '<=', 4.0), Linear('gap', {'x': 1.0, 'y': -1.0}, lo=-2.0)),
    structure=Structure({'profit': LINEAR, 'load': LINEAR}),
)
LP_ORACLE = analytic(SPACE, linear={'profit': [3.0, 2.0], 'load': [1.0, 1.0]}, constant={'profit': 1.0})


def _drive(problem: Problem, oracle, method: str, tmp_path: Path) -> StructuredSolver:
    solver = StructuredSolver(problem, oracle.capability, method)
    outcome = run(problem, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=10))
    assert outcome.stop.reason is Termination.SOLVER_DONE
    return solver


def test_p14_an_analytic_lp_is_solved_through_the_driver_and_agrees_with_linprog(tmp_path: Path) -> None:
    solver = _drive(LP, LP_ORACLE, 'lp', tmp_path)
    reference = linprog([-3.0, -2.0], A_ub=[[1.0, 1.0], [-1.0, 1.0]], b_ub=[4.0, 2.0], bounds=[(0.0, 4.0), (-1.0, 3.0)])
    assert solver.solution.halt is Halt.CONVERGED
    np.testing.assert_allclose(solver.solution.x, reference.x, atol=1e-8)
    design = solver.design()
    assert design['x'] == pytest.approx(reference.x[0]) and design['y'] == pytest.approx(reference.x[1])
    assert solver.optimum.cell.values['profit'] == pytest.approx(-reference.fun + 1.0)


def test_p14_an_analytic_qp_is_solved_through_the_driver_and_agrees_with_scipy(tmp_path: Path) -> None:
    h = np.array([[2.0, 0.5], [0.5, 1.0]])
    problem = Problem(
        SPACE,
        (Objective('f'),),
        constraints=(BlackBox('load', '>=', 3.0),),
        structure=Structure({
            'f': frozenset({Form.QUADRATIC, Form.CONVEX}),
            'load': frozenset({Form.QUADRATIC, Form.CONVEX}),
        }),
    )
    oracle = analytic(SPACE, linear={'f': [-1.0, -1.0], 'load': [1.0, 1.0]}, quadratic={'f': h})
    solver = _drive(problem, oracle, 'qp', tmp_path)
    reference = minimize(
        lambda v: 0.5 * v @ h @ v - v.sum(),
        [1.0, 1.0],
        bounds=[(0.0, 4.0), (-1.0, 3.0)],
        constraints=({'type': 'ineq', 'fun': lambda v: v.sum() - 3.0},),
        method='SLSQP',
        tol=1e-12,
    )
    assert solver.solution.halt is Halt.CONVERGED
    np.testing.assert_allclose(solver.solution.x, reference.x, atol=1e-6)


def test_p14_one_table_names_the_coefficients_each_form_needs() -> None:
    assert COEFFICIENTS_FOR == {
        Form.LINEAR: Information.LINEAR_COEFFICIENTS,
        Form.QUADRATIC: Information.QUADRATIC_COEFFICIENTS,
    }
    assert Information.LINEAR_COEFFICIENTS in LP_ORACLE.capability.provides


def test_p14_planted_control_a_linear_problem_without_the_capability_is_refused_by_name() -> None:
    values_only = Capability(outputs=('profit', 'load'))
    verdict = match(Requires.of('lp'), LP, values_only)
    assert isinstance(verdict, Ineligible) and 'LINEAR_COEFFICIENTS' in verdict.reason
    with pytest.raises(Refusal, match='LINEAR_COEFFICIENTS'):
        StructuredSolver(LP, values_only, 'lp')
    assert match(Requires.of('lp'), LP, LP_ORACLE.capability) != verdict  # the control: declared, it runs


def test_p14_a_log_scaled_variable_is_not_linear_and_is_refused() -> None:
    space = Space((Real('x', 1.0, 4.0, scale='log'),))
    problem = Problem(space, (Objective('f'),), structure=Structure({'f': LINEAR}))
    with pytest.raises(Refusal, match='log'):
        StructuredSolver(problem, analytic(space, linear={'f': [1.0]}).capability, 'lp')
