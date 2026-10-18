"""P9 rulings (b) and (c): Newton-CG's Hessian-vector products, and the stopping bar every method shares.

(b) A declared hvp is used; gradient differencing is a NAMED fallback chosen by the caller and
recorded in the solution's evidence -- never a silent default.

(c) A method stops when the KKT residual is ``<= rtol * |grad f(x0)|`` OR ``<= atol``, ``atol`` in the
normalised objective's units per unit-cube width, both declared. The planted control is a start
whose gradient is ~0: the relative bar is then ~0, and the method must stop on ``atol`` -- not
divide by zero, not run to its iteration limit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Record, Status
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess_prod

from optimi_lab.core.errors import Refusal
from optimi_lab.operator import Linearization
from optimi_lab.oracle import Action, Capability, Information, Result, from_function
from optimi_lab.problem import Objective, Problem, Real, Space
from optimi_lab.run import Budget, run
from optimi_lab.solver import METHODS, GradientSolver, Halt, kkt_bar, newton_cg

X0 = np.array([-1.2, 1.0, -0.5, 0.3])
LO, HI = -2 * np.ones(4), 2 * np.ones(4)


def _counted():
    calls = []

    def evaluate(x: np.ndarray) -> Linearization:
        calls.append(x)
        return Linearization(x=x, f=[rosen(x)], df=[rosen_der(x)])

    return evaluate, calls


def test_p9b_newton_cg_uses_a_declared_hvp_and_says_so() -> None:
    evaluate, calls = _counted()
    found = newton_cg(evaluate, X0, LO, HI, hvp=rosen_hess_prod)
    reference = minimize(rosen, X0, jac=rosen_der, hessp=rosen_hess_prod, method='trust-ncg', options={'gtol': 1e-10})
    assert found.halt is Halt.CONVERGED and np.max(np.abs(found.x - reference.x)) <= 1e-5
    assert found.evidence == ('hvp:declared',)
    assert len(calls) <= found.iterations + 2  # one evaluation per step: no gradient differencing


def test_p9b_gradient_differencing_is_named_and_recorded() -> None:
    evaluate, calls = _counted()
    found = newton_cg(evaluate, X0, LO, HI, hvp='finite_difference')
    assert found.halt is Halt.CONVERGED and found.evidence == ('hvp:finite_difference',)
    assert len(calls) > 2 * found.iterations  # the fallback pays a gradient per product


def test_p9b_newton_cg_has_no_silent_default() -> None:
    evaluate, _ = _counted()
    with pytest.raises(Refusal, match='finite_difference'):
        newton_cg(evaluate, X0, LO, HI)


SPACE = Space(tuple(Real(f'x{i}', -2.0, 2.0) for i in range(4)))
PROBLEM = Problem(SPACE, (Objective('f'),))


def _oracle(declare_hvp: bool):
    provides = {Information.VALUES, Information.VJP} | ({Information.HVP} if declare_hvp else set())
    directions = []

    def evaluate(action: Action) -> Result:
        x = np.array([action.design[f'x{i}'] for i in range(4)])
        cell = Cell(Status.OK, values={'f': rosen(x)})
        extras = {'gradient': {'f': 4.0 * rosen_der(x)}}
        if action.direction is not None:
            v = np.array(action.direction)
            directions.append(v)
            extras['hvp'] = {'f': 16.0 * rosen_hess_prod(x, v)}  # unit coordinates: (d x / d u)^2 = 16
        return Result(action, cell, **extras)

    return from_function(evaluate, Capability(outputs=('f',), provides=frozenset(provides))), directions


def test_p9b_the_solver_asks_the_oracle_for_hvp_with_a_direction(tmp_path: Path) -> None:
    oracle, directions = _oracle(declare_hvp=True)
    solver = GradientSolver(PROBLEM, oracle.capability, 'newton_cg', start={f'x{i}': X0[i] for i in range(4)})
    run(PROBLEM, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=2000))
    assert solver.solution.halt is Halt.CONVERGED and directions
    assert solver.solution.evidence == ('hvp:declared',)
    assert np.max(np.abs(np.array(list(solver.design().values())) - 1.0)) <= 1e-5


def test_p9b_an_oracle_without_hvp_needs_the_named_fallback(tmp_path: Path) -> None:
    oracle, directions = _oracle(declare_hvp=False)
    with pytest.raises(Refusal, match='finite_difference'):
        GradientSolver(PROBLEM, oracle.capability, 'newton_cg')
    solver = GradientSolver(PROBLEM, oracle.capability, 'newton_cg', hvp='finite_difference')
    run(PROBLEM, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=4000))
    assert solver.solution.evidence == ('hvp:finite_difference',) and not directions


def test_p9c_the_bar_is_the_larger_of_the_relative_and_the_absolute_floor() -> None:
    assert kkt_bar(np.array([2.0, -4.0]), rtol=1e-6, atol=1e-9) == pytest.approx(4e-6)
    assert kkt_bar(np.zeros(2), rtol=1e-6, atol=1e-9) == pytest.approx(1e-9)
    with pytest.raises(Refusal, match='atol'):
        kkt_bar(np.zeros(2), rtol=1e-6, atol=0.0)


def _flat(x: np.ndarray) -> Linearization:
    """``(x - 0.5)^4``: at x0 = 0.5 + 1e-6 the gradient is ~4e-18 -- the relative bar is ~0."""
    return Linearization(
        x=x, f=[float(np.sum((x - 0.5) ** 4))], df=[4 * (x - 0.5) ** 3], c=[x[0] - 2.0], dc=[[1.0, 0.0]]
    )


def _box(x: np.ndarray) -> Linearization:
    p = _flat(x)
    return Linearization(x=p.x, f=p.f, df=p.df)


@pytest.mark.parametrize('name', sorted(set(METHODS) - {'mgda', 'pareto_continuation'}))
def test_p9c_planted_control_a_flat_start_stops_on_atol(name: str) -> None:
    box = name in {'lbfgsb', 'projected_gradient', 'newton_cg'}
    settings = {'hvp': 'finite_difference'} if name == 'newton_cg' else {}
    found = METHODS[name](_box if box else _flat, np.full(2, 0.5 + 1e-6), np.zeros(2), np.ones(2), **settings)
    assert found.halt is Halt.CONVERGED
    assert (
        found.iterations <= 50
    )  # far below every iteration limit: the floor stopped it (IPM still drives its barrier down)
