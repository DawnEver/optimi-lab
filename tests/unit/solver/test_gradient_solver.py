"""P9: the gradient methods driven by the run loop through ``GradientSolver``, against scipy.optimize.

The oracle declares its gradient (``Information.VJP``) through the P8 capability; the gradient is in
the space's UNIT coordinates, so each test's oracle applies the decode map's chain rule itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Reason, Record, Status
from scipy.optimize import minimize

from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Information, Result, from_function, with_finite_differences
from optimi_lab.problem import BlackBox, Integer, Linear, Normalization, Objective, Problem, Real, Relation, Scale
from optimi_lab.problem import Space as DesignSpace
from optimi_lab.run import Budget, Termination, run
from optimi_lab.solver import GradientSolver, Halt

SLOPES = frozenset({Information.VALUES, Information.VJP})
NAMES = ('a', 'b', 'c', 'd')
SPACE = DesignSpace(tuple(Real(n, 1.0, 5.0) for n in NAMES))
#: ``min a d (a + b + c) + c``  s.t.  ``abcd >= 25`` (oracle output) and ``a + b + c + d = 11`` (Linear).
PROBLEM = Problem(
    SPACE,
    (Objective('f'),),
    constraints=(
        BlackBox('product', Relation.AT_LEAST, 25.0),
        Linear('total', dict.fromkeys(NAMES, 1.0), lo=11.0, hi=11.0),
    ),
)
START = {'a': 1.5, 'b': 4.0, 'c': 3.0, 'd': 2.5}
TOLERANCE = 1e-5


def agrees(found: np.ndarray, reference: np.ndarray, tolerance: float = TOLERANCE) -> bool:
    return bool(np.max(np.abs(np.asarray(found) - np.asarray(reference))) <= tolerance)


def _values(x: np.ndarray) -> dict[str, float]:
    return {'f': x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2], 'product': float(np.prod(x))}


def _gradients(x: np.ndarray, planted: float = 0.0) -> dict[str, np.ndarray]:
    df = np.array([
        x[3] * (2 * x[0] + x[1] + x[2]) + planted,
        x[0] * x[3],
        x[0] * x[3] + 1,
        x[0] * (x[0] + x[1] + x[2]),
    ])
    dp = np.array([np.prod(x) / v for v in x])
    return {'f': 4.0 * df, 'product': 4.0 * dp}  # d design / d unit = hi - lo = 4


def _oracle(planted: float = 0.0, *, flaky: bool = False):
    seen: set[tuple[float, ...]] = set()

    def evaluate(action: Action) -> Result:
        x = np.array([action.design[n] for n in NAMES])
        key = tuple(x)
        if flaky and key not in seen:
            seen.add(key)
            return Result(action, Cell(Status.TIMEOUT, reason=Reason('timeout', 'licence queue')))
        values = _values(x)
        if Information.VJP not in action.want:
            return Result(action, Cell(Status.OK, values=values))
        return Result(action, Cell(Status.OK, values=values), gradient=_gradients(x, planted))

    return from_function(evaluate, Capability(outputs=('f', 'product'), provides=SLOPES))


def _reference() -> np.ndarray:
    found = minimize(
        lambda x: _values(x)['f'],
        np.array([START[n] for n in NAMES]),
        method='SLSQP',
        bounds=[(1.0, 5.0)] * 4,
        constraints=[
            {'type': 'ineq', 'fun': lambda x: np.prod(x) - 25},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 11},
        ],
        options={'ftol': 1e-12},
    )
    assert found.success
    return found.x


def _drive(tmp_path: Path, method: str, oracle, *, evaluations: int = 2000, record: str = 'r.jsonl', **settings):
    solver = GradientSolver(PROBLEM, oracle.capability, method, start=START, **settings)
    outcome = run(PROBLEM, oracle, solver, Record(tmp_path / record), Budget(evaluations=evaluations))
    return solver, outcome


@pytest.mark.parametrize('method', ['sqp', 'interior_point', 'augmented_lagrangian'])
def test_p9_a_constrained_method_run_by_the_loop_agrees_with_scipy(tmp_path: Path, method: str) -> None:
    solver, outcome = _drive(tmp_path, method, _oracle())
    assert outcome.stop.reason is Termination.SOLVER_DONE
    assert solver.solution.halt is Halt.CONVERGED, solver.solution.reason
    found = np.array([solver.design()[n] for n in NAMES])
    assert agrees(found, _reference())


def test_p9_finite_differences_stand_in_for_a_values_only_oracle(tmp_path: Path) -> None:
    values_only = from_function(
        lambda action: _values(np.array([action.design[n] for n in NAMES])), Capability(outputs=('f', 'product'))
    )
    with pytest.raises(Refusal, match='finite_differences'):
        GradientSolver(PROBLEM, values_only.capability, 'sqp', start=START)
    differenced = with_finite_differences(values_only, PROBLEM)
    solver, _ = _drive(tmp_path, 'sqp', differenced)
    found = np.array([solver.design()[n] for n in NAMES])
    assert agrees(found, _reference(), 1e-4)


def test_p9_a_retryable_result_is_asked_again_and_the_answer_does_not_move(tmp_path: Path) -> None:
    steady, _ = _drive(tmp_path, 'sqp', _oracle(), record='steady.jsonl')
    flaky, _ = _drive(tmp_path, 'sqp', _oracle(flaky=True), record='flaky.jsonl')
    assert flaky.solution.halt is Halt.CONVERGED
    assert np.array_equal(flaky.solution.x, steady.solution.x)


def test_p9_an_interrupted_run_resumes_by_replay_to_the_same_solution(tmp_path: Path) -> None:
    whole, whole_run = _drive(tmp_path, 'sqp', _oracle(), record='whole.jsonl')
    _, cut = _drive(tmp_path, 'sqp', _oracle(), record='cut.jsonl', evaluations=5)
    assert cut.stop.reason is not Termination.SOLVER_DONE
    resumed, resumed_run = _drive(tmp_path, 'sqp', _oracle(), record='cut.jsonl')
    assert np.array_equal(resumed.solution.x, whole.solution.x)
    assert resumed_run.ledger.evaluations == whole_run.ledger.evaluations


def test_p9_a_box_method_reads_sense_normalisation_and_a_log_scale(tmp_path: Path) -> None:
    space = DesignSpace((Real('x', 1.0, 3.0), Real('y', 0.1, 1.0, scale=Scale.LOG)))
    problem = Problem(space, (Objective('g', sense='maximize', normalization=Normalization(reference=1.0, scale=2.0)),))

    def evaluate(action: Action) -> Result:
        x, y = action.design['x'], action.design['y']
        value = -((x - 2.0) ** 2) - (y - 0.3) ** 2
        gradient = np.array([-2 * (x - 2.0) * 2.0, -2 * (y - 0.3) * y * np.log(10.0)])
        return Result(action, Cell(Status.OK, values={'g': value}), gradient={'g': gradient})

    oracle = from_function(evaluate, Capability(outputs=('g',), provides=SLOPES))
    solver = GradientSolver(problem, oracle.capability, 'lbfgsb', start={'x': 1.2, 'y': 0.9})
    run(problem, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=500))
    assert solver.solution.halt is Halt.CONVERGED
    assert agrees(np.array(list(solver.design().values())), np.array([2.0, 0.3]))


def test_p9_a_discrete_variable_is_refused() -> None:
    problem = Problem(DesignSpace((Real('x', 0.0, 1.0), Integer('k', 0, 3))), (Objective('f'),))
    with pytest.raises(Refusal, match='Real or Field'):
        GradientSolver(problem, Capability(outputs=('f',), provides=SLOPES), 'lbfgsb')


def test_p9_planted_control_a_wrong_declared_gradient_does_not_agree(tmp_path: Path) -> None:
    solver, _ = _drive(tmp_path, 'sqp', _oracle(planted=-1.0), evaluations=400)
    found = np.array([solver.design()[n] for n in NAMES]) if solver.done else None
    assert found is None or not (solver.solution.halt is Halt.CONVERGED and agrees(found, _reference()))
