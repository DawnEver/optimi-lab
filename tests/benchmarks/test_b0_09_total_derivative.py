"""B0-09 -- the wrong belief: "the derivative of an output is its partial derivative in the design".

An output that depends on the design THROUGH a solved state ``u`` (``K(x) u = b``, the shape of every
FEA/MEC/SDM output) has the TOTAL derivative ``df/dx = f_x - lambda^T R_x`` with ``K^T lambda = f_u``
(the adjoint, plan II.7). The partial ``f_x`` alone is what a hand-written gradient most often
returns, and it is wrong by exactly the state's contribution.

The check is :func:`optimi_lab.oracle.check_gradient`: a declared gradient against finite
differences of the oracle's own values. The adjoint passes; the partial is NAMED -- that is both the
B0-09 counterexample and the check's planted control. Then a gradient method run on the adjoint
gradient agrees with scipy.optimize (reference oracle, tests only).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from lab_commons.outcomes import Cell, Record, Status
from scipy.optimize import minimize

from optimi_lab.oracle import Action, Capability, Information, Result, check_gradient, from_function
from optimi_lab.problem import Objective, Problem, Real, Space
from optimi_lab.run import Budget, run
from optimi_lab.solver import GradientSolver, Halt

LO, HI = 0.5, 2.0
SPACE = Space(tuple(Real(f'k{i}', LO, HI) for i in range(3)))
PROBLEM = Problem(SPACE, (Objective('f'),))
#: The state system: a chain whose i-th diagonal entry is stiffened by x_i.
BASE = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
LOAD = np.array([1.0, 0.5, 2.0])
WEIGHT = np.array([1.0, -2.0, 1.5])


def _solve(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = BASE + np.diag(x)
    return k, np.linalg.solve(k, LOAD)


def _value(x: np.ndarray) -> float:
    _, u = _solve(x)
    return float(WEIGHT @ u + 0.5 * x @ x)


def _total(x: np.ndarray) -> np.ndarray:
    k, u = _solve(x)
    adjoint = np.linalg.solve(k.T, WEIGHT)
    return x - adjoint * u  # f_x - lambda^T (dK/dx_i) u, dK/dx_i = e_i e_i^T


def _partial(x: np.ndarray) -> np.ndarray:
    return x


def _oracle(derivative) -> object:
    def evaluate(action: Action) -> Result:
        x = np.array([action.design[f'k{i}'] for i in range(3)])
        values = {'f': _value(x)}
        if Information.VJP not in action.want:
            return Result(action, Cell(Status.OK, values=values))
        return Result(action, Cell(Status.OK, values=values), gradient={'f': (HI - LO) * derivative(x)})

    return from_function(
        evaluate, Capability(outputs=('f',), provides=frozenset({Information.VALUES, Information.VJP}))
    )


DESIGN = {'k0': 0.8, 'k1': 1.4, 'k2': 1.9}


def test_b0_09_the_adjoint_total_derivative_agrees_with_finite_differences() -> None:
    assert asyncio.run(check_gradient(_oracle(_total), PROBLEM, DESIGN))['f'] <= 1e-6


def test_b0_09_the_partial_derivative_alone_is_named() -> None:
    assert asyncio.run(check_gradient(_oracle(_partial), PROBLEM, DESIGN))['f'] > 1e-2


def test_b0_09_a_gradient_method_on_the_total_derivative_agrees_with_scipy(tmp_path: Path) -> None:
    oracle = _oracle(_total)
    solver = GradientSolver(PROBLEM, oracle.capability, 'lbfgsb', start=DESIGN)
    run(PROBLEM, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=500))
    assert solver.solution.halt is Halt.CONVERGED
    reference = minimize(
        _value,
        np.array(list(DESIGN.values())),
        jac=_total,
        method='L-BFGS-B',
        bounds=[(LO, HI)] * 3,
        options={'ftol': 1e-15, 'gtol': 1e-12},
    )
    found = np.array([solver.design()[f'k{i}'] for i in range(3)])
    assert np.max(np.abs(found - reference.x)) <= 1e-5
