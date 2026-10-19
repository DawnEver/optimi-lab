"""D65-D66 -- multi-fidelity BO on Forrester's pair, TuRBO on Hartmann-6, against their closed-form optima.

Objective gaps are in each function's own (dimensionless) unit; cost is in the levels' declared unit.
Measured values (2026-10-18) are quoted beside each bar.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.oracle import Capability, from_function
from optimi_lab.problem import BlackBox, Fidelity, Level, Objective, Problem, Real, Space
from optimi_lab.solver import bo, mfbo, turbo

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from catalogue import drive

#: Forrester et al. (2007): ``f(x) = (6x - 2)^2 sin(12x - 4)`` on [0, 1], minimum -6.020740 at x = 0.75725.
FORRESTER_MIN = -6.020740
#: Objective units; measured: mfbo 3.4e-5 at cost 97 (27 coarse + 7 fine), bo 0.11 at cost 120.
FORRESTER_GAP = 1e-3
COSTS = {'coarse': 1.0, 'fine': 10.0}
#: Hartmann-6's minimum (Dixon and Szego).
HARTMANN6_MIN = -3.32237
#: Objective units: the largest MEDIAN gap over seeds 1-5 at 80 evaluations, batch 4. Measured medians:
#: 0.035 on Linux (one BLAS thread), 0.129 on Windows. A single seed is one chaotic trajectory that a
#: last-bit BLAS difference redirects (seed 1: 0.243 on Windows, 0.382 on Linux), so the claim is the median.
HARTMANN6_GAP = 0.3


def _forrester(x: float, level: str | None) -> float:
    high = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    return float(high if level in ('fine', None) else 0.5 * high + 10 * (x - 0.5) - 5)


def _two_level() -> tuple[Problem, object]:
    fidelity = Fidelity(tuple(Level(name, cost) for name, cost in COSTS.items()))
    problem = Problem(Space((Real('x', 0.0, 1.0),)), (Objective('f'),), fidelity=fidelity)
    oracle = from_function(
        lambda a: {'f': _forrester(a.design['x'], a.fidelity)}, Capability(outputs=('f',), fidelities=tuple(COSTS))
    )
    return problem, oracle


def _hartmann6() -> tuple[Problem, object]:
    a = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    p = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    def f(action):
        x = np.array([action.design[f'x{i}'] for i in range(6)])
        return {'f': float(-np.sum(alpha * np.exp(-np.sum(a * (x - p) ** 2, axis=1))))}

    problem = Problem(Space(tuple(Real(f'x{i}', 0.0, 1.0) for i in range(6))), (Objective('f'),))
    return problem, from_function(f, Capability(outputs=('f',)))


def test_multi_fidelity_bo_beats_single_fidelity_bo_at_equal_cost():
    problem, oracle = _two_level()
    solver = mfbo(problem, oracle.capability, seed=1)
    results = drive(solver, oracle, 34)
    cost = sum(COSTS[r.action.fidelity] for r in results)
    assert cost <= 100.0
    assert solver.best.action.fidelity == 'fine' and solver.best.cell.values['f'] - FORRESTER_MIN <= FORRESTER_GAP
    assert {r.action.fidelity for r in results} == set(COSTS)


def test_planted_single_fidelity_bo_at_the_same_cost_misses_the_bar():
    """The control: ten truth evaluations (cost 100) of plain BO do not reach the bar."""
    plain = Problem(Space((Real('x', 0.0, 1.0),)), (Objective('f'),))
    oracle = from_function(lambda a: {'f': _forrester(a.design['x'], None)}, Capability(outputs=('f',)))
    solver = bo(plain, oracle.capability, acquisition='ei', seed=1)
    drive(solver, oracle, 10)
    assert solver.best.cell.values['f'] - FORRESTER_MIN > FORRESTER_GAP


def test_multi_fidelity_bo_refusals():
    problem, oracle = _two_level()
    with pytest.raises(Refusal, match='non-increasing'):
        mfbo(problem, oracle.capability, initial=(3, 6))
    single = Problem(problem.space, problem.objectives)
    with pytest.raises(Refusal, match='two levels'):
        mfbo(single, oracle.capability)
    constrained = Problem(
        problem.space, problem.objectives, constraints=(BlackBox('g', '<=', 1.0),), fidelity=problem.fidelity
    )
    with pytest.raises(Refusal, match='penalise'):
        mfbo(constrained, oracle.capability)


def test_turbo_reaches_hartmann6s_minimum():
    gaps = []
    for seed in range(1, 6):
        problem, oracle = _hartmann6()
        solver = turbo(problem, oracle.capability, batch=4, seed=seed)
        drive(solver, oracle, 80)
        gaps.append(solver.best.cell.values['f'] - HARTMANN6_MIN)
    assert np.median(gaps) <= HARTMANN6_GAP, gaps


def test_turbo_doubles_on_successes_halves_on_failures_and_restarts_below_the_floor():
    problem = Problem(Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))), (Objective('f'),))
    flat = from_function(lambda _: {'f': 1.0}, Capability(outputs=('f',)))
    solver = turbo(problem, flat.capability, batch=4, seed=0)
    solver._y, solver._outstanding = [5.0, 4.0], [4.0]
    for _ in range(3):
        solver._y.append(solver._y[-1] - 1.0)
        solver._outstanding = [solver._y[-1]]
        solver._adapt()
    assert solver.length == 1.6
    solver._y.append(solver._y[-1])
    solver._outstanding = [solver._y[-1]]
    solver._adapt()
    assert solver.length == 0.8
    # a flat function never succeeds: the region shrinks below its floor and the search restarts
    restarted = turbo(problem, flat.capability, batch=4, seed=0)
    drive(restarted, flat, 40)
    assert restarted.restarts >= 1


def test_turbo_refuses_constraints_and_a_foreign_surrogate():
    problem, oracle = _hartmann6()
    constrained = Problem(problem.space, problem.objectives, constraints=(BlackBox('g', '<=', 1.0),))
    with pytest.raises(Refusal, match='penalise'):
        turbo(constrained, oracle.capability)
    with pytest.raises(Refusal, match='GaussianProcess'):
        turbo(problem, oracle.capability, surrogate=object())
