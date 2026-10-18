"""D128 mixed-integer linear programs: branch and bound with Gomory mixed-integer cuts.

Closed form: ``max y s.t. -x + y <= 1, 3x + 2y <= 12, 2x + 3y <= 12, x, y >= 0`` has its LP
vertex at ``(1.8, 2.8)`` and, over the integers, the value 2 (enumerated below on ``0..6``).
Random instances are checked against ``scipy.optimize.milp`` (HiGHS), TEST-ONLY.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint
from scipy.optimize import milp as highs_milp

from optimi_lab.solver import Halt, LinearProgram, milp

#: Agreement of the optimal value with enumeration / HiGHS, relative to ``1 + |value|`` (dimensionless).
VALUE_TOLERANCE = 1e-7

SMALL = LinearProgram([0.0, -1.0], a_ub=[[-1.0, 1.0], [3.0, 2.0], [2.0, 3.0]], b_ub=[1.0, 12.0, 12.0])


def enumerated_optimum(problem: LinearProgram, top: int) -> float:
    points = (np.array(p, dtype=float) for p in itertools.product(range(top + 1), repeat=problem.width))
    return min(float(problem.c @ p) for p in points if np.all(problem.a_ub @ p <= problem.b_ub + 1e-12))


@pytest.mark.parametrize('cut_rounds', [0, 5])
def test_d128_the_small_instance_matches_enumeration_with_or_without_cuts(cut_rounds: int) -> None:
    found = milp(SMALL, [True, True], cut_rounds=cut_rounds)
    assert found.halt is Halt.CONVERGED
    assert abs(found.value - enumerated_optimum(SMALL, 6)) <= VALUE_TOLERANCE
    assert found.bound == found.value
    assert np.allclose(found.x, np.round(found.x))


def test_d128_gomory_cuts_are_added_and_shrink_the_tree() -> None:
    with_cuts, without = milp(SMALL, [True, True]), milp(SMALL, [True, True], cut_rounds=0)
    assert 'cuts:gomory_mixed_integer:0' in without.evidence
    assert 'cuts:gomory_mixed_integer:0' not in with_cuts.evidence
    assert with_cuts.iterations <= without.iterations


@pytest.mark.parametrize('seed', range(8))
def test_d128_gate_random_mixed_integer_programs_agree_with_highs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, m = 6, 4
    a = rng.integers(-3, 6, size=(m, n)).astype(float)
    integrality = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    problem = LinearProgram(-rng.uniform(1, 5, n), a_ub=a, b_ub=rng.uniform(5, 20, m), hi=np.full(n, 4.0))
    reference = highs_milp(
        problem.c,
        integrality=integrality.astype(int),
        bounds=Bounds(problem.lo, problem.hi),
        constraints=LinearConstraint(a, -np.inf, problem.b_ub),
    )
    found = milp(problem, integrality)
    assert found.halt is Halt.CONVERGED
    assert abs(found.value - reference.fun) <= VALUE_TOLERANCE * (1 + abs(reference.fun))
    assert np.max(a @ found.x - problem.b_ub) <= 1e-7


def test_d128_an_integer_infeasible_program_is_named() -> None:
    # 2x = 1 with x integer: the LP relaxation is feasible, no integer point is.
    found = milp(LinearProgram([1.0], a_eq=[[2.0]], b_eq=[1.0], hi=[5.0]), [True])
    assert found.halt is Halt.INFEASIBLE


def test_d128_a_node_limit_reports_the_open_bound_below_the_incumbent() -> None:
    rng = np.random.default_rng(5)
    a = rng.integers(1, 9, size=(1, 12)).astype(float)
    problem = LinearProgram(-rng.integers(1, 9, 12).astype(float), a_ub=a, b_ub=[float(a.sum()) / 2], hi=np.ones(12))
    found = milp(problem, np.ones(12, dtype=bool), cut_rounds=0, node_limit=10)
    assert found.halt is Halt.ITERATION_LIMIT
    assert found.bound < found.value
    assert milp(problem, np.ones(12, dtype=bool), cut_rounds=0).value == found.value  # closed: 25 nodes


def test_d128_planted_control_dropping_integrality_returns_the_fractional_lp_vertex() -> None:
    found = milp(SMALL, [False, False])
    assert abs(found.value + 2.8) <= VALUE_TOLERANCE
