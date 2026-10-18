"""D85, D95-D99 -- the single-score solvers against scipy.optimize (a TEST-ONLY oracle, R1).

Objective values are dimensionless here, so each tolerance is in those units; positions are in the
unit cube. Nelder-Mead is held to scipy's trajectory EXACTLY -- the same points in the same order --
because it implements scipy's step; the others to scipy's optimum within a stated gap.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize, rosen

from optimi_lab.core import Refusal, SampleSpec
from optimi_lab.oracle import Capability, from_function
from optimi_lab.problem import BlackBox, Integer, Objective, Problem, Real, Space
from optimi_lab.solver import CMAES, MADS, NelderMead, bobyqa, cmaes, ga, mads, map_elites, nelder_mead
from optimi_lab.solver.map_elites import Descriptor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from catalogue import drive

CENTRE = np.array([0.3, 0.7, 0.55])


def _bowl_problem(width: int = 3) -> tuple[Problem, object]:
    problem = Problem(Space(tuple(Real(f'x{i}', 0.0, 1.0) for i in range(width))), (Objective('f'),))
    oracle = from_function(
        lambda a: {'f': float(np.sum((np.array([a.design[f'x{i}'] for i in range(width)]) - CENTRE[:width]) ** 2))},
        Capability(outputs=('f',)),
    )
    return problem, oracle


def _rosen_problem(width: int = 4) -> tuple[Problem, object]:
    problem = Problem(Space(tuple(Real(f'x{i}', -2.0, 2.0) for i in range(width))), (Objective('f'),))
    oracle = from_function(
        lambda a: {'f': float(rosen(np.array([a.design[f'x{i}'] for i in range(width)])))},
        Capability(outputs=('f',)),
    )
    return problem, oracle


def test_nelder_mead_visits_exactly_the_points_scipy_visits():
    problem, oracle = _bowl_problem()
    solver = nelder_mead(problem, oracle.capability)
    asked: list[np.ndarray] = []
    original = solver.ask
    solver.ask = lambda capacity: [asked.append(problem.space.encode(a.design)) or a for a in original(capacity)]
    drive(solver, oracle, 10_000)
    seen: list[np.ndarray] = []

    def f(x):
        seen.append(np.array(x))
        return float(np.sum((x - CENTRE) ** 2))

    minimize(f, np.full(3, 0.5), method='Nelder-Mead', bounds=[(0.0, 1.0)] * 3)
    assert len(asked) == len(seen)
    assert np.allclose(np.array(asked), np.array(seen), rtol=0.0, atol=1e-15)


def test_a_planted_coefficient_change_breaks_the_trajectory_match():
    """The control: the comparison above can fail -- the adaptive coefficients walk another path."""
    problem, oracle = _bowl_problem()
    solver = nelder_mead(problem, oracle.capability, adaptive=True)
    results = drive(solver, oracle, 10_000)
    plain = drive(nelder_mead(problem, oracle.capability), oracle, 10_000)
    assert len(results) != len(plain) or not all(
        r.action.identity.key == p.action.identity.key for r, p in zip(results, plain, strict=False)
    )


@pytest.mark.parametrize(('factory', 'budget', 'gap'), [(cmaes, 6000, 1e-8), (bobyqa, 3000, 1e-6)])
def test_a_local_or_evolution_strategy_reaches_scipys_rosenbrock_optimum(factory, budget, gap):
    problem, oracle = _rosen_problem()
    reference = minimize(rosen, np.zeros(4), method='L-BFGS-B', bounds=[(-2.0, 2.0)] * 4).fun
    solver = factory(problem, oracle.capability, seed=3, **({'start': np.full(4, 0.5)} if factory is not cmaes else {}))
    drive(solver, oracle, budget)
    assert solver.best[1] <= reference + gap


def test_mads_reaches_scipys_optimum_of_a_smooth_bowl():
    """MADS on Rosenbrock reaches only 0.027 in 6000 evaluations (measured 2026-10-18, open point); on
    a bowl it must meet scipy's optimum within 1e-8.
    """
    problem, oracle = _bowl_problem()
    reference = minimize(lambda x: np.sum((x - CENTRE) ** 2), np.full(3, 0.5), method='L-BFGS-B', bounds=[(0, 1)] * 3)
    solver = mads(problem, oracle.capability, seed=3)
    drive(solver, oracle, 6000)
    assert solver.best[1] <= reference.fun + 1e-8


def test_cmaes_concludes_and_names_why():
    problem, oracle = _bowl_problem()
    solver = cmaes(problem, oracle.capability, seed=0)
    drive(solver, oracle, 50_000)
    assert solver.done and isinstance(solver, CMAES) and 'tol' in solver.stop
    assert np.allclose(solver.best[0], CENTRE, atol=1e-4)


def test_the_genetic_algorithm_approaches_the_optimum_of_one_score():
    problem, oracle = _bowl_problem()
    results = drive(ga(problem, oracle.capability, spec=SampleSpec(n_samples=30, seed=2), seed=2), oracle, 3000)
    assert min(r.cell.values['f'] for r in results) < 1e-4


def test_a_single_score_solver_refuses_two_objectives_without_a_scalarisation():
    problem = Problem(Space((Real('x', 0.0, 1.0),)), (Objective('f'), Objective('g')))
    for factory in (ga, cmaes, nelder_mead, mads, bobyqa):
        with pytest.raises(Refusal, match='scalarise'):
            factory(problem, Capability(outputs=('f', 'g')))


def test_a_local_search_refuses_a_discrete_variable():
    problem = Problem(Space((Integer('n', 1, 5),)), (Objective('f'),))
    with pytest.raises(Refusal, match='Real or Field'):
        nelder_mead(problem, Capability(outputs=('f',)))


def test_mads_progressive_barrier_reaches_the_constrained_optimum_from_an_infeasible_start():
    """Min x + y s.t. (x - 0.6)^2 + (y - 0.6)^2 <= 0.09: the optimum is 1.2 - 0.3 sqrt(2) at 45 degrees."""
    problem = Problem(
        Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))), (Objective('f'),), constraints=(BlackBox('g', '<=', 0.09),)
    )
    oracle = from_function(
        lambda a: {'f': a.design['x'] + a.design['y'], 'g': (a.design['x'] - 0.6) ** 2 + (a.design['y'] - 0.6) ** 2},
        Capability(outputs=('f', 'g')),
    )
    solver = mads(problem, oracle.capability, start=np.array([0.02, 0.02]), seed=1)
    drive(solver, oracle, 4000)
    assert isinstance(solver, MADS) and solver.best is not None
    assert solver.best[1] == pytest.approx(1.2 - 0.3 * np.sqrt(2.0), abs=1e-3)  # measured gap 4.3e-4


def test_map_elites_fills_its_grid_with_each_cells_own_optimum():
    """F = (y - 0.5)^2 with x binned in 5 cells: every cell's elite reaches y = 0.5, f below 1e-3."""
    problem = Problem(Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))), (Objective('f'),))
    oracle = from_function(lambda a: {'f': (a.design['y'] - 0.5) ** 2}, Capability(outputs=('f',)))
    solver = map_elites(
        problem, oracle.capability, descriptors=(Descriptor('x', 0.0, 1.0, 5),), spec=SampleSpec(n_samples=20, seed=1)
    )
    drive(solver, oracle, 2000)
    assert set(solver.elites) == {(k,) for k in range(5)}
    assert max(value for _, value, _ in solver.elites.values()) < 1e-3


def test_map_elites_refuses_a_descriptor_that_is_neither_output_nor_variable():
    problem, oracle = _bowl_problem()
    with pytest.raises(Refusal, match='descriptors'):
        map_elites(problem, oracle.capability, descriptors=(Descriptor('torque', 0.0, 1.0, 3),))


def test_nelder_mead_is_a_sequential_solver_that_concludes():
    problem, oracle = _bowl_problem()
    solver = nelder_mead(problem, oracle.capability)
    drive(solver, oracle, 10_000)
    assert isinstance(solver, NelderMead) and solver.done and solver.ask(5) == []
