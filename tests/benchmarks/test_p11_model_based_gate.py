"""P11 gate -- the model-based solvers at an EXPENSIVE budget, against closed-form optima; and B0-08.

THE ORACLE. The plan names BoTorch at equal budget; BoTorch is not installed (nor in the dev extra),
so the reference here is what BoTorch itself would be judged by: the closed-form optimum (Branin's
0.397887) and the closed-form ZDT1 front (``catalogue.py``), at a budget a population method cannot
use -- 40 evaluations -- with a uniform random search at the SAME budget as the control that the
bar separates an algorithm from none. Bars are in the problems' own (dimensionless) units; the
measured values (2026-10-18, seed 1) are quoted beside each.
"""

import numpy as np
import pytest
from catalogue import CATALOGUE, RandomSearch, drive, front_of
from lab_commons.outcomes import Evidence

from optimi_lab.metric import hypervolume
from optimi_lab.oracle import Capability, from_function
from optimi_lab.problem import Objective, Problem, Real, Space
from optimi_lab.solver import bo, ehvi, nehvi, parego

BUDGET = 40
#: Share of the true ZDT1 hypervolume to (1.1, 6) reached at BUDGET; measured parego 0.973, ehvi 0.984,
#: nehvi 0.981, random search 0.535.
HV_SHARE = 0.95
ZDT1_REFERENCE = np.array([1.1, 6.0])
#: Objective units above Branin's minimum 0.397887 at BUDGET; measured log-EI 0.004 at 30 evaluations.
BRANIN_GAP = 0.05


def _branin() -> tuple[Problem, object]:
    problem = Problem(Space((Real('x', -5.0, 10.0), Real('y', 0.0, 15.0))), (Objective('f'),))

    def f(action):
        x, y = action.design['x'], action.design['y']
        value = (y - 5.1 / (4 * np.pi**2) * x**2 + 5 / np.pi * x - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x)
        return {'f': float(value + 10)}

    return problem, from_function(f, Capability(outputs=('f',)))


def _share(results, b) -> float:
    return hypervolume(front_of(b.problem, results), ZDT1_REFERENCE) / hypervolume(b.front(2000), ZDT1_REFERENCE)


@pytest.mark.parametrize('factory', [parego, ehvi, nehvi], ids=['parego', 'ehvi', 'nehvi'])
def test_a_multi_objective_model_reaches_the_zdt1_front_at_an_expensive_budget(factory):
    b = CATALOGUE['zdt1']
    settings = {} if factory is parego else {'reference': ZDT1_REFERENCE}
    assert _share(drive(factory(b.problem, b.oracle.capability, seed=1, **settings), b.oracle, BUDGET), b) >= HV_SHARE


def test_bo_reaches_branins_minimum_at_an_expensive_budget():
    problem, oracle = _branin()
    solver = bo(problem, oracle.capability, seed=1)
    drive(solver, oracle, BUDGET)
    assert solver.best.cell.values['f'] - 0.397887 <= BRANIN_GAP


def test_random_search_at_the_same_budget_misses_both_bars():
    """The control: the bars separate a model-based method from no method at all."""
    b = CATALOGUE['zdt1']
    assert _share(drive(RandomSearch(b.problem, b.oracle.capability, seed=1), b.oracle, BUDGET), b) < HV_SHARE
    problem, oracle = _branin()
    best = min(r.cell.values['f'] for r in drive(RandomSearch(problem, oracle.capability, seed=1), oracle, BUDGET))
    assert best - 0.397887 > BRANIN_GAP


def test_b0_08_a_model_based_solver_reports_only_what_was_evaluated():
    """The surrogate's optimum steers an ask; the reported best and front are told observations."""
    problem, oracle = _branin()
    solver = bo(problem, oracle.capability, seed=1)
    evaluated = {r.action.identity.key for r in drive(solver, oracle, 15)}
    assert solver.best.action.identity.key in evaluated and solver.best.cell.evidence is Evidence.OBSERVED
    b = CATALOGUE['zdt1']
    mobo = ehvi(b.problem, b.oracle.capability, seed=1, reference=ZDT1_REFERENCE)
    told = {r.action.identity.key for r in drive(mobo, b.oracle, 15)}
    assert {r.action.identity.key for r in mobo.front} <= told
    assert all(r.cell.evidence is Evidence.OBSERVED for r in mobo.front)
