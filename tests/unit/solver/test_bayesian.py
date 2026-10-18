"""D61-D64 -- Bayesian optimisation, ParEGO, EHVI/NEHVI and their fantasies, on closed-form optima.

Objective gaps are in each problem's objective unit (Branin's and the bowl's are dimensionless);
positions are in the unit cube. Every run is seeded, so one run is one trajectory.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Reason, Status

from optimi_lab.core import Refusal
from optimi_lab.metric import hypervolume
from optimi_lab.model import RadialBasis
from optimi_lab.oracle import Action, Capability, Result, from_function
from optimi_lab.problem import BlackBox, Linear, Objective, Problem, Real, Space
from optimi_lab.solver import ACQUISITIONS, bo, ehvi, maximise, nehvi, parego

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from catalogue import drive

#: Branin's global minimum (three minimisers), its objective unit.
BRANIN_MIN = 0.397887
#: Objective units: the largest gap any acquisition may leave on Branin after 30 evaluations.
BRANIN_GAP = 0.1
#: Share of the true front's hypervolume a multi-objective method reaches in 30 evaluations.
HV_SHARE = 0.9
#: Unit-cube distance: the polished maximiser of a smooth acquisition against its closed-form argmax.
ARGMAX = 1e-4


def _branin() -> tuple[Problem, object]:
    problem = Problem(Space((Real('x', -5.0, 10.0), Real('y', 0.0, 15.0))), (Objective('f'),))

    def f(action):
        x, y = action.design['x'], action.design['y']
        value = (y - 5.1 / (4 * np.pi**2) * x**2 + 5 / np.pi * x - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x)
        return {'f': float(value + 10)}

    return problem, from_function(f, Capability(outputs=('f',)))


def _bowl(bound: float) -> tuple[Problem, object]:
    """``(x - 0.8)^2 + (y - 0.8)^2`` subject to ``x + y <= bound``: optimum ``2 (0.8 - bound / 2)^2``."""
    problem = Problem(
        Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))), (Objective('f'),), constraints=(BlackBox('g', '<=', bound),)
    )
    oracle = from_function(
        lambda a: {'f': (a.design['x'] - 0.8) ** 2 + (a.design['y'] - 0.8) ** 2, 'g': a.design['x'] + a.design['y']},
        Capability(outputs=('f', 'g')),
    )
    return problem, oracle


def _two_objectives() -> tuple[Problem, object]:
    problem = Problem(Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))), (Objective('f1'), Objective('f2')))
    oracle = from_function(
        lambda a: {'f1': a.design['x'], 'f2': (1 + a.design['y']) * (1 - np.sqrt(a.design['x']))},
        Capability(outputs=('f1', 'f2')),
    )
    return problem, oracle


def test_maximise_polishes_onto_the_closed_form_argmax():
    target = np.array([0.3, 0.71])

    def acquisition(points):
        return -np.sum((points - target) ** 2, axis=1)

    polished = maximise(acquisition, 2, rng=np.random.default_rng(0), candidates=64, polish=2)
    assert np.linalg.norm(polished - target) <= ARGMAX
    # the control: the Sobol candidates alone do not land on it
    coarse = maximise(acquisition, 2, rng=np.random.default_rng(0), candidates=64, polish=0)
    assert np.linalg.norm(coarse - target) > ARGMAX


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_every_acquisition_finds_branins_minimum(acquisition):
    problem, oracle = _branin()
    solver = bo(problem, oracle.capability, acquisition=acquisition, seed=1)
    drive(solver, oracle, 30)
    assert solver.best.cell.values['f'] - BRANIN_MIN <= BRANIN_GAP


@pytest.mark.parametrize(('bound', 'optimum'), [(1.0, 0.18), (0.2, 0.98)])
@pytest.mark.parametrize('acquisition', ['log_ei', 'ei', 'pi', 'log_pi'])
def test_constrained_ei_finds_the_constrained_optimum_even_from_an_infeasible_start(bound, optimum, acquisition):
    """At ``x + y <= 0.2`` most of the initial design is infeasible: the acquisition is PF until one is not."""
    problem, oracle = _bowl(bound)
    solver = bo(problem, oracle.capability, acquisition=acquisition, seed=2)
    drive(solver, oracle, 25)
    assert solver.best.cell.values['g'] <= bound
    assert solver.best.cell.values['f'] - optimum <= 5e-3


def test_a_batch_is_built_on_fantasies_and_a_repeated_pick_without_them_is_not_spread():
    problem, oracle = _branin()
    solver = bo(problem, oracle.capability, initial=6, batch=3, seed=4)
    drive(solver, oracle, 6)
    batch = np.array([problem.space.encode(a.design) for a in solver.ask(3)])
    gaps = [np.linalg.norm(batch[i] - batch[j]) for i in range(3) for j in range(i)]
    assert min(gaps) > 1e-3
    # the control: two picks with nothing pending and one generator state are one point
    solver._rng = np.random.default_rng(9)
    first = solver._pick(np.zeros((0, 2)))
    solver._rng = np.random.default_rng(9)
    assert np.array_equal(first, solver._pick(np.zeros((0, 2))))


def test_an_asynchronous_slot_is_conditioned_on_what_is_still_pending():
    problem, oracle = _branin()
    solver = bo(problem, oracle.capability, initial=6, seed=4)
    drive(solver, oracle, 6)
    first = solver.ask(1)[0]
    second = solver.ask(1)[0]
    assert np.linalg.norm(problem.space.encode(first.design) - problem.space.encode(second.design)) > 1e-3
    solver.tell([Result(second, Cell(Status.OK, values={'f': 5.0}))])
    solver.tell([Result(first, Cell(Status.TIMEOUT, reason=Reason('timeout', 'licence queue')))])
    assert not solver._pending and len(solver._x) == 7


def test_a_failed_point_leaves_the_pending_set_and_the_data():
    problem, oracle = _branin()
    solver = bo(problem, oracle.capability, initial=3, seed=0)
    asked = solver.ask(3)
    solver.tell([Result(asked[0], Cell(Status.ERROR, reason=Reason('error', 'mesh')))])
    solver.tell([Result(a, Cell(Status.OK, values={'f': 1.0 + i})) for i, a in enumerate(asked[1:])])
    assert not solver._pending and len(solver._x) == 2
    stranger = Action.of(problem, oracle.capability, {'x': 0.0, 'y': 0.0})
    with pytest.raises(Refusal, match='did not ask'):
        solver.tell([Result(stranger, Cell(Status.OK, values={'f': 1.0}))])


@pytest.mark.parametrize('factory', [parego, ehvi, nehvi])
def test_the_multi_objective_methods_reach_the_known_front(factory):
    """The front is ``f2 = 1 - sqrt(f1)`` at ``y = 0``; its hypervolume to (1.1, 1.1) from a 2001-point sweep.

    Measured 2026-10-18 at 30 evaluations, seed 1: ParEGO 0.933, EHVI 0.972, NEHVI 0.972 of it.
    """
    problem, oracle = _two_objectives()
    solver = factory(problem, oracle.capability, seed=1)
    drive(solver, oracle, 30)
    front = np.array([[r.cell.values['f1'], r.cell.values['f2']] for r in solver.front])
    f1 = np.linspace(0.0, 1.0, 2001)
    exact = hypervolume(np.column_stack([f1, 1 - np.sqrt(f1)]), np.array([1.1, 1.1]))
    assert hypervolume(front, np.array([1.1, 1.1])) >= HV_SHARE * exact


def test_ehvi_batches_on_believed_fronts():
    problem, oracle = _two_objectives()
    solver = ehvi(problem, oracle.capability, initial=6, batch=2, seed=3)
    drive(solver, oracle, 6)
    batch = np.array([problem.space.encode(a.design) for a in solver.ask(2)])
    assert np.linalg.norm(batch[0] - batch[1]) > 1e-3


def test_refusals_name_what_is_missing():
    problem, oracle = _bowl(1.0)
    with pytest.raises(Refusal, match='takes no constraint'):
        bo(problem, oracle.capability, acquisition='kg')
    with pytest.raises(Refusal, match='acquisition is one of'):
        bo(problem, oracle.capability, acquisition='thompson')
    with pytest.raises(Refusal, match='GaussianProcess'):
        bo(problem, oracle.capability, surrogate=RadialBasis())
    linear = Problem(problem.space, problem.objectives, constraints=(Linear('l', {'x': 1.0}, hi=0.5),))
    with pytest.raises(Refusal, match='penalise'):
        bo(linear, oracle.capability)
    with pytest.raises(Refusal, match='bo\\(\\)'):
        parego(problem, oracle.capability)
    two, two_oracle = _two_objectives()
    with pytest.raises(Refusal, match='scalarise'):
        bo(two, two_oracle.capability)
