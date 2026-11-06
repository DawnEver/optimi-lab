"""Model-based solvers fit their model on the declared risk over shared scenarios; the rest refuse, naming why.

The benchmark is the population solvers' (``test_robust_population.py``): one variable against one
uniform parameter ``xi ~ U(-0.5, 0.5)``, ``f(x, xi) = (x - 0.3)^2 + xi (1 - x)``. Over a fixed draw
of scenarios the sample-average optimum is closed form: the objective is ``(x - 0.3)^2 + m (1 - x)``
with ``m`` the risk of the drawn ``xi`` (their mean; their maximum for the worst case, since
``1 - x >= 0``), minimised at ``x = 0.3 + m / 2``. It is computed from the very draws the solver
asked, read off its actions, so the bar is the estimator's own optimum and not the population one.

A BlackBox ``x + xi <= 1`` under the worst case holds for every draw only while ``x <= 1 - max xi``;
maximising ``x`` under it lands exactly there.

Positions are in the variable's own (dimensionless) unit.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import Risk
from optimi_lab.oracle import Action, Capability, from_function
from optimi_lab.problem import (
    BlackBox,
    Chance,
    Fidelity,
    Level,
    Objective,
    Problem,
    Real,
    Space,
    Uncertainty,
    Uniform,
)
from optimi_lab.solver import Requires, bo, ehvi, match, mfbo, parego, turbo
from optimi_lab.solver.requires import Eligible, Ineligible

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from catalogue import drive

SPACE = Space((Real('x', 0.0, 1.0),))
XI = Uncertainty((Uniform('xi', -0.5, 0.5),))
SCENARIOS = 8
#: The largest distance of the best told design from the sample-average optimum, in units of ``x``.
POSITION = 0.03
#: The model-based solvers the registry declares with a risk estimator.
ROBUST_MODELS = ('bo', 'parego', 'ehvi', 'nehvi')


def _f(action: Action) -> dict[str, float]:
    x, xi = action.design['x'], action.scenario['xi']
    return {'f': (x - 0.3) ** 2 + xi * (1 - x), 'g': x + xi, 'neg_x': -x}


def _oracle(problem: Problem, fidelities: tuple[str, ...] = ()):
    outputs = problem.outputs
    return from_function(
        lambda a: {k: v for k, v in _f(a).items() if k in outputs}, Capability(outputs=outputs, fidelities=fidelities)
    )


def _draws(results) -> np.ndarray:
    """The shared scenario draws, read off the actions of one design."""
    first = results[0].action.design
    return np.array([r.action.scenario['xi'] for r in results if r.action.design == first])


@pytest.mark.parametrize(
    ('risk', 'aggregate'), [(Risk('mean'), np.mean), (Risk('worst_case'), np.max)], ids=['mean', 'worst_case']
)
def test_bo_settles_on_the_sample_optimum_of_the_declared_measure(risk, aggregate):
    problem = Problem(SPACE, (Objective('f'),), uncertainty=XI)
    oracle = _oracle(problem)
    solver = bo(problem, oracle.capability, seed=1, risk=risk, scenarios=SCENARIOS)
    results = drive(solver, oracle, SCENARIOS * 16)
    optimum = 0.3 + float(aggregate(_draws(results))) / 2.0
    assert solver.best is not None
    assert solver.best.action.design['x'] == pytest.approx(optimum, abs=POSITION)


def test_a_blackbox_constraint_holds_under_the_declared_measure_of_its_margin():
    problem = Problem(SPACE, (Objective('neg_x'),), constraints=(BlackBox('g', '<=', 1.0),), uncertainty=XI)
    oracle = _oracle(problem)
    solver = bo(problem, oracle.capability, seed=1, risk=Risk('worst_case'), scenarios=SCENARIOS)
    results = drive(solver, oracle, SCENARIOS * 16)
    cap = 1.0 - float(np.max(_draws(results)))
    assert solver.best is not None
    assert solver.best.action.design['x'] <= cap + 1e-12  # observed feasible in every scenario
    assert solver.best.action.design['x'] == pytest.approx(cap, abs=POSITION)


def test_every_pick_is_asked_once_per_scenario_with_common_draws():
    problem = Problem(SPACE, (Objective('f'),), uncertainty=XI)
    solver = bo(problem, _oracle(problem).capability, initial=4, scenarios=SCENARIOS)
    actions = solver.ask(100)
    assert len(actions) == 4 * SCENARIOS
    assert len({a.identity.key for a in actions}) == 4 * SCENARIOS
    draws = [sorted(a.scenario['xi'] for a in actions[i : i + SCENARIOS]) for i in range(0, 32, SCENARIOS)]
    assert all(d == draws[0] for d in draws)


def test_a_capacity_below_one_design_asks_nothing_rather_than_half_a_design():
    problem = Problem(SPACE, (Objective('f'),), uncertainty=XI)
    solver = bo(problem, _oracle(problem).capability, scenarios=SCENARIOS)
    assert list(solver.ask(SCENARIOS - 1)) == []


@pytest.mark.parametrize('factory', [parego, ehvi])
def test_a_pareto_model_runs_on_risk_rows(factory):
    problem = Problem(SPACE, (Objective('f'), Objective('neg_x')), uncertainty=XI)
    oracle = _oracle(problem)
    solver = factory(problem, oracle.capability, seed=1, risk=Risk('cvar', alpha=0.9), scenarios=SCENARIOS)
    drive(solver, oracle, SCENARIOS * 10)
    assert solver.front


@pytest.mark.parametrize('name', ROBUST_MODELS)
def test_the_registry_declares_the_model_risk_estimator(name):
    objectives = (Objective('f'),) if name == 'bo' else (Objective('f'), Objective('neg_x'))
    problem = Problem(SPACE, objectives, constraints=(BlackBox('g', '<=', 1.0),), uncertainty=XI)
    assert match(Requires.of(name), problem, _oracle(problem).capability) == Eligible()


def test_a_chance_constraint_is_refused_by_a_model_naming_why():
    problem = Problem(
        SPACE, (Objective('f'),), constraints=(Chance('cap', 'g', '<=', 1.0, probability=0.9),), uncertainty=XI
    )
    with pytest.raises(Refusal, match='estimates no probability'):
        bo(problem, _oracle(problem).capability, scenarios=SCENARIOS)
    assert isinstance(match(Requires.of('bo'), problem, _oracle(problem).capability), Ineligible)


def test_turbo_refuses_uncertainty_naming_why():
    problem = Problem(SPACE, (Objective('f'),), uncertainty=XI)
    with pytest.raises(Refusal, match='turbo has no risk estimator'):
        turbo(problem, _oracle(problem).capability)
    assert isinstance(match(Requires.of('turbo'), problem, _oracle(problem).capability), Ineligible)


def test_mfbo_refuses_uncertainty_naming_why():
    fidelity = Fidelity((Level('coarse', 1.0), Level('fine', 10.0)))
    problem = Problem(SPACE, (Objective('f'),), uncertainty=XI, fidelity=fidelity)
    with pytest.raises(Refusal, match='mfbo has no risk estimator'):
        mfbo(problem, _oracle(problem, ('coarse', 'fine')).capability)
    assert isinstance(match(Requires.of('mfbo'), problem, _oracle(problem).capability), Ineligible)


def test_risk_settings_without_declared_uncertainty_are_refused():
    problem = Problem(SPACE, (Objective('f'),))
    with pytest.raises(Refusal, match='uncertainty'):
        bo(problem, _oracle(problem).capability, risk=Risk('mean'))
