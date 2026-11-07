"""Population solvers estimate a declared risk over the problem's Uncertainty and select on it.

The benchmark is one variable against one uniform parameter ``xi ~ U(-0.5, 0.5)``:
``f(x, xi) = (x - 0.3)^2 + xi (1 - x)``. Its expectation is minimised at ``x = 0.3``, its worst
case (``xi = 0.5``) at ``x = 0.55``, its CVaR at 90 % (the tail mean of xi is 0.45) at
``x = 0.525`` -- three measures, three different optima, so a solver that ignores the measure
lands on the wrong one. A chance constraint ``P(x + xi <= 1) >= 0.9`` caps ``x`` at 0.6.
"""

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Record

from optimi_lab.core import SampleSpec
from optimi_lab.operator import Risk
from optimi_lab.oracle import Action, Capability, from_function
from optimi_lab.problem import Chance, Objective, Problem, Real, Space, Uncertainty, Uniform
from optimi_lab.run import Budget, run
from optimi_lab.solver import Requires, cmaes, ga, match, nsga2
from optimi_lab.solver.requires import Eligible

SPACE = Space((Real('x', 0.0, 1.0),))
XI = Uncertainty((Uniform('xi', -0.5, 0.5),))
ROBUST = (
    'nsga2',
    'nsga3',
    'unsga3',
    'rnsga3',
    'spea2',
    'sms_emoa',
    'rvea',
    'mode',
    'mopso',
    'moead',
    'ga',
    'cmaes',
    'ipop_cmaes',
    'map_elites',
)


def _f(action: Action) -> dict[str, float]:
    x, xi = action.design['x'], action.scenario['xi']
    return {'f': (x - 0.3) ** 2 + xi * (1 - x), 'g': x + xi, 'neg_x': -x}


ORACLE = from_function(_f, Capability(outputs=('f', 'g', 'neg_x')))


def _oracle(problem: Problem):
    outputs = problem.outputs
    return from_function(lambda a: {k: v for k, v in _f(a).items() if k in outputs}, Capability(outputs=outputs))


def _solve(problem: Problem, factory, path: Path, **settings):
    oracle = _oracle(problem)
    first = {} if factory is cmaes else {'spec': SampleSpec(n_samples=10, seed=1)}  # CMA-ES draws its own
    solver = factory(problem, oracle.capability, seed=1, **first, **settings)
    run(problem, oracle, solver, Record(path), Budget(evaluations=10 * settings['scenarios'] * 20))
    following = solver.ask(10**6)  # the next generation, converged
    return following, float(np.median([a.design['x'] for a in following]))


@pytest.mark.parametrize(
    ('risk', 'optimum'),
    [(Risk('mean'), 0.3), (Risk('worst_case'), 0.55), (Risk('cvar', alpha=0.9), 0.525)],
)
@pytest.mark.parametrize('factory', [ga, cmaes])
def test_the_population_settles_on_the_optimum_of_the_declared_measure(factory, risk, optimum, tmp_path):
    problem = Problem(SPACE, (Objective('f'),), uncertainty=XI)
    assert _solve(problem, factory, tmp_path / 'r.jsonl', risk=risk, scenarios=16)[1] == pytest.approx(
        optimum, abs=0.04
    )


def test_a_chance_constraint_is_estimated_over_the_scenarios(tmp_path):
    problem = Problem(
        SPACE, (Objective('neg_x'),), constraints=(Chance('cap', 'g', '<=', 1.0, probability=0.9),), uncertainty=XI
    )
    following, best = _solve(problem, ga, tmp_path / 'c.jsonl', scenarios=20)
    xi = np.sort([a.scenario['xi'] for a in following[:20]])  # the shared draw, read off one design's actions
    # Of 20 scenarios at least 18 must hold: the sample-average optimum sits under the 18th smallest draw.
    assert best == pytest.approx(1.0 - xi[17], abs=0.02)
    assert best == pytest.approx(0.6, abs=0.1)


def test_every_scenario_of_one_design_is_one_action_carrying_its_draw():
    problem = Problem(SPACE, (Objective('f'),), uncertainty=XI)
    solver = nsga2(problem, ORACLE.capability, spec=SampleSpec(n_samples=4, seed=0), scenarios=8)
    actions = solver.ask(100)
    assert len(actions) == 4 * 8
    assert len({a.identity.key for a in actions}) == 32
    draws = sorted(a.scenario['xi'] for a in actions[:8])
    assert draws == sorted(a.scenario['xi'] for a in actions[8:16])  # common random numbers


def test_a_scenario_on_a_problem_without_uncertainty_is_refused():
    with pytest.raises(ValueError, match='uncertainty'):
        Action.of(Problem(SPACE, (Objective('f'),)), ORACLE.capability, {'x': 0.5}, scenario={'xi': 0.0})


def test_risk_settings_without_declared_uncertainty_are_refused():
    with pytest.raises(ValueError, match='uncertainty'):
        ga(Problem(SPACE, (Objective('f'),)), ORACLE.capability, risk=Risk('mean'))


@pytest.mark.parametrize('name', ROBUST)
def test_the_registry_declares_the_risk_estimator(name):
    problem = Problem(
        SPACE, (Objective('f'),), constraints=(Chance('cap', 'g', '<=', 1.0, probability=0.9),), uncertainty=XI
    )
    assert match(Requires.of(name), problem, ORACLE.capability) == Eligible()


def test_an_unknown_measure_is_refused_naming_the_known_ones():
    with pytest.raises(ValueError, match='cvar'):
        Risk('median')
