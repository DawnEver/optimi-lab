"""Every registered solver an oracle drives is built BY NAME; the algebraic ones are refused naming why."""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import Risk
from optimi_lab.oracle import Capability, Information, analytic
from optimi_lab.problem import (
    BlackBox,
    Fidelity,
    Form,
    Level,
    Linear,
    Objective,
    Problem,
    Real,
    Sense,
    Space,
    Structure,
)
from optimi_lab.problem.declarations import Uncertainty
from optimi_lab.solver import SEEDING, SOLVERS, UNDRIVEN, Seeding, build_solver

SPACE = Space((Real('x', 0.0, 4.0), Real('y', -1.0, 3.0)))
ONE = Problem(SPACE, (Objective('f'),))
TWO = Problem(SPACE, (Objective('f'), Objective('g', Sense.MAXIMIZE)))
LEVELS = Problem(SPACE, (Objective('f'),), fidelity=Fidelity((Level('coarse', 1.0), Level('fine', 10.0))))
LP = Problem(
    SPACE,
    (Objective('profit', Sense.MAXIMIZE),),
    constraints=(BlackBox('load', '<=', 4.0), Linear('gap', {'x': 1.0, 'y': -1.0}, lo=-2.0)),
    structure=Structure({'profit': frozenset({Form.LINEAR}), 'load': frozenset({Form.LINEAR})}),
)
LP_ORACLE = analytic(SPACE, linear={'profit': [3.0, 2.0], 'load': [1.0, 1.0]}, constant={'profit': 1.0})
_CONVEX = frozenset({Form.QUADRATIC, Form.CONVEX})
QP = Problem(
    SPACE,
    (Objective('f'),),
    constraints=(BlackBox('load', '>=', 3.0),),
    structure=Structure({'f': _CONVEX, 'load': _CONVEX}),
)
QP_ORACLE = analytic(
    SPACE, linear={'f': [-1.0, -1.0], 'load': [1.0, 1.0]}, quadratic={'f': np.array([[2.0, 0.5], [0.5, 1.0]])}
)
VALUES = Capability(('f',), version='values')
PAIR = Capability(('f', 'g'), version='values')
STAGED = Capability(('f',), fidelities=('coarse', 'fine'), version='staged')
SLOPES = Capability(('f',), version='slopes', provides=frozenset({Information.VALUES, Information.VJP}))
PAIR_SLOPES = Capability(('f', 'g'), version='slopes', provides=frozenset({Information.VALUES, Information.VJP}))

#: The settings a solver cannot run without, as a case would declare them.
SETTINGS = {
    'map_elites': {'descriptors': {'x': [4]}},
    'multistart': {'local': 'nelder_mead'},
    'rnsga3': {'aspiration': [[0.5, 0.5]]},
    'scalarisation_scan': {'local': 'nelder_mead', 'weights': [[0.5, 0.5]]},
    'newton_cg': {'hvp': 'finite_difference'},
}


def _declared(name: str) -> tuple[Problem, Capability]:
    """A problem and oracle build the solver *name* is eligible on, read off its registered needs."""
    requires = SOLVERS[name]
    if requires.forms:
        return (QP, QP_ORACLE.capability) if Form.QUADRATIC in requires.forms else (LP, LP_ORACLE.capability)
    if requires.fidelity:
        return LEVELS, STAGED
    pareto = requires.objectives.value != 'single' and name not in ('moead', 'scalarisation_scan')
    several = pareto or name in ('moead', 'parego', 'scalarisation_scan', 'mgda', 'pareto_continuation')
    if Information.VJP in requires.information:
        return (TWO, PAIR_SLOPES) if several else (ONE, SLOPES)
    return (TWO, PAIR) if several else (ONE, VALUES)


@pytest.mark.parametrize('name', sorted(set(SOLVERS) - set(UNDRIVEN)))
def test_a_driven_solver_is_built_by_its_registered_name(name: str) -> None:
    problem, capability = _declared(name)
    solver = build_solver(
        name, problem, capability, seed=3, population=4, initial=None, generations=2, settings=SETTINGS.get(name)
    )
    assert callable(solver.ask) and callable(solver.tell)


def test_the_gradient_solver_lbfgsb_is_built_by_name_from_one_start() -> None:
    start = np.array([[0.25, 0.5]])
    solver = build_solver('lbfgsb', ONE, SLOPES, seed=0, population=1, initial=start, generations=4)
    asked = solver.ask(1)
    assert asked and asked[0].design == {'x': pytest.approx(1.0), 'y': pytest.approx(1.0)}
    assert Information.VJP in asked[0].want
    with pytest.raises(Refusal, match='ONE start point'):
        build_solver('lbfgsb', ONE, SLOPES, seed=0, population=2, initial=np.full((2, 2), 0.5), generations=4)
    with pytest.raises(Refusal, match='only after'):
        build_solver('lbfgsb', ONE, VALUES, seed=0, population=1, initial=None, generations=4)


def test_the_algebraic_solvers_no_oracle_drives_are_refused_naming_why() -> None:
    assert set(UNDRIVEN) == {'admm', 'benders', 'cp', 'dynamic_programming', 'milp'}
    assert set(SOLVERS) - set(SEEDING) - set(UNDRIVEN) == {'lp', 'qp'}
    for name in UNDRIVEN:
        with pytest.raises(Refusal, match=f'{name} is not driven through an oracle'):
            build_solver(name, ONE, VALUES, seed=0, population=4, initial=None, generations=1)


def test_a_solver_drawing_its_own_first_batch_refuses_rows_it_would_ignore() -> None:
    rows = np.full((4, 2), 0.5)
    assert SEEDING['bo'] is Seeding.MODEL
    with pytest.raises(Refusal, match='draws its own first batch'):
        build_solver('bo', ONE, VALUES, seed=0, population=4, initial=rows, generations=1)
    assert build_solver('nsga2', ONE, VALUES, seed=0, population=4, initial=rows, generations=1).ask(8)


def test_settings_restating_a_builder_declaration_and_missing_required_ones_are_refused() -> None:
    with pytest.raises(Refusal, match='builder-owned'):
        build_solver('nsga2', ONE, VALUES, seed=0, population=4, initial=None, generations=1, settings={'seed': 1})
    with pytest.raises(Refusal, match=r'settings\.descriptors'):
        build_solver('map_elites', ONE, VALUES, seed=0, population=4, initial=None, generations=1)


def test_a_robust_declaration_reaches_the_solver_as_declared() -> None:
    robust = Problem(SPACE, (Objective('f'),), uncertainty=Uncertainty.declared([('t', 'normal', 0.0, 1.0, None)]))
    solver = build_solver(
        'ga',
        robust,
        VALUES,
        seed=0,
        population=4,
        initial=None,
        generations=1,
        risk=Risk('cvar', alpha=0.9),
        scenarios=3,
    )
    asked = solver.ask(100)
    assert len(asked) == 4 * 3 and {tuple(a.scenario) for a in asked} == {('t',)}
