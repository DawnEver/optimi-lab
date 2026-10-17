"""D29 -- the B0 counterexamples the plan names for the base: each states the wrong belief it refutes.

P8's gate is B0-06, B0-12 and B0-18: they pass here. B0-08, B0-11, B0-16 and B0-20 are answered by
the base as well. B0-02, B0-03, B0-04, B0-05 and B0-14 need machinery from later phases and are
``xfail(strict=True)`` naming the item that answers them -- when it lands, the test turns red and is
promoted, never skipped.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from catalogue import RandomSearch
from lab_commons.outcomes import Cell, Cost, Evidence, Reason, Record, Status

from optimi_lab.archive import Archive
from optimi_lab.oracle import Action, Capability, Result, from_function
from optimi_lab.problem import Categorical, Fidelity, Level, Linear, Objective, Problem, Real, Space
from optimi_lab.problem import Path as PathConstraint
from optimi_lab.run import Budget, run
from optimi_lab.solver import REGISTRY, MultiObjective

SPACE = Space((Real('x', 0.0, 1.0),))
CAPABILITY = Capability(outputs=('f', 'g'), version='maxwell 24.1')
PROBLEM = Problem(SPACE, (Objective('f'), Objective('g')))


def test_b0_06_a_resource_timeout_then_a_success_is_never_infeasible(tmp_path: Path) -> None:
    attempts: list[float] = []

    def flaky(action: Action) -> Result:
        attempts.append(action.design['x'])
        if attempts.count(action.design['x']) == 1:
            return Result(
                action, Cell(Status.TIMEOUT, reason=Reason('timeout', 'licence queue'), cost=Cost(wall_s=1.0))
            )
        return Result(action, Cell(Status.OK, values={'f': 1.0, 'g': 2.0}, cost=Cost(wall_s=1.0)))

    archive = Archive(PROBLEM, Record(tmp_path / 'r.jsonl'))
    oracle = from_function(flaky, CAPABILITY)
    action = Action.of(PROBLEM, CAPABILITY, {'x': 0.5})
    first = asyncio.run(oracle(action))
    archive.tell(first)
    assert first.cell.status.family.name == 'RETRYABLE'
    assert archive.feasible() == () and archive.complete() == ()
    archive.tell(asyncio.run(oracle(action)))
    assert [r.cell.status for r in archive.complete()] == [Status.OK]


def test_b0_12_the_cache_key_holds_the_whole_evaluation_identity() -> None:
    design = {'x': 0.5}
    base = Action.of(PROBLEM, CAPABILITY, design).identity
    other_solver = Action.of(PROBLEM, Capability(outputs=('f', 'g'), version='maxwell 25.0'), design).identity
    other_problem = Problem(
        SPACE, (Objective('f'), Objective('g')), constraints=(Linear('budget', {'x': 1.0}, hi=0.9),)
    )
    other_condition = Action.of(other_problem, CAPABILITY, design).identity
    meshes = Problem(
        SPACE, (Objective('f'), Objective('g')), fidelity=Fidelity((Level('coarse', 1.0), Level('fine', 9.0)))
    )
    graded = Capability(outputs=('f', 'g'), fidelities=('coarse', 'fine'), version='maxwell 24.1')
    coarse, fine = (Action.of(meshes, graded, design, fidelity=f).identity for f in ('coarse', 'fine'))
    assert len({base, other_solver, other_condition, coarse, fine}) == 5


def test_b0_18_a_seed_does_not_reproduce_a_run_the_record_does(tmp_path: Path) -> None:
    """Submission order, completion order, data and versions are all on record; replay re-derives the run."""
    oracle = from_function(lambda a: {'f': a.design['x'], 'g': 1 - a.design['x']}, CAPABILITY)
    first = run(
        PROBLEM,
        oracle,
        RandomSearch(PROBLEM, CAPABILITY, seed=3, batch=4),
        Record(tmp_path / 'r.jsonl'),
        Budget(evaluations=8),
    )
    kinds = [e.kind for e in first.archive.record.replay()]
    assert kinds.count('submitted') == kinds.count('outcome') == 8
    assert all('oracle' in r.action.identity.canonical for r in first.archive.results())
    replayed = RandomSearch(PROBLEM, CAPABILITY, seed=3, batch=4)
    counted = []
    again = run(
        PROBLEM,
        from_function(lambda a: counted.append(a) or {'f': 0.0, 'g': 0.0}, CAPABILITY),
        replayed,
        Record(tmp_path / 'r.jsonl'),
        Budget(evaluations=8),
    )
    assert counted == [] and [r.cell.values['f'] for r in replayed.told] == [
        r.cell.values['f'] for r in first.archive.results()
    ]
    assert again.ledger.evaluations == 8


def test_b0_08_a_surrogate_optimum_is_not_the_truth_until_re_evaluated(tmp_path: Path) -> None:
    archive = Archive(PROBLEM, Record(tmp_path / 'r.jsonl'))
    archive.tell(Result(Action.of(PROBLEM, CAPABILITY, {'x': 0.4}), Cell(Status.OK, values={'f': 1.0, 'g': 1.0})))
    archive.predict(
        Result(
            Action.of(PROBLEM, CAPABILITY, {'x': 1.0}),
            Cell(Status.OK, values={'f': 0.0, 'g': 0.0}, evidence=Evidence.PREDICTED),
        )
    )
    assert [r.action.design['x'] for r in archive.front()] == [0.4]
    assert [r.action.design['x'] for r in archive.front(Evidence.PREDICTED)] == [1.0]


def test_b0_11_a_category_has_no_distance_until_embedded() -> None:
    with pytest.raises(ValueError, match='embedding='):
        Space((Categorical('coolant', ('air', 'water', 'oil')),)).encode({'coolant': 'air'})


def test_b0_16_one_simulation_that_yields_two_outputs_is_one_evaluation(tmp_path: Path) -> None:
    oracle = from_function(lambda _: {'f': 1.0, 'g': 2.0}, CAPABILITY)
    outcome = run(
        PROBLEM,
        oracle,
        RandomSearch(PROBLEM, CAPABILITY, seed=0, batch=5),
        Record(tmp_path / 'r.jsonl'),
        Budget(evaluations=5),
    )
    assert outcome.ledger.evaluations == 5


def test_b0_20_every_registered_solver_declares_how_it_treats_many_objectives() -> None:
    assert REGISTRY and all(isinstance(r.objectives, MultiObjective) for r in REGISTRY.values())
    assert REGISTRY['moead'].objectives is MultiObjective.WEIGHTED
    assert REGISTRY['nsga2'].objectives is MultiObjective.NATIVE_PARETO


@pytest.mark.xfail(strict=True, reason='B0-02: needs D84 scalarisation with declared reach and D110 scans (P12)')
def test_b0_02_a_weighted_sum_cannot_reach_the_middle_of_a_concave_front() -> None:
    from optimi_lab.operator import weighted_sum  # noqa: PLC0415 -- the missing module IS the expected failure

    assert weighted_sum.reach == 'convex_hull'


@pytest.mark.xfail(strict=True, reason='B0-03: needs D117 estimators for joint chance constraints (P13)')
def test_b0_03_two_95_percent_chances_are_a_90_25_percent_joint_chance() -> None:
    from optimi_lab.operator import joint_probability  # noqa: PLC0415

    assert joint_probability((0.95, 0.95)) == pytest.approx(0.9025)


@pytest.mark.xfail(strict=True, reason='B0-04: needs D138 path quantifier with interpolation (P15)')
def test_b0_04_feasible_at_the_nodes_is_not_feasible_on_the_path() -> None:
    nodes = np.sin(2 * np.pi * np.arange(0.0, 3.0))
    constraint = PathConstraint('peak', 'x', '<=', 0.5)
    assert constraint.violation(nodes) > 0.0


@pytest.mark.xfail(strict=True, reason='B0-05: needs D142 bilevel declaration (P15)')
def test_b0_05_simultaneous_is_not_bilevel() -> None:
    from optimi_lab.problem import Bilevel  # noqa: PLC0415

    assert Bilevel


@pytest.mark.xfail(strict=True, reason='B0-14: needs D116 CVaR risk operator (P13)')
def test_b0_14_cvar_of_a_discrete_tail_is_the_tail_value() -> None:
    from optimi_lab.operator import cvar  # noqa: PLC0415

    assert cvar(np.array([0.0, 10.0]), np.array([0.9, 0.1]), alpha=0.9) == 10.0
