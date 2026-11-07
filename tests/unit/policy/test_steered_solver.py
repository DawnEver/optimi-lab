"""A steered NSGA-II/III: the variation policy drives the SBX/PM rates, the family keeps its survival."""

from __future__ import annotations

from pathlib import Path

import pytest
from lab_commons.outcomes import Record

from optimi_lab.core import Refusal
from optimi_lab.operator import crowding_select, reference_select
from optimi_lab.oracle import Capability, from_function
from optimi_lab.policy import VariationPolicy, build_steered_solver
from optimi_lab.problem import Objective, Problem, Real, Space
from optimi_lab.run import Budget, run
from optimi_lab.solver import EvolutionarySolver, ga, nsga2, nsga3

PROBLEM = Problem(Space((Real('x', 0.0, 1.0),)), (Objective('f'), Objective('g')))
ORACLE = from_function(lambda a: {'f': a.design['x'], 'g': 1.0 - a.design['x']}, Capability(outputs=('f', 'g')))


def _policy(calls: list) -> VariationPolicy:
    def rule(results):
        calls.append(len(results))
        return {'mutation_rate': 0.5}

    return VariationPolicy(tuple, rule, generations=2, settings={})


@pytest.mark.parametrize(('factory', 'selection'), [(nsga2, crowding_select), (nsga3, reference_select)])
def test_the_policy_drives_the_variation_and_the_family_keeps_its_selection(factory, selection, tmp_path: Path) -> None:
    calls: list = []
    policy = _policy(calls)
    solver = build_steered_solver(
        factory, PROBLEM, ORACLE.capability, seed=1, population=6, initial=None, variation=policy
    )
    assert isinstance(solver, EvolutionarySolver)
    assert solver._selection is selection
    run(PROBLEM, ORACLE, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=24))
    assert len(calls) == 2 and policy.settings == {'mutation_rate': 0.5}


def test_another_family_is_refused() -> None:
    with pytest.raises(Refusal, match='nsga2 or nsga3'):
        build_steered_solver(ga, PROBLEM, ORACLE.capability, seed=1, population=6, initial=None, variation=_policy([]))
