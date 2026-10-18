"""D29 -- the catalogue's fronts are true fronts, and every benchmark runs through the P8 loop."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from catalogue import CATALOGUE, RandomSearch
from lab_commons.outcomes import Record

from optimi_lab.core.pareto import non_dominated_sorting
from optimi_lab.metric import igd
from optimi_lab.oracle import Action
from optimi_lab.run import Budget, Termination, run


@pytest.mark.parametrize('name', sorted(CATALOGUE))
def test_d29_the_true_front_is_mutually_non_dominated(name: str) -> None:
    front = CATALOGUE[name].front(60)
    assert len(non_dominated_sorting(front)) == len(front)


@pytest.mark.parametrize('name', ['zdt1', 'zdt2'])
def test_d29_the_oracle_reaches_its_closed_form_front_at_g_equal_one(name: str) -> None:
    bench = CATALOGUE[name]
    design = {v.name: 0.0 for v in bench.problem.space} | {'x0': 0.25}
    result = asyncio.run(bench.oracle(Action.of(bench.problem, bench.oracle.capability, design)))
    expected = bench.front(5)[1]  # f1 = 0.25
    assert [result.cell.values['f1'], result.cell.values['f2']] == pytest.approx(expected.tolist())


def test_d29_planted_control_igd_of_a_front_against_itself_is_zero_and_a_shifted_one_is_not() -> None:
    reference = CATALOGUE['zdt1'].front(50)
    assert igd(reference, reference) == 0.0
    shifted = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert igd(shifted, np.array([[0.0, 1.0], [1.0, 1.0]])) == 0.5  # nearest distances 1 and 0, by hand


@pytest.mark.parametrize('name', sorted(CATALOGUE))
def test_d29_every_benchmark_runs_through_the_loop_and_its_front_is_feasible(name: str, tmp_path: Path) -> None:
    bench = CATALOGUE[name]
    solver = RandomSearch(bench.problem, bench.oracle.capability, seed=1)
    outcome = run(bench.problem, bench.oracle, solver, Record(tmp_path / f'{name}.jsonl'), Budget(evaluations=60))
    assert outcome.stop.reason is Termination.EVALUATIONS
    front = outcome.archive.front()
    assert front and all(outcome.archive.violation(r) == 0.0 for r in front)
    values = np.array([[o.minimized(r.cell.values[o.name]) for o in bench.problem.objectives] for r in front])
    assert np.isfinite(igd(values, bench.front(40)))
