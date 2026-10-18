"""D110-D113, D115 -- scans, multistart, restarts, multi-fidelity schedules and warm starts.

Each is checked against what it promises: a scan's reach is its scalarisation's (B0-02), a
multistart finds the global basin a single start misses, IPOP doubles its population, successive
halving evaluates ``n, n/eta, ...`` points rung by rung, and a warm start hands the archive's
front to the next population as its first generation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Record

from optimi_lab.archive import Archive
from optimi_lab.core import Refusal, SampleSpec
from optimi_lab.operator import tchebycheff, weighted_sum
from optimi_lab.oracle import Capability, from_function
from optimi_lab.problem import Fidelity, Level, Objective, Problem, Real, Space
from optimi_lab.run import Budget, run
from optimi_lab.solver import (
    ASHA,
    Chain,
    SuccessiveHalving,
    hyperband,
    ipop_cmaes,
    match,
    multistart,
    nelder_mead,
    nsga2,
    restarts,
    scalarisation_scan,
    warm_start,
)
from optimi_lab.solver.requires import SOLVERS, Ineligible

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from catalogue import drive

LINE = Space((Real('t', 0.0, 1.0),))


def _concave() -> tuple[Problem, object]:
    """f1 = t, f2 = 1 - t^2: every t is Pareto-optimal and the front is concave."""
    problem = Problem(LINE, (Objective('f1'), Objective('f2')))
    oracle = from_function(
        lambda a: {'f1': a.design['t'], 'f2': 1.0 - a.design['t'] ** 2}, Capability(outputs=('f1', 'f2'))
    )
    return problem, oracle


@pytest.mark.parametrize(('scalarisation', 'reaches_middle'), [(weighted_sum, False), (tchebycheff, True)])
def test_a_scan_reaches_what_its_scalarisation_declares(scalarisation, reaches_middle):
    """B0-02 in a run: the weighted-sum scan returns only the two ends of a concave front."""
    problem, oracle = _concave()
    weights = np.column_stack([np.linspace(0.05, 0.95, 7), 1 - np.linspace(0.05, 0.95, 7)])
    chain = scalarisation_scan(
        problem, oracle.capability, local=nelder_mead, weights=weights, scalarisation=scalarisation
    )
    drive(chain, oracle, 20_000)
    ends = [solver.best[0][0] for solver in chain.finished]
    middle = [t for t in ends if 0.05 < t < 0.95]
    assert chain.done and len(ends) == 7
    assert bool(middle) is reaches_middle


def test_multistart_finds_the_basin_one_start_misses():
    """F has a local minimum at t = 0.15 (f = 0.2) and the global at t = 0.85 (f = 0)."""
    problem = Problem(LINE, (Objective('f'),))
    oracle = from_function(
        lambda a: {'f': min(20 * (a.design['t'] - 0.15) ** 2 + 0.2, 20 * (a.design['t'] - 0.85) ** 2)},
        Capability(outputs=('f',)),
    )
    single = nelder_mead(problem, oracle.capability, start=np.array([0.1]))
    drive(single, oracle, 2000)
    chain = multistart(problem, oracle.capability, local=nelder_mead, starts=8, seed=0)
    drive(chain, oracle, 20_000)
    assert single.best[1] == pytest.approx(0.2, abs=1e-6)
    assert min(s.best[1] for s in chain.finished) == pytest.approx(0.0, abs=1e-6)


def test_ipop_doubles_the_population_at_every_restart():
    problem = Problem(Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))), (Objective('f'),))
    oracle = from_function(
        lambda a: {'f': (a.design['x'] - 0.3) ** 2 + (a.design['y'] - 0.6) ** 2}, Capability(outputs=('f',))
    )
    chain = ipop_cmaes(problem, oracle.capability, count=3, population=6)
    drive(chain, oracle, 100_000)
    assert [s._lambda for s in chain.finished] == [6, 12, 24]


def test_restarts_refuse_a_count_below_one():
    with pytest.raises(Refusal, match='count'):
        restarts(lambda _: None, 0)


def _graded() -> tuple[Problem, object]:
    fidelity = Fidelity((Level('coarse', 1.0), Level('medium', 3.0), Level('fine', 9.0)))
    problem = Problem(LINE, (Objective('f'),), fidelity=fidelity)
    capability = Capability(outputs=('f',), fidelities=('coarse', 'medium', 'fine'))
    return problem, from_function(lambda a: {'f': (a.design['t'] - 0.4) ** 2}, capability)


def test_successive_halving_keeps_a_third_per_rung_through_the_truth():
    problem, oracle = _graded()
    sh = SuccessiveHalving(problem, oracle.capability, configs=27, eta=3, seed=1)
    results = drive(sh, oracle, 1000)
    per_rung = [sum(r.action.fidelity == level for r in results) for level in ('coarse', 'medium', 'fine')]
    assert per_rung == [27, 9, 3] and sh.done


def test_hyperband_runs_one_bracket_per_start_rung():
    problem, oracle = _graded()
    chain = hyperband(problem, oracle.capability, eta=3, seed=0)
    drive(chain, oracle, 10_000)
    assert isinstance(chain, Chain) and chain.done and len(chain.finished) == 3


def test_asha_promotes_only_a_point_in_the_top_third_of_its_rung():
    problem, oracle = _graded()
    asha = ASHA(problem, oracle.capability, eta=3, seed=2)
    results = drive(asha, oracle, 60)
    coarse = {r.action.design['t']: r.cell.values['f'] for r in results if r.action.fidelity == 'coarse'}
    promoted = {r.action.design['t'] for r in results if r.action.fidelity == 'medium'}
    assert promoted and len(promoted) <= len(coarse) // 3 + 1
    ranked = sorted(coarse, key=coarse.get)
    assert promoted <= set(ranked[: len(ranked) // 3 + 1])


def test_a_multi_fidelity_schedule_is_ineligible_without_fidelity_levels():
    problem = Problem(LINE, (Objective('f'),))
    assert isinstance(match(SOLVERS['hyperband'], problem, Capability(outputs=('f',))), Ineligible)
    with pytest.raises(Refusal, match='fidelity'):
        SuccessiveHalving(problem, Capability(outputs=('f',)), configs=3)


def test_a_warm_start_hands_the_archive_front_to_the_next_population(tmp_path):
    problem, oracle = _concave()
    first = run(
        problem, oracle, nsga2(problem, oracle.capability, spec=SampleSpec(n_samples=10, seed=1)),
        Record(tmp_path / 'r.jsonl'), Budget(evaluations=30),
    )  # fmt: skip
    rows = warm_start(first.archive, 10)
    front = {round(float(problem.space.encode(r.action.design)[0]), 12) for r in first.archive.front()}
    assert rows.shape == (min(10, len(first.archive.complete())), 1)
    assert front <= {round(float(t), 12) for t in rows[:, 0]} or len(front) > 10
    solver = nsga2(problem, oracle.capability, initial=rows, seed=2)
    assert [problem.space.encode(a.design)[0] for a in solver.ask(100)] == pytest.approx(rows[:, 0].tolist())


def test_an_empty_archive_cannot_warm_start(tmp_path):
    problem, _ = _concave()
    with pytest.raises(Refusal, match='no complete result'):
        warm_start(Archive(problem, Record(tmp_path / 'empty.jsonl')), 5)


def test_a_chain_refuses_a_tell_after_it_concluded():
    chain = Chain([])
    assert chain.done and chain.ask(3) == []
    with pytest.raises(Refusal, match='concluded'):
        chain.tell([])
