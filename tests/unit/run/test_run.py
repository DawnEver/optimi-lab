"""D22-D26: the async loop, the budget ledger, named termination, resume by replay, and dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Cost, Reason, Record, Status

from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Oracle, Result
from optimi_lab.problem import Fidelity, Level, Objective, Problem, Real, Space
from optimi_lab.run import Budget, Ledger, Termination, run, serial

PROBLEM = Problem(Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))), (Objective('f'), Objective('g')))
CAPABILITY = Capability(outputs=('f', 'g'), version='toy')


class _Toy:
    """A deterministic oracle with a fixed cost, so two runs write byte-identical records."""

    capability = CAPABILITY

    def __init__(self, timeout_first: frozenset[float] = frozenset()) -> None:
        self.calls: list[dict] = []
        self._timeout_first = set(timeout_first)

    async def __call__(self, action: Action) -> Result:
        self.calls.append(dict(action.design))
        x, y = action.design['x'], action.design['y']
        if x in self._timeout_first:
            self._timeout_first.discard(x)
            return Result(action, Cell(Status.TIMEOUT, reason=Reason('timeout'), cost=Cost(wall_s=2.0)))
        return Result(action, Cell(Status.OK, values={'f': x + y, 'g': 1 - x * y}, cost=Cost(wall_s=1.0)))


class _Search:
    """A seeded random search: deterministic in what it is told, so a replay re-asks the same designs."""

    def __init__(
        self, problem: Problem, capability: Capability, seed: int, *, batch: int = 3, stop_after: int | None = None
    ) -> None:
        self._problem, self._capability = problem, capability
        self._rng = np.random.default_rng(seed)
        self._batch, self._stop_after = batch, stop_after
        self.told: list[Result] = []
        self._retry: list[Action] = []

    def ask(self, capacity: int) -> Sequence[Action]:
        retry, self._retry = self._retry[:capacity], self._retry[capacity:]
        fresh = [
            Action.of(
                self._problem, self._capability, self._problem.space.decode(self._rng.random(self._problem.space.width))
            )
            for _ in range(min(self._batch, capacity) - len(retry))
        ]
        return [*retry, *fresh]

    def tell(self, results: Sequence[Result]) -> None:
        self.told.extend(results)
        self._retry.extend(r.action for r in results if not r.cell.status.decided)

    @property
    def done(self) -> bool:
        return self._stop_after is not None and len(self.told) >= self._stop_after


def _run(path: Path, budget: Budget, oracle: Oracle | None = None, **solver_options: object):
    oracle = oracle or _Toy()
    solver = _Search(PROBLEM, CAPABILITY, seed=7, **solver_options)
    return run(PROBLEM, oracle, solver, Record(path), budget), solver, oracle


def test_d22_the_loop_asks_dispatches_files_and_tells(tmp_path: Path) -> None:
    outcome, solver, oracle = _run(tmp_path / 'r.jsonl', Budget(evaluations=7))
    assert outcome.stop.reason is Termination.EVALUATIONS
    assert len(oracle.calls) == 7 == len(solver.told) == len(outcome.archive.complete())
    assert outcome.ledger.evaluations == 7


def test_d22_planted_control_an_oracle_missing_an_output_is_refused_before_any_evaluation(tmp_path: Path) -> None:
    narrow = _Toy()
    narrow.capability = Capability(outputs=('f',), version='toy')
    with pytest.raises(Refusal, match=r"\['g'\]"):
        run(PROBLEM, narrow, _Search(PROBLEM, CAPABILITY, 1), Record(tmp_path / 'r.jsonl'), Budget(evaluations=3))
    assert narrow.calls == []


def test_d23_the_ledger_counts_one_evaluation_per_result_and_its_fidelity_cost_b0_16() -> None:
    problem = Problem(PROBLEM.space, PROBLEM.objectives, fidelity=Fidelity((Level('mec', 1.0), Level('fea', 30.0))))
    capability = Capability(outputs=('f', 'g'), fidelities=('mec', 'fea'))
    ledger = Ledger(problem)
    for fidelity in ('mec', 'fea', 'fea'):
        action = Action.of(problem, capability, {'x': 0.1, 'y': 0.2}, fidelity=fidelity)
        ledger.charge(Result(action, Cell(Status.OK, values={'f': 1.0, 'g': 2.0}, cost=Cost(wall_s=0.5))))
    assert (ledger.evaluations, ledger.cost, ledger.wall_s, ledger.unmeasured) == (3, 61.0, 1.5, 0)
    ledger.charge(Result(action, Cell(Status.OK, values={'f': 1.0, 'g': 2.0})))
    assert ledger.unmeasured == 1  # an unknown wall time is counted as unknown, never as zero


def test_d23_planted_control_a_budget_with_no_limit_is_refused() -> None:
    with pytest.raises(Refusal, match='limit'):
        Budget()
    with pytest.raises(Refusal, match='> 0'):
        Budget(evaluations=0)


def test_d24_each_termination_reason_is_named(tmp_path: Path) -> None:
    assert _run(tmp_path / 'a.jsonl', Budget(wall_s=4.5))[0].stop.reason is Termination.WALL
    assert _run(tmp_path / 'b.jsonl', Budget(cost=4.0))[0].stop.reason is Termination.COST
    assert _run(tmp_path / 'c.jsonl', Budget(evaluations=50), stop_after=5)[0].stop.reason is Termination.SOLVER_DONE
    assert _run(tmp_path / 'd.jsonl', Budget(evaluations=50), batch=0)[0].stop.reason is Termination.NOTHING_ASKED
    assert 'after 6 evaluation(s)' in _run(tmp_path / 'e.jsonl', Budget(evaluations=6))[0].stop.detail


def test_d24_b0_06_a_timed_out_design_is_retried_not_judged(tmp_path: Path) -> None:
    first = np.random.default_rng(7).random(2)
    x = PROBLEM.space.decode(first)['x']
    outcome, solver, _ = _run(tmp_path / 'r.jsonl', Budget(evaluations=6), oracle=_Toy(frozenset({x})))
    statuses = [r.cell.status for r in solver.told if r.action.design['x'] == x]
    assert statuses == [Status.TIMEOUT, Status.OK]
    assert outcome.archive.result(solver.told[0].action.identity).cell.status is Status.OK


def test_d25_gate_a_serial_run_resumes_by_replay_to_an_identical_state(tmp_path: Path) -> None:
    whole, whole_solver, _ = _run(tmp_path / 'whole.jsonl', Budget(evaluations=12))
    _run(tmp_path / 'split.jsonl', Budget(evaluations=5))
    resumed, resumed_solver, oracle = _run(tmp_path / 'split.jsonl', Budget(evaluations=12))
    assert len(oracle.calls) == 7  # the first five came back from the record, not the oracle
    assert Record(tmp_path / 'whole.jsonl').latest() == Record(tmp_path / 'split.jsonl').latest()
    assert [r.action.identity for r in whole_solver.told] == [r.action.identity for r in resumed_solver.told]
    assert whole.ledger == resumed.ledger


def test_d25_planted_control_a_different_seed_does_not_replay(tmp_path: Path) -> None:
    _run(tmp_path / 'r.jsonl', Budget(evaluations=5))
    other = _Search(PROBLEM, CAPABILITY, seed=8)
    oracle = _Toy()
    run(PROBLEM, oracle, other, Record(tmp_path / 'r.jsonl'), Budget(evaluations=10))
    assert len(oracle.calls) == 10  # nothing matched the record, so nothing was replayed


def test_d26_dispatch_is_injected_and_the_record_keeps_ask_and_completion_order_b0_18(tmp_path: Path) -> None:
    async def reversed_dispatch(oracle: Oracle, actions: Sequence[Action]) -> AsyncIterator[Result]:
        results = [await oracle(a) for a in actions]
        for result in reversed(results):
            yield result

    solver = _Search(PROBLEM, CAPABILITY, seed=7)
    outcome = run(
        PROBLEM, _Toy(), solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=3), dispatch=reversed_dispatch
    )
    events = list(outcome.archive.record.replay())
    submitted = [e.key for e in events if e.kind == 'submitted']
    completed = [e.key for e in events if e.kind == 'outcome']
    assert completed == submitted[::-1]  # the record keeps completion order
    assert [r.action.identity.key for r in solver.told] == submitted  # the solver is told in ask order


def test_d26_planted_control_an_identity_asked_twice_is_evaluated_once(tmp_path: Path) -> None:
    class Twice(_Search):
        def ask(self, capacity: int) -> Sequence[Action]:
            first = super().ask(1)
            return [*first, *first]

    oracle = _Toy()
    solver = Twice(PROBLEM, CAPABILITY, seed=7)
    outcome = run(PROBLEM, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=3))
    assert len(oracle.calls) == 3 == outcome.ledger.evaluations
    assert len(solver.told) == 6 and solver.told[0] is solver.told[1]


def test_d26_serial_yields_in_order() -> None:
    async def collect() -> list[float]:
        actions = [Action.of(PROBLEM, CAPABILITY, {'x': x, 'y': 0.0}) for x in (0.1, 0.2, 0.3)]
        return [r.action.design['x'] async for r in serial(_Toy(), actions)]

    assert asyncio.run(collect()) == [0.1, 0.2, 0.3]
