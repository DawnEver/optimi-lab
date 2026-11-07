"""A pooled dispatch: a caller-owned executor is the pool -- its width the cap, its initializer per worker."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lab_commons.outcomes import Record

from optimi_lab.oracle import Action, Capability, from_function
from optimi_lab.problem import Objective, Problem, Real, Space
from optimi_lab.run import Budget, Termination, pooled, run, serial

PROBLEM = Problem(Space((Real('x', 0.0, 1.0),)), (Objective('f'),))


class _Asks:
    """Asks *points* once, then is done."""

    def __init__(self, points) -> None:
        self.points, self.told, self.done = points, [], False

    def ask(self, capacity: int):
        self.done = True
        return [Action.of(PROBLEM, ORACLE.capability, {'x': x}) for x in self.points]

    def tell(self, results) -> None:
        self.told.extend(results)


ORACLE = from_function(lambda a: {'f': a.design['x'] ** 2}, Capability(outputs=('f',)))


def test_a_pooled_run_tells_the_same_results_as_a_serial_one(tmp_path: Path) -> None:
    points = (0.1, 0.5, 0.9, 0.3)
    serial_solver, pooled_solver = _Asks(points), _Asks(points)
    run(PROBLEM, ORACLE, serial_solver, Record(tmp_path / 's.jsonl'), Budget(evaluations=10), dispatch=serial)
    with ThreadPoolExecutor(2) as executor:
        done = run(
            PROBLEM,
            ORACLE,
            pooled_solver,
            Record(tmp_path / 'p.jsonl'),
            Budget(evaluations=10),
            dispatch=pooled(executor),
        )
    assert done.stop.reason is Termination.SOLVER_DONE
    assert [r.cell.values for r in pooled_solver.told] == [r.cell.values for r in serial_solver.told]


def test_the_lowest_priority_starts_first_and_the_initializer_runs_per_worker(tmp_path: Path) -> None:
    started: list[float] = []
    initialised: list[str] = []

    def evaluate(action: Action) -> dict[str, float]:
        started.append(action.design['x'])
        return {'f': action.design['x']}

    oracle = from_function(evaluate, Capability(outputs=('f',)))
    with ThreadPoolExecutor(1, initializer=lambda: initialised.append(threading.current_thread().name)) as executor:
        run(
            PROBLEM,
            oracle,
            _Asks((0.9, 0.1, 0.5)),
            Record(tmp_path / 'r.jsonl'),
            Budget(evaluations=10),
            dispatch=pooled(executor, priority=lambda action: action.design['x']),
        )
    assert started == [0.1, 0.5, 0.9]
    assert len(initialised) == 1
