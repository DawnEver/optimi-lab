"""The run: ask, file, dispatch, file, tell -- until a named reason stops it.

EXECUTION IS INJECTED. ``dispatch(oracle, actions)`` yields results as they finish; this package
ships only :func:`serial`. A cluster, a process pool or a vendor queue is a caller's dispatch.

RESUME IS REPLAY. Every action the solver asks is looked up in the record first: the k-th ask of an
identity is answered with the k-th outcome recorded for it -- so a TIMEOUT that was retried replays
as the TIMEOUT and then the retry -- and charged to the ledger exactly as it was the first time; only
an ask the record has no answer for is dispatched. Re-driving a deterministic solver over an
interrupted run's record therefore reaches the state the uninterrupted run would have reached -- the
same results told in the same order, the same ledger -- and evaluates only what was never finished.
A submission with no outcome (a run killed mid-batch) is simply asked again.

ONE EVALUATION PER IDENTITY. An identity asked twice in a batch is dispatched once; one that already
has a DECIDED result is answered from the archive and not charged again (it costs nothing). Each
batch is told to the solver in ASK order, so a deterministic solver sees the same sequence live and
on replay; the completion order is what the record keeps (B0-18).
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass

from lab_commons.outcomes import Record

from optimi_lab.archive import Archive
from optimi_lab.oracle import Action, Oracle, Result
from optimi_lab.problem import Problem
from optimi_lab.run.ledger import Budget, Ledger, Stop, Termination
from optimi_lab.solver.protocol import Solver

__all__ = ['Dispatch', 'Run', 'run', 'run_async', 'serial']

#: ``dispatch(oracle, actions)`` -> results as they finish, in any order.
Dispatch = Callable[[Oracle, Sequence[Action]], AsyncIterator[Result]]


async def serial(oracle: Oracle, actions: Sequence[Action]) -> AsyncIterator[Result]:
    """One action at a time, in order."""
    for action in actions:
        yield await oracle(action)


@dataclass(frozen=True, slots=True)
class Run:
    """What a run leaves: its *archive*, its *ledger*, and why it *stop*-ped."""

    archive: Archive
    ledger: Ledger
    stop: Stop


async def run_async(
    problem: Problem, oracle: Oracle, solver: Solver, record: Record, budget: Budget, *, dispatch: Dispatch = serial
) -> Run:
    """Drive *solver* against *oracle* on *problem*, filing everything in *record*, within *budget*."""
    oracle.capability.require(problem)
    archive = Archive(problem, record)
    ledger = Ledger(problem)
    replay: dict[str, deque[Result]] = defaultdict(deque)
    for recorded in archive.history():
        replay[recorded.action.identity.key].append(recorded)
    while True:
        stop = ledger.exhausted(budget)
        if stop is None and solver.done:
            stop = Stop(Termination.SOLVER_DONE, f'the solver concluded after {ledger.evaluations} evaluation(s)')
        if stop is not None:
            return Run(archive, ledger, stop)
        remaining = ledger.remaining(budget)
        actions = list(solver.ask(sys.maxsize if remaining is None else remaining))
        if not actions:
            return Run(
                archive,
                ledger,
                Stop(Termination.NOTHING_ASKED, f'the solver asked nothing after {ledger.evaluations} evaluation(s)'),
            )
        answers = await _answer(
            actions, replay=replay, archive=archive, oracle=oracle, dispatch=dispatch, ledger=ledger
        )
        solver.tell([answers[action.identity.key] for action in actions])


async def _answer(
    actions: list[Action],
    *,
    replay: dict[str, deque[Result]],
    archive: Archive,
    oracle: Oracle,
    dispatch: Dispatch,
    ledger: Ledger,
) -> dict[str, Result]:
    """One result per asked identity: replayed, reused, or dispatched -- each identity at most once."""
    answers: dict[str, Result] = {}
    fresh: dict[str, Action] = {}
    for action in actions:
        key = action.identity.key
        if key in answers or key in fresh:
            continue
        known, current = replay.get(key), archive.result(key)
        if known:
            answers[key] = known.popleft()
            ledger.charge(answers[key])
        elif current is not None and current.cell.status.decided:
            answers[key] = current
        else:
            archive.submit(action)
            fresh[key] = action
    async for result in dispatch(oracle, list(fresh.values())):
        archive.tell(result)
        ledger.charge(result)
        answers[result.action.identity.key] = result
    return answers


def run(
    problem: Problem, oracle: Oracle, solver: Solver, record: Record, budget: Budget, *, dispatch: Dispatch = serial
) -> Run:
    """The synchronous door onto :func:`run_async`."""
    return asyncio.run(run_async(problem, oracle, solver, record, budget, dispatch=dispatch))
