"""Routing: a caller policy declines a solve or picks the evaluator for each proposed action.

A :class:`ScreeningOracle` is an :class:`~optimi_lab.oracle.Oracle` that also says how many
evaluators it declares, what a SCREENED disposition of an action is, which action an evaluator index
turns it into, and how a whole batch is published. :class:`RoutingPolicy` sits between a solver and
the run: it maps identities, never searches. A screened action is filed and told without dispatch or
an evaluation charge; the solver always sees its own original identities, the archive the ACTUAL
fidelity. The initial batch is evaluated as drawn -- a router decides offspring from results it has.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from numbers import Real as RealNumber
from typing import Protocol, runtime_checkable

import numpy as np
from lab_commons.outcomes import Record

from optimi_lab.archive import Archive
from optimi_lab.core.errors import refuse
from optimi_lab.oracle import Action, Oracle, Result
from optimi_lab.problem import Problem
from optimi_lab.solver.protocol import Solver

__all__ = ['RouteChoice', 'RouteRule', 'RoutingPolicy', 'ScreeningOracle', 'natural']

#: ``router(points, results)`` -> one evaluator index (or ``RouteChoice.SCREENED``) per unique proposed
#: action; *points* are natural coordinates in space order, *results* the told results in told order.
RouteRule = Callable[[np.ndarray, tuple[Result, ...]], np.ndarray]


class RouteChoice(IntEnum):
    """A policy may decline a solve; any other choice is a declared evaluator index."""

    SCREENED = -1


@runtime_checkable
class ScreeningOracle(Oracle, Protocol):
    """An oracle with *evaluators* legs, a screened disposition, routed actions and a batch door."""

    evaluators: int

    def screened(self, action: Action) -> Result:
        """The SCREENED result *action* is filed and told with; no solve is spent."""
        ...

    def routed(self, action: Action, index: int) -> Action:
        """*action* as evaluator *index* answers it (its fidelity), or *action* itself for one evaluator."""
        ...

    def publish_batch(self, actions: Sequence[Action], on_result: Callable[[Result], None]) -> None:
        """Answer every action, calling *on_result* as each completes; return only when all have."""
        ...


def natural(problem: Problem, action: Action) -> tuple[float, ...]:
    """*action*'s design as natural coordinates in space order; a non-numeric coordinate is refused."""
    values = tuple(action.design[name] for name in problem.space.names)
    if any(isinstance(value, bool) or not isinstance(value, RealNumber) for value in values):
        refuse(f'a routed design is continuous and numeric, got {values!r}')
    return tuple(float(value) for value in values)


class RoutingPolicy:
    """*solver* behind *router*'s choices on *oracle*, filing screened answers in *record*."""

    def __init__(
        self, solver: Solver, oracle: ScreeningOracle, problem: Problem, record: Record, router: RouteRule | None
    ) -> None:
        self.solver, self.oracle, self.problem, self.record, self.router = solver, oracle, problem, record, router
        self.order: tuple[Action, ...] = ()
        self.original: dict[str, Action] = {}
        self.screened: dict[str, Result] = {}
        self.batch_ends: list[int] = []
        # Only answers ALREADY TOLD, never the archive's later generations; references, not a store.
        self.observed: dict[str, Result] = {}

    def results(self) -> tuple[Result, ...]:
        """The told results, in told order (one per original identity)."""
        return tuple(self.observed.values())

    def ask(self, capacity: int) -> list[Action]:
        # The run reads an empty ask as termination, so a batch the policy screened entirely is told
        # here and the next is asked; each such tell consumes one bounded batch, so this cannot loop.
        while not self.solver.done:
            selected = self._ask_batch(capacity)
            if selected or not self.order:
                return selected
        return []

    def _ask_batch(self, capacity: int) -> list[Action]:
        self.order = tuple(self.solver.ask(capacity))
        if any(not isinstance(action, Action) for action in self.order):
            refuse('ask(capacity) must return Actions, not an untyped coordinate array')
        self.original, self.screened = {}, {}
        if not self.order:
            return []
        archive = Archive(self.problem, self.record)
        told = self.results()
        # ONE decision per identity, as dispatch makes one evaluation: repeated parents share it.
        unique: dict[str, Action] = {}
        for action in self.order:
            unique.setdefault(action.identity.key, action)
        proposed = tuple(unique.values())
        points = np.array([natural(self.problem, action) for action in proposed])
        routed = self.router is not None and bool(told)
        choices = np.asarray(self.router(points, told)) if routed else np.zeros(len(proposed), dtype=int)
        if (
            choices.shape != (len(proposed),)
            or not np.issubdtype(choices.dtype, np.integer)
            or np.any((choices < RouteChoice.SCREENED) | (choices >= self.oracle.evaluators))
        ):
            refuse('router must return one declared evaluator index or -1 (screened) per unique action')
        by_identity = {action.identity.key: int(index) for action, index in zip(proposed, choices, strict=True)}
        selected = []
        for action in self.order:
            index = by_identity[action.identity.key]
            if index == RouteChoice.SCREENED:
                known = self.observed.get(action.identity.key)
                if known is not None and known.cell.status.decided:
                    selected.append(action)
                    self.original[action.identity.key] = action
                    continue
                result = self.oracle.screened(action)
                archive.submit(action)
                persisted = archive.result(action.identity)
                if persisted is None or not persisted.cell.status.decided:
                    archive.tell(result)
                self.screened[action.identity.key] = result
            else:
                routed_action = self.oracle.routed(action, index)
                selected.append(routed_action)
                archive.submit(routed_action)
                self.original[routed_action.identity.key] = action
        if not selected:
            self.tell(())
        return selected

    def tell(self, results: Sequence[Result]) -> None:
        answers = dict(self.screened)
        observed = dict(self.screened)
        for result in results:
            original = self.original[result.action.identity.key]
            answers[original.identity.key] = Result.from_extras(original, result.cell, result.extras())
            observed[original.identity.key] = result
        for action in self.order:
            result = observed[action.identity.key]
            self.observed[result.action.identity.key] = result
        self.batch_ends.append(len(self.observed))
        self.solver.tell([answers[action.identity.key] for action in self.order])

    @property
    def done(self) -> bool:
        return self.solver.done
