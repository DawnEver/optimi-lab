"""What a caller plugs into a study: who answers points, how they are filed, and what a search runs on.

A design POINT is a tuple of natural coordinates in space order. The library never reads an answer:
it hands each one to the caller's :class:`Filing`, which owns the physical record format (one format,
one owner); an answer only has to carry its ``point``.

* :class:`Publisher` -- answers independent points, calling back as each completes.
* :class:`Filing` -- the caller's codec of independent-point answers over the study's record.
* :class:`Codec` -- opens the filing on the record, and binds a search: the problem, the
  :class:`~optimi_lab.policy.ScreeningOracle` and the record view the engine files into.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from lab_commons.outcomes import Record

from optimi_lab.policy import ScreeningOracle
from optimi_lab.problem import Problem

__all__ = ['Answer', 'Codec', 'Filing', 'Publisher', 'SearchBinding']


class Answer(Protocol):
    """What a publisher answers: anything carrying the *point* it answers; the filing reads the rest."""

    point: tuple[float, ...]


@runtime_checkable
class Publisher(Protocol):
    """``publish(points, on_result)``: answer every point, calling *on_result* as each completes.

    A publisher joins its workers before returning, including after a failure, so no callback can
    reach a study that already returned; the study serialises concurrent callbacks.
    """

    def publish(self, points: Sequence[tuple[float, ...]], on_result: Callable[[Answer], None]) -> None: ...


@runtime_checkable
class Filing(Protocol):
    """The caller's independent-point codec: what is reusable, submitted, filed, and read back."""

    def reusable(self, point: tuple[float, ...]) -> bool: ...

    def submit(self, point: tuple[float, ...]) -> None: ...

    def file(self, answer: Answer) -> object: ...

    def row(self, point: tuple[float, ...]) -> object | None: ...


@dataclass(frozen=True)
class SearchBinding:
    """What a search runs on: its *problem*, its *oracle*, and the *record* view results are filed in."""

    problem: Problem
    oracle: ScreeningOracle
    record: Record


@runtime_checkable
class Codec(Protocol):
    """Opens a study's :class:`Filing` on its record, and binds a search to the same filing."""

    def filing(self, record: Record) -> Filing: ...

    def searching(self, filing: Filing, record: Record) -> SearchBinding: ...
