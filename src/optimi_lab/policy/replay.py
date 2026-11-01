"""The run's replay view: decided evidence only, over the one authoritative record.

:func:`optimi_lab.run.run` replays every outcome the record holds. A routed search files policy
dispositions there too: replaying a ``SCREENED`` answer as if the oracle had given it would spend a
phantom evaluation and change what is later told for an action actually evaluated under the same
identity, and a RETRYABLE answer must not prevent a fresh attempt after recovery. ``SCREENED`` is
not a decided status, so both are one rule: the replay shows submissions, caller events and decided
outcomes. Every write still goes through to the record, which keeps all of them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from lab_commons.outcomes import Cell, Event, Expectation, Identity, Record, Status

__all__ = ['DecidedReplay']


class DecidedReplay(Record):
    """*record*, replaying only events without a cell or with a decided one; writes pass through."""

    def __init__(self, record: Record) -> None:
        self.record = record
        self.path = record.path

    def replay(self) -> Iterator[Event]:
        return (event for event in self.record.replay() if event.cell is None or event.cell.status.decided)

    def submit(self, identity: Identity | str, data: Mapping[str, Any] | None = None) -> bool:
        return self.record.submit(identity, data)

    def outcome(
        self,
        identity: Identity | str,
        cell: Cell,
        data: Mapping[str, Any] | None = None,
        attempt: int | None = None,
        expectation: Expectation | None = None,
    ) -> bool:
        return self.record.outcome(identity, cell, data, attempt, expectation)

    def append(self, event: Event) -> bool:
        return self.record.append(event)

    def attempts(self, identity: Identity | str) -> int:
        return self.record.attempts(identity)

    def latest(self) -> dict[str, Cell]:
        return self.record.latest()

    def decided(self) -> dict[str, Cell]:
        return self.record.decided()

    def pending(self) -> list[str]:
        return self.record.pending()

    def by_status(self, status: Status) -> list[str]:
        return self.record.by_status(status)
