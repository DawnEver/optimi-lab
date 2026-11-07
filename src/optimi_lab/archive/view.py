"""One filed ``Result`` read against its problem -- the row a report holds, never a second store.

A :class:`ResultView` answers, from the result and the problem alone, whether the row is rankable,
how far it violates its constraints, its objective vector and whether it is feasible. :func:`views`
reads an archive in FIRST-SUBMITTED order (independent of completion order), predictions last;
:func:`open_record` is the append-only record a run files into.

EQUALITY IS THE READING'S. A ``Result`` compares by identity and every reopen of a record decodes
fresh ones, so two views are equal when they read the same action identity AND the same cell --
a view of a resumed archive equals the view of the live one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from lab_commons.outcomes import SUBMITTED, Record, Status

from optimi_lab.archive.archive import Archive
from optimi_lab.core.errors import refuse
from optimi_lab.oracle import Result
from optimi_lab.problem import Problem

__all__ = ['ResultView', 'open_record', 'views']


def open_record(path: Path | Record | None = None, *, root: Path | None = None) -> Record:
    """The record a run files into: *path* as given, a Record at it, or a fresh ``root/<uuid>/archive.jsonl``."""
    if isinstance(path, Record):
        return path
    if path is not None:
        return Record(Path(path))
    if root is None:
        refuse('a fresh record needs root= (the caller owns where runs are written); this package resolves no path')
    return Record(Path(root) / uuid4().hex / 'archive.jsonl')


@dataclass(frozen=True, eq=False)
class ResultView:
    """*result* read against *problem*. See the module docstring."""

    problem: Problem
    result: Result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResultView):
            return NotImplemented
        same_action = self.result.action.identity.key == other.result.action.identity.key
        return same_action and self.result.cell == other.result.cell

    def __hash__(self) -> int:
        return hash(self.result.action.identity.key)

    @property
    def uncomputed(self) -> tuple[str, ...]:
        """The declared objectives the cell holds no number for, in declaration order."""
        return tuple(o.name for o in self.problem.objectives if o.name not in self.result.cell.values)

    @property
    def rankable(self) -> bool:
        """Every declared output is OK -- the cell's own status, not merely a PARTIAL number."""
        return self.result.cell.status is Status.OK

    @property
    def violation(self) -> float:
        """The total scaled violation (``Result.violation``, the one formula); 0 when every constraint holds."""
        return self.result.violation(self.problem)

    @property
    def feasible(self) -> bool:
        """Rankable and violating nothing; partial evidence cannot establish feasibility."""
        return self.rankable and self.violation == 0.0

    def scores(self) -> tuple[float, ...]:
        """The objective vector in declaration order and natural sense; refused when any was not computed."""
        if self.uncomputed:
            refuse(
                f'this result did not compute {list(self.uncomputed)}; no sentinel score stands in for a number '
                f'that was never computed'
            )
        return tuple(float(self.result.cell.values[o.name]) for o in self.problem.objectives)


def views(archive: Archive) -> tuple[ResultView, ...]:
    """Every result of *archive* in first-submitted order, then its held predictions."""
    observed = {result.action.identity.key: result for result in archive.results()}
    keys = dict.fromkeys(event.key for event in archive.record.replay() if event.kind == SUBMITTED)
    ordered = (*(observed[key] for key in keys if key in observed), *archive.predictions())
    return tuple(ResultView(archive.problem, result) for result in ordered)
