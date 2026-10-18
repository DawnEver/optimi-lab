"""The archive: views over one ``lab_commons.outcomes.Record``, never a second store of results.

Every submission and every OBSERVED outcome is an event in the record -- the submission carries the
action, the outcome carries the cell and the result's extras -- so reopening the file rebuilds the
archive exactly, and the record's own finality rule (the first decision is final) holds here too.

PREDICTIONS ARE KEPT APART (B0-08). A surrogate's answer is a ``Result`` whose cell has evidence
``PREDICTED``; :meth:`Archive.tell` refuses it and :meth:`Archive.predict` holds it in memory only,
so no view reads a prediction unless it is asked for that evidence level by name.

The views answer one question each:

* :meth:`complete` -- every ``OK`` result.
* :meth:`feasible` -- complete, and total scaled violation ``<= tol`` (``tol`` is in the constraints'
  scale units, which is what makes it comparable across constraints).
* :meth:`front` -- the feasible non-dominated results at ONE fidelity and at least one evidence level.
* :meth:`pending` / :meth:`retryable` -- submitted with no outcome / with a RETRYABLE outcome.
* :meth:`at_fidelity` -- every complete result at one level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lab_commons.outcomes import OUTCOME, SUBMITTED, Evidence, Identity, Record, Status

from optimi_lab.core.errors import refuse
from optimi_lab.core.pareto import non_dominated_sorting
from optimi_lab.oracle import Action, Information, Result
from optimi_lab.problem import Problem

__all__ = ['Archive']


def _action(data: dict[str, Any]) -> Action:
    identity = Identity(data['identity'], data['key'])
    want = frozenset(Information(i) for i in data['want'])
    return Action(data['design'], data['fidelity'], want, data['replicate'], identity)


class Archive:
    """The results of one *problem*, filed in *record*. See the module docstring."""

    def __init__(self, problem: Problem, record: Record) -> None:
        self.problem = problem
        self.record = record
        self._actions: dict[str, Action] = {}
        self._results: dict[str, Result] = {}
        self._predictions: list[Result] = []
        self._history: list[Result] = []
        for event in record.replay():
            if event.kind == SUBMITTED:
                self._actions[event.key] = _action(dict(event.data))
            elif event.kind == OUTCOME:
                result = Result.from_extras(self._actions[event.key], event.cell, event.data)
                self._results[event.key] = result
                self._history.append(result)

    def submit(self, action: Action) -> bool:
        """File *action* as submitted; False when it already was."""
        self._actions.setdefault(action.identity.key, action)
        data = action.to_plain() | {'identity': action.identity.canonical, 'key': action.identity.key}
        return self.record.submit(action.identity, data)

    def tell(self, result: Result) -> bool:
        """File an OBSERVED-or-better *result*; a prediction is refused (use :meth:`predict`)."""
        if result.cell.evidence is Evidence.PREDICTED:
            refuse('a PREDICTED result is not an observation; file it with Archive.predict, which keeps it apart')
        result.conform(self.problem)
        key = result.action.identity.key
        if key not in self._actions:
            self.submit(result.action)
        written = self.record.outcome(result.action.identity, result.cell, result.extras())
        self._results[key] = result
        return written

    def predict(self, result: Result) -> None:
        """Hold a PREDICTED *result* apart from the record."""
        if result.cell.evidence is not Evidence.PREDICTED:
            refuse(
                f'only a PREDICTED result is a prediction; this one is {result.cell.evidence.name}, tell it instead (OBSERVED)'
            )
        self._predictions.append(result.conform(self.problem))

    def result(self, identity: Identity | str) -> Result | None:
        """The current result filed under *identity*, if any."""
        return self._results.get(identity.key if isinstance(identity, Identity) else identity)

    def results(self) -> tuple[Result, ...]:
        """Every current observed result, in first-filed order."""
        return tuple(self._results.values())

    def history(self) -> tuple[Result, ...]:
        """Every outcome the record held when this archive was opened, in recorded order."""
        return tuple(self._history)

    def predictions(self) -> tuple[Result, ...]:
        return tuple(self._predictions)

    def pending(self) -> tuple[Action, ...]:
        """Submitted, with no outcome yet."""
        return tuple(self._actions[key] for key in self._actions if key not in self._results)

    def retryable(self) -> tuple[Result, ...]:
        """Results whose current status is RETRYABLE."""
        return tuple(r for r in self._results.values() if not r.cell.status.decided)

    def complete(self) -> tuple[Result, ...]:
        return tuple(r for r in self._results.values() if r.cell.status is Status.OK)

    def violation(self, result: Result) -> float:
        """Total scaled violation of *result* under this archive's problem."""
        return result.violation(self.problem)

    def feasible(self, tol: float = 0.0) -> tuple[Result, ...]:
        return tuple(r for r in self.complete() if self.violation(r) <= tol)

    def at_fidelity(self, fidelity: str | None) -> tuple[Result, ...]:
        return tuple(r for r in self.complete() if r.action.fidelity == fidelity)

    def front(
        self, evidence: Evidence = Evidence.OBSERVED, *, fidelity: str | None = None, tol: float = 0.0
    ) -> tuple[Result, ...]:
        """Feasible non-dominated results at *fidelity* with evidence ``>= evidence``.

        Asking for PREDICTED evidence admits the held predictions too; nothing else does.
        """
        if self.problem.fidelity is not None and fidelity is None:
            refuse(
                'this problem has several fidelity levels; name the fidelity of the front, a front across levels compares unlike values'
            )
        pool = [r for r in self.feasible(tol) if r.action.fidelity == fidelity and r.cell.evidence >= evidence]
        if evidence is Evidence.PREDICTED:
            pool += [
                r
                for r in self._predictions
                if r.cell.status is Status.OK and r.action.fidelity == fidelity and self.violation(r) <= tol
            ]
        if not pool:
            return ()
        matrix = np.array([[o.minimized(r.cell.values[o.name]) for o in self.problem.objectives] for r in pool])
        return tuple(pool[i] for i in non_dominated_sorting(matrix))
