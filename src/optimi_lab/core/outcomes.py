"""What an evaluation produced — a value AND the outcome behind it, per cell.

The old package had one way to say "this point produced nothing": fold a single ``inf`` SENTINEL
CONSTANT into the output matrix — defined byte-identically in two modules — and hand the result to
an optimizer that read it as "very bad". Two consequences were measured: its training path's
"normalize infinities" lines were no-ops, because the sentinel they tested for WAS ``inf``, and
the matrices carrying it reached scikit-learn, which raised ``ValueError: Input y contains
infinity``. That constant's spelling is retired and registered with its replacement in
``tests/architecture/test_a_retired_spelling_stays_retired.py``, which is the one file allowed to
write it: a worked example in a docstring is how a dead name comes back.

Here an :class:`Evaluation` carries an :class:`Outcome` per cell and its constructor is the only
way in: a cell marked :attr:`Outcome.OK` must hold a finite number or construction refuses, a
cell marked anything else has its number REPLACED by NaN, and the number is therefore never
readable as a value unless the outcome says it is one. The vocabulary mirrors the states the
downstream consumer already distinguishes; ``produced_a_value`` is True for ``OK`` alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from optimi_lab.core.constraints import ConstraintSet
from optimi_lab.core.errors import coerce_enum, refuse
from optimi_lab.core.space import ObjectiveSet

__all__ = ['Evaluation', 'Outcome']


class Outcome(StrEnum):
    """Why a cell of an evaluation holds what it holds."""

    OK = 'ok'
    REFUSED = 'refused'
    ERROR = 'error'
    TIMEOUT = 'timeout'
    CANCELLED = 'cancelled'

    @property
    def produced_a_value(self) -> bool:
        """Whether this outcome carries a number the caller may read. True for ``OK`` alone."""
        return self is Outcome.OK


@dataclass(frozen=True, slots=True, eq=False)
class _EvaluationFields:
    inputs: np.ndarray
    values: np.ndarray
    outcomes: np.ndarray | None = None
    constraints: np.ndarray | None = None


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Evaluation(_EvaluationFields):
    """A batch of evaluated points: the points, their values, and the outcome of each value.

    ``inputs`` has shape ``(n_points, n_var)``; ``values`` has shape ``(n_points, n_obj)``, and
    a cell whose outcome is not ``OK`` is replaced by NaN here whatever the caller passed;
    ``outcomes`` has the same shape as ``values``, holds :class:`Outcome` members or their
    values, and defaults to all-``OK``.

    The two value arrays are aligned cell by cell, not row by row: a run can produce some of a
    point's objectives and fail on the others, which is what the old ``solution_type``-per-result
    model could not express without a whole-row sentinel.

    ``constraints`` has shape ``(n_points, n_con)``, one column per declared
    :class:`~optimi_lab.core.space.Constraint`, and defaults to no column at all. In the caller's
    frame a cell is the constrained quantity itself; in the minimization frame
    (:meth:`to_minimization`) it is that quantity's normalised SHORTFALL, 0 when satisfied, so a
    proposer reads a point's violation as :func:`~optimi_lab.core.pareto.total_violation` of its
    row. A constraint that did not compute is NaN -- never an infinity, which would read as a
    number -- and leaves its point incomplete: feasibility unknown is not feasibility.
    """

    def __init__(
        self,
        inputs: np.ndarray,
        values: np.ndarray,
        outcomes: np.ndarray | None = None,
        constraints: np.ndarray | None = None,
    ) -> None:
        inputs = np.asarray(inputs, dtype=float)
        values = np.asarray(values, dtype=float)
        if inputs.ndim != 2:
            refuse(f'inputs must be a 2-D array with one row per point, got shape {inputs.shape}')
        if values.ndim != 2:
            refuse(f'values must be a 2-D array with one column per objective, got shape {values.shape}')
        if values.shape[0] != inputs.shape[0]:
            refuse(f'values must hold one row per point, got {values.shape[0]} row(s) for {inputs.shape[0]} point(s)')

        outcomes = self._coerce_outcomes(outcomes, values.shape)
        produced = outcomes == Outcome.OK
        offending = np.argwhere(produced & ~np.isfinite(values))
        if offending.size:
            cell = tuple(int(position) for position in offending[0])
            refuse(
                f'cell {list(cell)} is marked {Outcome.OK.value!r} but holds {float(values[cell])!r}; a point that '
                f'produced no value must carry one of {[member.value for member in Outcome]} instead of a number'
            )
        constraints = self._coerce_constraints(constraints, inputs.shape[0])
        _EvaluationFields.__init__(self, inputs, np.where(produced, values, np.nan), outcomes, constraints)

    @staticmethod
    def _coerce_constraints(constraints: np.ndarray | None, n_points: int) -> np.ndarray:
        """Return a ``(n_points, n_con)`` float matrix, or refuse a misshapen or infinite one."""
        if constraints is None:
            return np.zeros((n_points, 0))
        constraints = np.asarray(constraints, dtype=float)
        if constraints.ndim != 2 or constraints.shape[0] != n_points:
            refuse(
                f'constraints must be a 2-D array with one row per point and one column per constraint, got shape '
                f'{constraints.shape} for {n_points} point(s)'
            )
        if np.any(np.isinf(constraints)):
            refuse(
                'a constraint cell holds an infinity; a constraint that did not compute is NaN, so it cannot be '
                'read as a number'
            )
        return constraints

    @staticmethod
    def _coerce_outcomes(outcomes: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
        """Return ``shape``-shaped object array of :class:`Outcome`, or refuse naming the members."""
        if outcomes is None:
            # Not `np.full`: it stores a `StrEnum` member as its plain `str` value, so the default
            # cells read `'ok'` and fail every `is Outcome.OK` test. `fill` keeps the member.
            default = np.empty(shape, dtype=object)
            default.fill(Outcome.OK)
            return default
        given = np.asarray(outcomes, dtype=object)
        if given.shape != shape:
            refuse(f'outcomes must hold one entry per value, expected shape {shape}, got {given.shape}')
        coerced = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            coerced[index] = coerce_enum(Outcome, given[index], f'the outcome at cell {list(index)}')
        return coerced

    @classmethod
    def stack(cls, evaluations: Sequence[Evaluation]) -> Evaluation:
        """Concatenate batches along the point axis.

        Refuses an empty list, or a set of batches that disagree on ``n_obj`` or ``n_var``: a
        stack that broadcast the narrower side would misalign every column after it.
        """
        if not evaluations:
            refuse('cannot stack an empty batch list; there is no evaluation to return')
        n_obj = {evaluation.n_obj for evaluation in evaluations}
        n_var = {evaluation.inputs.shape[1] for evaluation in evaluations}
        n_con = {evaluation.n_con for evaluation in evaluations}
        if len(n_obj) != 1 or len(n_var) != 1 or len(n_con) != 1:
            refuse(
                f'stacked batches must share a width, got n_obj={sorted(n_obj)}, n_var={sorted(n_var)} and '
                f'n_con={sorted(n_con)}; these batches do not describe the same problem'
            )
        return cls(
            inputs=np.concatenate([evaluation.inputs for evaluation in evaluations]),
            values=np.concatenate([evaluation.values for evaluation in evaluations]),
            outcomes=np.concatenate([evaluation.outcomes for evaluation in evaluations]),
            constraints=np.concatenate([evaluation.constraints for evaluation in evaluations]),
        )

    @property
    def n_obj(self) -> int:
        """Number of objective columns."""
        return self.values.shape[1]

    @property
    def n_con(self) -> int:
        """Number of constraint columns; 0 for an unconstrained batch."""
        return self.constraints.shape[1]

    def __len__(self) -> int:
        return self.inputs.shape[0]

    @property
    def is_complete(self) -> bool:
        """Whether every point produced every objective and every constraint."""
        return bool(np.all(self.complete_mask()))

    def complete_mask(self) -> np.ndarray:
        """Boolean mask, shape ``(n_points,)``: every objective ``OK`` and every constraint computed."""
        return np.all(self.outcomes == Outcome.OK, axis=1) & np.all(np.isfinite(self.constraints), axis=1)

    def point_outcomes(self) -> tuple[Outcome, ...]:
        """One outcome per point: ``OK``, or the first outcome that stopped it producing one."""
        per_point = []
        for row in self.outcomes:
            outcome = Outcome.OK
            for cell in row:
                if cell is not Outcome.OK:
                    outcome = cell
                    break
            per_point.append(outcome)
        return tuple(per_point)

    def value(self, row: int, column: int) -> float:
        """Return the value at ``row``/``column``, refusing if that cell produced none.

        Reading a cell without checking its outcome is what turned an unevaluated point into a
        number; the only cells this returns are the ones whose outcome says they hold one.
        """
        outcome = self.outcomes[row, column]
        if not outcome.produced_a_value:
            refuse(
                f'cell [{row}, {column}] carries outcome {outcome.value!r}, so it holds no value; '
                f'only {Outcome.OK.value!r} cells are readable'
            )
        return float(self.values[row, column])

    def complete(self) -> Evaluation:
        """Return the points whose every objective produced a value; may hold no point at all."""
        mask = self.complete_mask()
        return Evaluation(
            inputs=self.inputs[mask],
            values=self.values[mask],
            outcomes=self.outcomes[mask],
            constraints=self.constraints[mask],
        )

    def require_complete(self) -> Evaluation:
        """Return the complete points, refusing if any point is incomplete.

        Refuses naming the points and their outcomes: a caller that needs every declared point
        has no honest way to proceed with a front built from some of them.
        """
        if not self.is_complete:
            per_point = self.point_outcomes()
            incomplete = np.flatnonzero(~self.complete_mask()).tolist()
            reasons = sorted({per_point[index].value for index in incomplete})
            msg = (
                f'point(s) {incomplete} produced no value for every objective and constraint, with outcome(s) '
                f'{reasons}; '
                f'drop them with complete() or evaluate them again — a point that did not compute is not a data point'
            )
            refuse(msg)
        return self

    def to_minimization(self, objectives: ObjectiveSet, constraints: ConstraintSet | None = None) -> Evaluation:
        """Return this batch in the frame a proposer ranks in, as a new batch.

        Every maximizing objective is negated, and every constraint cell becomes its normalised
        shortfall under ``constraints`` -- the declaration the matrix's columns were measured
        against, required exactly when the batch has constraint columns. Inputs and outcomes are
        carried over unchanged.
        """
        if constraints is None:
            if self.n_con:
                refuse(
                    f'this batch carries {self.n_con} constraint column(s), and a raw constraint value is not a '
                    f'shortfall; pass the ConstraintSet they were measured against'
                )
            shortfalls = self.constraints
        else:
            shortfalls = constraints.shortfall(self.constraints)
        return Evaluation(
            inputs=self.inputs,
            values=objectives.to_minimization(self.values),
            outcomes=self.outcomes,
            constraints=shortfalls,
        )
