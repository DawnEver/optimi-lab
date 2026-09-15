"""What an evaluation produced — a value AND the outcome behind it, per cell.

The old package had one way to say "this point produced nothing": fold ``PENALTY_VALUE =
np.inf`` into the output matrix, defined byte-identically in two modules, and hand the result to
an optimizer that read it as "very bad". Two consequences were measured: the "normalize
infinities" lines in ``train_surrogate_models`` were no-ops because the sentinel they tested for
WAS ``inf``, and the matrices carrying it reached scikit-learn, which raised
``ValueError: Input y contains infinity``.

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
class Evaluation:
    """A batch of evaluated points: the points, their values, and the outcome of each value.

    ``inputs`` has shape ``(n_points, n_var)``; ``values`` has shape ``(n_points, n_obj)``, and
    a cell whose outcome is not ``OK`` is replaced by NaN here whatever the caller passed;
    ``outcomes`` has the same shape as ``values``, holds :class:`Outcome` members or their
    values, and defaults to all-``OK``.

    The two value arrays are aligned cell by cell, not row by row: a run can produce some of a
    point's objectives and fail on the others, which is what the old ``solution_type``-per-result
    model could not express without a whole-row sentinel.
    """

    inputs: np.ndarray
    values: np.ndarray
    outcomes: np.ndarray | None = None

    def __post_init__(self) -> None:
        inputs = np.asarray(self.inputs, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if inputs.ndim != 2:
            refuse(f'inputs must be a 2-D array with one row per point, got shape {inputs.shape}')
        if values.ndim != 2:
            refuse(f'values must be a 2-D array with one column per objective, got shape {values.shape}')
        if values.shape[0] != inputs.shape[0]:
            refuse(f'values must hold one row per point, got {values.shape[0]} row(s) for {inputs.shape[0]} point(s)')

        outcomes = self._coerce_outcomes(values.shape)
        produced = outcomes == Outcome.OK
        offending = np.argwhere(produced & ~np.isfinite(values))
        if offending.size:
            cell = tuple(int(position) for position in offending[0])
            refuse(
                f'cell {list(cell)} is marked {Outcome.OK.value!r} but holds {float(values[cell])!r}; a point that '
                f'produced no value must carry one of {[member.value for member in Outcome]} instead of a number'
            )
        object.__setattr__(self, 'inputs', inputs)
        object.__setattr__(self, 'values', np.where(produced, values, np.nan))
        object.__setattr__(self, 'outcomes', outcomes)

    def _coerce_outcomes(self, shape: tuple[int, int]) -> np.ndarray:
        """Return ``shape``-shaped object array of :class:`Outcome`, or refuse naming the members."""
        if self.outcomes is None:
            return np.full(shape, Outcome.OK, dtype=object)
        given = np.asarray(self.outcomes, dtype=object)
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
        if len(n_obj) != 1 or len(n_var) != 1:
            refuse(
                f'stacked batches must share a width, got n_obj={sorted(n_obj)} and n_var={sorted(n_var)}; '
                f'these batches do not describe the same problem'
            )
        return cls(
            inputs=np.concatenate([evaluation.inputs for evaluation in evaluations]),
            values=np.concatenate([evaluation.values for evaluation in evaluations]),
            outcomes=np.concatenate([evaluation.outcomes for evaluation in evaluations]),
        )

    @property
    def n_obj(self) -> int:
        """Number of objective columns."""
        return self.values.shape[1]

    def __len__(self) -> int:
        return self.inputs.shape[0]

    @property
    def is_complete(self) -> bool:
        """Whether every cell of every point produced a value."""
        return bool(np.all(self.outcomes == Outcome.OK))

    def complete_mask(self) -> np.ndarray:
        """Boolean mask, shape ``(n_points,)``, True where EVERY objective of the point is ``OK``."""
        return np.all(self.outcomes == Outcome.OK, axis=1)

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
        return Evaluation(inputs=self.inputs[mask], values=self.values[mask], outcomes=self.outcomes[mask])

    def require_complete(self) -> Evaluation:
        """Return the complete points, refusing if any point is incomplete.

        Refuses naming the points and their outcomes: a caller that needs every declared point
        has no honest way to proceed with a front built from some of them.
        """
        if not self.is_complete:
            per_point = self.point_outcomes()
            incomplete = [index for index, outcome in enumerate(per_point) if not outcome.produced_a_value]
            reasons = sorted({per_point[index].value for index in incomplete})
            msg = (
                f'point(s) {incomplete} produced no value for every objective, with outcome(s) {reasons}; '
                f'drop them with complete() or evaluate them again — a point that did not compute is not a data point'
            )
            refuse(msg)
        return self

    def to_minimization(self, objectives: ObjectiveSet) -> Evaluation:
        """Return this batch with every maximizing objective negated, as a new batch.

        Inputs and outcomes are carried over unchanged; only the values move frames.
        """
        return Evaluation(
            inputs=self.inputs,
            values=objectives.to_minimization(self.values),
            outcomes=self.outcomes,
        )
