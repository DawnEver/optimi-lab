"""The surrogate pool fitted to a run's own results: which members, taught what, scored how.

:func:`fit_surrogate` teaches each registry model of a pool the run's fully computed (``OK``) results
at the target evidence and keeps its cross-validated scores beside the fit, so the steering gate and
a report read the same objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from lab_commons.outcomes import Evidence, Status

from optimi_lab.model.protocol import Fitted
from optimi_lab.model.registry import surrogate
from optimi_lab.model.validation import scored_fit
from optimi_lab.oracle import Result
from optimi_lab.policy.routing import natural
from optimi_lab.problem import BlackBox, Problem

__all__ = [
    'GP_MAX_ROWS',
    'MIN_SURROGATE_POINTS',
    'MODEL_SETTINGS',
    'STEERING_MODELS',
    'SURROGATE_POOL',
    'ScoredModel',
    'SurrogateFit',
    'black_boxes',
    'fit_surrogate',
    'targets',
]

#: Three deliberately unlike regressors: a pool can only be measured where its members disagree.
SURROGATE_POOL: Final[tuple[str, ...]] = ('polynomial', 'radial_basis', 'nearest_neighbours')
#: The selectable steering models; ``gp`` also models the constraints (probability of feasibility).
STEERING_MODELS: Final[dict[str, tuple[str, ...]]] = {'pool': SURROGATE_POOL, 'gp': ('gaussian_process',)}
#: Registry settings per model key; a key absent here takes the registry defaults.
MODEL_SETTINGS: Final[dict[str, dict]] = {'gaussian_process': {'family': 'matern52', 'restarts': 1}}
#: The most (most recent) rows the GP is taught: an exact ARD fit is O(n^3).
GP_MAX_ROWS: Final[int] = 400
#: The fewest fully computed rows a fit is attempted on: one row in each of five folds, a margin over
#: the two that cross-validation needs at all.
MIN_SURROGATE_POINTS: Final[int] = 8


@dataclass(frozen=True)
class ScoredModel:
    """One registry model under its *key*, fitted on every row, with its out-of-fold *scores*."""

    key: str
    scores: dict[str, float]
    fitted: Fitted


@dataclass(frozen=True)
class SurrogateFit:
    """The pool fitted to one run's results: *results* is the subset it was TAUGHT.

    *feasibility*: the columns after the objectives are the black-box constraint values and proposals
    are filtered by probability of feasibility. *span* is each column's training range.
    """

    models: tuple[ScoredModel, ...]
    problem: Problem
    results: tuple[Result, ...]
    target_evidence: Evidence = Evidence.OBSERVED
    feasibility: bool = False
    span: tuple[float, ...] = ()


def black_boxes(problem: Problem) -> tuple[BlackBox, ...]:
    """The black-box constraints of *problem*, in declaration order: the columns a feasibility fit models."""
    return tuple(c for c in problem.constraints if isinstance(c, BlackBox))


def targets(problem: Problem, result: Result, feasibility: bool) -> tuple[float, ...]:
    """*result*'s fitted columns: its objective values, then (with *feasibility*) each black-box value."""
    names = [o.name for o in problem.objectives] + ([c.name for c in black_boxes(problem)] if feasibility else [])
    return tuple(float(result.cell.values[name]) for name in names)


def fit_surrogate(
    problem: Problem,
    results: Sequence[Result],
    *,
    evidence: Evidence = Evidence.OBSERVED,
    pool: tuple[str, ...] = SURROGATE_POOL,
    feasibility: bool = False,
    max_rows: int | None = None,
) -> SurrogateFit | None:
    """*pool* fitted to the OK results at *evidence* or better -- or None below :data:`MIN_SURROGATE_POINTS`."""
    taught = tuple(r for r in results if r.cell.status is Status.OK and r.cell.evidence >= evidence)
    if len(taught) < MIN_SURROGATE_POINTS:
        return None
    if max_rows is not None:
        taught = taught[-max_rows:]
    feasibility = feasibility and bool(black_boxes(problem))
    inputs = np.asarray([natural(problem, r.action) for r in taught], dtype=float)
    values = np.asarray([targets(problem, r, feasibility) for r in taught], dtype=float)
    models = []
    for key in pool:
        fit = scored_fit(surrogate(key, **MODEL_SETTINGS.get(key, {})), inputs, values)
        models.append(ScoredModel(key, fit.scores, fit.fitted))
    return SurrogateFit(
        tuple(models), problem, taught, evidence, feasibility, tuple(float(v) for v in np.ptp(values, axis=0))
    )
