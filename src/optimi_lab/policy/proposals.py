"""What a fitted model proposes: a fresh candidate batch, its predictions, and the front chosen from them.

The predicted numbers only ever reach a sort and the accuracy log, never a result: what leaves
:func:`surrogate_selection` is natural design POINTS.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from optimi_lab.core.pareto import non_dominated_sorting
from optimi_lab.core.sampling import SampleSpec
from optimi_lab.model.bench import ColumnBound, feasibility_probability
from optimi_lab.policy.surrogate_fit import ScoredModel, SurrogateFit, black_boxes
from optimi_lab.problem import BlackBox, Relation, Space

__all__ = [
    'FALLBACK_PROPOSALS',
    'FEASIBLE_PROBABILITY',
    'GP_CALIBRATED_ROWS',
    'GP_SD_INFLATION',
    'SURROGATE_STEERING_SEED_OFFSET',
    'Selection',
    'feasible_front',
    'predictive',
    'probability_of_feasibility',
    'proposal_accuracy',
    'sd_inflation',
    'surrogate_candidates',
    'surrogate_proposals',
    'surrogate_selection',
]

#: Below :data:`GP_CALIBRATED_ROWS` the GP's predictive sd is widened by this factor (1.96 / 1.45, the
#: z its 95 % interval actually covered at n=96) before it feeds a probability of feasibility.
GP_SD_INFLATION: Final[float] = 1.4
GP_CALIBRATED_ROWS: Final[int] = 200
#: A candidate is proposed only when its probability of meeting every bound is at least this ...
FEASIBLE_PROBABILITY: Final[float] = 0.5
#: ... else the most probably feasible this many are, so a round still learns where feasibility is.
FALLBACK_PROPOSALS: Final[int] = 8
#: The candidate batch is seeded at the run seed PLUS this: under the run's own seed it would BE the
#: initial sample, where the model's error is smallest by construction and every point already solved.
SURROGATE_STEERING_SEED_OFFSET: Final[int] = 1

_GP: Final = 'gaussian_process'


@dataclass(frozen=True)
class Selection:
    """The proposed natural *points* and what the model predicted at them."""

    points: tuple[tuple[float, ...], ...]
    mean: np.ndarray
    sd: np.ndarray | None
    pof: np.ndarray | None


def sd_inflation(training_rows: int) -> float:
    """The factor the GP's predictive sd is widened by at this training size."""
    return GP_SD_INFLATION if training_rows < GP_CALIBRATED_ROWS else 1.0


def predictive(model: ScoredModel, drawn: np.ndarray, *, training_rows: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Predicted mean and, for the GP, the calibrated observation sd (latent variance plus noise)."""
    if model.key != _GP:
        return np.asarray(model.fitted.predict(drawn).mean, dtype=float), None
    predicted = model.fitted.predict(drawn, variance=True)
    noise = np.asarray(model.fitted.noise, dtype=float)[None, :]  # type: ignore[attr-defined]
    sd = np.sqrt(np.asarray(predicted.variance, dtype=float) + noise)
    return np.asarray(predicted.mean, dtype=float), sd * sd_inflation(training_rows)


def probability_of_feasibility(mean: np.ndarray, sd: np.ndarray, constraints: Sequence[BlackBox]) -> np.ndarray:
    """Per row, the probability every bound holds, the columns (in *constraints* order) independent normals."""
    bounds = [ColumnBound(i, c.relation is Relation.AT_LEAST, c.bound) for i, c in enumerate(constraints)]
    return feasibility_probability(mean, sd, bounds)


def feasible_front(minimizing: np.ndarray, pof: np.ndarray) -> np.ndarray:
    """The predicted front among the likely-feasible candidates, else the most probably feasible."""
    likely = np.flatnonzero(pof >= FEASIBLE_PROBABILITY)
    if likely.size == 0:
        return np.argsort(-pof, kind='stable')[:FALLBACK_PROPOSALS]
    return likely[np.asarray(non_dominated_sorting(minimizing[likely])).ravel()]


def proposal_accuracy(
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    span: np.ndarray,
    sd: np.ndarray | None = None,
    pof: np.ndarray | None = None,
    feasible: np.ndarray | None = None,
) -> dict:
    """On-policy calibration: range-normalised RMSE, 95 % coverage, feasible share, forecast and Brier."""
    error = predicted - actual
    scale = np.where(span > 0, span, 1.0)
    observed = None if feasible is None else np.asarray(feasible, dtype=float)
    return {
        'n': len(actual),
        'nrmse': float(np.mean(np.sqrt(np.mean(error**2, axis=0)) / scale)),
        'coverage95': None if sd is None else float(np.mean(np.abs(error) <= 1.96 * sd)),
        'feasible_share': None if observed is None else float(observed.mean()),
        'pof_mean': None if pof is None else float(np.mean(pof)),
        'brier': None if pof is None or observed is None else float(np.mean((pof - observed) ** 2)),
    }


def surrogate_candidates(space: Space, *, candidates: int, seed: int) -> np.ndarray:
    """The natural batch the model ranks: the library's sampler at *seed* + the steering offset."""
    return space.natural_sample(SampleSpec(n_samples=candidates, seed=seed + SURROGATE_STEERING_SEED_OFFSET))


def surrogate_selection(fit: SurrogateFit, model: ScoredModel, *, candidates: int, seed: int) -> Selection:
    """The distinct candidates at *model*'s predicted (likely-feasible) front, with its predictions."""
    problem = fit.problem
    drawn = surrogate_candidates(problem.space, candidates=candidates, seed=seed)
    mean, sd = predictive(model, drawn, training_rows=len(fit.results))
    k = len(problem.objectives)
    minimizing = np.column_stack([[o.minimized(v) for v in mean[:, i]] for i, o in enumerate(problem.objectives)])
    pof = None
    if fit.feasibility and sd is not None:
        pof = probability_of_feasibility(mean[:, k:], sd[:, k:], black_boxes(problem))
        front = feasible_front(minimizing, pof)
    else:
        front = np.asarray(non_dominated_sorting(minimizing)).ravel()
    chosen: dict[tuple[float, ...], int] = {}
    for index in front:
        chosen.setdefault(tuple(float(value) for value in drawn[int(index)]), int(index))
    at = list(chosen.values())
    return Selection(tuple(chosen), mean[at], None if sd is None else sd[at], None if pof is None else pof[at])


def surrogate_proposals(fit: SurrogateFit, model: ScoredModel, *, candidates: int, seed: int) -> tuple:
    """Only the points of :func:`surrogate_selection`."""
    return surrogate_selection(fit, model, candidates=candidates, seed=seed).points
