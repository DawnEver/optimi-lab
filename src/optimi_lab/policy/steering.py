"""Surrogate steering: a fitted model chooses WHICH points are worth solving, never what they are worth.

:func:`fit_surrogate` fits a pool of registry models to a run's own fully computed results and keeps
each member's cross-validated scores. :func:`steer` lets the best member RANK a fresh candidate batch
and then has every chosen point evaluated for real; what crosses from the model is design POINTS (the
indices of its predicted front), so no predicted number can become a result.

The gate is the score just measured: a pool whose best cross-validated ``r2`` is at or below
:data:`SURROGATE_STEERING_FLOOR_R2` -- no better than predicting the mean on held-out rows -- proposes
nothing, and :class:`Steering` says which member scored what against which floor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.oracle import Result
from optimi_lab.policy.proposals import Selection, proposal_accuracy, surrogate_selection
from optimi_lab.policy.routing import natural
from optimi_lab.policy.surrogate_fit import ScoredModel, SurrogateFit, black_boxes, targets

__all__ = ['SURROGATE_STEERING_FLOOR_R2', 'Evaluation', 'Steering', 'best_model', 'steer', 'steering_refusal']

#: The cross-validated ``r2`` a member must score strictly above to choose a point: 0 is the mean.
SURROGATE_STEERING_FLOOR_R2: Final[float] = 0.0


@dataclass(frozen=True)
class Steering:
    """What a steering phase did, INCLUDING nothing: exactly one of *best* (who passed) and *refusal* (why not)."""

    best: tuple[str, float] | None
    refusal: str | None
    proposed: int = 0
    solved: int = 0
    served: int = 0
    added: int = 0
    #: The model's accuracy on the points it proposed, once solved (:func:`proposal_accuracy`).
    accuracy: dict | None = None

    def __post_init__(self) -> None:
        if (self.refusal is None) == (self.best is None):
            msg = (
                f'a steering record says either WHY the model was not consulted or WHICH member passed the gate, '
                f'never both and never neither -- this one carries refusal={self.refusal!r} and best={self.best!r}'
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class Evaluation:
    """What evaluating proposals returned: their *results* (proposal order), fresh *solved*, reused *served*."""

    results: tuple[Result, ...]
    solved: int
    served: int


def best_model(models: Sequence[ScoredModel]) -> tuple[str, float, ScoredModel] | None:
    """The member with the best cross-validated ``r2``; a member that reported none is skipped."""
    best: tuple[str, float, ScoredModel] | None = None
    for member in models:
        measured = member.scores.get('r2')
        if measured is not None and (best is None or float(measured) > best[1]):
            best = (member.key, float(measured), member)
    return best


def steering_refusal(models: Sequence[ScoredModel]) -> str | None:
    """WHY this pool may not choose a design point, or None when it may."""
    best = best_model(models)
    if best is None:
        return (
            'no pool member reported a cross-validated r2, and the r2 is what says whether a fit beats '
            'predicting the mean -- a pool with no score has no order to offer'
        )
    name, score, _ = best
    if score > SURROGATE_STEERING_FLOOR_R2:
        return None
    return (
        f'no pool member scored above the r2 floor {SURROGATE_STEERING_FLOOR_R2:g} (best: {name} r2 '
        f'{score:.6g}). An r2 at or below the floor is a model no better than predicting the mean on the '
        f'rows it did not train on, so a preference taken from it would be arbitrary. No point was chosen '
        f"and no solve was spent on the model's opinion -- the front above is the run's own"
    )


def steer(
    fit: SurrogateFit | None,
    known: Sequence[Result],
    evaluate: Callable[[np.ndarray], Evaluation],
    *,
    candidates: int,
    seed: int,
) -> Steering:
    """Gate *fit*, let its best member choose points, have *evaluate* solve every one, and report it.

    *known* are the run's results before the round; ``added`` counts proposals that were not among them.
    """
    if fit is None:
        return Steering(
            best=None,
            refusal='no model was fitted to consult, so no point was chosen and no solve was spent '
            '(the surrogate line above names why nothing could be fitted)',
        )
    refusal = steering_refusal(fit.models)
    if refusal is not None:
        return Steering(best=None, refusal=refusal)
    best = best_model(fit.models)
    assert best is not None, 'a passed steering gate must name the model whose measured score passed'
    selection = surrogate_selection(fit, best[2], candidates=candidates, seed=seed)
    evaluated = evaluate(np.asarray(selection.points, dtype=float).reshape(len(selection.points), -1))
    points = {natural(fit.problem, r.action) for r in known}
    return Steering(
        best=(best[0], best[1]),
        refusal=None,
        proposed=len(selection.points),
        solved=evaluated.solved,
        served=evaluated.served,
        added=sum(natural(fit.problem, r.action) not in points for r in evaluated.results),
        accuracy=_accuracy(fit, selection, evaluated.results),
    )


def _accuracy(fit: SurrogateFit, selection: Selection, results: Sequence[Result]) -> dict | None:
    """:func:`proposal_accuracy` over the proposals that came back OK, or None if none did."""
    by_point = {natural(fit.problem, r.action): r for r in results}
    kept = [i for i, p in enumerate(selection.points) if p in by_point and by_point[p].cell.status is Status.OK]
    if not kept or not fit.span:
        return None
    solved = [by_point[selection.points[i]] for i in kept]
    k = len(fit.span)
    actual = np.asarray([targets(fit.problem, r, fit.feasibility) for r in solved], dtype=float)
    feasible = np.asarray([r.violation(fit.problem) == 0.0 for r in solved]) if black_boxes(fit.problem) else None
    return proposal_accuracy(
        selection.mean[kept][:, :k],
        actual,
        span=np.asarray(fit.span),
        sd=None if selection.sd is None else selection.sd[kept][:, :k],
        pof=None if selection.pof is None else selection.pof[kept],
        feasible=feasible,
    )
