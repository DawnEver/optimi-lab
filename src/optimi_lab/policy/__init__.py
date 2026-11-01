"""L8 policy: what a caller decides around a solver -- budgets, variation rates, routing, replay, steering.

* :class:`Batches` -- a generation budget counted in told batches.
* :class:`VariationPolicy` -- SBX/PM rates a :data:`Steer` updates per generation.
* :class:`RoutingPolicy` -- a :data:`RouteRule` screens or routes each action of a :class:`ScreeningOracle`.
* :class:`DecidedReplay` -- the run replays decided outcomes only (never SCREENED or retryable ones).
* surrogate steering -- a pool is fitted (:mod:`~optimi_lab.policy.surrogate_fit`), proposes
  (:mod:`~optimi_lab.policy.proposals`) and is gated (:mod:`~optimi_lab.policy.steering`): it chooses
  which points are worth solving, never what they are worth.
"""

from optimi_lab.policy.batches import Batches
from optimi_lab.policy.proposals import (
    FALLBACK_PROPOSALS,
    FEASIBLE_PROBABILITY,
    GP_CALIBRATED_ROWS,
    GP_SD_INFLATION,
    SURROGATE_STEERING_SEED_OFFSET,
    Selection,
    feasible_front,
    predictive,
    probability_of_feasibility,
    proposal_accuracy,
    sd_inflation,
    surrogate_candidates,
    surrogate_proposals,
    surrogate_selection,
)
from optimi_lab.policy.replay import DecidedReplay
from optimi_lab.policy.routing import RouteChoice, RouteRule, RoutingPolicy, ScreeningOracle, Steer, natural
from optimi_lab.policy.steering import (
    SURROGATE_STEERING_FLOOR_R2,
    Evaluated,
    Steering,
    best_model,
    steer_round,
    steering_refusal,
)
from optimi_lab.policy.surrogate_fit import (
    GP_MAX_ROWS,
    MIN_SURROGATE_POINTS,
    MODEL_SETTINGS,
    STEERING_MODELS,
    SURROGATE_POOL,
    ScoredModel,
    SurrogateFit,
    fit_surrogate,
)
from optimi_lab.policy.variation import VariationPolicy

__all__ = [
    'FALLBACK_PROPOSALS',
    'FEASIBLE_PROBABILITY',
    'GP_CALIBRATED_ROWS',
    'GP_MAX_ROWS',
    'GP_SD_INFLATION',
    'MIN_SURROGATE_POINTS',
    'MODEL_SETTINGS',
    'STEERING_MODELS',
    'SURROGATE_POOL',
    'SURROGATE_STEERING_FLOOR_R2',
    'SURROGATE_STEERING_SEED_OFFSET',
    'Batches',
    'DecidedReplay',
    'Evaluated',
    'RouteChoice',
    'RouteRule',
    'RoutingPolicy',
    'ScoredModel',
    'ScreeningOracle',
    'Selection',
    'Steer',
    'Steering',
    'SurrogateFit',
    'VariationPolicy',
    'best_model',
    'feasible_front',
    'fit_surrogate',
    'natural',
    'predictive',
    'probability_of_feasibility',
    'proposal_accuracy',
    'sd_inflation',
    'steer_round',
    'steering_refusal',
    'surrogate_candidates',
    'surrogate_proposals',
    'surrogate_selection',
]
