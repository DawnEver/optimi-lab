"""A search strategy end to end: the search (or the initial evaluation), then its surrogate rounds.

Round 0 fits the pool to what the search filed. Round *i* steers at ``seed + i * ROUND_SEED_STRIDE +
retry``, retrying a fresh candidate batch while the model proposes only points already solved; it
stops at a refusal or a round that neither solved nor added anything, and refits between rounds.
"""

from __future__ import annotations

import numpy as np
from lab_commons.outcomes import Evidence

from optimi_lab.core.errors import refuse
from optimi_lab.oracle import Action
from optimi_lab.policy import (
    GP_MAX_ROWS,
    STEERING_MODELS,
    Evaluated,
    Steering,
    SurrogateFit,
    fit_surrogate,
    steer_round,
)
from optimi_lab.study.engine import Searched, evaluate, opened, search
from optimi_lab.study.protocols import SearchBinding
from optimi_lab.study.strategy import (
    ROUND_SEED_STRIDE,
    EvaluationPolicy,
    SearchStrategy,
    SurrogateRound,
    SurrogateStrategy,
)

__all__ = ['searched_study']


def _fit(strategy: SurrogateStrategy | None, bound: SearchBinding, searched: Searched) -> SurrogateFit | None:
    results, evidence = searched.ordered_results(), searched.evidence
    if strategy is None or strategy.model == 'pool':
        pool = {} if strategy is None or strategy.pool is None else {'pool': strategy.pool}
        return fit_surrogate(bound.problem, results, evidence=evidence, **pool)
    # The GP also models the constraints and is taught at most the most recent GP_MAX_ROWS rows.
    return fit_surrogate(
        bound.problem,
        results,
        evidence=evidence,
        pool=strategy.pool or STEERING_MODELS[strategy.model],
        feasibility=True,
        max_rows=GP_MAX_ROWS,
    )


def searched_study(
    bound: SearchBinding, strategy: SearchStrategy, policy: EvaluationPolicy, initial: np.ndarray, evidence: Evidence
) -> tuple[Searched, SurrogateFit | None, Steering | None, tuple[SurrogateRound, ...]]:
    """Run *strategy* from the natural *initial* points on *bound*; the last fit, steering and the rounds."""
    problem, oracle, record = bound.problem, bound.oracle, bound.record
    if strategy.surrogate is not None and (policy.router is not None or policy.variation is not None):
        refuse('surrogate batch feedback cannot silently ignore admission/variation control')
    if strategy.generations:
        searched = search(
            problem,
            oracle,
            record,
            algorithm=strategy.algorithm,
            population=len(initial),
            generations=strategy.generations,
            seed=strategy.seed,
            initial=initial,
            router=policy.router,
            variation=policy.variation,
        )
    else:
        searched = evaluate(problem, oracle, initial, record, evidence=evidence)
    feedback = strategy.surrogate
    if feedback is None:
        return searched, _fit(None, bound, searched) if strategy.report_fit else None, None, ()

    def evaluated(points: np.ndarray) -> Evaluated:
        after = evaluate(problem, oracle, points, record, evidence=searched.evidence)
        names = problem.space.names
        actions = (Action.of(problem, oracle.capability, dict(zip(names, map(float, p), strict=True))) for p in points)
        results = tuple(r for a in actions if (r := after.archive.result(a.identity)) is not None)
        return Evaluated(results, after.solved, after.served)

    fit = _fit(feedback, bound, searched)
    if feedback.on_round is not None:
        feedback.on_round(SurrogateRound(0, 0, searched, fit, None))
    rounds: list[SurrogateRound] = []
    steering = None
    for index in range(1, feedback.rounds + 1):
        training = len(fit.results) if fit is not None else 0
        for retry in range(min(ROUND_SEED_STRIDE, max(8, 2 * len(searched.ordered_results()) // feedback.candidates))):
            known = searched.ordered_results()
            seed = strategy.seed + index * ROUND_SEED_STRIDE + retry
            steering = steer_round(fit, known, evaluated, candidates=feedback.candidates, seed=seed)
            if steering.refusal is None:
                searched = Searched(
                    opened(problem, oracle, record),
                    searched.search_refusal,
                    searched.evidence,
                    searched.batch_ends,
                    searched.execution,
                )
            if steering.refusal is not None or steering.added or steering.solved:
                break
        done = SurrogateRound(index, training, searched, fit, steering)
        rounds.append(done)
        if feedback.on_round is not None:
            feedback.on_round(done)
        assert steering is not None, 'each nonempty surrogate round attempts at least one candidate batch'
        if steering.refusal is not None or not (steering.added or steering.solved):
            break
        if index < feedback.rounds:
            fit = _fit(feedback, bound, searched)
    return searched, fit, steering, tuple(rounds)
