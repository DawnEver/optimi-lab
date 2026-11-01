"""L8 policy: the batch budget, steered variation, routing with screening, and the decided-only replay."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import SUBMITTED, Cell, Event, Reason, Record, Status

from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.policy import Batches, DecidedReplay, RouteChoice, RoutingPolicy, ScreeningOracle, VariationPolicy
from optimi_lab.problem import Objective, Problem, Real, Space

PROBLEM = Problem(Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))), (Objective('f'), Objective('g')))


class Toy:
    """A one-evaluator screening oracle: f = x + y, g = 1 - x y."""

    capability = Capability(outputs=('f', 'g'), version='toy')
    evaluators = 1

    def __init__(self) -> None:
        self.published: list[int] = []

    def answer(self, action: Action) -> Result:
        x, y = action.design['x'], action.design['y']
        return Result(action, Cell(Status.OK, values={'f': x + y, 'g': 1 - x * y}))

    async def __call__(self, action: Action) -> Result:
        return self.answer(action)

    def screened(self, action: Action) -> Result:
        return Result(action, Cell(Status.SCREENED, reason=Reason('screened', 'the policy spent no solve')))

    def routed(self, action: Action, index: int) -> Action:
        return action

    def publish_batch(self, actions, on_result) -> None:
        self.published.append(len(actions))
        for action in actions:
            on_result(self.answer(action))


def _action(x: float, y: float) -> Action:
    return Action.of(PROBLEM, Toy.capability, {'x': x, 'y': y})


class Scripted:
    """Asks the scripted batches in order and keeps what it was told."""

    def __init__(self, batches: Sequence[Sequence[Action]]) -> None:
        self.batches = [list(b) for b in batches]
        self.told: list[list[Result]] = []

    def ask(self, capacity: int) -> list[Action]:
        return self.batches.pop(0) if self.batches else []

    def tell(self, results) -> None:
        self.told.append(list(results))

    @property
    def done(self) -> bool:
        return not self.batches


def test_the_screened_choice_is_minus_one_and_the_toy_is_a_screening_oracle() -> None:
    assert RouteChoice.SCREENED == -1
    assert isinstance(Toy(), ScreeningOracle)


def test_the_batch_budget_counts_tells_not_fresh_evaluations() -> None:
    inner = Scripted([[_action(0.1, 0.1)]] * 5)
    bounded = Batches(inner, 2)
    assert not bounded.done and bounded.last_batch_complete is None
    bounded.tell([Toy().screened(_action(0.1, 0.1))])
    assert bounded.last_batch_complete is False and not bounded.done
    bounded.tell([Toy().answer(_action(0.1, 0.1))])
    assert bounded.last_batch_complete is True and bounded.done


def test_steered_variation_updates_its_rates_for_the_declared_generations_only() -> None:
    seen: list[int] = []

    def rates(results):
        seen.append(len(results))
        return {'mutation_rate': 0.0, 'crossover_rate': 0.0}

    policy = VariationPolicy(lambda: (1, 2, 3), rates, 2, {'eta_mutation': 15.0})
    population = np.array([[0.2, 0.3], [0.6, 0.7]])
    rng = np.random.default_rng(0)
    for _ in range(3):
        children = policy(population, 2, rng, values=np.zeros((2, 2)), violation=np.zeros(2))
    assert seen == [3, 3]
    assert policy.settings == {'eta_mutation': 15.0, 'mutation_rate': 0.0, 'crossover_rate': 0.0}
    np.testing.assert_allclose(np.sort(children, axis=0), np.sort(population, axis=0))


def test_the_router_sees_observed_told_order_and_a_screened_action_is_never_dispatched(tmp_path: Path) -> None:
    record = Record(tmp_path / 'r.jsonl')
    first = [_action(0.1, 0.2), _action(0.3, 0.4)]
    second = [_action(0.5, 0.6), _action(0.7, 0.8), _action(0.5, 0.6)]
    inner = Scripted([first, second])
    calls: list[tuple[np.ndarray, tuple[float, ...]]] = []

    def router(points, results):
        calls.append((points, tuple(r.cell.values['f'] for r in results)))
        return np.array([0, RouteChoice.SCREENED])

    toy = Toy()
    policy = RoutingPolicy(inner, toy, PROBLEM, record, router)
    asked = policy.ask(10)
    assert [a.identity.key for a in asked] == [a.identity.key for a in first]
    policy.tell([toy.answer(a) for a in reversed(asked)])
    asked = policy.ask(10)
    np.testing.assert_allclose(calls[0][0], [[0.5, 0.6], [0.7, 0.8]])
    assert calls[0][1] == pytest.approx((0.3, 0.7))
    assert [a.identity.key for a in asked] == [second[0].identity.key, second[0].identity.key]
    policy.tell([toy.answer(asked[0])])
    told = inner.told[-1]
    assert [r.cell.status for r in told] == [Status.OK, Status.SCREENED, Status.OK]
    assert policy.batch_ends == [2, 4]
    assert [r.cell.status for r in policy.results()] == [Status.OK, Status.OK, Status.OK, Status.SCREENED]


def test_a_router_answer_outside_the_declared_evaluators_is_refused(tmp_path: Path) -> None:
    inner = Scripted([[_action(0.1, 0.2)], [_action(0.3, 0.4)]])
    toy = Toy()
    policy = RoutingPolicy(inner, toy, PROBLEM, Record(tmp_path / 'r.jsonl'), lambda _p, _r: np.array([1]))
    policy.tell([toy.answer(a) for a in policy.ask(5)])
    with pytest.raises(Refusal, match='router'):
        policy.ask(5)


def test_the_decided_replay_hides_screened_and_retryable_outcomes_but_writes_through(tmp_path: Path) -> None:
    record = Record(tmp_path / 'r.jsonl')
    replay = DecidedReplay(record)
    replay.submit('a')
    replay.outcome('a', Cell(Status.SCREENED, reason=Reason('screened')))
    replay.outcome('b', Cell(Status.TIMEOUT, reason=Reason('timeout')))
    replay.outcome('c', Cell(Status.OK, values={'f': 1.0}))
    replay.append(Event('note', 'n', data={'k': 1}))
    kinds = [(e.kind, e.key) for e in replay.replay()]
    assert kinds == [(SUBMITTED, 'a'), ('outcome', 'c'), ('note', 'n')]
    assert len(list(record.replay())) == 5
    assert replay.attempts('a') == 1 and set(replay.latest()) == {'a', 'b', 'c'} and set(replay.decided()) == {'c'}
