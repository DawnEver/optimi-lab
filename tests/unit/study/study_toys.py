"""Shared toys for the study tests: a screening oracle, a point publisher, an in-memory filing, a codec."""

from __future__ import annotations

from dataclasses import dataclass, field

from lab_commons.outcomes import Cell, Reason, Record, Status

from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import Objective, Problem, Real, Sense, Space
from optimi_lab.study import SearchBinding

SPACE = Space((Real('a', 0.0, 1.0), Real('b', 2.0, 4.0)))
PROBLEM = Problem(SPACE, (Objective('y'), Objective('z', Sense.MAXIMIZE)))
AXES = (
    {'addresses': [['x', 0]], 'hi': 1.0, 'kind': 'design', 'lo': 0.0, 'name': 'a', 'samples': None, 'unit': 'm'},
    {'addresses': [['y']], 'hi': 4.0, 'kind': 'design', 'lo': 2.0, 'name': 'b', 'samples': None, 'unit': '1'},
)
READINGS = ({'name': 'y', 'unit': 'V'},)


class Toy:
    """A one-evaluator screening oracle: y = a + b, z = a - b / 4; it counts the points it published."""

    capability = Capability(outputs=('y', 'z'), version='toy')
    evaluators = 1

    def __init__(self) -> None:
        self.published: list[tuple[float, float]] = []

    def answer(self, action: Action) -> Result:
        a, b = float(action.design['a']), float(action.design['b'])
        return Result(action, Cell(Status.OK, values={'y': a + b, 'z': a - b / 4}))

    async def __call__(self, action: Action) -> Result:
        return self.answer(action)

    def screened(self, action: Action) -> Result:
        return Result(action, Cell(Status.SCREENED, reason=Reason('screened', 'the policy spent no solve')))

    def routed(self, action: Action, index: int) -> Action:
        return action

    def publish_batch(self, actions, on_result) -> None:
        for action in actions:
            self.published.append((float(action.design['a']), float(action.design['b'])))
            on_result(self.answer(action))


@dataclass(frozen=True)
class Answer:
    point: tuple[float, ...]
    y: float


class Publisher:
    """Answers each point with y = a + b and remembers what it was asked."""

    def __init__(self) -> None:
        self.asked: list[tuple[float, ...]] = []

    def publish(self, points, on_result) -> None:
        for point in points:
            self.asked.append(point)
            on_result(Answer(point, point[0] + point[1]))


@dataclass
class Filing:
    """An in-memory point codec: the record keeps only the study_* events in these tests."""

    rows: dict = field(default_factory=dict)
    submitted: list = field(default_factory=list)

    def reusable(self, point) -> bool:
        return point in self.rows

    def submit(self, point) -> None:
        self.submitted.append(point)

    def file(self, answer) -> None:
        self.rows[answer.point] = answer

    def row(self, point):
        return self.rows.get(point)


class Codec:
    """One filing per study; a search runs the toy oracle over the raw record."""

    def __init__(self, oracle: Toy | None = None) -> None:
        self.oracle = oracle or Toy()
        self.filed = Filing()

    def filing(self, record: Record) -> Filing:
        return self.filed

    def searching(self, filing: Filing, record: Record) -> SearchBinding:
        return SearchBinding(PROBLEM, self.oracle, record)
