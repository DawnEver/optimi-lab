"""What a study declares: fixed or sampled designs, bounded feedback, or an explicit search.

Sampling constructs inputs; it is not a kind of study. A fixed design records readings with no
invented objectives; only a :class:`SearchStrategy` runs an optimizer. The class NAMES are recorded
(``study_strategy`` events), so renaming one is a record-format change.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite, prod
from types import MappingProxyType
from typing import Final

from lab_commons.outcomes import Identity

from optimi_lab.core import SampleSpec
from optimi_lab.core.errors import refuse
from optimi_lab.policy import STEERING_MODELS, RouteRule, Steer, Steering, SurrogateFit
from optimi_lab.solver.decomposition import moead
from optimi_lab.solver.evolutionary import mode, nsga2, nsga3
from optimi_lab.solver.swarm import mopso
from optimi_lab.study.protocols import Publisher

__all__ = [
    'ALGORITHMS',
    'ROUND_SEED_STRIDE',
    'EvaluationPolicy',
    'FeedbackDesign',
    'FidelityEvaluation',
    'FixedDesign',
    'SampleDesign',
    'SearchStrategy',
    'SurrogateRound',
    'SurrogateStrategy',
    'algorithm_factory',
    'canonical_subject',
]

_FACTORIES: Final = {'mode': mode, 'moead': moead, 'mopso': mopso, 'nsga2': nsga2, 'nsga3': nsga3}
#: The population solvers a study names.
ALGORITHMS: Final = tuple(_FACTORIES)
#: Surrogate round *i*, retry *r* steers at ``seed + i * ROUND_SEED_STRIDE + r``.
ROUND_SEED_STRIDE: Final = 1000


def algorithm_factory(name: str | Callable) -> Callable:
    """The solver factory *name* declares; a callable is already one and is returned as given."""
    if callable(name):
        return name
    if name not in _FACTORIES:
        refuse(f'unknown algorithm {name!r}; choose {ALGORITHMS}')
    return _FACTORIES[name]


def canonical_subject(value: Mapping[str, object]) -> Mapping[str, object]:
    """*value* through its canonical identity encoding (exact plain data), recursively immutable."""
    return _freeze(json.loads(Identity.of(value).canonical)['subject'])


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({name: _freeze(item) for name, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class FixedDesign:
    """Ordered independent natural points, optionally with the exact sweep grid (*shape*, *order*) and *groups*."""

    points: tuple[tuple[float, ...], ...]
    shape: tuple[int, ...] = ()
    groups: tuple[str, ...] = ()
    order: str = 'C'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'points', tuple(tuple(float(c) for c in point) for point in self.points))
        object.__setattr__(self, 'shape', tuple(self.shape))
        object.__setattr__(self, 'groups', tuple(self.groups))
        if not self.points:
            refuse('a fixed design must declare at least one point')
        if len(set(self.points)) != len(self.points):
            refuse(
                'duplicate design points need an explicit replicate identity; implicit experimental replication is not supported'
            )
        if self.shape and (any(size < 1 for size in self.shape) or prod(self.shape) != len(self.points)):
            refuse('sweep shape must describe every ordered point exactly')
        if self.groups and len(self.groups) != len(self.points):
            refuse('sweep groups must name one group per ordered point')
        if self.order not in ('C', 'F'):
            refuse('sweep order must be C or F')


@dataclass(frozen=True)
class SampleDesign:
    """A sampling operator whose natural output becomes a fixed design."""

    spec: SampleSpec


@dataclass(frozen=True)
class FeedbackDesign:
    """Sequential DoE, which need not optimize anything: *propose* reads the rows filed so far; None is STOP."""

    initial: FixedDesign | SampleDesign
    propose: Callable[[tuple[object, ...]], FixedDesign | None]
    max_batches: int

    def __post_init__(self) -> None:
        if self.max_batches < 1:
            refuse('feedback needs a finite positive max_batches including the initial batch')


@dataclass(frozen=True)
class SurrogateRound:
    """Round *index* (0 is the initial fit): rows taught, the search state after it, its fit and steering."""

    index: int
    training_rows: int
    result: object
    fit: SurrogateFit | None
    steering: Steering | None


@dataclass(frozen=True)
class SurrogateStrategy:
    """Generic fit / propose / evaluate feedback; *model* is a key of ``STEERING_MODELS``."""

    rounds: int = 1
    candidates: int = 1
    pool: tuple[str, ...] | None = None
    on_round: Callable[[SurrogateRound], None] | None = field(default=None, compare=False)
    model: str = 'pool'

    def __post_init__(self) -> None:
        if self.rounds < 0 or self.candidates < 1:
            refuse('surrogate feedback needs nonnegative rounds and positive candidates')
        if self.model not in STEERING_MODELS:
            refuse(f'unknown steering model {self.model!r}; declared: {sorted(STEERING_MODELS)}')


@dataclass(frozen=True)
class SearchStrategy:
    """An explicit optimization intent over the problem the codec binds."""

    initial: FixedDesign | SampleDesign
    algorithm: str | Callable = 'nsga2'
    generations: int = 1
    seed: int = 0
    surrogate: SurrogateStrategy | None = None
    report_fit: bool = False

    def __post_init__(self) -> None:
        if self.generations < 0 or (self.generations == 0 and self.surrogate is None):
            refuse('optimization needs positive generations or explicit surrogate feedback')
        algorithm_factory(self.algorithm)


@dataclass(frozen=True)
class FidelityEvaluation:
    """One more evaluator build: its publisher, its nonempty physical identity and comparable cost."""

    evaluator: Publisher
    evaluation_identity: Mapping[str, object]
    cost: float

    def __post_init__(self) -> None:
        if not self.evaluation_identity:
            refuse('a fidelity evaluator needs its explicit nonempty physical evaluation_identity')
        if not isfinite(self.cost) or self.cost <= 0:
            refuse('a fidelity evaluation cost must be finite and positive')
        object.__setattr__(self, 'evaluation_identity', canonical_subject(self.evaluation_identity))


@dataclass(frozen=True)
class EvaluationPolicy:
    """Admission and fidelity (*router*) and variation-rate (*variation*) control, independent of any evaluator name."""

    router: RouteRule | None = None
    fidelities: tuple[FidelityEvaluation, ...] = ()
    primary_cost: float | None = None
    variation: Steer | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'fidelities', tuple(self.fidelities))
        if any(not isinstance(item, FidelityEvaluation) for item in self.fidelities):
            msg = 'fidelities must be explicit FidelityEvaluation declarations'
            raise TypeError(msg)
        if self.fidelities and self.router is None:
            refuse('additional fidelity evaluators need an explicit selection policy')
        if self.fidelities and self.primary_cost is None:
            refuse('primary_cost must declare the primary evaluator comparable cost alongside fidelities')
        if self.primary_cost is not None and (
            not self.fidelities or not isfinite(self.primary_cost) or self.primary_cost <= 0
        ):
            refuse('primary_cost needs additional fidelity declarations and must be finite and positive')
