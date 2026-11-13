"""What a study declares: fixed or sampled designs, bounded feedback, or an explicit search.

Sampling constructs inputs; it is not a kind of study. A fixed design records readings with no
invented objectives; only a :class:`SearchStrategy` runs an optimizer. The class NAMES are recorded
(``study_strategy`` events), so renaming one is a record-format change.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, prod
from types import MappingProxyType
from typing import Final

import numpy as np
from lab_commons.outcomes import Identity

from optimi_lab.core import SampleSpec
from optimi_lab.core.errors import refuse
from optimi_lab.operator import Risk
from optimi_lab.policy import STEERING_MODELS, RouteRule, Steering, SurrogateFit, VariationRule
from optimi_lab.solver.build import UNDRIVEN
from optimi_lab.solver.decomposition import moead
from optimi_lab.solver.evolutionary import mode, nsga2, nsga3
from optimi_lab.solver.requires import SOLVERS
from optimi_lab.solver.seeding import SEEDING, Seeding, require_seedable
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
class _FixedDesignFields:
    points: tuple[tuple[float, ...], ...]
    shape: tuple[int, ...]
    groups: tuple[str, ...]
    order: str


@dataclass(frozen=True, init=False)
class FixedDesign(_FixedDesignFields):
    """Ordered independent natural points, optionally with the exact sweep grid (*shape*, *order*) and *groups*."""

    def __init__(
        self,
        points: Sequence[Sequence[float]],
        shape: Sequence[int] = (),
        groups: Sequence[str] = (),
        order: str = 'C',
    ) -> None:
        points = tuple(tuple(float(c) for c in point) for point in points)
        shape, groups = tuple(shape), tuple(groups)
        if not points:
            refuse('a fixed design must declare at least one point')
        if len(set(points)) != len(points):
            refuse(
                'duplicate design points need an explicit replicate identity; implicit experimental replication is not supported'
            )
        if shape and (any(size < 1 for size in shape) or prod(shape) != len(points)):
            refuse('sweep shape must describe every ordered point exactly')
        if groups and len(groups) != len(points):
            refuse('sweep groups must name one group per ordered point')
        if order not in ('C', 'F'):
            refuse('sweep order must be C or F')
        _FixedDesignFields.__init__(self, points, shape, groups, order)


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
    """An explicit optimization intent over the problem the codec binds.

    *algorithm* names any registered solver an oracle drives (``optimi_lab.solver.SOLVERS`` but
    :data:`~optimi_lab.solver.seeding.UNDRIVEN`), or is a population factory. *settings* are the solver's
    own; *risk* and *scenarios* the robust declaration over the problem's uncertainty. *solves* stops the
    search at that many oracle evaluations (a scenario is one) in place of *generations*; surrogate
    feedback evaluates outside that budget, so the two are refused together. A solver that draws
    its own first batch is seeded by COUNT from *initial*, which must then be an unanchored Latin-hypercube
    sample (:func:`~optimi_lab.solver.seeding.require_seedable`).
    """

    initial: FixedDesign | SampleDesign
    algorithm: str | Callable = 'nsga2'
    generations: int = 1
    seed: int = 0
    surrogate: SurrogateStrategy | None = None
    report_fit: bool = False
    settings: Mapping[str, object] = field(default_factory=dict)
    risk: Risk | None = None
    scenarios: int | None = None
    solves: int | None = None

    def __post_init__(self) -> None:
        if self.generations < 0 or (self.generations == 0 and self.surrogate is None):
            refuse('optimization needs positive generations or explicit surrogate feedback')
        if self.solves is not None and self.solves < 1:
            refuse(f'a solve budget needs solves >= 1, got {self.solves}')
        if self.scenarios is not None and self.scenarios < 1:
            refuse(f'scenarios={self.scenarios!r} is a count of shared scenario draws of at least 1')
        if self.solves is not None and self.surrogate is not None:
            refuse('a solve budget caps the search; surrogate feedback evaluates outside it')
        if callable(self.algorithm):
            return
        if self.algorithm not in SOLVERS or self.algorithm in UNDRIVEN:
            refuse(f'unknown algorithm {self.algorithm!r}; choose {sorted(set(SOLVERS) - set(UNDRIVEN))}')
        if self.algorithm in SEEDING:
            first = self.initial.spec if isinstance(self.initial, SampleDesign) else np.array(self.initial.points)
            require_seedable(self.algorithm, first)

    @property
    def draws_own(self) -> bool:
        """Whether the named solver draws its own first batch, so *initial* is read as a count only."""
        if callable(self.algorithm) or self.algorithm not in SEEDING:
            return False
        seeding = SEEDING[self.algorithm]
        one_start = isinstance(self.initial, FixedDesign) and len(self.initial.points) == 1
        return seeding.draws_own and not (one_start and seeding in (Seeding.START, Seeding.SLOPES))


@dataclass(frozen=True)
class _FidelityFields:
    evaluator: Publisher
    evaluation_identity: Mapping[str, object]
    cost: float


@dataclass(frozen=True, init=False)
class FidelityEvaluation(_FidelityFields):
    """One more evaluator build: its publisher, its nonempty physical identity and comparable cost."""

    def __init__(self, evaluator: Publisher, evaluation_identity: Mapping[str, object], cost: float) -> None:
        if not evaluation_identity:
            refuse('a fidelity evaluator needs its explicit nonempty physical evaluation_identity')
        if not isfinite(cost) or cost <= 0:
            refuse('a fidelity evaluation cost must be finite and positive')
        _FidelityFields.__init__(self, evaluator, canonical_subject(evaluation_identity), cost)


@dataclass(frozen=True)
class _PolicyFields:
    router: RouteRule | None
    fidelities: tuple[FidelityEvaluation, ...]
    primary_cost: float | None
    variation: VariationRule | None


@dataclass(frozen=True, init=False)
class EvaluationPolicy(_PolicyFields):
    """Admission and fidelity (*router*) and variation-rate (*variation*) control, independent of any evaluator name.

    *fidelities* with no *router* leave each point's level to the solver (``Seeding.LEVELS``); the search
    refuses a solver that does not pick one, and a router beside one that does (``engine._refuse_choosers``).
    """

    def __init__(
        self,
        router: RouteRule | None = None,
        fidelities: Sequence[FidelityEvaluation] = (),
        primary_cost: float | None = None,
        variation: VariationRule | None = None,
    ) -> None:
        fidelities = tuple(fidelities)
        if any(not isinstance(item, FidelityEvaluation) for item in fidelities):
            msg = 'fidelities must be explicit FidelityEvaluation declarations'
            raise TypeError(msg)
        if fidelities and primary_cost is None:
            refuse('primary_cost must declare the primary evaluator comparable cost alongside fidelities')
        if primary_cost is not None and (not fidelities or not isfinite(primary_cost) or primary_cost <= 0):
            refuse('primary_cost needs additional fidelity declarations and must be finite and positive')
        _PolicyFields.__init__(self, router, fidelities, primary_cost, variation)
