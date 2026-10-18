"""The population solver base, and the evolutionary generation most population algorithms share.

A population solver implements :class:`optimi_lab.solver.Solver`. Its first generation is drawn by
:func:`optimi_lab.core.sampling.sample` over the unit cube of the problem's space -- THE SAMPLE SIZE
IS THE POPULATION SIZE, one number -- unless ``initial`` hands it unit-cube rows (a warm start,
:func:`optimi_lab.solver.meta.warm_start`) or the algorithm draws its own (CMA-ES). Every later
generation is built when the whole of the current one has been told, in any order and in any number
of parts.

Each algorithm works in the unit cube and in the MINIMISED, normalised objective frame
(:meth:`optimi_lab.problem.Objective.minimized`), and ranks by Deb's rule on the total scaled
violation (:meth:`optimi_lab.oracle.Result.violation`). ``scalarise`` maps those rows to one score
per row, which is how a single-objective algorithm is pointed at several objectives (and how a
scalarisation scan builds its subproblems). A result that is not ``OK`` is DROPPED from the
generation, never scored.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core import SampleSpec, refuse, sample
from optimi_lab.operator import discrete_mutation, lattice, snap
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import Categorical, Problem

__all__ = [
    'LOOP_ARGUMENTS',
    'EvolutionarySolver',
    'PopulationSolver',
    'Scalarise',
    'build',
    'require_one_score',
    'score_rows',
    'settings_of',
]

#: Minimised objective rows ``(n, m)`` -> one score per row ``(n,)``.
Scalarise = Callable[[np.ndarray], np.ndarray]

#: Arguments every variation receives from the loop rather than from its settings.
LOOP_ARGUMENTS = frozenset({'population', 'width', 'rng', 'values', 'violation'})


def settings_of(owner: type) -> frozenset[str]:
    """Every keyword setting *owner* takes, its own and each base class's, read off their signatures."""
    names: set[str] = set()
    for cls in owner.__mro__[:-1]:
        parameters = inspect.signature(cls.__init__).parameters.values()
        names |= {p.name for p in parameters if p.kind is p.KEYWORD_ONLY}
    return frozenset(names)


def build[T](owner: type[T], problem: Problem, capability: Capability, what: str, **settings: object) -> T:
    """``owner(problem, capability, **settings)``, refusing a setting it does not take and naming those it does."""
    accepted = settings_of(owner)
    unknown = sorted(set(settings) - accepted)
    if unknown:
        refuse(f'{unknown} are not settings of {what}; the valid settings are {sorted(accepted)}')
    return owner(problem, capability, **settings)


def require_one_score(problem: Problem, scalarise: Scalarise | None, what: str) -> None:
    """Refuse a single-score algorithm on several objectives unless a scalarisation is supplied."""
    if scalarise is None and len(problem.objectives) > 1:
        refuse(
            f'{what} minimises one score and the problem has {len(problem.objectives)} objectives; '
            f'pass scalarise= (an optimi_lab.operator scalarisation bound to its weights)'
        )


def score_rows(
    problem: Problem, scalarise: Scalarise | None, results: Sequence[Result]
) -> tuple[np.ndarray, np.ndarray]:
    """``(values, violation)`` of OK *results*: minimised rows (or one scalarised column), violation per row."""
    values = np.array([[o.minimized(r.cell.values[o.name]) for o in problem.objectives] for r in results])
    values = values.reshape(len(results), len(problem.objectives))
    if scalarise is not None:
        values = np.asarray(scalarise(values), dtype=float).reshape(-1, 1)
    return values, np.array([r.violation(problem) for r in results])


class PopulationSolver(ABC):
    """A solver that keeps a population and builds the next generation once the current one is told.

    ``seed`` seeds every draw -- two runs with one seed and one sequence of results are one run,
    which is what lets the run layer resume by replay.
    """

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        spec: SampleSpec | None = None,
        initial: np.ndarray | None = None,
        scalarise: Scalarise | None = None,
        seed: int = 0,
    ) -> None:
        conditional = [v.name for v in problem.space if v.active_if is not None]
        if conditional:
            refuse(f'population solvers need every variable active; {conditional} are conditional (active_if)')
        unembedded = [v.name for v in problem.space if isinstance(v, Categorical) and v.embedding is None]
        if unembedded:
            refuse(f'categoricals {unembedded} have no metric; apply optimi_lab.oracle.embed to the space first')
        if initial is not None and spec is not None:
            refuse('pass spec= (draw the first generation) or initial= (hand it in), not both')
        self._problem = problem
        self._capability = capability
        self._width = problem.space.width
        self._lattice = lattice(problem.space)
        self._scalarise = scalarise
        self._rng = np.random.default_rng(seed)
        first = self._first(spec) if initial is None else self._require_initial(initial)
        self._pop_size = first.shape[0]
        self._generation = self._actions(first)
        self._asked = 0
        self._told: dict[str, Result] = {}
        self._results: list[Result] = []

    def _first(self, spec: SampleSpec | None) -> np.ndarray:
        """The first generation's unit-cube rows; an algorithm with its own first draw overrides this."""
        return sample(self._width, spec)

    def _require_initial(self, initial: np.ndarray) -> np.ndarray:
        initial = np.asarray(initial, dtype=float)
        if initial.ndim != 2 or initial.shape[1] != self._width or len(initial) < 1:
            refuse(f'initial must be (n >= 1, {self._width}) unit-cube rows, got shape {initial.shape}')
        if np.any((initial < 0.0) | (initial > 1.0)):
            refuse('initial rows are unit-cube points; some coordinates lie outside [0, 1]')
        return initial

    def _actions(self, unit: np.ndarray) -> list[Action]:
        space = self._problem.space
        return [Action.of(self._problem, self._capability, space.decode(row)) for row in unit]

    def ask(self, capacity: int) -> Sequence[Action]:
        """The next unasked actions of the current generation, at most *capacity*."""
        start, self._asked = self._asked, min(self._asked + capacity, len(self._generation))
        return self._generation[start : self._asked]

    def tell(self, results: Sequence[Result]) -> None:
        """Results of the current generation, in any order; the next one is built once all are in."""
        keys = {a.identity.key for a in self._generation}
        for result in results:
            if result.action.identity.key not in keys:
                refuse('a result was told that this generation did not ask for; tell each result to its own solver')
            self._told[result.action.identity.key] = result
        if len(self._told) == len(keys):
            self._admit(*self._arrays())
            self._generation, self._asked, self._told = self._actions(self._propose()), 0, {}

    @property
    def done(self) -> bool:
        """A population solver concludes only when an algorithm says so; the run's budget stops the rest."""
        return False

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(inputs, values, violation, mask)`` of the complete rows, in generation order."""
        results = [self._told[a.identity.key] for a in self._generation]
        mask = np.array([r.cell.status is Status.OK for r in results])
        if not mask.any():
            statuses = sorted({r.cell.status.name for r in results})
            refuse(
                f'none of the {len(results)} point(s) of this generation produced a value ({statuses}); '
                f'there is no population to build the next one from'
            )
        self._results = [r for r, ok in zip(results, mask, strict=True) if ok]
        inputs = np.array([self._problem.space.encode(r.action.design) for r in self._results])
        values, violation = score_rows(self._problem, self._scalarise, self._results)
        return inputs, values, violation, mask

    @abstractmethod
    def _admit(self, inputs: np.ndarray, values: np.ndarray, violation: np.ndarray, mask: np.ndarray) -> None:
        """Fold one complete generation into this solver's state, seeding it on the first call."""

    @abstractmethod
    def _propose(self) -> np.ndarray:
        """The next generation in the unit cube, shape ``(n_points, width)``, with at least one point."""


class EvolutionarySolver(PopulationSolver):
    """A (mu + lambda) generation: vary the population, then survive from parents and offspring together.

    ``variation(population, width, rng, values=..., violation=..., **variation_settings)`` makes
    offspring, of which the first ``n_offspring`` are kept (None keeps them all; 1 is a steady
    state); ``selection(values, n, violation=...)`` keeps the survivors; ``parent_selection`` picks
    the mating pool (None: every individual mates). Discrete variables are snapped onto their levels
    and mutated on their own ladder at ``discrete_mutation_rate`` (None: one block per genome).
    """

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        variation: Callable[..., np.ndarray],
        variation_settings: Mapping[str, object] | None = None,
        selection: Callable[..., np.ndarray],
        parent_selection: Callable[..., np.ndarray] | None = None,
        n_offspring: int | None = None,
        discrete_mutation_rate: float | None = None,
        **base: object,
    ) -> None:
        super().__init__(problem, capability, **base)
        settings = dict(variation_settings or {})
        accepted = set(inspect.signature(variation).parameters) - LOOP_ARGUMENTS
        unknown = sorted(set(settings) - accepted)
        if unknown:
            refuse(f'{unknown} are not settings of this variation; the valid settings are {sorted(accepted)}')
        if n_offspring is not None and n_offspring < 1:
            refuse(f'n_offspring must be at least 1, got {n_offspring!r}')
        self._variation, self._variation_settings = variation, settings
        self._selection, self._parent_selection = selection, parent_selection
        self._n_offspring = n_offspring
        blocks = len(self._lattice.blocks)
        default_rate = 1.0 / blocks if blocks else 0.0
        self._discrete_rate = default_rate if discrete_mutation_rate is None else discrete_mutation_rate
        self._population: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._violation: np.ndarray | None = None

    def _survive(self, values: np.ndarray, n: int, violation: np.ndarray) -> np.ndarray:
        return self._selection(values, n, violation=violation)

    def _admit(self, inputs: np.ndarray, values: np.ndarray, violation: np.ndarray, mask: np.ndarray) -> None:
        if self._population is None:
            self._population, self._values, self._violation = inputs, values, violation
            return
        inputs = np.vstack([self._population, inputs])
        values = np.vstack([self._values, values])
        violation = np.concatenate([self._violation, violation])
        keep = self._survive(values, min(self._pop_size, len(values)), violation)
        self._population, self._values, self._violation = inputs[keep], values[keep], violation[keep]

    def _propose(self) -> np.ndarray:
        population, values, violation = self._population, self._values, self._violation
        if self._parent_selection is not None:
            pool = self._parent_selection(values, min(self._pop_size, len(population)), self._rng, violation=violation)
            population, values, violation = population[pool], values[pool], violation[pool]
        children = self._variation(
            population, self._width, self._rng, values=values, violation=violation, **self._variation_settings
        )
        children = children[: self._n_offspring] if self._n_offspring is not None else children
        if self._lattice.blocks:
            children = discrete_mutation(
                snap(children, self._lattice), self._lattice, mutation_rate=self._discrete_rate, rng=self._rng
            )
        return children
