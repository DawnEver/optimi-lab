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
generation, never scored -- unless it is a decided one whose design breaks a constraint that reads the design
alone (:meth:`optimi_lab.oracle.Result.design_violation` > 0, the shape of a design refused before any solve):
that row is certainly infeasible and RANKED by that violation, so a box that is mostly refused is still walked
toward feasibility. It has no objective values; it carries the worst scored value of its generation per column
(zero when none scored), which Deb's rule never reads -- an infeasible row is ordered by violation alone.

A problem that declares :class:`~optimi_lab.problem.Uncertainty` is solved ROBUSTLY: ``scenarios``
draws of it are taken once from the seed and shared by every design (common random numbers, so two
designs differ by design and not by draw); each design is asked once per scenario
(:attr:`optimi_lab.oracle.Action.scenario`), and its row is the declared ``risk``
(:class:`optimi_lab.operator.Risk`, default the mean) of each scored column and of the violation
over those scenarios. A :class:`~optimi_lab.problem.Chance` constraint is estimated as the share of
scenarios that satisfy it; its violation is the shortfall from its probability. A design with any
scenario not ``OK`` is dropped whole.

A space with conditional (``active_if``) variables or unembedded categoricals is solved through the
declared ``embedding`` (:class:`optimi_lab.problem.Embedding`): every row an algorithm keeps is the
embedded row of a told design, and every row it proposes is decoded to a LEGAL design before it is
asked, so a variation may move a switched-off column or a one-hot block freely.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core import SampleSpec, refuse, sample
from optimi_lab.operator import Risk, discrete_mutation, lattice, snap
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import Embedding, Problem
from optimi_lab.solver.scenarios import Scenarios

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
        risk: Risk | None = None,
        scenarios: int | None = None,
        embedding: Embedding | None = None,
    ) -> None:
        self._embedding = Embedding.declared(problem.space, embedding)
        if initial is not None and spec is not None:
            refuse('pass spec= (draw the first generation) or initial= (hand it in), not both')
        self._problem = problem
        self._capability = capability
        self._width = self._embedding.width
        self._lattice = lattice(self._embedding.space)
        self._scalarise = scalarise
        self._rng = np.random.default_rng(seed)
        self._robust = Scenarios.of(problem, risk, scenarios, self._rng)
        first = self._first(spec) if initial is None else self._require_initial(initial)
        self._pop_size = first.shape[0]
        self._generation = self._actions(first)
        self._asked = 0
        self._told: dict[str, Result] = {}
        self._results: list[Result] = []
        self._seed: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[Result]]] = []  # seed rows not yet admitted
        self._seeded = False

    #: The fewest rows the algorithm's next generation can be bred from (two parents of a crossover).
    _breeding_minimum = 2

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
        embedding = self._embedding
        draws = (None,) if self._robust is None else self._robust.scenarios
        return [
            Action.of(self._problem, self._capability, embedding.decode(row), scenario=scenario)
            for row in unit
            for scenario in draws
        ]

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
            if self._seeded:
                self._admit(*self._arrays())
            unit = self._propose() if self._seeded else self._next_seed()
            self._generation, self._asked, self._told = self._actions(unit), 0, {}

    def _next_seed(self) -> np.ndarray:
        """Admit the seed once it holds ``_breeding_minimum`` rows (a tournament cannot draw from one), else top it up."""
        rows = self._arrays(empty_ok=True)
        if rows is not None:
            self._seed.append((*rows[:3], self._results))
        held = sum(len(batch[0]) for batch in self._seed)
        if held < min(self._breeding_minimum, self._pop_size):
            shortfall = max(self._pop_size, self._breeding_minimum) - held
            return sample(self._width, SampleSpec(n_samples=shortfall, seed=int(self._rng.integers(2**32))))
        inputs, values, violation = (np.concatenate([batch[k] for batch in self._seed]) for k in range(3))
        self._results = [result for batch in self._seed for result in batch[3]]
        self._seed, self._seeded = [], True
        self._admit(inputs, values, violation, np.ones(held, dtype=bool))
        return self._propose()

    @property
    def done(self) -> bool:
        """A population solver concludes only when an algorithm says so; the run's budget stops the rest."""
        return False

    def _arrays(self, *, empty_ok: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """``(inputs, values, violation, mask)`` of the complete rows in generation order; none: refused, or None."""
        results = [self._told[a.identity.key] for a in self._generation]
        if self._robust is not None:
            return self._robust_arrays(results, empty_ok=empty_ok)
        scored = np.array([r.cell.status is Status.OK for r in results])
        refused = np.array([
            not ok and r.cell.status.decided and r.design_violation(self._problem) > 0.0
            for r, ok in zip(results, scored, strict=True)
        ])
        mask = scored | refused
        if not mask.any() and empty_ok:
            return None
        if not mask.any():
            statuses = sorted({r.cell.status.name for r in results})
            refuse(
                f'none of the {len(results)} point(s) of this generation produced a value ({statuses}); '
                f'there is no population to build the next one from'
            )
        self._results = [r for r, kept in zip(results, mask, strict=True) if kept]
        inputs = np.array([self._embedding.encode(r.action.design) for r in self._results])
        ok = scored[mask]
        columns = 1 if self._scalarise is not None else len(self._problem.objectives)
        values, violation = np.zeros((len(ok), columns)), np.zeros(len(ok))
        if ok.any():
            values[ok], violation[ok] = score_rows(
                self._problem, self._scalarise, [r for r, is_ok in zip(self._results, ok, strict=True) if is_ok]
            )
            values[~ok] = values[ok].max(axis=0)
        violation[~ok] = [
            r.design_violation(self._problem) for r, is_ok in zip(self._results, ok, strict=True) if not is_ok
        ]
        return inputs, values, violation, mask

    def _robust_arrays(
        self, results: list[Result], *, empty_ok: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """``(inputs, values, violation, mask)`` per DESIGN: each column the declared risk over its scenarios."""
        robust = self._robust
        per = len(robust.scenarios)
        groups = [results[i : i + per] for i in range(0, len(results), per)]
        mask = np.array([all(r.cell.status is Status.OK for r in group) for group in groups])
        if not mask.any() and empty_ok:
            return None
        if not mask.any():
            refuse(
                f'none of the {len(groups)} design(s) of this generation produced a value in every one of its '
                f'{per} scenarios; there is no population to build the next one from'
            )
        kept = [group for group, ok in zip(groups, mask, strict=True) if ok]
        self._results = [group[0] for group in kept]
        inputs = np.array([self._embedding.encode(group[0].action.design) for group in kept])
        rows = [robust.row(*score_rows(robust.sampled, self._scalarise, group), group) for group in kept]
        return inputs, np.array([v for v, _ in rows]), np.array([c for _, c in rows]), mask

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
        breeding_minimum: int = 2,
        **base: object,
    ) -> None:
        self._breeding_minimum = int(breeding_minimum)
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
