"""The population solver base, and the evolutionary generation three of the five algorithms share.

A population solver implements :class:`optimi_lab.solver.Solver`. Its first generation is drawn by
:func:`optimi_lab.core.sampling.sample` over the unit cube of the problem's space -- THE SAMPLE SIZE
IS THE POPULATION SIZE, one number -- and every later generation is built when the whole of the
current one has been told, in any order and in any number of parts.

Each algorithm works in the unit cube and in the MINIMISED, normalised objective frame
(:meth:`optimi_lab.problem.Objective.minimized`), and ranks by Deb's rule on the total scaled
violation (:meth:`optimi_lab.oracle.Result.violation`). A result that is not ``OK`` is DROPPED from
the generation, never scored: the population stays ``pop_size`` wide while that many points produce
values and is as wide as the complete points otherwise, because a quota filled by duplicating a point
would misstate how much of the search space a run covered.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core import SampleSpec, refuse, sample
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import Categorical, Problem

__all__ = ['EvolutionarySolver', 'PopulationSolver']


class PopulationSolver(ABC):
    """A solver that keeps a population and builds the next generation once the current one is told.

    ``spec`` draws the first generation (its row count is ``pop_size``); ``seed`` seeds every later
    draw -- two runs with one seed and one sequence of results are one run, which is what lets the
    run layer resume by replay.
    """

    def __init__(
        self, problem: Problem, capability: Capability, *, spec: SampleSpec | None = None, seed: int = 0
    ) -> None:
        conditional = [v.name for v in problem.space if v.active_if is not None]
        if conditional:
            refuse(f'population solvers need every variable active; {conditional} are conditional (active_if)')
        unembedded = [v.name for v in problem.space if isinstance(v, Categorical) and v.embedding is None]
        if unembedded:
            refuse(f'categoricals {unembedded} have no metric; apply optimi_lab.oracle.embed to the space first')
        self._problem = problem
        self._capability = capability
        self._width = problem.space.width
        self._rng = np.random.default_rng(seed)
        first = sample(self._width, spec)
        self._pop_size = first.shape[0]
        self._generation = self._actions(first)
        self._asked = 0
        self._told: dict[str, Result] = {}

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
        """A population solver never concludes by itself; the run's budget stops it."""
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
        kept = [r for r, ok in zip(results, mask, strict=True) if ok]
        inputs = np.array([self._problem.space.encode(r.action.design) for r in kept])
        values = np.array([[o.minimized(r.cell.values[o.name]) for o in self._problem.objectives] for r in kept])
        violation = np.array([r.violation(self._problem) for r in kept])
        return inputs, values, violation, mask

    @abstractmethod
    def _admit(self, inputs: np.ndarray, values: np.ndarray, violation: np.ndarray, mask: np.ndarray) -> None:
        """Fold one complete generation into this solver's state, seeding it on the first call."""

    @abstractmethod
    def _propose(self) -> np.ndarray:
        """The next generation in the unit cube, shape ``(n_points, width)``, with at least one point."""


class EvolutionarySolver(PopulationSolver):
    """A generation of an evolutionary algorithm: vary a population, then survive from the union.

    One loop, three operator sets. ``variation`` turns a population into offspring under
    ``variation_settings`` -- its keyword rates, fixed for the run -- ``selection`` keeps the survivors from parents and offspring together, and ``parent_selection`` picks the
    mating pool from the population's own values -- None mates every individual with the one
    selected before it, which is what a differential evolution wants. The count asked of a
    selector is the population size where that many rows exist and every row available otherwise,
    so a run whose evaluations fail shrinks rather than padding itself with a duplicate.
    """

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        variation: Callable[..., np.ndarray],
        variation_settings: Mapping[str, float] | None = None,
        selection: Callable[..., np.ndarray],
        parent_selection: Callable[..., np.ndarray] | None = None,
        spec: SampleSpec | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(problem, capability, spec=spec, seed=seed)
        self._variation = variation
        settings = dict(variation_settings or {})
        accepted = set(inspect.signature(variation).parameters) - {'population', 'width', 'rng'}
        unknown = sorted(set(settings) - accepted)
        if unknown:
            refuse(f'{unknown} are not settings of this variation; the valid settings are {sorted(accepted)}')
        self._variation_settings = settings
        self._selection = selection
        self._parent_selection = parent_selection
        self._population: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._violation: np.ndarray | None = None

    def _admit(self, inputs: np.ndarray, values: np.ndarray, violation: np.ndarray, mask: np.ndarray) -> None:
        if self._population is None:
            self._population, self._values, self._violation = inputs, values, violation
            return
        combined_inputs = np.vstack([self._population, inputs])
        combined_values = np.vstack([self._values, values])
        combined_violation = np.concatenate([self._violation, violation])
        keep = self._selection(combined_values, min(self._pop_size, len(combined_values)), violation=combined_violation)
        self._population, self._values = combined_inputs[keep], combined_values[keep]
        self._violation = combined_violation[keep]

    def _propose(self) -> np.ndarray:
        population = self._population
        if self._parent_selection is not None:
            pool = self._parent_selection(
                self._values, min(self._pop_size, len(population)), self._rng, violation=self._violation
            )
            population = population[pool]
        return self._variation(population, self._width, self._rng, **self._variation_settings)
