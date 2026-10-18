"""Compositions of solvers: a chain, scalarisation scans, multistart, restarts (IPOP), and warm starts.

Every composition here is a :class:`Chain` -- solvers run one after another, the next built (lazily)
when the current one concludes -- so a scan, a multistart and a restart schedule differ only in how
their solvers are made:

* :func:`scalarisation_scan` -- one single-score solver per weight vector, each minimising a
  scalarisation of the objectives (B0-02: the scalarisation's declared reach is what the scan can
  reach, so a weighted-sum scan of a concave front returns its ends only);
* :func:`multistart` -- one local solver per start point of a scrambled Sobol set;
* :func:`restarts` -- ``count`` solvers from one factory taking the restart index; IPOP-CMA-ES is
  restarts whose population doubles (:func:`ipop_cmaes`).

:func:`warm_start` turns an archive's complete results into the unit-cube rows a population solver
takes as ``initial=``: its constrained Pareto ranking, crowding broken, best first. In a run over the
same record a re-asked design is answered from that record rather than evaluated again.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence

import numpy as np

from optimi_lab.archive import Archive
from optimi_lab.core import SampleKind, SampleSpec, refuse, sample
from optimi_lab.operator import Scalarisation, crowding_select, tchebycheff
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import Problem
from optimi_lab.solver.cmaes import cmaes
from optimi_lab.solver.population import score_rows
from optimi_lab.solver.protocol import Solver

__all__ = ['Chain', 'ipop_cmaes', 'multistart', 'restarts', 'scalarisation_scan', 'warm_start']


class Chain:
    """Run *solvers* one after another; done when the last has concluded."""

    def __init__(self, solvers: Iterable[Solver]) -> None:
        self._pending: Iterator[Solver] = iter(solvers)
        self._current: Solver | None = next(self._pending, None)
        self.finished: list[Solver] = []

    def _advance(self) -> None:
        while self._current is not None and self._current.done:
            self.finished.append(self._current)
            self._current = next(self._pending, None)

    def ask(self, capacity: int) -> Sequence[Action]:
        self._advance()
        return [] if self._current is None else self._current.ask(capacity)

    def tell(self, results: Sequence[Result]) -> None:
        if self._current is None:
            refuse('the chain has concluded; nothing it asked is outstanding')
        self._current.tell(results)

    @property
    def done(self) -> bool:
        self._advance()
        return self._current is None


def scalarisation_scan(
    problem: Problem,
    capability: Capability,
    *,
    local: Callable[..., Solver],
    weights: np.ndarray,
    scalarisation: Scalarisation = tchebycheff,
    reference: np.ndarray | None = None,
    **settings: object,
) -> Chain:
    """One ``local(problem, capability, scalarise=..., **settings)`` per weight row, chained.

    ``reference`` is the scalarisation's reference point in the minimised normalised frame (None:
    the origin, the ideal of a frame normalised onto ``[0, 1]``).
    """
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    if weights.shape[1] != len(problem.objectives):
        refuse(f'one weight per objective: weights {weights.shape} for {len(problem.objectives)} objectives')
    point = np.zeros(weights.shape[1]) if reference is None else np.asarray(reference, dtype=float)

    def bound(row: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return lambda values: scalarisation(values, row, point)

    return Chain(local(problem, capability, scalarise=bound(row), **settings) for row in weights)


def multistart(
    problem: Problem,
    capability: Capability,
    *,
    local: Callable[..., Solver],
    starts: int,
    seed: int = 0,
    **settings: object,
) -> Chain:
    """``local`` from each of *starts* scrambled-Sobol points of the unit cube, chained."""
    points = sample(problem.space.width, SampleSpec(kind=SampleKind.SOBOL, n_samples=starts, seed=seed))
    return Chain(
        local(problem, capability, start=point, seed=seed + index, **settings) for index, point in enumerate(points)
    )


def restarts(factory: Callable[[int], Solver], count: int) -> Chain:
    """``factory(0)``, then ``factory(1)`` once it concludes, ... ``count`` solvers in all."""
    if count < 1:
        refuse(f'restarts needs count >= 1, got {count!r}')
    return Chain(factory(index) for index in range(count))


def ipop_cmaes(
    problem: Problem,
    capability: Capability,
    *,
    count: int = 9,
    population: int | None = None,
    factor: int = 2,
    seed: int = 0,
    **settings: object,
) -> Chain:
    """IPOP-CMA-ES (Auger and Hansen, 2005): each restart multiplies the population by *factor*."""
    base = population if population is not None else 4 + int(3 * np.log(problem.space.width))
    return restarts(lambda k: cmaes(problem, capability, population=base * factor**k, seed=seed + k, **settings), count)


def warm_start(archive: Archive, n: int) -> np.ndarray:
    """The unit-cube rows of the archive's *n* best complete designs (constrained front, then crowding)."""
    problem = archive.problem
    results = archive.complete()
    if not results:
        refuse('the archive holds no complete result to warm-start from')
    values, violation = score_rows(problem, None, results)
    keep = crowding_select(values, min(n, len(results)), violation=violation)
    return np.array([problem.space.encode(results[i].action.design) for i in keep])
