"""MAP-Elites (Mouret and Clune, 2015): the best design found in every cell of a behaviour grid.

The behaviour of a design is a tuple of :class:`Descriptor` readings, each binned evenly on
``[lo, hi]`` (values outside fall in the end bins). A descriptor names a declared problem OUTPUT
(an objective or a constraint's output, read from the result) or a design VARIABLE (read from the
design) -- a result carrying an undeclared output is refused upstream, so there is no third source. A cell keeps
its elite by Deb's rule on one score (``scalarise`` for several objectives). Each generation draws
``batch`` elites uniformly and perturbs them with the iso+line operator (Vassiliades and Mouret,
2018): ``x_i + sigma_iso N(0, I) + sigma_line N(0, 1) (x_j - x_i)``, clipped to the unit cube.
The result is the grid itself (:attr:`MapElites.elites`), not a front.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.population import PopulationSolver, build, require_one_score

__all__ = ['Descriptor', 'MapElites', 'descriptors_of', 'map_elites']


@dataclass(frozen=True, slots=True)
class Descriptor:
    """One behaviour axis: oracle output *name*, binned *bins* times evenly over ``[lo, hi]`` (its unit)."""

    name: str
    lo: float
    hi: float
    bins: int

    def __post_init__(self) -> None:
        if not self.hi > self.lo or self.bins < 1:
            refuse(f'descriptor {self.name!r} needs hi > lo and bins >= 1, got {self.lo}, {self.hi}, {self.bins}')

    def bin(self, value: float) -> int:
        return int(np.clip(np.floor((value - self.lo) / (self.hi - self.lo) * self.bins), 0, self.bins - 1))


class MapElites(PopulationSolver):
    """MAP-Elites over *descriptors*; ``batch`` children per generation (None: the first sample's size)."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        descriptors: tuple[Descriptor, ...],
        batch: int | None = None,
        sigma_iso: float = 0.01,
        sigma_line: float = 0.2,
        **base: object,
    ) -> None:
        require_one_score(problem, base.get('scalarise'), 'map_elites')
        known = set(problem.outputs) | set(problem.space.names)
        missing = [d.name for d in descriptors if d.name not in known]
        if not descriptors or missing:
            refuse(f'descriptors name problem outputs or variables; {missing or "none"} are not among {sorted(known)}')
        super().__init__(problem, capability, **base)
        self._descriptors = descriptors
        self._batch = self._pop_size if batch is None else int(batch)
        self._sigma_iso, self._sigma_line = sigma_iso, sigma_line
        self.elites: dict[tuple[int, ...], tuple[np.ndarray, float, float]] = {}

    def _admit(self, inputs: np.ndarray, values: np.ndarray, violation: np.ndarray, mask: np.ndarray) -> None:
        del mask
        for point, value, shortfall, result in zip(inputs, values[:, 0], violation, self._results, strict=True):
            reading = {**result.action.design, **result.cell.values}
            cell = tuple(d.bin(float(reading[d.name])) for d in self._descriptors)
            held = self.elites.get(cell)
            if held is None or (shortfall, value) < (held[2], held[1]):
                self.elites[cell] = (point, float(value), float(shortfall))

    def _propose(self) -> np.ndarray:
        parents = np.array([elite[0] for elite in self.elites.values()])
        first = parents[self._rng.integers(0, len(parents), self._batch)]
        second = parents[self._rng.integers(0, len(parents), self._batch)]
        children = (
            first
            + self._sigma_iso * self._rng.standard_normal(first.shape)
            + self._sigma_line * self._rng.standard_normal((self._batch, 1)) * (second - first)
        )
        return np.clip(children, 0.0, 1.0)


def map_elites(problem: Problem, capability: Capability, **settings: object) -> MapElites:
    """MAP-Elites; ``**settings`` are :class:`MapElites`'s (``descriptors`` required) and the base solver's."""
    return build(MapElites, problem, capability, 'map_elites', **settings)


def descriptors_of(problem: Problem, declared: Mapping[str, Sequence[float] | float]) -> tuple[Descriptor, ...]:
    """The archive axes declared as ``{name: [lo, hi, bins]}``, each in its name's unit.

    A name is a declared problem output or a design variable; a bounded variable may give ``[bins]``
    (or a bare ``bins``) alone and is binned over its own bounds. Anything else is refused naming what
    can be binned.
    """
    if not isinstance(declared, Mapping) or not declared:
        refuse('map_elites needs descriptors = {NAME: [lo, hi, bins]}, one per archive axis')
    axes = {variable.name: variable for variable in problem.space}
    out = []
    for name, row in declared.items():
        values = list(row) if isinstance(row, (list, tuple)) else [row]
        if name in axes and len(values) == 1 and hasattr(axes[name], 'lo'):
            values = [axes[name].lo, axes[name].hi, values[0]]
        if (name not in problem.outputs and name not in axes) or len(values) != 3:
            refuse(
                f'descriptor {name!r} = {row!r}: name a declared output {list(problem.outputs)} as [lo, hi, bins] '
                f'or a design variable {sorted(axes)} as [lo, hi, bins] or [bins]'
            )
        out.append(Descriptor(str(name), float(values[0]), float(values[1]), int(values[2])))
    return tuple(out)
