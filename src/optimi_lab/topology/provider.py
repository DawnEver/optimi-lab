"""The PHYSICS-PROVIDER seam, and the problem the optimizer sees through it.

A provider is the physics and nothing else (:class:`PhysicsProvider`): its ``domain``, its declared
``responses`` and ``evaluate(physical) -> {name: (value, d value / d physical)}`` at ONE physical density
field. It never sees a design variable, a filter, a projection or an optimizer.

:class:`TopologyProblem` owns the rest. At design variables ``z`` it runs the :class:`~.density.DesignChain`,
asks the provider once per realization it needs, and assembles the
:class:`~optimi_lab.operator.Linearization` a gradient method reads:

* the ONE objective is minimised, normalised by its magnitude at the first evaluation. A ``robust``
  objective read on several realizations (eroded / nominal / dilated) is their KS aggregate
  (:func:`~.interpolation.ks`, sharpness *ks_rho*) -- a smooth worst case;
* each constraint ``value <= bound`` becomes ``(value - bound) / scale <= 0`` (``scale`` declared, else
  ``|bound|``, else 1), once per realization when ``robust``, else on the nominal one;
* every sensitivity is pulled back through projection, filter and symmetry ties to ``d / d z``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.operator import Linearization
from optimi_lab.topology.density import DesignChain, grayness
from optimi_lab.topology.domain import DesignDomain
from optimi_lab.topology.interpolation import ks

__all__ = ['Evaluation', 'PhysicsProvider', 'Response', 'Role', 'TopologyProblem']


class Role(StrEnum):
    """What the driver does with a response."""

    OBJECTIVE = 'objective'
    CONSTRAINT = 'constraint'


@dataclass(frozen=True, slots=True)
class Response:
    """A declared provider output: minimised (``OBJECTIVE``) or held ``<= bound`` (``CONSTRAINT``).

    ``robust``: read on every projected realization (else on the nominal one only); ``scale``: the
    constraint's normalisation, defaulting to ``|bound|`` (or 1 for a zero bound).
    """

    name: str
    role: Role
    bound: float | None = None
    scale: float | None = None
    robust: bool = True


class PhysicsProvider(Protocol):
    """The physics behind a topology problem; see the module docstring."""

    domain: DesignDomain
    responses: tuple[Response, ...]

    def evaluate(self, physical: np.ndarray) -> Mapping[str, tuple[float, np.ndarray]]: ...


@dataclass(frozen=True, slots=True, eq=False)
class Evaluation:
    """One evaluation: what the optimizer reads, plus the raw values a record keeps."""

    linearization: Linearization
    objective: float
    constraints: tuple[float, ...]
    physical: tuple[np.ndarray, ...]

    @property
    def grayness(self) -> float:
        """The grayness of the nominal realization."""
        return grayness(self.physical[len(self.physical) // 2])


class TopologyProblem:
    """*provider* seen through *chain*; see the module docstring."""

    def __init__(self, provider: PhysicsProvider, chain: DesignChain, *, ks_rho: float = 40.0) -> None:
        if chain.domain is not provider.domain:
            refuse("the chain's filter is built on another domain than the provider's")
        objectives = [r for r in provider.responses if r.role == Role.OBJECTIVE]
        if len(objectives) != 1:
            refuse(f'a topology problem has exactly one objective, got {[r.name for r in objectives]}')
        self.constraints = tuple(r for r in provider.responses if r.role == Role.CONSTRAINT)
        for r in self.constraints:
            if r.bound is None:
                refuse(f'constraint {r.name!r} needs a bound')
        self.provider, self.chain, self.ks_rho = provider, chain, float(ks_rho)
        self.objective = objectives[0]
        self._scale: float | None = None

    def _realizations(self, response: Response) -> range:
        return range(len(self.chain.etas)) if response.robust else range(self.chain.nominal, self.chain.nominal + 1)

    def evaluate(self, design: np.ndarray) -> Evaluation:
        """The :class:`Evaluation` at design variables *design*."""
        physical = self.chain.forward(design)
        needed = sorted({k for r in self.provider.responses for k in self._realizations(r)})
        answers = {k: self.provider.evaluate(physical[k]) for k in needed}
        n = self.chain.domain.n_cells

        def read(k: int, name: str) -> tuple[float, np.ndarray]:
            if name not in answers[k]:
                refuse(f'the provider declared {name!r} but did not return it; it returned {sorted(answers[k])}')
            value, slope = answers[k][name]
            slope = np.asarray(slope, dtype=float)
            if slope.shape != (n,):
                refuse(f'{name!r} carries a sensitivity of shape {slope.shape}; the domain has {n} cells')
            return float(value), slope

        def pull(k: int, slope: np.ndarray) -> np.ndarray:
            per = [None] * len(physical)
            per[k] = slope
            return self.chain.pull_back(per)

        reads = [read(k, self.objective.name) for k in self._realizations(self.objective)]
        values = np.array([v for v, _ in reads])
        if self._scale is None:
            self._scale = abs(float(values[len(values) // 2])) or 1.0
        scaled, weights = ks(values / self._scale, self.ks_rho) if len(reads) > 1 else (values[0] / self._scale, [1.0])
        raw = scaled * self._scale
        per = [None] * len(physical)
        for k, w, (_, slope) in zip(self._realizations(self.objective), weights, reads, strict=True):
            per[k] = w * slope
        df = self.chain.pull_back(per) / self._scale
        c, dc = [], []
        for r in self.constraints:
            scale = r.scale or abs(r.bound) or 1.0
            for k in self._realizations(r):
                value, slope = read(k, r.name)
                c.append((value - r.bound) / scale)
                dc.append(pull(k, slope) / scale)
        point = Linearization(design, [scaled], df[None, :], c=c, dc=np.array(dc).reshape(len(c), len(design)))
        return Evaluation(point, raw, tuple(c), physical)
