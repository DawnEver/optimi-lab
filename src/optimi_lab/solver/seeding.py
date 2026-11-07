"""How each registered solver takes its FIRST batch, and the guards a driver applies before building one.

:data:`SOLVERS` (``solver.requires``) says what a solver NEEDS; this table says the one fact it does not --
how the first batch is made -- because a declared initial design a solver silently replaces is a
declaration that lies:

* ``POPULATION`` -- the first generation is handed in as unit-cube rows (``initial=``) or drawn by a
  sample rule (``spec=``); a declared point list is honoured point for point.
* ``MODEL`` -- a Gaussian-process solver draws its own maximin Latin hypercube of ``initial`` points.
* ``LEVELS`` -- a multi-fidelity schedule draws its own per-level design over the declared Fidelity.
* ``SELF`` -- CMA-ES (and IPOP restarts) draws every generation from its own distribution.
* ``START`` -- a local search walks from ONE start (a one-row initial design, else the cube's centre).
* ``SLOPES`` -- a gradient method walks from one start as ``START`` does, asking values and slopes.
* ``STARTS`` / ``SCAN`` -- a chain of ``START`` searches from Sobol starts, or one per scalarisation weight.

A :data:`SEQUENTIAL` solver asks a few points per round, so its run is bounded by EVALUATIONS
(:func:`batch_bound`), not by ``generations + 1`` rounds. The algebraic solvers (``lp``, ``milp``, ``qp``,
``cp``, ``dynamic_programming``, ``benders``, ``admm``) read coefficients, not a first batch, and have no row.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np

from optimi_lab.core import SampleKind, SampleSpec, refuse
from optimi_lab.problem import Embedding, Problem, needs_embedding
from optimi_lab.solver.gradient import METHODS

__all__ = ['SEEDING', 'SEQUENTIAL', 'Seeding', 'batch_bound', 'require_seedable', 'search_embedding']


class Seeding(StrEnum):
    """How a solver takes its first batch -- see the module docstring."""

    POPULATION = 'population'
    MODEL = 'model'
    LEVELS = 'levels'
    SELF = 'self'
    START = 'start'
    STARTS = 'starts'
    SCAN = 'scan'
    SLOPES = 'slopes'

    @property
    def draws_own(self) -> bool:
        """Whether the solver draws its own first batch, so a declared SAMPLE is read as a count only."""
        return self is not Seeding.POPULATION


_P, _M, _L, _S, _START = Seeding.POPULATION, Seeding.MODEL, Seeding.LEVELS, Seeding.SELF, Seeding.START

#: Every solver that takes a first batch -> how it takes it.
SEEDING: Final[Mapping[str, Seeding]] = MappingProxyType({
    **dict.fromkeys(
        ('ga', 'map_elites', 'mode', 'moead', 'mopso', 'nsga2', 'nsga3', 'prescreened', 'rnsga3', 'rvea'), _P
    ),
    **dict.fromkeys(('sms_emoa', 'spea2', 'unsga3'), _P),
    **dict.fromkeys(('bo', 'ehvi', 'nehvi', 'parego', 'turbo'), _M),
    **dict.fromkeys(('asha', 'hyperband', 'mfbo'), _L),
    **dict.fromkeys(('cmaes', 'ipop_cmaes'), _S),
    **dict.fromkeys(('bobyqa', 'mads', 'model_trust_region', 'nelder_mead'), _START),
    'multistart': Seeding.STARTS,
    'scalarisation_scan': Seeding.SCAN,
    **dict.fromkeys(METHODS, Seeding.SLOPES),
})

#: Solvers that ask a few points per round: their run is bounded by evaluations, not generation rounds.
SEQUENTIAL: Final = frozenset({
    'asha',
    'bobyqa',
    'hyperband',
    'mads',
    'model_trust_region',
    'multistart',
    'nelder_mead',
    'scalarisation_scan',
    *METHODS,
})

#: The sample rules a MODEL solver's own maximin Latin hypercube honours.
_LATIN: Final = frozenset({SampleKind.LATIN_HYPERCUBE, SampleKind.MAXIMIN_LATIN_HYPERCUBE})


def _seeding(name: str) -> Seeding:
    if name not in SEEDING:
        refuse(f'solver {name!r} takes no first batch; the seeded solvers are {sorted(SEEDING)}')
    return SEEDING[name]


def batch_bound(name: str, population: int, generations: int) -> int:
    """The ask/tell rounds a search of *name* may take: ``generations + 1``, or its evaluations if sequential."""
    _seeding(name)
    return population * (generations + 1) if name in SEQUENTIAL else generations + 1


def require_seedable(name: str, initial: np.ndarray | SampleSpec | None) -> None:
    """Refuse an initial design that solver *name*, drawing its own first batch, would silently replace.

    *initial* is the declared first batch: unit-cube rows, a sample rule, or None (the solver's own).
    A ``START``/``SLOPES`` solver also takes exactly one row: its start.
    """
    seeding = _seeding(name)
    if seeding is Seeding.POPULATION or initial is None:
        return
    if seeding in (Seeding.START, Seeding.SLOPES) and isinstance(initial, np.ndarray) and len(initial) == 1:
        return
    if not isinstance(initial, SampleSpec) or initial.kind not in _LATIN or initial.anchor is not None:
        refuse(
            f'{name} draws its own first batch ({seeding.value}) and learns only from what it asked; declare the '
            f'initial design as an unanchored Latin-hypercube sample of n points, not {initial!r}'
        )


def search_embedding(problem: Problem) -> Embedding | None:
    """The embedding a solver reads a conditional *problem* through; None when its space needs none."""
    return Embedding(problem.space) if needs_embedding(problem.space) else None
