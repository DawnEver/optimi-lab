"""Steered variation: a policy sets the SBX/PM rates, the engine's own operator makes the children.

:func:`build_steered_solver` composes an NSGA-II/III population solver whose variation is a
:class:`VariationPolicy`: it keeps the family's survival operator and replaces only the variation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from optimi_lab.core import SampleSpec, refuse
from optimi_lab.operator import Risk, crowding_select, reference_select, tournament_select
from optimi_lab.oracle import Capability, Result
from optimi_lab.problem import Problem
from optimi_lab.solver import EvolutionarySolver, nsga2, nsga3, search_embedding
from optimi_lab.solver.evolutionary import sbx_polynomial

__all__ = ['VariationPolicy', 'VariationRule', 'build_steered_solver']

#: ``variation(results)`` -> variation-rate updates for the next generation, or None to keep them.
VariationRule = Callable[[tuple[Result, ...]], Mapping[str, float] | None]


class VariationPolicy:
    """:func:`sbx_polynomial` whose rates *variation* may update before each of the first *generations* calls.

    *read* returns the results the variation is shown (the routing policy's told results); *settings* are
    the starting rates. A variation answering None leaves the rates as they are.
    """

    def __init__(
        self,
        read: Callable[[], tuple[Result, ...]],
        variation: VariationRule,
        generations: int,
        settings: Mapping[str, object],
    ) -> None:
        self.read, self.variation = read, variation
        self.remaining = generations
        self.settings = dict(settings)

    def __call__(
        self, population: np.ndarray, width: int, rng: np.random.Generator, *, values: np.ndarray, violation: np.ndarray
    ) -> np.ndarray:
        if self.remaining > 0:
            updates = self.variation(self.read())
            if updates is not None:
                self.settings.update(updates)
            self.remaining -= 1
        return sbx_polynomial(population, width, rng, values=values, violation=violation, **self.settings)


def build_steered_solver(
    factory: Callable[..., EvolutionarySolver],
    problem: Problem,
    capability: Capability,
    *,
    seed: int,
    population: int,
    initial: np.ndarray | None,
    variation: VariationPolicy,
    risk: Risk | None = None,
    scenarios: int | None = None,
) -> EvolutionarySolver:
    """An NSGA-II (*factory* ``nsga2``) or NSGA-III (``nsga3``) solver whose SBX/PM rates *variation* drives.

    *initial* is unit-cube rows (None draws *population* by LHS); *risk* and *scenarios* are a robust
    declaration, handed on as declared; a conditional space is read through its embedding.
    """
    if factory not in (nsga2, nsga3):
        refuse('SBX/PM variation control steers nsga2 or nsga3; the chosen solver has other operator settings')
    robust = {'risk': risk, 'scenarios': scenarios, 'embedding': search_embedding(problem)}
    return EvolutionarySolver(
        problem,
        capability,
        seed=seed,
        spec=SampleSpec(n_samples=population, seed=seed) if initial is None else None,
        initial=initial,
        variation=variation,
        selection=crowding_select if factory is nsga2 else reference_select,
        parent_selection=tournament_select,
        **{key: value for key, value in robust.items() if value is not None},
    )
