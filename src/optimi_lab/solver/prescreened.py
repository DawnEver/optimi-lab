"""D67 -- the surrogate-prescreened evolutionary algorithm: breed many, model them, evaluate the promising.

An NSGA-II generation (SBX + polynomial mutation, tournament mating, fronts-and-crowding survival)
breeds ``oversample`` times the offspring it will ask. A surrogate fitted on EVERY told OK row so
far -- the minimised (or scalarised) objectives, and the total scaled violation as one more output
when the problem has constraints -- predicts each child, and the solver's own survival operator
picks the children to ask from the predicted rows:

* objectives -- the optimistic bound ``mean - kappa * std`` per objective (``kappa = 0`` reads the
  mean alone and takes a surrogate that gives no variance);
* violation -- the predicted total violation, floored at 0.

Only the picked children are evaluated, and only evaluated rows enter the population and the
model: a prediction ranks a child, never replaces its evaluation (B0-08). With one objective it is a
prescreened GA. The rebuilt 0.3 idea, now any L6 surrogate (:mod:`optimi_lab.model`).
"""

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.model import GaussianProcess, Give, Surrogate
from optimi_lab.operator import crowding_select, tournament_select
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.evolutionary import evolutionary, sbx_polynomial
from optimi_lab.solver.population import EvolutionarySolver

__all__ = ['Prescreened', 'prescreened']


class Prescreened(EvolutionarySolver):
    """An evolutionary generation whose offspring are prescreened by a surrogate before they are asked."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        surrogate: Surrogate | None = None,
        oversample: int = 4,
        kappa: float = 1.0,
        **base: object,
    ) -> None:
        surrogate = surrogate or GaussianProcess(restarts=0)
        if oversample < 1 or not kappa >= 0.0:
            refuse(f'oversample is >= 1 and kappa >= 0, got {oversample!r} and {kappa!r}')
        if kappa > 0.0 and Give.VARIANCE not in surrogate.gives:
            refuse(f'kappa = {kappa} reads a variance and {type(surrogate).__name__} gives none; pass kappa=0')
        super().__init__(problem, capability, **base)
        self._surrogate, self._oversample, self._kappa = surrogate, oversample, kappa
        self._seen_x: list[np.ndarray] = []
        self._seen_y: list[np.ndarray] = []

    def _admit(self, inputs: np.ndarray, values: np.ndarray, violation: np.ndarray, mask: np.ndarray) -> None:
        self._seen_x.extend(inputs)
        self._seen_y.extend(np.column_stack([values, violation]) if self._problem.constraints else values)
        super()._admit(inputs, values, violation, mask)

    def _propose(self) -> np.ndarray:
        bred = [super(Prescreened, self)._propose() for _ in range(self._oversample)]
        keep = len(bred[0])
        children = np.vstack(bred)
        fitted = self._surrogate.fit(np.array(self._seen_x), np.array(self._seen_y))
        p = fitted.predict(children, variance=self._kappa > 0.0)
        bound = p.mean if p.variance is None else p.mean - self._kappa * np.sqrt(p.variance)
        if self._problem.constraints:
            bound, violation = bound[:, :-1], np.maximum(p.mean[:, -1], 0.0)
        else:
            violation = np.zeros(len(children))
        chosen = self._selection(bound, keep, violation=violation)
        return children[chosen]


def prescreened(problem: Problem, capability: Capability, **settings: object) -> EvolutionarySolver:
    """The prescreened NSGA-II/GA; ``**settings`` are SBX's rates, the population base's and :class:`Prescreened`'s."""
    return evolutionary(
        problem,
        capability,
        'prescreened',
        variation=sbx_polynomial,
        selection=crowding_select,
        parent_selection=tournament_select,
        settings=settings,
        owner=Prescreened,
        own=frozenset({'surrogate', 'oversample', 'kappa'}),
    )
