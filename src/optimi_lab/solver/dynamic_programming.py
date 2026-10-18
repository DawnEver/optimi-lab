"""D129 finite-horizon dynamic programming over a declared, finite state set (Bellman 1957).

    V_T(s) = terminal(s),    V_t(s) = min_a [ cost(t, s, a) + E V_{t+1}(next) ]

*transition* ``(t, s, a)`` returns the next state, or -- with ``stochastic=True`` -- a sequence of
``(probability, next)`` pairs whose probabilities sum to one. A next state outside *states* is
refused naming the set (a grid that does not close is a modelling error, never silently clipped).
A state with no action has ``V = inf`` and no policy entry; an infinite cost marks a forbidden move.

Backward induction tabulates ``values[t][s]`` and ``policy[t][s]``; :meth:`DPSolution.plan` rolls a
deterministic policy forward from a start. This is the shape of a drive-cycle energy-management DP:
time steps, a discretised state of charge, a finite set of controls.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.solver.linear_program import StructuredSolution
from optimi_lab.solver.smooth import Halt

__all__ = ['DPSolution', 'dynamic_programming']

#: Probabilities of one stochastic transition must sum to one within this (dimensionless).
_PROBABILITY_SUM = 1e-9


@dataclass(frozen=True, slots=True, eq=False)
class DPSolution:
    """``values[t][s]`` (``t = 0..T``) and ``policy[t][s]`` (``t < T``); see the module docstring."""

    values: tuple[dict[Hashable, float], ...]
    policy: tuple[dict[Hashable, Hashable], ...]
    transition: Callable
    stochastic: bool

    def plan(self, start: Hashable) -> tuple[tuple[Hashable, ...], tuple[Hashable, ...], StructuredSolution]:
        """The visited states, the actions and the report of rolling the policy forward from *start*."""
        if self.stochastic:
            refuse('a stochastic policy has no single plan; read policy[t][s] at each realised state')
        value = self.values[0].get(start)
        if value is None:
            refuse(f'{start!r} is not a declared state')
        if not np.isfinite(value):
            return (start,), (), StructuredSolution(Halt.INFEASIBLE, np.zeros(0), np.inf, 0, reason='no feasible path')
        states, actions = [start], []
        for t, policy in enumerate(self.policy):
            action = policy[states[-1]]
            actions.append(action)
            states.append(self.transition(t, states[-1], action))
        report = StructuredSolution(Halt.CONVERGED, np.zeros(0), float(value), len(self.policy), bound=float(value))
        return tuple(states), tuple(actions), report


def dynamic_programming(
    horizon: int,
    states: Iterable[Hashable],
    actions: Callable[[int, Hashable], Iterable[Hashable]],
    transition: Callable[[int, Hashable, Hashable], Hashable | Sequence[tuple[float, Hashable]]],
    cost: Callable[[int, Hashable, Hashable], float],
    *,
    terminal: Callable[[Hashable], float] = lambda _: 0.0,
    stochastic: bool = False,
) -> DPSolution:
    """Backward induction; see the module docstring."""
    declared = tuple(states)
    if not declared or horizon < 1:
        refuse(f'a DP needs states and a horizon >= 1; got {len(declared)} state(s), horizon {horizon}')
    known = frozenset(declared)
    values: list[dict[Hashable, float]] = [{s: float(terminal(s)) for s in declared}]
    policy: list[dict[Hashable, Hashable]] = []

    def outcomes(t: int, s: Hashable, a: Hashable) -> Sequence[tuple[float, Hashable]]:
        result = transition(t, s, a)
        pairs = tuple(result) if stochastic else ((1.0, result),)
        if stochastic and abs(sum(p for p, _ in pairs) - 1.0) > _PROBABILITY_SUM:
            refuse(f'the transition probabilities at t={t}, s={s!r}, a={a!r} sum to {sum(p for p, _ in pairs)}')
        stranger = [n for _, n in pairs if n not in known]
        if stranger:
            refuse(
                f'transition at t={t}, s={s!r}, a={a!r} reaches {stranger}, outside the {len(known)} declared states'
            )
        return pairs

    for t in range(horizon - 1, -1, -1):
        later, now, chosen = values[0], {}, {}
        for s in declared:
            best, best_action = np.inf, None
            for a in actions(t, s):
                step = float(cost(t, s, a))
                if not np.isfinite(step):
                    continue
                total = step + sum(p * later[n] for p, n in outcomes(t, s, a))
                if total < best:
                    best, best_action = total, a
            now[s] = best
            if best_action is not None:
                chosen[s] = best_action
        values.insert(0, now)
        policy.insert(0, chosen)
    return DPSolution(tuple(values), tuple(policy), transition, stochastic)
