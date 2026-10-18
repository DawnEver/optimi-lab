"""D130 a small finite-domain constraint solver: backtracking search with forward checking, and branch and bound.

Variables have finite *domains*; a :class:`Predicate` is a predicate over a named *scope*. The search
assigns the variable with the smallest remaining domain first (MRV), and after each assignment
FORWARD-CHECKS every rule left with one unassigned variable, deleting the values it forbids; an
emptied domain backtracks at once. :func:`all_different` is the pairwise decomposition.

* No *objective*: the first solution (or, with ``count=True``, every one, counted).
* An *objective* ``(assignment) -> float`` is minimised; an optional *bound* ``(partial) -> float``
  is a LOWER bound on every completion, and a branch whose bound cannot beat the incumbent is cut.

"Small" is the claim: no global propagators, no learning -- the solver for a slot/pole table, a
winding layout or a sizing grid, not a scheduling engine. A *node_limit* reached is reported as
``ITERATION_LIMIT`` with the best assignment found.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from optimi_lab.core.errors import refuse, require_declared_names
from optimi_lab.solver.smooth import Halt

__all__ = ['CPSolution', 'Predicate', 'all_different', 'cp']


@dataclass(frozen=True, slots=True)
class Predicate:
    """``predicate(*values)`` over the variables named in *scope*, in that order."""

    scope: tuple[str, ...]
    predicate: Callable[..., bool]


def all_different(*names: str) -> tuple[Predicate, ...]:
    """The pairwise decomposition of all-different over *names*."""
    return tuple(Predicate((a, b), lambda u, v: u != v) for a, b in combinations(names, 2))


@dataclass(frozen=True, slots=True, eq=False)
class CPSolution:
    """Why it *halt*-ed, the best *assignment*, its *value*, the *solutions* counted, and *nodes* visited."""

    halt: Halt
    assignment: dict[str, object]
    value: float
    solutions: int
    nodes: int


class _Search:
    def __init__(self, domains, rules, *, objective, bound, count, node_limit) -> None:
        self.rules = rules
        self.watch = {name: [r for r in rules if name in r.scope] for name in domains}
        self.objective, self.bound, self.count, self.node_limit = objective, bound, count, node_limit
        self.best: dict[str, object] | None = None
        self.value, self.solutions, self.nodes = np.inf, 0, 0

    def stop(self) -> bool:
        if self.nodes >= self.node_limit:
            return True
        return self.best is not None and self.objective is None and not self.count

    def run(self, assigned: dict[str, object], domains: dict[str, list]) -> None:
        if self.stop():
            return
        free = [v for v in domains if v not in assigned]
        if not free:
            value = float(self.objective(assigned)) if self.objective is not None else 0.0
            self.solutions += 1
            if self.best is None or value < self.value:
                self.best, self.value = dict(assigned), value
            return
        name = min(free, key=lambda v: len(domains[v]))
        for choice in domains[name]:
            self.nodes += 1
            assigned[name] = choice
            pruned = self.forward_check(name, assigned, domains)
            if pruned is not None and not self.cut(assigned):
                self.run(assigned, pruned)
            del assigned[name]
            if self.stop():
                return

    def cut(self, assigned: dict[str, object]) -> bool:
        if self.objective is None or self.bound is None or self.best is None:
            return False
        return float(self.bound(assigned)) >= self.value

    def forward_check(self, name: str, assigned: dict, domains: dict[str, list]) -> dict[str, list] | None:
        pruned = dict(domains)
        pruned[name] = [assigned[name]]
        for rule in self.watch[name]:
            open_ = [v for v in rule.scope if v not in assigned]
            if not open_:
                if not rule.predicate(*(assigned[v] for v in rule.scope)):
                    return None
            elif len(open_) == 1:
                (u,) = open_
                kept = [x for x in pruned[u] if rule.predicate(*(x if v == u else assigned[v] for v in rule.scope))]
                if not kept:
                    return None
                pruned[u] = kept
        return pruned


def cp(
    domains: Mapping[str, Sequence],
    rules: Sequence[Predicate] = (),
    *,
    objective: Callable[[dict], float] | None = None,
    bound: Callable[[dict], float] | None = None,
    count: bool = False,
    node_limit: int = 1_000_000,
) -> CPSolution:
    """Search the domains; see the module docstring."""
    require_declared_names(tuple(domains), 'the CP variables')
    stranger = sorted({v for r in rules for v in r.scope} - set(domains))
    if stranger:
        refuse(f'rules name {stranger}, which are not among the variables {sorted(domains)}')
    empty = [v for v, d in domains.items() if not len(d)]
    if empty:
        refuse(f'variables {empty} have empty domains')
    search = _Search(domains, tuple(rules), objective=objective, bound=bound, count=count, node_limit=node_limit)
    search.run({}, {v: list(d) for v, d in domains.items()})
    if search.nodes >= node_limit:
        halt = Halt.ITERATION_LIMIT
    else:
        halt = Halt.CONVERGED if search.best is not None else Halt.INFEASIBLE
    return CPSolution(halt, search.best or {}, search.value, search.solutions, search.nodes)
