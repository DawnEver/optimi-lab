"""D128 mixed-integer linear programs: LP-based branch and bound with Gomory mixed-integer cuts at the root.

    minimise c.x   subject to the rows and bounds of a LinearProgram,   x_k integer where *integrality*

* Root: the LP relaxation (revised simplex), then up to *cut_rounds* rounds of GMI cuts
  (:mod:`optimi_lab.solver.cuts`) appended as rows and re-solved -- cut-and-branch.
* Tree: best-bound first (a heap on the parent's LP value), branching on the most fractional integer
  variable (``x_k <= floor`` / ``x_k >= ceil``); a node is pruned when infeasible or when its bound
  cannot beat the incumbent by more than the *gap* (relative to ``1 + |incumbent|``).
* Report: ``bound`` is the smallest open-node bound -- the proven lower bound -- equal to the value
  when the tree is closed. A node limit returns ``ITERATION_LIMIT`` with the incumbent (if any) and
  that bound; no incumbent at all is ``INFEASIBLE`` only when the tree CLOSED.

Integrality is judged at *integrality_tolerance* in each variable's own unit.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.solver.cuts import gomory_cuts, integer_columns
from optimi_lab.solver.linear import report
from optimi_lab.solver.linear_program import LinearProgram, StructuredSolution, standard_form
from optimi_lab.solver.simplex import simplex
from optimi_lab.solver.smooth import Halt

__all__ = ['milp']


def _root_cuts(problem: LinearProgram, integrality: np.ndarray, rounds: int) -> tuple[LinearProgram, int]:
    """*problem* with up to *rounds* rounds of GMI cuts appended, and how many cuts were added."""
    added = 0
    for _ in range(rounds):
        form = standard_form(problem)
        vertex = simplex(form)
        if vertex.halt is not Halt.CONVERGED:
            break
        rows, sides = gomory_cuts(form, vertex, integer_columns(form, integrality))
        if not sides.size:
            break
        problem = problem.with_rows(rows, sides)
        added += sides.size
    return problem, added


def milp(
    problem: LinearProgram,
    integrality,
    *,
    cut_rounds: int = 5,
    node_limit: int = 10_000,
    gap: float = 1e-9,
    integrality_tolerance: float = 1e-6,
) -> StructuredSolution:
    """Solve the mixed-integer program; see the module docstring."""
    integrality = np.asarray(integrality, dtype=bool).reshape(-1)
    if integrality.size != problem.width:
        refuse(f'integrality has {integrality.size} entr(ies) for {problem.width} variable(s)')
    lo = np.where(integrality, np.ceil(problem.lo - integrality_tolerance), problem.lo)
    hi = np.where(integrality, np.floor(problem.hi + integrality_tolerance), problem.hi)
    if np.any(lo > hi):
        return StructuredSolution(Halt.INFEASIBLE, problem.lo.copy(), np.nan, 0, reason='an integer range is empty')
    cut, added = _root_cuts(problem.with_bounds(lo, hi), integrality, cut_rounds)
    evidence = ['milp:branch_and_bound', f'cuts:gomory_mixed_integer:{added}']
    counter = itertools.count()
    heap: list[tuple[float, int, np.ndarray, np.ndarray]] = [(-np.inf, next(counter), lo, hi)]
    incumbent: StructuredSolution | None = None
    nodes = 0
    while heap and nodes < node_limit:
        bound, _, node_lo, node_hi = heapq.heappop(heap)
        if incumbent is not None and bound >= incumbent.value - gap * (1.0 + abs(incumbent.value)):
            continue
        nodes += 1
        node = cut.with_bounds(node_lo, node_hi)
        relaxed = report(node, simplex(standard_form(node)))
        if relaxed.halt is Halt.UNBOUNDED:
            return StructuredSolution(
                Halt.UNBOUNDED,
                relaxed.x,
                np.nan,
                nodes,
                reason='the LP relaxation is unbounded',
                evidence=tuple(evidence),
            )
        if relaxed.halt is not Halt.CONVERGED:
            continue
        if incumbent is not None and relaxed.value >= incumbent.value - gap * (1.0 + abs(incumbent.value)):
            continue
        distance = np.abs(relaxed.x - np.round(relaxed.x))
        distance[~integrality] = 0.0
        if distance.max(initial=0.0) <= integrality_tolerance:
            x = np.where(integrality, np.round(relaxed.x), relaxed.x)
            incumbent = StructuredSolution(Halt.CONVERGED, x, float(problem.c @ x), nodes)
            continue
        k = int(np.argmax(distance))
        down_hi, up_lo = node_hi.copy(), node_lo.copy()
        down_hi[k], up_lo[k] = np.floor(relaxed.x[k]), np.ceil(relaxed.x[k])
        heapq.heappush(heap, (relaxed.value, next(counter), node_lo, down_hi))
        heapq.heappush(heap, (relaxed.value, next(counter), up_lo, node_hi))
    evidence.append(f'nodes:{nodes}')
    closed = not heap or all(
        incumbent is not None and b >= incumbent.value - gap * (1.0 + abs(incumbent.value)) for b, *_ in heap
    )
    if incumbent is None:
        halt = Halt.INFEASIBLE if closed else Halt.ITERATION_LIMIT
        reason = 'every node is infeasible' if closed else f'no integer point within {node_limit} nodes'
        open_bound = min((b for b, *_ in heap), default=np.nan)
        return StructuredSolution(
            halt, problem.lo.copy(), np.nan, nodes, bound=open_bound, reason=reason, evidence=tuple(evidence)
        )
    bound = incumbent.value if closed else min(incumbent.value, *(b for b, *_ in heap))
    return StructuredSolution(
        Halt.CONVERGED if closed else Halt.ITERATION_LIMIT,
        incumbent.x,
        incumbent.value,
        nodes,
        bound=bound,
        reason='' if closed else f'the node limit {node_limit} left the tree open',
        evidence=tuple(evidence),
    )
