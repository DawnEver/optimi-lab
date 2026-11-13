---
created: 2026-11-13
---
# SQP reads its l1 slope from the violation a step removes, and raises nu until that step descends

Incident (a downstream spline-barrier search, 2026-11-13): an SQP refine whose objective gradient matched central
differences to 2e-5 ended `stalled -- no step decreased the l1 merit` from a start below its own floors.

- CAUSE: `l1_slope` was `grad f . d - nu * violation`, the slope of a step that removes ALL violation. When the
  linearised constraints admit no step inside the box, `active_set_qp` returns its least-violating step, which
  removes only part; the slope promised a decrease no step length had, so Armijo backtracked 30 times and stalled.
  And `nu` came from the multipliers alone, which an infeasible QP does not produce (they stay at their last values).
- FIX: `l1_reduction(point, d)` = violation minus the linearised violation at `d`; `l1_slope = grad f . d - nu *
  l1_reduction` (an upper bound on the merit derivative along ANY step, by convexity); `l1_penalty(...,
  descent=, reduction=)` raises `nu >= descent / (0.9 reduction)` when a step removes violation (Nocedal-Wright
  18.36). Equal to the old slope for a step that satisfies the linearisation.
- Planted: `tests/unit/solver/test_sqp_leaves_an_infeasible_linearisation.py` -- `x0^2 >= 0.25` from `x0 = 0.1` on
  the unit box (needs `d0 >= 1.2`, box allows 0.9): old slope -0.0025, true +0.0475, old sqp STALLED at iteration 0
  with the exact downstream reason; fixed sqp CONVERGED at (0.5, 0.5).
