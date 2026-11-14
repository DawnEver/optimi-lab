---
name: the-qp-never-adds-a-row-dependent-on-its-working-set
description: active_set_qp skips a blocking row numerically dependent on the working rows (relative rank test at sqrt(eps)); a box bound plus a parallel linearised floor no longer makes the KKT singular or cycles.
created: 2026-11-14
---
# The active-set QP never adds a row dependent on its working set

Incident (motronics HPC paramfloor6, job 8011077): an SQP refine halted `stalled after 0 iterations -- the active-set
QP did not settle within 230 iterations (cycling on a degenerate vertex)`. The start (the GA best) sat 2e-4 of its
range off a box lower bound while an Explicit floor whose gradient is nearly that bound's axis was active too.

Cause: `operator/qp.py::_working_set` added any blocking row with `a.p > tolerance`. A near-parallel row has a tiny
but positive `a.p`, blocks at alpha = 0, and enters a working set it is (numerically) dependent on: the KKT solve is
singular (refused, misread as an indefinite Hessian) or its multipliers are huge and opposite, so the pair is dropped
and re-added in turn. The module's claim that a blocking row is independent holds in exact arithmetic only.

Fix: a blocking candidate whose residual outside the working rows' span is <= sqrt(machine eps) of its own norm is
not added (`_dependent`, `_DEPENDENT`). At that vertex it is implied by the working set to that relative precision;
once the working set changes so it is independent again, it blocks normally. Chosen over Bland's rule (does not
cure this: the cycle is numerical dependence, not a combinatorial tie) and over a Goldfarb-Idnani exchange (exact,
but needs the dual ratio test; not warranted until a case needs it). No tolerance or iteration cap was changed.

Tests: `tests/unit/operator/test_qp.py::test_d31_a_degenerate_vertex_settles_on_the_optimum` (parallel,
near-parallel, scaled duplicates, duplicates plus a second bound, near-duplicates; near-parallel and near-duplicates
red before) and its planted non-degenerate control.
