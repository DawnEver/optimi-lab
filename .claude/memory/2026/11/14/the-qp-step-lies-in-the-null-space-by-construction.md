---
name: the-qp-step-lies-in-the-null-space-by-construction
description: active_set_qp computes its equality step by the null-space method (QR of the working rows), so at a vertex of near-parallel rows the step is exactly zero and the multipliers are read; the full KKT solve's eps*cond*|g| round-off made it take junk steps forever.
created: 2026-11-14
---
# The QP step lies in the working rows' null space by construction

Follows `the-qp-never-adds-a-row-dependent-on-its-working-set.md` (same day), which did NOT cure the real case:
HPC paramfloor7 (job 8011663, pinned to fbd2190) still halted `the active-set QP did not settle within 230
iterations (cycling on a degenerate vertex)`.

Reproduced (`tests/unit/solver/test_sqp_survives_a_degenerate_vertex.py`, found by a 5000-seed sweep that
failed 45 under fbd2190): four unit-cube variables, the start 1.7e-4 off a box lower bound, three floors driven by
that axis with 1e-2..1e-1 relative cross terms and slacks 2.67e-6/1.78e-4/1.68e-4, one floor exactly on, a linear
objective of magnitude ~70. The working set fills to four rows -- a vertex, condition ~1e4, multipliers ~1e4. It
was neither a dependent add nor an add/drop cycle: numerics-lab's full KKT solve returned |p| ~ 1.4e-10
(= eps * cond * |g|) where p is exactly zero, just over `tolerance * scale`, so the loop stepped by round-off
every iteration and never reached the multiplier test (one multiplier was -3.7e3, to be dropped).

Fix (`operator/qp.py::_null_space_step`): p = Z w with Z the orthonormal null-space basis from QR of the working
rows' transpose, w from the reduced Hessian's Cholesky; multipliers from R. `rows p = 0` now holds to machine
precision whatever the rows' conditioning, a vertex gives p = 0 exactly, and the step's residual round-off scales
with the REDUCED problem, not cond(rows). Bland's rule was not the fix: no add/drop cycle occurred. Tolerances and
the iteration cap are unchanged; the dependent-add guard stays (R must be nonsingular).
