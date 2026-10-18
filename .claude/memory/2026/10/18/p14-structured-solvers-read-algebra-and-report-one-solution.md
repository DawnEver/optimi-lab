---
name: p14-structured-solvers-read-algebra-and-report-one-solution
description: Track D phase P14 of PLAN-engines-and-one-outcome-vocabulary (motronics memory 2026/10/17) -- D126-D132 on rebuild/p14-structured (from integrate/main 329e959); LP simplex + Mehrotra IPM, QP route onto P9, MILP B&B with Gomory cuts, DP, small CP, Benders, ADMM; what was measured, what verify could not run, what is open.
metadata:
  type: project
created: 2026-10-18
accessed: 2026-10-18
---

# P14: structured solvers read algebra and report one solution

Plan: motronics-studio `.claude/memory/2026/10/17/PLAN-engines-and-one-outcome-vocabulary.md`, Track D, P14.
Base: `integrate/main` 329e959 (P8-P13 merged).

## Shape

- All L7, pure functions over ALGEBRA (matrices, finite sets, proximal maps), like P9's methods
  over `evaluate(x)` and P13's `saa`; none is an ask/tell driver.
- `linear_program`: `LinearProgram(c, a_ub, b_ub, a_eq, b_eq, lo, hi)` (lo 0 / hi inf default);
  ONE `standard_form` (shift, reflect, split; a slack per row) with the affine map `z = W x + w`
  when nothing is split; `StructuredSolution` is every P14 report. `Halt` gained `UNBOUNDED`
  (one vocabulary with the gradient methods). Duals are sensitivities `d value / d b`.
- D126 `simplex` (two-phase revised, Bland's rule, dependent rows dropped) and `interior_lp`
  (Mehrotra; normal equations on numerics-lab's `jittered_cholesky`); `lp(method=...)`. Basis
  solves are `numerics_lab.linear.solve_linear_system` on a CSC basis (native path, no warning).
- D127 `qp` is a ROUTE onto P9's `bound_qp` (bounds only, H PD) or `active_set_qp` (bounds
  folded into rows) -- no third QP method.
- D128 `milp`: best-bound B&B over the simplex; root rounds of GMI cuts (`cuts.py`) read off the
  optimal basis and written back on x through `W`; slacks taken as continuous (valid, weaker).
- D129 `dynamic_programming` (backward induction, deterministic or (p, next) transitions, a move
  off the declared grid REFUSED); D130 `cp` (MRV + forward checking, `all_different`, counting,
  B&B on a caller's lower bound; the scope type is `Predicate` because `Rule` is `operator.robust`'s).
- D131 `benders` (optimality cuts from recourse sensitivities, feasibility cuts from the phase-1
  program's; master by `milp` or `lp`); D132 `admm` (scaled consensus, `quadratic_prox` /
  `box_prox` / `l1_prox`, Boyd's residual stop).
- Registered: lp/milp/benders need declared LINEAR outputs and Linear rows; qp QUADRATIC+CONVEX;
  dynamic_programming SEPARABLE; admm CONVEX+SEPARABLE; cp takes Explicit/Linear rows.

## Measured (2026-10-18, direct runner -- see "Not run")

- LP: vertex/shadow prices exact (both methods); Beale's cycling example -1/20; 8 random
  programs per method agree with HiGHS (`linprog`) at 1e-7 relative, duals at 1e-4.
- QP: Duchi simplex projection to 1e-8; 6 random QPs satisfy their KKT certificate at 1e-6.
- MILP: 8 random 6-variable instances agree with HiGHS (`scipy.optimize.milp`) at 1e-7; cuts
  never grew the tree on the textbook instance; a 10-node limit reports bound -34.36 < -34.
- DP vs all 3^6 battery schedules exact; CP 92 eight-queens solutions, assignment = all 120
  permutations, the bound cut the tree. Benders = whole program (HiGHS) on 4 seeds x {lp, milp}
  master; ADMM = lasso soft threshold and the D127 box QP at 1e-5.

## Not run / open

- `lab_commons.dev.verify` passed once (D126, 63 tests), then refused: box memory (0 of 4 GB
  free), then "environment is STALE -- numerics-lab installed 4257a1c, HEAD e8551a5"; a lock
  bump is not this phase's. Every other result above is from calling the test functions directly.
- Architecture floors P14 pushed past headroom re-taken from direct readings (public surface,
  runtime purity, R4 importers 9 -> 13, relative tolerance, echo, cited tests, retired spelling,
  no-CJK, untimed wait). Still red, not P14's: the shared-registry adoption, private markers,
  the rules-page ratchet (`invariant.md` 33 -> 34) and modules that no longer import against
  the installed lab-commons kit (stale env).
- Open design question: no structured solver is reachable from a `Problem` yet -- `Problem`
  declares `Form.LINEAR` but carries no objective coefficients, so `match` can say `lp` is
  eligible while nothing can build its `LinearProgram`. Needs a ruling on where linear/quadratic
  coefficients are declared (L1 `Structure` vs an L2 capability that returns them).
- Deferred: an IPM that certifies infeasibility (homogeneous self-dual); dual simplex warm
  starts in B&B; cuts at tree nodes; a QP interior point (sparse, for transcription in P15);
  ADMM with general `A x + B z = c` and adaptive rho; stochastic DP with a rollout sampler.
