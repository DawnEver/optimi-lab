---
name: p9-gradient-methods-are-pure-functions-over-numerics-lab
description: Track D phase P9 of PLAN-engines-and-one-outcome-vocabulary (motronics memory 2026/10/17) -- D31-D41 on branch rebuild/p9-gradients; every gradient method is a pure function over evaluate(x) built on numerics-lab (B3 API), driven by the run loop through one replay adapter; scipy.optimize only as a test reference; numerics-lab declared as a floating git dependency whose remote does not exist yet.
metadata:
  type: project
created: 2026-10-18
accessed: 2026-10-18
---

# P9: gradient methods are pure functions over numerics-lab

Plan: motronics-studio `.claude/memory/2026/10/17/PLAN-engines-and-one-outcome-vocabulary.md`, Track D, P9.

## Shape

- L5 `optimi_lab.operator`: `Linearization` (f, df, c<=0, dc, h=0, dh at x), `kkt_residual`,
  `lagrangian_gradient`, Powell `damped_secant` (D31 KKT assembly); `active_set_qp` (N&W 16.3,
  phase 1 in (x, t), infeasible -> `feasible=False`) and `bound_qp` (Bertsekas projected Newton)
  (D31); l1 merit/slope/penalty and the augmented Lagrangian (D32).
- L7 `optimi_lab.solver`: `bounded` (projected_gradient = SPG, lbfgsb, newton_cg), `constrained`
  (sqp, interior_point dense/sparse, augmented_lagrangian), `moving_asymptotes` (mma, gcmma via
  `conservative=True`), `multiobjective` (mgda = Fliege-Svaiter on the box, pareto_continuation =
  epsilon-constraint homotopy with a secant predictor, two objectives only). D33-D41.
- A method is `method(evaluate, x0, lo, hi, **settings) -> Solution` (Halt, x, point, residual,
  front). `SmoothView` maps Problem + Result to a Linearization in the unit cube and the minimised
  normalised frame; design constraints are differentiated through the decode map.
- `GradientSolver` (ask/tell) RE-RUNS the method over the results told so far; the first untold
  point raises `_Needed` (a BaseException, so no `except Exception` swallows it) and becomes the
  action. Deterministic, thread-free, and resume-by-replay falls out. Cost: the method's own
  arithmetic once per evaluation (quadratic in evaluations) -- noise beside a field solve, visible on
  long cheap runs.
- `oracle.check_gradient`: declared gradient vs central differences of the oracle's values,
  relative per output. B0-09 (`tests/benchmarks/test_b0_09_total_derivative.py`): the adjoint total
  derivative passes, the partial alone is named.
- Registry: ten names in `SOLVERS` / `METHODS`; a values-only oracle is refused naming
  `finite_differences` (apply `with_finite_differences` first).

## Decisions taken here, and the three rulings that replaced some of them (coordinator, 2026-10-18)

- (a) L-BFGS-B is the real algorithm: generalised Cauchy point along the projected path
  (`operator.cauchy_point`), subspace minimisation over the free variables (CG, truncated to the
  box), strong-Wolfe capped at the box. The first version solved the box model exactly with
  `bound_qp` -- not L-BFGS-B, so the name lied; that step is gone. `bound_qp` itself stays as the D31
  operator under its true name. numerics-lab publishes no COMPACT L-BFGS matrix, so the model is
  `theta I` carried through the pairs by its dense `bfgs_update` (same matrix, dense) -- a compact
  representation belongs in numerics-lab.
- (b) A Hessian-vector product is asked WITH its direction: `Action.direction` (unit cube, part of the
  identity, required iff `Information.HVP` is wanted); `Result.hvp` is refused for an action without
  one. Newton-CG takes `hvp(x, v)` -- the oracle's when it declares HVP -- or the NAMED fallback
  `hvp="finite_difference"`; no default; the choice is in `Solution.evidence`
  (`hvp:declared` / `hvp:finite_difference`). No field was renamed, so nothing is registered retired.
  `jvp` has the same direction-less shape and is NOT fixed here.
- (c) Stopping: residual <= `rtol * |grad f(x0)|_max` OR <= `atol` (`kkt_bar`; `atol` > 0 in
  normalised objective units per unit-cube width, both settings declared; defaults `RTOL`, `ATOL`).
  Planted control: a start whose gradient is ~4e-18 stops on `atol` for every single-objective method.
  The `tolerance` setting is gone (a generic word, not registrable).
- MMA's dual is maximised by our own `lbfgsb` on `[0, inf)` with `atol=1e-8`: below ~1e-9 the dual
  gradient is rounding and the search stagnates (measured: 500 iterations, 9.6 s vs 0.02 s).
- requires-python raised to 3.13 (numerics-lab's floor). `solver/mma.py` would have been shadowed by
  the exported function; the module is `moving_asymptotes`, and `gcmma` is its own function.

## Findings for numerics-lab (not fixed there; local-only repo)

- `strong_wolfe`'s zoom divides by zero when the bracket collapses (`_cubic_minimizer`, a == b);
  `lbfgsb` catches `ZeroDivisionError` beside `WolfeSearchFailed` and falls back to Armijo. Fix
  belongs in numerics-lab.

## Not done / open

- numerics-lab is a floating runtime dependency (bare repository URL, no ref -- the components
  docs: an engine floats, a product overrides it with a path source). Its repository was created
  later the same day (main 9f16d44, Apache-2.0); the lock was then re-taken and holds it, and the
  venv installs it from the lock (the interim machine-local `.pth` is gone).
- Pre-existing kit-drift reds on main are untouched (listed in the hand-back report).
