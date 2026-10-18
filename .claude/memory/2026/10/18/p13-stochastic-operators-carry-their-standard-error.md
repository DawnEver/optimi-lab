---
name: p13-stochastic-operators-carry-their-standard-error
description: Track D phase P13 of PLAN-engines-and-one-outcome-vocabulary (motronics memory 2026/10/17) -- D116-D125 on rebuild/p13 (P9 + origin/main P12); risk, estimators, MC/RQMC robust evaluation, IS/subset/FORM/SORM/surrogate reliability, MLMC, SAA, OCBA; B0-03/B0-14 promoted; P9's Armijo fallback removed after numerics-lab fixed strong_wolfe; what was measured and what stays open.
metadata:
  type: project
created: 2026-10-18
accessed: 2026-10-18
---

# P13: stochastic operators carry their standard error

Plan: motronics-studio `.claude/memory/2026/10/17/PLAN-engines-and-one-outcome-vocabulary.md`, Track D, P13.
Base: `rebuild/p9-gradients` (f86c4d3), then origin/main (P12, 5b65a67) merged in exactly as P10's
fef6f45 resolved the same pair (for `core.qmc`, the one Sobol/Halton source).

## Shape

- L1: `Normal`/`Uniform` gain `quantile(u)` and `from_standard(z)`; `Uncertainty.at_unit` /
  `at_standard` map unit-cube / standard-normal rows to physical values (independent parameters, so
  the Rosenblatt map is exact).
- L5 `operator`: `risk` (expectation, mean_plus_std, value_at_risk, cvar Rockafellar-Uryasev with the
  straddling atom split, worst_case) D116; `estimators` (`Estimate(value, std_error, n)`,
  mean_estimate, probability, joint_probability, satisfied(Chance), replicated) D117; `robust`
  (`Rule` mc/sobol/halton, draw, robust_mean -- RQMC as scrambled replicates) D118; `reliability`
  (crude MC, importance_sampling D119, subset_simulation D120 Au-Beck modified Metropolis,
  surrogate_reliability D122 with an injected fit, in_standard_space); `form` (iHLRF FORM with
  numerics-lab's Armijo on the merit, Hohenbichler-Rackwitz SORM) D121; `multilevel` (Giles
  adaptive MLMC) D123; `allocation` (OCBA, largest-remainder integers) D125.
- L7 `solver.stochastic`: `sample_average` (scenario set -> Evaluate, any gradient method) and `saa`
  with Mak-Morton-Wood lower/upper bounds and gap D124.

## Measured (2026-10-18, this branch)

- SORM: Breitung's beta-form is 2.26 % off the Gauss-Hermite truth at beta = 3, k = 0.2; the
  Hohenbichler-Rackwitz psi-form 0.50 %, so the psi-form ships (gated 1 %).
- RQMC (scrambled Sobol, 16 replicates) standard error 12.6x below MC at 4096 points (gated 3x).
- OCBA two-stage (10 + 50 replicates, 5 designs, 2000 macro-replications): PCS 0.862 vs 0.8265 equal
  allocation (gated +0.02).
- MLMC call (eps 2e-3): converged, levels 0-4, V_l 2.2e-4 -> 3.2e-5 over levels 1-4, |error| 2.0e-3
  (gated 3 eps); an uncoupled coarse path costs 3.0x more at eps 1e-2 (V_4 0.041 vs 3.4e-5).
- lbfgsb without the Armijo fallback: the solver + benchmark suites fell from 204 s to 20 s.

## Decisions

- B0-03 restated on samples: the joint chance is read off the joint event matrix; two 95 % events
  give 90.25 % only when independent (comonotone gives 95 %, the product belief refuted at > 20 se).
- D122's chaos is NOT written here: P10 owns the PCE surrogate (D55). `surrogate_reliability` takes
  the fit injected; the tests use a test-local Hermite least squares as stand-in.
- P9's `lbfgsb` fallback hid three things, not one: numerics-lab's ZeroDivisionError on a collapsed
  bracket (now `WolfeSearchFailed`), the box cap (now a `capped` WolfeStep, More-Thuente's stpmax
  exit), and budget failures (now: drop the secant memory and retry steepest-descent once, then
  STALLED with the search's reason).

## Not done / open

- D122 production pairing with P10's PCE awaits P10's merge.
- No `Solver`/registry entry for SAA: it is a method composed from `sample_average` + any method;
  OCBA's sequential loop belongs to L8 replicate allocation (D147).
- numerics-lab still has no remote (P9's open point); the venv reaches it through a local `.pth`.
- Architecture reds present before this work (kit drift) untouched: see the hand-back list.
