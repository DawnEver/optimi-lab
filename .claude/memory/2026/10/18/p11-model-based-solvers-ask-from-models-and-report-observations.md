---
name: p11-model-based-solvers-ask-from-models-and-report-observations
description: Track D phase P11 of PLAN-engines-and-one-outcome-vocabulary (downstream plan, 2026-10-17) -- D59-D68 on rebuild/p11-model-based (from integrate/main c028663); acquisitions, BO/ParEGO/EHVI/NEHVI with Kriging-believer fantasies, multi-fidelity BO, TuRBO, prescreened EA, trust-region model management; what was measured, what verify could not run, what is deferred.
metadata:
  type: project
created: 2026-10-18
accessed: 2026-10-18
---

# P11: model-based solvers ask from models and report observations

Plan: the downstream plan `PLAN-engines-and-one-outcome-vocabulary`, Track D, P11.
Base: `integrate/main` c028663 (P8, P9, P10, P12, P13 merged).

## Shape

- L5 `operator.acquisition` (D59, D60): EI and log-EI (Ament et al. `log h`: direct, `erfcx`,
  asymptotic series below z = -1e3), PI/log-PI, UCB, discrete knowledge gradient (Frazier upper
  envelope), log probability of feasibility (constrained EI = log EI + log PF), EXACT EHVI for
  independent Gaussian objectives in any m (grid cells of the non-dominated region,
  `Psi = std h(.)`), exact single-point HVI, NEHVI over joint draws (an `Estimate`), ParEGO's
  augmented Tchebycheff on range-normalised columns.
- L6 additions to `FittedGaussianProcess`: `posterior_covariance`, `posterior`, `draw` (caller-held
  base normals), `noise`, `condition` (a fantasy at fixed theta and normalisation, never a refit).
- L7: `model_based` (ModelBased shell, `maximise` = Sobol + numerics-lab `minimize_lbfgs` polish in
  logit coordinates on model central differences, `margins`, `require_modelable`); `bayesian.bo`
  (D61); `pareto_bo.parego` / `ehvi` / `nehvi` (D62, D63); fantasies (D64) are the Kriging
  believer for every pending point and earlier pick -- one mechanism for batch and async; NEHVI
  carries pending in its jointly drawn baseline and draws each candidate conditionally on it;
  `fidelity_bo.mfbo` (D65, Huang's augmented EI on AR1 co-Kriging); `turbo.turbo` (D66, TuRBO-1);
  `prescreened.prescreened` (D67, any L6 surrogate ranks oversampled NSGA-II/GA children by the
  solver's own survival on `mean - kappa std`); `model_trust_region` (D68, Alexandrov et al. with
  ORBIT's certified-shrink rule and a local cubic RBF in `(x - x_c) / radius` coordinates).
- All registered in `SOLVERS`: bo/turbo/mfbo/model_trust_region SINGLE, parego WEIGHTED,
  ehvi/nehvi/prescreened NATIVE_PARETO. `best`/`front` read told observations only (B0-08).

## Measured (2026-10-18, by hand -- see "Not run")

- Branin 30 evaluations: log-EI gap 0.004, EI 0.046, PI 0.031, UCB 0.022, KG 0.012 (bar 0.1).
- Constrained bowl (x + y <= 0.2, mostly infeasible start): every EI/PI variant within 1e-3.
- ZDT1 (6 vars) 40 evaluations, HV share to (1.1, 6): ParEGO 0.973, EHVI 0.984, NEHVI 0.981,
  random 0.535 (gate 0.95).
- Forrester pair: mfbo gap 3.4e-5 at cost 97 vs single-fidelity BO 0.11 at cost 120.
- Hartmann-6 TuRBO batch 4 at 80: -3.080 / -3.306 / -3.272 (min -3.32237).
- ZDT1 160 evaluations: prescreened IGD 0.240 vs NSGA-II 0.746.
- Model management: 3-D bowl 2.3e-9 in 70; 2-D Rosenbrock 0.030 at 300 (BOBYQA 4e-4 at 200).

## Decisions

- Gate oracle: BoTorch is NOT installed (nor in the dev extra), so "BoTorch at equal budget" is
  replaced by the closed-form optimum/front at an expensive budget with random search at equal
  budget as the planted control (`tests/benchmarks/test_p11_model_based_gate.py`).
- ParEGO multiplies by the weights (Knowles); a divided ASF made a zero Das-Dennis weight dominant.
- mfbo's default nested design is 2(d+1) at the truth, x3 per cheaper level: at 6 + 3 points the
  coarse GP's likelihood was maximised by calling everything noise (AR1 scale 6e4).
- Model management's default model is a LOCAL cubic RBF scaled by the radius: a GP over all points
  collapsed the radius on 4-D Rosenbrock at f = 3.0; raw coordinates made the RBF KKT singular.
- `bayesian.py` (515 lines) was split, not pinned, under the 320-line band.

## Not run / open

- `lab_commons.dev.verify` refused EVERY run on this box (0.03-0.9 GB free of 4 GB asked); no
  test of this branch ran under verify. Each scenario was smoke-run by hand at the numbers above.
- Architecture SlackFloor reds (public surface, runtime purity, relative tolerance, echo,
  cited tests, retired spelling, untimed wait, no-CJK) were ALREADY red at c028663; re-taken in
  this branch's last commit from direct scan readings (the layering floor is the one P11 alone
  pushed past its headroom). Left red, present at c028663 with the lab-commons 96a90c8 kit:
  the shared-registry adoption (two IDs the registry no longer defines), project-file / managed
  file accounting, the dev index's family table, private markers in `pyproject.toml`, and seven
  stale placement-row readings.
- numerics-lab: `strong_wolfe` raises `ValueError` (not `WolfeSearchFailed`) on phi'(0) = -0.0 --
  a gradient that squares to zero; `maximise` declares atol 1e-150 around it.
- Deferred: TRMM with a LOW-FIDELITY model (Alexandrov's original corrected low-fidelity
  oracle) -- here the model is a data fit; SCBO (constrained TuRBO); a cost-aware BO acquisition
  over the run ledger; MF-BO batch/async (one pick at a time); an L8 policy wrapper for D143
  screening (the prescreened EA is the solver half).
