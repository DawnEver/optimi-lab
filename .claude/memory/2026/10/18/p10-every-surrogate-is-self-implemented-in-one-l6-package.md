---
name: p10-every-surrogate-is-self-implemented-in-one-l6-package
description: Track D phase P10 of PLAN-engines-and-one-outcome-vocabulary (motronics memory 2026/10/17) -- D42-D58 on branch rebuild/p10-surrogates (P9 + main with P12 merged); the scikit-learn-backed surrogate package is replaced by optimi_lab.model, every model self-implemented on numpy, scipy.linalg/special and numerics-lab; scikit-learn leaves the runtime and stays a test oracle; what each oracle showed and what was not verified.
metadata:
  type: project
created: 2026-10-18
accessed: 2026-10-18
---

# P10: every surrogate is self-implemented in one L6 package

Plan: motronics-studio `.claude/memory/2026/10/17/PLAN-engines-and-one-outcome-vocabulary.md`, Track D, P10.
Base: `rebuild/p9-gradients` f86c4d3, with optimi-lab main (P12, 5b65a67) merged in (fef6f45).

## Shape (`src/optimi_lab/model`, layer L6 -- the layering table's `model` row)

- D42 `protocol`: a surrogate is a frozen declaration; `fit -> Fitted`; `predict(x, variance=,
  gradient=) -> Prediction(mean (n,k), variance (n,k) | None, gradient (n,k,d) | None)`. Each declares
  `gives` (a set of `Give`); asking for anything else is refused naming what it gives. The variance is
  LATENT (no noise term). Models that need more than values take it as a further `fit` argument
  (GEK: gradients; co-Kriging: fidelity level per point).
- D43 `polynomial`: total-degree monomials, unpenalised intercept by centring; OLS / `Ridge`
  (augmented lstsq) / `Lasso` (cyclic coordinate descent, sklearn's objective).
- D44 `radial_basis`: eight kernels with scipy's signs and minimum tail degrees; the saddle system is
  numerics-lab `solve_kkt`.
- D45-D48 `kernel` + `gaussian_process` + `likelihood`: SE / Matern 1/2, 3/2, 5/2 with ARD;
  `theta = log[amplitude, lengths, noise]` (sklearn's order), one GP per output column on normalised
  values; the likelihood factor is numerics-lab `jittered_cholesky`; the fit is numerics-lab `minimize_lbfgs`,
  unconstrained in `u` with `theta = lo + (hi - lo)
  expit(u)` (`Box`); Dubrule's closed-form LOO.
- D49 `gradient_enhanced` (SE only, analytic likelihood gradient, nugget 1e-8 amplitude), D50
  `multi_output` (ICM, `B = W W^T + diag(kappa)`, one noise per output), D51 `co_kriging` (AR1,
  recursive, GLS scale/offset with the envelope-theorem gradient).
- D52 `neighbours`, D53 `tree` (CART best/random splitter, `Forest` = random forest / bagging /
  extra trees by settings), D54 `neural` (`Network` + `MultilayerPerceptron`, L-BFGS on the exact
  backprop gradient), D55 `chaos` (Legendre on the unit cube; mean, variance, first/total Sobol),
  D56 `ensemble` (CV-R2 weights, refused when every member scores <= 0), D57 `validation` (folds,
  `cross_validate`, `score`, `calibration`), `registry` (`SURROGATES`, `surrogate(name, **settings)`).
- D58: the scikit-learn-backed package and its test are deleted (spellings in the retired registry); scikit-learn moved to the `dev` extra
  (test oracle); the purity guard's runtime set is numpy / scipy / lab-commons / numerics-lab, with
  `sklearn` planted in its control; retired spellings registered (P10 block).

## Decisions taken here

- The hyperparameter and MLP fits call numerics-lab `minimize_lbfgs` (numerics-lab branch
  `feat/minimize-lbfgs` 23d2160, R4: one minimiser, in the engine); an earlier L6 copy was deleted.
  Bounds stay a sigmoid REPARAMETRISATION (`Box`), not a solver.
- GP outputs are fitted independently (own hyperparameters per column), unlike sklearn's shared
  kernel; correlated outputs are the ICM's job.
- Trees take `max_features` as a FRACTION only (no count reading); `max_leaf_nodes` is not offered.
- Merge resolution (fef6f45): P9's duplicate R1 scipy guard removed (P12's is the one guard);
  `Linearization` rebuilt on the frozen-fields pattern (main's NO-REFLECTION refused
  `object.__setattr__`); `gcmma` became a function so the registry names a shipped factory.

## What the oracles showed

- Exact (to rounding): OLS/ridge vs sklearn 1e-9, lasso 1e-6, RBF vs `RBFInterpolator` 1e-8 (all
  kernels, with and without smoothing), kNN 1e-12, depth-limited CART 1e-12, MLP forward pass and
  loss at sklearn's weights, scores and unshuffled folds 1e-12, GP likelihood + gradient at the same
  theta 1e-9 relative, GP mean/variance at the same theta, LOO vs n brute-force posteriors.
- sklearn's GP default `alpha=1e-10` is a second, unfitted nugget that moves the likelihood ~1e-8
  relative; the oracle runs with `alpha=0`.
- A fully grown sklearn tree is NOT deterministic in its thresholds: tied splits deep in the tree are
  broken by its random feature order (its own trees for seeds 0..4 disagree at 26-40 of 300 query
  points). The partition of the training data is determined and agrees exactly.
- Randomised ensembles (forest, extra trees, bagging, feature-subsampled forest) are within 25 % of
  sklearn's held-out RMSE; the no-resampling limit collapses to one tree exactly.
- GEK's value-and-gradient covariance reaches condition ~1e8: its likelihood agrees with
  `multivariate_normal` to 1e-7 relative, not 1e-9.

## Not verified / open

- `uv.lock` pins numerics-lab at gitea main 9f16d44, which lacks `minimize_lbfgs`: development runs
  through a venv-only `.pth` (inserted first on sys.path) to the numerics-lab `feat/minimize-lbfgs`
  worktree. RE-LOCK (`dep --lock-upgrade . --upgrade-package numerics-lab`) after the integrator merges
  that branch; until then a clean install of this branch cannot import `numerics_lab.minimize`.
- The ensemble of sklearn-matching randomised forests is a statistical band, not equality.
- No benchmark of fit time against sklearn; the GP/ICM likelihood forms the full inverse per
  evaluation (fine at tens to low hundreds of points, quadratic memory beyond).
- Pre-existing kit-drift reds on main are untouched (listed in the hand-back report).
