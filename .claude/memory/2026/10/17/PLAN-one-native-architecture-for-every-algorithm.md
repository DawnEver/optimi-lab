---
name: PLAN-one-native-architecture-for-every-algorithm
description: 2026-10-17 user-ratified plan to rebuild optimi-lab as one self-implemented decision library over two new lower packages -- gauss (equation solving, extracted from motronics) and ledger (task-outcome ledger shared by optimi-lab, motronics and lab-commons hpc); full layer-by-layer architecture, a 150-item native inventory, what stays outside and why, phases G/L/P8-P15 with acceptance, and the handoff to the motronics session
created: 2026-10-17
accessed: 2026-10-17
tags: [plan, architecture, algorithms, surrogates, gradients, stochastic, structured, topology, control, gauss, ledger, handoff]
metadata:
  type: project
---

# PLAN: one native architecture for every algorithm

Status: RATIFIED by the user 2026-10-17, in a session forked from motronics-studio-39. No code has
started. It continues motronics' `PLAN-optimization-stack-refactor-with-optimi-lab` (2026-10-16,
P0-P7 merged at optimi-lab 0.3). Read-only inputs: the OneDrive research package
`archived/261017-Optimization-Framework`, which has 29 motor algorithm families (A01-A29), 32
cross-domain methods (OR/ML/SC/CT), the B0 counterexamples and the API proposals.

## Naming and product architecture (user rulings later on 2026-10-17; these supersede the names used below)

- **Engines and doors.** Every external engine repository is named `<x>-lab` and imported as
  `<x>_lab`. In motronics, each engine has exactly ONE door module, named after a person; that door
  is the only place that imports the engine, and it translates product contracts into engine types.

| motronics door | engine |
|---|---|
| `gauss` | `numerics-lab` / `numerics_lab` |
| `pareto` | `optimi-lab` |
| `turing` | `ai-lab` |
| `euclid.winding` | `wdg-lab` |
| `core` | `lab-commons` |
| `contracts` | the task-outcome package; its name is still undecided |

  Below, "package `gauss`" means the engine `numerics-lab`, and "`ledger`" means the task-outcome
  package.
- **Versions.**
  - A library floats on the latest version of the engines it depends on: no pins. For example,
    optimi-lab floats on numerics-lab.
  - The product composes exactly. Every engine except lab-commons is a git submodule under
    `components/<name>`, and Python and Rust consume it by path.
  - A lab-commons mechanism (`[tool.lab_commons.components]`) bumps one pointer at a time after a
    forge verdict: merge if green; if red, keep the last green pointer and open an issue.
  - lab-commons itself stays a floating git dependency and is not mounted.
- **Candidate engines, NOT extracted now:** laplace's MNA/circuit solver; the mesh Rust crates.
- **Deferred proposal, not adopted** (user: keep the current names for now). Rename the remaining
  non-person modules:

| current | proposed person | alternative |
|---|---|---|
| `contracts` | `leibniz`, the universal-language idea | `frege` |
| `core` | `kelvin`, measurement and units | `steinmetz` |
| `workflow` | `lovelace`, steps arranged as a program | `babbage` |
| `cli` | `morse`, the human-machine interface | -- |

  If adopted, do it as one repository-wide rename at a quiet point, with retired names registered.

## 0. Rulings (user, 2026-10-17)

| # | ruling |
|---|---|
| R1 | Runtime depends only on numpy plus low-level mathematics: `scipy.linalg`, `scipy.sparse` / `scipy.sparse.linalg`, `scipy.special`. No scipy.optimize, scipy.stats algorithms, sklearn, pymoo, BoTorch, optuna, cma, NOMAD, OR-Tools or cvxpy at runtime. Those may appear ONLY in tests, as reference oracles. |
| R2 | Every algorithm and EVERY surrogate model is self-implemented under one architecture. |
| R3 | Gradients matter. motronics owns its FEA/MEC/SDM/MNA solvers, so adjoint gradients are reachable. |
| R4 | Topology optimisation is in scope: its pure-algorithm parts live here. Winding legality rules are NOT in scope for now. |
| R5 | Equation-solving numerics are ONE package, `gauss`, extracted from motronics as an independent sub-repository and moving to Rust long term. optimi-lab depends on it and keeps no numeric kernel of its own. |
| R6 | A new minimal package, `ledger`, owns the task-outcome ledger (status vocabulary, cell, identity, JSONL, replay). optimi-lab, motronics and lab-commons hpc all use it. |
| R7 | No backward compatibility anywhere in this plan. |
| R8 | optimi-lab's `uv.lock` is git-ignored by design; each checkout generates its own. |

Principles applied throughout: first principles, discard historical baggage, Occam's razor, single
source of truth, layered decoupling, names that match what things are.

## 1. Why this shape -- first principles

1. **Three different jobs, three owners.**
   - Solving equations (Ax=b, R(y)=0, y'=f(y), quadrature, factorisations) is `gauss`.
   - Choosing decisions (min f, a Pareto set, what to evaluate next) is `optimi-lab`.
   - Recording what happened to each task is `ledger`.

   Optimisation USES equation solving at every turn: KKT systems, Cholesky, line search and trust
   region globalisation, quasi-Newton updates, shooting and collocation, adjoint solves. So the
   solver layer must exist exactly once, below optimisation.
2. **Decompose by mathematical operator, not by algorithm name.**
   - An algorithm is a composition of primitives plus a capability declaration, and each primitive
     is written once.
   - Example: NSGA-II = sampling + SBX/PM + non-dominated sort + crowding.
   - Example: EHVI-BO = GP + hypervolume + acquisition + L-BFGS-B.
3. **What an algorithm can be used for is decided by the information the evaluator can supply.**
   - Classes: values, gradients (JVP/VJP/HVP), algebraic structure, random samples.
   - Matching is explicit. It returns eligible, needs an explicit transform (cost named), or
     ineligible with a reason. Nothing degrades silently.
4. **A result has a status, and the status says whose fault it is.**
   - A design verdict: invalid or infeasible.
   - Not a design verdict, so retriable: non-converged, timeout, infrastructure, licence, screened,
     pending, partial. B0-06: a timeout is never infeasibility.
   - Evidence levels are separate: predicted < observed < validated, plus certified.
5. **The ledger is the single source of facts.** Fronts, feasible sets and resumption are views of
   it. A prediction can never overwrite an observation (B0-08).
6. **Execution is injected.** optimi-lab never knows about processes or clusters. It depends on
   one protocol, `dispatch(tasks) -> results as they finish`, and ships only a serial
   implementation.

## 2. Repository graph (each package depends only on the packages below it)

```
ledger      task-outcome ledger: status vocabulary, cell, identity hash, JSONL events, replay
            deps: stdlib + numpy
gauss       equation solving: dense/sparse linear algebra (jittered Cholesky, KKT, CG/MINRES/GMRES,
            sparse direct), nonlinear solving and its globalisation (line search, trust region,
            quasi-Newton updates), time integration (incl. shooting), quadrature, random streams
            deps: numpy, scipy.linalg/sparse/special; Rust (faer) long term
optimi-lab  decisions: problem, oracle, archive views, metrics, operators, models, solvers, policies, run
            deps: gauss, ledger
lab-commons compute / hpc execution; HPC results.jsonl IS a ledger          deps: ledger
motronics   physics and business; hamilton/laplace call gauss; pareto calls optimi-lab
            deps: optimi-lab, gauss, ledger, lab-commons
```

### What is NOT unified, and why

A state snapshot of a time march (gauss `checkpoint`, `event_checkpoint`) is resume-from-step-k
state, not a ledger of independent tasks.
- An optimiser resumes by REPLAYING its ledger deterministically and needs no snapshot.
- A march resumes from a snapshot, because replaying it from the start is too expensive.
- So snapshots stay in gauss.

## 3. Package `ledger` (new, about a few hundred lines)

- **`Status`** -- two families:

| family | values | meaning |
|---|---|---|
| design verdict | `OK`, `INVALID_DESIGN`, `INFEASIBLE` | enters ranking; the last two require evidence (rule and domain) |
| no verdict | `NONCONVERGED`, `TIMEOUT`, `INFRA`, `LICENSE`, `SCREENED`, `PENDING`, `PARTIAL` | retriable; never infeasible |

- **`Evidence`**: `PREDICTED < OBSERVED < VALIDATED`; `CERTIFIED` for a mathematical certificate.
- **`Cell(status, value, reason, evidence)`**, with these invariants:
  - only `OK` carries a finite value;
  - any decided status other than `OK` carries a reason;
  - `PENDING` carries none;
  - NaN is never a status.
- **`Identity`**: a canonical encoding and SHA of a task's defining fields. Fields are sorted; floats
  are canonicalised by an explicit policy, never coarse rounding; inactive conditional fields are
  excluded.
- **`Ledger`**: append-only events (`submitted`, `result`, plus caller-defined kinds such as
  `asked`/`told`).
  - Idempotent: a second result for one identity is stored once and counted once.
  - Views: `latest(identity)`, `pending()`, `done()`, `by(status)`.
  - `replay()` yields events in order.
- **JSONL I/O**: one file; append with flush-and-fsync per event; reading tolerates a torn last line
  and refuses anything else.
- Consumers adopt it with NO compatibility layer:
  - optimi-lab archive;
  - motronics `contracts/ir/cell.py` and `SweepFrame` cells (e8d3bed0d9);
  - the campaign CSV checkpoint;
  - lab-commons HPC `results.jsonl` (motronics `PLAN-an-hpc-job-ships-one-file-not-a-checkout`).

## 4. Package `gauss` (extracted from motronics `src/motronics/gauss` + `rust/crates/gauss`)

### What exists today (motronics)

- `linear/`: native direct solve (faer in Rust), AMG, IC0, SSOR, refinement, constraints, Dirichlet,
  gauge, border storage, index-two DAE initial conditions.
- `nonlinear/`: Newton with line search, Picard, quasi-Newton strategies, local and joint initial
  solves, finite storage, material state, convergence.
- `time_integrators/`: march, multirate, periodic, shooting, waveform relaxation, explicit, ZOH,
  factor reuse, decimate, event/EEC checkpoints.
- Top-level modules: quadrature, spline and modal bases, transforms, circle root.

### Extraction work (gauss sub-repo project, G phases)

- **G0** -- cut the coupling. gauss imports these from `motronics.contracts`: `IntegratorKind`,
  `TransientStudy`, `SemiDiscreteProblem`, `HistoryProblem`, `DerivativeWeights`,
  `NEWMARK_BETA/GAMMA`, `ResidualAssemblyBound`, `InitialJointEvaluation`. Move their numerical
  definitions into `gauss.formulation`; motronics contracts then depends on gauss for them, not the
  reverse.
- **G1** -- separate the generic numerics from the FE-shaped ones (Dirichlet, gauge, border
  storage, `material_state`). Physics-free FE-shaped parts may stay as their own gauss module;
  anything that knows a material moves to hamilton.
- **G2** -- publish what optimi-lab needs, as public API with tests:
  - jittered Cholesky that reports the jitter used;
  - dense KKT solve, and sparse KKT (`scipy.sparse.linalg`, faer in Rust);
  - CG, MINRES, Steihaug-CG;
  - Armijo, strong-Wolfe and backtracking line search as standalone functions;
  - trust-region radius update, dogleg;
  - quasi-Newton updates (BFGS, L-BFGS two-loop, SR1, Broyden), shared with the nonlinear
    strategies;
  - Gauss-Legendre/Lobatto/Radau nodes;
  - shooting and collocation residual helpers;
  - random `Stream(seed, path)` with named child streams, so asynchronous completion order cannot
    change draws (B0-18).
- **G3** -- the Rust port follows measurement (`porting.md`: interpreted per-element dispatch is
  worth porting; compiled numpy paths are not).

## 5. Package `optimi-lab` -- architecture in detail

### Global invariants

- A layer imports only lower layers, pinned by an architecture test.
- Frozen dataclasses; numpy arrays with shape checks at each boundary.
- Unsupported means `Refusal`: name what is missing and how to supply it. Never NaN, never a silent
  default.
- No import-time side effects; no file I/O outside `run`, which goes through ledger.
- Absolute tolerances carry a unit; relative ones are preferred (`TOLERANCE-CARRIES-A-UNIT`).

### Layout

```
optimi_lab/
  problem/   L1   oracle/  L2   archive/ L3   metric/ L4
  operator/  L5   (sampling, variation, selection, scalarize, qp, merit, acquisition, risk,
                   estimator, transcribe, field)
  model/     L6
  solver/    L7   (population, direct, model_based, gradient, structured, stochastic, meta,
                   topology, control)
  policy/    L8   run/ L9
```

L0 is gone: it is gauss.

### L1 problem -- what is optimised

- **Variables:**

| kind | definition | invariant |
|---|---|---|
| `Real(lo, hi, unit, scale=linear|log)` | continuous | lo < hi, finite |
| `Integer(lo, hi, step)` | integer | operators emit lattice values only |
| `Categorical(levels, embedding=None)` | category | no metric unless an explicit embedding is declared (B0-11) |
| `Field(n_cells, lo, hi, neighbors)` | density/field (topology) | carries adjacency for filters |
| `active_if=Predicate` on any variable | conditional | an inactive variable is excluded from identity and from operators |

- **`Space`**: encode/decode between declared values and the internal vector; categorical values
  are encoded as unordered indices.
- **`Objective(name, sense, unit, normalization=(shift, scale))`**: the normalisation is FIXED in
  the problem, so cross-algorithm HV/IGD+ are comparable.
- **`Constraint`** kinds, each with sense, tolerance and unit:
  - `Linear(A, b)`;
  - `Explicit(fn)`, checkable before evaluation;
  - `BlackBox(name)`;
  - `Chance(name, eps, joint)`;
  - `Path(name, axis)`, holds for all t (B0-04).
- **`Structure`** (optional): linear/quadratic coefficients, a sparsity pattern, and a declared
  convexity. A declared convexity is a claim; the result's evidence records that it was relied on.
- **`Uncertainty`**: marginals, correlation, scenario sets for xi.
- **`Fidelity`**: ordered levels (e.g. MEC < FEA2D < FEA3D), each with a cost prior.
- **`Problem`**: all of the above plus an identity SHA. Self-checks run at construction.

### L2 oracle -- what an evaluator supplies

- **`Capability`**:
  - `values`, `jvp`, `vjp`, `hvp`, `structure`, `stochastic`;
  - the fidelities, nodes and outputs supported.
- **`Action`**:
  - design, fidelity, nodes (partial evaluation), requested outputs, scenario, stream path;
  - derivative request (which quantities are held fixed);
  - identity = `ledger.Identity` over these fields.
- **`Result`**:
  - action id, `ledger.Status`;
  - values (name -> value, unit, axes), derivatives, uncertainty;
  - quality (residual, convergence, mesh), cost (wall, CPU, licence), provenance.
- **`Oracle` protocol**: `capability()`, `submit(actions)`, `poll() -> results` (they may arrive in
  any order and in parts); a synchronous evaluator is wrapped by one adapter.
- **Transforms:**
  - finite differences: forward, central, complex step, cost n+1 / 2n named;
  - penalisation of a constraint;
  - categorical embedding.

### L3 archive -- views over the ledger

- Built on `ledger.Ledger`; the action identity is the ledger identity.
- Views: `complete()`, `feasible(tol)`, `front(evidence=OBSERVED)`, `pending()`, `at_fidelity(l)`.
- A front is computed only over one fully defined set of objectives and constraints; a partially
  evaluated point waits.
- Prediction rows carry `PREDICTED` and the model version, and live apart from observations.

### L4 metric

| group | contents |
|---|---|
| dominance | plain, constrained (Deb), epsilon, tolerance -- always chosen explicitly |
| front quality | exact HV (2D/3D, WFG for higher) and Monte Carlo HV with a confidence interval; HV contribution; IGD/IGD+ (reference set named); spread; attainment surface |
| statistics | confidence intervals; rank tests across seeds |
| analysis | Sobol (Saltelli), Morris, main effect, knee, TOPSIS, effective dimension, power law |

### L5 operator -- primitives (pure functions, unit-tested)

- **sampling**: grid, LHS (maximin-optimised), Sobol QMC (Joe-Kuo direction numbers), scrambled
  Halton, Poisson disk, anchor, stratified integer/categorical.
- **variation**:
  - SBX, polynomial;
  - DE rand/1, best/1, current-to-best/1 with binomial or exponential crossover;
  - PSO velocity and position;
  - integer step, categorical resample;
  - permutation OX, PMX, swap;
  - repair: project, reflect, hook.
- **selection**: fast and incremental non-dominated sort, crowding, Das-Dennis two-layer reference
  directions, RVEA angle penalty, HV contribution, SPEA2 strength and density, tournament, random.
- **scalarize**: weighted sum, Tchebycheff, augmented Tchebycheff/ASF, PBI, epsilon-constraint,
  NBI. Each declares which part of a front it can reach (B0-02).
- **qp**: dense active-set QP, bound-constrained QP, KKT assembly (linear algebra from gauss).
- **merit**: L1 penalty, augmented Lagrangian, Fletcher filter.
- **acquisition**: log-EI, PI, UCB, KG (discretised), EHVI (exact 2D, MC beyond), MC NEHVI with
  fantasies, ParEGO-ASF, constrained EI.
- **risk**: mean, variance, mean-variance, VaR, CVaR (Rockafellar-Uryasev, correct on discrete
  atoms, B0-14), worst case, individual and joint chance (B0-03).
- **estimator**: MC/QMC with confidence intervals, importance-sampling weights and effective sample
  size, control variates.
- **transcribe**:
  - time-axis reducers (integral, peak, time over limit) and the path quantifier;
  - multiple-shooting and collocation NLP assembly; the integrators come from gauss.
- **field**: density filter, sensitivity filter, Heaviside projection with beta continuation,
  SIMP/RAMP interpolation, volume constraint and its gradient.

### L6 model -- every surrogate, self-implemented

The protocol is `fit(X, Y, noise=None) -> Fitted` and `predict(X) -> (mean, var|None, grad|None)`.
- `fit` is total: it returns a model or refuses.
- Each model declares whether it gives a variance and a gradient.
- Predictions are written only as `PREDICTED`.

| model | content |
|---|---|
| polynomial RSM | least squares (QR), ridge (closed form), lasso (coordinate descent), degree selection |
| RBF | thin plate, Gaussian, multiquadric; polynomial tail; augmented system |
| GP / Kriging | RBF, Matern 3/2 and 5/2, ARD, linear, sum and product kernels; MLE by L-BFGS-B; closed-form LOO; mean, var and grad; noise |
| gradient-enhanced Kriging | uses adjoint gradients |
| multi-output GP | intrinsic coregionalisation (ICM) |
| multi-fidelity | AR1 co-Kriging (Kennedy-O'Hagan) |
| kNN | distance weighted |
| trees | CART, random forest, extra trees, bagging; variance from tree spread |
| MLP | forward and backward pass, Adam, early stopping; ensemble for variance |
| PCE | Hermite/Legendre bases, least squares or sparse regression; Sobol indices from coefficients |
| ensemble / mixture | weights from CV error |
| validation | k-fold CV, standardised-residual calibration |

### L7 solver -- algorithms

- Protocol: `init(problem, archive, budget)`, `ask(capacity) -> actions`,
  `tell(results)` (results may arrive out of order or in parts), `done() -> Termination`,
  `summary() -> SolveResult(status, guarantee, transforms_used)`.
- Each solver declares `Requires(info, variable kinds, constraint kinds, n_obj, scale)`.
- Structured and gradient solvers may take several internal steps between asks; ask/tell is a
  shell, not a straitjacket.

### L8 policy -- where the budget goes (the solver decides where to search)

- **Screening**: replaces `Router`, `routed` and the integer `SCREENED`.
- **Fidelity choice**: cost-aware, value-of-information.
- **Racing**: Hyperband/ASHA over fidelity.
- **Partial evaluation**: e.g. EM first, thermal only if worth it.
- **Replicate allocation**: OCBA.
- **The AI hook**: replaces `steer`/`Tunable`; motronics turing deciders enter here.

### L9 run -- the optimisation loop only

1. Ask the solver for as many actions as there is capacity.
2. Apply the policy.
3. `dispatch` the actions.
4. Record results in the ledger as they arrive, and tell the solver.

Around that loop:
- **Budget ledger**: evaluations, wall time, CPU, licence, fidelity-weighted cost; a cap on any of
  them.
- **Termination**: a named reason.
- **Resume**: rebuild by deterministic replay of the ledger events.
- **Dispatch**: `dispatch(tasks) -> iterator of results as they finish` is the ONE execution
  protocol. optimi-lab ships only a serial implementation; motronics and lab-commons supply the
  real ones.

### Capability matching (spans L1/L2/L7)

```
match(problem, oracle.capability(), solver.requires)
  -> Eligible
   | NeedsTransform([FiniteDifference(cost=n+1), Penalize(c), Embed(categorical), ...])
   | Ineligible(reason)
```

Every transform states its effect on objective and feasibility semantics, and a result lists the
transforms used. B0-20: the result says whether its front is native Pareto, a weighted scan or an
external scan.

## 6. Where each information class meets motronics

- **Values only** -- JMAG, FEMM, and any black box. Finite differences are the explicit transform.
- **Gradients** -- in-house FEA/MEC/SDM: Newton already assembles R_y, so df/dx = f_x - lambda^T R_x
  costs one extra linear solve per objective. MEC and MNA are algebraic networks, so they are the
  cheapest. Required anyway by GP MLE, acquisition maximisation, MLP training and MMA/SQP
  subproblems. Uses:
  - shape and parameter optimisation;
  - BH and loss identification;
  - surrogate and calibration fits;
  - MTPA and current trajectories;
  - local front refinement;
  - topology optimisation.
- **Algebraic structure** -- MTPA/MTPV QP, drive-cycle allocation (DP), thermal-network steady
  state, epsilon-constraint subproblems, topology selection (MILP).
- **Repeated sampling** -- manufacturing tolerance (Br, eccentricity, lamination), temperature,
  failure probability, cycle statistics. FEA is deterministic, so the randomness is xi; MEC is the
  low level of MLMC.

### Control co-design vs ordinary multi-objective optimisation

- **Nested** (a parameterised controller simulated per design) is an ORDINARY black-box MOO plus
  time-axis reducers and path quantifiers. No new algorithm is needed.
- **Simultaneous** (direct transcription) differs in five ways:
  - the decisions are functions and must be discretised;
  - constraints hold for all t, and nodes alone are not enough (B0-04);
  - the problem becomes a large sparse NLP;
  - it needs gradients and a sparse Jacobian;
  - nested, simultaneous and bilevel are different problems (B0-05), and non-anticipativity applies.

  optimi-lab supplies the transcription, sparse SQP/IPM and the bilevel declaration; laplace
  supplies the dynamics.

## 7. Full inventory (K = exists in 0.3, rebuilt; N = new)

### ledger
- L1 N `Status` families + `Evidence`.
- L2 N `Cell` and its invariants.
- L3 N `Identity`, with the canonicalisation policy.
- L4 N `Ledger` events, idempotence, views, replay.
- L5 N JSONL with a torn-tail rule.

### gauss (extraction; numbering G)
- G0 N formulation types moved in, so nothing imports motronics.
- G1 N generic and FE-shaped split.
- G2 N published API: jittered Cholesky, dense/sparse KKT, CG/MINRES/Steihaug, line searches, trust
  region/dogleg, quasi-Newton updates, Gauss nodes, shooting/collocation residuals, `Stream`.
- G3 Rust port by measurement.

### optimi-lab

**L1 problem**
- 1 K->N `Real` (+log).
- 2 N `Integer`.
- 3 N `Categorical` (+embedding).
- 4 N `Field`.
- 5 N conditional activation.
- 6 K->N `Space` encode/decode.
- 7 K->N `Objective` with fixed normalisation.
- 8 K->N `Constraint` kinds (Linear/Explicit/BlackBox/Chance/Path).
- 9 N `Structure`.
- 10 N `Uncertainty`.
- 11 N `Fidelity`.
- 12 N `Problem` + identity.

**L2 oracle**
- 13 N `Capability`.
- 14 N `Action`.
- 15 N `Result`.
- 16 N `Oracle` async protocol + sync adapter.
- 17 N finite difference (forward/central/complex step).
- 18 N penalise and embed transforms.

**L3 archive**
- 19 N archive over the ledger.
- 20 N views.
- 21 N separate prediction rows.

**L4 metric**
- 22 K dominance + Deb.
- 23 N epsilon/tolerance dominance.
- 24 K->N HV (exact 2D/3D, WFG, MC).
- 25 N HV contribution.
- 26 K->N IGD + IGD+.
- 27 N spread, attainment.
- 28 N CI, rank tests.
- 29 K Sobol.
- 30 N Morris.
- 31 K main effect, knee, TOPSIS, effective dimension, power law.

**L5 operator**
- Sampling:
  - 32 K grid/LHS/Poisson/anchor;
  - 33 N maximin LHS;
  - 34 N Sobol QMC;
  - 35 N scrambled Halton;
  - 36 N stratified integer/categorical.
- Variation:
  - 37 K SBX/PM/DE/PSO;
  - 38 N other DE variants;
  - 39 N integer/categorical/permutation;
  - 40 N repair.
- Selection:
  - 41 K non-dominated sort/crowding/reference/tournament;
  - 42 N incremental non-dominated sort;
  - 43 N RVEA angle penalty;
  - 44 N SPEA2 strength/density;
  - 45 N HV-contribution selection.
- Scalarisation:
  - 46 K Tchebycheff;
  - 47 N WS/ASF/PBI/epsilon/NBI with reach declarations.
- Local subproblems:
  - 48 N active-set QP, bound QP, KKT assembly;
  - 49 N merit functions (L1, AL, filter).
- Acquisition:
  - 50 N log-EI/PI/UCB/KG;
  - 51 N EHVI/NEHVI/ParEGO-ASF/constrained EI.
- 52 N risk operators.
- 53 N estimators (MC/QMC/IS/control variates).
- Transcription:
  - 54 N time reducers + path quantifier;
  - 55 N shooting/collocation NLP assembly.
- 56 N field primitives (filters, Heaviside+beta, SIMP/RAMP, volume).

**L6 model**
- 57 N protocol.
- 58 K->N polynomial RSM/ridge/lasso.
- 59 N RBF.
- GP:
  - 60 N kernels;
  - 61 N MLE;
  - 62 N LOO;
  - 63 N mean/var/grad.
- 64 N gradient-enhanced Kriging.
- 65 N ICM multi-output.
- 66 N AR1 co-Kriging.
- 67 K->N kNN.
- 68 K->N CART/forest/extra trees/bagging.
- 69 K->N MLP.
- 70 N PCE.
- 71 K->N ensemble/mixture.
- 72 N CV and calibration.

**L7 population**
- 73 K->N GA.
- 74 K->N NSGA-II.
- 75 N SPEA2.
- 76 K->N NSGA-III.
- 77 N U-/R-NSGA-III.
- 78 N RVEA.
- 79 K->N MOEA/D (Tchebycheff/PBI).
- 80 N SMS-EMOA.
- 81 K->N MODE.
- 82 K->N MOPSO.
- 83 N CMA-ES + IPOP.
- 84 N MAP-Elites.

**L7 local derivative-free**
- 85 N Nelder-Mead.
- 86 N pattern search/MADS with progressive barrier.
- 87 N quadratic-model trust region (BOBYQA-like).

**L7 model-based**
- 88 N BO (EI/KG).
- 89 N ParEGO.
- 90 N EHVI/NEHVI MOBO.
- 91 N batch/async (Kriging believer, fantasies).
- 92 N multi-fidelity BO.
- 93 N TuRBO.
- 94 K->N surrogate-prescreened EA (from routing/gate).
- 95 N trust-region model management.

**L7 gradient**
- 96 N L-BFGS-B.
- 97 N projected gradient.
- 98 N Newton-CG trust region.
- 99 N SQP.
- 100 N interior point (dense and sparse).
- 101 N augmented Lagrangian.
- 102 N MMA/GCMMA.
- 103 N MGDA.
- 104 N Pareto continuation.

**L7 structured**
- 105 N LP (simplex, IPM).
- 106 N QP (uses 48).
- 107 N MILP (branch and bound + cuts).
- 108 N dynamic programming.
- 109 N small CP (propagation + backtracking).
- 110 N Benders.
- 111 N ADMM.

**L7 stochastic**
- 112 N MC/QMC robust evaluation.
- 113 N importance sampling.
- 114 N subset simulation.
- 115 N FORM/SORM.
- 116 N PCE reliability.
- 117 N MLMC.
- 118 N SAA.
- 119 N OCBA.

**L7 meta**
- 120 N scalarisation scans to a front.
- 121 N multistart.
- 122 N restarts.
- 123 N Hyperband/ASHA.
- 124 N conflict learning (no-good regions).
- 125 N warm start from a ledger.

**L7 topology** (physics solve and adjoint sensitivities come from motronics hamilton)
- 126 N SIMP + MMA/OC.
- 127 N Heaviside-projection workflow with beta continuation.
- 128 N level set (HJ update, reinitialisation), later.
- 129 N ON/OFF binary, later.

**L7 control**
- 130 N nested co-design (composition of 54, no new algorithm).
- 131 N simultaneous co-design (55 + sparse SQP/IPM).
- 132 N bilevel declaration with optimistic/pessimistic rule.

**L8 policy**
- 133 K->N screening.
- 134 N fidelity choice.
- 135 N racing.
- 136 N partial evaluation.
- 137 N replicate allocation.
- 138 K->N AI hook.

**L9 run**
- 139 K->N async loop.
- 140 N budget ledger.
- 141 N termination reasons.
- 142 N resume by replay.
- 143 N `dispatch` protocol + serial implementation.

**Cross-cutting**
- 144 N capability registry and transforms.
- 145 N one-way layer architecture test.
- 146 N benchmarks in tests: ZDT, DTLZ, constrained problems, and the B0 counterexamples 02, 03, 04,
  05, 06, 08, 11, 12, 14, 16, 18, 20.
- 147 N reference-oracle equivalence tests (scipy, sklearn, pymoo, BoTorch, tests only).

**Deleted, no compatibility (148-150)**
- `Evaluation`, `Record`, `routed`, integer `SCREENED`, `Tunable`/`steer`, `build_proposer(pop_size)`,
  `SampleKind.IMPORT`, the root `benchmarks.py` (moves into tests).
- `Outcome` (replaced by `ledger.Status`).
- The sklearn dependency.

## 8. Stays outside optimi-lab (not pure algorithms; they CALL it)

| item | owner |
|---|---|
| Physics of topology/field distributions; shape sensitivity R_x; adjoint assembly | motronics hamilton / euclid |
| Dynamics for co-design, MPC, CBF, reachability | motronics laplace |
| Learned algorithm selection | motronics turing (through the L8 AI hook) |
| Winding legality rules | not in scope for now (R4) |
| Preference learning, SafeOpt on hardware, HJ reachability, SOS certificates | not in scope: hardware-in-the-loop or formal verification, with no current case |

## 9. Phases and acceptance (TDD, failing test first, planted control in each)

| phase | repo | scope | depends on | exit gate |
|---|---|---|---|---|
| L | ledger | L1-L5 | -- | invariants + replay + idempotence tests; torn-tail control |
| G0-G2 | gauss | extraction + published API | -- | motronics suites unchanged after the switch; G2 API tests; nothing imports motronics |
| P8 | optimi-lab | L1, L2, L3, L9 + items 144, 145 | L | B0-06/12/18 pass; serial run resumes by replay to identical state |
| P9 | optimi-lab | gradient family 96-104 + 48, 49 | G2 | matches scipy.optimize on CUTEst-style problems; B0-09 total derivative vs FD |
| P10 | optimi-lab | L6 models 57-72; sklearn removed | P9 | per-model equivalence with sklearn on fixed data; GP log-likelihood within tolerance |
| P11 | optimi-lab | model-based 88-95 + 50, 51 | P10 | BoTorch reference on ZDT/DTLZ at equal budget; B0-08 |
| P12 | optimi-lab | population 73-87 + operators 32-47 | P8 | pymoo reference fronts at equal seeds/budget; B0-02/11/20 |
| P13 | optimi-lab | stochastic 112-119 + 52, 53 | P9 | B0-03/14; failure probability vs analytic cases |
| P14 | optimi-lab | structured 105-111 | P9 | LP/QP/MILP vs scipy reference |
| P15 | optimi-lab | topology 126-129, control 130-132 | P9, P14 | MBB-beam / cantilever compliance reference; a collocation OCP vs known solution; B0-04/05 |
| P16 | motronics | pareto translates into L1/L2; oracles declare capability; ledger adopted by SweepFrame/campaign/HPC; adjoint gradients in hamilton | P8+ | one recorded T3 replay reproduces its front or every difference is named |

- Size: 15-25 k new lines in optimi-lab plus comparable tests; ledger is a few hundred lines; the
  gauss extraction is mostly moving code.
- Rust only where a measurement shows interpreter overhead.

## 10. Handoff to the motronics session (motronics-studio-39)

Its scope split (received 2026-10-17):
- **It owns** the `refactor/opt-refused-cells` and `refactor/opt-pools-onto-executor` lanes.
- **This plan owns** the optimisation library redesign, `pareto/`, and moving
  `OptimizationResult`/`ObjectiveRow` onto the frame.

What this plan changes for motronics, and when:
- `contracts/ir/cell.py` (`Cell`, `EvaluationOutcome`) and the campaign CSV checkpoint are REPLACED
  by `ledger` in P16. Lanes touching them now should not add new outcome kinds without matching the
  `ledger.Status` table in section 3.
- `src/motronics/gauss` leaves for its own repository in G0-G2. Code that imports
  `motronics.gauss.*` will import `gauss.*`, and the contracts formulation types move into
  `gauss.formulation`.
- `pareto/` becomes pure translation into optimi-lab L1/L2 in P16; the five routes become rows of
  (solver, policy, oracle) data.
- Nothing in this plan touches `core/utils/executor.py` or the six pools being moved.

## 11. Open points

- Naming inside packages may be refined at implementation time; the layer boundaries may not.
- Whether `ledger` lives in its own repository or as a top-level package of lab-commons with no dev
  tooling imported: decide at phase L. A separate repository is the default.
