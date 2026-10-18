---
name: p8-the-base-is-four-layers-over-one-outcome-record
description: Track D phase P8 of PLAN-engines-and-one-outcome-vocabulary (motronics memory 2026/10/17) -- problem / oracle / archive / run rebuilt on lab_commons.outcomes (A0 at lab-commons a3e0e2b); D1-D30 delivered on branch rebuild/p8-base, the 0.3 evaluation surface retired, the five population algorithms ported to the Solver protocol; what was not verified is listed.
metadata:
  type: project
created: 2026-10-18
accessed: 2026-10-18
---

# P8: the base is four layers over one outcome record

Plan: motronics-studio `.claude/memory/2026/10/17/PLAN-engines-and-one-outcome-vocabulary.md`, Track D, P8.

## Done (D1-D30)

- L1 `optimi_lab.problem`: Real (+log), Integer, Categorical (+embedding, no metric without it),
  Field, `When` (active_if), Space (canonical design drops inactive values; unit-cube encode/decode),
  Objective with a fixed Normalization, constraint kinds Linear/Explicit/BlackBox/Chance/Path,
  Structure, Uncertainty (explicit Generator only), Fidelity, Problem with an `outcomes.Identity`.
- L2 `optimi_lab.oracle`: Capability (information classes, outputs, fidelities, nodes, version),
  Action (identity = design + problem + fidelity + wanted information + replicate + oracle version,
  B0-12), Result = `outcomes.Cell` + gradient/jvp/hvp/uncertainty/quality/provenance/series, async
  Oracle + `from_function` (II.3 optimisation mapping), finite differences (forward, central,
  complex for a function; complex refused for an oracle), Penalty, embed (one-hot).
- L3 `optimi_lab.archive`: views over one `outcomes.Record`; predictions held apart (B0-08).
- L9 `optimi_lab.run`: async loop, Ledger (one evaluation per result, B0-16), named Termination,
  resume by replay (k-th ask of an identity answered by its k-th recorded outcome), injected
  dispatch + `serial`.
- D27 `solver.match` + `SOLVERS` registry with MultiObjective (B0-20); D28 layering test;
  D29 `tests/benchmarks` (ZDT1/2, DTLZ2, BNH, B0 tests; 02/03/04/05/14 strict xfail naming the item).
- D30: retired and registered (see the retired registry): Evaluation, Outcome, core Record/optimize,
  routing (Router, integer SCREENED), protocols (Proposer, Evaluator, Tunable/retune/steer),
  build_proposer, SampleKind.IMPORT, root benchmarks, VariableSet/ObjectiveSet/ConstraintSet/
  Direction, `*Proposer` classes and the `proposers` module.

## Behaviour changes

- lab-commons is a RUNTIME dependency for `lab_commons.outcomes` only (invariant page and purity
  guard updated; any other `lab_commons.*` import still refused).
- Population solvers work in the unit cube and the normalised minimised frame; the first sample is
  the population; a whole generation is told in any order before the next is built; conditional
  spaces and unembedded categoricals are refused. Settings are fixed at construction (no retune).
- The run tells each batch in ASK order; completion order lives in the record. An identity asked
  twice is evaluated once; a DECIDED result is reused uncharged.
- Sampling returns unit-cube points; the anchor is a unit-cube point.
- IGD/hypervolume moved to `optimi_lab.metric`; Sobol takes a Space of unconditional Reals.

## Not verified / open

- Architecture reds that predate this branch and come from the newer lab-commons kit (missing
  `boxlock` etc., family config, registry adoption, roster, agent guard, project files): untouched.
- R1 not yet met: sampling and sensitivity still use `scipy.stats.qmc`, surrogates use sklearn
  (P10/P12 items D58, D69-D71).
- A retried RETRYABLE outcome whose cell is byte-equal to the previous one is not re-written by the
  Record (A0 idempotence), so its replay would see one outcome fewer; harmless while cells carry
  measured cost, noted for the integrator.
- Problem.identity is recomputed on each access (NO-REFLECTION forbids caching on a frozen slot).
- motronics consumes the 0.3 names (track E); a floating install of this branch breaks it until E.
