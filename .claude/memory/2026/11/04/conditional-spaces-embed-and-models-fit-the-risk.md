---
created: 2026-11-04
---
# Conditional spaces embed, and model-based solvers fit the declared risk

User ruling 2026-11-04: conditional variables and robust search live in optimi-lab, not motronics.

- `problem.Embedding(space, categorical=one_hot|ordinal, inactive=neutral|indicator, neutral=0.5 | {name: u})`:
  finite rows for a model or a variation; decode goes through `Space.decode`, so it is always legal
  (neutral and indicator columns are never read back). `Embedding.declared` refuses a space that
  needs one (`needs_embedding`) when none is given -- an embedding is DECLARED, never defaulted.
- Taken as `embedding=` by `PopulationSolver` (all EAs, CMA-ES, MOEA/D, swarm, MAP-Elites,
  prescreened), `ModelBased` (bo, parego, ehvi, nehvi), `turbo`, `mfbo`; `warm_start(..., embedding)`.
  `oracle.embed` now delegates to `embed_categoricals` (one-hot by default).
- Registry: `Requires.conditional`; without it a conditional space is Ineligible (local solvers,
  model_trust_region, scan, hyperband/asha). Metric solvers list the `embed` transform.
- Risk on models: each pick asked once per shared scenario; the data row is the risk of each
  objective and BlackBox margin over its scenarios, so the GP models the risk estimate and the
  acquisition reads it at the predicted mean. Chance refused by a model (no probability estimate);
  turbo and mfbo refuse uncertainty, naming why. prescreened is now `_ROBUST_PARETO` (it is a
  population solver).
- `policy.routing.natural` reads a switched-off axis as the one NaN object `INACTIVE`.
- NOT RUN locally (ruling: HPC). Bars in the new tests are unmeasured first guesses: GAP 0.05 on the
  switch benchmark, POSITION 0.03 on the robust one. Tests: tests/unit/problem/test_embedding.py,
  tests/unit/solver/test_conditional_embedding.py, tests/unit/solver/test_robust_model_based.py,
  tests/unit/policy/test_routing_and_replay.py, plus the existing solver/requires/transforms suites.
