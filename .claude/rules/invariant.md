# Invariants — facts about THIS library that no unit test states

`optimi_lab` is a generic multi-objective optimization library. Everything below is a property a
reader could violate in one commit, and each names the mechanism that refuses it.

- **The RUNTIME is numpy, scipy, scikit-learn and `lab_commons.outcomes`, and nothing else.**
  `[project.dependencies]` is that list; another entry is a decision, not a convenience. No logging,
  unit, validation, plotting, path or config library -- callers own those choices, and one of them
  (motronics) would inherit ours. Refused by `tests/architecture/test_the_runtime_stays_pure.py`,
  which reads `pyproject.toml` and the import graph rather than trusting a docstring.
- **Of lab-commons, ONLY `lab_commons.outcomes` is runtime** (plan II.1/II.3: stdlib-only, one
  task-outcome vocabulary for every engine). The base install takes it without extras; the dev kit
  (`lab-commons[dev]`) stays in the `dev` extra. Both are bare `git+https` URLs, never pinned (R8):
  a pin is a ceiling nobody re-argued. Any other `lab_commons.*` import is refused by that guard.
- **Importing the package does nothing.** No file is read, no directory is created, no global
  (numpy print options, matplotlib backend, warnings filters, random seed) is mutated at import.
  A library that reconfigures its host process is unusable inside another library.
- **A member of a declared set is resolved through that set, and a stranger RAISES** — `Refusal`
  from `optimi_lab.core.errors.refuse`, with the valid set QUOTED in the message. `REGISTRY` /
  `SURROGATE_KEYS`, `SampleKind` / `SAMPLE_KINDS` and every `StrEnum` are those sets. A silent
  fallback to a default regressor or a default sampler is the forbidden shape: the caller asked
  for something specific and cannot tell it did not get it.
- **A point that did not compute is never a number.** An `outcomes.Cell` carries values only as
  `OK` or `PARTIAL`; every other status carries a reason and no value (II.3). Nothing
  downstream — a front, a metric, a surrogate fit — may reintroduce a placeholder number for a
  failed evaluation.
- **A benchmark metric is verified against a CLOSED-FORM value, never against its own output.**
  `igd` of a true front against itself is exactly 0; `hypervolume_2d` of a known box set is
  arithmetic done by hand. A golden captured from a previous run of the same function proves only
  that the function has not changed, which is not what a metric is for.
- **Randomness enters through an explicit seed or generator argument**, never `np.random` global
  state: two callers running concurrently in one process must not be able to perturb each other,
  and an unreproducible optimization run is not evidence of anything.
