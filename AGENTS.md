# optimi-lab

Multi-objective optimization and surrogate models, as pure numerics: arrays in, arrays out.

## How a change lands

The family flow, stated once in
[the integration branch](https://github.com/DawnEver/lab-commons/blob/integrate/main/docs-src/dev/integration.md):

- Cut `pr/<topic>` from `origin/integrate/main`, measure what you touched, commit, push, open a pull request.
- A producer claims the pushed head and posts `tests/<part>`; the FORGE merges the PR once every part
  stands PASS, the head merges cleanly and the merge passes the audit.
- `main` moves only by `integrate promote <sha>` on a full PASS.

The mechanisms are dev-lab's (`docs-src/dev/`); this repo declares the names in `pyproject.toml`.

<!-- BEGIN dev-lab family section -->

## Family principles

_Rendered by `dev_lab.dev.agentsection` from `dev_lab.dev.principles`; a hand edit here is refused._

- **First principles** (`first-principles`): Derive from the actual system and current needs, not habit or precedent;
  state shaping assumptions.
- **Discard historical baggage** (`no-historical-baggage`): Remove obsolete structures, duplicates and dead paths no
  current need uses -- never a capability.
- **Occam's razor** (`occams-razor`): Simplest design meeting the need; nothing hypothetical; never a capability.
- **Single source of truth** (`single-source-of-truth`): Each fact lives in one place; all else derives from it.
- **Layered decoupling** (`layered-decoupling`): Depend one way through narrow interfaces; never into internals or
  upward.
- **Names match what they do** (`names-match-behavior`): A name says what the thing does now.
- **No backward compatibility** (`no-backward-compatibility`): No shims, aliases or dual paths for old callers; callers
  move with the change.
- **Progressive disclosure** (`progressive-disclosure`): An entry point states what is needed first and points to
  detail; no duplication across levels.
- **Keep every valuable parameter and option** (`keep-valuable-options`): Generality is a requirement: keep every
  parameter and option a real case uses or plausibly needs. A reference match ADDS a selectable option and never
  replaces or deletes the alternative; numerical method choices are options even with no current caller. Every
  retirement is classified (renamed, baggage, or capability-dropped) and only a user ruling drops a capability.
- **Every wall teaches** (`every-wall-teaches`): Every refusal or failure an agent hits names its cause and its door;
  one that does not is a mechanism gap, and closing it (a hook row, a remedy, a test) comes before the task. Never route
  around a wall or hand the user a command. The loop: docs-src/dev/feedback.md (dev-lab).
- **Fan out without a barrier** (`no-barrier-fan-out`): Parallel work is independent items, each landed the moment it is
  done; nothing waits for the slowest sibling -- no barrier-style orchestration (e.g. a dynamic workflow that joins all
  agents) unless a step truly needs every result. How: docs-src/dev/fanout.md (dev-lab).
- **Deliver an image by its absolute path** (`images-by-absolute-path`): An image handed to the user is named by its
  ABSOLUTE path, never a relative one (IMAGES-BY-ABSOLUTE-PATH).
- **A worktree is named for its task** (`worktree-names-its-task`): A worktree's name has a physical meaning tied to its
  task; a hash-like name (`agent-<hex>`, an appended hash) is forbidden (WORKTREE-NAMES-ITS-TASK).
- **Native first** (`native-first`): Python declares and orchestrates, Rust computes. A stage Rust owns is written in
  Rust from its first line; a Python version of it is only a scratch prototype, never a second implementation;
  acceptance is against references (analytic, the vendor ladder, the previous implementation during a migration), never
  against a Python twin.
- **Everything in its place** (`everything-in-its-place`): Each thing lives in the one owner responsible for it; a
  misplaced thing moves, never leaving a pointer or a copy.
- **No detour** (`no-detour`): Every hop must add something; a rule that forces a detour is the defect. Complexity is
  measured, not guessed.

## Family workflow

_Each rule is cited by ID; its full statement and refusing mechanism live in `dev_lab.dev.rules`._

- `SHARED-CHECKOUT`: A box holds a SHARED, possibly concurrent checkout: there is NO single-agent mode, the remote IS
  the only shared medium, pushing is the obligation rather than the last step of landing, and a verdict cannot be
  delegated to someone who cannot see the code.
- `ONE-BRANCH-PER-SESSION`: The integration branch is the only long-lived branch a session works into: a pushed head is
  `pr/<task-slug>/<feature>`, one feature per PR, and every other branch is merged and then deleted, never left behind;
  a repo`s long-lived branches are DECLARED (the trunk plus the integration branch) so anything else on origin or on
  this box is debt, and unpushed work is reported, never deleted silently.
- `PRS-TARGET-INTEGRATION`: Every pull request targets the integration branch, one opened against the trunk is
  retargeted with a comment saying why, and each child branch carries ONE integration comment (or, without a PR, one
  `integration` commit status) updated in place by `forge integration`, never a new one per tick.
- `DOWNSTREAM-SYNCS-INTEGRATION`: A child branch (a lane, a feat branch, a pull-request head) merges ONLY the
  integration branch, so another child`s work reaches it once, through integration, unless the owner opts in to watch
  named refs recorded in the checkout`s own git config (`branch.<child>.labSyncFrom`), never in a file, and `forge sync`
  merges integration first, then the watched refs, refusing any other ref by name.
- `SUBAGENT-NO-PUSH`: A subagent never publishes (no push, no PR opened or readied, no skipped hook): it commits on its
  lane and hands the SHA back to the main session, which merges and pushes, the forge produces the verdict
  (VERDICTS-FROM-THE-FORGE), and a targeted measure of the tests it touched stays its own.
- `VERDICTS-FROM-THE-FORGE`: A verdict is a fact about (tree, env key, part), produced ONCE after the push by the forge
  serve host that claims its `tests/<part>` part and read from the forge: no pre-push hook runs a test, a full tier
  refuses outside the producer naming the remedy (push, then read the part), a targeted measure needs no claim, and a
  producer cites a stored PASS of its key rather than re-run.
- `TRUNK-MOVES-BY-FAST-FORWARD`: The trunk moves only by fast-forward to an integration commit whose every required
  `tests/<part>` part stands PASS at scope full, through `integrate promote`; it refuses with the missing parts named
  and the emergency door, and refuses a trunk that cannot fast-forward rather than merging into it.
- `MERGE-OR-DELETE`: A merged branch is DELETED and an unmerged one is MERGED, no third state: an `archive/*` tag or
  untracked rename hides work and is debt -- `branchset` lists it, deletes it when a declared branch holds its tip, else
  names it to merge.
- `AGENT-GUARD`: A hand-written test invocation, a bare network git verb and a stash are refused by the shared deny
  machinery rather than by prose, so the refusal is identical in every repo that adopts it.
- `NETWORK-RETRY-THEN-REPORT`: The forge is a REMOTE service: a network verb goes through the retry wrapper rather than
  being called directly, and a write that still fails is REPORTED with its diagnosis instead of hammered.
- `PRODUCTION-ENTRY-POINT`: Reproduce a behavior through the PRODUCTION entry point, never an ad-hoc script handed out
  beside it, and treat a behavior not reachable from there as a gap to FIX, the boundary being the SUBJECT, not the tree
  the code happens to live in.

<!-- END dev-lab family section -->

## Invariants — facts about THIS library that no unit test states

`optimi_lab` is a generic multi-objective optimization library. Everything below is a property a
reader could violate in one commit, and each names the mechanism that refuses it.

- **The RUNTIME is numpy, scipy, numerics-lab and `lab_commons.outcomes`, nothing else**
  (`test_the_runtime_stays_pure.py`); no logging/unit/plotting/config library; a downstream repo inherits ours.
  Equation solving is numerics-lab's (R4): no backtracking/Armijo or strong Wolfe search, secant two-loop, L-BFGS
  memory or minimizer, Broyden, Krylov, factorization or KKT solve here (`test_equation_solving_is_numerics_labs.py`);
  the trust region and dense BFGS/SR1 models are ours; scipy only `linalg`/`sparse`/`sparse.linalg`/`special`,
  `scipy.optimize`/`scipy.stats` in tests only (R1, `test_the_runtime_takes_only_the_r1_scipy_submodules`).
- **Of lab-commons, ONLY `lab_commons.outcomes` is runtime** (plan II.1/II.3: stdlib-only, one
  task-outcome vocabulary for every engine). The base install takes it without extras; the dev kit
  (`dev-lab`) stays in the `dev` dependency group. Both use bare `git+https` URLs (R8):
  a pin is a ceiling nobody re-argued. Any other `lab_commons.*` import is refused by that guard.
- **Importing the package does nothing.** No file is read, no directory is created, no global
  (numpy print options, matplotlib backend, warnings filters, random seed) is mutated at import.
  A library that reconfigures its host process is unusable inside another library.
- **A member of a declared set is resolved through that set, and a stranger RAISES** — `Refusal` from
  `optimi_lab.core.errors.refuse`, quoting the valid set. `SURROGATES`, `KERNEL_FAMILIES`, `RADIAL_KERNELS`,
  `SampleKind` / `SAMPLE_KINDS` and every `StrEnum` are those sets. A silent fallback to a default model or
  sampler is the forbidden shape: the caller asked for something specific and cannot tell it did not get it.
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

## Workflow — how work gets done in optimi-lab

- **The shared rules live in `dev_lab.dev.rules`; this repo supplies the MECHANISM.**
  `tests/architecture/test_optimi_lab_adopts_the_shared_registry.py` names, as a NAMED SET, every
  rule this repo enforces and every rule it does not. A rule added upstream reds that test until
  someone here decides about it — it never arrives adopted by default, and it never arrives
  silently unadopted.
- **A gap is DECLARED with a ceiling, never left silent.** The seven rules optimi-lab does not
  enforce are listed by name with a ceiling that may only go DOWN. Closing one deletes its name
  and lowers the number in the same edit.
- **This section and the invariants above are HARD CONSTRAINTS ONLY.** Reasoning, incident write-ups and
  measurements go to `.claude/memory/`; how a thing works goes in the module docstring next to it.
  A rule that no reader could violate does not belong here — it is a description.
- **A new public name is exported from its module's `__all__` in the same commit that defines it**,
  and one name has one definition across the package. The package `__init__` re-exports; it does
  not redefine.
- **Tests before commit.** Measure changed tests with `python -m dev_lab.dev.verify -- <paths>`.
  Full-suite verdicts belong to the forge after publishing the PR, not a developer's local run.
  A known failure is a strict `xfail` with its reason, never a `skip` that records nothing.
- **No `_legacy` / `_compat` shim and no deprecation alias.** This package was already rewritten
  once to remove that shape (`refactor(core)!: the library becomes pure numerics`); callers and
  tests move WITH the code, in the same commit.
