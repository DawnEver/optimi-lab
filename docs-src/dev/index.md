# Development

- **This repo's development MECHANISM is the family's, and it lives in `dev-lab`.** Until 2026-09-16 there was none written here at all — no `docs-src/` tree and no page describing how a verdict is produced, how a lane lands, or what the box rations. This directory is that tree's first content.
- **These pages are POINTED AT, never copied.** A shared document each repo copies is the fork the sharing removes. The table below is what `dev_lab.dev.devdocs.pointer_table` renders, so a page renamed there does not leave this one quietly wrong.
- **The rule split:** `.claude/rules/**` here is HARD CONSTRAINTS ONLY and is read on every turn; the universal rule STATEMENTS are `dev_lab.dev.rules` rows cited by ID; the MECHANISM is below.
- Reach the shared tree at `../../../dev-lab/docs-src/dev/` when dev-lab is checked out beside this repo, or read the same files on the forge. **The tree is rendered only as motronics-studio's `family` sub-site; this repo has no hand-written-markdown renderer** — `scripts/pdoc.py` builds the API reference and nothing else, so here these pages are read as markdown. Saying so is the point: a reader expecting a portal and finding markdown has been told something false.

## The family's pages

| page | what it covers |
|---|---|
| [Onboarding a repo](../../../dev-lab/docs-src/dev/onboarding.md) | The one call, every declaration a new repo supplies, and the order that matters |
| [Migrating a workstation](../../../dev-lab/docs-src/dev/migrate.md) | Start here if you used to run the gate before a push: what to stop, and the five commands that replace it |
| [The three participants](../../../dev-lab/docs-src/dev/the-three-participants.md) | Who does what: the human, the authors, the producers, the forge and the steward |
| [The verdict model](../../../dev-lab/docs-src/dev/verdict-model.md) | What a verdict is, the tiers, and which destination needs which forge parts |
| [Testing discipline](../../../dev-lab/docs-src/dev/testing-discipline.md) | The lightweight/heavy partition, xfail never skip, one test session per box |
| [Branch layers](../../../dev-lab/docs-src/dev/branch-layers.md) | Trunk, integration and child branches: what each proves and the one verb that moves it |
| [Alignment](../../../dev-lab/docs-src/dev/alignment.md) | How every branch keeps shared bookkeeping out of merge conflicts |
| [Fan-out](../../../dev-lab/docs-src/dev/fanout.md) | Parallel work in worktrees: environments, subagent bases, measuring, landing |
| [The shared checkout](../../../dev-lab/docs-src/dev/shared-checkout.md) | What breaks when several parties write one checkout, and the habits that prevent it |
| [Killed runs and orphans](../../../dev-lab/docs-src/dev/orphans.md) | What survives a stopped run, why a lock judges liveness, and how to reap |
| [Box resources](../../../dev-lab/docs-src/dev/box-resources.md) | What one workstation rations, and the broker every run declares its cost to |
| [The forge](../../../dev-lab/docs-src/dev/forge.md) | Parts, claims, `forge serve`, scheduling and `forge queue`: how a sha gets its verdict |
| [The integration branch](../../../dev-lab/docs-src/dev/integration.md) | The path of one change, the roles, the integration tick, `forge sync`, `integrate promote`, the merge audit |
| [Issues](../../../dev-lab/docs-src/dev/issues.md) | An issue is intent; its state is derived from refs and parts; claims and priority are labels and comments |
| [HPC clusters](../../../dev-lab/docs-src/dev/hpc.md) | A cluster is one more `forge serve` host: grants, the one-line deploy, runs, records |
| [Components](../../../dev-lab/docs-src/dev/components.md) | A product mounts its engines as submodules: the table, the derived-file check, the bump loop |
| [Artefacts](../../../dev-lab/docs-src/dev/artefacts.md) | A run's declared outputs: the store, its layout and its retention |
| [Retirement](../../../dev-lab/docs-src/dev/retirement.md) | The archive keeps old code, the registry keeps a dead spelling dead |
| [The docs pipeline](../../../dev-lab/docs-src/dev/docs-pipeline.md) | The three properties a docs builder holds, and why a skipped sub-site is announced |
| [Translations](../../../dev-lab/docs-src/dev/translations.md) | How a `<stem>.zh.md` beside its `<stem>.md` passes the CJK guard, and the non-ASCII goal |
| [Worktrees stay inside](../../../dev-lab/docs-src/dev/worktrees.md) | Every checkout lives under `<repo>/.claude/worktrees/<name>` |
| [One branch per session](../../../dev-lab/docs-src/dev/branch-set.md) | The declared branch set, merge-or-delete, and the push-time check |
| [Auto mode runs the doors](../../../dev-lab/docs-src/dev/auto-doors.md) | Local cleanup and run stops are checked doors both clients allow unprompted |
| [Refusals](../../../dev-lab/docs-src/dev/refusals.md) | Every deny refusal is a clause, a door and a pointer; the full text of each row |
| [Project files](../../../dev-lab/docs-src/dev/project-files.md) | Every project file is rendered from a family base plus a delta, or owned by the repo; hooks and CI are TOML |

## What this repo has, and what it does not

- **A mechanism a page describes is not automatically present here**, and assuming it is would be an instruction this repo cannot follow.
- PRESENT: the Makefile's lint and test targets, `scripts/pdoc.py`, and the lane and worktree hazards, which are properties of the box and of git rather than of any repo.
- NOT PRESENT: a tiered gate runner with a case library and live solver engines. That one is motronics-studio's, and the portable substitute the family ships is `python -m dev_lab.dev.verify`, which produces the same verdict shape without needing any of it.
- This repo's own subject — the optimisers, their operators and their problem definitions — has no page yet; when one is written it belongs here rather than in the shared tree, because its subject changes the moment you change repos.
