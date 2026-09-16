# Development

- **This repo's development MECHANISM is the family's, and it lives in `lab-commons`.** Until 2026-09-16 there was none written here at all — no `docs-src/` tree and no page describing how a verdict is produced, how a lane lands, or what the box rations. This directory is that tree's first content.
- **These pages are POINTED AT, never copied.** A shared document each repo copies is the fork the sharing removes. The table below is what `lab_commons.dev.devdocs.pointer_table` renders, so a page renamed there does not leave this one quietly wrong.
- **The rule split:** `.claude/rules/**` here is HARD CONSTRAINTS ONLY and is read on every turn; the universal rule STATEMENTS are `lab_commons.dev.rules` rows cited by ID; the MECHANISM is below.
- Reach the shared tree at `../../../lab-commons/docs-src/dev/` when lab-commons is checked out beside this repo, or read the same files on the forge. **There is no rendered portal for that tree yet, and this repo has no hand-written-markdown renderer either** — `scripts/pdoc.py` builds the API reference and nothing else, so these pages are read as markdown. Saying so is the point: a reader expecting a portal and finding markdown has been told something false.

## The family's pages

| page | what it covers |
|---|---|
| [The three participants](../../../lab-commons/docs-src/dev/the-three-participants.md) | Who does what: the human, the concurrent dev agents, the coordinator that absorbs their work |
| [The verdict model](../../../lab-commons/docs-src/dev/verdict-model.md) | What a verdict IS, why it starts INCONCLUSIVE, the three tiers and the destination each one gates |
| [Testing discipline](../../../lab-commons/docs-src/dev/testing-discipline.md) | The lightweight/heavy partition, xfail never skip, one test session at a time |
| [Branch layers](../../../lab-commons/docs-src/dev/branch-layers.md) | What a lane, an integration branch and `main` each PROVE, and why one cannot substitute for another |
| [Alignment](../../../lab-commons/docs-src/dev/alignment.md) | The framework every branch develops against so an integration conflicts on CODE, never on bookkeeping |
| [Fan-out](../../../lab-commons/docs-src/dev/fanout.md) | Parallel lanes in worktrees: environments, subagent bases, the shared hazards, landing |
| [The shared checkout](../../../lab-commons/docs-src/dev/shared-checkout.md) | Where a tool's correctness argument stops transferring when the checkout is not exclusive |
| [Killed runs and orphans](../../../lab-commons/docs-src/dev/orphans.md) | What survives a stopped run, why a lock does not time out, and the census a reaper needs |
| [Box resources](../../../lab-commons/docs-src/dev/box-resources.md) | What one workstation rations, the four defects measured in doing it by hand, and the broker shape |
| [The forge](../../../lab-commons/docs-src/dev/forge.md) | How `main` is protected on a self-hosted forge: the push whitelist, the status check, branch disposal |
| [Retirement](../../../lab-commons/docs-src/dev/retirement.md) | An archive and a retired-spelling registry are a PLACE and a RULE, and why the rule cannot live in the place |
| [The docs pipeline](../../../lab-commons/docs-src/dev/docs-pipeline.md) | The three properties a docs builder must hold, and why an unbuilt sub-site is announced rather than silent |

## What this repo has, and what it does not

- **A mechanism a page describes is not automatically present here**, and assuming it is would be an instruction this repo cannot follow.
- PRESENT: the Makefile's lint and test targets, `scripts/pdoc.py`, and the lane and worktree hazards, which are properties of the box and of git rather than of any repo.
- NOT PRESENT: a tiered gate runner with a case library and live solver engines. That one is motronics-studio's, and the portable substitute the family ships is `python -m lab_commons.dev.verify`, which produces the same verdict shape without needing any of it.
- This repo's own subject — the optimisers, their operators and their problem definitions — has no page yet; when one is written it belongs here rather than in the shared tree, because its subject changes the moment you change repos.
