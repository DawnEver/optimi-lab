"""optimi-lab ADOPTS the shared rules registry -- 51 of 58, with the other 7 named as gaps.

WHY THIS FILE EXISTS. `dev_lab.dev.rules` holds one canonical copy of the development rules
that are not about any repo's subject. Before this file, optimi-lab had no `.claude/` at all: its
rules were whatever the last reader remembered, so there was no statement to violate and nothing
that could red. The registry's `assert_adopted` is what turns "we follow the house rules" into a
check -- it resolves each rule this repo claims to enforce against a mechanism that must be a
TRACKED file in THIS tree.

WHAT IT PROVES, stated exactly, because overclaiming it would be the defect it guards against: every
rule is ACCOUNTED FOR -- enforced with a live mechanism, or declared absent -- and no ID is a typo.
It does NOT run the mechanisms and cannot say they pass. An existing red mechanism is still an
existing mechanism; whether it is green is the suite's business.

BOTH SETS ARE PINNED BY NAME AND TYPED OUT. Deriving `_ENFORCED` from `RULES` would silently absorb
a rule added upstream instead of making this repo decide about it -- and an integer pin could not
say WHICH rule moved, which is the failure mode the family already paid for once.

WHY NOT ALL OF THEM. optimi-lab is a LIBRARY: it has no network verb of its own, no persisted
format, no production CLI, no vendor arbitration and no physical units. Adopting
those rules by writing a mechanism that checks nothing would be worse than the gap -- so they are
DECLARED ABSENT, by name, under a ceiling that may only go down. Each entry in `_ABSENT` has its
reason in the comment above it, and on 2026-09-17 every one of those reasons was RE-MEASURED rather
than re-read: three of them did not survive it and became mechanisms.

TWO RULES ARRIVED UPSTREAM AND WERE ADOPTED ON THE SAME DAY, 2026-09-16:
`NO-CJK-IN-TRACKED-SOURCE` and `INJECTED-DOC-WIDTH-CEILING`, each driving the SHARED scanner in
`dev_lab.dev.cjk` / `dev_lab.dev.docwidth` over this tree's own corpus. Both measured CLEAN
here (53 files scanned, zero CJK; two injected docs, zero lines past 120 columns), so both declared
sets are EMPTY and each carries a FLOOR pinned at its measurement -- an empty declaration over an
unread corpus is the one way those guards could be vacuous. Two more arrived the same day with no
subject in this repo and went to `_ABSENT`, which is why the ceiling below moved UP; that move has
its own note where the number lives.

THE AUDIT OF 2026-09-15, because a count that moved without saying why is the thing this file
exists to prevent. The first version of this file claimed 12 of 28. Three specific overstatements
were alleged and each was checked against the code rather than taken on trust:

* **Three rules rest on a module that cannot be collected** -- REFUTED. The module is this one and
  the three are `NAMED-SETS-NOT-COUNTS`, `ESCAPE-HATCH-CEILING`, `RATCHET-TWO-SIDES`. It imports
  `dev_lab.dev`, which is genuinely missing from one shared venv's STALE installed
  `lab_commons` -- but that is a fact about an install, not about the file, the dev extra declares
  the dependency, and a missing import here is a COLLECTION ERROR, which is the loudest failure
  pytest has. A mechanism that reds when its dependency is absent is behaving.
* **`_SURFACE` plants nothing** -- CONFIRMED, and it was cited for SEVEN rules. Fixed rather than
  demoted: its five scans now route through `assert_floor`, `second_homes`, `oversize`, `skipping`
  and `pin_gap`, and a control drives each of those REAL functions on planted input.
* **`OVERSIZE_PINS` and `LAZY_IMPORT_PINS` are empty, so the ratchet never executes** -- CONFIRMED.
  Both are empty as correct MEASUREMENTS, so the live comparison ran an empty set against an empty
  set. Seeding a fake pin would be the waiver-nothing-uses side of the same defect; the two-sided
  control on `pin_gap` is what makes the empty answer mean clean rather than unread.

A FOURTH one was found that the review had not named, and it is the one that cost a rule.
`NAMED-SETS-NOT-COUNTS` cited three modules that merely OBEY it: nothing refused replacing a named
pin with an integer. A rule exemplified by the code is not a rule enforced by it, so it now cites
`test_a_pin_is_a_named_set.py`, which refuses a numeric module-level constant unless its name
declares it a threshold.

FOUR GAPS CLOSED in the same pass, each with a floor and a planted control: `FIX-THE-CAUSE` (the
one-name-one-home scan was already here and unclaimed), `RETIRED-NAMES-REGISTERED` (seven names
the two breaking refactors deleted, and the three docstrings still writing them), `MEMORY-SHAPE`
and `DOCS-SPLIT`. So the ceiling falls 16 -> 12 -- and had the audit gone the other way it would
have RISEN, which is the correct direction for an honesty pass and is not a thing to hide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.profile import RepoProfile
from dev_lab.dev.rules import RULES, Adoption, TestPath, assert_adopted, unadopted, waived

_ROOT: Final = Path(__file__).resolve().parents[2]

_PROFILE: Final = RepoProfile(
    app_name='optimi_lab',
    package='optimi_lab',
    root=_ROOT,
    lint_config=_ROOT / 'pyproject.toml',
)

_PURITY: Final = 'tests/architecture/test_the_runtime_stays_pure.py'
_SURFACE: Final = 'tests/architecture/test_the_public_surface_is_declared.py'
_STRANGER: Final = 'tests/architecture/test_a_declared_set_refuses_a_stranger.py'
_ADOPTION: Final = 'tests/architecture/test_optimi_lab_adopts_the_shared_registry.py'
_PIN_SHAPE: Final = 'tests/architecture/test_a_pin_is_a_named_set.py'
_RETIRED: Final = 'tests/architecture/test_a_retired_spelling_stays_retired.py'
_MEMORY: Final = 'tests/architecture/test_memory_lives_under_a_date.py'
_RULES_RATCHET: Final = 'tests/architecture/test_the_rules_pages_are_a_ratchet.py'
_NO_CJK: Final = 'tests/architecture/test_no_cjk_in_tracked_source.py'
_DOC_WIDTH: Final = 'tests/architecture/test_injected_doc_width_ceiling.py'
_HOOKS: Final = 'tests/architecture/test_the_declared_hooks_are_installed.py'
_BOUNDED: Final = 'tests/architecture/test_a_bounded_wait_names_its_remedy.py'
_ORIGIN: Final = 'tests/architecture/test_this_checkout_is_visible_on_origin.py'
_BOX: Final = 'tests/architecture/test_the_verdict_run_takes_the_box.py'
_DOOR: Final = 'tests/architecture/test_the_dependency_door_is_wired.py'
_DOOR_PORT: Final = 'scripts/dep.py'
_INSTALL_DOORS: Final = 'tests/architecture/test_the_install_doors_deliver_the_declared_kit.py'
_TOLERANCE: Final = 'tests/architecture/test_a_relative_tolerance_carries_its_floor.py'
_GUARD: Final = 'tests/architecture/test_the_agent_guard_is_live.py'
_GUARD_RULES: Final = 'scripts/deny_rules.py'
_CODEPLACE: Final = 'tests/architecture/test_code_lives_only_where_code_belongs.py'
_FAMCONFIG: Final = 'tests/architecture/test_the_family_config_is_rendered.py'
_PROJECT_FILES: Final = 'tests/architecture/test_every_project_file_has_one_source.py'
_NO_REFLECTION: Final = 'tests/architecture/test_no_reflection.py'
_ALLOW: Final = 'tests/architecture/test_no_allow_entry_names_a_denied_shape.py'
_ADMISSION: Final = 'tests/architecture/test_push_admission_is_delegated.py'
#: The branch-flow declaration: `[tool.lab_commons.branchset]`, `[parts]`, `[tests]`, `[integrator]`, `[hooks]`.
_PYPROJECT: Final = 'pyproject.toml'

#: THIS REPO'S MECHANISMS -- a rule ID to the tracked file(s) in THIS tree that refuse a violation
#: of it. Not the registry's own rows: those name motronics paths, which resolve in exactly one
#: checkout on earth and would grade this repo against a tree it does not have.
_MECHANISMS: Final[dict[str, tuple[str, ...]]] = {
    # The package docstring claims a three-distribution runtime; the guard reads the dependency
    # table and the import graph instead of believing it.
    'DECLARATION-LIES': (_PURITY,),
    # Every guard is called on PLANTED input: a dropped dependency's import, a silent module, a
    # second home for one name, a deferred import, an oversize module, a skip, a count pin, a
    # forked memory path, a stale frontmatter date, a resurrected spelling, each ratchet direction,
    # an entry point that names the box without entering it, a verdict held while the environment is pushed,
    # a tolerance written as a bare ratio.
    'PLANTED-CONTROL': (
        _PURITY,
        _SURFACE,
        _STRANGER,
        _PIN_SHAPE,
        _RETIRED,
        _MEMORY,
        _RULES_RATCHET,
        _BOX,
        _DOOR,
        _TOLERANCE,
    ),
    # No scanner reports green over a file list shorter than its measured floor, and `assert_floor`
    # is one function with one control rather than a line repeated per scan.
    'FLOOR-ON-EVERY-SCAN': (_PURITY, _SURFACE, _PIN_SHAPE, _RETIRED, _MEMORY, _RULES_RATCHET, _TOLERANCE),
    # No upper bound on a runtime requirement, and lab-commons is a bare git+https URL, never a sha.
    'LATEST-DEPENDENCIES': (_PURITY,),
    # `__all__` in every shipped module, and one public name with exactly one home.
    'PUBLIC-SURFACE-DECLARED': (_SURFACE,),
    # No import inside a function body; the exemption map is empty and is checked both ways.
    'NO-LAZY-IMPORT': (_SURFACE,),
    # A size band with the oversize set pinned by name rather than by a count.
    'MODULE-SIZE-ALARM': (_SURFACE,),
    # A skip records nothing, so the suite may not contain one.
    'XFAIL-NOT-SKIP': (_SURFACE,),
    # An unregistered surrogate key or sample kind RAISES `Refusal` and the message quotes the set.
    'UNSUPPORTED-RAISES': (_STRANGER,),
    # A numeric module-level constant in any architecture module is REFUSED unless its name
    # declares it a threshold. The three modules previously cited here merely obeyed the rule,
    # which is exemplification and not enforcement -- see the audit note in the docstring.
    'NAMED-SETS-NOT-COUNTS': (_PIN_SHAPE,),
    # The absent set carries `_ABSENT_CEILING`, which may only go down.
    'ESCAPE-HATCH-CEILING': (_ADOPTION,),
    # Two-sided pins: an arrival reds, and so does a waiver nothing uses. `pin_gap` and
    # `ratchet_breaks` each answer in BOTH directions and each has a control that drives them so,
    # which is what the live empty pin sets cannot do for themselves.
    'RATCHET-TWO-SIDES': (_ADOPTION, _SURFACE, _RULES_RATCHET),
    # One public name has one home: a second spelling is how a patched symptom outlives its cause,
    # and the scan reads the merged tree, which is the only place the clash is visible.
    'FIX-THE-CAUSE': (_SURFACE,),
    # Seven names the two breaking refactors deleted, each with its replacement, and a scan that
    # reaches PROSE -- where a retired spelling survives longest -- as well as code.
    'RETIRED-NAMES-REGISTERED': (_RETIRED,),
    # `.claude/memory/YYYY/MM/DD/`, plus the frontmatter date agreeing with the path it sits on.
    'MEMORY-SHAPE': (_MEMORY,),
    # The always-loaded rule pages are pinned per file, ratcheting down only.
    'DOCS-SPLIT': (_RULES_RATCHET,),
    # The shared scanner over this tree's whole tracked corpus, with the declared set EMPTY --
    # measured 2026-09-16 at 53 files scanned and zero CJK characters. Adopted at zero, so the
    # declaration's job is to refuse the FIRST arrival rather than to record a debt.
    'NO-CJK-IN-TRACKED-SOURCE': (_NO_CJK,),
    # The same shape over the injected-doc corpus (here: the two `.claude/rules/` pages), measured
    # 2026-09-16 at zero lines past 120 columns. It is the WIDTH dimension the line-count ratchet
    # in `_RULES_RATCHET` cannot see, over the same pages.
    'INJECTED-DOC-WIDTH-CEILING': (_DOC_WIDTH, _RULES_RATCHET),
    # CLOSED 2026-09-16, and it was VIOLATED here rather than merely unenforced: the reason on
    # record said this repo has no hooks directory, while the hook declaration named 19
    # hooks and ZERO were installed in the directory git consults. The mechanism is the family's
    # (`dev_lab.dev.hook_install`, which ASKS GIT for that directory instead of string-building
    # it); what is this repo's is the tree asked about, the one-stage named pin, and controls that
    # plant absent / installed / foreign in a real repository.
    'HOOKS-ARE-WIRED': (_HOOKS,),
    # CLOSED 2026-09-16. The old reason -- "no gate runner" -- answered about a RUNNER while the
    # rule is about a WAIT, and this package starts subprocesses. The wall is driven against a
    # planted GRANDCHILD that really inherits stdout (the shape `subprocess.run`'s timeout cannot
    # bound), the reaper is asked to kill this process's own tree and must refuse while still
    # reaping a planted stranger, and the refusal must name a DIFFERENT remedy in each of its three
    # states.
    'REFUSAL-NAMES-THE-REMEDY': (_BOUNDED,),
    # CLOSED 2026-09-16, and the first honest run of it found real debris: `fix/p0-integration-
    # blockers`, a branch with no origin counterpart, no upstream and no worktree. The old reason
    # was a wrong conclusion from a true premise -- the push obligation IS a fact about origin, and
    # git answers it without a network call. Named sets on both sides, and the stale-local-trunk
    # trap planted on real repositories with a real remote.
    'SHARED-CHECKOUT': (_ORIGIN,),
    # CLOSED 2026-09-17, and the reason on record was refuted by this repo's own entry point: it
    # said there is "no runner to queue on a box lock", which answers about a RUNNER while
    # `dev_lab.dev.verify` -- the only verdict command this repo has -- already took the box
    # across its run. So this repo was already a party to the rendezvous and had nothing that could
    # notice the box being dropped. The mechanism is the family's (`lab_commons.box`, since 96a90c8);
    # what is local is the assertion, read off the installed source with a planted control, that
    # THIS repo's entry point still routes its run through `verifybox.boxed`.
    'ONE-BOX-ONE-LOCK': (_BOX,),
    # CLOSED 2026-09-17. The old reason said nothing here issues a verdict another party cites;
    # `.verify/verify-*.log` carries an `env=` key computed when the run ENDS, which is exactly the
    # invariant the door exists for, and `dev_lab.dev.dep`'s own docstring named this repo as
    # the one supplying neither adapter. Since lab-commons 96a90c8 the exclusion is the kit's own
    # (the environment seat `verify` holds), and `scripts/dep.py` supplies the one repo answer left --
    # the verdict logs as the anchors -- so the report renders no gap, against a control port that
    # declares none and must render it.
    'ENV-MUTATION-THROUGH-THE-DOOR': (_DOOR, _DOOR_PORT),
    # ADOPTED 2026-09-17, and this repo is the one that needed NO repair -- which is a measurement
    # rather than a skip. All 22 installer commands in its declared door files are `uv pip install` /
    # `pip install`, both of which re-resolve a bare git URL every time; none consults `uv.lock`, so
    # none can serve the `0.2.2.dev26+gba3bf6de` this checkout's untracked lock still pins. The guard
    # is what keeps that true: a `uv sync` or a bare `uv run` added to a target or a hook reds.
    'INSTALL-DOOR-DELIVERS-THE-DECLARATION': (_INSTALL_DOORS,),
    # CLOSED 2026-09-17, in the one half of the rule that has a subject in a unitless package. The
    # unit half of the old reason still stands and is untouched; what does not need a unit is
    # `never write a rel= without the abs= floor it is combined with`, and this package compares
    # quantities that pass through zero, where a ratio admits only exactly zero. Every tracked
    # `approx` call is PARSED, adopted at zero (two call sites, both bare) with a floor on the
    # call sites read and a control that plants all four shapes.
    'TOLERANCE-CARRIES-A-UNIT': (_TOLERANCE,),
    # CLOSED 2026-09-17, and the reason on record was RETIRED BY ITS OWN TERMS rather than refuted.
    # It said the shared deny registry existed but "its ENGINE does not: the engine the registry's
    # own recipe points at is a `.js` file in another repo", and named the missing piece as "an
    # engine in the family rather than a file here". That engine landed as
    # `dev_lab.dev.agenthooks`, installed and verified by `dev_lab.dev.agent_guard`, so the
    # blocker is gone. Measured here before the change: NOTHING DECLARED, exit 2 -- no settings, no
    # matcher, no engine. The mechanism is the family's; what is this repo's is WHICH rules it may
    # honestly ship, which is `scripts/deny_rules.py`: five shipped, two (GIT-NETWORK-VERB,
    # RAW-PROCESS-KILL) declared absent because their exits -- a retry wrapper, a typeable
    # process-tree killer -- do not exist here. `_GUARD` pins the shipped set BY NAME, drives the
    # real engine under node in both directions, and its sanctioned row is this repo's own verdict
    # command, so the guard cannot seal the only road out of itself.
    #
    # FORGE-THROUGH-THE-DOOR rests on the SAME two files: its three FORGE-WRITE-* deny rows ship
    # through this guard, so a `gh`/`tea`/raw-API write is refused live and the exit they name is
    # `python -m dev_lab.dev.forge`, the client that stamps which machine and agent wrote.
    **dict.fromkeys(('AGENT-GUARD', 'FORGE-THROUGH-THE-DOOR', 'INJECTED-TEXT-IS-PROGRESSIVE'), (_GUARD, _GUARD_RULES)),
    # ADOPTED 2026-10-01 with `dev_lab.dev.codeplace`: the filesystem walk over this tree's
    # declared code roots. The mechanism is the family's; the roots and the pruned generated trees
    # are this repo's.
    'CODE-IN-CODE-ROOTS': (_CODEPLACE,),
    # ADOPTED 2026-10-01 with the forge's status and issue doors. This repo's ONE verdict command is
    # the `verify:` recipe, `python -m dev_lab.dev.verify`, which posts `lab/gate` on a clean HEAD
    # and nothing for INCONCLUSIVE; the Makefile base requires that exact line, so a recipe that grew
    # `--no-status` (or stopped calling verify) reds in `_FAMCONFIG`.
    'VERDICT-AS-STATUS': ('Makefile', _FAMCONFIG),
    # The issue state is derived from `Refs #N` / `Closes #N` in commit messages; this repo's half is
    # that its commits are SPELLED so the derivation can see them -- the base's `issue-ref` commit-msg
    # row, taken as is by `[tool.lab_commons.hooks]` and installed per `_HOOKS`. No label or field in
    # this tree keeps an issue's state.
    'ISSUE-IS-INTENT': (_PYPROJECT, _HOOKS),
    # ADOPTED with lab-commons 9ce57c3. PREVENT is the `WORKTREES-STAY-INSIDE` deny row rendered by
    # `scripts/deny_rules.py`; DETECT is the family detector over `git worktree list`; IGNORE is the
    # base's `.claude/` section in `.gitignore`.
    'WORKTREES-STAY-INSIDE': (_PROJECT_FILES, _GUARD_RULES, '.gitignore'),
    # ADOPTED with lab-commons 9ce57c3: every managed file is rendered from its base plus
    # `tests/architecture/_famconfig.py`, every config table is its section base plus this repo's
    # SectionDelta, the dev index holds the live pointer table, and `OWNED_HERE` is empty.
    'PROJECT-FILES-HAVE-ONE-SOURCE': (_PROJECT_FILES, 'tests/architecture/_famconfig.py'),
    # ADOPTED with lab-commons a2f50b9 at zero sites: the family scanner over every tracked `*.py`,
    # all four banned calls plus `__getattr__`, with an empty two-sided allow-set.
    'NO-REFLECTION': (_NO_REFLECTION,),
    # ADOPTED with lab-commons ec5e8c5: `[tool.lab_commons.branchset]` declares trunk `main` and no
    # session branches, and the family famtest polices origin and this box against it.
    'ONE-BRANCH-PER-SESSION': ('pyproject.toml', _ORIGIN),
    # ADOPTED with lab-commons dd33312: `scripts/deny_rules.py` renders one narrow allow row per
    # family door and tracked script entry point, and the Codex rules from the same table; the allow
    # test holds both files current and the block between its measured floor and headroom.
    'AUTO-MODE-RUNS-THE-DOORS': (_ALLOW, _GUARD_RULES, '.codex/rules/family-doors.rules'),
    # The subagent-scoped deny row SUBAGENT-NO-PUSH ships from `scripts/deny_rules.py` and is pinned by
    # name in `_GUARD`; it is the half of MAIN-SESSION-PUBLISHES this repo can refuse: a subagent's
    # push is denied, so only a main session publishes. REPORTS-LIVE-IN-MEMORY is a deny row of the
    # same guard (a shell redirect writing Markdown under `output/`), arrived with lab-commons 96a90c8.
    **dict.fromkeys(('MAIN-SESSION-PUBLISHES', 'SUBAGENT-NO-PUSH', 'REPORTS-LIVE-IN-MEMORY'), (_GUARD, _GUARD_RULES)),
    # ADOPTED 2026-10-18 with `[tool.lab_commons.hooks]` (the family base as is). The commit-msg
    # `merge-audit` row refuses committing a merge that loses or rewrites a test unnamed; the pre-push
    # `session-branch` row (`dev_lab.dev.sessionbranch`) refuses publishing a session branch from
    # another ref, a non-fast-forward, and a non-merge first-parent commit of `integrate/main` past
    # `[tool.lab_commons.integrator] merges_only_since`. `_HOOKS` pins the stages and their shims.
    # ONE-WRITER-PER-SESSION-BRANCH is git's own compare-and-swap plus that same fast-forward row.
    **dict.fromkeys(
        (
            'MERGE-DEVIATIONS-NAMED',
            'PUBLISH-IS-THE-LOCAL-REF',
            'SESSION-HISTORY-ONLY-GROWS',
            'INTEGRATION-IS-MERGES-ONLY',
            'ONE-WRITER-PER-SESSION-BRANCH',
        ),
        (_PYPROJECT, _HOOKS),
    ),
    # ADOPTED 2026-10-18 with `[tool.lab_commons.parts]`, `[tests]` and `[integrator]`: a verdict is a
    # `tests/<part>` the forge produces from the pushed sha (`produce` runs `dev_lab.dev.verify
    # --part`), `main` moves only by `integrate promote` on every part a full PASS, and the admission
    # table is the kit's with no local copy (`_ADMISSION`). VERDICT-BAR-IS-THE-INCREMENT was on record
    # as absent because no increment existed for a bar to be; the trunk gate is that increment.
    **dict.fromkeys(
        ('VERDICTS-FROM-THE-FORGE', 'TRUNK-MOVES-BY-FAST-FORWARD', 'VERDICT-BAR-IS-THE-INCREMENT'),
        (_PYPROJECT, _ADMISSION),
    ),
    # ADOPTED 2026-10-18: `forge sync` merges `integrate/main` only, and `forge integration` retargets a
    # PR opened against `main` -- both read `[tool.lab_commons.integrator]`. MERGE-OR-DELETE is the
    # branch-set apply over `[tool.lab_commons.branchset]` (`prune_merged = true`), policed by `_ORIGIN`.
    **dict.fromkeys(('DOWNSTREAM-SYNCS-INTEGRATION', 'PRS-TARGET-INTEGRATION'), (_PYPROJECT,)),
    'MERGE-OR-DELETE': (_PYPROJECT, _ORIGIN),
    # ADOPTED 2026-10-23 with `pr_merges_since` in `[tool.lab_commons.integrator]` and `pr/*` declared
    # ephemeral in the branch set: the forge alone merges a PR into integration (`forgemerge`), its review
    # advises (`forgereview`), admission refuses a second or stacked PR (`pradmit`) and a docs-only head
    # carries its parts (`forgecarry`) -- all server-side steps over this declared table, policed by `_ORIGIN`.
    **dict.fromkeys(
        (
            'INTEGRATION-TAKES-ONLY-FORGE-MERGES',
            'PR-REVIEW-IS-ADVISORY',
            'ONE-SESSION-ONE-PR',
            'NO-STACKED-PRS',
            'DOCS-ONLY-CARRIES',
        ),
        (_PYPROJECT, _ORIGIN),
    ),
}

_ENFORCED: Final = frozenset(_MECHANISMS)

#: NOT ENFORCED HERE, each with its reason -- and every reason below was RE-MEASURED on 2026-09-17
#: rather than re-read, because a reason that has never been re-measured is a claim and not a fact.
#: That pass is what moved three names out of this set; these eight survived it. Two kinds:
#:
#: * NO SUBJECT IN THIS REPO, measured: BAR-IS-A-CONSTANT and IMPLEMENT-EVERYTHING (no reference
#:   ladder and no accuracy matrix -- a metric here is exact arithmetic, not a measurement against a
#:   vendor, and there is no capability table a combination could be kept out of);
#:   PRODUCTION-ENTRY-POINT (the API IS the entry point and `pyproject.toml` declares no console
#:   script; the enforceable half of the rule is the placement scan, and `scripts/` holds two files,
#:   both of which drive a family mechanism and neither of which imports `optimi_lab` -- a corpus
#:   that small makes a placement guard's floor the guard's whole content);
#:   UNITS-GO-THROUGH-PINT (`Variable.unit` is a free-text LABEL this package never computes with,
#:   and `pyproject.toml` declares numpy/scipy/numerics-lab/lab-commons and nothing else -- pint is a dependency
#:   this library may not take, and narrowing some other scan to say otherwise would be a fake
#:   closure); REGISTRY-OWNS-THE-DECISION (there is one registry,
#:   `src/optimi_lab/model/registry.py`, the surrogate names already live in it, and
#:   `test_a_declared_set_refuses_a_stranger.py` already refuses an unregistered key -- what is
#:   missing is a SECOND kind to branch on, not a guard).
#:   FORMATS-UPGRADE-ON-READ (arrived with lab-commons 96a90c8, measured 2026-10-18: `git grep` over
#:   `src/` finds no `json.dump`, `pickle`, `np.save`, `write_text` or schema field -- this library
#:   persists no format of its own, so there is no older schema for a reader to upgrade).
#: * WORKFLOW MACHINERY THIS REPO GENUINELY DOES NOT HAVE, measured at the filesystem rather than
#:   asserted: NETWORK-RETRY-THEN-REPORT (this repo issues NO network verb of its own: re-measured
#:   2026-10-18, every `git clone/fetch/pull/push/ls-remote` spelling under `scripts/`, `src/` and
#:   `tests/` is a STRING FIXTURE inside `test_the_agent_guard_is_live.py`, fed to the deny engine
#:   rather than executed, so there is no verb here for `dev_lab.dev.netverb` to wrap).
#: The fourth kind is gone: the REAL GAPS this set used to hold (FIX-THE-CAUSE,
#: RETIRED-NAMES-REGISTERED, MEMORY-SHAPE, DOCS-SPLIT) were closed on 2026-09-15 and their names
#: were deleted here in the same edit that lowered the ceiling, which is the only way this number
#: is allowed to move.
_ABSENT: Final = frozenset({
    'BAR-IS-A-CONSTANT',
    'IMPLEMENT-EVERYTHING',
    'PRODUCTION-ENTRY-POINT',
    'UNITS-GO-THROUGH-PINT',
    'REGISTRY-OWNS-THE-DECISION',
    'FORMATS-UPGRADE-ON-READ',
    'NETWORK-RETRY-THEN-REPORT',
    # ARRIVED 2026-10-23 with the dev-lab cut-over, NOT YET MEASURED against this tree: the scratch
    # rule that replaced the archive-or-promote one (no declared `scratch/` README anchor here), the
    # stray-file census, the concept-split naming rule and the ephemeral-citation scan (its pre-commit
    # row runs; no forge-side test binds it here yet).
    'SCRATCH-IS-NEVER-COMMITTED',
    'SPLIT-BY-CONCEPT',
    'STRAY-FILE-REFUSED',
    'TRACKED-CITES-NO-EPHEMERAL',
})

#: The ceiling on the gap, MEASURED the day this file was written. It may only go DOWN FOR A RULE
#: THIS REPO ALREADY CARRIED: the hand that builds a mechanism deletes the name above and lowers
#: this number in the same edit.
#:
#: IT MOVED UP ONCE, 12 -> 14 on 2026-09-16, and the reason is stated here rather than left to be
#: read off the digit. The registry grew by FOUR rules that day. Two of them --
#: NO-CJK-IN-TRACKED-SOURCE and INJECTED-DOC-WIDTH-CEILING -- were ADOPTED in the same commit, with
#: mechanisms above, so they never touched this number. The other two are about a shared box and a
#: shared environment, neither of which exists here, and a ceiling that cannot admit an upstream
#: rule with no subject here would be cleared by the only move left: deleting it from `_ABSENT`,
#: which is the silent gap this file exists to prevent. So the ceiling absorbs the two it must and
#: nothing else -- every gap this repo could close is still on record, and none was reopened.
#:
#: AND DOWN AGAIN, 14 -> 11 later the same day, when three of the six workflow rules stopped being
#: machinery this repo does not have. Their MECHANISMS moved into `dev_lab.dev`
#: (`hook_install`, `bounded`, `checkout`) and this repo wired each one to its own tree.
#: HOOKS-ARE-WIRED was the sharpest: the reason on record claimed there was no subject here, and
#: there were 19 declared hooks with none installed.
#:
#: AND 11 -> 8 on 2026-09-17, on the same method applied to the rows that were left: MEASURE the
#: premise rather than reading it. Two of the three were the same mistake as HOOKS-ARE-WIRED in a
#: different spelling -- a reason that answered about machinery this repo does not have
#: (a gate runner, a shared box) while the rule is about a property this repo's ONE verdict command
#: already has (it takes the box seat; it writes a log carrying an `env=` key). The third,
#: TOLERANCE-CARRIES-A-UNIT, was true as far as it went and did not go far enough: the unit half has
#: no subject here and the `rel=` without `abs=` half needs no unit.
#:
#: FOUR ROWS WERE RE-MEASURED AND LEFT OPEN, which is the other honest outcome and not a lesser one.
#: The reasons above each of them now state a measurement rather than an assertion.
#:
#: AND 8 -> 7 later on 2026-09-17, for AGENT-GUARD, on a third route: neither a refuted premise nor
#: a re-measured one, but a reason that NAMED ITS OWN EXPIRY. It said the blocker was that the deny
#: engine lived in another repo as a `.js` file and that the missing piece was "an engine in the
#: family rather than a file here" -- and the family shipped one. A reason that states the condition
#: under which it stops applying is the cheapest kind to retire, and writing them that way is why
#: this number could move today without anybody re-litigating the gap.
#:
#: AND 7 -> 8 on 2026-10-07, A RISE for a rule this repo had never carried, stated as 12 -> 14 was:
#: upgrading lab-commons (dfc2e61 -> bb9968e) brought MERGE-DEVIATIONS-NAMED, measured at the tree
#: and admitted with its reason above rather than left in neither set.
#:
#: AND 8 -> 9 on 2026-10-09, the same kind of rise: lab-commons 2cca293 brought
#: the then-current name of what is now VERDICTS-FROM-THE-FORGE, measured and admitted.
#:
#: AND 9 -> 7 on 2026-10-18, when optimi-lab declared the family branch flow (`[tool.lab_commons.parts]`,
#: `[tests]`, `[integrator]`, `[hooks]`) and lab-commons 96a90c8 retired two IDs this file cited.
#: MERGE-DEVIATIONS-NAMED, VERDICT-BAR-IS-THE-INCREMENT and the retired verdict row (now
#: VERDICTS-FROM-THE-FORGE) left for mechanisms: each reason named the missing push wiring, and the
#: hook table installs it. FORMATS-UPGRADE-ON-READ arrived and was measured absent with its reason.
#:
#: AND 7 -> 11 on 2026-10-23, A RISE: moving the kit from lab-commons to dev-lab brought nine rules
#: this repo had never carried (SCRATCH-IS-NEVER-COMMITTED replacing an ID cited above); the five of
#: the PR flow were adopted with `pr_merges_since`, and the other four are in `_ABSENT` by name rather
#: than in neither set; closing one lowers this number.
_ABSENT_CEILING: Final = 11


def _adoption() -> Adoption:
    """The repo's adoption: its own mechanisms, plus the gaps it has on record."""
    mechanisms = {rule: tuple(TestPath(path) for path in paths) for rule, paths in _MECHANISMS.items()}
    return Adoption(app_name='optimi_lab', mechanisms=mechanisms, declared_absent=_ABSENT)


def test_optimi_lab_adopts_the_shared_registry() -> None:
    """THE CHECK. Every rule is enforced here by a tracked mechanism, or named as a gap."""
    assert_adopted(_PROFILE, _adoption())


def test_the_two_sets_account_for_the_registry_exactly() -> None:
    """A rule added upstream must be typed into one set or the other before anything else passes."""
    accounted = _ENFORCED | _ABSENT
    live = {rule.id for rule in RULES}
    assert accounted == live, (
        f'rules this repo says nothing about: {sorted(live - accounted)}; names that are not rules: '
        f'{sorted(accounted - live)}. A new universal rule is a decision optimi-lab makes for itself, '
        f'not a line that appears in a list nobody read.'
    )


def test_the_absent_set_may_only_shrink() -> None:
    """The ceiling on the escape hatch: closing a gap lowers the number in the same edit."""
    assert len(_ABSENT) <= _ABSENT_CEILING, (
        f'{len(_ABSENT)} rules are declared absent, above the {_ABSENT_CEILING} ceiling. A waiver nobody '
        f'must justify is how a check reaches zero without anything being fixed.'
    )


def test_the_gap_is_visible_rather_than_silent() -> None:
    """A declared gap and an unexamined rule are different facts, and only one is acceptable."""
    adoption = _adoption()
    assert unadopted(RULES, adoption) == (), 'a rule is cited by neither set; the check above says which.'
    assert set(waived(RULES, adoption)) == _ABSENT


def test_every_mechanism_this_repo_names_is_a_file_in_it() -> None:
    """The control for the check above, read off the filesystem rather than off git.

    `assert_adopted` resolves a mechanism against `git ls-files`, which answers for what is
    COMMITTED. This answers for what is on disk. The two disagree exactly when a mechanism was
    written and never added -- a file nobody else has -- and that is a state worth naming directly.
    """
    missing = sorted({path for paths in _MECHANISMS.values() for path in paths if not (_ROOT / path).is_file()})
    assert not missing, (
        f'{missing} are named as mechanisms and do not exist. A rule whose mechanism is a path that '
        f"resolves to nothing is prose wearing a guarantee's clothes."
    )
