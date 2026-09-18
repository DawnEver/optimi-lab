"""WHERE EVERY DEV FILE'S MECHANISM LIVES -- this repo's half of the family migration, as data.

WHY THIS FILE EXISTS, AND THE MEASUREMENT THAT FORCED IT. `lab_commons.dev` publishes 30 public
modules, and this repo imports 16 of them. The migration that produced those 16 was never written
down anywhere in this tree, so the remaining files under `scripts/` and `tests/architecture/` were
UNMEASURED -- and an unmeasured migration reads exactly like a finished one. motronics-studio has
had the answering shape since 2026-08-11 (`tests/architecture/layering/_placement_*.py`, composed
into `PLACEMENT`: measured 2026-09-17 at 71 rows, 44 STAYS, 13 MOVES, 14 SPLITS). This is the same
declaration for this repo, and it is a DECLARATION ONLY: not one file is moved by the commit that
lands it.

THE TEST THAT DECIDES A ROW, taken from the family's and re-stated in this repo's nouns: **does this
code name `optimi_lab` or an optimisation concept?** If it does, it is this repo's and it STAYS. If
it does not, it is a mechanism and it MOVES. The third answer is the interesting one: a file that
does both must SPLIT, and naming the SEAM -- which half is family, which is local -- is more useful
than forcing it into a box it does not fit.

FOUR PROPERTIES ARE ENFORCED by `test_the_migration_boundary_is_declared.py`, and only the last
three are more than bookkeeping:

1. COMPLETENESS, WITH A FLOOR. Every runnable file under `scripts/` and every module under
   `tests/architecture/` must carry a row, and the scan that collects them must read at least
   `PLACEMENT_FLOOR` files. Without the first a new file is silently unplaced; without the second a
   walk that reached nothing reports what a fully classified tree reports.
2. A `STAYS` CLAIM MUST HAVE EVIDENCE -- the file names something this repo owns.
3. A `MOVES` CLAIM MUST SURVIVE THE CONVERSE -- its CODE names nothing this repo owns, because
   moving a file that does would carry an optimisation fact into a vendor-neutral package.
4. A `STAYS` OR `SPLITS` FILE MUST BE MOSTLY THIS REPO'S, BY DENSITY. Property 2 is a HIT test and a
   hit is one line; property 4 asks how much NON-repo code surrounds it. A label the author picks is
   not a measurement, so `SPLITS` answers to the SAME bar as `STAYS` -- an unfinished split is a
   `BELOW_THE_BAR` row carrying its numbers, never a gentler standard.

WHAT THE MEASUREMENT CANNOT SEE, stated so a reader does not supply "everything":

* IT COUNTS LINES, NOT MEANING, and property 4 reads CODE rather than prose: a docstring naming
  optimisation forty times counts zero.
* THE IDENTIFIER BOUNDARY CUTS BOTH WAYS. A repo fact written as a PATH, a NUMBER or a STRING
  LITERAL is invisible here. Five rows below are honest `BELOW_THE_BAR` entries for exactly that
  reason and say so.
* IT IS PYTHON-ONLY, which costs nothing here: measured 2026-09-17, `scripts/` holds three `.py`
  files and one `README.md`, and there is no shell script in either scanned tree. The wdg-lab twin
  of this table needs a named shell exemption; this one does not, and
  `test_the_scanned_trees_hold_no_unmeasurable_file` is what keeps that true rather than assumed.

THE `lab_commons.dev` MODULES THIS REPO DOES NOT IMPORT, and the ones whose absence is a FACT rather
than a gap (measured 2026-09-17 -- this tree imports `agent_guard`, `agenthooks`, `bounded`,
`boxlock`, `boxwait`, `checkout`, `cjk`, `dep`, `docsite`, `docwidth`, `hook_adoption`,
`hook_install`, `hooks`, `profile`, `rules`, `verify`):

* `selfbuild` and `shadow_build` -- NO SUBJECT, and the absence is correct rather than owed. Both
  are about a native extension rebuilt from the checkout you are standing in; measured 2026-09-17,
  this repo has no `rust/` tree, no `Cargo.toml`, no maturin and no compiled extension anywhere.
  There is nothing here to build a wheel from, so importing either would be a capability with no
  question to answer. The sibling wdg-lab has the subject and hand-rolled it.
* `netverb` -- NO SUBJECT IN THE SCANNED TREES. Measured 2026-09-17, no file under `scripts/` or
  `tests/architecture/` issues a network verb; the only git this tree runs is `git ls-files` and
  `git cherry`, which are local and reach no forge. `GIT-NETWORK-VERB` is declared absent in
  `scripts/deny_rules.py` for a DIFFERENT reason (no retry wrapper to name as a remedy), and the two
  readings agree.
* `ab_bench` -- SUBJECT PRESENT IN THE PACKAGE, ABSENT FROM THIS TABLE'S TREES. A library of
  interchangeable regressors and samplers is exactly the in-process seam `ab_bench` times, and one
  of its properties (a candidate that legitimately REFUSES an input another accepts) describes this
  package's registries. But measured 2026-09-17, nothing under `scripts/` or `tests/architecture/`
  times anything, interleaves anything or reports a median -- so there is no file to place, and
  saying where the subject actually lives is the row.
* `testfacts` -- NO HAND-ROLLED COPY HERE, AND NONE IS OWED, which is the difference from the
  sibling: that repo has a guard asking which modules SKIP and how many times, which is exactly the
  question `testfacts` answers, and it now delegates the whole reading. This tree reads its own test
  files with `ast` in `test_a_relative_tolerance_carries_its_floor.py` and
  `test_the_public_surface_is_declared.py` for a DIFFERENT question (an `approx` ratio with no floor
  under it, an `__all__` that resolves), neither of which `testfacts` answers. Correctly absent --
  the reason is the QUESTION being different, not the reading technique being different.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from lab_commons.dev.famtests import density, placement
from lab_commons.dev.famtests.placement import SPLITS, STAYS, Placement

__all__ = [
    'BELOW_THE_BAR',
    'CEILING_ADMITS',
    'CEILING_REFUSES',
    'DELEGATION_HOMES',
    'MINIMUM_ADMITS',
    'MINIMUM_REFUSES',
    'MIN_REPO_DENSITY_PCT',
    'NOT_PLACED',
    'NOUN',
    'OWN_MECHANISM_CEILING',
    'PLACEMENT',
    'PLACEMENT_FLOOR',
    'PLACEMENT_HEADROOM',
    'REPO_NOUNS',
    'RUNNABLE_SUFFIXES',
    'SCANNED',
    'SHELL_ROWS',
    'is_justified',
    'measure',
    'nouns_in_code',
    'nouns_in_prose',
    'placed_files',
    'repo_root',
    'stays_rows',
]


def repo_root() -> Path:
    """Resolve this checkout: `tests/architecture/_placement.py` sits two directories below it."""
    return Path(__file__).resolve().parents[2]


#: The two trees this roster answers for. `src/` is the shipped library and has its own guards;
#: this table is about DEV files, which is where the family migration happens.
SCANNED: Final = ('scripts', 'tests/architecture')

#: A runnable file is a Python module or a shell script. The `.sh` half is here because the SCAN
#: must be able to see one: this tree holds none today, and a suffix set that could not match a
#: shell script would report that fact whether or not it stayed true.
RUNNABLE_SUFFIXES: Final = ('.py', '.sh')

#: Never authored, or carrying no knowledge to place: build output and agent memory. `__init__.py`
#: is NOT here: the kit's walk always omits it, because it carries no knowledge and so follows
#: whatever its directory does.
NOT_PLACED: Final = ('__pycache__', '.claude')

#: Words that make a file THIS repo's rather than any repo's. A HIT IS EVIDENCE, NOT A VERDICT:
#: nothing here classifies anything, and every row below was written by a human who read the file.
#: Matched with identifier boundaries, so `sampler` does not fire on `resampler`.
REPO_NOUNS: Final = (
    'optimi_lab',
    'optimi',
    'optimization',
    'optimisation',
    'optimizer',
    'optimizers',
    'objective',
    'objectives',
    'pareto',
    'sampler',
    'samplers',
    'regressor',
    'regressors',
    'surrogate',
    'surrogates',
    'nsga',
    'moea',
)

#: THE ALTERNATION, built by `density.noun_pattern` with this repo's BOUNDARY POLICY stated rather
#: than baked into a regex nobody re-reads. `match_identifier_parts=False` is optimi-lab's answer and
#: the family does not agree on it: both labs and one motronics roster match WHOLE identifiers, so
#: `sampler` does not fire inside `resampler`, while motronics' scripts roster widened to identifier
#: PARTS after measuring a file at 0.00% with its noun sitting inside a longer name. The kit refuses
#: the empty vocabulary outright, which a hand-built alternation could not.
NOUN: Final = density.noun_pattern(REPO_NOUNS, match_identifier_parts=False)

#: WHAT COUNTS AS DELEGATION rather than this file's own mechanism. A line merely CALLING something
#: imported from here is wiring, and counting it scores a COMPLETED migration as new local code --
#: measured upstream on a real re-point as own 39 -> 40 where the honest reading is 39 -> 31.
DELEGATION_HOMES: Final = ('lab_commons',)

#: MEASURED 2026-09-17: `scripts/` holds 3 runnable files and `tests/architecture/` holds 20 modules
#: once this roster's own two files are counted, for 23 rows. RE-MEASURED 2026-09-18 at 27 rows,
#: the arrival being `test_the_roster_is_re_read_against_the_kit.py`; the floor does not move
#: with an arrival, because a floor with no room below it is a second pin on the population
#: rather than a refusal of an unread walk. The floor sits below that with room
#: for ordinary deletion and far above the zero a broken walk returns. Finding NOTHING is vacuous
#: rather than green.
PLACEMENT_FLOOR: Final = 18

#: THE OTHER SIDE OF ``PLACEMENT_FLOOR``, which no roster in this family ever wrote: how far past its
#: floor the population may grow before the number stops separating a classified tree from an unread
#: walk and must be RE-MEASURED. Today's reading is 27 - 18 = 9.
PLACEMENT_HEADROOM: Final = 15

#: THE SHELL ROWS, NAMED, AND EMPTY. `measure` parses Python, so a `.sh` file in a scanned tree
#: cannot be measured by section 4 at all -- and this tree declares `.sh` RUNNABLE while holding
#: NONE of them. Before this set existed, the day one arrived `stays_rows` would have handed it to
#: `ast.parse` and section 4 would have ERRORED rather than failed: the worst of the three outcomes,
#: because it names no file and reads as a broken test rather than as an unmeasured row. wdg-lab's
#: roster has the answering shape and this is it. EMPTY is legal and is the state to be in; what it
#: buys is that a `.sh` row leaves section 4 by NAME rather than by crashing it.
SHELL_ROWS: Final[frozenset[str]] = frozenset()


# --------------------------------------------------------------------------- THE MANIFEST

#: Keys are repo-relative POSIX paths -- DATA, so a row written on Windows matches on Linux.
PLACEMENT: Final[dict[str, Placement]] = {
    # ------------------------------------------------------------------ scripts/
    'scripts/deny_rules.py': Placement(
        STAYS,
        'A BINDER, and the file says so itself: the STATEMENT of every denied shape already moved to '
        '`lab_commons.dev.hooks`, and what is left is the REMEDY half, which is a fact about one '
        "tree -- `app_name='optimi_lab'`, the two rules this repo can offer an exit for, and the two "
        'it declares ABSENT (GIT-NETWORK-VERB, RAW-PROCESS-KILL). Measured 2026-09-17: own=17 lines, '
        'well under the 50-line binder ceiling. ITS wdg-lab TWIN IS NEAR-IDENTICAL and that is not '
        'move evidence but the opposite: the differences are exactly the three things that ought to '
        'differ -- the app name, the prose, and one call -- so both files are already the thin local '
        'half a finished migration leaves behind.',
    ),
    'scripts/dep.py': Placement(
        STAYS,
        'A BINDER over `lab_commons.dev.dep`: the env_key, the refusal while a verdict is in flight '
        'and the anchor retirement are all upstream, and this file supplies the only two answers '
        "that are not -- who holds this box's seat, and that this tree's verdict records are "
        '`.verify/verify-*.log`. Measured 2026-09-17: own=22 lines, under the ceiling. The seam is '
        'ALREADY CUT here, which is why this is a `STAYS` and not a `SPLITS`: `Port` is the shape '
        'upstream defined for exactly this residue, and this file is the residue.',
    ),
    'tests/architecture/test_the_install_doors_deliver_the_declared_kit.py': Placement(
        STAYS,
        'THE DECIDING FACT IS THE DOOR SET, WHICH IS FIVE PATHS IN THIS CHECKOUT AND NOTHING ELSE: '
        '`Makefile`, `README.md`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` and '
        '`scripts/dep.py`. Which files in a tree can move an environment is not portable -- the '
        'sibling lab has a git hook and an update script where this one has a CI workflow -- so the '
        "named set IS the deliverable. Everything around it is the family's: `scan_doors`, "
        '`floating_requirements`, `reverting` and `assert_doors_deliver` are all '
        '`lab_commons.dev.installdoor`, which is why this is a BINDER and a `STAYS` rather than a '
        '`SPLITS` -- the seam is already cut upstream, exactly as it is for `scripts/dep.py`. THE '
        'SIDE WAS MEASURED, NOT PICKED, 2026-09-17: own=21 repo=0 -> 0.00%, which is under the 3.0% '
        'bar and comfortably under the 50-line binder ceiling, so it is admitted by the BINDER arm '
        'alone and needs no BELOW_THE_BAR row. THE ZERO IS HONEST RATHER THAN A MISCLASSIFICATION: '
        'every repo fact in the file is a PATH STRING in `_DOORS` or a NUMBER in `_DOOR_FLOOR`, and '
        'an identifier scan cannot see either. MOVES was refused by the door set -- moving it would '
        'put five paths that are true of this checkout only into a package three repos share.',
    ),
    'scripts/pdoc.py': Placement(
        STAYS,
        'THE DRIVER ALREADY MOVED and this file is the record of it: the pdoc invocation, the image '
        'walk and the `check=False` that made an errored build indistinguishable from a real one are '
        'now `lab_commons.dev.docsite`. What is left is DATA -- the `optimi_lab` package, the edit '
        'URL, the logo, the version footer -- plus the browser branch that stays local so this '
        "repo's own test can patch the name in this repo's own namespace. Measured 2026-09-17: "
        'own=25 project=3 -> 12.00%, the densest file under `scripts/`, and under the ceiling too.',
    ),
    # ------------------------------------------------------------ tests/architecture/
    'tests/architecture/_placement.py': Placement(
        STAYS,
        'THIS TABLE ANSWERS FOR ITSELF, which is not decoration: a roster exempt from its own '
        'completeness check is the declaration-that-lies shape one level up. It is 23 rows of '
        'optimi-lab facts -- which registries, which retired spellings, which runtime, which rules '
        'pages -- and the noun density comes from the row KEYS and reasons, which are code here '
        'rather than prose. The measurement machinery beside them is the family shape and the '
        'module docstring says so. THE PREDICTION THIS ROW CARRIED IS WHAT HAPPENED: it said that if '
        '`lab_commons.dev` grew a `placement` module the machinery would go and the rows would stay. '
        '`famtests.placement` and `famtests.density` shipped on 2026-09-18 and the walk, the docstring '
        'blanker, the delegation reader, the `Density` class and `measure_density` are DELETED from '
        'here -- what is left of the machinery is five BINDERS supplying the arguments the kit refuses '
        'to default. The row stays STAYS rather than becoming a SPLITS because there is no longer a '
        'family half in it to split off. RE-MEASURED 2026-09-18 after the adoption: own=402 '
        'project=30 -> 7.46%.',
    ),
    'tests/architecture/_famconfig.py': Placement(
        STAYS,
        'PURE DECLARATION, AND IT IS ENTIRELY THIS REPO. The mechanism -- base, renderer, delta, '
        'ceiling, drop-with-a-reason -- already lives once in `lab_commons.dev.famconfig`, so what '
        'is left here is the lines that are facts about optimi-lab: the `usr/local/` tree, the '
        'viztracer `result.json`, the coredumpy `dumps/`, the eight stock hook ids this repo takes '
        'beyond the family eleven, and the reason `.pre-commit-config.yaml` is refused. There is no '
        'family half left to cut, which is exactly why the seam upstream was worth cutting. MEASURED '
        '2026-09-17: own=66 repo=2 -> 3.03%, which clears the 3.0% bar by a THIRD OF A POINT, so the '
        'honest reading is that the density admits this row rather than argues it -- the real '
        'evidence is that every added line resolves against this tree and no other, and a gitignore '
        'pattern names a directory, which is the shape a noun scan cannot see.',
    ),
    'tests/architecture/test_a_bounded_wait_names_its_remedy.py': Placement(
        SPLITS,
        'THE SEAM IS CONTROL VERSUS SUBJECT. FAMILY: the wall that terminates a process TREE, the '
        'width a refusal must consult, and the three-state sentence are all `lab_commons.dev.bounded` '
        'already, and the controls that drive them have no optimisation in them -- MEASURED '
        '2026-09-17, wdg-lab asks the SAME question with a completely disjoint implementation (303 '
        'lines across the two files, 303 of them differing), which is the worst case of a duplicated '
        'question: two answers and no shared line. LOCAL: that the waits in THIS tree are '
        '`scripts/pdoc.py` driving a documentation build and the verify entry point driving ruff and '
        'pytest. Measured: own=43 repo=0 -> 0.00%, admitted by the 50-line ceiling alone, so this '
        'row carries no evidence of density -- only no room to hide a mechanism.',
    ),
    'tests/architecture/test_a_declared_set_refuses_a_stranger.py': Placement(
        STAYS,
        'THE DOMAIN FACT IS THE SUBJECT: a library of interchangeable parts IS a set of registries -- '
        'which regressor, which sampler, which direction -- and the failure it refuses is the silent '
        'one where an unrecognised key falls back to a default and the caller gets a run it never '
        'asked for. That sentence is false of every repo that is not an optimiser. Measured '
        '2026-09-17: own=36 repo=4 -> 11.11%, and under the ceiling as well.',
    ),
    'tests/architecture/test_a_pin_is_a_named_set.py': Placement(
        SPLITS,
        'THE SEAM IS RULE VERSUS POPULATION. FAMILY: `NAMED-SETS-NOT-COUNTS` is a family rule stated '
        'in motronics-studio\'s `.claude/rules/rem/integration.md` -- "a count pin is blind to a swap, '
        'and invites being lowered" -- and the scan that refuses a `_COUNT` where a `frozenset` '
        "belongs has nothing repo-shaped in it. LOCAL: the pins it walks are this package's, and so "
        'is the correction this file records, that the rule was claimed as enforced on 2026-09-15 '
        'while citing three modules that merely HAPPEN to use named sets. Measured 2026-09-17: '
        'own=67 repo=0 -> 0.00%, over the ceiling and at zero density -- a `BELOW_THE_BAR` row.',
    ),
    'tests/architecture/test_a_relative_tolerance_carries_its_floor.py': Placement(
        SPLITS,
        'THE SEAM IS SCAN VERSUS SCOPE. FAMILY: walking tracked `.py` files for an `approx` call '
        'giving a RATIO with no floor under it is a pytest fact, true in every repo of the family '
        'and answered nowhere else in it. LOCAL: the two MEASURED floors (2 `approx` call sites, 40 '
        'Python files) and the SCOPE this file is careful to bound -- `Variable.unit` is a free-text '
        'label this package never computes with, pint is not a dependency it may take, and '
        '`UNITS-GO-THROUGH-PINT` stays declared absent rather than being answered by narrowing this '
        'scan. Measured 2026-09-17: own=71 repo=0 -> 0.00%; a `BELOW_THE_BAR` row whose repo facts '
        'are two integers, which is the shape the noun scan cannot see.',
    ),
    'tests/architecture/test_a_retired_spelling_stays_retired.py': Placement(
        STAYS,
        'THE REGISTRY HAS A SUBJECT AND THE SUBJECT IS THIS PACKAGE: two breaking refactors in '
        'September 2026 deleted public `optimi_lab` names outright, with no compat shim and no '
        'deprecation alias, and recorded which ones only in their commit messages. The scan that '
        'refuses a resurrection is small; the LIST of what was retired is the file, and it means '
        'nothing anywhere else. Measured 2026-09-17: own=80 repo=8 -> 10.00%.',
    ),
    'tests/architecture/test_every_blocking_wait_declares_a_ceiling.py': Placement(
        STAYS,
        'A BINDER over `lab_commons.dev.famtests.untimedwaits`: the walk, the AST reader, the two '
        'named call sets, the arm and the planted control are all upstream, and this file supplies '
        'only the three answers the kit refuses to guess -- which trees are walked, what a file in '
        'each is called, and the two numbers that bound the population. MEASURED 2026-09-18: own=16 '
        'repo=0 -> 0.00%, thirty-four lines under the 50-line binder ceiling, so it is admitted by '
        'the BINDER arm alone and owes no `BELOW_THE_BAR` row. THE ZERO IS HONEST RATHER THAN A '
        'MISCLASSIFICATION: every repo fact in the file is a PATH STRING in `ROOTS` or a NUMBER in '
        'the two floors, and an identifier scan can see neither. MOVES was refused by the roots -- '
        'moving them would put one checkout tree layout into a package three repos share, and a '
        'guessed pair does not raise, it walks a tree that is not there and reports CLEAN.',
    ),
    'tests/architecture/test_injected_doc_width_ceiling.py': Placement(
        SPLITS,
        'THE SEAM IS DECLARATION VERSUS CONTROL. FAMILY: the 120-column ceiling, the injected-doc '
        'corpus and the two-sided ratchet are already `lab_commons.dev.docwidth`, and the planted '
        'controls that drive it are generic too -- MEASURED 2026-09-17, 58% of the lines across this '
        "file and wdg-lab's twin are identical (291 lines total, 123 differing). LOCAL: this tree's "
        'declared `path:line` over-width sites. Measured: own=36 repo=0 -> 0.00%, admitted by the '
        '50-line ceiling alone.',
    ),
    'tests/architecture/test_memory_lives_under_a_date.py': Placement(
        SPLITS,
        'CORRECTED 2026-09-17, AND THE TWO ROSTERS HAD DISAGREED ABOUT THE SAME FILE. This row read '
        "MOVES while wdg-lab's twin read SPLITS, and the measurement supports wdg-lab: the two "
        'copies are 33.2% identical -- the LOWEST pair in the whole census, against 88.9% for the '
        'allow guard -- and the difference is STRUCTURAL, five declared memory trees there against '
        'one here. A shape that has to be re-derived per tree is not one file waiting to move. '
        'FAMILY: `.claude/memory/YYYY/MM/DD/` with frontmatter agreeing with the directory it sits '
        'in is a fleet convention answered independently in three repos. LOCAL: the tree this repo '
        'declares. RE-MEASURED against the density bar rather than relabelled after wdg-lab: '
        'own=87 repo=0 -> 0.00%, so it joins `BELOW_THE_BAR` with its numbers instead of arriving '
        'as a clean SPLITS -- the label had to answer the same bar as STAYS, and it does not yet.',
    ),
    'tests/architecture/test_no_allow_entry_names_a_denied_shape.py': Placement(
        SPLITS,
        'EXECUTED 2026-09-17, AND CLOSING THE HOLE CAUGHT THIS REPO. The body is '
        '`lab_commons.dev.famtests.allowguard` and the local copy is deleted; the census that '
        "decided it measured this file and wdg-lab's twin 88.9% identical over 324 lines with ZERO "
        'repo nouns in code. FAMILY: the glob instantiation and its cases (the `Bash(...)` spelling '
        'belongs to the agent client, not to any repo), the engine driving, and the `probed` floor '
        'that now refuses a scan of zero rows. LOCAL: the sanctioned exit a red is redirected to '
        "and the two planted rows, which are this repo's because `deny_rules.py` DROPS a rule "
        'whose remedy a repo lacks -- a row denied here is permitted elsewhere. THE FIRST HONEST '
        'RUN REDDED: `.claude/settings.json` declares no `permissions` block at all, so this guard '
        'has been green over ZERO probed rows for its whole life, and the property arm is a strict '
        'xfail carrying that measurement until a human adds the first allow row. RE-MEASURED: '
        'own=22 repo=0 -> 0.00%, down from own=77, admitted by the 50-line binder ceiling.',
    ),
    'tests/architecture/test_no_cjk_in_tracked_source.py': Placement(
        SPLITS,
        'THE SEAM IS THE SAME ONE THE DOC-WIDTH ROW HAS. FAMILY: the scan, the CJK ranges, the '
        'exemptions and the two-sided ratchet are already `lab_commons.dev.cjk`, and the control '
        "shape is shared -- MEASURED 2026-09-17, 60% of the lines across this file and wdg-lab's "
        "twin are identical (252 total, 100 differing). LOCAL: this tree's corpus and its DECLARED "
        'SET of files still carrying CJK. Measured: own=38 repo=0 -> 0.00%, admitted by the 50-line '
        'ceiling alone.',
    ),
    'tests/architecture/test_optimi_lab_adopts_the_shared_registry.py': Placement(
        STAYS,
        "IT IS THIS REPO'S HALF OF THE SHARED REGISTRY BY CONSTRUCTION: `lab_commons.dev.rules` "
        "authors each rule's universal statement and `assert_adopted` demands that the adopting tree "
        'name the MECHANISM in its own files. 25 of 32 enforced with the other 7 named as gaps is a '
        'statement about optimi-lab and is false of every other checkout. Measured 2026-09-17: '
        'own=98 repo=3 -> 3.06%, which clears the bar by the narrowest margin in this table -- '
        'stated rather than rounded, because the file is a table of rule IDs and mechanism paths, '
        'and neither spelling is a noun this scan can see.',
    ),
    'tests/architecture/test_the_agent_guard_is_live.py': Placement(
        SPLITS,
        'EXECUTED 2026-09-17, AND THIS REPO GAINED AN ARM RATHER THAN ONLY LOSING LINES. The seven '
        'arms are `lab_commons.dev.famtests.agentguard` and the local copy is deleted. THE FAMILY '
        "BODY TOOK WDG-LAB'S STRUCTURE, which is the integration rule applied to assertions: the "
        'deny arm now NAMES the rule it expects, where this file asked only whether SOMETHING '
        'refused -- a question upstream measured being fooled by registry order on two real rules '
        'that both match `git push --force`. LOCAL: the five-name shipped set, the six '
        'command-plus-rule rows and the four sanctioned exits. RE-MEASURED after the adoption: '
        'own=42 repo=0 -> 0.00%, down from own=79, admitted by the 50-line binder ceiling.',
    ),
    'tests/architecture/test_the_family_config_is_rendered.py': Placement(
        SPLITS,
        'EXECUTED 2026-09-17, AND THE CHEAPEST OF THE FIVE BECAUSE THE SEAM WAS ALREADY CUT HERE. '
        'This file already imported `DELTAS`/`EXTRA_HOOK_IDS`/`REPO` from a local `_famconfig`; '
        'what was missing was the other side, and it is now '
        '`lab_commons.dev.famtests.configrender`. LOCAL: the two floors, the re-render remedy, the '
        'planted edit, the five ids that keep pre-push, the pre-push bit, the pin-resolution arm -- '
        'and, new today, `PRECOMMIT_STAGE_MOVE`, the thirteen ids the narrowing takes, which this '
        'repo had never pinned and could not have seen a hook silently JOINING without. THREE ARMS '
        'WERE DELETED rather than ported: the kit already drives all three anchor refusals against '
        'this same base. RE-MEASURED: own=67 repo=0 -> 0.00%, down from own=118, a `BELOW_THE_BAR` '
        'row.',
    ),
    'tests/architecture/test_the_declared_hooks_are_installed.py': Placement(
        SPLITS,
        'EXECUTED 2026-09-17, AND HALF OF IT HAD ALREADY MOVED. `lab_commons.dev.hook_install` has '
        'owned the MEASUREMENT since 2026-09-16; what was still forked three ways was the six-arm '
        'TEST, now `lab_commons.dev.famtests.hookinstall`. LOCAL: the tree under test and the '
        'two-stage named set -- and nothing else, which is why this is a BINDER. ONE ARM WAS '
        'DELETED and its absence is the finding: `assert all(stage.hook_ids ...)` cannot fail, '
        'because `declared_stages` groups ids BY stage and the empty case is unreachable through '
        'the only path that builds a report. RE-MEASURED after the adoption: own=11 repo=0 -> '
        '0.00%, down from own=55, admitted by the 50-line binder ceiling.',
    ),
    'tests/architecture/test_the_dependency_door_is_wired.py': Placement(
        SPLITS,
        'THE HIGHEST TEXTUAL OVERLAP MEASURED ANYWHERE IN THIS ROSTER, and still not a move. '
        "MEASURED 2026-09-17: this file and wdg-lab's twin are 402 lines together with 34 differing "
        '-- 91.5% identical. FAMILY: that both adapters are supplied and that the GAP is rendered if '
        "either goes is a property of `lab_commons.dev.dep`'s `Port`, and the control that plants a "
        "holder rather than taking the real box seat is generic. LOCAL: it imports this repo's "
        '`scripts/dep.py` and asserts the port is named `optimi-lab`. Measured: own=89 repo=6 -> '
        '6.74%, which clears the bar -- the six hits are the local half, so the split is real and '
        'small rather than absent.',
    ),
    'tests/architecture/test_the_migration_boundary_is_declared.py': Placement(
        SPLITS,
        'THE SEAM IS CHECKS VERSUS DATA, and it is the cleanest one in this table because the split '
        'is already cut across two files. FAMILY: all five sections -- completeness with a floor, '
        'STAYS-needs-evidence, the MOVES converse, the density bar with SPLITS held to it, and the '
        'planted controls that prove the measurement can still red -- are the shape motronics-studio '
        "proved in `tests/architecture/layering/`, and wdg-lab's copy of this file is the third. THE "
        'FAMILY HALF IS NOW PAID: the completeness arm, both sides of the placement floor, the '
        'stale-debt refusal, the two bar-bounding arms and the meter control are `famtests.placement` '
        'and `famtests.density`, and four hand-written density controls collapsed into one kit call. '
        'LOCAL: the shell-row ceiling with its planted control, the prose-versus-code asymmetry pinned '
        'on two live rows, and the two planted shapes that use this repo`s nouns. RE-MEASURED '
        '2026-09-18: own=156 project=5 -> 3.21%, which clears the bar -- AND ALL FIVE HITS ARE PLANTED '
        'CONTROL FIXTURES, the strings a control writes into a temporary tree to prove the noun scan '
        'can still fire, not facts this file asserts about optimi-lab. The number is TRUE and the '
        'reading behind it is weak, which is stated here rather than left for the density to imply. '
        'THE FIRST DRAFT OF THIS ROW PREDICTED 0.00% AND WAS WRONG, and the strict xfail it was given '
        'XPASSED -- which is the ratchet working in the direction nobody plans for.',
    ),
    'tests/architecture/test_the_public_surface_is_declared.py': Placement(
        STAYS,
        'IT SCANS `src/optimi_lab` AND NOTHING ELSE, and the twin comparison refutes a move rather '
        "than supporting one: MEASURED 2026-09-17, this file and wdg-lab's same-named module are 462 "
        'lines together with 410 differing -- 89% DIFFERENT, so the shared name is not shared code. '
        "Its three properties are stated in this package's terms and its size band is this "
        "package's datum (320, measured against a largest module of 261). Measured: own=153 repo=1 "
        '-> 0.65%, a `BELOW_THE_BAR` row.',
    ),
    'tests/architecture/test_the_rules_pages_are_a_ratchet.py': Placement(
        SPLITS,
        'EXECUTED 2026-09-18, kit `0.2.2.dev74+ga177ba62f`. This row spent two days as the only '
        'ADOPTION-NOT-YET-EXECUTED in the roster; the whole of its local mechanism is now DELETED '
        'and `lab_commons.dev.famtests.rulespages` drives the verdict. FAMILY: the reading, the '
        'four-movement comparison, the vacuity floor and -- NEW HERE -- the TOTAL CEILING, which '
        'is the one arm this copy never had and the reason the swap is not a wash: every per-page '
        'pin is a LOCAL decision, so nothing capped their sum and a third and fourth page would '
        'have been two defensible pins and one document nobody sized. LOCAL: the corpus definition '
        '(this repo WALKS `.claude/rules/`, lab-commons takes its own from git, and the kit ships '
        'no default because neither is right for the other), the two pins, the ceiling and the '
        'floor. BOTH CONTROLS WENT UPSTREAM to `tests/test_famtests_rulespages.py` rather than '
        'being copied: a duplicate control that cannot diverge from its original is maintenance '
        'cost with no evidence value. THE SWAP WAS NOT A RENAME: exactly ONE of the six public '
        'names survived the move (`ratchet_breaks`; `rule_pages` became `page_lines`, and every '
        'repo fact became a keyword with no default), so every call site was re-read against the '
        'new signatures rather than mapped. RE-MEASURED: own=25 repo=0 -> 0.00%, well under the '
        '50-line binder ceiling and justified as a binder rather than by density; what is left is '
        'four numbers and two paths, the shape the noun scan cannot see. '
        'THE CENSUS MISS THIS ROW RECORDED IS CLOSED, AND THE RECORD IS KEPT BECAUSE THE MISS WAS '
        'WORTH MORE THAN THE ROW. On kit `0.2.2.dev71+gf6c46bb90` `supersede.take_census` graded '
        'this pair UNTOUCHED while it was already upstream, because BOTH detectors were blind at '
        'once -- `rulespages` named no consumer path, so PROVENANCE could not fire, and this file '
        'imported nothing from the kit, so IMPORT could not either. That is the '
        'live-fork-that-imports-nothing shape the instrument names as the worst case it was built '
        'for, invisible whenever provenance is ALSO absent. BOTH HALVES ARE NOW FIXED AND BY '
        'DIFFERENT ROUTES: upstream turned provenance into DATA (`_provenance_rows.py`, audited by '
        '`undeclared_modules`, so a published module naming nobody REDS -- forgetting is the '
        'failure rather than the absence of one) and declares this pair `supersedes` with this '
        "exact path; and this file now imports the kit, so IMPORT fires too. The registry's three "
        'kinds are why that declaration is readable at all: `supersedes` (the consumer file was '
        'replaced -- the only kind that opens a case), `adopted_by` (the consumer delegates and '
        'STAYS) and `original` (no consumer fork). Prose could not tell the first two apart, which '
        'is how a mention of `scripts/gate/runner.py` became a false positive. '
        'ONE UPSTREAM EXPECTATION IS NOW HISTORICAL AND IS REPORTED RATHER THAN EDITED: '
        "lab-commons's `tests/_supersede_rows.py` pins this file as a fixture case, hand-read "
        '`already_in_the_kit` and expected PARTIAL on the strength of its three `test_*` functions '
        'plus `PAGE_FLOOR` and `rule_pages` being remainder. Those are exactly what this commit '
        'deleted, so the fixture now describes a file that no longer has that remainder. That is '
        "lab-commons's row to re-take, not this roster's to edit.",
    ),
    'tests/architecture/test_the_roster_is_re_read_against_the_kit.py': Placement(
        SPLITS,
        'THE SEAM IS THE ANSWERS VERSUS THE ASSERTIONS, and the family half is ALREADY '
        'PUBLISHED. FAMILY, IMPORTED: `lab_commons.dev.supersede` (kit `0.2.2.dev74+ga177ba62f`) '
        'holds both detectors, the seven grades, the ruler and both floor refusals -- nothing '
        'here re-implements any of it. LOCAL: the four answers that module refuses to guess, and '
        'it refuses them for the reason `LAB_CZ_BASE_REF` is the worked example of -- the roster '
        'source, this checkout as the root, `lab_commons.dev` spelled at full depth, and the two '
        'floors. THE ROW THIS FILE EXISTS FOR IS THE ONE NOBODY HAD: the density bar asks whether '
        'a file is mostly generic and never asks whether the family already expressed it, so a '
        'row stayed pending after its subject landed upstream and no mechanism noticed. THE SPLIT '
        'IT STILL OWES: the assertion body -- census floors, the stale-MOVES arm, the named '
        'waiver and the planted control -- is what every consumer of `supersede` will write, and '
        'belongs beside it in `lab_commons.dev.famtests` as a body a repo parametrizes. THE '
        'EVIDENCE THAT IT SHOULD is the same shape the other famtests rows cite: wdg-lab gained '
        'this file on the same day, and the two differ only in the four answers. MEASURED '
        '2026-09-18: own=88 repo=0 -> 0.00%, so it is a `BELOW_THE_BAR` row and says so.',
    ),
    'tests/architecture/test_the_runtime_stays_pure.py': Placement(
        STAYS,
        "THE CLAIM IT CHECKS IS THIS PACKAGE'S OWN SENTENCE: `src/optimi_lab/__init__.py` states the "
        'runtime is numpy, scipy and scikit-learn -- "no plotting, no logging framework, no path '
        'resolution, no configuration file" -- and `pyproject.toml` repeats it in a comment. Until '
        'this file both were PROSE, and a `structlog` import added to one function would have left '
        'every word of the claim reading exactly as it does now. Which three distributions are '
        "permitted is this library's decision and nobody else's. Measured 2026-09-17: own=86 repo=2 "
        '-> 2.33%, a `BELOW_THE_BAR` row: the facts are three distribution NAMES, which are strings.',
    ),
    'tests/architecture/test_the_verdict_run_takes_the_box.py': Placement(
        SPLITS,
        'THE SEAM IS PROTOCOL VERSUS ENTRY POINT. FAMILY: that one CPU-saturating run holds the box '
        'at a time, and the control that proves the lock is taken and released, is '
        "`lab_commons.dev.boxlock`'s protocol -- MEASURED 2026-09-17, this file and wdg-lab's twin "
        "are 350 lines together with 54 differing, 84.6% identical. LOCAL: that this repo's ONE "
        'verdict command is `.venv/Scripts/python.exe -m lab_commons.dev.verify`, and the rules page '
        'that permits exactly that line. Measured: own=55 repo=2 -> 3.64%, which clears the bar.',
    ),
    'tests/architecture/test_this_checkout_is_visible_on_origin.py': Placement(
        SPLITS,
        'EXECUTED 2026-09-17, AND THE SEAM WAS NARROWER THAN THIS ROW USED TO READ. '
        '`lab_commons.dev.checkout` already answered "what is here that origin does not have" and '
        'this file already imported it; what was still forked were the assertions and, above all, '
        'the two planted git FIXTURES -- factored out here and inlined per function in the sibling '
        'lab, which is a control written twice and therefore a control that can be wrong in one '
        'copy. `lab_commons.dev.famtests.visibility` now ships `plant_checkout` and `commit` beside '
        'the assertions. LOCAL, each a keyword with NO DEFAULT upstream: the trunk NAME (the '
        '`LAB_CZ_BASE_REF` shape -- a repo judged against a guessed `main` reports itself clean), '
        'and the two two-sided branch pins, one of which still names '
        '`fix/p0-integration-blockers`. RE-MEASURED after the adoption: own=13 repo=0 -> 0.00%, '
        'down from own=74, admitted by the 50-line binder ceiling.',
    ),
}


# --------------------------------------------------------------------------- THE MEASUREMENT
#
# THE READERS AND THE METER ARE `lab_commons.dev.famtests.{placement,density}`. Everything below is a
# BINDER: it supplies the repo-shaped arguments the kit refuses to default, and holds no mechanism of
# its own. What used to be here -- the walk, the docstring/comment blanker, the delegation-import
# reader, the `Density` dataclass and `measure_density` -- was one of FOUR hand-rolled copies of one
# body in this family, and it is DELETED rather than wrapped.


def placed_files(root: Path | None = None) -> tuple[str, ...]:
    """Every runnable file this table must classify, as sorted repo-relative POSIX paths."""
    return placement.placed_files(root or repo_root(), trees=SCANNED, suffixes=RUNNABLE_SUFFIXES, not_placed=NOT_PLACED)


def nouns_in_prose(rel: str, root: Path | None = None) -> set[str]:
    """Every repo noun anywhere in *rel*, PROSE INCLUDED -- what property 2 reads.

    Property 2 asks only whether evidence EXISTS, and it is deliberately the generous reading: a
    file that explains in its docstring which optimisation fact it is about has said something true.
    That generosity is exactly why property 4 exists, and section 4's own controls pin that prose
    buys no DENSITY. Renamed from `nouns_in` in the same edit that adopted the kit, because the kit
    publishes a `nouns_in` that takes TEXT and this one takes a PATH -- one spelling for two
    signatures is how a caller passes the wrong thing and gets an answer anyway.
    """
    text = (root or repo_root()).joinpath(rel).read_text(encoding='utf-8', errors='replace')
    return density.nouns_in(text, noun=NOUN)


def nouns_in_code(rel: str, root: Path | None = None) -> set[str]:
    """Every repo noun in *rel*'s CODE, docstrings and comments blanked -- what property 3 reads.

    THE ASYMMETRY WITH `nouns_in_prose` IS DELIBERATE AND IT WAS MEASURED IN THE SIBLING. Property
    3's hazard is carrying a repo fact into a vendor-neutral package, and a fact only a SENTENCE
    carries is rewritten by the move itself. Reading prose there produces FALSE REFUSALS: on the
    first run of wdg-lab's copy of this guard, 2026-09-17, three correct `MOVES` rows were refused
    because their docstrings used an English word and named their own repo while explaining where a
    contradiction had been found. None of those is a domain fact.

    A non-Python file has no AST, so its whole text is its code: a shell script carries no docstring
    to be generous about.
    """
    path = (root or repo_root()).joinpath(rel)
    text = path.read_text(encoding='utf-8', errors='replace')
    if path.suffix != '.py':
        return density.nouns_in(text, noun=NOUN)
    return density.nouns_in('\n'.join(density.code_only(text)), noun=NOUN)


def measure(source: str) -> density.Density:
    """The density reading for *source*, with this repo's signal and delegation home supplied."""
    return density.measure_density(source, signals=(NOUN,), delegation_homes=DELEGATION_HOMES)


def is_justified(reading: density.Density) -> bool:
    """Whether a row clears THIS repo's two bars -- OURS or a BINDER, and nothing between."""
    return density.justified(reading, ceiling=OWN_MECHANISM_CEILING, minimum_pct=MIN_REPO_DENSITY_PCT)


#: CALIBRATED ON THIS TREE, 2026-09-17, and BOUNDED BY MEASUREMENT ON BOTH SIDES rather than chosen.
#: Every number below is `measure_density` run on the real file.
#:
#:     ADMIT   scripts/dep.py                              own= 22  repo=1   the reference binder
#:     ADMIT   test_no_cjk_in_tracked_source.py            own= 38  repo=0   declaration + control
#:     ADMIT   test_a_bounded_wait_names_its_remedy.py     own= 43  repo=0   the LARGEST admitted
#:     REFUSE  test_a_pin_is_a_named_set.py                own= 67  repo=0   carries its own scan
#:     REFUSE  test_a_relative_tolerance_carries_its_floor own= 71  repo=0   carries its own scan
#:
#: The ceiling must exceed 43 and fall below 67; 50 is the round number in that interval.
#:
#: IT IS NOT A FAMILY CONSTANT AND THE KIT DELIBERATELY DOES NOT SHIP ONE, which is the finding the
#: placement half exists to carry. Three of the family's four rosters read 50 and the fourth --
#: motronics' `scripts/` -- reads 40 over an interval of (35, 42) that EXCLUDES 50, while this
#: tree's (43, 67) excludes 40. A module shipping one number would have been wrong for one consumer
#: on that consumer's own evidence, and wrong in the ADMITTING direction, which is the silent one.
#: What ships instead is `assert_ceiling_is_bounded`, and the two readings below are its arguments --
#: until now this bracket lived in the comment above, where nothing could check it and where a bar
#: quietly widens to absorb the row that reds.
OWN_MECHANISM_CEILING: Final = 50

#: `test_a_bounded_wait_names_its_remedy.py`, own=43 -- the LARGEST reading the ceiling must ADMIT.
CEILING_ADMITS: Final = 43

#: `test_a_pin_is_a_named_set.py`, own=67 when the interval was bounded -- the SMALLEST reading the
#: ceiling must REFUSE. The calibration reading is not re-taken: a ceiling re-derived from today's
#: files is a number tuned to pass them.
CEILING_REFUSES: Final = 67

#: BOUNDED THE SAME WAY, measured 2026-09-17: above `test_the_public_surface_is_declared.py` (0.65%,
#: which scans this package and still reads near zero) and at or below
#: `test_optimi_lab_adopts_the_shared_registry.py` (3.06%, which is this repo's adoption table and
#: could not belong anywhere else). 3.0 is the round number in that interval, and the interval here
#: is TIGHT -- 3.06% clears it by six hundredths -- which is stated rather than hidden: a bar this
#: close to a live row is a bar whose next re-measurement may move a label. All four rosters in the
#: family measured 3.0 independently and the kit STILL does not ship it: four independent
#: measurements of a number are evidence FOR the number, not a licence to stop measuring it.
MIN_REPO_DENSITY_PCT: Final = 3.0

#: `test_optimi_lab_adopts_the_shared_registry.py`, 3.06% -- the SMALLEST percentage the bar must
#: ADMIT, and the tightest bracket in the family.
MINIMUM_ADMITS: Final = 3.06

#: `test_the_public_surface_is_declared.py`, 0.65% -- the LARGEST percentage the bar must REFUSE.
MINIMUM_REFUSES: Final = 0.65

#: THE SHORTFALL ON RECORD. Six `STAYS`/`SPLITS` rows fail the density guard today.
#:
#: TWO ROWS LEFT THIS SET ON 2026-09-18 AND THEY LEFT BY BEING DELETED rather than re-worded:
#: `test_memory_lives_under_a_date.py` (own 87 -> 21) and
#: `test_the_roster_is_re_read_against_the_kit.py` (own 88 -> 21) both adopted their family halves
#: and fell under the binder ceiling. A strict xfail that starts passing is a row to REMOVE, which is
#: the direction this ratchet exists to make cheap. Two others JOINED for the opposite reason: their
#: own counts rose when the kit adoption gave them headroom constants and planted controls to carry.
#:
#: Every row is xfailed STRICTLY with its measurement and its date, so each must be REMOVED in the
#: same commit that fixes its file, and no new row can join the list by accident. Loosening either
#: bar to absorb them was available and is refused: it would delete the finding rather than record it.
#:
#: CONFIDENCE IS NOT UNIFORM AND SAYING SO IS THE POINT. Four read 0.00%, which is the same reading a
#: genuine misclassification gives. `test_the_runtime_stays_pure.py` (2.38%),
#: `test_the_public_surface_is_declared.py` (0.62%) and `_famconfig.py` (1.23%) sit under the bar
#: with hits, and for those this says "no evidence of density", never "this row is wrong".
BELOW_THE_BAR: Final[dict[str, str]] = {
    'tests/architecture/test_a_pin_is_a_named_set.py': (
        'MEASURED 2026-09-17: own=67 repo=0 -> 0.00%, seventeen lines over the binder ceiling. THE '
        'SPLIT IT OWES: the scan that refuses a count where a named set belongs is the family rule '
        '`NAMED-SETS-NOT-COUNTS` with no optimisation in it, and belongs in `lab_commons.dev`. What '
        "stays is this package's pins. The zero is honest rather than a misclassification: the "
        'pins are frozensets of module NAMES, which are strings.'
    ),
    'tests/architecture/test_a_relative_tolerance_carries_its_floor.py': (
        'MEASURED 2026-09-17: own=71 repo=0 -> 0.00%. THE SPLIT IT OWES: walking tracked `.py` files '
        'for an `approx` ratio with no floor under it is a pytest fact and belongs upstream, where '
        'the sibling repos would get it too -- neither of them has this check at all. What stays is '
        'the two measured floors and the scope statement that keeps `UNITS-GO-THROUGH-PINT` '
        'honestly absent rather than answered by a narrowed scan.'
    ),
    'tests/architecture/test_the_family_config_is_rendered.py': (
        'MEASURED 2026-09-17 AFTER ADOPTING `famtests.configrender`: own=67 repo=0 -> 0.00%, down '
        'from own=118. A NEW ROW WHOSE DIRECTION IS THE FINDING: the verdict half left and the '
        'declaration half stayed, so this is the residue rather than a mechanism. THE READING IS '
        'HONEST: what is left is two FLOORS, a re-render remedy, a planted edit pair, a boolean '
        'about an installed hook and hook IDS -- numbers and strings to the last one, the same '
        'shape `_famconfig.py` beside it is recorded under. THE SPLIT IT STILL OWES: the '
        'pin-resolution arm, whose question (does every declared id still RESOLVE at the base pin) '
        'is a family question this repo happens to be the first to ask, and belongs beside '
        '`assert_declared_ids_survive` upstream.'
    ),
    'tests/architecture/_famconfig.py': (
        'RE-MEASURED 2026-09-17 AFTER DECLARING `PRECOMMIT_STAGE_MOVE`: own=81 repo=1 -> 1.23%, '
        'thirty-one lines over the binder ceiling, up from own=65 repo=1 -> 1.54% at the '
        '`.pre-commit-config.yaml` adoption. The thirteen new lines are the ids the stage narrowing '
        'takes, which the family body requires as an EQUALITY and this repo had never pinned. THE '
        'DIRECTION IS EXPECTED AND IS NOT A REGRESSION: this file is DATA, so it grows when a fact '
        'is declared rather than when a mechanism is written. THE READING IS HONEST AND THE ROW '
        'SAYS WHAT MOVED BEFORE THAT TOO: before the `.pre-commit-config.yaml` adoption this file '
        'read own=66 repo=2 -> 3.03%, a hair over the bar on a '
        'margin of ONE line -- and that line was the reason string of the refusal this adoption '
        'deleted, which named '
        'the repo only because a waiver is prose. Deleting the waiver with its subject deleted the '
        'density too, which is the clearest possible demonstration that 3.03% was never evidence of '
        'anything. THE REMEDY IS NOT A MOVE AND NOT A THINNING: this file is DATA computed by '
        'nothing, and its repo facts are `.gitignore` PATTERNS (`usr/local/`, `result.json`, '
        '`dumps/`), YAML HOOK IDS and a CEILING -- strings and numbers to the last one, which the '
        'module docstring above already names as the boundary this scan cannot cross. THE SEAM IT '
        'OWES IS ALREADY CUT: every mechanism it touches is `lab_commons.dev.famconfig`, and what '
        'is left is the residue. It sits under a bar written for a different shape.'
    ),
    'tests/architecture/test_the_public_surface_is_declared.py': (
        'MEASURED 2026-09-17: own=153 repo=1 -> 0.65%, far under the bar and far over the ceiling. '
        'THE READING IS HONEST AND THE ROW SAYS WHAT IT MISSES: this file scans `src/optimi_lab` '
        'and reports module names, so its repo facts are a PATH and the strings it prints. THE '
        'REMEDY IS NOT A MOVE -- the twin comparison refutes that outright (89% of the lines across '
        'this file and wdg-lab\'s same-named module differ). At 0.65% this is "no evidence of '
        'density", not "this row is wrong".'
    ),
    'tests/architecture/test_the_runtime_stays_pure.py': (
        'MEASURED 2026-09-17: own=86 repo=2 -> 2.33%, under the 3.0% bar and over the ceiling. The '
        'reading is honest: the fact this file enforces is WHICH THREE DISTRIBUTIONS are permitted, '
        "and `numpy`, `scipy` and `scikit-learn` are other people's names, so a scan for this "
        "repo's nouns cannot see the very thing the guard is about. THE REMEDY IS NOT TO THIN IT: "
        'the import walk is twenty lines and the permitted set is the deliverable. It sits under a '
        'bar written for a different shape, which is what this row records rather than argues away.'
    ),
}


def stays_rows() -> list:
    """Return every `STAYS` and `SPLITS` row as a param, strict-xfailed where `BELOW_THE_BAR` says so.

    `SPLITS` IS PARAMETRIZED HERE DELIBERATELY AND THAT IS THE CEILING ON THE LABEL. `SPLITS` is the
    escape hatch the density guard's own failure message offers, and an escape hatch needs a CEILING
    rather than just a reason: holding it to the SAME bar as `STAYS` means relabelling a red row
    cannot green it. A finished split leaves a binder or a file that is genuinely ours, which is
    precisely what `STAYS` must satisfy, so a separate gentler bar would let the label the author
    picked decide the standard -- and a label the author picks is not a measurement.
    """
    params = []
    rows = sorted(rel for rel, row in PLACEMENT.items() if row.side in (STAYS, SPLITS) and rel not in SHELL_ROWS)
    for rel in rows:
        reason = BELOW_THE_BAR.get(rel)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(rel, marks=marks))
    return params
