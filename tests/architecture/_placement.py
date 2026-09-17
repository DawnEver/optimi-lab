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
  LITERAL is invisible here. Four rows below are honest `BELOW_THE_BAR` entries for exactly that
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
* `testfacts` -- NO HAND-ROLLED COPY HERE, which is the difference from the sibling. wdg-lab reads
  its test files' declarations with a regex; this tree reads its own with `ast` in
  `test_a_relative_tolerance_carries_its_floor.py` and `test_the_public_surface_is_declared.py`,
  for a DIFFERENT question (an `approx` ratio with no floor under it, an `__all__` that resolves),
  neither of which `testfacts` answers. Correctly absent.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

__all__ = [
    'BELOW_THE_BAR',
    'FAMILY_PACKAGE',
    'MIN_REPO_DENSITY_PCT',
    'MOVES',
    'OWN_MECHANISM_CEILING',
    'PLACEMENT',
    'PLACEMENT_FLOOR',
    'REPO_NOUNS',
    'RUNNABLE_SUFFIXES',
    'SCANNED',
    'SPLITS',
    'STAYS',
    'Density',
    'Placement',
    'measure_density',
    'nouns_in',
    'nouns_in_code',
    'placed_files',
    'repo_root',
    'stays_rows',
]

STAYS: Final = 'stays'
MOVES: Final = 'moves'
SPLITS: Final = 'splits'


@dataclass(frozen=True)
class Placement:
    """One file's side, and the FACT that decides it. The reason is the deliverable."""

    side: str
    why: str


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

#: Never authored, or carrying no knowledge to place: build output, agent memory, and the empty
#: package markers whose placement is whatever their directory does.
_NOT_PLACED: Final = ('__pycache__', '.claude')

#: The family package every row is measured against. A line that merely CALLS something imported
#: from it is delegation, not this file's own mechanism -- which is what makes a binder legible.
FAMILY_PACKAGE: Final = 'lab_commons'

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

_NOUN: Final = re.compile(r'(?<![A-Za-z0-9_])(' + '|'.join(REPO_NOUNS) + r')(?![A-Za-z0-9_])', re.IGNORECASE)

#: MEASURED 2026-09-17: `scripts/` holds 3 runnable files and `tests/architecture/` holds 20 modules
#: once this roster's own two files are counted, for 23 rows. The floor sits below that with room
#: for ordinary deletion and far above the zero a broken walk returns. Finding NOTHING is vacuous
#: rather than green.
PLACEMENT_FLOOR: Final = 18


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
        'module docstring says so; it lives here rather than upstream because motronics-studio '
        'holds its own copy and a third repo is not yet evidence that one home would fit all three. '
        'If `lab_commons.dev` grows a `placement` module this row becomes a SPLITS, and the seam is '
        'already named: the machinery is family, the rows are ours. Measured 2026-09-17: own=417 '
        'repo=32 -> 7.67%.',
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
        MOVES,
        'THREE COPIES OF ONE FLEET CONVENTION. `.claude/memory/YYYY/MM/DD/` with frontmatter that '
        "agrees with the directory it sits in is a family shape, and this file, wdg-lab's twin "
        "(MEASURED 2026-09-17: 380 lines together, 246 differing) and motronics-studio's "
        '`tests/architecture/docs/test_memory_files_live_under_a_dated_directory.py` are three '
        'independent answers to it. It names ZERO repo nouns in code, so the converse holds -- and '
        'that is the measurable difference from the wdg-lab twin, which holds two in-package memory '
        'trees as data and is therefore a SPLITS there. The trees this one declares are top-level '
        'and are configuration a shared mechanism would take as an argument.',
    ),
    'tests/architecture/test_no_allow_entry_names_a_denied_shape.py': Placement(
        MOVES,
        'THE SECOND-STRONGEST DUPLICATION EVIDENCE IN THIS TABLE: MEASURED 2026-09-17, this file and '
        "wdg-lab's twin are 324 lines together with 36 differing -- 88.9% identical. The subject is "
        'the contradiction between `.claude/settings.json` and `.claude/hooks/deny-rules.json`, two '
        'hand-written files about which commands an agent may issue, and neither is about '
        'optimisation. It names ZERO repo nouns in code. The natural home is beside '
        '`lab_commons.dev.hook_adoption`, which already owns the render half of the same pair.',
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
        MOVES,
        'THE MECHANISM IS ALREADY PUBLISHED AND THE FILE ONLY DRIVES IT: engine, rules and wiring '
        'reported BY NAME is `lab_commons.dev.agent_guard`, and this file asserts its three parts '
        'are installed in this checkout. MEASURED 2026-09-17: 65% of the lines across this file and '
        "wdg-lab's twin are identical (394 total, 136 differing), and it names ZERO repo nouns in "
        'code. What is repo-specific about the answer -- which rules ship here -- lives in '
        '`scripts/deny_rules.py`, which has its own row and STAYS.',
    ),
    'tests/architecture/test_the_family_config_is_rendered.py': Placement(
        MOVES,
        'THE CONVERSE HOLDS AND THE FILE ASSERTS NOTHING OF ITS OWN. MEASURED 2026-09-17: own=75 '
        'repo=0 -> 0.00%, and ZERO repo nouns in CODE -- every artefact name, every ceiling and '
        'every drop reason it checks is read at run time off `_famconfig`, the sibling that STAYS. '
        'Eight properties, all generic: the declared-or-refused completeness pin, the re-render '
        'comparison, the anti-fork arm, the last-match-wins negation property, the waiver that must '
        'still refuse, the waiver subject read off the live YAML, the planted control and the '
        'mode pin. MEASURED the same day, wdg-lab wrote six of these eight against a different '
        'delta, which is the definition of a move rather than a copy.',
    ),
    'tests/architecture/test_the_declared_hooks_are_installed.py': Placement(
        MOVES,
        'A DECLARATION-VERSUS-INSTALLATION CHECK WITH NO SUBJECT OF ITS OWN: a '
        '`.pre-commit-config.yaml` declares hooks, `pre-commit install` is a separate act on a '
        'separate machine, and `lab_commons.dev.hook_install` already owns the reading of both '
        "sides. MEASURED 2026-09-17: 66% of the lines across this file and wdg-lab's twin are "
        'identical (295 total, 99 differing), and it names ZERO repo nouns in code. Its numbers (19 '
        'hooks at pre-commit, zero installed on 2026-09-16) are read off the configuration at run '
        'time rather than restated, so the move carries no data with it.',
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
        "proved in `tests/architecture/layering/`, and wdg-lab's copy of this file is the third. "
        'LOCAL: nothing but the import of `_placement`. MEASURED 2026-09-17, in the commit that '
        'created it: own=154 repo=6 -> 3.90%, which clears the bar -- AND ALL SIX HITS ARE PLANTED '
        'CONTROL FIXTURES, the strings a control writes into a temporary tree to prove the noun '
        'scan can still fire, not facts this file asserts about optimi-lab. The number is TRUE and '
        'the reading behind it is weak, which is stated here rather than left for the density to '
        'imply: the split is owed on the machinery, not refuted by this margin. THE FIRST DRAFT OF '
        'THIS ROW PREDICTED 0.00% AND WAS WRONG, and the strict xfail it was given XPASSED -- which '
        'is the ratchet working in the direction nobody plans for.',
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
        MOVES,
        'A SECOND COPY OF A SHAPE MOTRONICS-STUDIO ALREADY HOLDS: its '
        '`tests/architecture/ratchets/test_rules_line_ratchet.py` pins the same thing, that '
        '`.claude/rules/*.md` is re-read on every turn so every line is a cost paid on every '
        'request. The mechanism -- count the lines of each always-loaded page, refuse all four '
        'movements, and pin PER FILE rather than as a total -- has nothing about optimisation in it '
        'and names ZERO repo nouns in code, so the converse holds. Its `_MEASURED` budgets are DATA '
        'a shared mechanism would take as an argument, which is what makes this a move rather than a '
        'split.',
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
        MOVES,
        'THREE COPIES AND A PUBLISHED MECHANISM. `lab_commons.dev.checkout` already answers "what is '
        'here that origin does not have" -- measured against the REMOTE and with `git cherry` rather '
        'than ancestry, because those are the two ways this audit has been observed to lie. This '
        "file, wdg-lab's twin (MEASURED 2026-09-17: 346 lines together, 178 differing) and "
        "motronics-studio's `tests/architecture/repo/test_a_working_lane_is_visible_on_origin.py` "
        'are three answers to it. It names ZERO repo nouns in code: the push obligation is a fact '
        'about origin, and nothing in it is a fact about optimisation.',
    ),
}


# --------------------------------------------------------------------------- THE MEASUREMENT


def placed_files(root: Path | None = None) -> Iterator[str]:
    """Every runnable file this table must classify, as repo-relative POSIX paths."""
    base = root or repo_root()
    for tree in SCANNED:
        for path in sorted((base / tree).rglob('*')):
            if not path.is_file() or path.suffix not in RUNNABLE_SUFFIXES:
                continue
            if any(part in _NOT_PLACED for part in path.parts) or path.name == '__init__.py':
                continue
            yield path.relative_to(base).as_posix()


def nouns_in(rel: str, root: Path | None = None) -> set[str]:
    """Every repo noun anywhere in *rel*, PROSE INCLUDED -- what property 2 reads.

    Property 2 asks only whether evidence EXISTS, and it is deliberately the generous reading: a
    file that explains in its docstring which optimisation fact it is about has said something true.
    That generosity is exactly why property 4 exists, and section 4's own controls pin that prose
    buys no DENSITY.
    """
    text = (root or repo_root()).joinpath(rel).read_text(encoding='utf-8', errors='replace')
    return {match.group(1).lower() for match in _NOUN.finditer(text)}


def nouns_in_code(rel: str, root: Path | None = None) -> set[str]:
    """Every repo noun in *rel*'s CODE, docstrings and comments blanked -- what property 3 reads.

    THE ASYMMETRY WITH `nouns_in` IS DELIBERATE AND IT WAS MEASURED IN THE SIBLING. Property 3's
    hazard is carrying a repo fact into a vendor-neutral package, and a fact only a SENTENCE carries
    is rewritten by the move itself. Reading prose there produces FALSE REFUSALS: on the first run
    of wdg-lab's copy of this guard, 2026-09-17, three correct `MOVES` rows were refused because
    their docstrings used the English word "slot" and named their own repo while explaining where a
    contradiction had been found. None of those is a domain fact.

    A non-Python file has no AST, so its whole text is its code: a shell script carries no docstring
    to be generous about.
    """
    path = (root or repo_root()).joinpath(rel)
    text = path.read_text(encoding='utf-8', errors='replace')
    if path.suffix != '.py':
        return {match.group(1).lower() for match in _NOUN.finditer(text)}
    code, _ = _code_and_delegation(text)
    return {match.group(1).lower() for line in code for match in _NOUN.finditer(line)}


def _code_and_delegation(source: str) -> tuple[list[str], set[str]]:
    """Lines with comments and docstrings blanked, plus the names bound from the family package."""
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) or not body:
            continue
        head = body[0]
        if isinstance(head, ast.Expr) and isinstance(head.value, ast.Constant) and isinstance(head.value.value, str):
            docstrings.add((head.lineno, head.col_offset))

    lines = source.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        prose = token.type == tokenize.COMMENT or (token.type == tokenize.STRING and token.start in docstrings)
        if not prose:
            continue
        (first, start_col), (last, end_col) = token.start, token.end
        for row in range(first, last + 1):
            text = lines[row - 1]
            begin = start_col if row == first else 0
            end = end_col if row == last else len(text)
            lines[row - 1] = text[:begin] + ' ' * (end - begin) + text[end:]

    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if FAMILY_PACKAGE in (node.module or '').split('.'):
                bound.update(alias.asname or alias.name for alias in node.names)
                bound.add(FAMILY_PACKAGE)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if FAMILY_PACKAGE in alias.name.split('.'):
                    bound.add(alias.asname or alias.name.split('.')[-1])
                    bound.add(FAMILY_PACKAGE)
    return [line for line in lines if line.strip()], bound


@dataclass(frozen=True)
class Density:
    """What the density guard measured about one file. A REASON, not a bool, so a failure prints it."""

    #: Code lines that are neither prose nor a delegation to the family package.
    own: int
    #: Of those, the ones naming a repo noun.
    repo: int

    @property
    def percent(self) -> float:
        return 100.0 * self.repo / self.own if self.own else 0.0

    @property
    def justified(self) -> bool:
        """Justified by being OURS or by being a BINDER, and by nothing in between."""
        return self.own <= OWN_MECHANISM_CEILING or self.percent >= MIN_REPO_DENSITY_PCT


def measure_density(source: str) -> Density:
    """Run the whole measurement on TEXT, so a control can plant a shape instead of a file."""
    code, delegated = _code_and_delegation(source)
    if delegated:
        delegation = re.compile(
            r'(?<![A-Za-z0-9_])(' + '|'.join(re.escape(name) for name in sorted(delegated)) + r')(?![A-Za-z0-9_])'
        )
        own = [line for line in code if not delegation.search(line)]
    else:
        # No family import, so nothing to subtract: an empty alternation matches at position zero
        # on EVERY line, which would read every file in the tree as a binder.
        own = list(code)
    return Density(own=len(own), repo=sum(1 for line in own if _NOUN.search(line)))


#: CALIBRATED ON THIS TREE, 2026-09-17, and BOUNDED BY MEASUREMENT ON BOTH SIDES rather than chosen.
#: Every number below is `measure_density` run on the real file.
#:
#:     ADMIT   scripts/dep.py                              own= 22  repo=1   the reference binder
#:     ADMIT   test_no_cjk_in_tracked_source.py            own= 38  repo=0   declaration + control
#:     ADMIT   test_a_bounded_wait_names_its_remedy.py     own= 43  repo=0   the LARGEST admitted
#:     REFUSE  test_a_pin_is_a_named_set.py                own= 67  repo=0   carries its own scan
#:     REFUSE  test_a_relative_tolerance_carries_its_floor own= 71  repo=0   carries its own scan
#:
#: The ceiling must exceed 43 and fall below 67; 50 is the round number in that interval. It is the
#: same value motronics-studio measured independently in its own tree and the same one wdg-lab's
#: roster measured in its own, which is evidence that the bar is a family property rather than a
#: number tuned to make today's rows pass -- four rows do not clear the density bar, and they are
#: recorded in
#: `BELOW_THE_BAR` rather than bought off by moving this to 90.
OWN_MECHANISM_CEILING: Final = 50

#: BOUNDED THE SAME WAY, measured 2026-09-17: above `test_the_public_surface_is_declared.py` (0.65%,
#: which scans this package and still reads near zero) and at or below
#: `test_optimi_lab_adopts_the_shared_registry.py` (3.06%, which is this repo's adoption table and
#: could not belong anywhere else). 3.0 is the round number in that interval, and the interval here
#: is TIGHT -- 3.06% clears it by six hundredths -- which is stated rather than hidden: a bar this
#: close to a live row is a bar whose next re-measurement may move a label.
MIN_REPO_DENSITY_PCT: Final = 3.0

#: THE SHORTFALL ON RECORD. Four `STAYS`/`SPLITS` rows fail the density guard today. They are
#: xfailed STRICTLY with their measurement and its date, so each must be REMOVED in the same commit
#: that fixes its file, and no new row can join the list by accident. Loosening either bar to absorb
#: them was available and is refused: it would delete the finding rather than record it.
#:
#: CONFIDENCE IS NOT UNIFORM AND SAYING SO IS THE POINT. Two read 0.00%, which is the same reading
#: a genuine misclassification gives. `test_the_runtime_stays_pure.py` (2.33%) and
#: `test_the_public_surface_is_declared.py` (0.65%) sit under the bar with hits, and for those this
#: says "no evidence of density", never "this row is wrong".
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
    for rel in sorted(rel for rel, placement in PLACEMENT.items() if placement.side in (STAYS, SPLITS)):
        reason = BELOW_THE_BAR.get(rel)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(rel, marks=marks))
    return params
