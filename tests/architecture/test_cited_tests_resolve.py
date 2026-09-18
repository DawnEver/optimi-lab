"""PROSE THAT POINTS AT A GUARD MUST POINT AT A GUARD THAT EXISTS -- and four in this tree did not.

WHY THIS IS A GUARD AND NOT A PROOFREADING HABIT. A comment or docstring naming a repo-relative
`tests/.../test_*.py` path, or naming a bare test function, tells every reader that a specific check
governs the thing it heads. Being prose, nothing reads it back. THE FAILURE IS WORSE THAN A STALE
NAME: a reader who goes looking finds NO file and reasonably concludes the rule is unguarded, so the
prose routes people AROUND a check that is working one directory over -- measured here 2026-09-18,
`_placement.py` claimed a named guard kept its Python-only scope honest, that name has never existed
anywhere in this family, and the real guard sits 350 lines below it in the file that reads the table.

THE OTHER THREE WERE THE CROSS-REPO FORM, which is the same lie wearing a truthful sentence: two
cited an upstream lab-commons test module by its lab-commons-relative path, so the path was REAL
there and resolved to nothing HERE. Spelling the repo in front of the path fixes it and is not
cosmetic: it is the difference between a reader finding the control and concluding there is none.
It also puts the token out of this scan's reach by construction, since a citation preceded by a
slash is not a repo-relative one.

THE WHOLE READING IS THE FAMILY'S. `lab_commons.dev.famtests.citedtests` holds the walk, the prose
reader, the two citation patterns, the resolution rule, the three arms, the exemption arm and the
planted control; `lab_commons.dev.floors` holds both sides of each floor. What is left here is the
half the kit refuses to guess -- every repo-shaped fact arrives as a keyword argument with NO
DEFAULT -- and this file is where optimi-lab's eight answers are written down.

THE PATH FORM AND THE NAME FORM ARE PINNED BY SEPARATE FLOORS because they are separate populations
of very different size, and a single floor over the union would be set by the larger and would never
notice the smaller going silent.

THREE FLOORS, OVER THREE DIFFERENT POPULATIONS, AND THE THIRD IS THE ONE NOBODY WRITES. The file
count says the walk reached the tree. The test-function count says the set every name resolves
AGAINST is really there -- a `tests/` tree that moved would empty it while the file walk stayed
healthy. Neither can notice that the walk reached every file and read only its `#` lines: the
originating incident left 14 dangling citations sitting in DOCSTRINGS, green for weeks, under a
file-count floor of 1400. `DOCSTRING_CITATION_FLOOR` is that third population, and a reader narrowed
back to comments leaves the first two green and reds this one, which is the whole point. IT MATTERS
MORE HERE THAN ANYWHERE: this tree's prose carries its citations almost entirely in docstrings
rather than in `#` comments, so losing the docstring half would lose essentially the whole check
while the other two floors held.

WHAT THIS DOES NOT PROVE, stated so the row cannot overclaim. Nothing here reads Markdown, so a
dangling citation in `docs-src/` or a rules page is invisible. Resolution is deliberately WEAK --
exact name, module stem, or a wrapped prefix or suffix of a real name -- because test names in this
family are long and comments wrap them; the cost is that a wrong name which happens to prefix a real
one passes, and the benefit is that this guard CANNOT raise a false alarm, which is what decides
whether a mechanism survives contact rather than being deleted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import citedtests

#: This checkout: `tests/architecture/<this file>` sits two directories below it.
ROOT: Final = Path(__file__).resolve().parents[2]

#: THE TREES WHERE A CITATION IS A POINTER RATHER THAN A RECORD. `tests/` is in WHOLE rather than
#: reached into by waiver header, which is why `WAIVER_DIRS` below is empty: this repo has no
#: file-level header naming a governing ratchet, and ALL FOUR offenders found here lived under
#: `tests/architecture/` -- a scan of `src/` and `scripts/` alone would have reported this tree
#: clean. `examples/` is absent for a measured reason rather than an assumed one: it holds no `.py`
#: file at all in this checkout, so declaring it would add a pointer tree that contributes nothing.
POINTER_DIRS: Final = ('src', 'scripts', 'tests')

#: ROOT CONFIGURATION FILES READ AS PROSE. `#` starts a comment in all three formats, so they need no
#: separate reader. This argument exists because the originating incident hid in one: a lint config
#: at the repo root cited a guard deleted weeks earlier and survived, for the single reason that the
#: walk was two `.py` trees and a file at the root is in neither.
ROOT_CONFIGS: Final = ('pyproject.toml', '.pre-commit-config.yaml', 'Makefile')

#: THE FILE-LEVEL WAIVER HEADER AND THE TREES SEARCHED FOR IT. The header is the family's spelling;
#: the tree list is EMPTY, and by construction rather than by omission -- that mechanism exists to
#: reach INTO a test tree that is otherwise unwalked, and `POINTER_DIRS` already walks `tests/`
#: whole. Adding a directory here would read the same files twice and change no answer.
WAIVER_HEADER: Final = 'ratcheted by'
WAIVER_DIRS: Final[tuple[str, ...]] = ()

#: WHERE THE TESTS LIVE -- the set every citation resolves against, and the argument a guess destroys
#: loudly rather than silently: a `test_dir` resolving to nothing empties the defined set, at which
#: point every citation in the tree is dangling and the name arm fails on all of them at once.
TEST_DIR: Final = 'tests'

#: FILES WHOSE PROSE IS A DATED LOG rather than directions to a reader, and rewriting a path inside
#: one falsifies the record. EMPTY, which is the STRICTEST form this exemption can take and is
#: honest: measured 2026-09-18, this repo's four offenders were all live pointers, none of them a
#: historical record, so nothing is owed a keeper. The exemption arm below asserts it anyway -- a
#: keeper naming a deleted file is a waiver nothing uses, and it does not announce itself.
HISTORY_KEEPERS: Final[tuple[str, ...]] = ()

#: PHRASES THAT MAKE A SENTENCE A RECORD RATHER THAN A POINTER. EMPTY for the same measured reason,
#: and deliberately so: the two files here that describe arms they USED TO HOLD were repaired by
#: naming the upstream module that holds them now, which is the remedy, rather than by excusing the
#: retired spelling with a marker, which would have kept a name resolving to nothing in this
#: checkout while reading as a considered decision.
HISTORY_MARKERS: Final[tuple[str, ...]] = ()

#: TOKENS SHAPED LIKE A TEST NAME THAT ARE NOT CITATIONS -- vocabulary, and one repo's vocabulary
#: silently removing another's real citations is the guessed-exclusion failure. EMPTY: measured
#: 2026-09-18 this tree's prose produced no such token, so declaring one would be a widening bought
#: on speculation.
NOT_CITATIONS: Final[tuple[str, ...]] = ()

#: MEASURED 2026-09-18: the walk reads 61 files. Floor below it, far above the zero a mistyped
#: pointer directory returns; headroom is how far the population may grow before the floor is
#: RE-MEASURED, never how much slack the floor may be given. Compare the sibling lab's 463 over the
#: same reading -- an 8x difference, which is why this number cannot have a default.
SCANNED_FILE_FLOOR: Final = 50
SCANNED_FILE_HEADROOM: Final = 20

#: MEASURED 2026-09-18: 182 `def test_*` under `tests/`. THIS IS A DIFFERENT POPULATION FROM THE ONE
#: ABOVE and neither floor stands in for the other -- the name arm resolves against THIS set, so a
#: test tree that moved would empty it while the file walk stayed perfectly healthy.
DEFINED_TEST_FLOOR: Final = 150
DEFINED_TEST_HEADROOM: Final = 45

#: MEASURED 2026-09-18: 21 citations sitting on DOCSTRING lines. THE THIRD POPULATION, and the one a
#: file-count floor is blind to: a prose reader narrowed back to `#` lines leaves both floors above
#: green, empties the offender set for the wrong reason, and reds only here.
DOCSTRING_CITATION_FLOOR: Final = 20
DOCSTRING_CITATION_HEADROOM: Final = 15


def scan() -> citedtests.CitedScan:
    """One walk, handed to every arm, with this repo's eight answers supplied."""
    return citedtests.take_scan(
        ROOT,
        pointer_dirs=POINTER_DIRS,
        root_configs=ROOT_CONFIGS,
        waiver_header=WAIVER_HEADER,
        waiver_dirs=WAIVER_DIRS,
        test_dir=TEST_DIR,
        history_keepers=HISTORY_KEEPERS,
        history_markers=HISTORY_MARKERS,
        not_citations=NOT_CITATIONS,
    )


def test_no_prose_cites_a_test_FILE_that_is_not_on_disk() -> None:
    """THE PATH ARM, with the walk's file floor bound on both sides FIRST."""
    citedtests.assert_no_dangling_path_citation(scan(), floor=SCANNED_FILE_FLOOR, headroom=SCANNED_FILE_HEADROOM)


def test_no_prose_cites_a_test_NAME_that_resolves_to_nothing() -> None:
    """THE NAME ARM, the commoner form, floored over the DEFINED TEST FUNCTIONS it resolves against."""
    citedtests.assert_no_dangling_function_citation(scan(), floor=DEFINED_TEST_FLOOR, headroom=DEFINED_TEST_HEADROOM)


def test_the_prose_half_of_the_walk_is_really_being_read() -> None:
    """THE THIRD FLOOR. A file-count floor cannot notice that every DOCSTRING went unread."""
    citedtests.assert_the_prose_half_is_reached(
        scan(), floor=DOCSTRING_CITATION_FLOOR, headroom=DOCSTRING_CITATION_HEADROOM
    )


def test_every_history_exemption_names_a_file_that_is_here() -> None:
    """A keeper naming a deleted file has stopped covering anything while still reading as a decision."""
    citedtests.assert_every_exemption_is_real(ROOT, history_keepers=HISTORY_KEEPERS)


def test_the_readers_still_convict_a_planted_dangling_citation(tmp_path: Path) -> None:
    """THE CONTROL, both ways, through the SHIPPED readers and this repo's own vocabulary.

    The kit plants every distinction at once, each offender beside an honest neighbour that must NOT
    be named: a dangling path against a resolving one, a dangling name against a wrapped fragment of
    a real one, a citation in a DOCSTRING against the same path in ordinary CODE -- a path in code is
    DATA -- and a dangling citation in a ROOT CONFIG, which is the form the original incident hid in.
    """
    citedtests.assert_the_readers_still_convict(tmp_path, history_markers=HISTORY_MARKERS, not_citations=NOT_CITATIONS)
