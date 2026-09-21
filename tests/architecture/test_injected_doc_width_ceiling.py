"""`INJECTED-DOC-WIDTH-CEILING` -- optimi-lab's corpus and declared sites, driven by the kit's arms.

The ceiling, the injected-doc corpus definition and the two-sided ratchet live in
`lab_commons.dev.docwidth`; the VERDICTS over them, the escape hatch's ceiling and the planted
control live in `lab_commons.dev.famtests.injectedwidth`. This module supplies only what a shared
body cannot hold: THIS tree's corpus, its DECLARED set, that set's ceiling, and the floor with its
headroom.

ADOPTED 2026-09-18, alongside the CJK guard beside it and for the same reason: the kit shipped this
body at `0.2.2.dev98+ga330b5561` and the roster census immediately reported this path as a
`NAMED_ONLY` row -- a file the kit names and that imports none of it. The kit's own docstring
records that the two guards were priced as possibly ONE body and refused on three measured
differences (their surfaces share no name, this one keys its declaration by `path:line` where the
other keys by FILE, and only this one has a ceiling on its hatch), so they are adopted as
neighbours rather than merged.

MEASURED HERE 2026-09-16 and re-measured 2026-09-18: of 79 tracked paths, exactly 2 are injected
documents by the shared definition (`AGENTS.md` / `CLAUDE.md` by basename anywhere in the tree --
this repo has neither -- plus everything under `.claude/rules/`, which here is `invariant.md` and
`workflow.md`). Scanning those two finds ZERO lines past the ceiling, so `DECLARED` is empty and
there is no width debt to record. NOTHING WAS REFLOWED to reach that: the pages were already inside.

THE SAME PAGES ARE ALSO PINNED BY LINE COUNT in `test_the_rules_pages_are_a_ratchet.py`, and the two
pins measure different things on purpose: a rewrite can lower a line count while doubling a line's
width, and a count-only pin reads that as an improvement. This is the missing dimension, not a
duplicate.

THE FLOOR IS THE POINT OF A ZERO MEASUREMENT, AND IT NOW HAS ITS SECOND SIDE. An empty declared set
against an empty found set is the vacuous green this family refuses: a corpus definition that
stopped matching anything reports exactly what these two compliant pages report. The floor used to
be pinned AT the measured 2 on the argument that the corpus only grows -- but `assert_width_floor`
is ONE-SIDED, so "only grows" was an intention rather than a property and growth was silently
absorbed. `FILES_READ_HEADROOM` makes it a property: at 2 + 1 the third rules page still passes and
the FOURTH forces the floor to be re-measured, which is what pinning AT a measurement was trying to
say all along.

`test_the_scan_reads_every_doc_handed_to_it` CLOSES THE OTHER HALF OF THE SAME GAP: the floor counts
what the scan READ, and a file handed in that it could not decode is a file not checked however
large the corpus is. A count alone cannot tell those apart.

`DECLARED_CEILING` IS 0, WHICH IS LEGAL AND IS THE STRONGEST VALUE -- the opposite of a floor. Every
declared site is prose an agent loads on every turn and pays for each time, so the number may only
go DOWN; at zero there is no hatch to grow one justified line at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.docwidth import WIDTH_CEILING, injected_docs, is_injected_doc
from lab_commons.dev.famtests import injectedwidth
from lab_commons.dev.rules import tracked_files

_ROOT: Final = Path(__file__).resolve().parents[2]

#: THE DECLARED SET, as `path:line` -- exactly how the ratchet keys its own found set. A NAMED SET
#: rather than a count, so a fixed site must be DELETED here in the same edit that fixes it and a
#: newly over-width line cannot arrive silently. Empty: measured 2026-09-16, re-measured 2026-09-18.
DECLARED: Final[frozenset[str]] = frozenset()

#: The largest `DECLARED` may be. ZERO -- there is no width debt here and no hatch to widen.
DECLARED_CEILING: Final = 0

#: Re-measured 2026-09-18: the injected-doc corpus here is 2 files, `.claude/rules/invariant.md` and
#: `.claude/rules/workflow.md`. A scan reaching fewer has stopped matching the corpus.
FILES_READ_FLOOR: Final = 2

#: How far past the floor the corpus may grow before the floor stops binding and must be re-taken.
#: 2 + 1 = 3 against today's 2, which is as tight as this kit allows (a headroom of 0 is refused as
#: misdeclared). A fourth rules page reds this constant instead of passing unremarked.
FILES_READ_HEADROOM: Final = 1

_WHAT: Final = 'INJECTED-DOC-WIDTH-CEILING'


def _corpus() -> tuple[str, ...]:
    """The injected docs of THIS repo, narrowed from what git tracks by the shared definition."""
    return tuple(injected_docs(sorted(tracked_files(_ROOT))))


def test_the_declared_sites_are_the_named_set_and_the_floor_binds_on_both_sides() -> None:
    """THE CHECK: the reading is proved to be a reading, then the ratchet, then the hatch's ceiling."""
    docs = _corpus()
    scan = injectedwidth.take_scan([_ROOT / name for name in docs], root=_ROOT, ceiling=WIDTH_CEILING)
    injectedwidth.assert_widths_are_the_named_set(
        scan,
        declared=DECLARED,
        declared_ceiling=DECLARED_CEILING,
        floor=FILES_READ_FLOOR,
        headroom=FILES_READ_HEADROOM,
        what=_WHAT,
    )


def test_the_scan_reads_every_doc_handed_to_it() -> None:
    """A file the scan could not open is a file it did not check -- never a compliant one.

    The floor above counts what was READ; this compares that against what was HANDED IN, which is
    the only arm that can see a file the scanner silently failed to decode.

    The body is `lab_commons.dev.famtests.injectedwidth.assert_every_document_handed_in_was_read`,
    which also refuses `undecodable` -- the field this file's own comparison never read. The corpus
    and the paths handed to it are this repo's two answers; the comparison is the family's.
    """
    paths = [_ROOT / name for name in _corpus()]
    scan = injectedwidth.take_scan(paths, root=_ROOT, ceiling=WIDTH_CEILING)
    injectedwidth.assert_every_document_handed_in_was_read(scan, handed=paths)


def test_a_planted_overwide_document_is_named_and_a_line_at_the_ceiling_is_not(tmp_path: Path) -> None:
    """CONTROL, BOTH DIRECTIONS, on a REAL tree: one column over convicts, exactly AT the ceiling does not."""
    injectedwidth.assert_the_scanner_still_convicts(tmp_path, ceiling=WIDTH_CEILING)


def test_the_declaration_is_the_shape_the_ratchet_keys_on() -> None:
    """A declaration is a set of `path:line` SITES, never a count -- the shape the ratchet acts on."""
    injectedwidth.assert_the_declaration_is_the_shape_the_ratchet_keys_on(DECLARED)


def test_the_corpus_is_this_repo_rules_pages() -> None:
    """WHAT IS BEING CHECKED, named -- a floor over the wrong corpus is a floor over nothing.

    This is the axis the planted control above is blind to: it plants a TREE and proves nothing about
    the narrowing. A basename set that lost `AGENTS.md`, or a prefix that stopped matching nested
    pages, passes every control and is caught only here and by the floor.
    """
    assert set(_corpus()) == {'.claude/rules/invariant.md', '.claude/rules/workflow.md'}, _corpus()
    assert not is_injected_doc('.claude/memory/2026/09/15/entry.md'), 'a dated memory entry is not injected'
    assert is_injected_doc('AGENTS.md'), 'the basename half of the corpus definition has stopped matching'
