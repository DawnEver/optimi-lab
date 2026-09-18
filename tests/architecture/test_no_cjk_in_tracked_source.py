"""`NO-CJK-IN-TRACKED-SOURCE` -- this repo's corpus and declared set, driven by the kit's arms.

The scan, the five CJK ranges, the exemption list and the two-sided ratchet live in
`lab_commons.dev.cjk`; the VERDICTS over them, and the planted control, live in
`lab_commons.dev.famtests.trackedcjk`. This module supplies only what a shared body cannot hold:
THIS tree's corpus, its DECLARED set, and its floor with that floor's headroom.

ADOPTED 2026-09-18. Until then this file held its own copies of the kit's arms, and the moment the
kit shipped `trackedcjk` at `0.2.2.dev98+ga330b5561` the roster census reported this path as a
`NAMED_ONLY` row -- a file the kit names and that imports none of it. That red is what this change
answers; it was the kit upgrade that raised it, not any edit of ours.

`DECLARED` IS EMPTY AND IS A MEASUREMENT RATHER THAN AN ASPIRATION. Re-measured 2026-09-18:
`git ls-files` names 79 tracked paths, of which 77 are inside the scan (2 sit under an exempt
prefix, none fail to decode), and ZERO CJK characters remain. The set exists to refuse the FIRST CJK
character that arrives, not to record a debt.

WHICH IS EXACTLY WHY THE FLOOR IS NOT OPTIONAL, AND WHY IT IS NOW TWO-SIDED. An empty declared set
matched against an empty found set is the vacuous green this family refuses: a scan that reached no
files at all reports precisely what this clean tree reports. `FILES_READ_FLOOR` is what tells them
apart. It was 53, set AT a measurement of 53 on the argument that this corpus is small enough that
any shrinkage is a defect -- and the corpus then grew to 77 without a word, because
`cjk.assert_floor` is ONE-SIDED: it refuses an under-read tree and is silent about a floor the tree
has outgrown. At 53 against 77 it had 24 files of slack and would have passed a scan that lost two
thirds of the corpus. The remedy a slack floor names is to RE-MEASURE THE FLOOR, never to widen the
headroom, so it is re-taken at 70 and `FILES_READ_HEADROOM` is the side that reds the next time the
corpus outgrows it instead of letting it rot silently a second time.

THE EXEMPTION MATCH IS ON A PATH SEGMENT, not a root prefix, and this tree's count would be a
different number had it been otherwise -- `test_the_exemptions_are_matched_as_a_path_segment` pins
that property against the real shared predicate rather than trusting this docstring for it.

THE PLANTED CJK CHARACTER IS BUILT AT RUNTIME FROM THE SHARED RANGES, never written as a literal or
as an escape this file's text would carry -- and that is now the kit's property rather than this
file's promise. This module is itself tracked source and is scanned by the very guard it installs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.cjk import EXEMPT_PREFIXES, Scan, scan_files
from lab_commons.dev.famtests import trackedcjk
from lab_commons.dev.rules import tracked_files

_ROOT: Final = Path(__file__).resolve().parents[2]

#: THE DECLARED SET -- repo-relative POSIX paths, exactly as the scan names them. A NAMED SET rather
#: than a count, so a cleaned file must be DELETED here in the same edit and a newly dirty file
#: cannot arrive silently. Empty: measured 2026-09-16, re-measured 2026-09-18.
DECLARED: Final[frozenset[str]] = frozenset()

#: Re-measured 2026-09-18: 79 tracked paths, 77 of them read by the scan. Set below the measurement
#: on purpose -- a floor refuses an UNREAD tree, it is not a second pin on the count.
FILES_READ_FLOOR: Final = 70

#: How far past the floor this corpus may grow before the floor stops binding and must be re-taken.
#: 70 + 15 = 85 against today's 77. This is the side the old floor did not have, and its absence is
#: how a floor set AT 53 was allowed to drift to 53-against-77 unremarked.
FILES_READ_HEADROOM: Final = 15

_WHAT: Final = 'NO-CJK-IN-TRACKED-SOURCE'


def scan() -> Scan:
    """THE REAL SCAN over THIS repo's tracked corpus -- the one input a shared module cannot supply."""
    names = sorted(tracked_files(_ROOT))
    return trackedcjk.take_scan([_ROOT / name for name in names], root=_ROOT, exempt_prefixes=EXEMPT_PREFIXES)


def test_the_declared_set_is_exact_and_the_floor_binds_on_both_sides() -> None:
    """THE CHECK: the reading is proved to be a reading, then the ratchet is read in both directions."""
    trackedcjk.assert_no_undeclared_cjk(
        scan(),
        declared=DECLARED,
        floor=FILES_READ_FLOOR,
        headroom=FILES_READ_HEADROOM,
        what=_WHAT,
    )


def test_a_planted_violation_is_named_and_a_clean_file_is_not(tmp_path: Path) -> None:
    """CONTROL, BOTH DIRECTIONS, through the REAL scanner: a planted character convicts, ASCII does not."""
    trackedcjk.assert_the_scanner_still_convicts(tmp_path, exempt_prefixes=EXEMPT_PREFIXES)


def test_the_exemptions_are_matched_as_a_path_segment() -> None:
    """A nested exempt tree is exempt and a name merely STARTING with one is not.

    This repo carries a nested `.claude/memory/` under `src/optimi_lab/`, so a root-only prefix match
    would put dated agent notes back into the scanned population, and a substring match would open
    a `notarchived/` tree nobody named.
    """
    trackedcjk.assert_the_exemptions_match_a_path_segment(EXEMPT_PREFIXES)
    assert EXEMPT_PREFIXES, 'the shared exemption list is empty, so nothing above was exercised'


def test_the_declaration_is_the_shape_the_ratchet_keys_on() -> None:
    """A declaration is a set of PATHS, never a count -- the shape the ratchet can act on."""
    trackedcjk.assert_the_declaration_is_the_shape_the_ratchet_keys_on(DECLARED)


def test_neither_this_file_nor_the_kit_body_carries_a_literal() -> None:
    """A guard that forbids CJK may not be written with one -- BOTH bodies are scanned by it."""
    trackedcjk.assert_the_source_is_itself_clean()
    text = Path(__file__).read_text(encoding='utf-8')
    assert text.isascii(), 'the guard that forbids CJK may not be written with any non-ASCII character'


def test_this_guard_is_itself_inside_the_corpus_it_scans() -> None:
    """The floor COUNTS files; this NAMES one, so a corpus that stopped reaching tests/ still reds."""
    assert not scan_files([Path(__file__)], root=_ROOT).exempted, (
        'this guard sits under an exempt prefix, so it scans nothing of itself'
    )
