"""`NO-CJK-IN-TRACKED-SOURCE` -- this repo's declared set, driven by the REAL shared scanner.

The scan, the CJK ranges, the exemption list and the two-sided ratchet all live in
`lab_commons.dev.cjk`, authored once for the whole family. This module supplies only the two things
a shared registry cannot hold: THIS tree's corpus (its tracked files) and THIS tree's DECLARED SET
of files still carrying CJK.

`_DECLARED` IS EMPTY, AND THAT IS A MEASUREMENT RATHER THAN AN ASPIRATION. Measured 2026-09-16:
`git ls-files` names 55 tracked paths here; two of them sit under an exempt prefix
(`.claude/memory/`, `archived/` -- see `lab_commons.dev.cjk.EXEMPT_PREFIXES`), every remaining one
decodes as UTF-8, so the scan reads 53 files and finds zero CJK characters. optimi-lab is therefore
adopting this guard at ZERO, with nothing to ratchet down: the declaration exists to refuse the
FIRST CJK character that arrives, not to record a debt.

WHICH IS EXACTLY WHY THE FLOOR IS NOT OPTIONAL HERE. An empty declared set matched against an empty
found set is the vacuous green this family refuses by default: a scan that reached no files at all
reports precisely what this clean tree reports. `_FILES_READ_FLOOR` is what tells them apart, and it
is set AT the measured 53 rather than below it -- unlike a large, growing tree, this corpus is small
enough that any real shrinkage (an exemption widened by accident, a root that stopped resolving) is
a defect to look at rather than noise to absorb. Adding files only ever raises the count.

THE EXEMPTION MATCH IS ON A PATH SEGMENT, not a root prefix, and this test would have measured a
different number had it been otherwise -- `test_the_exemptions_are_matched_as_a_path_segment` pins
that property against the real shared predicate rather than trusting this docstring for it.

THE PLANTED CJK CHARACTER IS BUILT AT RUNTIME, `chr(0x4E2D)`, NEVER WRITTEN AS A LITERAL OR AS AN
ESCAPE THIS FILE'S TEXT WOULD CARRY. This module is itself tracked source and is scanned by the very
guard it installs; a control that plants a real character in its own source would be the one
violation the guard could never report. The file's text stays ASCII and the runtime value is the
real thing the scanner refuses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.cjk import EXEMPT_PREFIXES, Occurrence, assert_floor, exempted, ratchet, scan_files
from lab_commons.dev.rules import tracked_files

_ROOT: Final = Path(__file__).resolve().parents[2]

#: THE DECLARED SET -- repo-relative POSIX paths, exactly as `scan_files(..., root=_ROOT)` names
#: them. A NAMED SET rather than a count, so a file that is cleaned must be DELETED here in the same
#: edit and a newly dirty file cannot arrive silently. Empty: measured 2026-09-16, this tree carries
#: no CJK in tracked, non-exempt, decodable source.
DECLARED: Final[frozenset[str]] = frozenset()

#: Measured 2026-09-16: 55 tracked paths, 2 exempt, 0 undecodable, so the scan reads 53. Pinned AT
#: the measurement because this corpus only grows; a scan reading fewer files than this has stopped
#: reaching the tree, and its empty answer would mean nothing.
FILES_READ_FLOOR: Final = 53


def scan():
    """THE REAL SCAN over THIS repo's tracked corpus -- the one input a shared module cannot supply."""
    names = sorted(tracked_files(_ROOT))
    return scan_files([_ROOT / name for name in names], root=_ROOT)


def test_the_cjk_scan_reaches_a_plausible_corpus() -> None:
    """FLOOR: a clean tree and an unread one both report zero occurrences -- only this tells them apart."""
    found = scan()
    assert_floor(found.files_read, FILES_READ_FLOOR, what='NO-CJK-IN-TRACKED-SOURCE')


def test_the_declared_set_is_exact_and_two_sided() -> None:
    """THE PIN. `ratchet` refuses an undeclared violation and an orphaned declaration alike."""
    found = scan()
    problems = ratchet(found.occurrences, DECLARED)
    assert not problems, '\n'.join(problems)


def test_a_planted_undeclared_violation_is_named(tmp_path: Path) -> None:
    """CONTROL, direction one: plant a real CJK character in a real file, call the REAL scanner.

    The character is composed at runtime so that this module's own source text stays ASCII.
    """
    planted = tmp_path / 'caller.py'
    planted.write_text(f"LABEL = '{chr(0x4E2D)}'\n", encoding='utf-8')
    found = scan_files([planted], root=tmp_path)
    assert [occurrence.path for occurrence in found.occurrences] == ['caller.py'], (
        f'the scanner read {found.files_read} file(s) and reported {found.occurrences} for a planted CJK '
        f'character. A scan that cannot fail on a planted violation is not evidence about the tree.'
    )
    problems = ratchet(found.occurrences, DECLARED)
    assert len(problems) == 1, problems
    assert 'UNDECLARED' in problems[0], problems[0]
    assert 'caller.py:1' in problems[0], problems[0]


def test_a_planted_clean_file_is_green(tmp_path: Path) -> None:
    """CONTROL, direction two: the same real scanner over ASCII source reports nothing at all."""
    clean = tmp_path / 'caller.py'
    clean.write_text("LABEL = 'plain ascii'\n", encoding='utf-8')
    found = scan_files([clean], root=tmp_path)
    assert found.files_read == 1, 'the control must actually read its planted file.'
    assert found.occurrences == (), f'a clean file was reported as carrying CJK: {found.occurrences}'
    assert ratchet(found.occurrences, DECLARED) == ()


def test_an_orphaned_declaration_is_also_refused() -> None:
    """THE OTHER SIDE OF THE RATCHET: a declared file no longer carrying CJK is a stale waiver."""
    problems = ratchet((), declared=frozenset({'src/optimi_lab/cleaned_up.py'}))
    assert len(problems) == 1, problems
    assert 'ORPHANED' in problems[0], problems[0]
    assert 'cleaned_up.py' in problems[0], problems[0]


def test_the_exemptions_are_matched_as_a_path_segment() -> None:
    """The property that makes the 53 above the right number, asserted against the REAL predicate.

    A root-only prefix match would excuse nothing nested and refuse nothing spurious here today, so
    the tree alone cannot witness the difference -- these four cases can.
    """
    assert exempted('.claude/memory/2026/09/15/entry.md')
    assert exempted('src/optimi_lab/.claude/memory/entry.md'), 'a nested memory tree is exempt too.'
    assert not exempted('src/optimi_lab/notarchived/model.py'), 'a substring is not a path segment.'
    assert not exempted('tests/architecture/test_no_cjk_in_tracked_source.py')
    assert EXEMPT_PREFIXES, 'the shared exemption list is empty, so nothing above was exercised.'


def test_the_planted_control_uses_no_literal_cjk() -> None:
    """A guard that forbids CJK may not be written with one -- THIS FILE IS SCANNED BY ITSELF."""
    text = Path(__file__).read_text(encoding='utf-8')
    assert text.isascii(), 'the guard that forbids CJK may not be written with any non-ASCII character.'
    assert 'chr(0x4E2D)' in text, 'the planted control must build its character at runtime.'


def test_the_declaration_is_the_shape_the_ratchet_keys_on() -> None:
    """A declaration is a set of PATHS, never a count -- the shape `ratchet` can act on."""
    assert isinstance(DECLARED, frozenset)
    assert ratchet((Occurrence(path='x.py', line=1, col=1, char=chr(0x4E2D)),), frozenset({'x.py'})) == ()
