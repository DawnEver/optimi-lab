"""`INJECTED-DOC-WIDTH-CEILING` -- this repo's declared over-width sites, driven by the REAL scanner.

The ceiling (120 columns), the injected-doc corpus definition and the two-sided ratchet all live in
`lab_commons.dev.docwidth`, authored once for the family. This module supplies only what a shared
registry cannot hold: THIS tree's real injected-doc corpus, and the DECLARED SET of `path:line`
sites still past the ceiling.

MEASURED HERE 2026-09-16, and it had never been measured in this repo before this pass: of 55
tracked paths, exactly 2 are injected documents by the shared definition (`AGENTS.md` / `CLAUDE.md`
by basename anywhere in the tree -- this repo has neither -- plus everything under
`.claude/rules/`, which here is `invariant.md` and `workflow.md`). Scanning those two finds ZERO
lines past 120 columns, so `DECLARED` is empty and there is no width debt to record. NOTHING WAS
REFLOWED to reach that: the measurement found the pages already inside the ceiling.

THE SAME PAGES ARE ALSO PINNED BY LINE COUNT in `test_the_rules_pages_are_a_ratchet.py`, and the two
pins measure different things on purpose: a rewrite can lower a line count while doubling a line's
width, and a count-only pin reads that as an improvement. This is the missing dimension, not a
duplicate -- and because nothing here is rewrapped, this pass cannot move the line-count pin at all.

THE FLOOR IS THE POINT OF A ZERO MEASUREMENT. An empty declared set against an empty found set is
the vacuous green this family refuses: a corpus definition that stopped matching anything reports
exactly what these two compliant pages report. `FILES_READ_FLOOR` is pinned AT the measured 2 -- the
corpus can only grow, and a rules page that stops being seen by the scan is a defect rather than
noise. `test_the_scan_reads_every_doc_handed_to_it` closes the other half of the same gap: a file
handed in and not read is a file not checked, however large the corpus is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.docwidth import (
    WIDTH_CEILING,
    Overwidth,
    assert_width_floor,
    injected_docs,
    is_injected_doc,
    scan_widths,
    width_ratchet,
)
from lab_commons.dev.rules import tracked_files

_ROOT: Final = Path(__file__).resolve().parents[2]

#: THE DECLARED SET, as `path:line` -- exactly how `width_ratchet` keys its own found set. A NAMED
#: SET rather than a count, so a fixed site must be DELETED here in the same edit that fixes it and
#: a newly over-width line cannot arrive silently. Empty: measured 2026-09-16, no injected document
#: in this tree has a line past `WIDTH_CEILING` columns.
DECLARED: Final[frozenset[str]] = frozenset()

#: Measured 2026-09-16: the injected-doc corpus here is 2 files, `.claude/rules/invariant.md` and
#: `.claude/rules/workflow.md`. Pinned AT the measurement because the corpus only grows; a scan
#: reaching fewer files has stopped matching the corpus, and its empty answer would mean nothing.
FILES_READ_FLOOR: Final = 2


def scan():
    """THE REAL SCAN over THIS repo's injected docs, plus how many were handed to it."""
    docs = injected_docs(sorted(tracked_files(_ROOT)))
    return scan_widths([_ROOT / name for name in docs], root=_ROOT), len(docs)


def test_the_width_scan_reaches_a_plausible_injected_corpus() -> None:
    """FLOOR: a compliant corpus and an unread one both report zero -- only this tells them apart."""
    found, _handed_in = scan()
    assert_width_floor(found.files_read, FILES_READ_FLOOR, what='INJECTED-DOC-WIDTH-CEILING')


def test_the_scan_reads_every_doc_handed_to_it() -> None:
    """A file the scan could not open is a file it did not check -- never a compliant one."""
    found, handed_in = scan()
    assert found.files_read == handed_in, (
        f'{handed_in} injected docs were handed to the scan and it read {found.files_read}; '
        f'undecodable: {found.undecodable}.'
    )


def test_the_declared_set_is_exact_and_two_sided() -> None:
    """THE PIN. `width_ratchet` refuses an undeclared over-width line and an orphaned waiver alike."""
    found, _handed_in = scan()
    problems = width_ratchet(found.overwidth, DECLARED)
    assert not problems, '\n'.join(problems)


def test_a_planted_overwide_document_is_named(tmp_path: Path) -> None:
    """CONTROL, direction one: a real file one column over the ceiling, through the REAL scanner."""
    planted = tmp_path / 'AGENTS.md'
    planted.write_text('ok\n' + 'x' * (WIDTH_CEILING + 1) + '\n', encoding='utf-8')
    assert is_injected_doc('AGENTS.md'), 'the planted file must be in the corpus the guard defines.'
    found = scan_widths([planted], root=tmp_path)
    assert [(site.line, site.width) for site in found.overwidth] == [(2, WIDTH_CEILING + 1)], (
        f'the scanner reported {found.overwidth} for a line one column over the ceiling. A scan that '
        f'cannot fail on a planted violation is not evidence about the corpus.'
    )
    problems = width_ratchet(found.overwidth, DECLARED)
    assert len(problems) == 1, problems
    assert 'UNDECLARED' in problems[0], problems[0]
    assert 'AGENTS.md:2' in problems[0], problems[0]


def test_a_planted_document_at_the_ceiling_is_green(tmp_path: Path) -> None:
    """CONTROL, direction two: a line exactly AT the ceiling is compliant, and reports nothing."""
    clean = tmp_path / 'AGENTS.md'
    clean.write_text('x' * WIDTH_CEILING + '\n', encoding='utf-8')
    found = scan_widths([clean], root=tmp_path)
    assert found.files_read == 1, 'the control must actually read its planted file.'
    assert found.overwidth == (), f'a line at exactly {WIDTH_CEILING} columns is not over it: {found.overwidth}'
    assert width_ratchet(found.overwidth, DECLARED) == ()


def test_an_orphaned_declaration_is_also_refused() -> None:
    """THE OTHER SIDE OF THE RATCHET: a declared site no longer over width is a stale waiver."""
    problems = width_ratchet((), declared=frozenset({'.claude/rules/workflow.md:12'}))
    assert len(problems) == 1, problems
    assert 'ORPHANED' in problems[0], problems[0]
    assert 'workflow.md:12' in problems[0], problems[0]


def test_the_corpus_is_this_repo_rules_pages() -> None:
    """WHAT IS BEING CHECKED, named -- a floor over the wrong corpus is a floor over nothing."""
    docs = injected_docs(sorted(tracked_files(_ROOT)))
    assert set(docs) == {'.claude/rules/invariant.md', '.claude/rules/workflow.md'}, docs
    assert not is_injected_doc('.claude/memory/2026/09/15/entry.md'), 'a dated memory entry is not injected.'
    assert width_ratchet((Overwidth(path='a.md', line=3, width=999),), frozenset({'a.md:3'})) == ()
