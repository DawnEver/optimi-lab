"""The always-loaded rule pages may only SHRINK, and they are pinned per file, not as a total.

`.claude/rules/*.md` is re-read on every turn, so every line in it is a cost paid on every
request. Nothing measured that cost in this repo until now, which is the whole of the DOCS-SPLIT
gap: the two pages existed and could grow without limit, and mechanism could migrate into them
from the module docstrings that own it.

THE ADMISSION CRITERION for a line in an always-loaded file, and the way past this ratchet is
never to raise a number:

1. **NOT ENFORCED.** If a test already refuses it, the prose goes and the test stays. Every guard
   in `tests/architecture/` is a line these pages do not have to carry.
2. **NOT DERIVABLE** from another line in the set -- the pages are one document for deduplication.
3. **NOT LOCAL.** A rule governing one module belongs in that module's docstring, where the code
   consults it. A declaration next to its enforcement cannot drift; prose in a rule page can.
4. **COSTLY IF WRONG.** Without the line the DEFAULT behaviour is wrong, not merely suboptimal.

PINNED PER FILE, and the reason is the one this family has already paid for: a total cannot say
WHICH page grew, so a rule migrating from one file to the other -- which changes nothing about the
cost and hides the movement -- would be absorbed silently. The per-file map is also TWO-SIDED: a
page that arrives unpinned reds, and a pin naming a page that no longer exists reds too, because a
budget a deletion freed is given back in the same edit rather than banked as slack for the next
arrival.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parents[2]
_RULES: Final = _ROOT / '.claude' / 'rules'

#: Every always-loaded page, to its line count MEASURED 2026-09-15. Each number RATCHETS DOWN:
#: lower it in the commit that shrinks the page, never raise it. Raising one is a claim that a new
#: rule could not be expressed inside the budget AND that no existing line fails a criterion above,
#: and that claim belongs in the commit message where a reader can refuse it.
_MEASURED: Final[dict[str, int]] = {
    # The domain facts no test catches: what this library refuses to become. 33 lines.
    '.claude/rules/invariant.md': 33,
    # How work gets done here. 22 lines.
    '.claude/rules/workflow.md': 22,
}

#: A floor on the scan: below this the walk did not reach `.claude/rules/` at all, and an empty
#: page set is indistinguishable from a compressed one.
PAGE_FLOOR: Final = 2


def rule_pages(root: Path) -> dict[str, int]:
    """THE SCAN. Every markdown page under *root*, by repo-relative path, to its line count."""
    return {
        path.relative_to(_ROOT).as_posix() if path.is_relative_to(_ROOT) else path.as_posix(): len(
            path.read_text(encoding='utf-8').splitlines()
        )
        for path in sorted(root.rglob('*.md'))
    }


def ratchet_breaks(measured: dict[str, int], pinned: dict[str, int]) -> dict[str, str]:
    """THE COMPARISON, both directions at once: an unpinned page, a stale pin, a page that grew.

    A page that SHRANK is also named, because the freed budget is paid back in the same edit. That
    is the half of a ratchet that goes missing: without it a compression pass banks slack and the
    next arrival spends it without anyone deciding to.
    """
    breaks: dict[str, str] = {}
    for page, lines in sorted(measured.items()):
        if page not in pinned:
            breaks[page] = f'{lines} lines and unpinned -- an always-loaded page nobody budgeted'
        elif lines > pinned[page]:
            breaks[page] = f'grew {pinned[page]} -> {lines}; find a line failing an admission criterion'
        elif lines < pinned[page]:
            breaks[page] = f'shrank {pinned[page]} -> {lines}; lower the pin in this same edit'
    for page in sorted(set(pinned) - set(measured)):
        breaks[page] = 'pinned and gone -- delete the pin with the page, or the waiver outlives it'
    return breaks


def test_the_rule_pages_hold_their_measured_budget() -> None:
    """THE CHECK, with the number of pages actually read asserted before any verdict is given."""
    assert _RULES.is_dir(), f'{_RULES} does not exist -- this test scanned nothing.'
    measured = rule_pages(_RULES)
    assert len(measured) >= PAGE_FLOOR, (
        f'the scan reached {len(measured)} rule page(s), below the {PAGE_FLOOR} floor. A ratchet over a '
        f'directory the walk did not enter reports exactly what a compressed one reports.'
    )
    breaks = ratchet_breaks(measured, _MEASURED)
    assert not breaks, (
        f'the always-loaded rule set moved: {breaks}. Every line here is read on every turn, so the way '
        f'past this number is to spend an existing line, never to raise the budget.'
    )


def test_the_ratchet_refuses_all_four_movements() -> None:
    """THE CONTROL. Plant each movement and call the REAL comparison, not a copy of it.

    Four shapes, and the two that are easy to leave out are the last two: a pin with no page, and
    a page that shrank without its pin following. Both are the waiver-nothing-uses side.
    """
    pinned = {'a.md': 10, 'b.md': 20}
    assert ratchet_breaks({'a.md': 10, 'b.md': 20}, pinned) == {}, 'the clean case must be silent.'
    assert 'grew 10 -> 11' in ratchet_breaks({'a.md': 11, 'b.md': 20}, pinned)['a.md']
    assert 'shrank 20 -> 4' in ratchet_breaks({'a.md': 10, 'b.md': 4}, pinned)['b.md']
    assert 'unpinned' in ratchet_breaks({'a.md': 10, 'b.md': 20, 'c.md': 7}, pinned)['c.md']
    assert 'pinned and gone' in ratchet_breaks({'a.md': 10}, pinned)['b.md']


def test_the_scan_counts_a_planted_page(tmp_path: Path) -> None:
    """THE CONTROL FOR THE SCAN ITSELF: an unread page and a zero-line page are different answers."""
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'one.md').write_text('a\nb\nc\n', encoding='utf-8')
    (tmp_path / 'nested' / 'two.md').write_text('', encoding='utf-8')
    (tmp_path / 'three.txt').write_text('not markdown\n', encoding='utf-8')
    counted = {Path(name).name: lines for name, lines in rule_pages(tmp_path).items()}
    assert counted == {'one.md': 3, 'two.md': 0}, (
        f'the scan answered {counted}. It must recurse, count lines exactly, and read only markdown -- '
        f'a page it cannot see is a budget nobody is spending.'
    )
