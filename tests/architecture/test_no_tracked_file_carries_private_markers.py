"""A PUBLISHED repo: no tracked file may name a machine, a person, a home path or a chat id.

This repo is PUBLIC on its forge (checked 2026-10-02), so everything `git ls-files` lists is what a
push publishes. The scan is the kit's `lab_commons.dev.privatemarkers`: generic patterns committed
there, plus the machine-local denylist at ``~/.claude/private-markers``, reported by KIND only so the
check cannot leak what it guards. The kit's own suite holds the planted positives and negatives; this
file holds the facts that are this repo's -- that its tree is clean, and which names it publishes on
purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.privatemarkers import load_markers, scan_text, scan_tree, tracked_files

_ROOT: Final = Path(__file__).resolve().parents[2]

#: The fewest tracked files the scan must read. MEASURED 2026-10-02 at 85; a scan of nothing finds nothing.
_FILE_FLOOR: Final = 70

#: Names this repo publishes ON PURPOSE, masked before the machine-local denylist applies (it serves every
#: repo, the kit included, and the kit names no consumer). Its own name, and the two PUBLIC siblings it
#: declares as dependents in its own manifest and tests. Every other denylist entry still applies.
PUBLIC_NAMES: Final[tuple[str, ...]] = ('optimi-lab', 'wdg-lab', 'wdg_lab', 'motronics')


def test_the_scan_reads_this_tree() -> None:
    """FLOOR-ON-EVERY-SCAN: a clean result over an unread tree is the vacuous green."""
    assert len(tracked_files(_ROOT)) >= _FILE_FLOOR


def test_a_planted_home_path_in_a_real_file_is_found() -> None:
    """PLANTED-CONTROL: this repo's README with one home path appended, through the same reader."""
    text = (_ROOT / 'README.md').read_text(encoding='utf-8')
    # Assembled from halves so this file does not plant a hit in the tree scan below.
    planted = text + '\nsee C:' + '\\Users\\someone\\repo\n'
    hits = scan_text(planted, load_markers(), path='README.md', public=PUBLIC_NAMES)
    assert [kind for _, kind, _ in hits] == ['windows home path']


def test_no_tracked_file_carries_a_private_marker() -> None:
    """THE PROPERTY, over every tracked text file, with this host's denylist when it has one."""
    hits = scan_tree(_ROOT, load_markers(), public=PUBLIC_NAMES)
    assert hits == [], 'private values in tracked files:\n' + '\n'.join(hits)
