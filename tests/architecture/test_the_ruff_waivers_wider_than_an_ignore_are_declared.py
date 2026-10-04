"""THE TWO RUFF WAIVERS WIDER THAN AN IGNORE, declared for optimi_lab and judged by the kit.

``exclude`` drops every selector over a subtree and names no code; ``per-file-ignores`` drops a named
code over a glob. Every arm stated over the global ignore list is blind to both.
`lab_commons.dev.ruffwaivers` reads them across EVERY spelling ruff honours, from the config ruff
RESOLVES, and refuses both sides of the ratchet. These rows used to live in lab-commons under a
neutral name; a fact about this tree belongs in this tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.ruffwaivers import MACHINE_EXCLUDES, assert_waivers_declared

_ROOT: Final = Path(__file__).resolve().parents[2]

#: EVERY EXCLUDE THAT HIDES REAL SOURCE, MEASURED 2026-10-02 from the resolved config. `ignore` and
#: `output` are scratch trees; neither is source under review. It may only SHRINK.
TREE_EXCLUDES: Final[tuple[str, ...]] = ('ignore', 'output')

#: EVERY PER-FILE WAIVER as ``'<glob>::<code>'``. MEASURED 2026-10-02: none, and the empty set is a
#: stated answer -- the first one to arrive must arrive here with its reason.
PER_FILE_WAIVERS: Final[tuple[str, ...]] = ()


def test_every_exclude_that_hides_source_is_a_declared_row() -> None:
    """The widest waiver a ruff config can write may not arrive as a line nothing reads -- both sides."""
    live = assert_waivers_declared(_ROOT, 'exclude', TREE_EXCLUDES, machine=MACHINE_EXCLUDES)
    assert live == frozenset(TREE_EXCLUDES)


def test_no_per_file_waiver_is_undeclared() -> None:
    """Equality with the declared set, read across every spelling ruff honours."""
    assert assert_waivers_declared(_ROOT, 'per-file-ignores', PER_FILE_WAIVERS) == frozenset(PER_FILE_WAIVERS)
