"""COLLECTION-REACH, for THIS checkout: what the CI selection leaves the test tree able to COLLECT.

A sync that leaves the runner able to start can still leave a repo with NO VERDICT: pytest IMPORTS
every module it collects, and a module-scope import of a distribution the selection did not install
aborts that file having judged nothing. `dev_lab.dev.collectcensus` is the census over that join;
this file declares this repo's one selection and calls the kit over this checkout's WORKING TREE.
The row used to live in lab-commons under a neutral name; a fact about this tree belongs in this tree.

Nothing here installs, syncs or prunes, and nothing imports a scanned file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.collectcensus import CollectRow, assert_reaches

_HERE: Final = 'optimi-lab'
_ROOT: Final = Path(__file__).resolve().parents[2]

#: The declared dev dependency group and what it leaves the tree, compared by equality.
ROWS: Final[tuple[CollectRow, ...]] = (
    CollectRow(
        repo=_HERE,
        selected=(),
        groups=('dev',),
        errors=frozenset(),
        degrades=frozenset(),
        unresolved=frozenset({'sklearn'}),
        why=(
            'The dev dependency group installs the independent development kit and test oracles. '
            'This row checks the actual group axis, not an optional-dependency extra. sklearn is '
            'the scikit-learn test oracle; its import spelling remains an explicit unresolved '
            'supplier until the shared alias registry can derive it.'
        ),
    ),
)

#: The fewest test files the read must hold. MEASURED 2026-10-23 at 106; a tree that was not read
#: reports the same empty stranded set as a tree whose every import survives.
FILE_FLOORS: Final = {_HERE: 100}


def test_the_recorded_selection_still_leaves_the_tree_what_the_row_records() -> None:
    """THE CENSUS. Equality on errors, degrades AND unresolved, in both directions."""
    assert assert_reaches(ROWS, {_HERE: _ROOT}, FILE_FLOORS, here=_HERE, repo_floor=1, row_floor=1) == {_HERE}
