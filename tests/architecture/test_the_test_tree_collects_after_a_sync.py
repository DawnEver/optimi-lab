"""COLLECTION-REACH, for THIS checkout: what the CI selection leaves the test tree able to COLLECT.

A sync that leaves the runner able to start can still leave a repo with NO VERDICT: pytest IMPORTS
every module it collects, and a module-scope import of a distribution the selection did not install
aborts that file having judged nothing. `lab_commons.dev.collectcensus` is the census over that join;
this file declares this repo's one selection and calls the kit over this checkout's WORKING TREE.
The row used to live in lab-commons under a neutral name; a fact about this tree belongs in this tree.

Nothing here installs, syncs or prunes, and nothing imports a scanned file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.collectcensus import CollectRow, assert_reaches

_HERE: Final = 'optimi-lab'
_ROOT: Final = Path(__file__).resolve().parents[2]

#: The CI selection (`extras: 'dev'`) and what it leaves the tree. Three sets, each by EQUALITY.
ROWS: Final[tuple[CollectRow, ...]] = (
    CollectRow(
        repo=_HERE,
        selected=('dev',),
        errors=frozenset(),
        degrades=frozenset(),
        unresolved=frozenset(),
        why=(
            'A CLEAN REPO, and the clean answer is the claim: its `dev` extra is the only thing '
            'putting lab-commons in the environment at all, so this row is what says that the ONE '
            'selection CI makes also carries every import the test tree needs at module scope. An '
            'empty residue means no name needs a reason, and the first one to arrive reds here.'
        ),
    ),
)

#: The fewest test files the read must hold. MEASURED 2026-10-02 at 34; a tree that was not read
#: reports the same empty stranded set as a tree whose every import survives.
FILE_FLOORS: Final = {_HERE: 28}


def test_the_recorded_selection_still_leaves_the_tree_what_the_row_records() -> None:
    """THE CENSUS. Equality on errors, degrades AND unresolved, in both directions."""
    assert assert_reaches(ROWS, {_HERE: _ROOT}, FILE_FLOORS, here=_HERE, repo_floor=1, row_floor=1) == {_HERE}
