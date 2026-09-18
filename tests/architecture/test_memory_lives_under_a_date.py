"""Every memory entry sits under `YYYY/MM/DD/`, in the trees this repo declares, with its own date.

WHY A SHAPE GUARD AND NOT A DOCUMENTED CONVENTION. The convention is `.claude/memory/YYYY/MM/DD/`
across every repo in this fleet, and the fleet has already measured what happens without a check:
one writer composed its records under a literal `lanes/` directory instead of a date, grew a
seventeen-file parallel tree beside the real one, and the index builder listed every file without a
word. It survived for days because the layout was an instruction, and an instruction is not a
control.

THE READERS AND THE ARMS ARE THE FAMILY'S, and this file no longer holds a line of either:
`lab_commons.dev.famtests.datedmemory` ships the walk, the four readings, the four assertions and
the planted control, and `lab_commons.dev.floors` ships BOTH sides of the entry floor. What is left
here is the half the kit refuses to guess -- every repo fact arrives as a keyword argument with no
default, and this file is where optimi-lab's five answers are written down: which trees, what is not
an entry, what is never walked, and the two numbers that bound the population.

THE SILENT ENTRY IS NOW ITS OWN POPULATION, AND THAT IS A BEHAVIOUR CHANGE RATHER THAN A SWAP. This
file used to fold "no `created:` field" into `date_disagreements` as the reason string ``no
`created:` field``, so one assertion carried two populations and a pin over it could not have said
which half moved. The kit splits them -- :func:`~lab_commons.dev.famtests.datedmemory.silent` and
:func:`~lab_commons.dev.famtests.datedmemory.date_disagreements` -- and the split is what shipped, so
adopting it means this repo now answers the two questions separately. It also means a MISSING field
is no longer named twice: the kit asks only DATED markdown, because an undated file is already named
by the shape reading and one fix should not red two arms.

THE TREE WALK REPLACES A SINGLE HARD-CODED DIRECTORY. The old file asserted `.claude/memory` existed
and read only that; a second memory tree anywhere in this checkout was invisible to it. The trees are
now FOUND BY WALKING and compared by EQUALITY against the declaration below, which is the only reading
that can report a tree nobody decided on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import datedmemory

_ROOT: Final = Path(__file__).resolve().parents[2]

#: MEASURED 2026-09-18 through the kit readers: the walk finds ONE tree holding ONE entry. One is
#: the weakest a floor can be -- it separates a tree that exists from a directory that does not, and
#: nothing more. It goes UP with the tree; a walk reaching zero is refused rather than reported clean.
MEMORY_FILE_FLOOR: Final = 1

#: THE OTHER SIDE OF THE FLOOR, and the side no copy in this family ever wrote. A floor measured
#: against a tree that has since grown refuses only a total collapse and passes a walk that lost most
#: of its corpus. This is how far past its floor the population may grow before the floor is
#: RE-MEASURED -- never how much headroom the floor may be given.
MEMORY_FILE_HEADROOM: Final = 50

#: THE TREES, as a named set rather than a count. A second is a decision somebody makes here.
MEMORY_TREES: Final = ('.claude/memory',)

#: Directories the tree walk never descends into: build output and the virtual environment, neither
#: of which is authored and both of which are large enough to make the walk answer about something
#: else entirely.
NOT_WALKED: Final = frozenset({'.git', '.venv', '__pycache__', 'node_modules'})

#: NOT ENTRIES. `_meta.json` is written beside the entries by the memory tooling and reaches no index
#: and no other checkout; `.gitkeep` is how an empty tree stays tracked at all. Neither is a record
#: somebody wrote, and a real fork is in neither set and is still caught.
NON_ENTRY_NAMES: Final = frozenset({'_meta.json', '.gitkeep'})

#: THE SILENT ENTRIES ON RECORD -- dated markdown carrying no `created:` field. EMPTY is accepted
#: here and is the state the ratchet exists to REACH; what stops it being vacuous is the entry floor
#: above, which the check binds before it compares anything.
MEMORY_FRONTMATTER_DEBT: Final[frozenset[str]] = frozenset()

#: THE ENTRIES WHOSE HEADER CONTRADICTS THEIR DIRECTORY, pinned WITH the date each one claims: a name
#: alone cannot tell a copied-forward header from a moved directory.
MEMORY_DATE_DISAGREEMENT: Final[dict[str, str]] = {}


def scan() -> datedmemory.MemoryScan:
    """One walk over every declared tree, with this repo's five answers supplied."""
    return datedmemory.take_scan(
        _ROOT,
        trees=MEMORY_TREES,
        non_entry_names=NON_ENTRY_NAMES,
        not_walked=NOT_WALKED,
    )


def test_the_memory_trees_are_the_ones_this_repo_declares() -> None:
    """A second tree is a fork, and a name missing here is a tree that stopped being read."""
    datedmemory.assert_trees_are_the_named_set(_ROOT, declared=MEMORY_TREES, not_walked=NOT_WALKED)


def test_every_memory_entry_sits_under_its_own_date() -> None:
    """THE CHECK, BOTH sides of the floor asserted first, whatever route a file arrived by."""
    datedmemory.assert_every_entry_is_dated(scan(), floor=MEMORY_FILE_FLOOR, headroom=MEMORY_FILE_HEADROOM)


def test_the_entries_with_no_created_field_are_the_set_on_record() -> None:
    """THE DEBT, two-sided. A new silent entry reds; so does a pinned name that gained the field."""
    datedmemory.assert_silent_entries_are_the_named_set(scan(), declared=MEMORY_FRONTMATTER_DEBT)


def test_the_entries_that_contradict_their_directory_are_the_set_on_record() -> None:
    """The other half of the frontmatter property, pinned WITH the wrong date each one claims."""
    datedmemory.assert_dates_agree(scan(), declared=MEMORY_DATE_DISAGREEMENT)


def test_the_readers_still_convict_a_planted_fork_and_a_planted_stale_date(tmp_path: Path) -> None:
    """THE CONTROL. Plant all four shapes in a real tree and drive the REAL readers over it."""
    datedmemory.assert_the_readers_still_convict(tmp_path, non_entry_names=NON_ENTRY_NAMES)
