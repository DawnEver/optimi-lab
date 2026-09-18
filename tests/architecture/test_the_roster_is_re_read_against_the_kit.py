"""THE ROSTER IS RE-READ AGAINST THE KIT -- a declared side goes stale silently, and nothing noticed.

THE DEFECT, MEASURED BEFORE THIS FILE EXISTED. `_placement.py` decides a row's side by a DENSITY bar,
which answers *is this file mostly generic?*. That is a good question and a DIFFERENT one from *has
the family already expressed this?*, and nothing in this tree ever asked the second. So a row keeps
its side after its subject lands upstream, and the only thing that would have caught it is a human
re-reading every row. Across three tranches of the family's largest roster the error ran ONE WAY
every time -- 17 rows declared MOVES, 7 were real, 5 were splits and 5 were already in the kit.

THE WHOLE ASSERTION BODY IS THE FAMILY'S. `lab_commons.dev.supersede` holds the two detectors (a kit
module NAMING a consumer path, and the consumer IMPORTING that module), the seven grades, the ruler
and both floors; `lab_commons.dev.famtests.rostercensus` holds the arms that sit on top of it. Those
arms were hand-written TWICE on the same day -- here at 198 lines and in wdg-lab at 190 -- and,
diffed with each repo's own name blanked, the two files were IDENTICAL IN CODE: every line that
differed was a docstring. What is left here is the half the kit refuses to guess -- EVERY REPO-SHAPED
FACT ARRIVES AS AN ARGUMENT WITH NO DEFAULT -- which is where optimi-lab's four answers are written
down: which rows, which root, which dotted package, and how little a census may read before finding
nothing stops meaning anything.

THE WAIVER IS GONE, AND ITS DEPARTURE IS THE RATCHET'S OTHER SIDE FIRING. Until 2026-09-18 this file
carried `SUBPACKAGE_ADOPTED`, a NAMED SET of six rows that graded `NAMED_ONLY` -- the kit named the
file and nothing corroborated it -- while all six were FULLY ADOPTED, importing the exact module that
named them. The cause was inside the instrument: `imported_kit_modules` resolved
`from lab_commons.dev.famtests import allowguard` to the segment `famtests`, one level above the name
`kit_modules` publishes, so the IMPORT detector could not fire on ANY sub-package adoption. That was
routed back rather than absorbed and is fixed upstream in `1aa2738`. RE-MEASURED against
`0.2.2.dev83+g3c70b5835`: all six now grade `PARTIAL`, the `NAMED_ONLY` population is EMPTY, and the
arm is DELETED rather than pinned at zero -- a waiver nothing uses is as wrong as a capability that
disappeared, and the kit's own `assert_waiver_is_the_named_set` refuses an empty declaration for
exactly that reason. The day a new invisible adoption arrives, the arm comes back with its member.

WHAT THIS DOES NOT PROVE. The census is blind to a LIVE FORK that imports nothing and that no kit
docstring names; `supersede` says so itself and calls overlap a ruler rather than a detector. Green
here means no row is stale BY THE TWO DETECTORS, never that the roster is right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from _placement import MOVES, PLACEMENT
from lab_commons.dev import supersede
from lab_commons.dev.famtests import rostercensus

#: This checkout: `tests/architecture/<this file>` sits two directories below it.
ROOT: Final = Path(__file__).resolve().parents[2]

#: THE KIT DIRECTORY'S OWN DOTTED PATH, and the depth it implies is the whole reason it is spelled
#: in full. One level short resolves every `lab_commons.dev.x` import to the token `dev`, which
#: matches no kit module, so the IMPORT detector silently never fires and every row reads UNTOUCHED
#: -- which is what a fully forked tree reports. `take_census` refuses that rather than returning it.
KIT_PACKAGE: Final = 'lab_commons.dev'

#: THE KIT FLOOR, RE-MEASURED 2026-09-18 against `0.2.2.dev83+g3c70b5835`: `kit_modules` reads 51
#: modules, up from the 46 this floor was first set against. The floor sits well below that with room
#: for a module being renamed or withdrawn, and far above the zero a mis-resolved package returns --
#: against which every row grades UNTOUCHED, the answer a finished migration gives.
KIT_MODULE_FLOOR: Final = 35


def census() -> supersede.Census:
    """The roster of this checkout, judged against the installed kit, with all four answers stated here."""
    return supersede.take_census(
        {path: placement.side for path, placement in PLACEMENT.items()},
        supersede.kit_modules(rostercensus.kit_directory(package=KIT_PACKAGE)),
        root=ROOT,
        package=KIT_PACKAGE,
        row_floor=len(PLACEMENT),
        module_floor=KIT_MODULE_FLOOR,
    )


def test_the_census_reaches_the_whole_roster_and_a_kit_that_is_really_there() -> None:
    """THE FLOOR'S OWN ARM: every row is judged, and finding nothing must not read as green."""
    rostercensus.assert_reach(census(), rows_declared=len(PLACEMENT), module_floor=KIT_MODULE_FLOOR)


def test_no_row_whose_subject_the_kit_already_holds_is_still_declared_moves() -> None:
    """THE CHECK. A `MOVES` row the kit already supersedes is the stale-roster defect, named."""
    rostercensus.assert_no_stale_moves(census(), moves_side=MOVES)


def test_no_row_the_kit_names_is_left_uncorroborated() -> None:
    """THE OTHER DIRECTION, now that the waiver is empty: a `NAMED_ONLY` row is a real stale row.

    While the detector was blind to sub-package imports this set held six false positives and had to
    be waived by name. It is empty against `0.2.2.dev83+g3c70b5835`, so the honest arm is equality
    against nothing declared -- and an arrival here is a row the kit names whose file imports it not
    at all, which is precisely the stale roster this module exists to find.
    """
    found = rostercensus.named_only_paths(census())
    assert not found, (
        f'{sorted(found)} are named by a kit module and corroborated by nothing in their own text. '
        f'Either the row has not adopted what supersedes it, or the import detector has gone blind '
        f'to a new spelling -- read the file before waiving it.'
    )


def test_the_census_still_convicts_a_planted_supersession() -> None:
    """THE CONTROL: plant both shapes and call the REAL grader, so a silent census cannot pass."""
    rostercensus.assert_the_grader_still_convicts(moves_side=MOVES)
