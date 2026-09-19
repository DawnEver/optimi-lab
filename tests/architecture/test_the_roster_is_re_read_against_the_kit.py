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

THE WAIVER LEFT AND CAME BACK, AND BOTH DIRECTIONS ARE THE RATCHET WORKING -- read the entry on
`NAMED_BY_AN_EXAMPLE` below for today's single member and why it is a false positive inside the
instrument rather than an unadopted row. The first departure is kept here because it is the better
lesson. Until 2026-09-18 this file
carried `SUBPACKAGE_ADOPTED`, a NAMED SET of six rows that graded `NAMED_ONLY` -- the kit named the
file and nothing corroborated it -- while all six were FULLY ADOPTED, importing the exact module that
named them. The cause was inside the instrument: `imported_kit_modules` resolved
`from lab_commons.dev.famtests import allowguard` to the segment `famtests`, one level above the name
`kit_modules` publishes, so the IMPORT detector could not fire on ANY sub-package adoption. That was
routed back rather than absorbed and is fixed upstream in `1aa2738`. RE-MEASURED against
`0.2.2.dev83+g3c70b5835`: all six now grade `PARTIAL`, the `NAMED_ONLY` population is EMPTY, and the
arm was DELETED rather than pinned at zero -- a waiver nothing uses is as wrong as a capability that
disappeared, and the kit's own `assert_waiver_is_the_named_set` refuses an empty declaration for
exactly that reason. That entry ended "the day a new invisible adoption arrives, the arm comes back
with its member", and one day later it did -- for a cause the sentence did not anticipate, which is
the part worth keeping: the prediction named the SHAPE (a row the kit names and nothing corroborates)
and was right, while the CAUSE was neither a blind detector nor an unadopted row.

WHAT THIS DOES NOT PROVE. The census is blind to a LIVE FORK that imports nothing and that no kit
docstring names; `supersede` says so itself and calls overlap a ruler rather than a detector. Green
here means no row is stale BY THE TWO DETECTORS, never that the roster is right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from _placement import PLACEMENT
from lab_commons.dev import supersede
from lab_commons.dev.famtests import rostercensus
from lab_commons.dev.famtests.placement import MOVES

#: This checkout: `tests/architecture/<this file>` sits two directories below it.
ROOT: Final = Path(__file__).resolve().parents[2]

#: THE KIT DIRECTORY'S OWN DOTTED PATH, and the depth it implies is the whole reason it is spelled
#: in full. One level short resolves every `lab_commons.dev.x` import to the token `dev`, which
#: matches no kit module, so the IMPORT detector silently never fires and every row reads UNTOUCHED
#: -- which is what a fully forked tree reports. `take_census` refuses that rather than returning it.
KIT_PACKAGE: Final = 'lab_commons.dev'

#: THE KIT FLOOR, RE-MEASURED 2026-09-19 against `0.2.2.dev166+g7b7661ba9`: `kit_modules` reads 74
#: modules -- 46 when this floor was first set, then 51, 57, 61, 67, and 74 now. The floor must sit
#: below that with room for a module being renamed or withdrawn, and far above the zero a mis-resolved
#: package returns -- against which every row grades UNTOUCHED, the answer a finished migration gives.
#:
#: IT WAS RE-TAKEN AT 61 RATHER THAN LEFT AT 54, WHICH `SlackFloor` HAD ALREADY CONVICTED: 54 against
#: 74 is 20 clear, past the 16 headroom, so the floor had stopped separating a full read from a broken
#: one. The remedy for a floor the kit has outgrown is to RE-MEASURE THE FLOOR, never to widen the
#: headroom -- a headroom raised to swallow its own breach is the escape hatch with no ceiling. This
#: is the SECOND consecutive day the arm has fired, which is the arm working: lab-commons `main`
#: landed 23 commits between `dev140` and `dev166` and this repo refreshed through `scripts/dep.py`.
#:
#: 61 IS THIS CHECKOUT'S OWN NUMBER, NOT A NUMBER COPIED SIDEWAYS. The rule this repo has applied at
#: every re-take is a slack of 13 below the measured reading (48 was taken against 61, 54 against 67);
#: 74 - 13 = 61. The quantity being floored is the INSTALLED KIT, which is shared with wdg-lab, while
#: the headroom prices how much of it THIS repo will silently lose, which is not.
KIT_MODULE_FLOOR: Final = 61

#: How far past the floor the kit may grow before the floor stops binding and must be re-taken.
#: 61 + 16 = 77 against today's 74. The kit grew 67 -> 74 in a day, so this will bind again soon --
#: and that is the arm working, not a nuisance. The headroom is UNCHANGED at 16 on purpose: it is the
#: side that refuses, and the breach was answered by moving the floor.
KIT_MODULE_HEADROOM: Final = 16


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
    """THE FLOOR'S OWN ARM, NOW TWO-SIDED IN THE KIT: every row is judged, an under-read kit is
    refused, AND a floor the kit has outgrown is refused too.

    The second side used to be a local arm standing beside this call because `assert_reach` did not
    take a headroom. It does now, so the two halves are one argument again rather than one imported
    and one written by hand -- which is what let 35 survive two kit growths here without anyone
    noticing.
    """
    rostercensus.assert_reach(
        census(),
        rows_declared=len(PLACEMENT),
        module_floor=KIT_MODULE_FLOOR,
        module_headroom=KIT_MODULE_HEADROOM,
    )


def test_no_row_whose_subject_the_kit_already_holds_is_still_declared_moves() -> None:
    """THE CHECK. A `MOVES` row the kit already supersedes is the stale-roster defect, named."""
    rostercensus.assert_no_stale_moves(census(), moves_side=MOVES)


#: THE WAIVER CAME BACK 2026-09-19 WITH ONE MEMBER, WHICH IS THIS ARM'S OTHER SIDE FIRING. It was
#: DELETED on 2026-09-18 when the sub-package import bug was fixed upstream and its six members all
#: regraded `PARTIAL`; the file's own docstring said "the day a new invisible adoption arrives, the
#: arm comes back with its member." It has, and the member is declared rather than the assertion
#: loosened.
#:
#: THE READING, AND IT IS NOT AN ADOPTION. `famtests.storedreadings` shipped in
#: `0.2.2.dev166+g7b7661ba9` and its docstring names `scripts/dep.py` at line 82 -- as a WORKED
#: EXAMPLE of the calibration comment it deliberately does NOT read (`ADMIT scripts/dep.py own= 22
#: repo=1`). So the PROVENANCE detector fires on a sentence whose whole content is that the module
#: has nothing to do with that file, and the IMPORT detector correctly does not: `scripts/dep.py`
#: imports `lab_commons.dev.dep`, `boxlock` and `verify`, and importing `storedreadings` there would
#: be wrong. This is a FALSE POSITIVE INSIDE THE INSTRUMENT, the same class as the six of 2026-09-18
#: and a different cause, and the remedy is upstream: this family's own integration page already
#: rules that naming the SHAPE of a thing beats naming an EXAMPLE of it, because an example is a
#: spelling waiting to be retired. ROUTED BACK rather than absorbed. The day that docstring names a
#: shape instead of a path, this entry leaves and the arm goes with it.
NAMED_BY_AN_EXAMPLE: Final = {'scripts/dep.py': 'storedreadings'}


def test_no_row_the_kit_names_is_left_uncorroborated() -> None:
    """THE OTHER DIRECTION: a `NAMED_ONLY` row is a stale row unless this repo has READ it and said why.

    Compared by EQUALITY in both directions, which is what makes the waiver a ratchet rather than a
    hole: an ARRIVAL is a real stale row that nobody has looked at, and a DEPARTURE is a fix that
    must delete its entry in the same commit. The kit refuses an EMPTY declaration outright, so the
    arm cannot be kept alive at zero after its last member leaves.
    """
    rostercensus.assert_waiver_is_the_named_set(census(), waived=NAMED_BY_AN_EXAMPLE)


def test_the_census_still_convicts_a_planted_supersession() -> None:
    """THE CONTROL: plant both shapes and call the REAL grader, so a silent census cannot pass."""
    rostercensus.assert_the_grader_still_convicts(moves_side=MOVES)
