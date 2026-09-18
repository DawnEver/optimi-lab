"""THE ROSTER IS RE-READ AGAINST THE KIT -- a declared side goes stale silently, and nothing noticed.

THE DEFECT, MEASURED BEFORE THIS FILE EXISTED. `_placement.py` decides a row's side by a DENSITY bar,
which answers *is this file mostly generic?*. That is a good question and a DIFFERENT one from *has
the family already expressed this?*, and nothing in this tree ever asked the second. So a row keeps
its side after its subject lands upstream, and the only thing that would have caught it is a human
re-reading every row. Across three tranches of the family's largest roster the error ran ONE WAY
every time -- 17 rows declared MOVES, 7 were real, 5 were splits and 5 were already in the kit.

THE FAMILY HALF IS ALREADY PUBLISHED and this file does not re-implement any of it:
`lab_commons.dev.supersede` holds the two detectors (a kit module NAMING a consumer path, and the
consumer IMPORTING that module), the seven grades, the ruler and both floors. What is here is the
half that package refuses to guess -- EVERY REPO-SHAPED FACT ARRIVES AS AN ARGUMENT WITH NO DEFAULT,
and this file is where optimi-lab's four answers are written down: which rows, which root, which
dotted package, and how little a census may read before finding nothing stops meaning anything.

THE READING THAT MADE THIS FILE WORTH WRITING, kit `0.2.2.dev74+ga177ba62f`, 27 rows, 46 kit modules:
six rows grade `NAMED_ONLY` -- the kit names the file and NOTHING corroborates it -- and all six are
FULLY ADOPTED, importing the exact module that names them. The cause is a mismatch inside the
instrument: `kit_modules` recurses into public sub-packages and publishes `famtests/allowguard.py`
under the bare name `allowguard`, while `imported_kit_modules(package='lab_commons.dev')` resolves
`from lab_commons.dev.famtests import allowguard` to the segment at depth 2, which is `famtests`.
The two halves spell the same module differently, so the IMPORT detector cannot fire on ANY
sub-package adoption. `NAMED_ONLY` was added upstream after a measured false positive; this is a
second, structural one, and it is routed back rather than absorbed. MEASURED IN BOTH LABS ON THE
SAME DAY against the same kit: six rows each, the same six modules, so it is a property of the
instrument rather than of either tree.

That is why `SUBPACKAGE_ADOPTED` is a NAMED SET and not a tolerance. A waiver with no ceiling is how
a real stale row hides among six false ones, so every member is REQUIRED to prove its own adoption
by the text of the file naming it, the set is compared by EQUALITY in both directions, and it is
refused if it is empty. It shrinks to nothing the day the kit resolves a sub-package import, and
this file reds when it does -- a ratchet has two sides.

WHAT THIS DOES NOT PROVE. The census is blind to a LIVE FORK that imports nothing and that no kit
docstring names; `supersede` says so itself and calls overlap a ruler rather than a detector. Green
here means no row is stale BY THE TWO DETECTORS, never that the roster is right.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Final

import pytest
from _placement import MOVES, PLACEMENT, PLACEMENT_FLOOR
from lab_commons.dev import supersede

#: This checkout: `tests/architecture/<this file>` sits two directories below it.
ROOT: Final = Path(__file__).resolve().parents[2]

#: THE KIT DIRECTORY'S OWN DOTTED PATH, and the depth it implies is the whole reason it is spelled
#: in full. One level short resolves every `lab_commons.dev.x` import to the token `dev`, which
#: matches no kit module, so the IMPORT detector silently never fires and every row reads UNTOUCHED
#: -- which is what a fully forked tree reports. `take_census` refuses that rather than returning it.
KIT_PACKAGE: Final = 'lab_commons.dev'

#: THE ROW FLOOR IS THIS REPO'S, and it is the one `_placement` already measured for its own
#: completeness arm: re-deriving a second number here would be two answers to one question.
ROW_FLOOR: Final = PLACEMENT_FLOOR

#: THE KIT FLOOR, measured 2026-09-18 against `0.2.2.dev74+ga177ba62f`: `kit_modules` reads 46
#: modules. The floor sits well below that with room for a module being renamed or withdrawn, and
#: far above the zero a mis-resolved package returns -- against which every row grades UNTOUCHED,
#: the answer a finished migration gives.
KIT_MODULE_FLOOR: Final = 35

#: THE SIX ROWS THE IMPORT DETECTOR CANNOT SEE, each with the sub-package module it adopted. NOT a
#: tolerance and not a count: an integer here could not say WHICH row went quiet, and this set is
#: the only thing standing between a structurally invisible adoption and a genuinely stale row.
#: Every member is proved below by reading the file, so a row cannot be parked here to silence it.
SUBPACKAGE_ADOPTED: Final[dict[str, str]] = {
    'tests/architecture/test_no_allow_entry_names_a_denied_shape.py': 'allowguard',
    'tests/architecture/test_the_agent_guard_is_live.py': 'agentguard',
    'tests/architecture/test_the_declared_hooks_are_installed.py': 'hookinstall',
    'tests/architecture/test_the_family_config_is_rendered.py': 'configrender',
    'tests/architecture/test_the_rules_pages_are_a_ratchet.py': 'rulespages',
    'tests/architecture/test_this_checkout_is_visible_on_origin.py': 'visibility',
}

#: The dotted path a sub-package adoption is written as, so the proof below reads what the file
#: actually says rather than trusting the table above.
_SUBPACKAGE_IMPORT: Final = 'from lab_commons.dev.famtests import '


def kit_directory() -> Path:
    """Where `lab_commons.dev` was installed, resolved by SPEC rather than by importing it.

    The kit is read as SOURCE and never executed: a census must be able to judge a kit version that
    is not the one running, and importing the thing under measurement is how a measurement acquires
    a side effect.
    """
    spec = importlib.util.find_spec(KIT_PACKAGE)
    assert spec is not None and spec.submodule_search_locations, (
        f'{KIT_PACKAGE} does not resolve to a package in this environment, so there is no kit to '
        f'read. Reinstall through `python scripts/dep.py` rather than lowering the floor below.'
    )
    return Path(spec.submodule_search_locations[0])


def census() -> supersede.Census:
    """The roster of this checkout, judged against the installed kit, with all four answers stated here."""
    return supersede.take_census(
        {path: placement.side for path, placement in PLACEMENT.items()},
        supersede.kit_modules(kit_directory()),
        root=ROOT,
        package=KIT_PACKAGE,
        row_floor=ROW_FLOOR,
        module_floor=KIT_MODULE_FLOOR,
    )


def test_the_census_reaches_the_whole_roster_and_a_kit_that_is_really_there() -> None:
    """THE FLOOR'S OWN ARM: every row is judged, and finding nothing must not read as green."""
    taken = census()
    assert taken.rows_read == len(PLACEMENT), (
        f'the census judged {taken.rows_read} of {len(PLACEMENT)} rows. A row the census never read '
        f'is a row whose side nothing re-checks, which is the state this file exists to end.'
    )
    assert taken.modules_read >= KIT_MODULE_FLOOR, (
        f'{taken.modules_read} kit modules read, below the {KIT_MODULE_FLOOR} floor. Fix the '
        f'installed kit, never the floor: every row reads UNTOUCHED against a kit nobody read.'
    )


def test_no_row_whose_subject_the_kit_already_holds_is_still_declared_moves() -> None:
    """THE CHECK. A `MOVES` row the kit already supersedes is the stale-roster defect, named.

    This roster holds no `MOVES` row today, and that is why the arm is worth having rather than a
    reason to leave it out: what it guards is the row a later tranche writes, and a guard added
    after the row it was meant to catch is a guard that has already failed once.
    """
    stale = {
        claim.path: f'{claim.grade} against `{claim.kit_module}`' for claim in census().flagged if claim.side == MOVES
    }
    assert not stale, (
        f'{stale} are declared MOVES and the kit already holds their subject. A move with an '
        f'occupant is a duplicate running today, not work outstanding -- adopt and delete, then '
        f're-price the row.'
    )


def test_the_rows_the_import_detector_cannot_see_are_the_named_set() -> None:
    """The waiver, compared by EQUALITY: an arrival is a real stale row, a departure is the fix."""
    found = frozenset(claim.path for claim in census().claims if claim.grade == supersede.NAMED_ONLY)
    assert found == frozenset(SUBPACKAGE_ADOPTED), (
        f'arrived: {sorted(found - frozenset(SUBPACKAGE_ADOPTED))}; adopted-and-still-waived: '
        f'{sorted(frozenset(SUBPACKAGE_ADOPTED) - found)}. A NAMED_ONLY row is one the kit names '
        f'and nothing corroborates -- an arrival here is either a real stale row or a seventh '
        f'sub-package adoption, and only reading the file says which.'
    )
    assert found, 'the waiver set is empty, which contradicts the measurement it was written from'


@pytest.mark.parametrize(('path', 'module'), sorted(SUBPACKAGE_ADOPTED.items()))
def test_every_waived_row_proves_its_own_adoption(path: str, module: str) -> None:
    """THE CEILING ON THE WAIVER. Membership is earned by the file's own text, never by the table."""
    source = (ROOT / path).read_text(encoding='utf-8')
    assert f'{_SUBPACKAGE_IMPORT}{module}' in source, (
        f'{path} is waived as having adopted `{module}` and does not import it. This set exists '
        f'because the detector is blind to a sub-package import, not because a row may opt out of '
        f'being judged.'
    )
    assert path in PLACEMENT, f'{path} is waived and carries no placement row, so nothing places it'


def test_the_census_still_convicts_a_planted_supersession() -> None:
    """THE CONTROL: plant both shapes and call the REAL grader, so a silent census cannot pass."""
    module = supersede.KitModule(
        name='checkout',
        claims=frozenset({'scripts/repo/worktree_debris.py'}),
        universe=frozenset({'registered_worktrees', 'orphan_directories', 'stale_branches'}),
    )
    whole = supersede.Row(
        path='scripts/repo/worktree_debris.py',
        side=MOVES,
        public=frozenset({'registered_worktrees', 'orphan_directories'}),
        imports=frozenset(),
    )
    part = supersede.Row(
        path='scripts/repo/worktree_debris.py',
        side=MOVES,
        public=frozenset({'registered_worktrees', 'a_local_answer'}),
        imports=frozenset({'checkout'}),
    )
    assert supersede.grade_row(whole, [module]).grade == supersede.SUPERSEDED, 'a whole fork went unconvicted'
    graded = supersede.grade_row(part, [module])
    assert graded.grade == supersede.PARTIAL, 'a partial fork went unconvicted'
    assert graded.remainder == ('a_local_answer',), (
        f'the grader reported {graded.remainder}; the remainder IS the local half of the split, so '
        f'a caller must get the seam rather than a percentage.'
    )
    assert graded.flagged and supersede.grade_row(part, []).grade == supersede.UNTOUCHED, (
        'with no kit module to judge against the same row must read UNTOUCHED -- otherwise the '
        'convictions above are a property of the row rather than of the kit.'
    )
