"""THE MIGRATION BOUNDARY IS DECLARED -- and a new dev file cannot be silently unplaced.

The manifest, its reasons and the measurement machinery live in `_placement.py`; see that module's
docstring for why the table exists and what the density measurement cannot see. This file holds
only the checks: a table edited through the file that reads it drifts away from what it describes,
because the edit that makes a red go green and the edit that states what the repo believes end up
being the same keystroke.

FIVE SECTIONS, and each one fails on a mistake somebody would actually make:

1. COMPLETENESS, WITH A FLOOR -- every runnable file carries a row, no row names a file that is
   gone, and every row states a reason. The FLOOR is what stops a walk that reached nothing from
   reporting exactly what a fully classified tree reports.
2. `STAYS` NEEDS EVIDENCE -- a file said to be ours that names nothing we own is a guess with a
   label on it.
3. `MOVES` MUST SURVIVE THE CONVERSE -- a file said to be vendor-neutral whose CODE names an
   optimisation fact would carry that fact into a shared package. THE POPULATION IS EMPTY as of
   2026-09-18 and is pinned as a NAMED SET rather than held above a floor: this roster's one and
   only `MOVES` row was the rules-page ratchet, whose destination did not exist yet, and that
   adoption is now executed. A floor there would have made FINISHING the migration a red.
4. DENSITY -- `STAYS` and `SPLITS` answer to the same bar, with the planted controls that prove the
   measurement can still red at all.
5. THE PYTHON-ONLY LIMIT HAS A CEILING -- section 4 parses an AST, so a file it cannot parse would
   be unmeasured. The shell rows are a NAMED SET, EMPTY today, and `stays_rows` excludes it: before
   that, a `.sh` file arriving here would have been handed to `ast.parse` and section 4 would have
   ERRORED rather than failed.
"""

from __future__ import annotations

from pathlib import Path

import _placement
import pytest
from _placement import (
    BELOW_THE_BAR,
    CEILING_ADMITS,
    CEILING_REFUSES,
    MIN_REPO_DENSITY_PCT,
    MINIMUM_ADMITS,
    MINIMUM_REFUSES,
    OWN_MECHANISM_CEILING,
    PLACEMENT,
    PLACEMENT_FLOOR,
    PLACEMENT_HEADROOM,
    RUNNABLE_SUFFIXES,
    SHELL_ROWS,
    is_justified,
    measure,
    nouns_in_code,
    nouns_in_prose,
    placed_files,
    repo_root,
    stays_rows,
)
from lab_commons.dev import floors
from lab_commons.dev.famtests import density, placement
from lab_commons.dev.famtests.placement import MOVES, SPLITS, STAYS, Placement

_ROOT = repo_root()
_FILES = tuple(placed_files(_ROOT))


# --------------------------------------------------------------------------- 1. completeness


def test_the_roster_and_its_tree_agree_with_both_sides_of_the_floor() -> None:
    """COMPLETENESS, BOTH DIRECTIONS, over a walk bound to BOTH sides of its floor first.

    THE FLOOR COMES FIRST. Every assertion in this file is over `_FILES`; if that walk reached
    nothing, all of them would pass and the table would be describing an empty tree. THE HIGH SIDE
    COMES WITH IT, and it is the half no roster in this family ever wrote: a floor the tree has
    outgrown refuses only a total collapse.

    The two per-row arms below stay parametrized so an offender fails under its OWN name -- this one
    is the aggregate the kit ships, and it catches both directions at once.
    """
    placement.assert_every_file_is_placed(
        _FILES,
        declared=PLACEMENT,
        floor=PLACEMENT_FLOOR,
        headroom=PLACEMENT_HEADROOM,
        what='optimi-lab placement',
    )


@pytest.mark.parametrize('rel', _FILES)
def test_every_runnable_file_is_classified(rel: str) -> None:
    """Force a decision about a NEW dev file, rather than describing the tree on one past day.

    Without this the table is a snapshot, which is how a migration loses files and then reads as
    complete. PARAMETRIZED, and not only so each file fails under its own name: an aggregate
    `assert not missing` over a loop files one failure and names no file, so an unclassified one
    hides until the whole tree is classified.
    """
    assert rel in PLACEMENT, f'{rel} is unclassified; add a row to PLACEMENT saying which side and why'


@pytest.mark.parametrize('rel', sorted(PLACEMENT))
def test_the_table_names_no_file_that_is_gone(rel: str) -> None:
    """Refuse the converse, which is the half that rots quietly.

    A deleted file leaves a row asserting a placement for something nobody can look at, and that
    reads as coverage.
    """
    assert rel in set(_FILES), f'PLACEMENT names {rel}, which is not a runnable file in this tree'


@pytest.mark.parametrize('rel', sorted(PLACEMENT))
def test_every_row_states_a_reason(rel: str) -> None:
    """Refuse a label with no fact behind it -- a row without a reason outlives the guesser."""
    row = PLACEMENT[rel]
    assert row.side in {STAYS, MOVES, SPLITS}, rel
    assert len(row.why) > 60, f'{rel}: a placement without a reason is a guess with a label'


def test_the_scan_places_a_planted_file(tmp_path: Path) -> None:
    """Drive the REAL walk over a planted tree: one file of each runnable suffix, plus the two
    exclusions that would otherwise read as unplaced files -- build output and prose.
    """
    scripts = tmp_path / 'scripts'
    (scripts / '__pycache__').mkdir(parents=True)
    (tmp_path / 'tests' / 'architecture').mkdir(parents=True)
    (scripts / 'planted.py').write_text('x = 1\n', encoding='utf-8')
    (scripts / 'planted.sh').write_text('echo hi\n', encoding='utf-8')
    (scripts / 'README.md').write_text('prose\n', encoding='utf-8')
    (scripts / '__pycache__' / 'planted.cpython-313.pyc').write_bytes(b'\x00')
    (tmp_path / 'tests' / 'architecture' / 'test_planted.py').write_text('def test_x(): ...\n', encoding='utf-8')

    assert set(placed_files(tmp_path)) == {
        'scripts/planted.py',
        'scripts/planted.sh',
        'tests/architecture/test_planted.py',
    }


def test_the_floor_refuses_an_unread_tree(tmp_path: Path) -> None:
    """THE PLANTED CONTROL FOR THE FLOOR: an empty tree must not read as a fully classified one."""
    assert len(placed_files(tmp_path)) < PLACEMENT_FLOOR
    with pytest.raises(floors.FloorUnmet):
        placement.assert_every_file_is_placed(
            placed_files(tmp_path),
            declared=PLACEMENT,
            floor=PLACEMENT_FLOOR,
            headroom=PLACEMENT_HEADROOM,
            what='optimi-lab placement',
        )


def test_no_recorded_shortfall_names_a_file_the_manifest_no_longer_places() -> None:
    """A DEBT ROW WHOSE FILE IS GONE DESCRIBES NOTHING -- the half no roster in this family wrote.

    It still reads as a live shortfall, and its strict xfail can never fire: the parametrize that
    would have run it no longer holds the row, so the one arm that would have caught it is the arm
    the deletion removed. `test_every_recorded_shortfall_names_a_real_row` below asks a NEIGHBOURING
    question -- that the row is a STAYS or SPLITS -- and is parametrized over `BELOW_THE_BAR`, so it
    too goes quiet rather than red when the manifest drops the key. This is the aggregate that does
    not.
    """
    placement.assert_no_stale_debt(declared=PLACEMENT, debt=BELOW_THE_BAR, what='optimi-lab placement')


def test_each_bar_sits_inside_the_interval_its_own_two_measured_rows_draw() -> None:
    """THE BARS ARE BOUNDED BY EVIDENCE, and until now that bracket lived in a COMMENT.

    Neither bar is a family constant and the kit deliberately ships neither as a value: motronics'
    `scripts/` roster bounds the ceiling at 40 over an interval excluding 50, while this tree's
    excludes 40. What the kit ships is the ARM, which takes this repo's value together with the two
    real files that bracket it -- and a comment is exactly where a bar quietly widens to absorb the
    row that reds. The density minimum's bracket here is the tightest in the family, at six
    hundredths of a percent.
    """
    placement.assert_ceiling_is_bounded(
        OWN_MECHANISM_CEILING, admits=CEILING_ADMITS, refuses=CEILING_REFUSES, what='optimi-lab own-mechanism'
    )
    placement.assert_minimum_is_bounded(
        MIN_REPO_DENSITY_PCT, admits=MINIMUM_ADMITS, refuses=MINIMUM_REFUSES, what='optimi-lab repo-density'
    )


# --------------------------------------------------------------------------- 2. STAYS needs evidence


@pytest.mark.parametrize('rel', sorted(rel for rel, row in PLACEMENT.items() if row.side == STAYS))
def test_a_file_that_STAYS_names_something_this_repo_owns(rel: str) -> None:
    """Refuse a `STAYS` with no repo noun in it, because the cost of that guess is one-directional.

    The file never moves, and nobody afterwards can tell whether that was a decision.
    """
    assert nouns_in_prose(rel, _ROOT), f"{rel} is classified as this repo's but names nothing it owns"


# --------------------------------------------------------------------------- 3. MOVES must survive it


def test_a_file_that_MOVES_carries_no_repo_noun() -> None:
    """Refuse a move that would carry an optimisation fact into a vendor-neutral package.

    THE DISCRIMINATING PROPERTY, and the one that fails on a mistake somebody would actually make:
    such a fact is invisible in the new home BECAUSE the default works.

    IT READS CODE, NOT PROSE, and `nouns_in_code` records the measurement that forced the
    distinction -- in the sibling repo, prose hits refused three correct rows.

    A hit is not an order to reclassify. It is a demand to either strip the noun (usually: take it
    as data) or move the row to `SPLITS` with the seam named.

    NOT PARAMETRIZED, and that is a consequence of the set below going EMPTY on 2026-09-18. A
    parametrize over zero rows is a pytest SKIP, and a skip is a selected test that reported
    nothing -- the verdict runner refuses one, and rightly: it reads exactly like a table whose
    every move survived the converse. Looping reports the same per-row facts and always RUNS.
    """
    carried = {
        rel: sorted(nouns_in_code(rel, _ROOT))
        for rel, row in sorted(PLACEMENT.items())
        if row.side == MOVES and nouns_in_code(rel, _ROOT)
    }
    assert not carried, (
        f'declared vendor-neutral but naming this repo: {carried}. Strip the noun (take it as '
        f'data), or reclassify as SPLITS with the seam named.'
    )


def test_property_3_reads_code_and_property_2_reads_prose() -> None:
    """Pin the asymmetry between the two readers, because it is what an editor would "tidy" away.

    Collapsing them either way breaks a real row: read prose in property 3 and a `MOVES` row whose
    docstring merely NAMES this repo is refused; read code only in property 2 and a file whose repo
    fact is genuinely explained in its docstring loses its evidence. This asserts BOTH directions on
    live rows rather than describing them.
    """
    prose_only = 'tests/architecture/test_the_declared_hooks_are_installed.py'
    assert nouns_in_prose(prose_only, _ROOT), 'the prose reader stopped reading prose'
    assert not nouns_in_code(prose_only, _ROOT), 'the code reader started reading prose'

    in_code = 'tests/architecture/test_a_declared_set_refuses_a_stranger.py'
    assert nouns_in_code(in_code, _ROOT), 'the code reader stopped seeing a noun that is genuinely in code'


#: THE `MOVES` POPULATION, AS A NAMED SET RATHER THAN A FLOOR, and it is EMPTY as of 2026-09-18.
#: This roster carried exactly one `MOVES` row for its whole life --
#: `test_the_rules_pages_are_a_ratchet.py`, a mechanism whose destination did not exist yet -- and
#: that adoption is now EXECUTED, so the row is a `SPLITS` and the migration this table declares has
#: nothing left to move. The floor that used to stand here asserted the population was non-empty,
#: which made FINISHING the migration a red: a floor whose only satisfying state is unfinished work
#: is the waiver-nothing-uses failure wearing the other face. A named set is two-sided instead -- a
#: `MOVES` row ARRIVING reds until it is declared here, and one DISAPPEARING reds too.
#:
#: THE ANTI-VACUITY JOB THE FLOOR WAS DOING IS ALREADY CARRIED, and by a stronger arm than a
#: population count: `test_property_3_reads_code_and_property_2_reads_prose` drives the REAL
#: converse reader in BOTH directions on two live files -- one that names this repo in code and must
#: be caught, one that names it only in prose and must not be. That holds at zero rows and at fifty,
#: which a floor never did: a table of rows that all pass is equally consistent with a reader
#: returning nothing.
DECLARED_MOVES: frozenset[str] = frozenset()


def test_the_MOVES_population_is_the_one_this_roster_declares() -> None:
    """Pin the named set in both directions; the anti-vacuity job moved to the planted control."""
    live = frozenset(rel for rel, row in PLACEMENT.items() if row.side == MOVES)
    assert live == DECLARED_MOVES, (
        f'the MOVES population moved: arrived {sorted(live - DECLARED_MOVES)}, gone '
        f'{sorted(DECLARED_MOVES - live)}. Declare it here in the edit that classifies it.'
    )


# --------------------------------------------------------------------------- 4. density


@pytest.mark.parametrize('rel', stays_rows())
def test_a_file_that_STAYS_or_SPLITS_is_MOSTLY_this_repo(rel: str) -> None:
    """Refuse the middle case: general machinery with one repo constant welded into it.

    PROPERTY 4, and the one that fails on the mistake property 2 lets through. Property 2 is a HIT
    test and a hit is one line. This asks how much NON-repo code surrounds it: a claim is justified
    two ways and no third -- the file is OURS (dense enough) or it is a BINDER (small enough that
    there is no room in it for a mechanism somebody else should own).
    """
    reading = measure((_ROOT / rel).read_text(encoding='utf-8', errors='replace'))
    assert is_justified(reading), (
        f'{rel} claims {PLACEMENT[rel].side.upper()} on {reading.hits} of {reading.own} own code '
        f"lines ({reading.percent:.2f}%). That is neither dense enough to be this repo's "
        f'(>= {MIN_REPO_DENSITY_PCT}%) nor small enough to be a binder '
        f'(<= {OWN_MECHANISM_CEILING} own lines). Thin it against `lab_commons.dev` until only the '
        f'repo fact is left, or record it in BELOW_THE_BAR with its measurement and the seam it '
        f'owes. Relabelling it SPLITS does NOT exempt it: SPLITS answers to this same bar.'
    )


@pytest.mark.parametrize('rel', sorted(BELOW_THE_BAR))
def test_every_recorded_shortfall_names_a_real_row(rel: str) -> None:
    """Refuse a shortfall entry that outlived its file, which reads as a backlog nobody can act on.

    The strict xfail above would ERROR rather than say why, so the diagnosis belongs here.
    """
    assert PLACEMENT.get(rel) and PLACEMENT[rel].side in (STAYS, SPLITS), (
        f'BELOW_THE_BAR names {rel}, which is neither a STAYS nor a SPLITS row. Delete the entry.'
    )


def test_every_SPLITS_row_is_MEASURED_by_this_section() -> None:
    """Hold the ceiling on the escape hatch: a `SPLITS` label buys a seam, never a gentler bar.

    `stays_rows` parametrizes `STAYS` and `SPLITS` over the SAME measurement, and one character
    narrowing that to `== STAYS` would silently delete the arm while every other test here stayed
    green, because a parametrize that loses cases reports no failure.

    THE FLOOR MOVED 2026-09-19, AND WHERE IT MOVED TO IS THE POINT. It used to read
    `assert splits` -- a demand that the live manifest hold at least one `SPLITS` row. On 2026-09-19
    `test_a_pin_is_a_named_set.py` adopted `famtests.countpins`, the last split was paid, and the
    manifest went to thirty `STAYS` and nothing else. A FINISHED MIGRATION IS THE SUCCESS CONDITION
    OF THIS TABLE, so a floor that reds on it is a floor demanding the work stay unfinished, and the
    only ways to satisfy it are to invent a row or to hold one open. Neither is a measurement.

    WHAT KEEPS THIS NON-VACUOUS INSTEAD, and it is strictly stronger than the old floor was: the
    subset assertion below is floored on `stays_rows()` itself -- the population that must never
    collapse -- and `test_the_SPLITS_arm_would_be_MISSED_if_it_were_dropped` PLANTS a manifest that
    is nothing but a split and drives the REAL `stays_rows`. The one-character narrowing to
    `== STAYS` reds there whether or not this tree currently holds a split, which the old floor could
    not claim: it went green the moment any split existed, regardless of the arm.
    """
    splits = sorted(rel for rel, row in PLACEMENT.items() if row.side == SPLITS)
    measured = {param.values[0] for param in stays_rows()}
    assert measured, (
        'section 4 parametrized NOTHING, so this ceiling and every density row under it are '
        'measuring an empty tree -- and a collapsed `stays_rows` and a fully classified manifest '
        'are indistinguishable from here unless this fires'
    )
    assert set(splits) <= measured, (
        f'{sorted(set(splits) - measured)} carry the SPLITS label but are not parametrized into '
        f'section 4. A label the author picks is not a measurement: an unfinished split is a '
        f'BELOW_THE_BAR row, never an exemption.'
    )


def test_the_SPLITS_arm_would_be_MISSED_if_it_were_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plant a manifest that is nothing BUT a split, and call the REAL `stays_rows` on it.

    The test above reads TODAY's manifest, so it would keep passing on a tree where every `SPLITS`
    row happened to have been retired -- and then the arm could be deleted under it.
    """
    planted = {'scripts/planted_split.py': Placement(SPLITS, 'planted: a seam nobody measured, stated at length')}
    monkeypatch.setattr(_placement, 'PLACEMENT', planted)
    monkeypatch.setattr(_placement, 'BELOW_THE_BAR', {})
    assert [param.values[0] for param in stays_rows()] == ['scripts/planted_split.py'], (
        'a manifest holding one SPLITS row produced no section-4 parameter: the SPLITS arm of '
        '`stays_rows` is gone, and every split in this tree is now unmeasured'
    )


# ------------------------------------------------- the planted controls for section 4
#
# THESE PLANT THE SHAPE AND CALL THE REAL `measure`. A control that re-derived the traversal
# could only agree with itself, and a green section 4 over a tree whose rows all happen to pass
# proves nothing about whether the measurement can red at all.

_A_MECHANISM_WITH_A_NOUN_WELDED_IN = (
    '"""A docstring that talks at length about optimi_lab, the sampler, the regressor and pareto.\n\n'
    'Prose is not code, and this paragraph is the whole of what property 2 can see.\n"""\n'
    'NAMESPACE = "optimi_lab"\n' + ''.join(f'VALUE_{i} = {i}\n' for i in range(70))
)

_A_BINDER = (
    '"""Binds one fact."""\n'
    'from lab_commons.dev.dep import Port\n\n'
    'NAMESPACE = "optimi_lab"\n\n\n'
    'def take():\n    return Port(name=NAMESPACE)\n'
)


def test_the_density_guard_REJECTS_a_mechanism_with_one_repo_constant() -> None:
    """Refuse 71 own lines carrying one repo constant -- the DISCRIMINATING control.

    Property 2 is satisfied by that shape and this must not be. If this ever passes, section 4 has
    stopped filtering. It stays local because it is the one control that reads THIS repo's ceiling.
    """
    reading = measure(_A_MECHANISM_WITH_A_NOUN_WELDED_IN)
    assert reading.own > OWN_MECHANISM_CEILING
    assert not is_justified(reading), f'a {reading.own}-line mechanism with one constant was admitted'


def test_the_meter_still_convicts_prose_a_hit_and_a_delegation() -> None:
    """THE FAMILY CONTROL, driving the INSTALLED meter over three planted shapes in OUR OWN WORDS.

    This replaces four hand-written controls that said the same three things: prose buys no density,
    a code line naming a repo noun is a hit, and a line delegating to `lab_commons` is wiring rather
    than this file's mechanism. The last is the one with a measured cost -- counting delegation as
    own mechanism scores a COMPLETED migration as new local code, read on a real re-point upstream
    as own 39 -> 40 where the honest answer is 39 -> 31.

    The witness and the home are THIS repo's: a kit inventing either would drive the meter over its
    own idea of a noun instead of ours.
    """
    density.assert_the_meter_still_convicts(noun_witness='sampler', delegation_home='lab_commons')


def test_the_density_guard_ADMITS_a_large_file_that_is_genuinely_ours() -> None:
    """Admit length when the density is there, so section 4 does not simply punish size."""
    ours = ''.join(f'OPTIMI_{i} = "sampler"\nV{i} = {i}\n' for i in range(60))
    reading = measure(ours)
    assert reading.own > OWN_MECHANISM_CEILING
    assert is_justified(reading)


# --------------------------------------------------------------------------- 5. the parse ceiling


def test_the_shell_rows_are_exactly_the_named_set() -> None:
    """SECTION 4 IS PYTHON-ONLY, SO THE EXEMPTION IS NAMED AND TWO-SIDED -- and it is EMPTY today.

    THE DEFECT THIS CLOSES was live rather than hypothetical. This roster declares `.sh` RUNNABLE and
    holds none, and the arm it used to have merely asserted the absence. `stays_rows` had no shell
    exclusion, so the day a shell script landed it would have been parametrized into section 4,
    handed to `ast.parse` and the guard would have ERRORED rather than failed -- naming no file and
    reading as a broken test rather than as an unmeasured row. wdg-lab's `SHELL_ROWS` is the shape
    that answers it and this is now the same shape: `stays_rows` excludes the named set, and the set
    is compared by EQUALITY, so an arriving `.sh` file is a decision somebody makes here.

    EMPTY IS LEGAL AND IS THE STATE TO BE IN. The anti-vacuity job is carried by the suffix arm
    below, which pins that the scan can still SEE a shell script at all.
    """
    found = frozenset(rel for rel in _FILES if rel.endswith('.sh'))
    assert found == SHELL_ROWS, (
        f'shell files not named in SHELL_ROWS: {sorted(found - SHELL_ROWS)}; named but absent: '
        f'{sorted(SHELL_ROWS - found)}. Section 4 cannot measure a shell script, so each one is '
        f'declared here by name with its exemption stated -- adding one is an edit somebody means.'
    )


def test_a_shell_row_is_excluded_from_section_4_by_NAME_rather_than_by_crashing_it() -> None:
    """THE PLANTED CONTROL FOR THE EXEMPTION, because `SHELL_ROWS` is empty and proves nothing alone.

    An empty named set cannot show that the exclusion WORKS, and an exclusion that does not work is
    exactly the `ast.parse` error this arm exists to prevent. So plant a manifest holding one shell
    row and call the REAL `stays_rows`: the row must not come back as a section-4 parameter.
    """
    planted = {
        'scripts/planted.sh': Placement(STAYS, 'planted: a shell row no AST reader can measure, stated at length')
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_placement, 'PLACEMENT', planted)
        patch.setattr(_placement, 'BELOW_THE_BAR', {})
        patch.setattr(_placement, 'SHELL_ROWS', frozenset(planted))
        assert [param.values[0] for param in stays_rows()] == [], (
            'a shell row reached section 4, where `measure` will hand it to `ast.parse` and the '
            'guard will ERROR rather than name the file'
        )


def test_the_runnable_suffixes_can_still_SEE_a_file_this_tree_lacks() -> None:
    """Pin both directions of the suffix set, because narrowing it would GREEN the arm above.

    If `RUNNABLE_SUFFIXES` dropped `.sh`, a shell script arriving in `scripts/` would simply vanish
    from `_FILES`: section 1 would not demand a row for it and the named set above would find nothing
    to compare. The scan must be able to SEE the thing it reports the absence of.
    """
    assert '.sh' in RUNNABLE_SUFFIXES, (
        'the scan can no longer see a shell script, so "this tree holds none" became unfalsifiable'
    )
    assert set(RUNNABLE_SUFFIXES) == {'.py', '.sh'}
