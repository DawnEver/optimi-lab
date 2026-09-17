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
   optimisation fact would carry that fact into a shared package.
4. DENSITY -- `STAYS` and `SPLITS` answer to the same bar, with the planted controls that prove the
   measurement can still red at all.
5. THE PYTHON-ONLY LIMIT HAS A CEILING -- section 4 parses an AST, so a file it cannot parse would
   be unmeasured. This tree holds none today, and that is asserted rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import _placement
import pytest
from _placement import (
    BELOW_THE_BAR,
    MIN_REPO_DENSITY_PCT,
    MOVES,
    OWN_MECHANISM_CEILING,
    PLACEMENT,
    PLACEMENT_FLOOR,
    RUNNABLE_SUFFIXES,
    SPLITS,
    STAYS,
    Placement,
    measure_density,
    nouns_in,
    nouns_in_code,
    placed_files,
    repo_root,
    stays_rows,
)

_ROOT = repo_root()
_FILES = tuple(placed_files(_ROOT))


# --------------------------------------------------------------------------- 1. completeness


def test_the_scan_read_a_real_population() -> None:
    """Refuse a scan that read fewer files than this tree holds -- the FLOOR, and it comes first.

    Every assertion below is over `_FILES`; if that walk reached nothing, all of them would pass and
    the table would be describing an empty tree. Finding NOTHING is vacuous rather than green.
    """
    assert len(_FILES) >= PLACEMENT_FLOOR, (
        f'the placement scan read {len(_FILES)} runnable files, below its measured floor of '
        f'{PLACEMENT_FLOOR}. A walk that stopped early, a tree that moved, or a widened exclusion '
        f'all report what a fully classified tree reports. Fix the walk, or lower the floor IN THE '
        f'SAME EDIT that deletes the files it counted.'
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
    placement = PLACEMENT[rel]
    assert placement.side in {STAYS, MOVES, SPLITS}, rel
    assert len(placement.why) > 60, f'{rel}: a placement without a reason is a guess with a label'


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
    """Hold the other half of the floor: an empty tree must not read as a fully classified one."""
    assert len(tuple(placed_files(tmp_path))) < PLACEMENT_FLOOR


# --------------------------------------------------------------------------- 2. STAYS needs evidence


@pytest.mark.parametrize('rel', sorted(rel for rel, row in PLACEMENT.items() if row.side == STAYS))
def test_a_file_that_STAYS_names_something_this_repo_owns(rel: str) -> None:
    """Refuse a `STAYS` with no repo noun in it, because the cost of that guess is one-directional.

    The file never moves, and nobody afterwards can tell whether that was a decision.
    """
    assert nouns_in(rel, _ROOT), f"{rel} is classified as this repo's but names nothing it owns"


# --------------------------------------------------------------------------- 3. MOVES must survive it


@pytest.mark.parametrize('rel', sorted(rel for rel, row in PLACEMENT.items() if row.side == MOVES))
def test_a_file_that_MOVES_carries_no_repo_noun(rel: str) -> None:
    """Refuse a move that would carry an optimisation fact into a vendor-neutral package.

    THE DISCRIMINATING PROPERTY, and the one that fails on a mistake somebody would actually make:
    such a fact is invisible in the new home BECAUSE the default works.

    IT READS CODE, NOT PROSE, and `nouns_in_code` records the measurement that forced the
    distinction -- in the sibling repo, prose hits refused three correct rows.

    A hit is not an order to reclassify. It is a demand to either strip the noun (usually: take it
    as data) or move the row to `SPLITS` with the seam named.
    """
    carried = sorted(nouns_in_code(rel, _ROOT))
    assert not carried, (
        f'{rel} is declared vendor-neutral but names this repo: {carried}. Strip the noun (take it '
        f'as data), or reclassify as SPLITS with the seam named.'
    )


def test_property_3_reads_code_and_property_2_reads_prose() -> None:
    """Pin the asymmetry between the two readers, because it is what an editor would "tidy" away.

    Collapsing them either way breaks a real row: read prose in property 3 and a `MOVES` row whose
    docstring merely NAMES this repo is refused; read code only in property 2 and a file whose repo
    fact is genuinely explained in its docstring loses its evidence. This asserts BOTH directions on
    live rows rather than describing them.
    """
    prose_only = 'tests/architecture/test_the_declared_hooks_are_installed.py'
    assert nouns_in(prose_only, _ROOT), 'the prose reader stopped reading prose'
    assert not nouns_in_code(prose_only, _ROOT), 'the code reader started reading prose'

    in_code = 'tests/architecture/test_a_declared_set_refuses_a_stranger.py'
    assert nouns_in_code(in_code, _ROOT), 'the code reader stopped seeing a noun that is genuinely in code'


def test_the_MOVES_arm_is_not_empty() -> None:
    """Put a floor under section 3, because over zero `MOVES` rows it proves nothing.

    A vacuous green there reads exactly like a table whose every move survived the converse.
    """
    moves = sorted(rel for rel, row in PLACEMENT.items() if row.side == MOVES)
    assert moves, 'no MOVES row exists, so section 3 is measuring nothing'


# --------------------------------------------------------------------------- 4. density


@pytest.mark.parametrize('rel', stays_rows())
def test_a_file_that_STAYS_or_SPLITS_is_MOSTLY_this_repo(rel: str) -> None:
    """Refuse the middle case: general machinery with one repo constant welded into it.

    PROPERTY 4, and the one that fails on the mistake property 2 lets through. Property 2 is a HIT
    test and a hit is one line. This asks how much NON-repo code surrounds it: a claim is justified
    two ways and no third -- the file is OURS (dense enough) or it is a BINDER (small enough that
    there is no room in it for a mechanism somebody else should own).
    """
    density = measure_density((_ROOT / rel).read_text(encoding='utf-8', errors='replace'))
    assert density.justified, (
        f'{rel} claims {PLACEMENT[rel].side.upper()} on {density.repo} of {density.own} own code '
        f"lines ({density.percent:.2f}%). That is neither dense enough to be this repo's "
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

    FLOOR FIRST: over zero `SPLITS` rows this assertion is vacuous.
    """
    splits = sorted(rel for rel, row in PLACEMENT.items() if row.side == SPLITS)
    assert splits, 'no SPLITS row exists, so this ceiling is measuring nothing'
    measured = {param.values[0] for param in stays_rows()}
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
# THESE PLANT THE SHAPE AND CALL THE REAL `measure_density`. A control that re-derived the traversal
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
    stopped filtering.
    """
    density = measure_density(_A_MECHANISM_WITH_A_NOUN_WELDED_IN)
    assert density.own > OWN_MECHANISM_CEILING
    assert not density.justified, f'a {density.own}-line mechanism with one constant was admitted'


def test_prose_is_not_evidence_and_the_docstring_proves_it() -> None:
    """Count zero for a docstring naming four repo nouns, or section 4 is property 2 again."""
    prose_only = '"""optimi_lab sampler regressor pareto surrogate."""\n' + ''.join(f'V{i} = {i}\n' for i in range(70))
    assert measure_density(prose_only).repo == 0


def test_a_comment_is_not_evidence_either() -> None:
    """Count zero for comments too -- a `# optimi_lab` above every line would buy any density."""
    commented = ''.join(f'# optimi_lab owns this\nV{i} = {i}\n' for i in range(70))
    assert measure_density(commented).repo == 0


def test_the_density_guard_ADMITS_a_binder() -> None:
    """Admit the shape a FINISHED migration leaves behind, which is not decoration.

    A guard that rejected binders too would force `scripts/dep.py` and `scripts/deny_rules.py` to be
    reclassified as a mechanism they are not -- the opposite mistake, and equally invisible.
    """
    assert measure_density(_A_BINDER).justified


def test_delegation_is_excluded_from_a_files_OWN_mechanism() -> None:
    """Separate a binder from a mechanism: a line CALLING a `lab_commons` name is delegation."""
    assert measure_density(_A_BINDER).own <= 4, measure_density(_A_BINDER)


def test_the_density_guard_ADMITS_a_large_file_that_is_genuinely_ours() -> None:
    """Admit length when the density is there, so section 4 does not simply punish size."""
    ours = ''.join(f'OPTIMI_{i} = "sampler"\nV{i} = {i}\n' for i in range(60))
    density = measure_density(ours)
    assert density.own > OWN_MECHANISM_CEILING
    assert density.justified


# --------------------------------------------------------------------------- 5. the parse ceiling


def test_the_scanned_trees_hold_no_unmeasurable_file() -> None:
    """Refuse a file section 4 cannot parse, because an unmeasurable file is an unmeasured one.

    `measure_density` reads an AST, so a shell script in a scanned tree would carry a placement that
    no measurement ever tested. wdg-lab's roster has two and names them in a `SHELL_ROWS` set; this
    tree has NONE, and that is asserted rather than assumed -- the day a `.sh` file lands here, this
    reds and forces the same named-set decision rather than letting it in silently.
    """
    unmeasurable = sorted(rel for rel in _FILES if not rel.endswith('.py'))
    assert not unmeasurable, (
        f'{unmeasurable} are in a scanned tree and cannot be parsed by `measure_density`, so their '
        f'rows would be exempt from section 4. Declare them in a named set with the exemption '
        f'stated, as wdg-lab does, or give section 4 a reader for them.'
    )


def test_the_runnable_suffixes_can_still_SEE_a_file_this_tree_lacks() -> None:
    """Pin both directions of the suffix set, because narrowing it would GREEN the test above.

    If `RUNNABLE_SUFFIXES` dropped `.sh`, a shell script arriving in `scripts/` would simply vanish
    from `_FILES`: section 1 would not demand a row for it and section 5 would find nothing to
    refuse. The scan must be able to SEE the thing it reports the absence of.
    """
    assert '.sh' in RUNNABLE_SUFFIXES, (
        'the scan can no longer see a shell script, so "this tree holds none" became unfalsifiable'
    )
    assert set(RUNNABLE_SUFFIXES) == {'.py', '.sh'}
