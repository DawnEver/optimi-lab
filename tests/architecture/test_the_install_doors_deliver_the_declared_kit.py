"""Install doors either resolve floating requirements or reproduce the tracked lock.

The frozen developer installs are LOCKED, not fresh resolutions. The shared
reader requires a tracked, parseable lock before that classification is allowed.
The complete door census and planted untracked-lock control remain active.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from dev_lab.dev.doorcensus import DoorRow, assert_census, door_text
from dev_lab.dev.installdoor import (
    Delivery,
    UnlockedDoorsError,
    assert_doors_deliver,
    floating_requirements,
    unlocked,
)
from dev_lab.dev.synccensus import pruning_sites

_ROOT: Final = Path(__file__).resolve().parents[2]

#: The name this repo goes by in the census API.
_HERE: Final = 'optimi-lab'

#: EVERY FILE IN THIS TREE WHOSE CONTENT MOVES AN ENVIRONMENT, one row each, with what it measures.
#: ``commands`` and ``deliveries`` are compared by EQUALITY, never as a floor: a floor is satisfied by
#: every shorter declaration. These rows used to live in lab-commons under a neutral name; a fact
#: about this tree belongs in this tree. RE-MEASURED 2026-10-02.
DOOR_ROWS: Final[tuple[DoorRow, ...]] = (
    DoorRow(
        repo=_HERE,
        path='Makefile',
        commands=15,
        deliveries=frozenset({'INERT', 'RESOLVES', 'LOCKED'}),
        why=(
            '16 -> 15 ON 2026-10-18: `install-dev` lost `python -m pre-commit autoupdate`, which '
            'updated a configuration the retired pre-commit framework no longer reads; deliveries '
            'unchanged. Measured correct on 2026-09-17 and changed as a result of nothing, which is a result '
            'rather than a skip: this repo carried the hazard fully loaded -- a floating kit '
            'requirement and an untracked lock -- and simply had no door that fired it. Every install '
            'here is `uv pip install`, which re-resolves; the row keeps that true.'
        ),
    ),
    DoorRow(
        repo=_HERE,
        path='README.md',
        commands=6,
        deliveries=frozenset({'INERT', 'RESOLVES', 'LOCKED'}),
        why=(
            'A DOOR WITH A PERSON IN THE MIDDLE. What a README tells a human to type moves an '
            'environment exactly as a Makefile target does, and it is the door with no CI and no '
            'hook to catch it drifting away from the Makefile beside it.'
        ),
    ),
    DoorRow(
        repo=_HERE,
        path='.github/workflows/ci.yml',
        commands=0,
        deliveries=frozenset(),
        why=(
            'Zero because it is a thin caller of lab-commons` reusable `python-verify.yml`, so this '
            "repo's CI install door physically lives in another repo's tree, where the kit judges it "
            'against the names of the repos that run it. Judged here the door is invisible; the row '
            'is what says that was looked at rather than missed.'
        ),
    ),
    DoorRow(
        repo=_HERE,
        path='scripts/dep.py',
        commands=0,
        deliveries=frozenset(),
        why=(
            'Zero because this script composes its argv in Python through `dev_lab.dev.dep`, '
            'which spells `sys.executable -m pip install` -- correct by construction, and correct '
            'for a second reason, since pip re-clones a direct URL rather than treating it as '
            'satisfied. Declared as a door with a zero reading rather than left off the list.'
        ),
    ),
)

#: The door set every other arm reads, DERIVED from the rows so the paths are stated once.
_DOORS: Final[tuple[str, ...]] = tuple(row.path for row in DOOR_ROWS)

#: MEASURED 2026-09-17 at 22 installer-capable commands, 21 across these four files since 2026-10-18;
#: the per-file counts are held by EQUALITY in `DOOR_ROWS`. The floor sits under it, because "no
#: unlocked door" over a set that was never read is a vacuous green.
_DOOR_FLOOR: Final = 15


def test_this_repo_declares_the_kit_as_a_floating_requirement() -> None:
    """The rule's subject, asserted rather than assumed: with no floating requirement it is vacuous."""
    assert floating_requirements(_ROOT / 'pyproject.toml') == ('dev-lab', 'lab-commons', 'numerics-lab')


def test_every_install_door_already_delivers_the_declared_kit() -> None:
    """THE GUARD, and today it passes on a tree that needed no repair -- which is a result, not a skip."""
    names = floating_requirements(_ROOT / 'pyproject.toml')
    doors = assert_doors_deliver(_ROOT, list(_DOORS), names, floor=_DOOR_FLOOR)
    assert unlocked(doors) == ()
    assert Delivery.RESOLVES in {door.delivery for door in doors}, (
        'every door here classified INERT, so this scan is no longer watching an install path at all'
    )


def test_the_guard_fires_on_a_lock_consuming_target_planted_in_this_makefile(tmp_path: Path) -> None:
    """THE PLANTED CONTROL. A green scan proves nothing unless the same guard can refuse this tree.

    The plant is the one-line change somebody would actually make -- `uv sync` instead of
    `uv pip install -e .` -- applied to a copy of THIS repo's Makefile and pushed through the REAL
    guard, which then names the file and the line.
    """
    text = (_ROOT / 'Makefile').read_text(encoding='utf-8').replace('uv pip install -e "."', 'uv sync')
    (tmp_path / 'Makefile').write_text(text, encoding='utf-8')
    with pytest.raises(UnlockedDoorsError, match=r'Makefile:\d+'):
        assert_doors_deliver(tmp_path, ['Makefile'], ('lab-commons',), floor=1)


def test_every_declared_door_still_measures_what_its_row_records() -> None:
    """THE CENSUS, through the kit's `assert_census`, over this checkout's WORKING TREE.

    Equality per file in both directions.
    """
    seen = assert_census(
        _ROOT.parent, {_HERE: _ROOT.name}, DOOR_ROWS, repo_floor=1, door_floor=len(DOOR_ROWS), here=_HERE
    )
    assert set(seen) == {_HERE}


def test_the_pruning_doors_are_exactly_the_declared_frozen_group_installs() -> None:
    """Pin every pruning door by its command; a new or weakened installation must be declared."""
    texts = {(_HERE, row.path): door_text(_ROOT, row.path, at_head=False) or '' for row in DOOR_ROWS}
    assert pruning_sites(texts) == (
        (_HERE, 'Makefile', 'uv sync --frozen --group dev'),
        (_HERE, 'README.md', 'uv sync --frozen --group dev # for developer'),
    )
