"""INSTALL-DOOR-DELIVERS-THE-DECLARATION, and this repo's answer is that it was ALREADY CORRECT.

THE RULE'S SUBJECT IS HERE. `pyproject.toml` declares
``lab-commons[dev] @ git+https://github.com/DawnEver/lab-commons.git`` with no ref and `uv.lock` is
not tracked, so the requirement says "whatever that URL holds now" -- and this checkout's untracked
lock pins `0.2.2.dev26+gba3bf6de`, sixteen commits behind the declared build measured on 2026-09-17.
The hazard is fully loaded; what is missing is a door that fires it.

MEASURED 2026-09-17, AND NOTHING WAS CHANGED AS A RESULT. All 22 installer-capable commands across
the files below are `uv pip install` / `pip install` / `python -m`, and with uv 0.12.5 a
`uv pip install` re-resolves a bare git URL every time -- measured directly, transitively, and over
an already-installed older build. Not one command here consults the lock, so not one can serve it.
`scripts/dep.py` contributes no line to this scan because it builds its argv in Python through
`lab_commons.dev.dep`, which spells `sys.executable -m pip install`: correct by construction, and
correct for a second reason (pip re-clones a direct URL rather than treating it as satisfied).

SO THIS FILE EXISTS TO KEEP THAT TRUE RATHER THAN TO REPAIR ANYTHING, and the difference matters
because the sibling repo that DID revert did it through a git hook nobody had listed as a door. The
day a `uv sync` or a bare `uv run` is added to this Makefile or to a hook here, this reds.

THE DOOR SET IS OPTIMI-LAB'S OWN AND THAT IS WHY THIS FILE STAYS HERE, which `_placement` records as
the deciding fact. Which files in a tree can move an environment is not portable: this repo installs
through a `.github/workflows/ci.yml` the sibling lab does not have, and the sibling installs through
a `generate-changelog` git hook and an update script that optimi-lab does not have. The MECHANISM
around the set -- `scan_doors`, `floating_requirements`, `reverting`, `assert_doors_deliver` -- is
entirely `lab_commons.dev.installdoor`, so what is genuinely this package's is the five paths in
`_DOORS` and the floor under them, and that is the whole of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from lab_commons.dev.doorcensus import DoorRow, assert_census, door_text
from lab_commons.dev.installdoor import (
    Delivery,
    RevertingDoorsError,
    assert_doors_deliver,
    floating_requirements,
    reverting,
)
from lab_commons.dev.synccensus import pruning_sites

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
        commands=16,
        deliveries=frozenset({'INERT', 'RESOLVES'}),
        why=(
            'Measured correct on 2026-09-17 and changed as a result of nothing, which is a result '
            'rather than a skip: this repo carried the hazard fully loaded -- a floating kit '
            'requirement and an untracked lock -- and simply had no door that fired it. Every install '
            'here is `uv pip install`, which re-resolves; the row keeps that true.'
        ),
    ),
    DoorRow(
        repo=_HERE,
        path='README.md',
        commands=6,
        deliveries=frozenset({'INERT', 'RESOLVES'}),
        why=(
            'A DOOR WITH A PERSON IN THE MIDDLE. What a README tells a human to type moves an '
            'environment exactly as a Makefile target does, and it is the door with no CI and no '
            'hook to catch it drifting away from the Makefile beside it.'
        ),
    ),
    DoorRow(
        repo=_HERE,
        path='.pre-commit-config.yaml',
        commands=0,
        deliveries=frozenset(),
        why=(
            'ZERO, AND THIS IS THE SIBLING OF THE FILE THAT CARRIED THE DEFECT: the sibling lab '
            'reverted the kit on every push through a hook entry in a file of this exact name, and '
            'this repo has no such entry. The day a `uv run` hook is added here, this count moves '
            'off zero and the census says so before the first push does.'
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
            'Zero because this script composes its argv in Python through `lab_commons.dev.dep`, '
            'which spells `sys.executable -m pip install` -- correct by construction, and correct '
            'for a second reason, since pip re-clones a direct URL rather than treating it as '
            'satisfied. Declared as a door with a zero reading rather than left off the list.'
        ),
    ),
)

#: The door set every other arm reads, DERIVED from the rows so the paths are stated once.
_DOORS: Final[tuple[str, ...]] = tuple(row.path for row in DOOR_ROWS)

#: MEASURED 2026-09-17 at 22 installer-capable commands across those five files; the per-file counts
#: are held by EQUALITY in `DOOR_ROWS`. The floor sits under it, because "no reverting door" over a
#: set that was never read is a vacuous green.
_DOOR_FLOOR: Final = 15


def test_this_repo_declares_the_kit_as_a_floating_requirement() -> None:
    """The rule's subject, asserted rather than assumed: with no floating requirement it is vacuous."""
    assert floating_requirements(_ROOT / 'pyproject.toml') == ('lab-commons',)


def test_every_install_door_already_delivers_the_declared_kit() -> None:
    """THE GUARD, and today it passes on a tree that needed no repair -- which is a result, not a skip."""
    names = floating_requirements(_ROOT / 'pyproject.toml')
    doors = assert_doors_deliver(_ROOT, list(_DOORS), names, floor=_DOOR_FLOOR)
    assert reverting(doors) == ()
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
    with pytest.raises(RevertingDoorsError, match=r'Makefile:\d+'):
        assert_doors_deliver(tmp_path, ['Makefile'], ('lab-commons',), floor=1)


def test_every_declared_door_still_measures_what_its_row_records() -> None:
    """THE CENSUS, through the kit's `assert_census`, over this checkout's WORKING TREE.

    Equality per file in both directions, plus the kit's refusal of a tracked ``uv.lock`` -- the
    condition every lock-consuming door would rest on.
    """
    seen = assert_census(
        _ROOT.parent, {_HERE: _ROOT.name}, DOOR_ROWS, repo_floor=1, door_floor=len(DOOR_ROWS), here=_HERE
    )
    assert set(seen) == {_HERE}


def test_no_declared_door_holds_a_pruning_command() -> None:
    """THE SYNC CENSUS'S ANSWER FOR THIS REPO, and it is the empty set: no door here runs `uv sync`, so
    none chooses a population. The day a bare ``uv sync`` lands in one of them it reds here.
    """
    texts = {(_HERE, row.path): door_text(_ROOT, row.path, at_head=False) or '' for row in DOOR_ROWS}
    assert pruning_sites(texts) == ()
