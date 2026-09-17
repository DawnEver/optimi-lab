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
from lab_commons.dev.installdoor import (
    Delivery,
    RevertingDoorsError,
    assert_doors_deliver,
    floating_requirements,
    reverting,
)

_ROOT: Final = Path(__file__).resolve().parents[2]

#: Every file in this tree whose content moves an environment, as a NAMED SET. `README.md` is a door
#: with a person in the middle: it tells a human what to type. A count could not say which one came
#: off the list, and in the sibling repo the door that carried the defect was one nobody had listed.
_DOORS: Final[tuple[str, ...]] = (
    'Makefile',
    'README.md',
    '.pre-commit-config.yaml',
    '.github/workflows/ci.yml',
    'scripts/dep.py',
)

#: MEASURED 2026-09-17: 22 installer-capable commands across those five files. The floor sits under
#: it, because "no reverting door" over a set that was never read is a vacuous green.
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
