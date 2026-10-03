"""WORKTREES-STAY-INSIDE and PROJECT-FILES-HAVE-ONE-SOURCE, adopted from lab-commons 9ce57c3.

The mechanisms are the family's (`lab_commons.dev.famtests`); what is this repo's is the tree they
run over (the optimi_lab checkout), the deltas they judge it against, and the files it owns. Each test judges THIS checkout,
which is the point of moving these bodies out of the kit's suite: a test judges the repo it lives in.

THE SECTION DELTAS. The three ruff tables are copied from lab-commons
`tests/_famconfig_section_delta.py` at 9ce57c3 under `consumer-c` -- the 52-code shared tail and
nothing of this repo's own. The kit publishes no consumer delta for `[project]` or
`[tool.pytest.ini_options]`; those two were MEASURED here on 2026-10-03 with `inspect_section`:
`[project]` is the base exactly, and pytest adds `src` to `testpaths` (the doctest root).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from _famconfig import DELTAS, REPO
from lab_commons.dev.famconfig import SectionDelta
from lab_commons.dev.famtests import configrender, devdocs, famfiles, worktreeplace

_ROOT: Final = Path(__file__).resolve().parents[2]

#: The alias the kit uses for this repo in its family tables.
_ALIAS: Final = 'consumer-c'

#: Where the family's dev pages live relative to `docs-src/dev/` when lab-commons is checked out
#: beside this repo -- the base the committed pointer table is rendered against.
_DEVDOCS_BASE: Final = '../../../lab-commons/docs-src/dev'

_RERENDER: Final = 'python -m lab_commons.dev.famfiles --deltas tests/architecture/_famconfig.py'

#: The 52 ruff codes all three consumers ignore beyond the family's 10 (lab-commons 9ce57c3,
#: `CONSUMER_IGNORE_TAIL`). This repo's waiver set is exactly this tail.
_CONSUMER_IGNORE_TAIL: Final[tuple[str, ...]] = (
    'ANN001', 'ANN002', 'ANN003', 'ANN201', 'ANN202', 'ANN206', 'ARG002', 'B018', 'B904', 'C901',
    'D100', 'D101', 'D102', 'D103', 'D104', 'D105', 'D107', 'D205', 'D415', 'D417', 'DTZ005', 'E501',
    'ERA001', 'EXE', 'FBT', 'FIX', 'INP001', 'LOG015', 'N801', 'N802', 'N803', 'N805', 'N806', 'N815',
    'N816', 'NPY002', 'PLR0912', 'PLR0915', 'PLR2004', 'PLW2901', 'PT012', 'PT018', 'RUF012', 'RUF043',
    'S101', 'S311', 'SIM108', 'SIM113', 'SLF001', 'TD', 'TRY300', 'UP017',
)  # fmt: skip

SECTION_DELTAS: Final[dict[str, SectionDelta]] = {
    '[tool.ruff]': SectionDelta(repo=_ALIAS, added={}, dropped={}, ceiling=0),
    '[tool.ruff.lint]': SectionDelta(repo=_ALIAS, added={'ignore': _CONSUMER_IGNORE_TAIL}, dropped={}, ceiling=52),
    '[tool.ruff.format]': SectionDelta(repo=_ALIAS, added={}, dropped={}, ceiling=0),
    '[project]': SectionDelta(repo=_ALIAS, added={}, dropped={}, ceiling=0),
    '[tool.pytest.ini_options]': SectionDelta(repo=_ALIAS, added={'testpaths': ('src',)}, dropped={}, ceiling=1),
}

#: Tracked project-level files this repo owns beyond the family list. None, measured 2026-10-03.
OWNED_HERE: Final[dict[str, str]] = {}


def test_every_worktree_stays_inside() -> None:
    """WORKTREES-STAY-INSIDE: every linked worktree lives under `<main>/.claude/worktrees/`."""
    worktreeplace.assert_every_worktree_stays_inside(root=_ROOT)


def test_every_managed_file_is_rendered() -> None:
    """PROJECT-FILES-HAVE-ONE-SOURCE: each managed file is the family base plus this repo's delta."""
    famfiles.assert_every_managed_file_is_rendered(root=_ROOT, deltas=DELTAS, repo=REPO, rerender_hint=_RERENDER)


def test_every_project_file_is_accounted_for() -> None:
    """PROJECT-FILES-HAVE-ONE-SOURCE: no tracked project file is outside the family list and OWNED_HERE."""
    famfiles.assert_every_project_file_is_accounted_for(root=_ROOT, owned_here=OWNED_HERE)


def test_every_section_is_owned() -> None:
    """PROJECT-FILES-HAVE-ONE-SOURCE: every family config table is its section base plus our delta."""
    configrender.assert_every_section_is_owned(root=_ROOT, deltas=SECTION_DELTAS, repo=REPO)


def test_the_index_holds_the_live_table() -> None:
    """PROJECT-FILES-HAVE-ONE-SOURCE: `docs-src/dev/index.md` holds the kit's live pointer table."""
    devdocs.assert_the_index_holds_the_live_table(root=_ROOT, base=_DEVDOCS_BASE)
