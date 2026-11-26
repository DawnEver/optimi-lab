"""DECLARED-LAYOUT, NO-BACKUP-COPIES and ONE-FORMAT-PER-NEED over the optimi_lab checkout, through the kit body.

The table is `[layout]` of the root `layout.toml`; the walk, the predicate and the refusal text are
`dev_lab.dev.famtests.layout`'s. This file supplies only the floor on the top-level directories the walk reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.famtests.layout import assert_layout_holds

_ROOT: Final = Path(__file__).resolve().parents[2]

#: MEASURED 2026-11-26: the ten tracked top-level directories (`.claude`, `.codex`, `.github`, `archived`,
#: `docs-src`, `examples`, `scratch`, `scripts`, `src`, `tests`). A walk reading fewer has lost the tree.
TOP_LEVEL_FLOOR: Final = 10

#: The untracked directories a checkout adds on top (`.git`, `.venv`, `.venvs`, `output` and tool caches): 15
#: read on a bootstrapped workstation on 2026-11-26, so 6 above the floor is the measured spread, not slack.
TOP_LEVEL_HEADROOM: Final = 6


def test_the_tree_is_the_declared_layout() -> None:
    """Every directory matches its own row and every file a format that row allows."""
    assert_layout_holds(root=_ROOT, floor=TOP_LEVEL_FLOOR, headroom=TOP_LEVEL_HEADROOM)
