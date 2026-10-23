"""THE DOOR onto this repo's environment. The mechanism is the family's; the one answer is ours.

WHY THIS FILE IS A FEW FUNCTIONS AND NOT A DOOR. `dev_lab.dev.dep` measures the interpreter's
own prefix, computes the env_key, refuses a mutation while a verdict holds THIS environment's seat
(`dev_lab.dev.envseat.EnvironmentSeat`, which `python -m dev_lab.dev.verify` takes for the
whole of ruff plus pytest) and retires the anchors a moved key invalidated -- identically in every
repo, because none of that is repo-shaped. Who may block a change is therefore not a repo answer.
Exactly one question is not answerable from there, and it is the whole content of this file:

* WHICH FILES RECORD A VERDICT ABOUT THIS ENVIRONMENT. Here that is `.verify/verify-*.log`, the logs
  `run_verify` tees every step into and the only thing in this tree a later `env=` comparison can
  cite.

WHAT CHANGES BY SUPPLYING IT, stated exactly. Without it the door's report renders
`NO_ANCHORS_DECLARED`: H2 detected and its remedy with nowhere to land.
`tests/architecture/test_the_dependency_door_is_wired.py` is what refuses its quiet removal.

WHY A DEV SCRIPT AND NOT A PACKAGE MODULE. `optimi_lab` is a library about optimisation; where this
checkout keeps its verdict logs is not a fact about optimisation and does not ship. It is also not
domain code wearing a script's clothes -- it knows nothing except one path in this tree.

Usage: ``python scripts/dep.py <requirement> [...]`` -- add ``--dry-run`` to perform every CHECK and
no mutation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

from dev_lab.dev.dep import Port, mutate
from dev_lab.dev.verify import LOG_DIRECTORY
from lab_commons.log import emit

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ['ANCHOR_GLOB', 'main', 'port', 'repo_root', 'verdict_anchors']

#: How a verdict log is spelled by `dev_lab.dev.verify`. Kept as a name rather than inlined
#: twice, because the anchor half of the door is worth exactly as much as this pattern's accuracy.
ANCHOR_GLOB: Final = 'verify-*.log'


def repo_root() -> Path:
    """The root of THIS checkout, from this file's own location rather than from the caller's cwd."""
    return Path(__file__).resolve().parents[1]


def verdict_anchors(root: Path | None = None) -> tuple[Path, ...]:
    """Every verdict log in this tree -- the files a moved env_key turns into a statement about a
    dead environment.

    Sorted, so a report naming what it retired reads the same twice.
    """
    return tuple(sorted((root or repo_root()).joinpath(LOG_DIRECTORY).glob(ANCHOR_GLOB)))


def port(*, root: Path | None = None) -> Port:
    """optimi-lab's half of the door: its verdict anchors, so no gap is rendered."""
    return Port(name='optimi-lab', anchor_paths=lambda: verdict_anchors(root))


def main(argv: Sequence[str] | None = None) -> int:
    """Change this environment through the door, and print what it did and what it retired."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('requirements', nargs='+', help='the distributions to install')
    parser.add_argument('--dry-run', action='store_true', help='run every check and mutate nothing')
    parsed = parser.parse_args(argv)
    report = mutate(parsed.requirements, port=port(), dry_run=parsed.dry_run)
    emit(report.render(), flush=True)
    return 0 if report.returncode in (None, 0) else 1


if __name__ == '__main__':
    sys.exit(main())
