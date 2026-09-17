"""THE DOOR onto this repo's environment. The mechanism is the family's; the two answers are ours.

WHY THIS FILE IS THREE FUNCTIONS AND NOT A DOOR. `lab_commons.dev.dep` measures the interpreter's
own prefix, computes the env_key, refuses a mutation while a verdict is in flight and retires the
anchors a moved key invalidated -- identically in every repo, because none of that is repo-shaped.
Exactly two questions are not answerable from there, and they are the whole content of this file:

* WHO IS RUNNING A VERDICT IN THIS ENVIRONMENT RIGHT NOW. Here that is the box seat: this repo's
  only verdict command is `python -m lab_commons.dev.verify`, which holds `BoxLock` for the whole of
  ruff plus pytest. `BoxLock.holders` READS the records without taking them, which is the half that
  matters -- a probe that acquired would become a writer of the state it reports.
* WHICH FILES RECORD A VERDICT ABOUT THIS ENVIRONMENT. Here that is `.verify/verify-*.log`, the logs
  `run_verify` tees every step into and the only thing in this tree a later `env=` comparison can
  cite.

WHAT CHANGES BY SUPPLYING THEM, stated exactly. Before this, optimi-lab passed NEITHER, so the
door's report rendered `LOCK_UNDECLARED` and `NO_ANCHORS_DECLARED`: H1 -- a mutation DURING a verdict
run -- was UNGUARDED here rather than absent, and H2's remedy had nowhere to land. Both halves now
resolve, and the guard in
`tests/architecture/test_the_dependency_door_is_wired.py` is what refuses their quiet removal.

WHY A DEV SCRIPT AND NOT A PACKAGE MODULE. `optimi_lab` is a library about optimisation; where this
checkout keeps its verdict logs is not a fact about optimisation and does not ship. It is also not
domain code wearing a script's clothes -- it knows nothing except two paths in this tree.

Usage: ``python scripts/dep.py <requirement> [...]`` -- add ``--dry-run`` to perform every CHECK and
no mutation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from lab_commons.dev.boxlock import BoxLock
from lab_commons.dev.dep import Port, mutate
from lab_commons.dev.verify import LOG_DIRECTORY
from lab_commons.log import emit
from lab_commons.resources import Broker

__all__ = ['ANCHOR_GLOB', 'live_holders', 'main', 'port', 'repo_root', 'verdict_anchors']

#: How a verdict log is spelled by `lab_commons.dev.verify`. Kept as a name rather than inlined
#: twice, because the anchor half of the door is worth exactly as much as this pattern's accuracy.
ANCHOR_GLOB: Final = 'verify-*.log'


def repo_root() -> Path:
    """The root of THIS checkout, from this file's own location rather than from the caller's cwd."""
    return Path(__file__).resolve().parents[1]


def live_holders(*, broker: Broker | None = None) -> tuple[str, ...]:
    """Who is running a verdict on this box right now, NAMED -- the answer to H1's question.

    *broker* is injected so a control can plant a real holder in its own records directory rather
    than driving a re-implementation of this lookup, and so that planting one never touches the box
    a real run may be holding.
    """
    return tuple(holder.describe() for holder in BoxLock.holders(broker=broker))


def verdict_anchors(root: Path | None = None) -> tuple[Path, ...]:
    """Every verdict log in this tree -- the files a moved env_key turns into a statement about a
    dead environment.

    Sorted, so a report naming what it retired reads the same twice.
    """
    return tuple(sorted((root or repo_root()).joinpath(LOG_DIRECTORY).glob(ANCHOR_GLOB)))


def port(*, root: Path | None = None, broker: Broker | None = None) -> Port:
    """optimi-lab's half of the door: both adapters supplied, so neither gap is rendered."""
    return Port(
        name='optimi-lab',
        holders=lambda: live_holders(broker=broker),
        anchor_paths=lambda: verdict_anchors(root),
    )


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
