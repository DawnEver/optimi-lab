"""A test that runs a famtests body against a resource the cluster lacks wears that resource's marker.

The need is the body's fact, declared once in :mod:`dev_lab.dev.famtests.boxneeds`; this file only
points the check at optimi-lab: its test tree, its resource vocabulary, its floor. The floor is the six
box-body tests the scan reads today (host node, hook install, origin visibility): a scan that
finds fewer bodies than that has stopped reading the tree, not found it clean. It read ten until
dev-lab 04622431 ran the guard engine on a locked Node and dropped its engine-driving bodies
from boxneeds.NEEDS -- the population shrank at its source, so those tests lost the marker too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.famtests.boxneeds import assert_box_bodies_are_marked

ROOT: Final = Path(__file__).resolve().parents[2]


def test_every_box_body_carries_its_marker() -> None:
    assert_box_bodies_are_marked(
        tests=ROOT / 'tests', pyproject=(ROOT / 'pyproject.toml').read_text(encoding='utf-8'), floor=6
    )
