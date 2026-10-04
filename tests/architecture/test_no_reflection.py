"""NO-REFLECTION: no tracked Python calls getattr/hasattr/setattr/delattr or defines ``__getattr__``.

THE BODY IS THE FAMILY'S (`lab_commons.dev.famtests.noreflection`), over every tracked ``*.py``.
This file holds only the repo's answers: the allow-set, EMPTY because optimi_lab measured zero
reflection sites when the rule was adopted (lab-commons a2f50b9), and the read floor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import noreflection

_ROOT: Final = Path(__file__).resolve().parents[2]
_ALLOWED: Final[dict[str, str]] = {}
#: Below the 2026-10-04 measurement (64 tracked files) on purpose: a floor refuses an UNREAD tree.
_FILES_READ_FLOOR: Final = 50


def test_a_planted_reflection_site_of_every_shape_is_found() -> None:
    planted = 'def __getattr__(name):\n    return getattr(object(), name)\nhasattr(1, "x")\nsetattr(o, "a", 1)\ndelattr(o, "a")\n'
    assert noreflection.reflection_sites(planted, '<planted>') == [1, 2, 3, 4, 5]


def test_no_tracked_python_reflects() -> None:
    noreflection.assert_no_reflection(root=_ROOT, allowed=_ALLOWED, floor=_FILES_READ_FLOOR)
