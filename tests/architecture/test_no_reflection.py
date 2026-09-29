"""NO-GETATTR: `src/` and `tests/` carry zero ``getattr``/``hasattr`` calls and zero ``__getattr__`` hooks.

User directive 2026-09-26, family-wide: reflection is banned outright. A declared field is read by
attribute access, a name-keyed lookup through ``vars(obj)`` or an explicit mapping, and an optional
capability through a ``runtime_checkable`` Protocol. Zero is not a ceiling to walk down, so the pin
names offending sites rather than a count.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from lab_commons.dev.rules import tracked_files

_ROOT: Final = Path(__file__).resolve().parents[2]
_BANNED_CALLS: Final = frozenset({'getattr', 'hasattr'})
#: Below the 2026-09-29 measurement on purpose: a floor refuses an UNREAD tree.
_FILES_READ_FLOOR: Final = 40


def _reflection_sites(source: str, filename: str) -> list[int]:
    """Line numbers of every banned reflection call and ``__getattr__`` definition."""
    tree = ast.parse(source, filename=filename)
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALLS)
        or (isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == '__getattr__')
    )


def test_a_planted_getattr_hasattr_and_hook_are_all_found() -> None:
    planted = 'def __getattr__(name):\n    return getattr(object(), name)\nhasattr(object(), "x")\n'
    assert _reflection_sites(planted, '<planted>') == [1, 2, 3]


def test_src_and_tests_carry_no_reflection_at_all() -> None:
    names = [n for n in tracked_files(_ROOT) if n.startswith(('src/', 'tests/')) and n.endswith('.py')]
    files = [_ROOT / n for n in names if (_ROOT / n).is_file()]
    assert len(files) >= _FILES_READ_FLOOR, f'reflection scan read {len(files)} files, floor {_FILES_READ_FLOOR}'
    offenders = [
        f'{path.relative_to(_ROOT).as_posix()}:{line}'
        for path in files
        for line in _reflection_sites(path.read_text(encoding='utf-8'), str(path))
    ]
    assert not offenders, f'getattr/hasattr/__getattr__ is banned (directive 2026-09-26): {offenders}'
