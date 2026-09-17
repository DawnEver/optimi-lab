"""TOLERANCE-CARRIES-A-UNIT, in the one half of it that has a subject in a unitless package.

THE OLD REASON, RE-MEASURED. This repo declared the rule absent among the "no physical units" group:
`Variable.unit` is a free-text LABEL this package never computes with, pint is not a dependency it
may take, and there is no golden generated in millimetres waiting to be tightened by 1000x. All of
that is still TRUE and none of it is the whole rule. The rule's last clause is about a tolerance's
SHAPE rather than about a unit: *never write a `rel=` without the `abs=` floor it is combined with*
-- and that clause needs no unit at all, only a comparison whose two sides can both approach zero.

WHY IT BITES IN A PACKAGE LIKE THIS ONE, which is the reason it is worth a mechanism rather than a
sentence. `pytest.approx(x, rel=1e-9)` is a RATIO, so at `x == 0` it admits only exactly zero, and
this package's subject is full of quantities that pass through zero: a crowding distance on a front
of one, a weight in a decomposition, a velocity at a converged swarm, an r2 on a constant column.
A bare `rel=` there is not a loose test, it is a test whose tolerance silently becomes zero for the
inputs most likely to be interesting -- and `abs=` is what pins the floor it falls back to.

WHAT IS PROVED. Every `approx` call in the tracked corpus is PARSED, and one carrying `rel=` without
`abs=` is named. The scan is asserted to have read a floor of real call sites first, because its
natural answer is "no violations" -- which is also what a walk of the wrong directory returns.

ADOPTED AT ZERO, measured 2026-09-17: two `approx` call sites, both bare, neither carrying `rel=`.
So the declared set is EMPTY and this guard's job is to refuse the FIRST arrival rather than to
record a debt. The control below is what makes an empty answer mean clean rather than unread: it
plants all four shapes and calls the REAL scanner on them.

WHAT THIS DOES NOT CLAIM. It says nothing about absolute tolerances written as bare floats in the
package's own arithmetic, and nothing about units -- `UNITS-GO-THROUGH-PINT` remains declared absent
here with its own reason, and narrowing THIS scan is not a way of answering THAT rule.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parents[2]

#: The git executable, resolved rather than looked up on PATH at call time.
_GIT: Final = shutil.which('git') or 'git'

#: The floor on the scan: the tracked corpus really contains this many `approx` call sites, measured
#: 2026-09-17. Below it the walk did not reach the tests it reports on, and "no violations" would be
#: a statement about an empty list.
APPROX_CALL_FLOOR: Final = 2

#: The floor on the corpus itself, so a scan that parsed nothing cannot report clean either.
PYTHON_FILE_FLOOR: Final = 40


def tracked_python_files() -> list[Path]:
    """Every tracked `.py` file, asked of git -- what is COMMITTED, not what a walk happens to find."""
    listed = subprocess.run(
        [_GIT, 'ls-files', '--', '*.py'],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return [_ROOT / line for line in listed.stdout.splitlines() if line.strip()]


def _is_approx(node: ast.Call) -> bool:
    """A call to `approx`, however it was imported -- `pytest.approx(...)` or a bare `approx(...)`."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == 'approx'
    return isinstance(func, ast.Name) and func.id == 'approx'


def approx_calls(tree: ast.AST) -> list[ast.Call]:
    """Every `approx` call in *tree*, in source order."""
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _is_approx(node)]


def unfloored(tree: ast.AST) -> tuple[int, ...]:
    """THE GUARD. The line of every `approx` call giving a RATIO with no floor under it.

    A call with neither keyword is left alone: that is `approx`'s own default, which already
    combines a relative band with an absolute one. What is named is the call that OVERRODE the
    relative half and left the absolute half unstated -- the shape whose tolerance goes to zero
    exactly where the quantity does.
    """
    return tuple(
        call.lineno
        for call in approx_calls(tree)
        if {keyword.arg for keyword in call.keywords} & {'rel'}
        and 'abs' not in {keyword.arg for keyword in call.keywords}
    )


def test_no_relative_tolerance_is_written_without_its_floor() -> None:
    """THE CHECK, over the tracked corpus, with what the scan actually read asserted first."""
    files = tracked_python_files()
    assert len(files) >= PYTHON_FILE_FLOOR, (
        f'the scan listed {len(files)} tracked Python files, below the {PYTHON_FILE_FLOOR} floor. Finding no '
        f'bare tolerance in a corpus nobody parsed is not a clean corpus.'
    )
    trees = {path: ast.parse(path.read_text(encoding='utf-8')) for path in files}
    seen = sum(len(approx_calls(tree)) for tree in trees.values())
    assert seen >= APPROX_CALL_FLOOR, (
        f'the scan parsed {seen} `approx` call sites, below the {APPROX_CALL_FLOOR} floor measured when this '
        f'guard was written. Either the calls moved somewhere this walk cannot see, or it read the wrong tree.'
    )
    offenders = {
        path.relative_to(_ROOT).as_posix(): lines for path, tree in trees.items() if (lines := unfloored(tree))
    }
    assert not offenders, (
        f'{offenders} compare with `rel=` and no `abs=`. A ratio admits only exactly zero at zero, so that '
        f'tolerance vanishes precisely where the quantity it measures does. State the floor it falls back to.'
    )


def test_the_guard_tells_the_four_shapes_apart() -> None:
    """THE CONTROL. Plant each shape and call the REAL scanner -- an empty live answer means nothing
    unless the scanner is shown to answer at all.
    """
    planted = ast.parse(
        'import pytest\n'
        'from pytest import approx\n'
        'a = value == pytest.approx(0.0, rel=1e-9)\n'
        'b = value == pytest.approx(0.0, rel=1e-9, abs=1e-12)\n'
        'c = value == pytest.approx(0.0)\n'
        'd = value == approx(0.0, rel=1e-9)\n'
        'e = value == some.other.call(0.0, rel=1e-9)\n'
    )
    assert len(approx_calls(planted)) == 4, (
        f'the scanner found {len(approx_calls(planted))} approx calls in four planted ones -- it either misses '
        f'an import spelling or claims a call that is not approx at all.'
    )
    assert unfloored(planted) == (3, 6), (
        f'the guard answered {unfloored(planted)}. It must name the bare `rel=` under both import spellings, '
        f'leave the floored call and the defaulted call alone, and never reach a call that is not approx.'
    )
