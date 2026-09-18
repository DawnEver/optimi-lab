"""The RUNTIME of `optimi_lab` is numpy, scipy and scikit-learn -- checked, not claimed.

`src/optimi_lab/__init__.py` states "Dependencies are numpy, scipy and scikit-learn. No plotting,
no logging framework, no path resolution, no configuration file" and `pyproject.toml` repeats it in
a comment. Until this module, both were PROSE: a `structlog` import added to one function would
have left every word of that claim reading exactly as it does now. A declaration the code does not
enforce is the defect this family names first, so the claim is read off the two places it can
actually be violated -- the dependency table and the import graph.

THE GUARD IS ONE FUNCTION, CALLED TWICE. `foreign_roots` is what the real check runs and what the
planted control runs: a control that re-implemented the comparison would agree with itself and
prove nothing about the guard that ships.

`lab-commons` IS THE REASON THIS IS SHARP RATHER THAN THEORETICAL. It is adopted here, it pulls in
structlog, pint and pydantic, and it belongs to the `dev` extra for exactly that reason. The one
edit that breaks this library for its callers is moving that line up into `[project.dependencies]`,
and `test_lab_commons_is_dev_only_and_floats` is the line that refuses it.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path
from typing import Final

import pytest
from lab_commons.dev import floors

_ROOT: Final = Path(__file__).resolve().parents[2]
_SRC: Final = _ROOT / 'src' / 'optimi_lab'

#: The whole runtime dependency set. A fourth entry is a decision taken in this file, by name.
RUNTIME: Final = frozenset({'numpy', 'scipy', 'scikit-learn'})

#: The import roots those distributions provide, plus the package itself. `sklearn` is
#: `scikit-learn`'s import name and the two spellings are not derivable from one another.
RUNTIME_IMPORT_ROOTS: Final = frozenset({'numpy', 'scipy', 'sklearn', 'optimi_lab'})

#: A floor on the scan. Finding no foreign import across three files is not a pure package, it is a
#: walk that stopped reaching one.
#: RE-MEASURED 2026-09-18: 22 shipped modules.
SOURCE_FILE_FLOOR: Final = 15

#: THE OTHER SIDE OF ``SOURCE_FILE_FLOOR``. Today's reading is 22 - 15 = 7.
SOURCE_FILE_HEADROOM: Final = 12

#: Version specifiers that are a CEILING rather than a floor. An upper bound is a bet that a
#: release nobody has read will break us, and it is the shape `LATEST-DEPENDENCIES` refuses.
_CEILINGS: Final = ('<', '==', '~=')


def foreign_roots(module: ast.Module, allowed: frozenset[str]) -> frozenset[str]:
    """The top-level import roots in *module* that are neither stdlib nor in *allowed*.

    THE GUARD. Relative imports carry no root and are skipped; `from a.b import c` and `import a.b`
    both reduce to `a`, because a distribution is what a fresh install pulls and `a.b` never is.
    """
    roots: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split('.')[0])
    return frozenset(roots - allowed - set(sys.stdlib_module_names))


def _requirements() -> tuple[list[str], dict[str, list[str]]]:
    """`(runtime requirements, extras)` exactly as `pyproject.toml` declares them."""
    table = tomllib.loads((_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    project = table['project']
    return project.get('dependencies', []), project.get('optional-dependencies', {})


def _source_files() -> list[Path]:
    """Every shipped module, excluding the version file setuptools-scm generates."""
    return sorted(p for p in _SRC.rglob('*.py') if p.name != '__version__.py')


def _distribution(requirement: str) -> str:
    """The distribution name a requirement line names, without its marker, extra or specifier."""
    head = requirement.split(';', maxsplit=1)[0].split('@', maxsplit=1)[0].strip()
    for mark in ('[', '<', '>', '=', '~', '!', ' '):
        head = head.split(mark)[0]
    return head.strip()


def test_the_runtime_dependency_set_is_exactly_the_declared_one() -> None:
    """Set EQUALITY, both directions: an addition reds, and so does a stale name nothing installs."""
    runtime, _ = _requirements()
    declared = {_distribution(line) for line in runtime}
    assert declared == set(RUNTIME), (
        f'runtime dependencies are {sorted(declared)}, not {sorted(RUNTIME)}. Arrived and undeclared: '
        f'{sorted(declared - RUNTIME)} -- this library is imported BY other libraries, so a dependency '
        f'here is one every caller inherits. Declared but gone: {sorted(RUNTIME - declared)} -- delete '
        f'the name here in the same edit that dropped it, or this waiver outlives the thing it waived.'
    )


def test_lab_commons_is_dev_only_and_floats() -> None:
    """The shared dev kit is adopted at the DEV layer: in an extra, by bare URL, never pinned."""
    runtime, extras = _requirements()
    assert not any(_distribution(line) == 'lab-commons' for line in runtime), (
        'lab-commons is in [project.dependencies]. It brings structlog, pint and pydantic, which is '
        'the whole runtime this library refuses to impose on a caller. It belongs in the dev extra.'
    )
    lines = [line for entries in extras.values() for line in entries if _distribution(line) == 'lab-commons']
    assert lines, 'no extra declares lab-commons, so the adoption test beside this one runs for nobody.'
    for line in lines:
        assert 'git+https' in line, f'{line!r} does not resolve lab-commons from its repository.'
        tail = line.split('git+https', 1)[1]
        assert '@' not in tail, (
            f'{line!r} pins lab-commons to a ref. A pin is a ceiling nobody re-argued: the dev kit moves '
            f'forward with the fleet, and a repo held at a sha stops being told when a rule changed.'
        )


def test_no_runtime_requirement_carries_an_upper_bound() -> None:
    """A floor is a statement; a ceiling is a guess about a release nobody has read."""
    runtime, _ = _requirements()
    capped = [line for line in runtime if any(mark in line for mark in _CEILINGS)]
    assert not capped, (
        f'{capped} bound a dependency from above. Move to the LATEST and record the failure if one '
        f'appears; a cap added to make one red green is a cap no later reader can tell from a fact.'
    )


def test_no_shipped_module_imports_outside_the_runtime_set() -> None:
    """THE IMPORT GRAPH, which is where the docstring's claim is actually kept or broken."""
    files = _source_files()
    floors.assert_floor(len(files), floor=SOURCE_FILE_FLOOR, what='RUNTIME-PURITY (shipped modules)')
    floors.assert_floor_still_binds(
        len(files), floor=SOURCE_FILE_FLOOR, headroom=SOURCE_FILE_HEADROOM, what='RUNTIME-PURITY (shipped modules)'
    )
    offenders = {
        path.relative_to(_ROOT).as_posix(): sorted(found)
        for path in files
        if (found := foreign_roots(ast.parse(path.read_text(encoding='utf-8')), RUNTIME_IMPORT_ROOTS))
    }
    assert not offenders, (
        f'{offenders} import distributions outside the declared runtime. Either the import goes, or '
        f'the distribution is added to RUNTIME and to [project.dependencies] as a deliberate cost to '
        f'every caller -- but the package docstring may not keep saying three while the graph says four.'
    )


@pytest.mark.parametrize('planted', ['structlog', 'pint', 'pydantic', 'matplotlib', 'lab_commons'])
def test_the_guard_names_a_planted_heavyweight(planted: str) -> None:
    """THE CONTROL. Plant the import in a module and call the REAL guard, not a copy of it.

    Each planted name is a dependency this package deliberately dropped, so the control is also the
    list of what a well-meaning commit is most likely to add back.
    """
    module = ast.parse(f'import numpy\nfrom {planted} import thing\n\n\ndef f():\n    return thing\n')
    assert foreign_roots(module, RUNTIME_IMPORT_ROOTS) == frozenset({planted}), (
        f'the guard did not name a planted {planted!r} import. A scan that cannot fail on a planted '
        f'violation is not evidence about the tree it reports green.'
    )
