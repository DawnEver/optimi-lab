"""The RUNTIME of `optimi_lab` is numpy, scipy, numerics-lab and `lab_commons.outcomes` -- checked.

PLAN II.1 (2026-10-17) moved lab-commons into the base install for ONE module: `lab_commons.outcomes`,
the standard-library-only task-outcome vocabulary every engine shares. Any other lab-commons module
(structlog, pint and pydantic behind `log`, `units`, `config`) is still refused here, by the same
guard, which now matches an allowed import by its dotted PREFIX rather than its root.

`src/optimi_lab/__init__.py` states which distributions the runtime is, and that it has no plotting,
no logging framework, no path resolution and no configuration file; `pyproject.toml` repeats it in
a comment. Until this module, both were PROSE: a `structlog` import added to one function would
have left every word of that claim reading exactly as it does now. A declaration the code does not
enforce is the defect this family names first, so the claim is read off the two places it can
actually be violated -- the dependency table and the import graph.

THE GUARD IS ONE FUNCTION, CALLED TWICE. `foreign_roots` is what the real check runs and what the
planted control runs: a control that re-implemented the comparison would agree with itself and
prove nothing about the guard that ships.

`lab-commons` IS THE REASON THIS IS SHARP RATHER THAN THEORETICAL. Its other modules pull in
structlog, pint and pydantic; `foreign_roots` admits `lab_commons.outcomes` and names any other
`lab_commons.*` import, and `test_lab_commons_floats_and_the_dev_kit_stays_in_dev` keeps the dev kit
(`dev-lab`) in the development dependency group, not the runtime.

NUMERICS-LAB IS THE FIFTH ENTRY, by decision (plan R4, P9): equation solving is one engine and this
library keeps no numeric kernel of its own (`test_equation_solving_is_numerics_labs.py`). It floats
like lab-commons -- a bare repository URL, never a ref (`test_every_engine_floats`).
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path
from typing import Final

import pytest
from dev_lab.dev import floors

_ROOT: Final = Path(__file__).resolve().parents[2]
_SRC: Final = _ROOT / 'src' / 'optimi_lab'

#: The whole runtime dependency set. A sixth entry is a decision taken in this file, by name.
RUNTIME: Final = frozenset({'numpy', 'scipy', 'lab-commons', 'numerics-lab'})

#: The import roots those distributions provide, plus the package itself. scikit-learn left the
#: runtime with P10 (plan R1, D58): every surrogate is self-implemented and sklearn is a TEST oracle.
RUNTIME_IMPORT_ROOTS: Final = frozenset({
    'numpy',
    'scipy',
    'optimi_lab',
    'lab_commons.outcomes',
    'numerics_lab',
})

#: A floor on the scan. Finding no foreign import across three files is not a pure package, it is a
#: walk that stopped reaching one.
#: RE-TAKEN 2026-10-17 AT 37: the P8 base (problem, oracle, archive, run, solver) read 48 shipped modules.
#: RE-TAKEN 2026-10-18 by P12 (population methods): 63 shipped modules.
#: RE-TAKEN 2026-10-18 AT 66: the P9 + P12 merge read 72.
#: RE-TAKEN 2026-10-18 AT 78: P10 (surrogates, optimi_lab.model) read 84.
#: RE-TAKEN 2026-10-18 AT 65 after merging P9 (operator, gradient solvers) onto P12.
#: RE-TAKEN 2026-10-18 AT 95: P11 (model-based solvers, D59-D68) read 101; already red at c028663, which read 93.
#: RE-TAKEN 2026-10-18 AT 106: P14 (structured solvers, D126-D132) read 112; 329e959 (P11) read 101.
SOURCE_FILE_FLOOR: Final = 106

#: THE OTHER SIDE OF ``SOURCE_FILE_FLOOR``. Today's reading is 22 - 15 = 7.
SOURCE_FILE_HEADROOM: Final = 12

#: Version specifiers that are a CEILING rather than a floor. An upper bound is a bet that a
#: release nobody has read will break us, and it is the shape `LATEST-DEPENDENCIES` refuses.
_CEILINGS: Final = ('<', '==', '~=')


def foreign_roots(module: ast.Module, allowed: frozenset[str]) -> frozenset[str]:
    """The imports in *module* that are neither stdlib nor under an *allowed* dotted prefix.

    THE GUARD. Relative imports carry no root and are skipped. An import is named by its root (`a` for
    `a.b`), except under a root that is only PARTLY allowed (`lab_commons`, of which only `outcomes`
    is runtime), where the full module is named so the message says which part arrived.
    """
    roots: set[str] = set()
    names: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    partial = {prefix.split('.')[0] for prefix in allowed if '.' in prefix}
    for name in names:
        root = name.split('.')[0]
        if root in sys.stdlib_module_names or any(name == a or name.startswith(a + '.') for a in allowed):
            continue
        roots.add(name if root in partial else root)
    return frozenset(roots)


def _requirements() -> tuple[list[str], dict[str, list[str]]]:
    """`(runtime requirements, development groups)` from the manifest."""
    table = tomllib.loads((_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    project = table['project']
    return project.get('dependencies', []), table.get('dependency-groups', {})


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


def test_lab_commons_floats_and_the_dev_kit_stays_in_dev() -> None:
    """Runtime outcomes and the development kit are separate distributions and install scopes."""
    runtime, groups = _requirements()
    runtime_lines = [line for line in runtime if _distribution(line) == 'lab-commons']
    assert runtime_lines and all('[' not in line.split('@')[0] for line in runtime_lines), (
        f'{runtime_lines} -- the runtime takes lab-commons WITHOUT extras (plan II.1): its dev kit brings '
        f'what this library refuses to impose on a caller.'
    )
    assert all(_distribution(line) != 'dev-lab' for line in runtime)
    lines = [line for line in groups['dev'] if _distribution(line) == 'dev-lab']
    assert len(lines) == 1, 'the development group must install the independent development kit'
    assert not any('lab-commons[' in line for entries in groups.values() for line in entries)
    for line in [*runtime_lines, *lines]:
        assert 'git+https' in line, f'{line!r} does not resolve lab-commons from its repository.'
        tail = line.split('git+https', 1)[1]
        assert '@' not in tail, (
            f'{line!r} pins lab-commons to a ref. A pin is a ceiling nobody re-argued: libraries float (R8).'
        )


#: The engines this library depends on by repository; each floats (R8).
ENGINES: Final = ('lab-commons', 'numerics-lab')


@pytest.mark.parametrize('engine', ENGINES)
def test_every_engine_floats(engine: str) -> None:
    """An engine is a bare repository URL in the runtime: a ref would be a ceiling nobody re-argued (R8)."""
    runtime, _ = _requirements()
    lines = [line for line in runtime if _distribution(line) == engine]
    assert lines, f'{engine} is not a runtime dependency; ENGINES names it, so either it is or the name goes.'
    for line in lines:
        assert 'git+https' in line, f'{line!r} does not resolve {engine} from its repository.'
        assert '@' not in line.split('git+https', 1)[1], f'{line!r} pins {engine} to a ref; libraries float (R8).'


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


@pytest.mark.parametrize(
    'planted', ['structlog', 'pint', 'pydantic', 'matplotlib', 'sklearn', 'lab_commons', 'lab_commons.log', 'dev_lab']
)
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


#: R1 (plan PLAN-engines-and-one-outcome-vocabulary, 2026-10-17): the runtime takes from scipy only
#: these submodules; every optimisation, sampling and statistics ALGORITHM is self-implemented, and
#: scipy.optimize / scipy.stats appear only in tests, as reference oracles. scipy.linalg and
#: scipy.sparse.linalg LEFT the set 2026-10-18 (R4): every linear solve -- least squares, triangular,
#: Cholesky, KKT -- and its failure type are numerics-lab's (``numerics_lab.linear``).
R1_SCIPY: Final = frozenset({'scipy.sparse', 'scipy.special'})


def scipy_beyond_r1(module: ast.Module) -> frozenset[str]:
    """THE GUARD: every scipy module *module* imports that is not one of the R1 submodules."""
    names: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return frozenset(n for n in names if n.split('.')[0] == 'scipy' and n not in R1_SCIPY)


def test_the_runtime_takes_only_the_r1_scipy_submodules() -> None:
    files = _source_files()
    floors.assert_floor(len(files), floor=SOURCE_FILE_FLOOR, what='R1 scipy scan (shipped modules)')
    found = {
        p.relative_to(_ROOT).as_posix(): sorted(scipy_beyond_r1(ast.parse(p.read_text(encoding='utf-8'))))
        for p in files
    }
    offenders = {k: v for k, v in found.items() if v}
    assert not offenders, (
        f'{offenders} import scipy beyond R1 {sorted(R1_SCIPY)}; a linear solve is numerics_lab.linear.'
    )


def test_the_r1_guard_names_a_planted_scipy_algorithm() -> None:
    planted = ast.parse(
        'from scipy.stats import qmc\nimport scipy.optimize\nfrom scipy.special import ndtr\n'
        'from scipy.linalg import lstsq\nimport scipy.sparse as sp\n'
    )
    assert scipy_beyond_r1(planted) == {'scipy.stats', 'scipy.optimize', 'scipy.linalg'}
