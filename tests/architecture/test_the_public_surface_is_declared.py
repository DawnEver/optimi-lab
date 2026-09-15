"""The shape of the package: a declared surface, eager imports, modules inside the size band.

Three properties that are cheap to state, invisible when broken, and each one a thing a single
commit can undo. None of them is about optimization -- they are about whether a reader of this
package can trust what it says about itself.

* `__all__` in EVERY shipped module, and one name with one definition. A second spelling of the
  same name is a defect only the whole tree can see: each module reads fine on its own.
* No import inside a function body. The rule allows an unavoidable circular or heavy-optional
  import WITH its reason written down -- and this package has neither, since its entire runtime is
  three eagerly-imported distributions, so the exemption set is empty and must stay empty by name.
* A module size BAND, with the oversize set pinned by NAME. The band is repo data (this package's
  largest module is 261 lines); that there IS one is not.

EVERY SCAN HERE HAS A FLOOR, for the reason the family states: finding nothing is vacuous rather
than green, and a walk that silently reached two files reports exactly what a clean tree reports.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parents[2]
_SRC: Final = _ROOT / 'src' / 'optimi_lab'
_TESTS: Final = _ROOT / 'tests'

#: Floors. Measured when this file was written: 17 shipped modules, 4 test modules.
SOURCE_FILE_FLOOR: Final = 15
TEST_FILE_FLOOR: Final = 3

#: The size band, in lines. Past it a module is refactored, or it is pinned below by name with the
#: reason it may not be -- the band is not a suggestion and the pin is not a default.
MODULE_SIZE_BAND: Final = 320

#: Modules allowed past the band, by NAME. Empty, and that is a measurement: the largest module in
#: the package is 261 lines. A count could not say WHICH module grew; this set can only shrink.
OVERSIZE_PINS: Final = frozenset()

#: Functions allowed to import lazily, by NAME, each with its reason. Empty: nothing in this
#: package is circular and nothing in it is optional.
LAZY_IMPORT_PINS: Final[dict[str, str]] = {}


def _shipped() -> list[Path]:
    """Every shipped module, excluding the version file setuptools-scm generates."""
    return sorted(p for p in _SRC.rglob('*.py') if p.name != '__version__.py')


def _test_modules() -> list[Path]:
    """Every test module in the suite."""
    return sorted(p for p in _TESTS.rglob('test_*.py'))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'))


def declared_names(module: ast.Module) -> tuple[str, ...] | None:
    """The `__all__` entries of *module*, or None if it declares none.

    None and `()` are different answers and are kept apart on purpose: a module with no `__all__`
    exports whatever it happens to hold, while an empty one is a module that exports nothing and
    says so.
    """
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets
        ):
            return tuple(elt.value for elt in node.value.elts if isinstance(elt, ast.Constant))
    return None


def lazy_imports(module: ast.Module) -> frozenset[str]:
    """The names of functions in *module* that import inside their body."""
    out: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(inner, (ast.Import, ast.ImportFrom)) for inner in ast.walk(node)
        ):
            out.add(node.name)
    return frozenset(out)


def test_every_shipped_module_declares_its_public_surface() -> None:
    """A module that declares nothing exports everything it happens to hold, including its imports."""
    files = _shipped()
    assert len(files) >= SOURCE_FILE_FLOOR, (
        f'the scan reached {len(files)} modules, below the {SOURCE_FILE_FLOOR} floor -- a green over a '
        f'list this short says nothing about the package.'
    )
    silent = [p.relative_to(_ROOT).as_posix() for p in files if declared_names(_tree(p)) is None]
    assert not silent, (
        f'{silent} declare no `__all__`, so their public surface is whatever they happen to hold -- '
        f'including every name they imported. Declare the surface, even if it is empty.'
    )


def test_no_public_name_has_two_definitions() -> None:
    """One name, one home. `__init__` re-exports and is excluded; a leaf module may not shadow a leaf."""
    homes: defaultdict[str, list[str]] = defaultdict(list)
    for path in _shipped():
        if path.name == '__init__.py':
            continue
        for name in declared_names(_tree(path)) or ():
            homes[name].append(path.relative_to(_ROOT).as_posix())
    clashes = {name: where for name, where in homes.items() if len(where) > 1}
    assert not clashes, (
        f'{clashes} are each exported from more than one module. Two definitions of one name is the '
        f'defect only the merged tree can see: pick the home and re-export from the other.'
    )


def test_no_module_imports_lazily() -> None:
    """An import graph a reader (or a cost model) reads must be exact; a deferred import makes it a guess."""
    files = _shipped()
    assert len(files) >= SOURCE_FILE_FLOOR, f'the scan reached only {len(files)} modules.'
    found = {
        f'{path.relative_to(_ROOT).as_posix()}::{name}'
        for path in files
        for name in lazy_imports(_tree(path))
    }
    assert found == set(LAZY_IMPORT_PINS), (
        f'lazy imports that are not pinned: {sorted(found - set(LAZY_IMPORT_PINS))} -- this package has '
        f'three eagerly imported dependencies, so there is no circular and no optional import to defer. '
        f'Pins with nothing behind them: {sorted(set(LAZY_IMPORT_PINS) - found)} -- a waiver nothing uses '
        f'is a hole that reads as a decision.'
    )


def test_every_module_is_inside_the_size_band() -> None:
    """Two-sided: a module that grows past the band reds, and so does a pin nothing needs."""
    files = _shipped()
    assert len(files) >= SOURCE_FILE_FLOOR, f'the scan reached only {len(files)} modules.'
    over = {
        path.relative_to(_ROOT).as_posix()
        for path in files
        if len(path.read_text(encoding='utf-8').splitlines()) > MODULE_SIZE_BAND
    }
    assert over == set(OVERSIZE_PINS), (
        f'past the {MODULE_SIZE_BAND}-line band and unpinned: {sorted(over - set(OVERSIZE_PINS))} -- split '
        f'it, or pin it here with the reason it may not be split. Pinned but now inside the band: '
        f'{sorted(set(OVERSIZE_PINS) - over)} -- the budget a shrink freed is given back in the same edit, '
        f'never banked as slack for the next module.'
    )


def test_no_test_is_skipped() -> None:
    """A skip records nothing: it makes work that stopped working look like work never started."""
    files = _test_modules()
    assert len(files) >= TEST_FILE_FLOOR, (
        f'the scan reached {len(files)} test modules, below the {TEST_FILE_FLOOR} floor.'
    )
    skipper = re.compile(r'@\s*pytest\s*\.\s*mark\s*\.\s*skip|\bpytest\s*\.\s*skip\s*\(')
    skipped = [p.relative_to(_ROOT).as_posix() for p in files if skipper.search(p.read_text(encoding='utf-8'))]
    assert not skipped, (
        f'{skipped} skip a test. A known failure is an `xfail` carrying the reason it fails, so the day '
        f'it starts passing is a day somebody hears about.'
    )
