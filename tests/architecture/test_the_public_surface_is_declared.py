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

AND EVERY SCAN HERE NOW HAS A PLANTED CONTROL, which it did not until 2026-09-15. Before that
date this module was cited as the mechanism for SEVEN rules while containing no test that could
demonstrate any of its guards failing: `OVERSIZE_PINS` and `LAZY_IMPORT_PINS` are both EMPTY, so
`found == set(PINS)` compared an empty set to an empty set and the two-sided ratchet those rules
claim never executed in either direction. An empty set equalling an empty set is the same shape
that let an emptied table pass elsewhere in this family. The fix is not to seed a fake pin -- a
pin nothing needs is the OTHER side of the same defect -- it is to extract the comparison into
:func:`pin_gap`, :func:`oversize`, :func:`skipping` and :func:`assert_floor`, have the live tests
call exactly those, and have a control call the SAME function on PLANTED input. The control is
what proves the empty result means clean rather than unread.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pytest

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
#: Its EMPTINESS is what :func:`test_the_ratchet_refuses_both_an_arrival_and_a_stale_pin` exists
#: for: with nothing in it the live comparison below cannot fail, so the control carries the
#: evidence that it would.
OVERSIZE_PINS: Final = frozenset()

#: Functions allowed to import lazily, by NAME, each with its reason. Empty: nothing in this
#: package is circular and nothing in it is optional. Same emptiness, same control.
LAZY_IMPORT_PINS: Final[dict[str, str]] = {}

#: The skip marker, in either spelling. `xfail` is deliberately NOT matched: an xfail records the
#: failure and is the shape this rule asks for.
SKIP_MARKER: Final = re.compile(r'@\s*pytest\s*\.\s*mark\s*\.\s*skip|\bpytest\s*\.\s*skip\s*\(')


def _shipped() -> list[Path]:
    """Every shipped module, excluding the version file setuptools-scm generates."""
    return sorted(p for p in _SRC.rglob('*.py') if p.name != '__version__.py')


def _test_modules() -> list[Path]:
    """Every test module in the suite."""
    return sorted(p for p in _TESTS.rglob('test_*.py'))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'))


def _name(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def assert_floor(files: Iterable[Path], floor: int, what: str) -> list[Path]:
    """THE FLOOR, as one function every scan below calls, so one control answers for all of them.

    Returns the file list when it is long enough and raises otherwise. A scan that walked the
    wrong directory finds nothing and reports exactly what a clean tree reports; this is the line
    that tells the two apart.
    """
    found = list(files)
    assert len(found) >= floor, (
        f'the scan reached {len(found)} {what}, below the {floor} floor -- a green over a list this '
        f'short says nothing about the package, it says the walk did not reach it.'
    )
    return found


def declared_names(module: ast.Module) -> tuple[str, ...] | None:
    """The `__all__` entries of *module*, or None if it declares none.

    None and `()` are different answers and are kept apart on purpose: a module with no `__all__`
    exports whatever it happens to hold, while an empty one is a module that exports nothing and
    says so.
    """
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets):
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


def second_homes(files: Iterable[Path]) -> dict[str, list[str]]:
    """Public names exported from more than one leaf module, to the modules that export them.

    THE FIX-THE-CAUSE GUARD. Two spellings of one name is how a patched symptom outlives the thing
    it patched: each module reads fine alone, and only the merged tree can see that a caller's
    import now depends on which one it happened to reach. `__init__` re-exports by design and is
    excluded; a leaf may not shadow a leaf.
    """
    homes: defaultdict[str, list[str]] = defaultdict(list)
    for path in files:
        if path.name == '__init__.py':
            continue
        for name in declared_names(_tree(path)) or ():
            homes[name].append(_name(path))
    return {name: where for name, where in homes.items() if len(where) > 1}


def oversize(files: Iterable[Path], band: int) -> frozenset[str]:
    """The modules in *files* longer than *band* lines."""
    return frozenset(_name(path) for path in files if len(path.read_text(encoding='utf-8').splitlines()) > band)


def skipping(files: Iterable[Path]) -> frozenset[str]:
    """The test modules in *files* that skip a test rather than xfailing it."""
    return frozenset(_name(path) for path in files if SKIP_MARKER.search(path.read_text(encoding='utf-8')))


def pin_gap(found: Iterable[str], pins: Iterable[str]) -> tuple[list[str], list[str]]:
    """`(arrived and unpinned, pinned with nothing behind it)` -- THE TWO SIDES OF EVERY RATCHET.

    Both pin sets in this module are empty, so the live call returns `([], [])` from two empty
    inputs. That is a measurement about the package and NOT evidence that the comparison works,
    which is why this function is separate and why the control below drives it in each direction
    with planted input.
    """
    found_set, pin_set = set(found), set(pins)
    return sorted(found_set - pin_set), sorted(pin_set - found_set)


def test_every_shipped_module_declares_its_public_surface() -> None:
    """A module that declares nothing exports everything it happens to hold, including its imports."""
    files = assert_floor(_shipped(), SOURCE_FILE_FLOOR, 'modules')
    silent = [_name(p) for p in files if declared_names(_tree(p)) is None]
    assert not silent, (
        f'{silent} declare no `__all__`, so their public surface is whatever they happen to hold -- '
        f'including every name they imported. Declare the surface, even if it is empty.'
    )


def test_no_public_name_has_two_definitions() -> None:
    """One name, one home -- the cause fixed at the source, not a caller taught which import to take."""
    files = assert_floor(_shipped(), SOURCE_FILE_FLOOR, 'modules')
    clashes = second_homes(files)
    assert not clashes, (
        f'{clashes} are each exported from more than one module. Two definitions of one name is the '
        f'defect only the merged tree can see: pick the home and re-export from the other.'
    )


def test_no_module_imports_lazily() -> None:
    """An import graph a reader (or a cost model) reads must be exact; a deferred import makes it a guess."""
    files = assert_floor(_shipped(), SOURCE_FILE_FLOOR, 'modules')
    found = {f'{_name(path)}::{name}' for path in files for name in lazy_imports(_tree(path))}
    unpinned, stale = pin_gap(found, LAZY_IMPORT_PINS)
    assert not unpinned and not stale, (
        f'lazy imports that are not pinned: {unpinned} -- this package has three eagerly imported '
        f'dependencies, so there is no circular and no optional import to defer. Pins with nothing '
        f'behind them: {stale} -- a waiver nothing uses is a hole that reads as a decision.'
    )


def test_every_module_is_inside_the_size_band() -> None:
    """Two-sided: a module that grows past the band reds, and so does a pin nothing needs."""
    files = assert_floor(_shipped(), SOURCE_FILE_FLOOR, 'modules')
    unpinned, stale = pin_gap(oversize(files, MODULE_SIZE_BAND), OVERSIZE_PINS)
    assert not unpinned and not stale, (
        f'past the {MODULE_SIZE_BAND}-line band and unpinned: {unpinned} -- split it, or pin it here '
        f'with the reason it may not be split. Pinned but now inside the band: {stale} -- the budget a '
        f'shrink freed is given back in the same edit, never banked as slack for the next module.'
    )


def test_no_test_is_skipped() -> None:
    """A skip records nothing: it makes work that stopped working look like work never started."""
    files = assert_floor(_test_modules(), TEST_FILE_FLOOR, 'test modules')
    skipped = sorted(skipping(files))
    assert not skipped, (
        f'{skipped} skip a test. A known failure is an `xfail` carrying the reason it fails, so the day '
        f'it starts passing is a day somebody hears about.'
    )


# --------------------------------------------------------------------------------------------
# THE CONTROLS. Each plants the violation and calls the SAME function the live test above calls.
# A control that re-implemented the comparison would agree with itself and say nothing about the
# guard that ships.
# --------------------------------------------------------------------------------------------


def test_the_floor_refuses_a_walk_that_reached_nothing() -> None:
    """THE CONTROL FOR EVERY FLOOR, since all five scans route through one function."""
    with pytest.raises(AssertionError, match='below the 15 floor'):
        assert_floor([Path('a.py'), Path('b.py')], SOURCE_FILE_FLOOR, 'modules')
    assert len(assert_floor([Path(f'{i}.py') for i in range(15)], SOURCE_FILE_FLOOR, 'modules')) == 15


def test_the_surface_guard_names_a_planted_silent_module() -> None:
    """A module with no `__all__` answers None, and an empty declaration is a different answer."""
    assert declared_names(ast.parse('import numpy\n\n\ndef f():\n    return 1\n')) is None
    assert declared_names(ast.parse('__all__ = []\n')) == ()
    assert declared_names(ast.parse("__all__ = ['f']\n\n\ndef f():\n    return 1\n")) == ('f',)


def test_the_one_home_guard_names_a_planted_second_definition(tmp_path: Path) -> None:
    """PLANT the same exported name in two leaves and call the REAL guard on them."""
    first, second, init = tmp_path / 'a.py', tmp_path / 'b.py', tmp_path / '__init__.py'
    first.write_text("__all__ = ['sample']\n", encoding='utf-8')
    second.write_text("__all__ = ['sample', 'only_here']\n", encoding='utf-8')
    init.write_text("__all__ = ['sample']\n", encoding='utf-8')
    clashes = second_homes([first, second, init])
    assert set(clashes) == {'sample'}, (
        f'the guard did not name a planted second home: {clashes}. `only_here` has one home and the '
        f'`__init__` re-export is excluded by design, so exactly one name may come back.'
    )


def test_the_lazy_import_guard_names_a_planted_deferred_import() -> None:
    """PLANT an import inside a function body and call the REAL guard."""
    module = ast.parse(
        'import numpy\n\n\ndef eager():\n    return numpy\n\n\ndef deferred():\n    import json\n\n    return json\n'
    )
    assert lazy_imports(module) == frozenset({'deferred'}), (
        'the guard did not name a planted function-body import, or it named the eager one too.'
    )


def test_the_size_scan_names_a_planted_oversize_module(tmp_path: Path) -> None:
    """PLANT a module past the band and call the REAL scan; one inside it must not come back."""
    big, small = tmp_path / 'big.py', tmp_path / 'small.py'
    big.write_text('x = 1\n' * (MODULE_SIZE_BAND + 1), encoding='utf-8')
    small.write_text('x = 1\n' * MODULE_SIZE_BAND, encoding='utf-8')
    found = oversize([big, small], MODULE_SIZE_BAND)
    assert found == frozenset({_name(big)}), (
        f'the scan did not name a module {MODULE_SIZE_BAND + 1} lines long against a '
        f'{MODULE_SIZE_BAND}-line band, or it named the one exactly at the band: {sorted(found)}.'
    )


def test_the_skip_scan_names_a_planted_skip_and_leaves_an_xfail_alone(tmp_path: Path) -> None:
    """PLANT both spellings of a skip, and an xfail that must NOT be mistaken for one.

    THE PLANTED SPELLINGS ARE ASSEMBLED, never written out -- and that is evidence rather than
    style. Typed literally they made THIS module the live scan's first offender on 2026-09-15, so
    the control's own first run was also the demonstration that the scan fires. It is the same
    shape the registered-spelling half of this family names: a dead spelling survives longest in
    the prose explaining it, and a scan does not know prose from code (nor should it).
    """
    marked = tmp_path / 'test_marked.py'
    called = tmp_path / 'test_called.py'
    expected = tmp_path / 'test_xfail.py'
    token = 'sk' + 'ip'
    marked.write_text(f'@pytest.mark.{token}(reason="x")\ndef test_a(): ...\n', encoding='utf-8')
    called.write_text(f'def test_b():\n    pytest.{token}("x")\n', encoding='utf-8')
    expected.write_text('@pytest.mark.xfail(reason="residual 3e-4")\ndef test_c(): ...\n', encoding='utf-8')
    found = skipping([marked, called, expected])
    assert found == frozenset({_name(marked), _name(called)}), (
        f'the skip scan named {sorted(found)}. Both skip spellings must come back and the xfail must '
        f'not: an xfail records its failure, which is the shape this rule asks for.'
    )


def test_the_ratchet_refuses_both_an_arrival_and_a_stale_pin() -> None:
    """THE TWO-SIDED CONTROL, and the reason it exists: BOTH live pin sets are empty.

    `OVERSIZE_PINS` and `LAZY_IMPORT_PINS` hold nothing, so the two live comparisons above run an
    empty set against an empty set and can only pass. Driving the SAME function in each direction
    with planted input is what makes their green a measurement instead of a tautology.
    """
    assert pin_gap({'arrived.py'}, frozenset()) == (['arrived.py'], [])
    assert pin_gap(frozenset(), {'waived.py'}) == ([], ['waived.py'])
    assert pin_gap({'a', 'b'}, {'b', 'c'}) == (['a'], ['c'])
    assert pin_gap(frozenset(), frozenset()) == ([], []), 'the clean case must be the empty answer.'
