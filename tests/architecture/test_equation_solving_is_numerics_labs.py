"""R4, checked: optimi-lab keeps no numeric kernel of its own.

R4 (plan I.1): equation solving is ONE engine, numerics-lab; optimi-lab depends on it. So no shipped
module DEFINES a line search, a trust-region step, a quasi-Newton update, a Krylov method, a
factorisation or a KKT solve -- it calls numerics-lab's. The guard reads definitions by name against
:data:`KERNEL_NAMES`; and the gradient methods must actually import numerics-lab (a floor), so a
method that quietly inlined its own search would show up on one side or the other.

R1 (which scipy submodules the runtime may import) is guarded once, by
``test_the_runtime_stays_pure.py::test_the_runtime_takes_only_the_r1_scipy_submodules``.

Every guard is one function, called on the tree and on a planted violation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

from lab_commons.dev import floors

_ROOT: Final = Path(__file__).resolve().parents[2]
_SRC: Final = _ROOT / 'src' / 'optimi_lab'

#: A definition whose name, lower-cased without underscores, contains one of these is a numerics-lab kernel.
KERNEL_NAMES: Final = (
    'armijo',
    'wolfe',
    'backtrack',
    'dogleg',
    'steihaug',
    'updateradius',
    'twoloop',
    'bfgsupdate',
    'bfgsinverse',
    'sr1update',
    'broyden',
    'conjugategradient',
    'minres',
    'gmres',
    'cholesky',
    'solvekkt',
)
_KERNEL: Final = re.compile('|'.join(KERNEL_NAMES))

#: Gradient-method and operator modules that import numerics-lab; measured 2026-10-18 at 4.
NUMERICS_IMPORTER_FLOOR: Final = 4
NUMERICS_IMPORTER_HEADROOM: Final = 6


def kernels_defined(module: ast.Module) -> list[str]:
    """Every function or class defined in *module* whose name names a numerics-lab kernel."""
    return sorted(
        node.name
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and _KERNEL.search(node.name.lower().replace('_', ''))
    )


def _imports_numerics(module: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom) and (node.module or '').startswith('numerics_lab') for node in ast.walk(module)
    )


def _shipped() -> dict[str, ast.Module]:
    files = sorted(p for p in _SRC.rglob('*.py') if p.name != '__version__.py')
    return {p.relative_to(_ROOT).as_posix(): ast.parse(p.read_text(encoding='utf-8')) for p in files}


def test_no_shipped_module_defines_a_numeric_kernel() -> None:
    offenders = {name: found for name, module in _shipped().items() if (found := kernels_defined(module))}
    assert not offenders, f'{offenders} define numeric kernels numerics-lab owns (R4); call numerics_lab instead.'


def test_the_gradient_methods_import_numerics_lab() -> None:
    importers = [name for name, module in _shipped().items() if _imports_numerics(module)]
    floors.assert_floor(len(importers), floor=NUMERICS_IMPORTER_FLOOR, what='R4 (numerics-lab importers)')
    floors.assert_floor_still_binds(
        len(importers), floor=NUMERICS_IMPORTER_FLOOR, headroom=NUMERICS_IMPORTER_HEADROOM, what='R4 (importers)'
    )


def test_planted_controls_each_guard_names_its_violation() -> None:
    planted = ast.parse(
        'from scipy.optimize import minimize\nimport scipy.stats\nfrom scipy.sparse.linalg import spsolve\n\n\n'
        'def strong_wolfe_search(phi):\n    return phi\n\n\nclass LBFGSUpdate:\n    pass\n'
    )
    assert kernels_defined(planted) == ['LBFGSUpdate', 'strong_wolfe_search']
    assert _imports_numerics(ast.parse('from numerics_lab.line_search import armijo\n'))
    assert not _imports_numerics(planted)
