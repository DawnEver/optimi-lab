"""D28 -- the II.7 layers point one way: a module imports only its own layer and the ones below.

    core -> problem (L1) -> oracle (L2) -> archive (L3) -> metric (L4) -> operator (L5)
         -> model (L6) -> solver (L7) -> policy (L8) -> run (L9)

``core`` (refusals, dominance, point sets) sits under everything. ``analysis`` (sensitivity,
decision, conflict) reads finished samples and is a metric (L4). The package ``__init__`` is the facade over all of them and is the one module
exempt.

THE GUARD IS ONE FUNCTION, CALLED TWICE: :func:`against_the_arrow` runs on the real import graph and
on a planted upward edge, so a green here is a reading, not an echo. A subpackage the table does not
name is refused: a new layer is a decision, made in this table.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from dev_lab.dev import floors

_ROOT: Final = Path(__file__).resolve().parents[2]
_SRC: Final = _ROOT / 'src' / 'optimi_lab'

#: Subpackage (or top-level module) -> layer. The package facade and the version file are exempt.
LAYERS: Final[dict[str, int]] = {
    'core': 0,
    'problem': 1,
    'oracle': 2,
    'archive': 3,
    'metric': 4,
    'analysis': 4,
    'operator': 5,
    'model': 6,
    'solver': 7,
    'policy': 8,
    'run': 9,
}
EXEMPT: Final = frozenset({'__init__', '__version__'})

#: Shipped modules the walk must reach; re-measured 2026-10-17 at 43 with headroom 12.
#: RE-TAKEN 2026-10-18 by P12 (population methods): 63 shipped modules.
#: RE-TAKEN 2026-10-18 AT 75: P10 (surrogates, optimi_lab.model) read 84.
#: RE-TAKEN 2026-10-18 AT 92: P11 (model-based solvers, D59-D68) read 101.
#: RE-TAKEN 2026-10-19 AT 105: P14 driver (oracle.coefficients, solver.structured) read 114.
MODULE_FLOOR: Final = 105
MODULE_HEADROOM: Final = 20


def _part(module: str) -> str:
    """The layer key of a dotted ``optimi_lab.*`` module name."""
    parts = module.split('.')
    return parts[1] if len(parts) > 1 else '__init__'


def against_the_arrow(edges: list[tuple[str, str]]) -> list[str]:
    """Every ``importer -> imported`` edge that points UP the layers, or touches an undeclared part."""
    offences = []
    for importer, imported in edges:
        a, b = _part(importer), _part(imported)
        if a in EXEMPT or b == '__version__':
            continue
        if a not in LAYERS or b not in LAYERS:
            offences.append(
                f'{importer} -> {imported}: {sorted({a, b} - set(LAYERS) - EXEMPT)} is not a declared layer'
            )
        elif LAYERS[b] > LAYERS[a]:
            offences.append(f'{importer} (L{LAYERS[a]}) -> {imported} (L{LAYERS[b]}) points up the layers')
    return offences


def _edges() -> tuple[int, list[tuple[str, str]]]:
    files = sorted(p for p in _SRC.rglob('*.py') if p.name != '__version__.py')
    edges = []
    for path in files:
        importer = '.'.join(('optimi_lab', *path.relative_to(_SRC).with_suffix('').parts))
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            edges.extend((importer, n) for n in names if n.split('.')[0] == 'optimi_lab')
    return len(files), edges


def test_the_import_graph_points_down_the_layers() -> None:
    count, edges = _edges()
    floors.assert_floor(count, floor=MODULE_FLOOR, what='LAYERING (shipped modules)')
    floors.assert_floor_still_binds(
        count, floor=MODULE_FLOOR, headroom=MODULE_HEADROOM, what='LAYERING (shipped modules)'
    )
    assert not against_the_arrow(edges)


def test_the_guard_names_a_planted_upward_edge_and_a_stranger() -> None:
    planted = [
        ('optimi_lab.problem.space', 'optimi_lab.run.loop'),
        ('optimi_lab.oracle.result', 'optimi_lab.archive'),
        ('optimi_lab.gadgets.x', 'optimi_lab.core.errors'),
        ('optimi_lab.run.loop', 'optimi_lab.problem'),
    ]
    found = against_the_arrow(planted)
    assert len(found) == 3
    assert 'points up' in found[0] and 'points up' in found[1] and "['gadgets']" in found[2]
