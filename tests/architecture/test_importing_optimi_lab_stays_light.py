"""`import optimi_lab` loads a bounded module set, and none of the heavy ones.

The cost of importing this package is paid by every downstream process, including the ones that only
declare a problem. Measured 2026-11-26 in a downstream venv (cold, best of 3): `import optimi_lab` 9.7 s,
because `optimi_lab.solver` imported every solver and the model-based ones pulled
`learn_lab.models.response` (scipy.interpolate, scipy.optimize). Those solvers now import their
surrogate models where they build them (each site carries `noqa: PLC0415 -- heavy-optional` and is
pinned in `test_the_public_surface_is_declared.py::LAZY_IMPORT_PINS`).

WHY A MODULE SET AND NOT A CLOCK: wall time on a shared host varies by 3x between runs of the same
import, so a time cap is either loose enough to say nothing or flaky. The module set is exact. Two
guards, each one function called on the live import and on a planted violation:

* :data:`HEAVY` -- modules `import optimi_lab` must never load (a plotting stack, or the surrogate
  zoo with its scipy halves);
* :data:`MODULE_CEILING` -- the number of modules in ``sys.modules`` after `import optimi_lab`
  (measured 728 on 2026-11-26), so a new eager import of anything large shows up by count.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Final

#: Modules `import optimi_lab` must not load: optimi_lab's plotting stacks and learn_lab's model zoo.
HEAVY: Final = frozenset({
    'bokeh',
    'lab_commons.viz',
    'learn_lab.models.response',
    'matplotlib',
    'pandas',
    'scipy.interpolate',
    'scipy.optimize',
    'scipy.stats',
    'sympy',
})

#: `len(sys.modules)` after `import optimi_lab`: measured 728 on 2026-11-26 (893 before the deferral).
MODULE_CEILING: Final = 780


def loaded_by(statement: str) -> frozenset[str]:
    """The modules a fresh interpreter holds after running *statement*."""
    code = f'import json, sys\n{statement}\nprint(json.dumps(sorted(sys.modules)))'
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, timeout=600)
    return frozenset(json.loads(done.stdout.strip().splitlines()[-1]))


def heavy_in(modules: frozenset[str]) -> list[str]:
    """The :data:`HEAVY` modules (or their submodules) among *modules*."""
    return sorted(m for m in modules if any(m == h or m.startswith(f'{h}.') for h in HEAVY))


def over_ceiling(modules: frozenset[str], ceiling: int = MODULE_CEILING) -> int:
    """How many modules past *ceiling* the import loaded; 0 when within it."""
    return max(0, len(modules) - ceiling)


def test_importing_optimi_lab_loads_no_heavy_module_and_stays_under_the_ceiling() -> None:
    modules = loaded_by('import optimi_lab')
    assert 'optimi_lab.solver' in modules, 'the probe did not import optimi_lab at all'
    assert not heavy_in(modules), (
        f'`import optimi_lab` loads {heavy_in(modules)}: import it where it is used, as the model-based '
        f'solvers do (noqa: PLC0415 -- heavy-optional, pinned in LAZY_IMPORT_PINS).'
    )
    assert not over_ceiling(modules), (
        f'`import optimi_lab` loads {len(modules)} modules, past the ceiling of {MODULE_CEILING}: find the '
        f'new eager import with `python -X importtime -c "import optimi_lab"`.'
    )


def test_planted_controls_each_guard_names_its_violation() -> None:
    """PLANT a heavy import beside optimi_lab and call the SAME functions the live test calls."""
    planted = loaded_by('import optimi_lab\nimport scipy.interpolate')
    assert 'scipy.interpolate' in heavy_in(planted)
    assert over_ceiling(planted, ceiling=len(planted) - 1) == 1
    assert not over_ceiling(planted, ceiling=len(planted))
