"""Importing optimi-lab must not touch the cwd, ``sys.modules`` or the filesystem.

The defect this pins: ``optimi_lab/utils/__init__.py`` used to execute ``CONF`` and
``add_handle()`` at import, so merely importing the package created the per-run output
directory *and* attached a file + console handler pair that also rebound lab_commons'
process-wide active logger -- every ``log()`` call of every library in the process landed
in optimi-lab's handlers from then on.

The hazard is PLANTED rather than described: each probe runs in a fresh interpreter whose
cwd is an empty temporary directory and whose ``OPTIMI_LAB_HOME`` is cleared, so a
regression is visible as a new entry in that directory, a new directory under the
checkout's ``output/``, or a module that the import pulled in. Re-adding the removed lines
to ``__init__.py`` reds all of it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / 'src'

_PROBE = '''"""Fresh-interpreter probe: report what importing the package did. Prints JSON."""
import importlib
import json
import logging
import os
import sys
from pathlib import Path


def tree(root: Path) -> list:
    """Relative path of every entry under `root` (empty list when it does not exist)."""
    if not root.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*'))


# A host that already uses lab_commons' logging, to observe the process-wide redirect.
lab_log = importlib.import_module('lab_commons.log')

cwd = Path.cwd()
repo = Path(os.environ['OPTIMI_LAB_REPO'])
output_root = repo / 'output'

modules_before = set(sys.modules)
cwd_before = tree(cwd)
output_before = tree(output_root)
active_before = lab_log._active_logger.name

import optimi_lab.utils  # noqa: E402

print(
    json.dumps(
        {
            'cwd_before': cwd_before,
            'cwd_after': tree(cwd),
            'output_before': output_before,
            'output_after': tree(output_root),
            'new_modules': sorted(set(sys.modules) - modules_before),
            'handlers': [type(h).__name__ for h in logging.getLogger('optimi_lab').handlers],
            'active_before': active_before,
            'active_after': lab_log._active_logger.name,
        }
    )
)
'''


@pytest.fixture
def probe(tmp_path: Path) -> dict:
    """Run the probe in a fresh interpreter from an empty cwd; return its JSON report."""
    script = tmp_path / 'probe_import_side_effects.py'
    script.write_text(_PROBE, encoding='utf-8')
    run_dir = tmp_path / 'cwd'
    run_dir.mkdir()

    env = dict(os.environ)
    env['PYTHONPATH'] = str(_SRC)
    env['OPTIMI_LAB_REPO'] = str(_REPO_ROOT)
    env.pop('OPTIMI_LAB_HOME', None)  # keep output-root resolution honest

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    report['stderr'] = completed.stderr
    report['run_dir'] = str(run_dir)
    return report


def test_import_leaves_the_cwd_and_the_filesystem_unchanged(probe: dict) -> None:
    """No directory appears in the importing process's cwd, and none under ``output/``."""
    assert probe['cwd_before'] == []
    assert probe['cwd_after'] == [], (
        f'importing optimi_lab wrote into the caller\'s cwd: {probe["cwd_after"]} -- a path must never be '
        'resolved from Path.cwd()'
    )
    assert probe['output_after'] == probe['output_before'], (
        f'importing optimi_lab created {set(probe["output_after"]) - set(probe["output_before"])} under the '
        'checkout output root -- the per-run directory must not be resolved at import'
    )
    assert 'Failed to open file' not in probe['stderr'], (
        f'importing optimi_lab logged a config read failure: {probe["stderr"]}'
    )


def test_import_attaches_no_handler(probe: dict) -> None:
    """The ``'optimi_lab'`` logger carries no handler until ``add_handle()`` is asked for."""
    assert probe['handlers'] == [], (
        f'importing optimi_lab attached {probe["handlers"]} -- handler attachment is add_handle()\'s job, on request'
    )


def test_import_does_not_redirect_the_process_logger(probe: dict) -> None:
    """lab_commons' active logger is the host's before and after -- importing never rebinds it."""
    assert probe['active_before'] != 'optimi_lab', 'the probe did not plant a host binding'
    assert probe['active_after'] == probe['active_before'], (
        f'importing optimi_lab rebound the process-wide active logger '
        f'{probe["active_before"]!r} -> {probe["active_after"]!r}'
    )


def test_import_pulls_in_nothing_beyond_the_package(probe: dict) -> None:
    """``sys.modules`` gains the two package modules and nothing else.

    Both ``__init__.py`` files are inert, so an eager ``from .config import CONF`` (or any
    other import) is what this reds on -- it drags config, lab_commons, pint and pydantic
    into the host process before the host asks for any of them.
    """
    assert probe['new_modules'] == ['optimi_lab', 'optimi_lab.utils'], (
        f'importing optimi_lab.utils added {probe["new_modules"]} to sys.modules'
    )


_WRITE_PROBE = '''"""Fresh-interpreter probe: report what importing <OPTIMI_LAB_MODULE> wrote."""
import importlib
import json
import logging
import os
from pathlib import Path

lab_log = importlib.import_module('lab_commons.log')

cwd = Path.cwd()
output_root = Path(os.environ['OPTIMI_LAB_REPO']) / 'output'


def tree(root: Path) -> list:
    """Relative path of every entry under `root` (empty list when it does not exist)."""
    if not root.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*'))


cwd_before = tree(cwd)
output_before = tree(output_root)
active_before = lab_log._active_logger.name

importlib.import_module(os.environ['OPTIMI_LAB_MODULE'])

print(
    json.dumps(
        {
            'module': os.environ['OPTIMI_LAB_MODULE'],
            'cwd_before': cwd_before,
            'cwd_after': tree(cwd),
            'output_before': output_before,
            'output_after': tree(output_root),
            'handlers': [type(h).__name__ for h in logging.getLogger('optimi_lab').handlers],
            'active_before': active_before,
            'active_after': lab_log._active_logger.name,
        }
    )
)
'''


@pytest.fixture(params=['optimi_lab.optimizer', 'optimi_lab.plot', 'optimi_lab.sensitivity_analysis'])
def write_probe(request: pytest.FixtureRequest, tmp_path: Path) -> dict:
    """Import a heavy submodule in a fresh interpreter and report what it wrote.

    These are the modules a consumer actually imports, and they carry the defaults that a
    run-scoped path is easy to hide in: a function's DEFAULT ARGUMENT is evaluated when the
    module is imported, not when the function is called, so `fig_path: str =
    PathData.default_fig_path` created the per-run output directory during the import -- for
    every consumer, before a single figure was asked for.
    """
    script = tmp_path / 'probe_write.py'
    script.write_text(_WRITE_PROBE, encoding='utf-8')
    run_dir = tmp_path / 'cwd'
    run_dir.mkdir()

    env = dict(os.environ)
    env['PYTHONPATH'] = str(_SRC)
    env['OPTIMI_LAB_REPO'] = str(_REPO_ROOT)
    env['OPTIMI_LAB_MODULE'] = request.param
    env.pop('OPTIMI_LAB_HOME', None)

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    report['stderr'] = completed.stderr
    return report


def test_importing_a_submodule_writes_nothing(write_probe: dict) -> None:
    """No module's import creates the per-run output directory (or anything in the cwd)."""
    assert write_probe['cwd_before'] == []
    assert write_probe['cwd_after'] == [], (
        f'importing {write_probe["module"]} wrote into the caller\'s cwd: {write_probe["cwd_after"]}'
    )
    created = sorted(set(write_probe['output_after']) - set(write_probe['output_before']))
    assert created == [], (
        f'importing {write_probe["module"]} created {created} under the output root -- a default argument is '
        'evaluated at import, so a run-scoped path must be resolved inside the call instead'
    )
    assert 'Failed to open file' not in write_probe['stderr'], (
        f'importing {write_probe["module"]} logged a config read failure: {write_probe["stderr"]}'
    )


def test_importing_a_submodule_attaches_no_handler(write_probe: dict) -> None:
    """No module's import attaches the handler pair or rebinds the process logger."""
    assert write_probe['handlers'] == [], (
        f'importing {write_probe["module"]} attached {write_probe["handlers"]}'
    )
    assert write_probe['active_before'] != 'optimi_lab', 'the probe did not plant a host binding'
    assert write_probe['active_after'] == write_probe['active_before'], (
        f'importing {write_probe["module"]} rebound the process-wide active logger '
        f'{write_probe["active_before"]!r} -> {write_probe["active_after"]!r}'
    )
