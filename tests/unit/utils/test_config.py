import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from optimi_lab.utils import config
from optimi_lab.utils.config import Config, PathData, load_config, save_config
from optimi_lab.utils.file_io import read_toml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / 'src'

# A user config file as the loader consumes it: the block names the model's fields, with a
# value that DIFFERS from the in-code default so the read is visible in the result.
_CONFIG_TOML = """[core]

[utils]
log_file_format = "%(asctime)s %(levelname)s %(message)s"
log_console_format = "%(message)s %(asctime)s"
log_app_format = "%(levelname)s %(message)s %(asctime)s"
log_date_format = "%m/%d/%Y %H:%M:%S"
text_editor_command = "vim"
"""


@pytest.fixture(autouse=True)
def config_paths_are_left_where_they_were_found():
    """No test in this module may repoint the config paths for the rest of the session.

    ``PathData.config_file_path`` is a module global, so a test that assigns to it directly -- as
    the ``save_config`` test here used to -- leaves every LATER test and any in-process consumer
    resolving its config from a pytest temp directory. Patch through ``mocker.patch.object`` and
    this fixture confirms the undo happened. Nothing else in the suite reads that path in-process
    today, which is why the leak was silent rather than a red.
    """
    before = PathData.config_file_path, PathData.default_config_file_path

    yield

    assert (PathData.config_file_path, PathData.default_config_file_path) == before, (
        'this test left a patched config path behind'
    )


def test_load_config_valid_file(tmp_path: Path):
    """A config file that IS present is read, and its values land on the model."""
    config_file_path = tmp_path / 'config.toml'
    config_file_path.write_text(_CONFIG_TOML, encoding='utf-8')

    config_obj = load_config(config_file_path)

    assert isinstance(config_obj, Config)
    assert config_obj.utils.text_editor_command == 'vim'
    assert config_obj.utils.log_file_format == '%(asctime)s %(levelname)s %(message)s'


def test_load_config_falls_back_to_defaults_when_the_file_is_absent(tmp_path: Path):
    """An absent config file is the normal case: in-code defaults, and it is not an error.

    ``read_toml`` logs its failure at ERROR before raising, so a reader wrapped in
    try/except printed 'Failed to open file ...' for every optional-and-missing file.
    """
    config_obj = load_config(tmp_path / 'absent.toml')

    assert isinstance(config_obj, Config)
    assert config_obj.utils.text_editor_command == 'notepad.exe'


@pytest.fixture
def config_obj() -> Config:
    """A ``Config`` whose fields make the write visible in the file it produces.

    ``log_file_format`` and ``text_editor_command`` DIFFER from the model defaults, so they must
    reach the file; ``log_console_format`` / ``log_app_format`` / ``log_date_format`` are set to
    the values the model already defaults to, so they must NOT -- ``load_config`` refills them on
    the next read.
    """
    return Config(
        core={},
        utils={
            'log_file_format': '%(asctime)s %(levelname)s %(message)s',
            'log_console_format': '%(message)s %(asctime)s',
            'log_app_format': '%(levelname)s %(message)s %(asctime)s',
            'log_date_format': '%m/%d/%Y %H:%M:%S',
            'text_editor_command': '',
        },
    )


# What `save_config` must put on disk for `config_obj`: the fields differing from the model
# defaults. Written out by hand rather than derived from `model_dump`, so that the assertions
# below cannot agree with the code by construction -- they did, while this was a no-op test:
# `save_toml` was mocked, so the function under test wrote nothing at all, and the file the
# assertions saw held only the DEFAULT CONFIG that `check_path` had seeded.
_SAVED_FIELDS = {
    'core': {},
    'utils': {
        'log_file_format': '%(asctime)s %(levelname)s %(message)s',
        'text_editor_command': '',
    },
}


def test_save_config_writes_the_non_default_fields_to_the_path_it_is_given(config_obj: Config, tmp_path: Path):
    """An explicit path is written directly, and the module global is left alone.

    The parent directory does not exist yet, so this also pins that the write creates it.
    """
    config_file_path = tmp_path / 'nested' / 'config.toml'
    configured_path_before = PathData.config_file_path

    save_config(config_obj, config_file_path)

    assert read_toml(config_file_path) == _SAVED_FIELDS
    assert PathData.config_file_path == configured_path_before, 'an explicit path must not repoint the global'


def test_save_config_without_a_path_writes_to_the_configured_path(config_obj: Config, mocker, tmp_path: Path):
    """``config_file_path=None`` resolves ``PathData.config_file_path`` AT CALL TIME.

    The paths are patched through ``mocker.patch.object`` rather than assigned, so the module
    global is left exactly as it was found: the bare assignment this replaces leaked a temp path
    into every later test in the session.
    """
    config_file_path = tmp_path / 'usr' / 'local' / 'config.toml'
    default_config_file_path = tmp_path / 'usr' / 'default' / 'default.config.toml'
    default_config_file_path.parent.mkdir(parents=True)
    default_config_file_path.write_text(
        "[core]\n\n[utils]\nresend_api_key='your resend api key here'\ntext_editor_command='start'\n",
        encoding='utf-8',
    )
    mocker.patch.object(PathData, 'config_file_path', config_file_path)
    mocker.patch.object(PathData, 'default_config_file_path', default_config_file_path)

    save_config(config_obj)

    assert read_toml(config_file_path) == _SAVED_FIELDS, 'the seeded default must be overwritten, not left behind'


def test_utils_field_validator():
    """Test the field validator for text_editor_command."""
    utils = config.Utils(
        log_file_format='%(asctime)s %(levelname)s %(message)s',
        log_console_format='%(message)s %(asctime)s',
        log_app_format='%(levelname)s %(message)s %(asctime)s',
        log_date_format='%m/%d/%Y %H:%M:%S',
    )
    assert utils.text_editor_command == 'notepad.exe'


_PROBE = '''"""Fresh-interpreter probe: resolve the config from a directory that is not a repo."""
import json
from pathlib import Path

from optimi_lab.utils.config import PathData

cwd = Path.cwd()
print(
    json.dumps(
        {
            'cwd': str(cwd),
            'cwd_tree': sorted(p.relative_to(cwd).as_posix() for p in cwd.rglob('*')),
            'root_path_abs': str(PathData.root_path_abs),
            'config_file_path': str(PathData.config_file_path),
            'usr_local_path': str(PathData.usr_local_path),
        }
    )
)
'''


@pytest.fixture(params=['checkout', 'wheel'])
def probe(request: pytest.FixtureRequest, tmp_path: Path) -> dict:
    """Import the config module in a fresh interpreter whose cwd is an empty temp dir.

    Two layouts, because they resolve differently and only one of them can see the
    checkout:

    - ``checkout``: ``PYTHONPATH`` is the repo's ``src/``, so ``_detect_repo_root`` finds the
      checkout and the config root must be the repo root;
    - ``wheel``: the package is copied to a bare temp directory with NO enclosing ``src/`` +
      ``pyproject.toml`` -- the site-packages layout an installed consumer has. Detection
      fails there, which is the case that used to fall back to ``Path.cwd()`` and made the
      consumer log a config read failure from ITS repo root on every interpreter start.

    Either way the cwd is an empty directory, so a cwd-derived path is visible as a path
    pointing into it.
    """
    if request.param == 'checkout':
        python_path = _SRC
        expected_root = _REPO_ROOT
    else:
        site_packages = tmp_path / 'site-packages'
        shutil.copytree(
            _SRC / 'optimi_lab',
            site_packages / 'optimi_lab',
            ignore=shutil.ignore_patterns('__pycache__'),
        )
        python_path = site_packages
        expected_root = site_packages / 'optimi_lab'

    script = tmp_path / 'probe_config_root.py'
    script.write_text(_PROBE, encoding='utf-8')
    run_dir = tmp_path / 'cwd'
    run_dir.mkdir()

    env = dict(os.environ)
    env['PYTHONPATH'] = str(python_path)
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
    report['layout'] = request.param
    report['expected_root'] = str(expected_root)
    report['run_dir'] = str(run_dir)
    return report


def test_config_root_never_comes_from_the_callers_cwd(probe: dict) -> None:
    """The config root is the checkout's root, or the package's own directory in a wheel.

    The hazard is planted: the probe's cwd is an EMPTY temporary directory, and the config
    root is asserted against a path that has nothing to do with it.
    """
    assert probe['cwd'] == probe['run_dir']
    assert probe['root_path_abs'] == probe['expected_root'], (
        f'[{probe["layout"]}] config root resolved to {probe["root_path_abs"]} while running in '
        f'{probe["cwd"]} -- a config path must not be a function of the caller\'s cwd'
    )
    assert Path(probe['config_file_path']) == Path(probe['expected_root']) / 'usr' / 'local' / 'config.toml'
    assert Path(probe['usr_local_path']) == Path(probe['expected_root']) / 'usr' / 'local'


def test_config_lookup_writes_nothing_under_the_callers_cwd(probe: dict) -> None:
    """Neither the probe's cwd nor anything in it is created or read."""
    assert probe['cwd_tree'] == [], f'the config lookup created {probe["cwd_tree"]} in the caller\'s cwd'


def test_absent_config_is_not_reported_as_an_error(probe: dict) -> None:
    """No 'Failed to open file' line for the optional config file nobody wrote.

    The default config path is absent in BOTH layouts, which is why the reader must test for
    absence instead of catching the reader's OSError -- that OSError is logged at ERROR
    first, so catching it still printed the line.
    """
    assert 'Failed to open file' not in probe['stderr'], (
        f'[{probe["layout"]}] importing the config module logged a read failure: {probe["stderr"]}'
    )
