import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from optimi_lab.utils import config
from optimi_lab.utils.config import Config, PathData, load_config, save_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / 'src'


def test_load_config_valid_file(tmp_path: Path):
    """A config file that IS present is read, and its values land on the model."""
    config_file_path = tmp_path / 'config.toml'
    config_file_path.write_text(
        '\n'.join(
            [
                '[core]',
                '',
                '[utils]',
                'log_file_format = "%(asctime)s %(levelname)s %(message)s"',
                'log_console_format = "%(message)s %(asctime)s"',
                'log_app_format = "%(levelname)s %(message)s %(asctime)s"',
                'log_date_format = "%m/%d/%Y %H:%M:%S"',
                'text_editor_command = "vim"',
            ]
        ),
        encoding='utf-8',
    )

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


def test_save_config(mocker, tmp_path):
    """Test saving a configuration file."""
    mocker.patch('optimi_lab.utils.config.save_toml')
    config_obj = Config(
        core={},
        utils={
            'log_file_format': '%(asctime)s %(levelname)s %(message)s',
            'log_console_format': '%(message)s %(asctime)s',
            'log_app_format': '%(levelname)s %(message)s %(asctime)s',
            'log_date_format': '%m/%d/%Y %H:%M:%S',
            'text_editor_command': '',
        },
    )
    config_file_path = tmp_path / 'config.toml'
    save_config(config_obj, config_file_path)
    config.save_toml(file_path=config_file_path, dict_data=config_obj.model_dump(exclude_defaults=True))

    PathData.config_file_path = config_file_path
    save_config(config_obj)
    config.save_toml(file_path=config_file_path, dict_data=config_obj.model_dump(exclude_defaults=True))


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
