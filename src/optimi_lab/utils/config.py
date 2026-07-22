"""Centralized loading and parsing of configuration files, and provide a unified access interface.
Singleton pattern: ensure configuration consistency.
"""

import datetime
from pathlib import Path

from lab_commons.paths import _detect_repo_root

from .file_io import check_path, read_toml, save_toml
from .quantities import BaseModel_with_q, pydantic_config_dict_with_q_case_insensitive

__all__ = ['CONF', 'PathData', 'load_config', 'save_config']


current_time = datetime.datetime.now()

# An editable / source checkout lives under '<root>/src/optimi_lab/...'; an
# installed wheel lives under '<site-packages>/optimi_lab/...' with no 'src'
# component. Anchor writable state to the repo root in the former case and to the
# current working directory in the latter, so importing the installed package
# never raises (and never writes into site-packages).
# NOTE: pass an in-repo anchor (`start=Path(__file__)`) explicitly -- lab_commons'
# `_detect_repo_root()` defaults its upward search to ITS OWN `__file__`, which is
# site-packages once lab_commons is installed as a dependency, so it would never
# detect the optimi-lab source checkout otherwise (motronics hit the same bug first;
# see motronics-studio's plan-lab-commons-standalone.md PHASE 2).
root_path = _detect_repo_root(start=Path(__file__)) or Path.cwd()


class PathData:
    """Naming conventions and path-related attributes.

    - xx_name
        - folder name without trailing '/'
        - file name with extension
    - xx_name_base
        - file name without extension
    - xx_path
        - folder relative path with '/'
        - file relative path with extension
    - xx_path_abs
        - file absolute path
    """

    # Anchor all package data / writable paths to the package root, never the
    # current working directory: the library must import cleanly regardless of
    # where the consumer runs from (it is consumed as a dependency, not only from
    # the repo root). NOTE: a fuller redesign that drops the ``usr`` layer and
    # writes user state under a proper data dir is tracked in the motronics
    # backlog (MANUAL-20260703-176).
    root_path_abs: Path = root_path
    usr_local_path: Path = root_path / 'usr/local/'
    usr_default_path: Path = root_path / 'usr/default/'
    usr_lib_path: Path = root_path / 'usr/lib/'
    default_empty_file_path: Path = usr_default_path / 'empty.toml'

    config_file_path: Path = usr_local_path / 'config.toml'
    default_config_file_path: Path = usr_default_path / 'default.config.toml'

    """
    Log files
    """
    filename_base_current_time_sec: str = current_time.strftime(r'%H-%M-%S')
    filename_base_current_time_min: str = current_time.strftime(r'%H-%M')
    filename_base_current_time_hour: str = current_time.strftime(r'%H')
    filename_base_current_time_day: str = current_time.strftime(r'%Y-%m-%d')
    log_filename: str = f'{filename_base_current_time_hour}.log'
    log_folder_name: str = filename_base_current_time_day
    log_folder_parent_path: Path = root_path / 'output/logs/'
    log_folder_path: Path = log_folder_parent_path / log_folder_name
    case_workdir_path: Path = log_folder_path
    report_folder_path: Path = root_path / 'output/reports/'
    """
    Optimizer files
    """
    optimizer_filename: str = f'{filename_base_current_time_sec}.opt.toml'
    optimizer_file_path: Path = log_folder_path / optimizer_filename
    surrogate_model_path: Path = f'{log_folder_path}{filename_base_current_time_sec}.surrogate_model.pkl'
    intelligent_algorithm_path: Path = f'{log_folder_path}{filename_base_current_time_sec}.intelligent_algorithm_.pkl'

    """
    Figures
    """
    default_fig_name: str = f'{filename_base_current_time_sec}.png'
    default_fig_path: Path = log_folder_path / default_fig_name


# Best-effort creation of writable scratch folders and the user config. Must never
# be fatal at import time: a read-only or packaged install (which ships no default
# config outside src/) simply falls back to the in-code defaults in load_config().
def _ensure_runtime_paths() -> None:
    try:
        for dir_path in [PathData.log_folder_path, PathData.report_folder_path]:
            check_path(target_path=dir_path, is_dir=True)
        if Path(PathData.default_config_file_path).exists():
            check_path(
                target_path=PathData.config_file_path,
                default_path=PathData.default_config_file_path,
            )
    except OSError:
        pass


_ensure_runtime_paths()


class Core(BaseModel_with_q): ...


class Utils(BaseModel_with_q):
    r"""Attributes:
    log_file_format(str): Log file format.
        Use double quotes, supports "\n" and "\t".
        Refer to the standard library logging/__init__.py: Formatter
    log_console_format(str): Console log format
    log_app_format(str): Application log format
    log_date_format(str): Date format.
        A wrong format may cause an infinite loop!!!
        ```python
        ...\\Lib\\logging\\__init__.py", line 650, in formatTime
            s = time.strftime(datefmt, ct)
        ValueError: Invalid format string
        ```
    text_editor_command(str): Text editor command.
    """

    model_config = pydantic_config_dict_with_q_case_insensitive
    # logs
    log_file_format: str = '%(asctime)s %(levelname)s %(message)s \tlocation: %(filename)s line%(lineno)d'
    log_console_format: str = '%(message)s %(asctime)s'
    log_app_format: str = '%(levelname)s %(message)s %(asctime)s'
    log_date_format: str = '%m/%d/%Y %H:%M:%S'
    text_editor_command: str = 'notepad.exe'
    resend_api_key: str = ''


class Config(BaseModel_with_q):
    """Global configuration file."""

    core: Core
    utils: Utils


def load_config(config_file_path: Path = PathData.config_file_path) -> Config:
    """Args:
        config_file_path(Path): Path to the configuration file
    Returns:
        Config: Parsed configuration object
    Raises:
        ValidationError: If parsing the configuration fails.

    Falls back to the in-code defaults when the config file is absent (e.g. a
    packaged install that ships no user config), so importing the library never
    requires a writable/pre-populated config directory.
    """
    try:
        config_data = read_toml(config_file_path)
    except (FileNotFoundError, OSError):
        config_data = {}
    config_data.setdefault('core', {})
    config_data.setdefault('utils', {})
    return Config.model_validate(config_data)


def save_config(config: Config, config_file_path: Path | None = None) -> None:
    """Args:
    config(Config): Configuration object
    config_file_path(Path): Path to save the configuration file.
    """
    if config_file_path is None:
        check_path(PathData.config_file_path, PathData.default_config_file_path)
        config_file_path = PathData.config_file_path

    dict_data = config.model_dump(exclude_defaults=True)
    save_toml(file_path=config_file_path, dict_data=dict_data)


CONF = load_config()
