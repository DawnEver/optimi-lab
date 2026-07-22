"""Centralized loading and parsing of configuration files, and provide a unified access interface.
Singleton pattern: ensure configuration consistency.
"""

from pathlib import Path

from lab_commons.paths import _detect_repo_root, run_output_dir, run_stamp

from .file_io import check_path, read_toml, save_toml
from .quantities import BaseModel_with_q, pydantic_config_dict_with_q_case_insensitive

__all__ = ['CONF', 'PathData', 'load_config', 'save_config']


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


class classproperty:
    """Descriptor that makes a method accessible as a class-level property.

    ``PathData.log_folder_path`` reads like a class attribute but computes lazily
    on first access, delegating to ``lab_commons.paths`` for the per-run stamp.
    """

    def __init__(self, fget) -> None:
        self.fget = fget

    def __get__(self, obj, owner=None) -> object:
        return self.fget(owner)


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

    Time-dependent paths delegate to ``lab_commons.paths`` (``run_date()`` /
    ``run_stamp()`` / ``run_output_dir()``) so every artifact of a single process
    lands in the same per-run folder without manual datetime formatting. All such
    attributes are lazy ``classproperty`` descriptors -- computed on first access
    and stable for the rest of the process.
    """

    # --- Static paths (computed once at class-definition time) ---

    root_path_abs: Path = root_path
    usr_local_path: Path = root_path / 'usr/local/'
    usr_default_path: Path = root_path / 'usr/default/'
    usr_lib_path: Path = root_path / 'usr/lib/'
    default_empty_file_path: Path = usr_default_path / 'empty.toml'

    config_file_path: Path = usr_local_path / 'config.toml'
    default_config_file_path: Path = usr_default_path / 'default.config.toml'

    report_folder_path: Path = root_path / 'output/reports/'

    # --- Lazy, per-run time-dependent paths ---
    # Delegated to lab_commons.paths; each property is computed on FIRST ACCESS
    # and stable for the rest of the process (run_date/run_stamp memoize internally).

    @classproperty
    def log_folder_path(cls) -> Path:
        """Per-run log output directory, created on first access.

        Delegates to ``lab_commons.paths.run_output_dir()`` so the directory is
        created lazily (no import-time filesystem write) and the path is memoized
        per process via the shared ``run_date()`` / ``run_stamp()``.
        """
        return run_output_dir('optimi-lab', root=cls.root_path_abs / 'output')

    @classproperty
    def case_workdir_path(cls) -> Path:
        """Convenience alias -- same as :attr:`log_folder_path`."""
        return cls.log_folder_path

    @classproperty
    def log_filename(cls) -> str:
        """Per-run log file name ``<HH-MM-SS>.log`` (via ``run_stamp()``)."""
        return f'{run_stamp()}.log'

    @classproperty
    def optimizer_file_path(cls) -> Path:
        """Per-run optimizer state path ``<log_folder>/<HH-MM-SS>.opt.toml``."""
        return cls.log_folder_path / f'{run_stamp()}.opt.toml'

    @classproperty
    def surrogate_model_path(cls) -> Path:
        """Per-run surrogate model path ``<log_folder>/<HH-MM-SS>.surrogate_model.pkl``."""
        return cls.log_folder_path / f'{run_stamp()}.surrogate_model.pkl'

    @classproperty
    def intelligent_algorithm_path(cls) -> Path:
        """Per-run intelligent algorithm path ``<log_folder>/<HH-MM-SS>.intelligent_algorithm_.pkl``."""
        return cls.log_folder_path / f'{run_stamp()}.intelligent_algorithm_.pkl'

    @classproperty
    def default_fig_name(cls) -> str:
        """Per-run default figure name ``<HH-MM-SS>.png`` (via ``run_stamp()``)."""
        return f'{run_stamp()}.png'

    @classproperty
    def default_fig_path(cls) -> Path:
        """Per-run default figure path ``<log_folder>/<HH-MM-SS>.png``."""
        return cls.log_folder_path / cls.default_fig_name


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
