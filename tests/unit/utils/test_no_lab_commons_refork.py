"""Anti-drift guard: the shared infra layer is the external ``lab_commons``, never a
resurrected in-tree re-implementation of its mechanics.

optimi-lab's ``utils/{logger,config,file_io,exceptions,quantities,multiprocessing}.py`` are all
thin bindings over ``lab_commons.{log,paths,file_io,exceptions,units,em,multiprocess}`` (see
motronics-studio's ``plan-lab-commons-standalone.md`` — full-set migration, 2026-07-22). A shared
package nobody imports (or that gets re-forked locally) is worse than none, so this is a check the
CODE consults, not a comment: it reds the moment either the generic mechanics reappear locally or
``lab_commons`` stops resolving.
"""

import importlib
from pathlib import Path

import lab_commons.em as _em

from optimi_lab.utils import quantities
from optimi_lab.utils.exceptions import ParameterException, QuantityException
from optimi_lab.utils.file_io import list_files_in_dir, read_toml
from optimi_lab.utils.multiprocessing import MultiProcessing, MultiProcessingParameters

_SRC = Path(__file__).resolve().parents[3] / 'src'


def test_lab_commons_is_importable() -> None:
    """``lab_commons`` (the git-URL dependency) resolves -- the swap is wired, not just declared.

    Use :func:`importlib.import_module` for the submodules: ``lab_commons.__init__``
    re-exports a ``log`` FUNCTION, which shadows the ``lab_commons.log`` submodule under
    plain attribute access.
    """
    lc_log = importlib.import_module('lab_commons.log')
    lc_paths = importlib.import_module('lab_commons.paths')

    assert callable(lc_log.get_logger)
    assert callable(lc_paths.run_output_dir)


def test_logger_binds_lab_commons_not_a_local_fork() -> None:
    """``optimi_lab.utils.logger`` imports its mechanics from ``lab_commons.log``.

    A local re-implementation of ``get_logger``/``add_handle``/``log``/``timer`` bodies
    (the fork this migration retired) would not import them at all.
    """
    logger_path = _SRC / 'optimi_lab' / 'utils' / 'logger.py'
    text = logger_path.read_text(encoding='utf-8')
    assert 'from lab_commons.log import' in text, (
        f'{logger_path} no longer binds lab_commons.log -- the local logging mechanics were re-forked.'
    )


def test_config_anchors_repo_root_via_lab_commons() -> None:
    """``optimi_lab.utils.config`` resolves its repo root through ``lab_commons.paths``.

    Must pass its own ``start=Path(__file__)`` anchor (the extraction bug motronics hit
    first): ``lab_commons``' own default anchor is site-packages once it is installed as a
    dependency, so a consumer that calls the bare function never detects its own checkout.
    """
    config_path = _SRC / 'optimi_lab' / 'utils' / 'config.py'
    text = config_path.read_text(encoding='utf-8')
    assert 'from lab_commons.paths import' in text, (
        f'{config_path} no longer resolves its repo root via lab_commons.paths -- re-forked locally.'
    )
    assert 'start=Path(__file__)' in text, (
        f'{config_path} must anchor `_detect_repo_root(start=Path(__file__))` on its own file, '
        'not lab_commons defaults (which resolve to site-packages once installed).'
    )


def test_file_io_and_generic_exceptions_come_from_lab_commons() -> None:
    """`optimi_lab.core.utils.{file_io,exceptions}` are thin re-export shims (plan §3); if someone
    re-implements read_toml / ParameterException locally, its __module__ stops being lab_commons and
    this reds -- the check the code consults so the fork cannot silently return.
    """
    assert read_toml.__module__ == 'lab_commons.file_io'
    assert list_files_in_dir.__module__ == 'lab_commons.file_io'
    assert ParameterException.__module__ == 'lab_commons.exceptions'
    assert QuantityException.__module__ == 'lab_commons.exceptions'


def test_units_come_from_lab_commons() -> None:
    """`optimi_lab.utils.quantities` re-exports tier-1 (`lab_commons.units`) + tier-2
    (`lab_commons.em`) and DEFINES no units of its own -- optimi-lab's own ``PydanticQuantity``
    core-schema design is retired. If someone re-defines get_quantity_type or a *Type here, the
    identity check breaks and this reds.
    """
    assert quantities.get_quantity_type.__module__ == 'lab_commons.units'
    assert quantities.BaseModel_with_q.__module__ == 'lab_commons.units'
    # the shared EM types/constants are the SAME objects lab_commons.em defines (not re-created)
    assert quantities.TorqueType is _em.TorqueType
    assert quantities.Q_0Nm is _em.Q_0Nm


def test_multiprocess_comes_from_lab_commons() -> None:
    """The parallel-run helper must resolve to `lab_commons.multiprocess`, not an optimi-lab re-fork."""
    assert MultiProcessing.__module__ == 'lab_commons.multiprocess'
    assert MultiProcessingParameters.__module__ == 'lab_commons.multiprocess'
