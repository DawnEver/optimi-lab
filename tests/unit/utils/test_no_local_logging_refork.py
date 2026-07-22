"""Anti-drift guard: the shared logging/paths layer is the external ``lab_commons``, never
a resurrected in-tree re-implementation of its mechanics.

optimi-lab's ``utils/logger.py`` and ``utils/config.py`` are thin bindings over
``lab_commons.log`` / ``lab_commons.paths`` (see motronics-studio's
``plan-lab-commons-standalone.md``). A shared package nobody imports (or that gets
re-forked locally) is worse than none, so this is a check the CODE consults, not a
comment: it reds the moment either the generic handler-wiring mechanics reappear locally
or ``lab_commons`` stops resolving.
"""

import importlib
from pathlib import Path

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
