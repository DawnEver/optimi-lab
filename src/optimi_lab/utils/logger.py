"""optimi-lab logging binding.

Thin optimi-lab-specific layer over the shared ``lab_commons`` primitives: it binds the
named ``'optimi_lab'`` logger and wires the file/console handlers from optimi-lab's
config + PathData. The generic mechanics (named logger, handler pair, log/timer helpers)
live in ``lab_commons.log``; only the domain wiring (warning filters, config-driven handler
setup) stays here.

IMPORTING THIS MODULE ATTACHES NOTHING. Creating the named logger is free, but the handler
pair and the binding of that logger as lab_commons' process-wide active sink -- the two
things that redirect every ``log()`` call in the process -- happen only in
:func:`add_handle`, which a caller must ask for. See ``optimi_lab/utils/__init__.py``.
"""

import logging
import warnings

from lab_commons.log import (
    add_handle as _shared_add_handle,
)
from lab_commons.log import (
    get_logger,
    log,
    log_decorator,
    set_active_logger,
    timer,
)
from pint import UnitStrippedWarning as pint_UnitStrippedWarning

__all__ = ['add_handle', 'bind_active_logger', 'log', 'log_decorator', 'timer']

for warning in [
    # unitstrippedwarning: the unit of the quantity is stripped when downcasting to ndarray.
    pint_UnitStrippedWarning,
]:
    warnings.filterwarnings('ignore', category=warning)

# Dedicated named logger so handlers never collide with the root logger (or any other
# library sharing the process, e.g. lab-commons' other consumers). NOT bound as the
# active sink here: that is a process-wide redirect and belongs to add_handle().
logger = get_logger('optimi_lab')


def bind_active_logger() -> None:
    """Bind optimi-lab's named logger as lab_commons' process-wide active sink.

    ``log()`` / ``log_decorator()`` / ``timer()`` are lab_commons helpers that write to one
    process-global logger, so binding is a mutation of the WHOLE process, not of
    optimi-lab. It therefore happens only when a caller asks: :func:`add_handle` calls
    this (handler pair first, then routing), and a caller that wants optimi-lab's routing
    without its file/console handlers can call it alone.
    """
    set_active_logger(logger)


def add_handle() -> None:
    """Attach the optimi-lab file + console handlers and route ``log()`` through them.

    The explicit opt-in for optimi-lab's logging: resolves the per-run output directory
    (CREATING it) and attaches the config-driven handler pair. Nothing calls this
    implicitly -- an ``import optimi_lab.utils`` no longer does.

    Because config.py depends on logger.py (config.py -> file_io.py -> logger.py), we
    cannot import config.py at module level here to avoid circular imports.
    """
    from .config import CONF, PathData, Utils  # noqa: PLC0415

    utils_config: Utils = CONF.utils
    # Handlers filter at INFO (matching the pre-lab_commons hardcoded level), but
    # lab_commons' shared add_handle also raises the LOGGER's own level to match --
    # restore it to DEBUG-permissive right after, so anything a caller attaches its own
    # handler to (e.g. a test capturing DEBUG output) still sees everything the logger
    # itself lets through; only the two config-driven handlers stay INFO-and-up.
    _shared_add_handle(
        logger,
        log_file=PathData.log_folder_path / PathData.log_filename,
        level=logging.INFO,
        file_format=utils_config.log_file_format,
        console_format=utils_config.log_console_format,
        date_format=utils_config.log_date_format,
    )
    logger.setLevel(logging.DEBUG)
    # Bind LAST, so the process-wide redirect lands on a logger whose handler pair already
    # exists. This is the half of the old import-time behaviour that reached out of
    # optimi-lab: it moved every other library's log() calls into these handlers.
    bind_active_logger()
