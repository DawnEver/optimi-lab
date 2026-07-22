"""optimi-lab logging binding.

Thin optimi-lab-specific layer over the shared ``lab_commons`` primitives: it binds the
named ``'optimi_lab'`` logger and wires the file/console handlers from optimi-lab's
config + PathData. The generic mechanics (named logger, handler pair, log/timer helpers)
live in ``lab_commons.log``; only the domain wiring (warning filters, config-driven handler
setup) stays here.
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

__all__ = ['add_handle', 'log', 'log_decorator', 'timer']

for warning in [
    # unitstrippedwarning: the unit of the quantity is stripped when downcasting to ndarray.
    pint_UnitStrippedWarning,
]:
    warnings.filterwarnings('ignore', category=warning)

# Dedicated named logger so handlers never collide with the root logger (or any other
# library sharing the process, e.g. lab-commons' other consumers). Bound as the active
# sink for the module-level log()/timer() helpers.
logger = get_logger('optimi_lab')
set_active_logger(logger)


def add_handle() -> None:
    """Add the optimi-lab file + console logging handlers based on Config.

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
