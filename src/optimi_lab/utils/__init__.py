"""Avoid circular imports; ensure config.py is loaded before calling add_handle()."""

from .config import CONF
from .logger import add_handle

CONF
add_handle()
