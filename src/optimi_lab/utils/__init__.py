"""optimi-lab's utility layer: importing it has NO side effects.

Nothing here runs on import -- no config file is read, no output directory is created,
no log handler is attached to anything, and lab_commons' process-wide active logger is
not rebound. The previous revision executed ``CONF`` and ``add_handle()`` right here, so
a bare ``import optimi_lab.utils`` (which any ``import optimi_lab.utils.<anything>``
performs) created the per-run output directory and attached a file + console handler pair
-- and, because ``add_handle()`` also bound that logger as the active sink, every
``log()`` call of every library in the process was redirected into optimi-lab's handlers.
A library that is merely imported must not do that to its host.

A caller that wants optimi-lab's configured logging asks for it explicitly::

    from optimi_lab.utils.logger import add_handle

    add_handle()

``optimi_lab.utils.config`` is likewise no longer loaded on import; a caller that needs it
imports it by name.
"""
