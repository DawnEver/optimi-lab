"""Parallel-run helper — re-exported from the shared ``lab_commons.multiprocess`` (extracted, plan §3).

The timeout/retry-aware ``mp.Pool`` runner and its pydantic parameter model live in lab_commons now;
its dependencies (units/em/exceptions/log) are all lab_commons too, so the extraction is a clean
re-point (decoupling from a project's own quantities module was the plan's one blocker for this
module). optimi-lab has no multiprocessing of its own — this is a thin re-export.
"""

from lab_commons.multiprocess import (
    MultiProcessing,
    MultiProcessingParameters,
    global_mp_params,
    machine_cores,
)

__all__ = ['MultiProcessing', 'MultiProcessingParameters', 'global_mp_params', 'machine_cores']
