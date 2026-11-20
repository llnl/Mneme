"""
GPU profiling helpers (lazy-loaded native profiler binding).

This module provides a small Python wrapper around Mneme's native profiling
library. The profiler is **loaded lazily** to avoid premature initialization
side-effects in GPU runtimes (notably HSA), which may perform tool initialization
during shared-library load time.

Why lazy-load?
  - Mneme spawns worker processes (fork). Some GPU profiling/tooling stacks
    initialize at import / dlopen time, which is unsafe or undesirable pre-fork.
  - By deferring the load until :func:`init_profiler` is called inside each worker,
    the profiling runtime is initialized in the correct process context.

Public API:
  - :func:`init_profiler`        Initialize (load) the profiling library.
  - :func:`gpu_profile_start`    Start profiling for a kernel name, returns correlation id.
  - :func:`gpu_profile_stop`     Stop profiling and return recorded timestamps/records.

Notes:
  - This is an internal module; callers are expected to call :func:`init_profiler`
    once per process before using start/stop.
  - The native library and its ABI are considered the source of truth.
"""

from ctypes import POINTER, c_char_p, c_uint64, c_int64, c_double, c_int

from mneme.mneme_logging import logger

from .llvm import ffi
from .llvm.common import _encode_string
from .llvm.utils import get_profile_library

profile_lib = None


def _init_profile():
    """
    Lazily load and bind the native profiling library.

    This function configures the ctypes signatures for the exported Mneme profiling
    C-API entry points. It is intentionally not executed at import time to prevent
    GPU-tool initialization from running in the parent process before worker fork.

    Side effects
    ------------
    Sets the global ``profile_lib`` handle and attaches ``argtypes`` / ``restype``
    to the relevant exported functions.
    """

    global profile_lib
    if profile_lib is None:
        profile_lib = ffi.load_library_so(get_profile_library())

        profile_lib.MnemePy_startProfile.argtypes = [c_char_p]
        profile_lib.MnemePy_startProfile.restype = c_int64

        profile_lib.MnemePy_stopProfile.argtypes = [
            c_uint64,
            POINTER(c_int64),
            c_uint64,
        ]

        profile_lib.MnemePy_getNumRecords.argtypes = [
            c_uint64,
            c_int64,
        ]
        profile_lib.MnemePy_getNumRecords.restype = c_int64

        profile_lib.MnemePy_initProfiler.argtypes = []

        # The counter exports exist only in the CUPTI profile library.
        if hasattr(profile_lib, "MnemePy_counterAvailable"):
            profile_lib.MnemePy_counterAvailable.argtypes = []
            profile_lib.MnemePy_counterAvailable.restype = c_int

            profile_lib.MnemePy_counterMetricName.argtypes = []
            profile_lib.MnemePy_counterMetricName.restype = c_char_p

            profile_lib.MnemePy_counterInit.argtypes = [c_int]
            profile_lib.MnemePy_counterInit.restype = c_int

            profile_lib.MnemePy_counterStart.argtypes = [c_int64]
            profile_lib.MnemePy_counterStart.restype = c_int

            profile_lib.MnemePy_counterStop.argtypes = [POINTER(c_double), c_int64]
            profile_lib.MnemePy_counterStop.restype = c_int64


def gpu_profile_start(kernel_name: str):
    """
    Start GPU profiling for a kernel.

    Parameters
    ----------
    kernel_name : str
        Kernel name used as a label by the profiling backend.

    Returns
    -------
    int
        Correlation identifier used to match start/stop calls.

    Raises
    ------
    RuntimeError
        If the profiling library has not been initialized via :func:`init_profiler`.
    """
    if profile_lib is None:
        raise RuntimeError("Profile library is not initialized")
    return int(profile_lib.MnemePy_startProfile(_encode_string(kernel_name)))


def gpu_profile_stop(correlation_id: int, launches: int):
    """
    Stop GPU profiling and return recorded profiling values.

    Parameters
    ----------
    correlation_id : int
        Correlation identifier returned by :func:`gpu_profile_start`.
    launches : int
        Number of target kernel launches to wait for.

    Returns
    -------
    list[int]
        List of profiling records returned by the native backend (typically GPU
        timestamps or counter values, depending on the profiler implementation).

    Raises
    ------
    RuntimeError
        If the profiling library has not been initialized via :func:`init_profiler`,
        or fewer than ``launches`` records arrived.
    """
    if profile_lib is None:
        raise RuntimeError("Profile library is not initialized")

    num_records = profile_lib.MnemePy_getNumRecords(correlation_id, launches)

    if num_records < launches:
        raise RuntimeError(
            f"Profiler delivered {num_records} of {launches} records "
            f"for token={correlation_id}"
        )

    logger.debug(f"Profiler contains {num_records} records")
    arr = (c_int64 * num_records)()

    _ = profile_lib.MnemePy_stopProfile(correlation_id, arr, num_records)
    return [int(r) for r in arr]


def init_profiler() -> None:
    """
    Initialize the Mneme profiling backend for the current process.

    This should be called inside each worker process (post-fork) before any calls
    to :func:`gpu_profile_start` / :func:`gpu_profile_stop`.
    """
    _init_profile()


def counters_available() -> bool:
    """Whether the native profile library can collect hardware counters."""
    _init_profile()
    if not hasattr(profile_lib, "MnemePy_counterAvailable"):
        return False
    return bool(profile_lib.MnemePy_counterAvailable())


def counter_metric_name() -> str:
    """Name of the (currently hard-coded) hardware counter that is collected."""
    _init_profile()
    return profile_lib.MnemePy_counterMetricName().decode()


_counters_ready = False


def gpu_counter_start(device_id: int, max_ranges: int) -> None:
    """
    Open a hardware-counter session for up to ``max_ranges`` kernel launches.

    The first call in a process also does the one-time counter setup.
    """
    global _counters_ready
    if profile_lib is None:
        raise RuntimeError("Profile library is not initialized")
    if not _counters_ready:
        if profile_lib.MnemePy_counterInit(device_id) != 0:
            raise RuntimeError("Counter setup failed")
        _counters_ready = True
    if profile_lib.MnemePy_counterStart(max_ranges) != 0:
        raise RuntimeError("Could not start the counter session")


def gpu_counter_stop(expected: int) -> list[float]:
    """
    Close the counter session and return one value per recorded launch.

    ``expected`` is the number of launches made inside the session.
    """
    if profile_lib is None:
        raise RuntimeError("Profile library is not initialized")
    capacity = expected + 4
    arr = (c_double * capacity)()
    recorded = profile_lib.MnemePy_counterStop(arr, capacity)
    if recorded < 0:
        raise RuntimeError("Counter collection failed")
    if recorded != expected:
        logger.warning(
            f"Counter session recorded {recorded} ranges for {expected} launches"
        )
    return [float(arr[i]) for i in range(min(recorded, capacity))]
