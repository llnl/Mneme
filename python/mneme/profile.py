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

from ctypes import POINTER, c_char_p, c_uint64, c_int64

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
