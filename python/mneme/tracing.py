"""Optional host tracing. Set MNEME_TRACE_DIR before importing Mneme."""

import atexit
import json
import os
import sys
from ctypes import CDLL, c_char_p, c_uint64
from functools import wraps

enabled = bool(os.environ.get("MNEME_TRACE_DIR"))
_lib = None


def _native():
    global _lib

    if _lib is None:
        from mneme.llvm import ffi

        # grrr... make sure the sink syncs itself; unfortunately LLVM's python
        # lock is inherited by a fork so two threads can hold it
        lib = CDLL(None, handle=ffi.lib._handle)
        lib.MnemePy_TraceTimestamp.argtypes = []
        lib.MnemePy_TraceTimestamp.restype = c_uint64

        lib.MnemePy_TraceComplete.argtypes = [c_char_p, c_uint64, c_uint64, c_char_p]
        lib.MnemePy_TraceComplete.restype = None

        lib.MnemePy_TraceFlush.argtypes = []
        lib.MnemePy_TraceFlush.restype = None

        lib.MnemePy_TraceDisable.argtypes = [c_char_p]
        lib.MnemePy_TraceDisable.restype = None

        _lib = lib

    return _lib


def _disable(error):
    global enabled

    if enabled:
        enabled = False
        reason = str(error)

        try:
            _native().MnemePy_TraceDisable(reason.encode("utf-8"))
        except Exception:
            print(f"mneme: tracing disabled: {reason}", file=sys.stderr)


def flush():
    if enabled:
        try:
            _native().MnemePy_TraceFlush()
        except Exception as error:
            _disable(error)


class _NoopScope:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def annotate(self, **attributes):
        pass


_NOOP = _NoopScope()


class _Scope:
    def __init__(self, name, attributes, flush_after):
        self.name = name
        self.attributes = attributes
        self.flush_after = flush_after
        self.start = 0
        self.arguments = {}

    def __enter__(self):
        try:
            self.start = _native().MnemePy_TraceTimestamp()
            self.owner = os.getpid()

            if self.start and self.attributes is not None:
                self.arguments.update(self.attributes())
        except Exception as error:
            self.start = 0
            _disable(error)

        return self

    def annotate(self, **attributes):
        if self.start:
            try:
                self.arguments.update(attributes)
            except Exception as error:
                _disable(error)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.start and enabled and self.owner == os.getpid():
                end = _native().MnemePy_TraceTimestamp()

                if exc_type is not None:
                    self.arguments["error"] = exc_type.__name__

                _native().MnemePy_TraceComplete(
                    self.name.encode("utf-8"),
                    self.start,
                    end,
                    json.dumps(
                        self.arguments,
                        ensure_ascii=True,
                        allow_nan=False,
                    ).encode("utf-8"),
                )
        except Exception as error:
            _disable(error)
        finally:
            if self.flush_after:
                flush()

        return False


def span(name, attributes=None, *, flush_after=False):
    """Time a scope; evaluate the optional attributes callback only when enabled."""

    return _Scope(name, attributes, flush_after) if enabled else _NOOP


def traced(name, *, attributes=None, flush_after=False):
    """Return the original function when tracing was disabled at import time."""

    def decorate(function):
        if not enabled:
            return function

        @wraps(function)
        def wrapped(*args, **kwargs):
            metadata = (
                None if attributes is None else lambda: attributes(*args, **kwargs)
            )

            with span(name, metadata, flush_after=flush_after):
                return function(*args, **kwargs)

        return wrapped

    return decorate


if enabled:
    atexit.register(flush)
