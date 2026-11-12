import pytest
from unittest.mock import patch, MagicMock

from mneme.proteus import jit
from mneme.llvm.module import ModuleRef
from mneme.llvm.buffer import MemBufferRef


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


class DummyModule(ModuleRef):
    """A minimal dummy ModuleRef for type checking."""

    def __init__(self):
        # ModuleRef typically stores a pointer + context,
        # but we don’t need real LLVM refs for unit tests.
        self._ptr = MagicMock()
        self._ctx = MagicMock()

    def _dispose(self):
        """Disable LLVM disposal so tests don't hit C destructor logic."""
        pass

    def __del__(self):
        """Suppress destructor behavior."""
        pass


# ------------------------------------------------------------
# pruneIR
# ------------------------------------------------------------


def test_pruneIR_accepts_ModuleRef():
    m = DummyModule()
    with patch("mneme.proteus.jit.ffi.lib", autospec=True) as fake_lib:
        fake_lib.ProteusPY_pruneIR = MagicMock()
        jit.pruneIR(m)
        fake_lib.ProteusPY_pruneIR.assert_called_once()


def test_pruneIR_raises_for_wrong_type():
    with pytest.raises(TypeError):
        jit.pruneIR(mod="not-a-module")


# ------------------------------------------------------------
# optimize
# ------------------------------------------------------------


def test_optimize_invokes_ffi_correctly():
    m = DummyModule()
    with patch("mneme.proteus.jit.ffi.lib", autospec=True) as fake_lib:
        fake_lib.ProteusPY_isValidPassPipeline = MagicMock(return_value=True)
        fake_lib.LLVMPY_DisposeString = MagicMock()
        fn = fake_lib.ProteusPY_optimize = MagicMock()
        jit.optimize(m, "gfx942", "default<O2>", 2)
        assert fn.called
        args = fn.call_args[0]
        assert args[0] is m  # ModuleRef
        assert isinstance(args[1], bytes)
        assert isinstance(args[2], bytes)


def test_optimize_invalid_codegen_level():
    m = DummyModule()
    with pytest.raises(ValueError):
        jit.optimize(m, "gfx942", "O2", 99)


def test_optimize_empty_opt_level_skips():
    m = DummyModule()
    with patch("mneme.proteus.jit.ffi.lib", autospec=True) as fake_lib:
        fn = fake_lib.ProteusPY_optimize = MagicMock()
        jit.optimize(m, "gfx942", "", 2)
        fn.assert_not_called()


# ------------------------------------------------------------
# internalize
# ------------------------------------------------------------


def test_internalize_calls_ffi():
    m = DummyModule()
    with patch("mneme.proteus.jit.ffi.lib", autospec=True) as fake_lib:
        fn = fake_lib.ProteusPY_internalize = MagicMock()
        jit.internalize(m, "kernel")
        fn.assert_called_once()
        args = fn.call_args[0]
        assert args[0] is m
        assert isinstance(args[1], bytes)


# ------------------------------------------------------------
# codegen_object
# ------------------------------------------------------------
@patch("mneme.llvm.buffer.ffi.lib")
def test_codegen_object_returns_MemBufferRef(fake_buflib):
    fake_buflib.LLVMPY_DisposeMemBuffer = MagicMock()
    m = DummyModule()
    fake_buf = MagicMock()

    with patch("mneme.proteus.jit.ffi.lib") as fake_lib:
        fn = fake_lib.ProteusPY_codeGenObject = MagicMock()

        buf = jit.codegen_object(m, "gfx942", 3)
        assert isinstance(buf, MemBufferRef)


def test_codegen_object_invalid_opt_level():
    m = DummyModule()
    with pytest.raises(RuntimeError):
        jit.codegen_object(m, "gfx942", 0)


# ------------------------------------------------------------
# link_llvm_modules
# ------------------------------------------------------------


def test_link_llvm_modules_calls_ffi_with_c_array():
    fake_mod = MagicMock()

    # Patch BOTH ffi.lib and get_global_context
    with patch("mneme.proteus.jit.ffi.lib") as fake_lib, patch(
        "mneme.proteus.jit.get_global_context", return_value="CTX"
    ) as fake_ctx:

        fake_lib.ProteusPY_linkModules = MagicMock(return_value=fake_mod)

        out = jit.link_llvm_modules(
            modules=["a.ll", "b.ll"],
            kernel_name="kernel",
            prune=True,
            internalize=True,
        )

        # Validate call
        fake_lib.ProteusPY_linkModules.assert_called_once()

        # Validate return wrapped in ModuleRef
        assert isinstance(out, jit.ModuleRef)


# ------------------------------------------------------------
# specialize_args
# ------------------------------------------------------------


def test_specialize_args_checks_index_count():
    m = DummyModule()
    with pytest.raises(RuntimeError):
        jit.specialize_args(
            m, 123, "kernel", None, num_args=1, specialize_indexes=[0, 1]
        )


def test_specialize_args_invokes_ffi():
    m = DummyModule()
    with patch("mneme.proteus.jit.ffi.lib", autospec=True) as fake_lib:
        fn = fake_lib.ProteusPY_specializeArguments = MagicMock(return_value=42)
        h = jit.specialize_args(
            m,
            10,
            "kernel",
            kernel_args=MagicMock(),
            num_args=3,
            specialize_indexes=[0, 2],
        )
        assert h == 42
        assert fn.called


# ------------------------------------------------------------
# specialize_dims
# ------------------------------------------------------------


def test_specialize_dims_invokes_ffi():
    m = DummyModule()
    with patch("mneme.proteus.jit.ffi.lib", autospec=True) as fake_lib:
        fn = fake_lib.ProteusPY_specializeDims = MagicMock(return_value=7)

        from mneme.mneme_types import dim3

        dd = dim3(1, 2, 3)
        h = jit.specialize_dims(m, 11, "kernel", dd, dd)
        assert h == 7
        assert fn.called


# ------------------------------------------------------------
# set_launch_bounds
# ------------------------------------------------------------


def test_set_launch_bounds_checks_max_threads():
    m = DummyModule()
    with pytest.raises(RuntimeError):
        jit.set_launch_bounds(m, 11, "kernel", 4096, 1)


def test_set_launch_bounds_invokes_ffi():
    m = DummyModule()
    with patch("mneme.proteus.jit.ffi.lib", autospec=True) as fake_lib:
        fn = fake_lib.ProteusPY_setLaunchBounds = MagicMock(return_value=99)
        h = jit.set_launch_bounds(m, 11, "kernel", 128, 1)
        assert h == 99
        assert fn.called
