import pytest
from unittest.mock import MagicMock, patch
from ctypes import c_void_p

from mneme.device import (
    DeviceModule,
    DeviceFunction,
    get_device_arch,
    get_device_count,
    set_device,
)
from mneme.llvm.buffer import MemBufferRef
from mneme.mneme_types import dim3


#
# DeviceModule tests
#

def test_device_module_from_membuffer_validates_type():
    buf = MemBufferRef.__new__(MemBufferRef)
    # Never initialized, so keep its finalizer from disposing it.
    buf._closed = True

    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePY_getDeviceObject = MagicMock(return_value=c_void_p(111))
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()

        dm = DeviceModule.from_MemBuffer(buf)
        assert isinstance(dm, DeviceModule)


def test_device_module_from_membuffer_rejects_wrong_type():
    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()
        with pytest.raises(TypeError):
            DeviceModule.from_MemBuffer("not a membuffer")


def test_device_module_get_function():
    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()
        fake_lib.MnemePY_getKernelFunctionFromImage = MagicMock(return_value=c_void_p(999))

        dm = DeviceModule(c_void_p(111))
        func = dm.get_function("kernel")

        assert isinstance(func, DeviceFunction)
        assert func._kernel_name == "kernel"
        fake_lib.MnemePY_getKernelFunctionFromImage.assert_called_once()


def test_device_module_dispose_invalidates_functions():
    func_ptr = c_void_p(123)

    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()

        dm = DeviceModule(c_void_p(111))
        f = DeviceFunction(func_ptr, dm, "k")
        dm._functions.add(f)

        # Dispose module
        dm._dispose()
        assert not f.valid
        fake_lib.MnemePY_DisposeDeviceObject.assert_called_once()

#
# DeviceFunction tests
#

def test_device_function_lazy_properties():
    with patch("mneme.llvm.ffi.ObjectRef.detach", lambda self: None), \
     patch("mneme.llvm.ffi.ObjectRef.close", lambda self: None), \
     patch("mneme.llvm.ffi.ObjectRef._dispose", lambda self: None), \
        patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()
        fake_lib.MnemePy_getLocalMemUsage = MagicMock(return_value=10)
        fake_lib.MnemePy_getRegisterUsage = MagicMock(return_value=20)
        fake_lib.MnemePy_getConstMemUsage = MagicMock(return_value=30)

        dm = DeviceModule(c_void_p(111))
        f = DeviceFunction(c_void_p(1), dm, "k")
        f._as_parameter_ = 0
        dm._as_parameter_ = 0

        assert f.local_mem == 10
        assert f.reg_usage == 20
        assert f.const_mem == 30


def test_device_function_invalidate_sets_flag():
    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()

        dm = DeviceModule(c_void_p(111))
        f = DeviceFunction(c_void_p(1), dm, "k")
        assert f.valid

        f.invalidate()
        assert not f.valid


def test_device_function_launch_success():
    with patch("mneme.device.ffi.lib") as fake_lib:
        fake_lib.MnemePy_launchKernelFunction = MagicMock()
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()

        dm = DeviceModule(c_void_p(222))
        f = DeviceFunction(c_void_p(333), dm, "kernel")

        f.launch(dim3(1, 1, 1), dim3(32, 1, 1))
        fake_lib.MnemePy_launchKernelFunction.assert_called_once()


def test_device_function_launch_fails_if_ptr_is_null():
    with patch("mneme.device.ffi.lib") as fake_lib:
        fake_lib.MnemePy_launchKernelFunction = MagicMock()
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()

        dm = DeviceModule(c_void_p(111))
        f = DeviceFunction(c_void_p(None), dm, "k")

        with pytest.raises(RuntimeError):
            f.launch(dim3(1, 1, 1), dim3(1, 1, 1))


def test_device_function_launch_fails_if_module_gone():
    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()
        dm = DeviceModule(c_void_p(111))
        f = DeviceFunction(c_void_p(1), dm, "k")

        # simulate GC of the module
        f._module_ref = lambda: None

        with pytest.raises(RuntimeError):
            f.launch(dim3(1, 1, 1), dim3(1, 1, 1))


def test_device_function_profile_calls_ffi():
    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePy_profile = MagicMock()
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()

        dm = DeviceModule(c_void_p(123))
        f = DeviceFunction(c_void_p(999), dm, "ker")

        buf = MagicMock()  # Memory state mock

        f.profile(
            grid_dim=dim3(1, 1, 1),
            block_dim=dim3(32, 1, 1),
            prologue_state=buf,
            epilogue_state=buf,
            shared_mem_size=0,
            iterations=5,
        )

        fake_lib.MnemePy_profile.assert_called_once()


def test_device_function_profile_fails_if_module_gc():
    with patch("mneme.llvm.ffi.ObjectRef.detach", lambda self: None), \
     patch("mneme.llvm.ffi.ObjectRef.close", lambda self: None), \
     patch("mneme.llvm.ffi.ObjectRef._dispose", lambda self: None), \
     patch("mneme.device.ffi.lib") as fake_lib:
        fake_lib.MnemePy_launchKernelFunction = MagicMock()
        fake_lib.MnemePY_DisposeDeviceObject = MagicMock()

        dm = DeviceModule(c_void_p(111))
        f = DeviceFunction(c_void_p(1), dm, "k")

        f._module_ref = lambda: None
        f._as_parameter_ = 0

        with pytest.raises(RuntimeError):
            f.profile(dim3(1, 1, 1), dim3(1, 1, 1), None, None, 0)


#
# Device helper function tests
#

def test_get_device_arch():
    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePy_getDeviceArch = MagicMock(return_value=b"gfx90a")

        arch = get_device_arch()
        assert arch == "gfx90a"
        fake_lib.MnemePy_getDeviceArch.assert_called_once()


def test_get_device_count():
    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePy_getDeviceCount = MagicMock(return_value=3)

        assert get_device_count() == 3


def test_set_device_calls_ffi():
    with patch("mneme.device.ffi.lib", autospec=True) as fake_lib:
        fake_lib.MnemePy_setDevice = MagicMock()

        set_device(2)
        fake_lib.MnemePy_setDevice.assert_called_once_with(2)

