#pragma once

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"

#include <assert.h>
#include <cstddef>
#include <dlfcn.h>

namespace mneme {

template <DeviceVendors VendorTypes> struct RecorderRuntimeFunctions {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;

  void *rtLib = nullptr;

  DeviceError_t (*origLaunchKernel)(const void *func, dim3 gridDim,
                                    dim3 blockDim, void **args,
                                    size_t sharedMem,
                                    DeviceStream_t stream) = nullptr;
  DeviceError_t (*origMallocDevice)(void **ptr, size_t size) = nullptr;
  DeviceError_t (*origMallocPinned)(void **ptr, size_t size,
                                    unsigned int flags) = nullptr;
  DeviceError_t (*origMallocManaged)(void **ptr, size_t size,
                                     unsigned int flags) = nullptr;
  DeviceError_t (*origFreeDevice)(void *devPtr) = nullptr;
  DeviceError_t (*origFreeAsync)(void *devPtr, DeviceStream_t stream) = nullptr;
  DeviceError_t (*origFreeHost)(void *ptr) = nullptr;
  DeviceError_t (*origSetDeviceID)(int id) = nullptr;
  DeviceError_t (*origGetDeviceID)(int *id) = nullptr;

  RecorderRuntimeFunctions() {
    rtLib = MnemeDeviceRT::getRTLib();

    reinterpret_cast<void *&>(origLaunchKernel) =
        dlsym(rtLib, MnemeDeviceRT::getLaunchKernelFnName());
    assert(origLaunchKernel &&
           "Expected non-null kernel-launch function pointer");

    reinterpret_cast<void *&>(origMallocDevice) =
        dlsym(rtLib, MnemeDeviceRT::getDeviceMallocFnName());
    assert(origMallocDevice &&
           "Expected non-null device malloc function pointer");

    reinterpret_cast<void *&>(origMallocPinned) =
        dlsym(rtLib, MnemeDeviceRT::getPinnedMallocFnName());
    assert(origMallocPinned &&
           "Expected non-null pinned malloc function pointer");

    reinterpret_cast<void *&>(origMallocManaged) =
        dlsym(rtLib, MnemeDeviceRT::getManagedMallocFnName());
    assert(origMallocManaged &&
           "Expected non-null managed malloc function pointer");

    reinterpret_cast<void *&>(origFreeHost) =
        dlsym(rtLib, MnemeDeviceRT::getPinnedFreeFnName());
    assert(origFreeHost && "Expected non-null Free Pinned Function");

    reinterpret_cast<void *&>(origFreeDevice) =
        dlsym(rtLib, MnemeDeviceRT::getDeviceFreeFnName());
    assert(origFreeDevice && "Expected non-null Device free function pointer");

    reinterpret_cast<void *&>(origFreeAsync) =
        dlsym(rtLib, MnemeDeviceRT::getAsyncFreeFnName());
    assert(origFreeAsync && "Expected non-null async free function pointer");

    reinterpret_cast<void *&>(origSetDeviceID) =
        dlsym(rtLib, MnemeDeviceRT::getDeviceSetIDFnName());
    assert(origSetDeviceID && "Expected non-null set device id fn name");

    reinterpret_cast<void *&>(origGetDeviceID) =
        dlsym(rtLib, MnemeDeviceRT::getDeviceGetIDFnName());
    assert(origGetDeviceID && "Expected non-null get device id fn name");
  }
};

template <DeviceVendors VendorTypes> class RecorderBackend {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;

  virtual ~RecorderBackend() = default;

  virtual bool setMetadataForPointer(const void *ptr, Metadata md) = 0;
  virtual bool getMetadataForPointer(const void *ptr, Metadata &md) const = 0;
  virtual bool eraseMetadataForPointer(const void *ptr) = 0;

  virtual DeviceError_t rtMalloc(void **ptr, size_t size) = 0;
  virtual DeviceError_t rtManagedMalloc(void **ptr, size_t size,
                                        unsigned int flags) = 0;
  virtual DeviceError_t rtHostMalloc(void **ptr, size_t size,
                                     unsigned int flags) = 0;
  // Remember an allocation that the vendor runtime served directly, so that
  // freeing it is forwarded instead of treated as an unknown address.
  virtual void trackPassthroughAlloc(void *ptr, size_t size,
                                     const char *api) = 0;
  virtual DeviceError_t rtFree(void *ptr) = 0;
  virtual DeviceError_t rtFreeAsync(void *ptr, DeviceStream_t stream) = 0;
  virtual DeviceError_t rtHostFree(void *ptr) = 0;
  virtual DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim,
                                       dim3 &BlockDim, void **Args,
                                       size_t SharedMem,
                                       DeviceStream_t Stream) = 0;
  virtual DeviceError_t rtSetDevice(int deviceID) = 0;
  virtual DeviceError_t rtGetDevice(int *deviceID) = 0;
};

} // namespace mneme
