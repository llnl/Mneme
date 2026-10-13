#pragma once

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"

#include <cstddef>

namespace mneme {

template <DeviceVendors VendorTypes> struct RecorderRuntimeFunctions {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;

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
    lookup(origLaunchKernel, MnemeDeviceRT::getLaunchKernelFnName());
    lookup(origMallocDevice, MnemeDeviceRT::getDeviceMallocFnName());
    lookup(origMallocPinned, MnemeDeviceRT::getPinnedMallocFnName());
    lookup(origMallocManaged, MnemeDeviceRT::getManagedMallocFnName());
    lookup(origFreeHost, MnemeDeviceRT::getPinnedFreeFnName());
    lookup(origFreeDevice, MnemeDeviceRT::getDeviceFreeFnName());
    lookup(origFreeAsync, MnemeDeviceRT::getAsyncFreeFnName());
    lookup(origSetDeviceID, MnemeDeviceRT::getDeviceSetIDFnName());
    lookup(origGetDeviceID, MnemeDeviceRT::getDeviceGetIDFnName());
  }

private:
  template <typename FnT> static void lookup(FnT &Fn, const char *Name) {
    Fn = getRuntimeFn<VendorTypes, FnT>(Name);
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
