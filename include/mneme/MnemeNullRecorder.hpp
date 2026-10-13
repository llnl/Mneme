#pragma once

#include "mneme/MnemeRecorderBackend.hpp"

#include <cstddef>

namespace mneme {

template <DeviceVendors VendorTypes>
class NullRecorder final : public RecorderBackend<VendorTypes> {
  using Base = RecorderBackend<VendorTypes>;
  using DeviceError_t = typename Base::DeviceError_t;
  using DeviceStream_t = typename Base::DeviceStream_t;

  RecorderRuntimeFunctions<VendorTypes> Runtime;

public:
  bool setMetadataForPointer(const void *, Metadata) override { return false; }
  bool getMetadataForPointer(const void *, Metadata &) const override {
    return false;
  }
  bool eraseMetadataForPointer(const void *) override { return false; }

  DeviceError_t rtMalloc(void **ptr, size_t size) override {
    return Runtime.origMallocDevice(ptr, size);
  }

  DeviceError_t rtManagedMalloc(void **ptr, size_t size,
                                unsigned int flags) override {
    return Runtime.origMallocManaged(ptr, size, flags);
  }

  DeviceError_t rtHostMalloc(void **ptr, size_t size,
                             unsigned int flags) override {
    return Runtime.origMallocPinned(ptr, size, flags);
  }

  void trackPassthroughAlloc(void *, size_t, const char *) override {}

  DeviceError_t rtFree(void *ptr) override {
    return Runtime.origFreeDevice(ptr);
  }

  DeviceError_t rtFreeAsync(void *ptr, DeviceStream_t stream) override {
    return Runtime.origFreeAsync(ptr, stream);
  }

  DeviceError_t rtHostFree(void *ptr) override {
    return Runtime.origFreeHost(ptr);
  }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) override {
    return Runtime.origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem,
                                    Stream);
  }

  DeviceError_t rtSetDevice(int deviceID) override {
    return Runtime.origSetDeviceID(deviceID);
  }

  DeviceError_t rtGetDevice(int *deviceID) override {
    return Runtime.origGetDeviceID(deviceID);
  }
};

} // namespace mneme
