#pragma once

#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeNullRecorder.hpp"
#include "mneme/MnemeRecorderBackend.hpp"
#include "mneme/MnemeRecordingBackend.hpp"

#include <memory>
#include <utility>

namespace mneme {

template <DeviceVendors VendorTypes> class MnemeRecorder {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;

private:
  static NullRecorder<VendorTypes>& getStaticNullRecorder() {
    static NullRecorder<VendorTypes> instance;
    return instance;
  }

  // non-owning pointer to avoid destruction issues during shutdown
  RecorderBackend<VendorTypes>* Backend = nullptr;

public:
  MnemeRecorder(const MnemeRecorder&) = delete;
  MnemeRecorder& operator=(const MnemeRecorder&) = delete;
  MnemeRecorder(MnemeRecorder&&) = delete;
  MnemeRecorder& operator=(MnemeRecorder&&) = delete;

  MnemeRecorder() {
    if (RecordingPolicy::get().isRecordingEnabledForCurrentRank())
      Backend = new RecordingBackend<VendorTypes>();
    else
      Backend = &getStaticNullRecorder();
  }

  ~MnemeRecorder() {
    // repoint to the static NullRecorder before deleting so Backend doesn't point to freed mem
    RecorderBackend<VendorTypes>* old = Backend;
    Backend = &getStaticNullRecorder();
    if (old && old != &getStaticNullRecorder())
      delete old;
  }

  bool setMetadataForPointer(const void *ptr, Metadata md) {
    return Backend->setMetadataForPointer(ptr, std::move(md));
  }

  bool getMetadataForPointer(const void *ptr, Metadata &md) const {
    return Backend->getMetadataForPointer(ptr, md);
  }

  bool eraseMetadataForPointer(const void *ptr) {
    return Backend->eraseMetadataForPointer(ptr);
  }

  DeviceError_t rtMalloc(void **ptr, size_t size) {
    return Backend->rtMalloc(ptr, size);
  }

  DeviceError_t rtManagedMalloc(void **ptr, size_t size, unsigned int flags) {
    return Backend->rtManagedMalloc(ptr, size, flags);
  }

  DeviceError_t rtHostMalloc(void **ptr, size_t size, unsigned int flags) {
    return Backend->rtHostMalloc(ptr, size, flags);
  }

  DeviceError_t rtFree(void *ptr) { return Backend->rtFree(ptr); }

  DeviceError_t rtHostFree(void *ptr) { return Backend->rtHostFree(ptr); }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) {
    return Backend->rtLaunchKernel(func, GridDim, BlockDim, Args, SharedMem,
                                   Stream);
  }

  DeviceError_t rtSetDevice(int deviceID) {
    return Backend->rtSetDevice(deviceID);
  }

  DeviceError_t rtGetDevice(int *deviceID) {
    return Backend->rtGetDevice(deviceID);
  }
};

} // namespace mneme
