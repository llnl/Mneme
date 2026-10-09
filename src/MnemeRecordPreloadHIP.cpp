#include "MnemeAnnotationRuntime.hpp"
#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeCrashHandler.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeRecord.hpp"
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <utility>

using namespace mneme;

class MnemeRecorderHIPPreload
    : public MnemeRecorder<mneme::DeviceVendors::HIP> {
private:
  static constexpr bool hasFatBinEnd = false;
  MnemeRecorderHIPPreload(MnemeRecorderHIPPreload &) = delete;
  MnemeRecorderHIPPreload(MnemeRecorderHIPPreload &&) = delete;

  MnemeRecorderHIPPreload() {
    // Installed on the first intercepted HIP call, i.e. after main has started.
    installCrashHandler();
  }

public:
  static MnemeRecorderHIPPreload &instance() {
    static MnemeRecorderHIPPreload Recorder{};
    return Recorder;
  }
};

using Recorder = MnemeRecorderHIPPreload;

extern "C" {
hipError_t hipMalloc(void **ptr, size_t size) {
  LOG_DEBUG("Entering Mneme to Malloc pointer of size : {}", size);
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtMalloc(ptr, size);
}

hipError_t hipMallocManaged(void **ptr, size_t size, unsigned int flags) {
  LOG_DEBUG("Entering Mneme to Malloc Managed pointer of size : {}", size);
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtManagedMalloc(ptr, size, flags);
};

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  LOG_DEBUG("Entering Mneme to Malloc 'Host|Pinned' pointer of size : {}",
            size);
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtHostMalloc(ptr, size, flags);
}

hipError_t hipHostAlloc(void **ptr, size_t size, unsigned int flags) {
  return forwardAlloc<Recorder>(
      "hipHostAlloc", [&] { return std::pair{*ptr, size}; }, ptr, size, flags);
}

hipError_t hipMallocHost(void **ptr, size_t size) {
  return forwardAlloc<Recorder>(
      "hipMallocHost", [&] { return std::pair{*ptr, size}; }, ptr, size);
}

hipError_t hipMemAllocHost(void **ptr, size_t size) {
  return forwardAlloc<Recorder>(
      "hipMemAllocHost", [&] { return std::pair{*ptr, size}; }, ptr, size);
}

hipError_t hipExtMallocWithFlags(void **ptr, size_t sizeBytes,
                                 unsigned int flags) {
  return forwardAlloc<Recorder>(
      "hipExtMallocWithFlags", [&] { return std::pair{*ptr, sizeBytes}; }, ptr,
      sizeBytes, flags);
}

hipError_t hipMallocPitch(void **ptr, size_t *pitch, size_t width,
                          size_t height) {
  return forwardAlloc<Recorder>(
      "hipMallocPitch", [&] { return std::pair{*ptr, *pitch * height}; }, ptr,
      pitch, width, height);
}

hipError_t hipMemAllocPitch(hipDeviceptr_t *dptr, size_t *pitch,
                            size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes) {
  return forwardAlloc<Recorder>(
      "hipMemAllocPitch", [&] { return std::pair{*dptr, *pitch * height}; },
      dptr, pitch, widthInBytes, height, elementSizeBytes);
}

hipError_t hipMalloc3D(hipPitchedPtr *pitchedDevPtr, hipExtent extent) {
  return forwardAlloc<Recorder>(
      "hipMalloc3D",
      [&] {
        return std::pair{pitchedDevPtr->ptr,
                         pitchedDevPtr->pitch * extent.height * extent.depth};
      },
      pitchedDevPtr, extent);
}

hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream) {
  return forwardAlloc<Recorder>(
      "hipMallocAsync", [&] { return std::pair{*dev_ptr, size}; }, dev_ptr,
      size, stream);
}

hipError_t hipMallocFromPoolAsync(void **dev_ptr, size_t size,
                                  hipMemPool_t mem_pool, hipStream_t stream) {
  return forwardAlloc<Recorder>(
      "hipMallocFromPoolAsync", [&] { return std::pair{*dev_ptr, size}; },
      dev_ptr, size, mem_pool, stream);
}

hipError_t hipFree(void *ptr) {
  LOG_DEBUG("Entering Mneme to Free pointer");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtFree(ptr);
};

hipError_t hipFreeAsync(void *dev_ptr, hipStream_t stream) {
  LOG_DEBUG("Entering Mneme to FreeAsync pointer");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtFreeAsync(dev_ptr, stream);
}

hipError_t hipHostFree(void *ptr) {
  LOG_DEBUG("Entering Mneme to HostFree pointer");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtHostFree(ptr);
}

hipError_t hipFreeHost(void *ptr) {
  LOG_DEBUG("Entering Mneme to FreeHost pointer");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtHostFree(ptr);
}

hipError_t hipSetDevice(int deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtSetDevice(deviceID);
}

hipError_t hipGetDevice(int *deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtGetDevice(deviceID);
}

hipError_t __proteus_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                               void **KernelArgs, uint64_t ShmemSize,
                               void *Stream) {
  LOG_DEBUG("Enetering Mneme to launch kernel");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtLaunchKernel(Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
                              static_cast<hipStream_t>(Stream));
}

bool mneme_set_metadata_for_ptr(const void *ptr, mneme::Metadata md) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.setMetadataForPointer(ptr, std::move(md));
}

bool mneme_get_metadata_for_ptr(const void *ptr, mneme::Metadata *md) {
  if (!md)
    return false;

  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.getMetadataForPointer(ptr, *md);
}

bool mneme_erase_metadata_for_ptr(const void *ptr) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.eraseMetadataForPointer(ptr);
}
}
