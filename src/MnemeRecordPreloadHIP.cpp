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

// Mneme serves only hipMalloc. Other allocation functions go to the HIP
// runtime, and their pointers are tracked so that hipFree forwards them.
static void trackAlloc(void *Ptr, size_t Size, const char *Api) {
  MnemeRecorderHIPPreload::instance().trackPassthroughAlloc(Ptr, Size, Api);
}

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
  static auto Orig =
      getRuntimeFn<DeviceVendors::HIP,
                   hipError_t (*)(void **, size_t, unsigned int)>(
          "hipHostAlloc");
  auto ret = Orig(ptr, size, flags);
  if (ret == hipSuccess)
    trackAlloc(*ptr, size, "hipHostAlloc");
  return ret;
}

hipError_t hipMallocHost(void **ptr, size_t size) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::HIP, hipError_t (*)(void **, size_t)>(
          "hipMallocHost");
  auto ret = Orig(ptr, size);
  if (ret == hipSuccess)
    trackAlloc(*ptr, size, "hipMallocHost");
  return ret;
}

hipError_t hipMemAllocHost(void **ptr, size_t size) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::HIP, hipError_t (*)(void **, size_t)>(
          "hipMemAllocHost");
  auto ret = Orig(ptr, size);
  if (ret == hipSuccess)
    trackAlloc(*ptr, size, "hipMemAllocHost");
  return ret;
}

hipError_t hipExtMallocWithFlags(void **ptr, size_t sizeBytes,
                                 unsigned int flags) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::HIP,
                   hipError_t (*)(void **, size_t, unsigned int)>(
          "hipExtMallocWithFlags");
  auto ret = Orig(ptr, sizeBytes, flags);
  if (ret == hipSuccess)
    trackAlloc(*ptr, sizeBytes, "hipExtMallocWithFlags");
  return ret;
}

hipError_t hipMallocPitch(void **ptr, size_t *pitch, size_t width,
                          size_t height) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::HIP,
                   hipError_t (*)(void **, size_t *, size_t, size_t)>(
          "hipMallocPitch");
  auto ret = Orig(ptr, pitch, width, height);
  if (ret == hipSuccess)
    trackAlloc(*ptr, *pitch * height, "hipMallocPitch");
  return ret;
}

hipError_t hipMemAllocPitch(hipDeviceptr_t *dptr, size_t *pitch,
                            size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::HIP,
                   hipError_t (*)(hipDeviceptr_t *, size_t *, size_t, size_t,
                                  unsigned int)>("hipMemAllocPitch");
  auto ret = Orig(dptr, pitch, widthInBytes, height, elementSizeBytes);
  if (ret == hipSuccess)
    trackAlloc(*dptr, *pitch * height, "hipMemAllocPitch");
  return ret;
}

hipError_t hipMalloc3D(hipPitchedPtr *pitchedDevPtr, hipExtent extent) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::HIP,
                   hipError_t (*)(hipPitchedPtr *, hipExtent)>("hipMalloc3D");
  auto ret = Orig(pitchedDevPtr, extent);
  if (ret == hipSuccess)
    trackAlloc(pitchedDevPtr->ptr,
               pitchedDevPtr->pitch * extent.height * extent.depth,
               "hipMalloc3D");
  return ret;
}

hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream) {
  static auto Orig = getRuntimeFn<DeviceVendors::HIP,
                                  hipError_t (*)(void **, size_t, hipStream_t)>(
      "hipMallocAsync");
  auto ret = Orig(dev_ptr, size, stream);
  if (ret == hipSuccess)
    trackAlloc(*dev_ptr, size, "hipMallocAsync");
  return ret;
}

hipError_t hipMallocFromPoolAsync(void **dev_ptr, size_t size,
                                  hipMemPool_t mem_pool, hipStream_t stream) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::HIP,
                   hipError_t (*)(void **, size_t, hipMemPool_t, hipStream_t)>(
          "hipMallocFromPoolAsync");
  auto ret = Orig(dev_ptr, size, mem_pool, stream);
  if (ret == hipSuccess)
    trackAlloc(*dev_ptr, size, "hipMallocFromPoolAsync");
  return ret;
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
