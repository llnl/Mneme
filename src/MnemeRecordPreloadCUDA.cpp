#include "MnemeAnnotationRuntime.hpp"
#include "mneme/MnemeCrashHandler.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeRecord.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <utility>

#ifdef __GNUC__
#define __align__(n) __attribute__((aligned(n)))
#else
#define __align__(n) __declspec(align(n))
#endif

using namespace mneme;

class MnemeRecorderCUDAPreload
    : public MnemeRecorder<mneme::DeviceVendors::CUDA> {
private:
  static constexpr bool hasFatBinEnd = true;
  MnemeRecorderCUDAPreload(MnemeRecorderCUDAPreload &) = delete;
  MnemeRecorderCUDAPreload(MnemeRecorderCUDAPreload &&) = delete;
  MnemeRecorderCUDAPreload() {
    // NOTE: This is important to be called in the initializer. As it enforces
    // the initialization/de-initialization order.
    // FIXME: Fix de-init fiasco order in some proper way
    LOG_DEBUG("Initializing preloaded library");
    // Installed on the first intercepted CUDA call, i.e. after main has started.
    installCrashHandler();
  }

public:
  static MnemeRecorderCUDAPreload &instance() {
    static MnemeRecorderCUDAPreload Recorder{};
    return Recorder;
  }
};

using Recorder = MnemeRecorderCUDAPreload;

extern "C" {
cudaError_t cudaMalloc(void **ptr, size_t size) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Malloc pointer of size : {}", size);
  return mneme.rtMalloc(ptr, size);
}

cudaError_t cudaMallocManaged(void **ptr, size_t size, unsigned int flags) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Malloc Managed pointer of size : {}", size);
  return mneme.rtManagedMalloc(ptr, size, flags);
};

cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Malloc 'Host|Pinned' pointer of size : {}",
            size);
  return mneme.rtHostMalloc(ptr, size, flags);
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
  return forwardAlloc<Recorder>(
      "cudaMallocHost", [&] { return std::pair{*ptr, size}; }, ptr, size);
}

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width,
                            size_t height) {
  return forwardAlloc<Recorder>(
      "cudaMallocPitch", [&] { return std::pair{*devPtr, *pitch * height}; },
      devPtr, pitch, width, height);
}

cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr, cudaExtent extent) {
  return forwardAlloc<Recorder>(
      "cudaMalloc3D",
      [&] {
        return std::pair{pitchedDevPtr->ptr,
                         pitchedDevPtr->pitch * extent.height * extent.depth};
      },
      pitchedDevPtr, extent);
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream) {
  return forwardAlloc<Recorder>(
      "cudaMallocAsync", [&] { return std::pair{*devPtr, size}; }, devPtr,
      size, hStream);
}

cudaError_t cudaMallocAsync_ptsz(void **devPtr, size_t size,
                                 cudaStream_t hStream) {
  return forwardAlloc<Recorder>(
      "cudaMallocAsync_ptsz", [&] { return std::pair{*devPtr, size}; }, devPtr,
      size, hStream);
}

cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size,
                                    cudaMemPool_t memPool,
                                    cudaStream_t stream) {
  return forwardAlloc<Recorder>(
      "cudaMallocFromPoolAsync", [&] { return std::pair{*ptr, size}; }, ptr,
      size, memPool, stream);
}

cudaError_t cudaMallocFromPoolAsync_ptsz(void **ptr, size_t size,
                                         cudaMemPool_t memPool,
                                         cudaStream_t stream) {
  return forwardAlloc<Recorder>(
      "cudaMallocFromPoolAsync_ptsz", [&] { return std::pair{*ptr, size}; },
      ptr, size, memPool, stream);
}

cudaError_t cudaFree(void *ptr) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to Free pointer");
  return mneme.rtFree(ptr);
};

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to FreeAsync pointer");
  return mneme.rtFreeAsync(devPtr, hStream);
}

// Used instead of cudaFreeAsync when compiling with a per-thread default
// stream, where stream 0 means the calling thread's stream.
cudaError_t cudaFreeAsync_ptsz(void *devPtr, cudaStream_t hStream) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to FreeAsync pointer");
  return mneme.rtFreeAsync(devPtr, hStream ? hStream : cudaStreamPerThread);
}

cudaError_t cudaFreeHost(void *ptr) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  LOG_DEBUG("Entering Mneme to FreeHost pointer");
  return mneme.rtHostFree(ptr);
}

cudaError_t cudaSetDevice(int deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtSetDevice(deviceID);
}

cudaError_t cudaGetDevice(int *deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtGetDevice(deviceID);
}

cudaError_t __proteus_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                               void **KernelArgs, uint64_t ShmemSize,
                               void *Stream) {
  LOG_DEBUG("Enetering Mneme to launch kernel");
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.rtLaunchKernel(Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
                              static_cast<cudaStream_t>(Stream));
}

bool mneme_set_metadata_for_ptr(const void *ptr, mneme::Metadata md) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.setMetadataForPointer(ptr, std::move(md));
}

bool mneme_get_metadata_for_ptr(const void *ptr, mneme::Metadata *md) {
  if (!md)
    return false;

  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.getMetadataForPointer(ptr, *md);
}

bool mneme_erase_metadata_for_ptr(const void *ptr) {
  auto &mneme = MnemeRecorderCUDAPreload::instance();
  return mneme.eraseMetadataForPointer(ptr);
}

}
