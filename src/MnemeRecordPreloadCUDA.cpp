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

// Mneme serves only cudaMalloc. Other allocation functions go to the CUDA
// runtime, and their pointers are tracked so that cudaFree forwards them.
static void trackAlloc(void *Ptr, size_t Size, const char *Api) {
  MnemeRecorderCUDAPreload::instance().trackPassthroughAlloc(Ptr, Size, Api);
}

using MallocAsyncFn = cudaError_t (*)(void **, size_t, cudaStream_t);
using MallocFromPoolAsyncFn = cudaError_t (*)(void **, size_t, cudaMemPool_t,
                                              cudaStream_t);

static cudaError_t mallocAsync(MallocAsyncFn Orig, const char *Api,
                               void **devPtr, size_t size,
                               cudaStream_t hStream) {
  auto ret = Orig(devPtr, size, hStream);
  if (ret == cudaSuccess)
    trackAlloc(*devPtr, size, Api);
  return ret;
}

static cudaError_t mallocFromPoolAsync(MallocFromPoolAsyncFn Orig,
                                       const char *Api, void **ptr, size_t size,
                                       cudaMemPool_t memPool,
                                       cudaStream_t stream) {
  auto ret = Orig(ptr, size, memPool, stream);
  if (ret == cudaSuccess)
    trackAlloc(*ptr, size, Api);
  return ret;
}

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
  static auto Orig =
      getRuntimeFn<DeviceVendors::CUDA, cudaError_t (*)(void **, size_t)>(
          "cudaMallocHost");
  auto ret = Orig(ptr, size);
  if (ret == cudaSuccess)
    trackAlloc(*ptr, size, "cudaMallocHost");
  return ret;
}

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width,
                            size_t height) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::CUDA,
                   cudaError_t (*)(void **, size_t *, size_t, size_t)>(
          "cudaMallocPitch");
  auto ret = Orig(devPtr, pitch, width, height);
  if (ret == cudaSuccess)
    trackAlloc(*devPtr, *pitch * height, "cudaMallocPitch");
  return ret;
}

cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr, cudaExtent extent) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::CUDA,
                   cudaError_t (*)(cudaPitchedPtr *, cudaExtent)>(
          "cudaMalloc3D");
  auto ret = Orig(pitchedDevPtr, extent);
  if (ret == cudaSuccess)
    trackAlloc(pitchedDevPtr->ptr,
               pitchedDevPtr->pitch * extent.height * extent.depth,
               "cudaMalloc3D");
  return ret;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::CUDA, MallocAsyncFn>("cudaMallocAsync");
  return mallocAsync(Orig, "cudaMallocAsync", devPtr, size, hStream);
}

cudaError_t cudaMallocAsync_ptsz(void **devPtr, size_t size,
                                 cudaStream_t hStream) {
  static auto Orig =
      getRuntimeFn<DeviceVendors::CUDA, MallocAsyncFn>("cudaMallocAsync_ptsz");
  return mallocAsync(Orig, "cudaMallocAsync", devPtr, size, hStream);
}

cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size,
                                    cudaMemPool_t memPool,
                                    cudaStream_t stream) {
  static auto Orig = getRuntimeFn<DeviceVendors::CUDA, MallocFromPoolAsyncFn>(
      "cudaMallocFromPoolAsync");
  return mallocFromPoolAsync(Orig, "cudaMallocFromPoolAsync", ptr, size,
                             memPool, stream);
}

cudaError_t cudaMallocFromPoolAsync_ptsz(void **ptr, size_t size,
                                         cudaMemPool_t memPool,
                                         cudaStream_t stream) {
  static auto Orig = getRuntimeFn<DeviceVendors::CUDA, MallocFromPoolAsyncFn>(
      "cudaMallocFromPoolAsync_ptsz");
  return mallocFromPoolAsync(Orig, "cudaMallocFromPoolAsync", ptr, size,
                             memPool, stream);
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
