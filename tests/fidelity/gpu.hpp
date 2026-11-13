#pragma once

// Keep the workload source and launch geometry identical across GPU backends.
#ifdef FIDELITY_HIP
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
using GPUProperties = hipDeviceProp_t;
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
using GPUProperties = cudaDeviceProp;
#endif

#include <stdexcept>
#include <string>
#include <vector>

inline void check(GPU(Error_t) error) {
  if (error != GPU(Success))
    throw std::runtime_error(GPU(GetErrorString)(error));
}

template <typename T> struct Buffer {
  T *data = nullptr;
  size_t size;

  explicit Buffer(size_t count) : size(count) {
    check(GPU(Malloc)(reinterpret_cast<void **>(&data), size * sizeof(T)));
  }

  ~Buffer() {
    (void)GPU(Free)(data);
  }

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  std::vector<T> read() const {
    std::vector<T> result(size);
    check(GPU(Memcpy)(result.data(), data, size * sizeof(T),
                     GPU(MemcpyDeviceToHost)));
    return result;
  }
};
