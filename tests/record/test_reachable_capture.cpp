// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" MNEME_RR_KERNELS=struct_kernel MNEME_CAPTURE_MODE=full %build/test_reachable_capture%ext | %FILECHECK %s --check-prefix=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR-FULL
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" MNEME_RR_KERNELS=struct_kernel MNEME_CAPTURE_MODE=reachable %build/test_reachable_capture%ext | %FILECHECK %s --check-prefix=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR-REACHABLE
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" MNEME_RR_KERNELS=lambda_kernel MNEME_CAPTURE_MODE=full %build/test_reachable_capture%ext | %FILECHECK %s --check-prefix=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR-LAMBDA-FULL
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" MNEME_RR_KERNELS=lambda_kernel MNEME_CAPTURE_MODE=reachable %build/test_reachable_capture%ext | %FILECHECK %s --check-prefix=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR-LAMBDA-REACHABLE
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" MNEME_RR_KERNELS=pinned_kernel MNEME_CAPTURE_MODE=reachable %build/test_reachable_capture%ext 2> "%t.$$.err" | %FILECHECK %s --check-prefix=CHECK
// RUN: %FILECHECK %s --check-prefix=CHECK-WARN < "%t.$$.err"
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR-PINNED
// RUN: rm -rf "%t.$$.mneme" "%t.$$.err"
// clang-format on

#include <iostream>

#include "mneme/DeviceTraits.hpp"

using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#define HOST_MALLOC(ptr, size) hipHostMalloc(ptr, size, 0)
#define HOST_FREE hipHostFree
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#define HOST_MALLOC(ptr, size) cudaHostAlloc(ptr, size, 0)
#define HOST_FREE cudaFreeHost
#endif

// The output pointer travels inside a by-value struct so that reachable
// capture has to look through the aggregate to find it.
struct Params {
  float *out;
  int n;
};

__global__ void struct_kernel(const float *in, Params p) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < p.n)
    p.out[idx] = in[idx] * 2.0f;
}

// RAJA-style forall: the kernel takes the loop body by value, so the captured
// pointers arrive inside the closure object rather than as direct arguments.
template <typename Body> __global__ void lambda_kernel(Body body) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  body(idx);
}

// Pinned host memory is not tracked by the recorder, so this pointer cannot be
// resolved to a captured allocation.
__global__ void pinned_kernel(float *m, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    m[idx] = 1.0f;
}

int main() {
  const int N = 64;

  float *in = nullptr, *out = nullptr, *scratch = nullptr, *pinned = nullptr;
  MnemeDeviceRT::DeviceMalloc(reinterpret_cast<void **>(&in),
                              N * sizeof(float));
  MnemeDeviceRT::DeviceMalloc(reinterpret_cast<void **>(&out),
                              N * sizeof(float));
  // Never passed to a kernel, so only full capture records it.
  MnemeDeviceRT::DeviceMalloc(reinterpret_cast<void **>(&scratch),
                              N * sizeof(float));
  HOST_MALLOC(reinterpret_cast<void **>(&pinned), N * sizeof(float));

  struct_kernel<<<1, N>>>(in, Params{out, N});
  const float scale = 2.0f;
  lambda_kernel<<<1, N>>>([=] __device__(int idx) {
    if (idx < N)
      out[idx] = in[idx] * scale;
  });
  pinned_kernel<<<1, N>>>(pinned, N);

  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Device error: " << EC.value() << "\n";
    return -1;
  }

  std::cout << "OK\n";
  MnemeDeviceRT::DeviceFree(in);
  MnemeDeviceRT::DeviceFree(out);
  MnemeDeviceRT::DeviceFree(scratch);
  HOST_FREE(pinned);
  return 0;
}

// clang-format off
// CHECK: OK
// CHECK-RR-FULL: DemangledName: struct_kernel(float const*, Params)
// CHECK-RR-FULL: CaptureMode: full
// CHECK-RR-FULL: NumBlobs: 3
// CHECK-RR-REACHABLE: DemangledName: struct_kernel(float const*, Params)
// CHECK-RR-REACHABLE: CaptureMode: reachable
// CHECK-RR-REACHABLE: NumBlobs: 2
// CHECK-RR-LAMBDA-FULL: DemangledName: void lambda_kernel<{{.*}}>({{.*}})
// CHECK-RR-LAMBDA-FULL: CaptureMode: full
// CHECK-RR-LAMBDA-FULL: NumBlobs: 3
// CHECK-RR-LAMBDA-REACHABLE: DemangledName: void lambda_kernel<{{.*}}>({{.*}})
// CHECK-RR-LAMBDA-REACHABLE: CaptureMode: reachable
// CHECK-RR-LAMBDA-REACHABLE: NumBlobs: 2
// CHECK-WARN: [mneme] Kernel {{.*}}pinned_kernel{{.*}} argument 0 offset 0 points to {{.*}} which is not a tracked allocation or registered global; it will not be captured
// CHECK-RR-PINNED: DemangledName: pinned_kernel(float*, int)
// CHECK-RR-PINNED: CaptureMode: reachable
// CHECK-RR-PINNED: NumBlobs: 0
// clang-format on
