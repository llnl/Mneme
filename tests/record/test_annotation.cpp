// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_DATA_DIR="%t.$$.mneme" %build/test_annotation%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

#include <iostream>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"

using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

// A simple kernel that takes a float pointer argument so the preload captures
// at least one annotated blob in the prologue/epilogue snapshots.
__global__ void annotation_kernel(float *data, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    data[idx] = static_cast<float>(idx);
}

int main() {
  const int N = 64;

  float *d_ptr = nullptr;
  MnemeDeviceRT::DeviceMalloc(reinterpret_cast<void **>(&d_ptr),
                              N * sizeof(float));

  // Annotate the device pointer before the kernel launch so the preload can
  // attach metadata to the recorded snapshot.
  mneme::annotate(d_ptr, mneme::Metadata{
                             .builtin = mneme::BuiltinDType::F32,
                             .threshold = 0.001,
                             .threshold_kind = mneme::ThresholdKind::Relative,
                             .norm = mneme::Norm::Linf,
                             .tag = std::string("test_ptr"),
                         });

  annotation_kernel<<<1, N>>>(d_ptr, N);

  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Device error: " << EC.value() << "\n";
    MnemeDeviceRT::DeviceFree(d_ptr);
    return -1;
  }

  std::cout << "OK\n";
  MnemeDeviceRT::DeviceFree(d_ptr);
  return 0;
}

// clang-format off
// CHECK: OK
// CHECK-RR: DemangledName: annotation_kernel(float*, int)
// CHECK-RR: NumModules: 1
// CHECK-RR: NumInstances: 1
// CHECK-RR: BlobAnnotation: threshold=0.001 threshold_kind=1 builtin=9 norm=3 tag=test_ptr
// clang-format on
