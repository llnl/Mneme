// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_DATA_DIR="%t.$$.mneme" %build/kernel%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

#include <climits>
#include <cstdio>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

// CHECK-RR:DemangledName: kernel()
// CHECK-RR:NumModules: 1
// CHECK-RR:NumInstances: 1
__global__ void kernel() { printf("Kernel\n"); }

int main() {
  // CHECK-RR:BlockDims:(1, 1, 1)
  // CHECK-RR:GridDims:(1, 1, 1)
  kernel<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }
  return 0;
}

// CHECK: Kernel
