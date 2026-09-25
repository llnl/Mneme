// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_SKIP_RECORDINGS=2 MNEME_MAX_RECORDINGS=2 LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_DATA_DIR="%t.$$.mneme" %build/test_skip_recordings%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

#include <cstdio>
#include <iostream>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

// CHECK-RR:DemangledName: kernel()
// CHECK-RR:NumModules: 1
// CHECK-RR:NumInstances: 2
__global__ void kernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("Kernel launch BlockDim: %d GridDim: %d\n", (int)blockDim.x,
           (int)gridDim.x);
}

int main() {
  for (int tid = 1; tid <= 4; tid++) {
    dim3 blockDim(tid * 32, 1, 1);
    dim3 gridDim(tid, 1, 1);
    kernel<<<gridDim, blockDim>>>();
    auto EC =
        MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
    if (EC) {
      std::cout << "Error when running benchmark " << EC.value() << "\n";
      return -1;
    }
  }
  return 0;
}

// clang-format off
// CHECK:Kernel launch BlockDim: 32 GridDim: 1
// CHECK:Kernel launch BlockDim: 64 GridDim: 2
// CHECK:Kernel launch BlockDim: 96 GridDim: 3
// CHECK:Kernel launch BlockDim: 128 GridDim: 4

// CHECK-RR-DAG:BlockDims:(96, 1, 1)
// CHECK-RR-DAG:GridDims:(3, 1, 1)
// CHECK-RR-DAG:BlockDims:(128, 1, 1)
// CHECK-RR-DAG:GridDims:(4, 1, 1)

// clang-format on
