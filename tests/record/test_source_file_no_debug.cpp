// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_DATA_DIR="%t.$$.mneme" MNEME_COPY_SOURCE=1 %build/test_source_file_no_debug%ext > "%t.$$.out" 2>&1
// RUN: %FILECHECK %s --check-prefix=CHECK < "%t.$$.out"
// RUN: %FILECHECK %s --check-prefix=CHECK-WARN < "%t.$$.out"
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR
// RUN: ls "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-LS --implicit-check-not=RecordedSource_
// RUN: rm -rf "%t.$$.mneme" "%t.$$.out"
// clang-format on

// This test is built with -g0, so the recorded kernel has no line-table debug
// info. Recording it must omit every source field and warn instead, and must
// copy nothing even though MNEME_COPY_SOURCE is set.

#include <climits>
#include <cstdio>
#include <iostream>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

// CHECK-WARN: has no line-table debug info; compile with -gline-tables-only

// CHECK-RR:DemangledName: no_debug_kernel()
// CHECK-RR:NumInstances: 1
// CHECK-RR-NOT:Source
// CHECK-RR:BlockDims:

// CHECK-LS: .json
__global__ void no_debug_kernel() { printf("NoDebugKernel\n"); }

int main() {
  no_debug_kernel<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }
  return 0;
}

// CHECK: NoDebugKernel
