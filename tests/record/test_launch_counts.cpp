// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_MAX_RECORDINGS=1 LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_PAGE_SIZE=%PG MNEME_DATA_DIR="%t.$$.mneme" %build/test_launch_counts%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-MAX
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_SKIP_RECORDINGS=10 LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_PAGE_SIZE=%PG MNEME_DATA_DIR="%t.$$.mneme" %build/test_launch_counts%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: ls "%t.$$.mneme" | %FILECHECK %s --allow-empty --check-prefix=CHECK-SKIP
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_SKIP_RECORDINGS=1 LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_PAGE_SIZE=%PG MNEME_DATA_DIR="%t.$$.mneme" %build/test_launch_counts%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-SKIP-ONE
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_RR_KERNELS="count_kernel" LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_PAGE_SIZE=%PG MNEME_DATA_DIR="%t.$$.mneme" %build/test_launch_counts%ext other | %FILECHECK %s --check-prefixes=CHECK,CHECK-OTHER
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-REGEX
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

// clang-format off
// CHECK-MAX: DemangledName: count_kernel()
// CHECK-MAX: NumInstances: 1
// CHECK-MAX: TotalLaunches: 5
// CHECK-MAX: UnrecordedLaunches: 2
// CHECK-MAX: Recorded: Grid:(1, 1, 1) Block:(32, 1, 1) SharedMem:0 Occurrences:3

// CHECK-SKIP-NOT: {{RecordedIR_|DeviceState|\.json}}

// CHECK-SKIP-ONE: NumInstances: 2
// CHECK-SKIP-ONE: TotalLaunches: 5
// CHECK-SKIP-ONE: UnrecordedLaunches: 0
// CHECK-SKIP-ONE-DAG: Recorded: Grid:(1, 1, 1) Block:(32, 1, 1) SharedMem:0 Occurrences:3
// CHECK-SKIP-ONE-DAG: Recorded: Grid:(2, 1, 1) Block:(64, 1, 1) SharedMem:0 Occurrences:2

// CHECK-REGEX-NOT: other_kernel
// CHECK-REGEX: DemangledName: count_kernel()
// CHECK-REGEX: NumInstances: 2
// CHECK-REGEX: TotalLaunches: 5
// CHECK-REGEX: UnrecordedLaunches: 0
// CHECK-REGEX-NOT: other_kernel
// clang-format on
__global__ void count_kernel() {}
__global__ void other_kernel() {}

static bool launch(dim3 GridDim, dim3 BlockDim) {
  count_kernel<<<GridDim, BlockDim>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  dim3 SmallGrid(1, 1, 1), SmallBlock(32, 1, 1);
  dim3 LargeGrid(2, 1, 1), LargeBlock(64, 1, 1);
  for (int I = 0; I < 2; I++)
    if (!launch(SmallGrid, SmallBlock) || !launch(LargeGrid, LargeBlock))
      return -1;
  if (!launch(SmallGrid, SmallBlock))
    return -1;

  printf("Launched count_kernel 5 times\n");

  if (argc > 1) {
    for (int I = 0; I < 4; I++)
      other_kernel<<<1, 1>>>();
    auto EC =
        MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
    if (EC) {
      std::cout << "Error when running benchmark " << EC.value() << "\n";
      return -1;
    }
    printf("Launched other_kernel 4 times\n");
  }
  return 0;
}

// CHECK: Launched count_kernel 5 times
// CHECK-OTHER: Launched other_kernel 4 times
