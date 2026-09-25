// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: env MNEME_MAX_RECORDINGS=1 LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_DATA_DIR="%t.$$.mneme" %build/test_crash_handling%ext > %t.out 2>&1 || true
// RUN: %FILECHECK %s --check-prefixes=CHECK < %t.out
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

// A crash while mneme is recording should be reported by the preload's signal
// handler rather than terminating silently.

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

__global__ void simple_kernel() {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Just a simple kernel that does nothing
  if (idx == 0) {
    printf("Kernel executing\n");
  }
}

int main(int argc, char** argv) {
  // Launch a simple kernel - this will initialize the MnemeRecorderHIPPreload
  // singleton via the preload library, which installs the crash handler
  dim3 blockDim(32, 1, 1);
  dim3 gridDim(1, 1, 1);
  simple_kernel<<<gridDim, blockDim>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running kernel " << EC.value() << std::endl;
    return -1;
  }

  // Use fprintf to stderr and flush to ensure message appears before crash
  fprintf(stderr, "Kernel completed successfully\n");
  fflush(stderr);

  // Now trigger a crash to test signal handler
  // CHECK: Kernel completed successfully
  // CHECK: === PROGRAM CRASHED (see core dump) ===

  // Trigger segmentation fault
  int* null_ptr = nullptr;
  *null_ptr = 42; // This will cause SIGSEGV, which should be caught by signal handler

  // Should never reach here
  fprintf(stderr, "ERROR: Should not reach this line\n");
  return 0;
}
