// clang-format off
// RUN: rm -rf "%t.$$.mneme" "%t.$$.log" && mkdir -p "%t.$$.mneme" "%t.$$.log"
// Logs go to a file so they cannot split the device printf lines on stdout.
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_LOG_DIR="%t.$$.log" MNEME_DATA_DIR="%t.$$.mneme" %build/indirect_launcher_tpl_multi_arg%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR
// RUN: rm -rf "%t.$$.mneme" "%t.$$.log"
// clang-format on

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

// CHECK-RR: NumModules: 1
// CHECK-RR: NumInstances: 1

__global__ void kernel(int arg) { printf("Kernel one; arg = %d\n", arg); }

// CHECK-RR: NumModules: 1
// CHECK-RR: NumInstances: 1
__global__ void kernel_two(int arg) { printf("Kernel two; arg = %d\n", arg); }

template <typename T>
MnemeDeviceRT::DeviceError_t launcher(T kernel_in, int a) {
  void *args[] = {&a};
  return MnemeDeviceRT::deviceLaunchKernel((const void *)kernel_in, 1, 1, args,
                                           0, 0);
}

int main() {
  auto indirect = reinterpret_cast<const void *>(&kernel);
  auto EC = MnemeDeviceRT::DeviceErrorCheck(launcher(kernel, 42));
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }

  EC = MnemeDeviceRT::DeviceErrorCheck(launcher(kernel_two, 24));
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }

  return 0;
}

// CHECK: Kernel one; arg = 42
// CHECK: Kernel two; arg = 24
