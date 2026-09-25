// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_FILTER_MODE=config MNEME_FILTER_CONFIG="%S/Inputs/source_file_kernel.json" LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_DATA_DIR="%t.$$.mneme" MNEME_COPY_SOURCE=1 %build/test_source_file%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR-SINGLE
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_FILTER_MODE=config MNEME_FILTER_CONFIG="%S/Inputs/multi_line_kernel.json" LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_DATA_DIR="%t.$$.mneme" MNEME_COPY_SOURCE=1 %build/test_source_file%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR-MULTI
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_FILTER_MODE=config MNEME_FILTER_CONFIG="%S/Inputs/lambda_kernel.json" LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_DATA_DIR="%t.$$.mneme" MNEME_COPY_SOURCE=1 %build/test_source_file%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR-LAMBDA
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

// Recording a kernel defined in its translation unit must report the absolute
// source path and, with MNEME_COPY_SOURCE, copy the file into the record dir.
// The helper and the lambda body are defined after the kernels that use them
// so a range that leaks past a kernel's closing brace fails SourceEndText.

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

__device__ __forceinline__ int triangular(int N);

// CHECK-RR-SINGLE:DemangledName: source_file_kernel()
// CHECK-RR-SINGLE:SourceFile: {{.*}}/test_source_file.cpp
// CHECK-RR-SINGLE:SourceLine: [[SLINE:[0-9]+]]{{$}}
// CHECK-RR-SINGLE:SourceEndLine: [[SLINE]]{{$}}
// CHECK-RR-SINGLE:SourceCopy: RecordedSource_{{[0-9a-f]+}}_test_source_file.cpp
// CHECK-RR-SINGLE:SourceMD5: ok
// CHECK-RR-SINGLE:SourceText: __global__ void source_file_kernel()
// CHECK-RR-SINGLE:SourceEndText: __global__ void source_file_kernel()
__global__ void source_file_kernel() { printf("Kernel\n"); }

// CHECK-RR-MULTI:DemangledName: multi_line_kernel()
// CHECK-RR-MULTI:SourceFile: {{.*}}/test_source_file.cpp
// CHECK-RR-MULTI:SourceCopy: RecordedSource_{{[0-9a-f]+}}_test_source_file.cpp
// CHECK-RR-MULTI:SourceMD5: ok
// CHECK-RR-MULTI:SourceText: __global__ void multi_line_kernel() {
// CHECK-RR-MULTI:SourceEndText: } // multi_line_kernel end
__global__ void multi_line_kernel() {
  int Sum = 0;
  for (int I = 0; I < 4; ++I)
    Sum += triangular(I);
  printf("Multi %d\n", Sum);
} // multi_line_kernel end

__device__ __forceinline__ int triangular(int N) {
  int Sum = 0;
  for (int I = 0; I <= N; ++I)
    Sum += I;
  return Sum;
}

// CHECK-RR-LAMBDA:DemangledName: {{.*}}lambda_kernel<
// CHECK-RR-LAMBDA:SourceFile: {{.*}}/test_source_file.cpp
// CHECK-RR-LAMBDA:SourceCopy: RecordedSource_{{[0-9a-f]+}}_test_source_file.cpp
// CHECK-RR-LAMBDA:SourceMD5: ok
// CHECK-RR-LAMBDA:SourceText: __global__ void lambda_kernel(DeviceCallable
// CHECK-RR-LAMBDA:SourceEndText: } // lambda_kernel end
template <typename DeviceCallable>
__global__ void lambda_kernel(DeviceCallable Func) {
  Func();
} // lambda_kernel end

int main() {
  source_file_kernel<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }

  multi_line_kernel<<<1, 1>>>();
  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }

  auto Body = [=] __device__() { printf("Lambda\n"); };
  lambda_kernel<<<1, 1>>>(Body);
  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }

  return 0;
}

// CHECK: Kernel
// CHECK: Multi 10
// CHECK: Lambda
