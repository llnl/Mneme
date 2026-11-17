// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// The read analysis always optimizes at O3, so the record run's Proteus opt
// level must not change which leaves are read.
// RUN: PROTEUS_OPT_LEVEL=0 MNEME_PROFILE_ARG_VALUES=1 MNEME_FILTER_MODE=scoped LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_arg_values%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %PYTHON %S/read-arg-values.py "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-PROFILE
// RUN: %PYTHON %S/read-arg-values.py "%t.$$.mneme" --top 6 | %FILECHECK %s --check-prefix=CHECK-INF
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_PROFILE_ARG_VALUES=1 MNEME_ARG_VALUE_CAP=4 MNEME_FILTER_MODE=scoped LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_arg_values%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %PYTHON %S/read-arg-values.py "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-CAP
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_PROFILE_ARG_VALUES=1 MNEME_MAX_RECORDINGS=1 LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_arg_values%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RECORD
// RUN: %PYTHON %S/read-arg-values.py "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-PROFILE
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_PROFILE_ARG_VALUES=1 MNEME_KERNEL_TIMING=profiler MNEME_FILTER_MODE=scoped LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_arg_values%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %PYTHON %S/read-arg-values.py "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-TIMING
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_FILTER_MODE=scoped LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_arg_values%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: ls "%t.$$.mneme" | %FILECHECK %s --allow-empty --check-prefix=CHECK-OFF
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <new>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

// clang-format off
// Kernels are listed by demangled name, leaves in argument layout order.
// The box kernel reads only lo[0] and hi[0].
// CHECK-PROFILE: Kernel: box_kernel(Box, double*) Launches: 3
// CHECK-PROFILE-NEXT: Leaf:arg0.0[0] Type:i32 NumDistinctValues:1 Untracked:0 TopValues:0(3)
// CHECK-PROFILE-NEXT: Leaf:arg0.1[0] Type:i32 NumDistinctValues:2 Untracked:0 TopValues:7(2),15(1)
// CHECK-PROFILE-NEXT: Leaf:out{{.*}} Type:ptr{{.*}} NumDistinctValues:1 Untracked:0
// CHECK-PROFILE: Kernel: params_kernel(Params, long) Launches: 4
// CHECK-PROFILE-NEXT: Leaf:arg0.0 Type:i32 NumDistinctValues:1 Untracked:0 TopValues:100(4)
// CHECK-PROFILE-NEXT: Leaf:arg0.1 Type:i8 NumDistinctValues:1 Untracked:0 TopValues:1(4)
// CHECK-PROFILE-NEXT: Leaf:arg0.2 Type:double NumDistinctValues:1 Untracked:0 TopValues:0.25(4)
// CHECK-PROFILE-NEXT: Leaf:arg0.3 Type:ptr{{.*}} NumDistinctValues:1 Untracked:0
// CHECK-PROFILE-NEXT: Leaf:step Type:i64 NumDistinctValues:2 Untracked:0 TopValues:0(2),1(2)
// Garbage in the explicit tail padding must not count as different values.
// CHECK-PROFILE: Kernel: resource_kernel(Resource) Launches: 3
// CHECK-PROFILE-NEXT: Leaf:arg0.0 Type:ptr{{.*}} NumDistinctValues:1 Untracked:0
// CHECK-PROFILE-NEXT: Leaf:arg0.1 Type:i32 NumDistinctValues:1 Untracked:0 TopValues:3(3)
// CHECK-PROFILE-NEXT: Unsupported:arg0.2: [4 x i8] (padding or char array)
// CHECK-PROFILE: Kernel: scale_kernel(double*, double, int) Launches: 6
// CHECK-PROFILE-NEXT: Leaf:out{{.*}} Type:ptr{{.*}} NumDistinctValues:1 Untracked:0
// CHECK-PROFILE-NEXT: Leaf:factor Type:double NumDistinctValues:6 Untracked:0 TopValues:0.5(1),1(1),1.5(1)
// CHECK-PROFILE-NEXT: Leaf:n Type:i32 NumDistinctValues:3 Untracked:0 TopValues:64(3),128(2),256(1)
// CHECK-PROFILE: Kernel: void forall_kernel<main::'lambda'(int)>(main::'lambda'(int), int){{.*}} Launches: 2
// CHECK-PROFILE-NEXT: Leaf:arg0.0 Type:ptr{{.*}} NumDistinctValues:1 Untracked:0
// CHECK-PROFILE-NEXT: Leaf:arg0.1 Type:double NumDistinctValues:1 Untracked:0 TopValues:2(2)
// CHECK-PROFILE-NEXT: Leaf:n Type:i32 NumDistinctValues:1 Untracked:0 TopValues:64(2)
// The byte of the empty functor takes three values but is not read, so it is
// not profiled.
// CHECK-PROFILE: Kernel: void reduce_kernel<Plus>(double*, Plus) Launches: 3
// CHECK-PROFILE-NEXT: Leaf:out{{.*}} Type:ptr{{.*}} NumDistinctValues:1 Untracked:0
// CHECK-PROFILE-NOT: Leaf:
// CHECK-PROFILE-NOT: Unsupported

// CHECK-INF: Leaf:factor {{.*}} TopValues:0.5(1),1(1),1.5(1),2(1),2.5(1),inf(1){{$}}

// Launches with values past the cap are counted only as untracked.
// CHECK-CAP: Kernel: scale_kernel(double*, double, int) Launches: 6
// CHECK-CAP: Leaf:factor Type:double NumDistinctValues:4 Untracked:2
// CHECK-CAP-NEXT: Leaf:n Type:i32 NumDistinctValues:3 Untracked:0 TopValues:64(3),128(2),256(1)

// Kernels that record nothing still report their timing in the profile.
// CHECK-TIMING: Kernel: scale_kernel(double*, double, int) Launches: 6
// CHECK-TIMING-NEXT: Timing: TimedLaunches:6 TotalTimeNs:{{[1-9][0-9]*$}}

// CHECK-RECORD: DemangledName: scale_kernel(double*, double, int)
// CHECK-RECORD: TotalLaunches: 6

// CHECK-OFF-NOT: profile
// clang-format on

struct Params {
  int nx;
  char flag;
  double h;
  float *buf;
};

// The constructor makes Clang give the struct explicit tail padding.
struct Resource {
  __host__ __device__ Resource(float *buf, int device)
      : buf(buf), device(device) {}
  float *buf;
  int device;
};

struct Box {
  int lo[3];
  int hi[3];
};

__global__ void scale_kernel(double *out, double factor, int n) {
  if (threadIdx.x < n)
    out[threadIdx.x] *= factor;
}

__global__ void params_kernel(Params p, long step) {
  if (threadIdx.x == 0 && p.flag)
    p.buf[0] += p.h * step + p.nx;
}

__global__ void box_kernel(Box b, double *out) {
  if (threadIdx.x == 0)
    out[0] = b.lo[0] + b.hi[0];
}

__global__ void resource_kernel(Resource r) {
  if (threadIdx.x == 0)
    r.buf[0] += r.device;
}

// Like rocPRIM's binary operators: an empty class still occupies one byte that
// nothing writes.
struct Plus {
  __device__ double operator()(double A, double B) const { return A + B; }
};

template <typename Op> __global__ void reduce_kernel(double *out, Op op) {
  if (threadIdx.x == 0)
    out[0] = op(out[0], out[1]);
}

template <typename F> __global__ void forall_kernel(F f, int n) {
  int i = threadIdx.x;
  if (i < n)
    f(i);
}

static bool synchronize() {
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return false;
  }
  return true;
}

int main() {
  double *Out;
  float *Buf;
  MnemeDeviceRT::DeviceMalloc((void **)&Out, 256 * sizeof(double));
  MnemeDeviceRT::DeviceMalloc((void **)&Buf, sizeof(float));
  MnemeDeviceRT::DeviceMemset(Out, 0, 256 * sizeof(double));
  MnemeDeviceRT::DeviceMemset(Buf, 0, sizeof(float));

  const int Sizes[] = {64, 64, 64, 128, 128, 256};
  // The last factor checks that infinity still makes valid JSON.
  for (int I = 0; I < 6; I++)
    scale_kernel<<<1, 256>>>(Out, I < 5 ? 0.5 * (I + 1) : INFINITY, Sizes[I]);

  for (int I = 0; I < 4; I++) {
    Params P;
    // Different garbage in the padding must not count as different values.
    std::memset(&P, 0x10 + I, sizeof(P));
    P.nx = 100;
    P.flag = 1;
    P.h = 0.25;
    P.buf = Buf;
    int Block = I < 2 ? 32 : 64;
    params_kernel<<<1, Block>>>(P, I % 2);
  }

  const int His[] = {7, 15, 7};
  for (int I = 0; I < 3; I++) {
    Box B = {{0, 1, 2}, {His[I], 8, 9}};
    box_kernel<<<I == 0 ? 1 : 2, 32>>>(B, Out);
  }

  for (int I = 0; I < 3; I++) {
    alignas(Resource) unsigned char Storage[sizeof(Resource)];
    std::memset(Storage, 0x10 + I, sizeof(Storage));
    new (Storage) Resource(Buf, 3);
    // An explicit argument array passes the garbage padding to the launch.
    void *Args[] = {Storage};
    MnemeDeviceRT::deviceLaunchKernel((const void *)resource_kernel, dim3(1),
                                      dim3(32), Args, 0, nullptr);
  }

  for (int I = 0; I < 3; I++) {
    // The byte of the empty functor holds different garbage on every launch.
    unsigned char Op = 0x10 + I;
    void *Args[] = {&Out, &Op};
    MnemeDeviceRT::deviceLaunchKernel((const void *)reduce_kernel<Plus>,
                                      dim3(1), dim3(32), Args, 0, nullptr);
  }

  double Alpha = 2.0;
  double *X = Out;
  auto Scale = [=] __device__(int i) { X[i] *= Alpha; };
  for (int I = 0; I < 2; I++)
    forall_kernel<<<1, 64>>>(Scale, 64);

  if (!synchronize())
    return -1;
  printf("Launched all kernels\n");
  MnemeDeviceRT::DeviceFree(Out);
  MnemeDeviceRT::DeviceFree(Buf);
  return 0;
}

// CHECK: Launched all kernels
