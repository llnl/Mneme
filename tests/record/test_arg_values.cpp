// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// The read analysis always optimizes at O3, so the record run's Proteus opt
// level must not change which leaves are read.
// RUN: PROTEUS_OPT_LEVEL=0 MNEME_PROFILE_ARG_VALUES=1 MNEME_FILTER_MODE=scoped LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_arg_values%ext 2> "%t.$$.err" | %FILECHECK %s --check-prefixes=CHECK
// RUN: %FILECHECK %s --check-prefix=CHECK-WARN --input-file="%t.$$.err"
// RUN: %FILECHECK %s --check-prefix=CHECK-JSON --input-file="%t.$$.mneme/profile/ArgValues.json"
// RUN: %PYTHON %S/read-arg-values.py "%t.$$.mneme" --signatures | %FILECHECK %s --check-prefix=CHECK-SIG
// RUN: %PYTHON %S/read-arg-values.py "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-PROFILE
// RUN: %PYTHON %S/read-arg-values.py "%t.$$.mneme" --top 6 | %FILECHECK %s --check-prefix=CHECK-INF
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_PROFILE_ARG_VALUES=1 MNEME_ARG_VALUE_CAP=4 MNEME_FILTER_MODE=scoped LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_arg_values%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %PYTHON %S/read-arg-values.py "%t.$$.mneme" --signatures | %FILECHECK %s --check-prefix=CHECK-CAP
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
// RUN: rm -rf "%t.$$.mneme" "%t.$$.err"
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
// Kernels are listed by demangled name; argument leaves come first, in layout
// order, then the launch leaves. Pointers are not leaves.
// The box kernel reads only lo[0] and hi[0].
// CHECK-PROFILE: Kernel: box_kernel(Box, double*) Launches: 3 Signatures: 3 Untracked: 0
// CHECK-PROFILE-NEXT: Leaf:arg0.0[0] Type:i32 NumDistinctValues:1 Overflow:0 TopValues:0(3)
// CHECK-PROFILE-NEXT: Leaf:arg0.1[0] Type:i32 NumDistinctValues:2 Overflow:0 TopValues:7(2),15(1)
// CHECK-PROFILE-NEXT: Leaf:GridDims.x Type:i32 NumDistinctValues:2 Overflow:0 TopValues:2(2),1(1)
// CHECK-PROFILE-NEXT: Leaf:GridDims.y Type:i32 NumDistinctValues:1 Overflow:0 TopValues:1(3)
// CHECK-PROFILE-NEXT: Leaf:GridDims.z Type:i32 NumDistinctValues:1 Overflow:0 TopValues:1(3)
// CHECK-PROFILE-NEXT: Leaf:BlockDims.x Type:i32 NumDistinctValues:1 Overflow:0 TopValues:32(3)
// CHECK-PROFILE-NEXT: Leaf:BlockDims.y Type:i32 NumDistinctValues:1 Overflow:0 TopValues:1(3)
// CHECK-PROFILE-NEXT: Leaf:BlockDims.z Type:i32 NumDistinctValues:1 Overflow:0 TopValues:1(3)
// CHECK-PROFILE-NEXT: Leaf:SharedMem Type:i64 NumDistinctValues:1 Overflow:0 TopValues:0(3)
// CHECK-PROFILE: Kernel: params_kernel(Params, long) Launches: 4 Signatures: 4 Untracked: 0
// CHECK-PROFILE-NEXT: Leaf:arg0.0 Type:i32 NumDistinctValues:1 Overflow:0 TopValues:100(4)
// CHECK-PROFILE-NEXT: Leaf:arg0.1 Type:i8 NumDistinctValues:1 Overflow:0 TopValues:1(4)
// CHECK-PROFILE-NEXT: Leaf:arg0.2 Type:double NumDistinctValues:1 Overflow:0 TopValues:0.25(4)
// CHECK-PROFILE-NEXT: Leaf:step Type:i64 NumDistinctValues:2 Overflow:0 TopValues:0(2),1(2)
// CHECK-PROFILE-NEXT: Leaf:GridDims.x
// CHECK-PROFILE: Leaf:BlockDims.x Type:i32 NumDistinctValues:2 Overflow:0 TopValues:32(2),64(2)
// Garbage in the explicit tail padding must not count as different values.
// CHECK-PROFILE: Kernel: resource_kernel(Resource) Launches: 3 Signatures: 1 Untracked: 0
// CHECK-PROFILE-NEXT: Leaf:arg0.1 Type:i32 NumDistinctValues:1 Overflow:0 TopValues:3(3)
// CHECK-PROFILE-NEXT: Leaf:GridDims.x
// CHECK-PROFILE: Kernel: scale_kernel(double*, double, int) Launches: 6 Signatures: 6 Untracked: 0
// CHECK-PROFILE-NEXT: Leaf:factor Type:double NumDistinctValues:6 Overflow:0 TopValues:0.5(1),1.0(1),1.5(1)
// CHECK-PROFILE-NEXT: Leaf:n Type:i32 NumDistinctValues:3 Overflow:0 TopValues:64(3),128(2),256(1)
// CHECK-PROFILE-NEXT: Leaf:GridDims.x
// Launches whose signature is new once the signature table is full are
// untracked, but still counted per leaf.
// CHECK-PROFILE: Kernel: sweep_kernel(double*, int, int, int) Launches: 4352 Signatures: 4096 Untracked: 256
// CHECK-PROFILE-NEXT: Leaf:a Type:i32 NumDistinctValues:17 Overflow:0 TopValues:0(256),1(256),2(256)
// CHECK-PROFILE-NEXT: Leaf:b Type:i32 NumDistinctValues:16 Overflow:0 TopValues:0(272),1(272),2(272)
// CHECK-PROFILE-NEXT: Leaf:c Type:i32 NumDistinctValues:16 Overflow:0 TopValues:0(272),1(272),2(272)
// CHECK-PROFILE-NEXT: Leaf:GridDims.x
// Reads that cannot be followed keep every leaf of the argument.
// CHECK-PROFILE: Kernel: table_kernel(Table, double*) Launches: 2 Signatures: 1 Untracked: 0
// CHECK-PROFILE-NEXT: Leaf:arg0.0 Type:i32 NumDistinctValues:1 Overflow:0 TopValues:4(2)
// CHECK-PROFILE-NEXT: Leaf:arg0.1[0] Type:float NumDistinctValues:1 Overflow:0 TopValues:0.1(2)
// CHECK-PROFILE-NEXT: Leaf:arg0.1[1] Type:float NumDistinctValues:1 Overflow:0 TopValues:0.5(2)
// CHECK-PROFILE-NEXT: Leaf:arg0.1[2] Type:float NumDistinctValues:1 Overflow:0 TopValues:1.5(2)
// CHECK-PROFILE-NEXT: Leaf:arg0.1[3] Type:float NumDistinctValues:1 Overflow:0 TopValues:2.5(2)
// CHECK-PROFILE-NEXT: Leaf:GridDims.x
// CHECK-PROFILE: Kernel: void forall_kernel<main::'lambda'(int)>(main::'lambda'(int), int){{.*}} Launches: 2 Signatures: 1 Untracked: 0
// CHECK-PROFILE-NEXT: Leaf:arg0.1 Type:double NumDistinctValues:1 Overflow:0 TopValues:2.0(2)
// CHECK-PROFILE-NEXT: Leaf:n Type:i32 NumDistinctValues:1 Overflow:0 TopValues:64(2)
// CHECK-PROFILE-NEXT: Leaf:GridDims.x
// The byte of the empty functor takes three values but is not read, so the
// kernel has only launch leaves.
// CHECK-PROFILE: Kernel: void reduce_kernel<Plus>(double*, Plus) Launches: 3 Signatures: 1 Untracked: 0
// CHECK-PROFILE-NEXT: Leaf:GridDims.x

// CHECK-INF: Leaf:factor {{.*}} TopValues:0.5(1),1.0(1),1.5(1),2.0(1),2.5(1),inf(1){{$}}

// Floats are stored as their bits: 0.5, infinity, and 0.1f.
// CHECK-JSON-DAG: ["0x3fe0000000000000",1]
// CHECK-JSON-DAG: ["0x7ff0000000000000",1]
// CHECK-JSON-DAG: ["0x3dcccccd",2]

// Signatures tie argument values to launch dims, in first-seen order.
// CHECK-SIG: Kernel: box_kernel(Box, double*)
// CHECK-SIG: Signature:0,7,1,1,1,32,1,1,0 Launches:1
// CHECK-SIG-NEXT: Signature:0,15,2,1,1,32,1,1,0 Launches:1
// CHECK-SIG-NEXT: Signature:0,7,2,1,1,32,1,1,0 Launches:1
// CHECK-SIG: Kernel: params_kernel(Params, long)
// CHECK-SIG: Signature:100,1,0.25,0,1,1,1,32,1,1,0 Launches:1
// CHECK-SIG-NEXT: Signature:100,1,0.25,1,1,1,1,32,1,1,0 Launches:1
// CHECK-SIG-NEXT: Signature:100,1,0.25,0,1,1,1,64,1,1,0 Launches:1
// CHECK-SIG-NEXT: Signature:100,1,0.25,1,1,1,1,64,1,1,0 Launches:1
// CHECK-SIG: Kernel: void reduce_kernel<Plus>(double*, Plus)
// CHECK-SIG: Signature:1,1,1,32,1,1,0 Launches:3

// The fifth and sixth factors arrive after the value cap of four, so they
// share the overflow index.
// CHECK-CAP: Kernel: scale_kernel(double*, double, int) Launches: 6 Signatures: 6 Untracked: 0
// CHECK-CAP-NEXT: Leaf:factor Type:double NumDistinctValues:4 Overflow:2 TopValues:0.5(1),1.0(1),1.5(1)
// CHECK-CAP-NEXT: Leaf:n Type:i32 NumDistinctValues:3 Overflow:0 TopValues:64(3),128(2),256(1)
// CHECK-CAP: Signature:*,128,1,1,1,256,1,1,0 Launches:1
// CHECK-CAP-NEXT: Signature:*,256,1,1,1,256,1,1,0 Launches:1
// CHECK-CAP: Kernel: sweep_kernel(double*, int, int, int) Launches: 4352 Signatures: 125 Untracked: 0
// CHECK-CAP-NEXT: Leaf:a Type:i32 NumDistinctValues:4 Overflow:3328 TopValues:0(256),1(256),2(256)
// CHECK-CAP-NEXT: Leaf:b Type:i32 NumDistinctValues:4 Overflow:3264 TopValues:0(272),1(272),2(272)
// CHECK-CAP-NEXT: Leaf:c Type:i32 NumDistinctValues:4 Overflow:3264 TopValues:0(272),1(272),2(272)

// CHECK-WARN: [mneme] Argument value profile keeps every leaf of argument arg0 of kernel {{.*}}table_kernel{{.*}} because its reads cannot be followed
// CHECK-WARN-NOT: table_kernel

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

// The variable index keeps the read analysis from following the reads of t.
struct Table {
  int n;
  float w[4];
};

__global__ void table_kernel(Table t, double *out) {
  if (threadIdx.x < t.n)
    out[threadIdx.x] = t.w[threadIdx.x % 4];
}

__global__ void sweep_kernel(double *out, int a, int b, int c) {
  if (threadIdx.x == 0)
    out[0] += a + b + c;
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

  for (int I = 0; I < 2; I++) {
    Table T = {4, {0.1f, 0.5f, 1.5f, 2.5f}};
    table_kernel<<<1, 32>>>(T, Out);
  }

  // 17 * 16 * 16 distinct signatures: the first 4096 fill the table and the
  // 256 launches with A == 16 are untracked.
  for (int A = 0; A < 17; A++)
    for (int B = 0; B < 16; B++)
      for (int C = 0; C < 16; C++)
        sweep_kernel<<<1, 32>>>(Out, A, B, C);

  if (!synchronize())
    return -1;
  printf("Launched all kernels\n");
  MnemeDeviceRT::DeviceFree(Out);
  MnemeDeviceRT::DeviceFree(Buf);
  return 0;
}

// CHECK: Launched all kernels
