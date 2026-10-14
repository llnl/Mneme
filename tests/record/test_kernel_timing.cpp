// clang-format off
// %TIMING is the timer under test, the profiler unless lit gets -D TIMING.
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_KERNEL_TIMING=%TIMING MNEME_MAX_RECORDINGS=1 MNEME_RR_KERNELS="count_kernel" LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_kernel_timing%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-MAX -DTIMING=%TIMING
// RUN: %CHECK_TIMING "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-INVARIANTS
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_KERNEL_TIMING=%TIMING MNEME_SKIP_RECORDINGS=1 MNEME_RR_KERNELS="count_kernel" LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_kernel_timing%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-SKIP
// RUN: %CHECK_TIMING "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-INVARIANTS
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_KERNEL_TIMING=%TIMING MNEME_RR_KERNELS="spin_kernel" LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_kernel_timing%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-SPIN
// RUN: %CHECK_TIMING "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-INVARIANTS
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: MNEME_KERNEL_TIMING=%TIMING MNEME_RR_KERNELS="count_kernel" LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_kernel_timing%ext capture 2>&1 | %FILECHECK %s --check-prefixes=CHECK-CAPTURED
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-CAPTURE
// RUN: %CHECK_TIMING "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-INVARIANTS
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: env -u MNEME_KERNEL_TIMING LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_DATA_DIR="%t.$$.mneme" %build/test_kernel_timing%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: cat "%t.$$.mneme"/*.json | %FILECHECK %s --check-prefix=CHECK-UNTIMED
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

// clang-format off
// The launch with an unrecorded configuration is timed but belongs to no
// instance, so it shows up as unrecorded time.
// CHECK-MAX: DemangledName: count_kernel()
// CHECK-MAX: TotalLaunches: 5
// CHECK-MAX: UnrecordedLaunches: 2
// CHECK-MAX: TimingSource: [[TIMING]]{{$}}
// CHECK-MAX: TotalTimeNs: {{[1-9][0-9]*$}}
// CHECK-MAX: UnrecordedTimeNs: {{[1-9][0-9]*$}}
// CHECK-MAX: BlockDims:(32, 1, 1)
// CHECK-MAX-NEXT: GridDims:(1, 1, 1)
// CHECK-MAX-NEXT: SharedMem:0 Occurrences:[[OCC:3]]
// CHECK-MAX-NEXT: Timing: TimedLaunches:[[OCC]] SumNs:{{[1-9][0-9]*}} MinNs:{{[1-9][0-9]*}} MaxNs:{{[1-9][0-9]*}} RecordedNs:{{[1-9][0-9]*$}}

// The skipped launch is timed into the instance that is recorded later.
// CHECK-SKIP: TotalLaunches: 5
// CHECK-SKIP: UnrecordedLaunches: 0
// CHECK-SKIP: UnrecordedTimeNs: {{0$}}
// CHECK-SKIP: SharedMem:{{.*}} Occurrences:[[OCC1:[0-9]+]]
// CHECK-SKIP-NEXT: Timing: TimedLaunches:[[OCC1]] SumNs:{{[1-9][0-9]*}}
// CHECK-SKIP: SharedMem:{{.*}} Occurrences:[[OCC2:[0-9]+]]
// CHECK-SKIP-NEXT: Timing: TimedLaunches:[[OCC2]] SumNs:{{[1-9][0-9]*}}

// Each launch spins for 1 ms, so anything below 0.5 ms is implausible.
// CHECK-SPIN: DemangledName: spin_kernel(unsigned long)
// CHECK-SPIN: TotalLaunches: 2
// CHECK-SPIN: UnrecordedTimeNs: {{0$}}
// CHECK-SPIN: SharedMem:{{.*}} Occurrences:2
// CHECK-SPIN-NEXT: Timing: TimedLaunches:2 SumNs:{{[0-9]+}} MinNs:{{([5-9][0-9]{5}|[1-9][0-9]{6,})}} MaxNs:

// Launches captured into a graph are counted but not timed, and the exit
// drain does not wait for them.
// CHECK-CAPTURED-NOT: No duration arrived
// CHECK-CAPTURED: Captured count_kernel twice
// CHECK-CAPTURED-NOT: No duration arrived
// CHECK-CAPTURE: TotalLaunches: 3
// CHECK-CAPTURE: Occurrences:3
// CHECK-CAPTURE-NEXT: Timing: TimedLaunches:1 SumNs:{{[0-9]+}} MinNs:{{[0-9]+}} MaxNs:{{[0-9]+}} RecordedNs:{{[0-9]+$}}

// CHECK-INVARIANTS-NOT: TimingError
// CHECK-INVARIANTS: TimingInvariants: ok

// CHECK-UNTIMED-NOT: {{TimingSource|TotalTimeNs|TimedLaunches|TimeNs|RecordedLaunchTimeNs}}
// clang-format on
__global__ void count_kernel() {}

__device__ uint64_t ticks() {
#ifdef MNEME_ENABLE_HIP
  return wall_clock64();
#else
  uint64_t Ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(Ns));
  return Ns;
#endif
}

__global__ void spin_kernel(uint64_t Ticks) {
  uint64_t Start = ticks();
  while (ticks() - Start < Ticks)
    ;
}

static bool sync() {
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return false;
  }
  return true;
}

static uint64_t ticksPerMillisecond() {
#ifdef MNEME_ENABLE_HIP
  int KHz = 0;
  if (hipDeviceGetAttribute(&KHz, hipDeviceAttributeWallClockRate, 0) !=
      hipSuccess)
    return 0;
  return KHz;
#else
  return 1000000;
#endif
}

static bool launchCaptured() {
#ifdef MNEME_ENABLE_HIP
  hipStream_t Stream;
  hipGraph_t Graph;
  hipGraphExec_t Exec;
  if (hipStreamCreate(&Stream) != hipSuccess)
    return false;
  count_kernel<<<1, 32, 0, Stream>>>();
  if (hipStreamSynchronize(Stream) != hipSuccess ||
      hipStreamBeginCapture(Stream, hipStreamCaptureModeGlobal) != hipSuccess)
    return false;
  count_kernel<<<1, 32, 0, Stream>>>();
  count_kernel<<<1, 32, 0, Stream>>>();
  if (hipStreamEndCapture(Stream, &Graph) != hipSuccess ||
      hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0) != hipSuccess ||
      hipGraphLaunch(Exec, Stream) != hipSuccess ||
      hipStreamSynchronize(Stream) != hipSuccess)
    return false;
  return hipGraphExecDestroy(Exec) == hipSuccess &&
         hipGraphDestroy(Graph) == hipSuccess &&
         hipStreamDestroy(Stream) == hipSuccess;
#else
  cudaStream_t Stream;
  cudaGraph_t Graph;
  cudaGraphExec_t Exec;
  if (cudaStreamCreate(&Stream) != cudaSuccess)
    return false;
  count_kernel<<<1, 32, 0, Stream>>>();
  if (cudaStreamSynchronize(Stream) != cudaSuccess ||
      cudaStreamBeginCapture(Stream, cudaStreamCaptureModeGlobal) !=
          cudaSuccess)
    return false;
  count_kernel<<<1, 32, 0, Stream>>>();
  count_kernel<<<1, 32, 0, Stream>>>();
  if (cudaStreamEndCapture(Stream, &Graph) != cudaSuccess ||
      cudaGraphInstantiate(&Exec, Graph, 0) != cudaSuccess ||
      cudaGraphLaunch(Exec, Stream) != cudaSuccess ||
      cudaStreamSynchronize(Stream) != cudaSuccess)
    return false;
  return cudaGraphExecDestroy(Exec) == cudaSuccess &&
         cudaGraphDestroy(Graph) == cudaSuccess &&
         cudaStreamDestroy(Stream) == cudaSuccess;
#endif
}

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "capture") == 0) {
    if (!launchCaptured()) {
      std::cout << "Error when capturing count_kernel\n";
      return -1;
    }
    printf("Captured count_kernel twice\n");
    return 0;
  }

  dim3 SmallGrid(1, 1, 1), SmallBlock(32, 1, 1);
  dim3 LargeGrid(2, 1, 1), LargeBlock(64, 1, 1);
  for (int I = 0; I < 2; I++) {
    count_kernel<<<SmallGrid, SmallBlock>>>();
    count_kernel<<<LargeGrid, LargeBlock>>>();
  }
  count_kernel<<<SmallGrid, SmallBlock>>>();
  if (!sync())
    return -1;

  uint64_t Ticks = ticksPerMillisecond();
  if (!Ticks)
    return -1;
  for (int I = 0; I < 2; I++)
    spin_kernel<<<1, 1>>>(Ticks);
  if (!sync())
    return -1;

  printf("Launched kernels\n");
  return 0;
}

// CHECK: Launched kernels
