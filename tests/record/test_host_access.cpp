// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG MNEME_DATA_DIR="%t.$$.mneme" %build/test_host_access%ext > %t.out 2>&1 || true
// RUN: %FILECHECK %s --check-prefixes=CHECK < %t.out
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

// On an integrated GPU (e.g. MI300A) device allocations are host-accessible,
// and applications and GPU-aware MPI rely on it. Device memory handed out by
// mneme while recording must stay host-accessible in both directions.
// Discrete GPUs make no such guarantee, so the checks are skipped there.

#include <cstdio>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

__global__ void fill(int *Data, int N) {
  int Idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (Idx < N)
    Data[Idx] = 7 + Idx;
}

__global__ void readback(const int *Data, int *Out) { *Out = Data[5]; }

static bool isIntegrated() {
  int Integrated = 0;
#ifdef MNEME_ENABLE_HIP
  hipDeviceGetAttribute(&Integrated, hipDeviceAttributeIntegrated, 0);
#elif defined(MNEME_ENABLE_CUDA)
  cudaDeviceGetAttribute(&Integrated, cudaDevAttrIntegrated, 0);
#endif
  return Integrated;
}

int main() {
  constexpr int N = 1 << 20;
  int *Data = nullptr;
  int *Out = nullptr;
  auto EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceMalloc((void **)&Data, N * sizeof(int)));
  if (!EC)
    EC = MnemeDeviceRT::DeviceErrorCheck(
        MnemeDeviceRT::DeviceMalloc((void **)&Out, sizeof(int)));
  if (EC) {
    printf("Error allocating device memory %s\n", EC.value().c_str());
    return -1;
  }

  if (!isIntegrated()) {
    // CHECK: Host read: {{PASS|SKIPPED}}
    // CHECK: Host write: {{PASS|SKIPPED}}
    printf("Host read: SKIPPED (discrete GPU)\n");
    printf("Host write: SKIPPED (discrete GPU)\n");
    return 0;
  }

  fill<<<N / 256, 256>>>(Data, N);
  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    printf("Error when running kernel %s\n", EC.value().c_str());
    return -1;
  }
  fflush(stdout);

  int Wrong = 0;
  for (int I = 0; I < N; ++I)
    Wrong += ((volatile int *)Data)[I] != 7 + I;
  printf("Host read: %s (Data[5]=%d, expected 12, %d/%d wrong)\n",
         Wrong ? "FAIL" : "PASS", ((volatile int *)Data)[5], Wrong, N);
  fflush(stdout);

  ((volatile int *)Data)[5] = 4242;
  readback<<<1, 1>>>(Data, Out);
  int Seen = -1;
  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
      &Seen, Out, sizeof(int), MnemeDeviceRT::MemcpyDeviceToHostKind()));
  if (EC) {
    printf("Error copying result %s\n", EC.value().c_str());
    return -1;
  }
  printf("Host write: %s (device read %d, expected 4242)\n",
         Seen == 4242 ? "PASS" : "FAIL", Seen);

  MnemeDeviceRT::DeviceFree(Out);
  MnemeDeviceRT::DeviceFree(Data);
  return 0;
}
