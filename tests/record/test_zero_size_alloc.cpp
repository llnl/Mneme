// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG MNEME_DATA_DIR="%t.$$.mneme" %build/test_zero_size_alloc%ext > %t.out 2>&1 || true
// RUN: %FILECHECK %s --check-prefixes=CHECK < %t.out
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

// Applications may request zero bytes, e.g. for an empty array. The vendor
// runtimes return success for this, and recording must not abort or hand the
// same address to a later allocation.

#include <cstdio>

#ifdef MNEME_ENABLE_HIP
#include <hip/hip_runtime.h>
#define RT(Name) hip##Name
#elif defined(MNEME_ENABLE_CUDA)
#include <cuda_runtime.h>
#define RT(Name) cuda##Name
#endif

static int Failures = 0;

static void expectOK(RT(Error_t) EC, const char *What) {
  if (EC == RT(Success)) {
    printf("%s: OK\n", What);
  } else {
    ++Failures;
    printf("%s: FAIL (%s)\n", What, RT(GetErrorString)(EC));
  }
  fflush(stdout);
}

__global__ void fill(int *Data, int N) {
  int Idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (Idx < N)
    Data[Idx] = Idx;
}

int main() {
  constexpr int N = 1 << 10;
  void *Empty = nullptr;
  int *Data = nullptr;
  void *Empty2 = nullptr;

  expectOK(RT(Malloc)(&Empty, 0), "zero-size alloc");
  printf("zero-size alloc pointer: %p\n", Empty);
  expectOK(RT(Malloc)((void **)&Data, N * sizeof(int)),
           "alloc after zero-size");
  expectOK(RT(Malloc)(&Empty2, 0), "second zero-size alloc");

  if (Data && (Data == Empty || Data == Empty2)) {
    ++Failures;
    printf("zero-size and non-empty allocations share address %p\n", Data);
  }

  fill<<<N / 256, 256>>>(Data, N);
  expectOK(RT(DeviceSynchronize)(), "kernel");
  int Last = -1;
  expectOK(RT(Memcpy)(&Last, Data + N - 1, sizeof(int), RT(MemcpyDeviceToHost)),
           "copy back");
  if (Last != N - 1) {
    ++Failures;
    printf("kernel result: FAIL (got %d)\n", Last);
  }

  expectOK(RT(Free)(Empty), "zero-size free");
  expectOK(RT(Free)(Empty2), "second zero-size free");
  expectOK(RT(Free)(Data), "free");

  printf("Done: %d failures\n", Failures);
  return Failures != 0;
}

// CHECK: zero-size alloc: OK
// CHECK: alloc after zero-size: OK
// CHECK: second zero-size alloc: OK
// CHECK: kernel: OK
// CHECK: zero-size free: OK
// CHECK: second zero-size free: OK
// CHECK: free: OK
// CHECK: Done: 0 failures
