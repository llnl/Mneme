// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG MNEME_DATA_DIR="%t.$$.mneme" %build/test_passthrough_alloc%ext > %t.out 2>&1 || true
// RUN: %FILECHECK %s --check-prefixes=CHECK < %t.out
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

// Mneme serves only hipMalloc/cudaMalloc. Memory from the other allocation
// functions comes from the vendor runtime, and applications may release it
// with the device free (or the async/host free). Recording must forward those
// frees instead of aborting on an address it did not allocate.

#include <cstdio>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#ifdef MNEME_ENABLE_HIP
#include <hip/hip_runtime.h>
#define RT(Name) hip##Name
#define MAKE_EXTENT make_hipExtent
#elif defined(MNEME_ENABLE_CUDA)
#include <cuda_runtime.h>
#define RT(Name) cuda##Name
#define MAKE_EXTENT make_cudaExtent
#endif

using Error_t = RT(Error_t);

static int Failures = 0;

static void expectOK(Error_t EC, const char *What) {
  if (EC == RT(Success)) {
    printf("%s: OK\n", What);
  } else {
    ++Failures;
    printf("%s: FAIL (%s)\n", What, RT(GetErrorString)(EC));
  }
  fflush(stdout);
}

static void skip(const char *What) { printf("%s: SKIPPED\n", What); }

int main() {
  constexpr size_t Size = 1 << 20;
  void *P = nullptr;
  size_t Pitch = 0;

  expectOK(RT(MallocManaged)(&P, Size, RT(MemAttachGlobal)), "managed alloc");
  expectOK(RT(Free)(P), "managed free");

  expectOK(RT(MallocPitch)(&P, &Pitch, 1000, 100), "pitch alloc");
  expectOK(RT(Free)(P), "pitch free");

  RT(PitchedPtr) Pitched;
  expectOK(RT(Malloc3D)(&Pitched, MAKE_EXTENT(1000, 10, 10)), "3D alloc");
  expectOK(RT(Free)(Pitched.ptr), "3D free");

  RT(Stream_t) Stream;
  expectOK(RT(StreamCreate)(&Stream), "stream create");

  expectOK(RT(MallocAsync)(&P, Size, Stream), "async alloc");
  expectOK(RT(FreeAsync)(P, Stream), "async alloc, async free");
  expectOK(RT(MallocAsync)(&P, Size, Stream), "async alloc");
  expectOK(RT(StreamSynchronize)(Stream), "stream sync");
  expectOK(RT(Free)(P), "async alloc, device free");

  RT(MemPool_t) Pool;
  expectOK(RT(DeviceGetDefaultMemPool)(&Pool, 0), "default pool");
  expectOK(RT(MallocFromPoolAsync)(&P, Size, Pool, Stream), "pool alloc");
  expectOK(RT(FreeAsync)(P, Stream), "pool alloc, async free");
  expectOK(RT(MallocFromPoolAsync)(&P, Size, Pool, Stream), "pool alloc");
  expectOK(RT(StreamSynchronize)(Stream), "stream sync");
  expectOK(RT(Free)(P), "pool alloc, device free");

  // Memory allocated by Mneme may also be released with the async free.
  expectOK(RT(Malloc)(&P, Size), "device alloc");
  expectOK(RT(FreeAsync)(P, Stream), "device alloc, async free");

  expectOK(RT(StreamSynchronize)(Stream), "stream sync");
  expectOK(RT(StreamDestroy)(Stream), "stream destroy");

  expectOK(RT(HostAlloc)(&P, Size, RT(HostAllocDefault)), "host alloc");
  expectOK(RT(FreeHost)(P), "host alloc, host free");
  expectOK(RT(MallocHost)(&P, Size), "malloc host");
  expectOK(RT(FreeHost)(P), "malloc host, host free");

#ifdef MNEME_ENABLE_HIP
  // HIP also releases pinned host memory with hipFree.
  expectOK(hipHostMalloc(&P, Size, hipHostMallocDefault), "hip host malloc");
  expectOK(hipFree(P), "hip host malloc, device free");
  expectOK(hipHostMalloc(&P, Size, hipHostMallocDefault), "hip host malloc");
  expectOK(hipHostFree(P), "hip host malloc, host free");
  expectOK(hipHostAlloc(&P, Size, hipHostAllocDefault), "hip host alloc");
  expectOK(hipFree(P), "hip host alloc, device free");
  expectOK(hipMemAllocHost(&P, Size), "hip mem alloc host");
  expectOK(hipHostFree(P), "hip mem alloc host, host free");
  expectOK(hipExtMallocWithFlags(&P, Size, hipDeviceMallocFinegrained),
           "hip ext malloc");
  expectOK(hipFree(P), "hip ext malloc, device free");
  expectOK(hipMemAllocPitch(&P, &Pitch, 1000, 100, 4), "hip mem alloc pitch");
  expectOK(hipFree(P), "hip mem alloc pitch, device free");
#else
  skip("hip host malloc, device free");
  skip("hip host malloc, host free");
  skip("hip host alloc, device free");
  skip("hip mem alloc host, host free");
  skip("hip ext malloc, device free");
  skip("hip mem alloc pitch, device free");
#endif

  printf("Done: %d failures\n", Failures);
  return Failures != 0;
}

// CHECK: managed free: OK
// CHECK: pitch free: OK
// CHECK: 3D free: OK
// CHECK: async alloc, async free: OK
// CHECK: async alloc, device free: OK
// CHECK: pool alloc, async free: OK
// CHECK: pool alloc, device free: OK
// CHECK: device alloc, async free: OK
// CHECK: host alloc, host free: OK
// CHECK: malloc host, host free: OK
// CHECK: hip host malloc, device free: {{OK|SKIPPED}}
// CHECK: hip host malloc, host free: {{OK|SKIPPED}}
// CHECK: hip host alloc, device free: {{OK|SKIPPED}}
// CHECK: hip mem alloc host, host free: {{OK|SKIPPED}}
// CHECK: hip ext malloc, device free: {{OK|SKIPPED}}
// CHECK: hip mem alloc pitch, device free: {{OK|SKIPPED}}
// CHECK: Done: 0 failures
