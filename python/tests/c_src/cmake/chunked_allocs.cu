#include <cstdio>
#include <cstdlib>
#include <vector>

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)

#ifdef __ENABLE_CUDA__
#include <cuda_runtime.h>
#define DEV_PREFIX cuda
#elif defined(__ENABLE_HIP__)
#include <hip/hip_runtime.h>
#define DEV_PREFIX hip
#endif

#define prefix(name) CAT(DEV_PREFIX, name)

// Sums the ends of each buffer through a device-resident pointer table and
// bumps them.
__global__ void ends(int **Table, const size_t *Lens, int NumBufs, long *Out) {
  int B = blockIdx.x * blockDim.x + threadIdx.x;
  if (B >= NumBufs)
    return;
  int *Buf = Table[B];
  size_t N = Lens[B];
  Out[B] = Buf[0] + Buf[N - 1];
  Buf[0] += 1;
  Buf[N - 1] += 1;
}

static void *alloc(size_t Bytes) {
  void *P = nullptr;
  if (prefix(Malloc)(&P, Bytes) != prefix(Success)) {
    printf("malloc failed\n");
    exit(1);
  }
  return P;
}

int main() {
  // Freed right away, so later buffers reuse the kept chunk.
  for (int I = 0; I < 50; ++I) {
    prefix(Free)(alloc(16 << 20));
    prefix(Free)(alloc(1000));
  }

  constexpr size_t MiB = 1 << 20;
  std::vector<size_t> Sizes = {100,           3000,     MiB,
                               4 * MiB + 100, 9 * MiB,  16 * MiB,
                               33 * MiB + 4,  64 * MiB, 5 * MiB};
  std::vector<int *> Bufs;
  for (size_t Bytes : Sizes)
    Bufs.push_back(static_cast<int *>(alloc(Bytes)));

  // Frees two neighbors so the next buffer best fits their merged range.
  prefix(Free)(Bufs[4]);
  prefix(Free)(Bufs[5]);
  Sizes.erase(Sizes.begin() + 4, Sizes.begin() + 6);
  Bufs.erase(Bufs.begin() + 4, Bufs.begin() + 6);
  Sizes.push_back(24 * MiB);
  Bufs.push_back(static_cast<int *>(alloc(Sizes.back())));

  int NumBufs = Bufs.size();
  std::vector<size_t> Lens(NumBufs);
  for (int B = 0; B < NumBufs; ++B) {
    Lens[B] = Sizes[B] / sizeof(int);
    int First = B * 1000, Last = B * 1000 + 7;
    prefix(Memset)(Bufs[B], 0, Sizes[B]);
    prefix(Memcpy)(Bufs[B], &First, sizeof(int), prefix(MemcpyHostToDevice));
    prefix(Memcpy)(Bufs[B] + Lens[B] - 1, &Last, sizeof(int),
                   prefix(MemcpyHostToDevice));
  }

  int **Table = static_cast<int **>(alloc(NumBufs * sizeof(int *)));
  prefix(Memcpy)(Table, Bufs.data(), NumBufs * sizeof(int *),
                 prefix(MemcpyHostToDevice));
  size_t *DLens = static_cast<size_t *>(alloc(NumBufs * sizeof(size_t)));
  prefix(Memcpy)(DLens, Lens.data(), NumBufs * sizeof(size_t),
                 prefix(MemcpyHostToDevice));
  long *Out = static_cast<long *>(alloc(NumBufs * sizeof(long)));
  prefix(Memset)(Out, 0, NumBufs * sizeof(long));

  ends<<<1, 64>>>(Table, DLens, NumBufs, Out);
  prefix(DeviceSynchronize)();

  std::vector<long> HOut(NumBufs);
  prefix(Memcpy)(HOut.data(), Out, NumBufs * sizeof(long),
                 prefix(MemcpyDeviceToHost));
  int Bad = 0;
  for (int B = 0; B < NumBufs; ++B)
    Bad += HOut[B] != 2 * B * 1000 + 7;
  printf("%s\n", Bad ? "FAIL" : "PASS");
  return Bad != 0;
}
