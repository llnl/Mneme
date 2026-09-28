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

// Sums each buffer through a device-resident pointer table and bumps its data.
__global__ void gather(int **Table, const int *Lens, int NumBufs,
                       const int *Big, long *Out) {
  int B = blockIdx.x;
  if (B >= NumBufs)
    return;
  int *Buf = Table[B];
  if (threadIdx.x == 0) {
    long Sum = Big[B];
    for (int I = 0; I < Lens[B]; ++I) {
      Sum += Buf[I];
      Buf[I] += 1;
    }
    Out[B] = Sum;
  }
}

static int *alloc(size_t Bytes) {
  void *P = nullptr;
  if (prefix(Malloc)(&P, Bytes) != prefix(Success)) {
    printf("malloc failed\n");
    exit(1);
  }
  return static_cast<int *>(P);
}

int main() {
  // Filled and emptied first so chunks are unmapped and their addresses reused.
  std::vector<void *> Churn;
  for (int I = 0; I < 3000; ++I)
    Churn.push_back(alloc(4096));
  for (void *P : Churn)
    prefix(Free)(P);

  // Freed before the launch, so the first recorded blob of its chunk is not
  // page aligned.
  int *Head = alloc(256);

  constexpr int NumBufs = 48;
  std::vector<int *> Bufs(NumBufs);
  std::vector<int> Lens(NumBufs);
  std::vector<int *> Dropped;
  for (int B = 0; B < NumBufs; ++B) {
    Lens[B] = 7 + B * 53;
    Bufs[B] = alloc(Lens[B] * sizeof(int));
    std::vector<int> H(Lens[B]);
    for (int I = 0; I < Lens[B]; ++I)
      H[I] = B * 1000 + I;
    prefix(Memcpy)(Bufs[B], H.data(), H.size() * sizeof(int),
                   prefix(MemcpyHostToDevice));
    Dropped.push_back(alloc(100 + B * 11));
  }
  for (int B = 0; B < NumBufs; B += 3)
    prefix(Free)(Dropped[B]);

  int *Big = alloc(3 << 20);
  std::vector<int> HBig((3 << 20) / sizeof(int), 5);
  for (int B = 0; B < NumBufs; ++B)
    HBig[B] = B;
  prefix(Memcpy)(Big, HBig.data(), 3 << 20, prefix(MemcpyHostToDevice));

  int **Table = reinterpret_cast<int **>(alloc(NumBufs * sizeof(int *)));
  prefix(Memcpy)(Table, Bufs.data(), NumBufs * sizeof(int *),
                 prefix(MemcpyHostToDevice));
  int *DLens = alloc(NumBufs * sizeof(int));
  prefix(Memcpy)(DLens, Lens.data(), NumBufs * sizeof(int),
                 prefix(MemcpyHostToDevice));
  long *Out = reinterpret_cast<long *>(alloc(NumBufs * sizeof(long)));
  prefix(Memset)(Out, 0, NumBufs * sizeof(long));

  prefix(Free)(Head);

  gather<<<NumBufs, 64>>>(Table, DLens, NumBufs, Big, Out);
  prefix(DeviceSynchronize)();

  std::vector<long> HOut(NumBufs);
  prefix(Memcpy)(HOut.data(), Out, NumBufs * sizeof(long),
                 prefix(MemcpyDeviceToHost));
  int Bad = 0;
  for (int B = 0; B < NumBufs; ++B) {
    long N = Lens[B];
    long Expect = B + N * B * 1000 + N * (N - 1) / 2;
    Bad += HOut[B] != Expect;
  }
  printf("%s\n", Bad ? "FAIL" : "PASS");
  return Bad != 0;
}
