#include <cstdlib>
#include <iostream>

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

// The kernel only reads these, so the host-set values exist nowhere in the
// device IR: replay must load them by name and must not fold their zero
// initializers.
__constant__ double coeff[4];
__device__ double offset;

__global__ void scaleOffset(const double *in, double *out, size_t size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size)
    out[tid] = in[tid] * coeff[tid % 4] + offset;
}

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "Wrong CLI: Proper one is " << argv[0] << " '<num elements>'\n";
    return 1;
  }

  size_t numElements = std::atoi(argv[1]);
  const double hostCoeff[4] = {1.0, 2.0, 3.0, 4.0};
  const double hostOffset = 0.5;
  prefix(MemcpyToSymbol)(coeff, hostCoeff, sizeof(hostCoeff));
  prefix(MemcpyToSymbol)(offset, &hostOffset, sizeof(hostOffset));

  double *hostIn = new double[numElements];
  double *hostOut = new double[numElements];
  for (size_t i = 0; i < numElements; i++)
    hostIn[i] = i;

  double *in, *out;
  prefix(Malloc)((void **)&in, numElements * sizeof(double));
  prefix(Malloc)((void **)&out, numElements * sizeof(double));
  prefix(Memcpy)(in, hostIn, numElements * sizeof(double),
                 prefix(MemcpyHostToDevice));

  const int threads = 256;
  int numBlocks = (numElements + threads - 1) / threads;
  scaleOffset<<<numBlocks, threads>>>(in, out, numElements);
  prefix(DeviceSynchronize)();

  prefix(Memcpy)(hostOut, out, numElements * sizeof(double),
                 prefix(MemcpyDeviceToHost));
  int failures = 0;
  for (size_t i = 0; i < numElements; i++)
    if (hostOut[i] != hostIn[i] * hostCoeff[i % 4] + hostOffset)
      failures++;

  delete[] hostIn;
  delete[] hostOut;
  prefix(Free)(in);
  prefix(Free)(out);
  std::cout << "device globals: " << failures << " failures\n";
  return failures != 0;
}
