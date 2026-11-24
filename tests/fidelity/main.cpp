#include "gpu.hpp"
#include "profile.hpp"
#include "rotate.hpp"
#include "stencil.hpp"
#include "nbody.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

constexpr int MatrixSize = 2048;
constexpr int StencilSize = 128;
constexpr int Particles = 1024;
constexpr int Elements = 262144;
constexpr int MixSteps = 256;

__global__ void prepare_matrix(float *matrix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < MatrixSize * MatrixSize)
    matrix[i] = float(i);
}

__global__ void prepare_stencil(double *input, double *output, double *sigma) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int volume = StencilSize * StencilSize * StencilSize;
  if (i < volume) {
    input[i] = i % 19;
    // Reset every invocation so a replay that does no work cannot verify.
    output[i] = 0;
    for (int component = 0; component < 9; ++component)
      sigma[i + component * volume] = (i + component * volume) % 19;
  }
}

__global__ void prepare_particles(Particle *particles, float *energy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Particles)
    return;

  Particle p{};
  for (int axis = 0; axis < 3; ++axis) {
    p.pos[axis] = float((i * (17 + axis * 6) + axis * 11) % 1021) / 1021.f;
    p.vel[axis] = float((i + axis) % 7 - 3) * 1.e-4f;
  }
  p.mass = float(1 + i % 31);
  particles[i] = p;
  energy[i] = 0;
}

__host__ __device__ uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  return value ^ (value >> 16);
}

__global__ void fidelity_prepare(uint32_t *input, uint32_t *output,
                                uint32_t seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < Elements) {
    input[i] = seed + uint32_t(i & 255);
    output[i] = 0;
  }
}

__global__ void fidelity_consumer(const uint32_t *input, uint32_t *output,
                                 size_t count, int steps) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    uint32_t value = input[i];
    for (int step = 0; step < steps; ++step)
      value = mix(value);
    output[i] = value | 1u;
  }
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct VerificationError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void verify(bool condition, const char *message) {
  if (!condition)
    throw VerificationError(message);
}

// CPU face-flux reference, independent of the shared-memory tile traversal.
double stencil_reference(int x, int y, int z) {
  constexpr int n = StencilSize, volume = n * n * n;
  auto psi = [](int a, int b, int c) {
    return double((c + n * (b + n * a)) % 19);
  };

  auto flux = [&](int axis, int a, int b, int c) {
    std::array<int, 3> p{a, b, c}, q = p;
    --q[axis];
    double total = 0;
    for (int d = 0; d < 3; ++d) {
      double gradient;
      if (d == axis) {
        gradient = psi(p[0], p[1], p[2]) - psi(q[0], q[1], q[2]);
      } else {
        auto lo = p, hi = p, qlo = q, qhi = q;
        --lo[d];
        ++hi[d];
        --qlo[d];
        ++qhi[d];
        gradient = (psi(hi[0], hi[1], hi[2]) - psi(lo[0], lo[1], lo[2]) +
                    psi(qhi[0], qhi[1], qhi[2]) - psi(qlo[0], qlo[1], qlo[2])) /
                   4.;
      }

      double sigma = (c + n * (b + n * a) + (axis * 3 + d) * volume) % 19;
      total += sigma * gradient;
    }
    return total;
  };

  return flux(0, x + 1, y, z) - flux(0, x, y, z) + flux(1, x, y + 1, z) -
         flux(1, x, y, z) + flux(2, x, y, z + 1) - flux(2, x, y, z);
}

int main(int argc, char **argv) {
  try {
    require(argc == 2 || argc == 4,
            "usage: fidelity CASE [PROFILE_LIBRARY KERNEL_NAME]");

    // Load the profiler before GPU initialization; recording runs omit it.
    Profile profile(argc == 4 ? argv[2] : nullptr);
    check(GPU(SetDevice)(0));
    GPUProperties properties{};
    check(GPU(GetDeviceProperties)(&properties, 0));

    std::string name = argv[1];
    dim3 grid, block;
    const char *target = argc == 4 ? argv[3] : "";

    if (name == "rotate") {
      Buffer<float> matrix(MatrixSize * MatrixSize);
      prepare_matrix<<<matrix.size / 256, 256>>>(matrix.data);
      grid = dim3((MatrixSize / 2 + 255) / 256);
      block = dim3(256);

      // The first four rotations supply context; only the fifth is timed.
      for (int i = 0; i < 5; ++i) {
        if (i == 4)
          profile.begin(target);
        rotate_matrix_parallel<<<grid, block>>>(matrix.data, MatrixSize);
      }

      // Five quarter turns produce the same matrix as one clockwise rotation.
      auto result = matrix.read();
      for (int x = 0; x < MatrixSize; ++x)
        for (int y = 0; y < MatrixSize; ++y)
          verify(result[x * MatrixSize + y] ==
                     float((MatrixSize - 1 - y) * MatrixSize + x),
                 "matrix verification failed");
    } else if (name == "stencil") {
      constexpr int volume = StencilSize * StencilSize * StencilSize;
      Buffer<double> input(volume), output(volume), sigma(9 * volume);
      grid = dim3((StencilSize - 2 + XTILE - 1) / XTILE,
                  (StencilSize - 2) / (BSIZE - 2),
                  (StencilSize - 2) / (BSIZE - 2));
      block = dim3(BSIZE, BSIZE);

      for (int i = 0; i < 5; ++i) {
        // Queue preparation directly before the target without a host sync.
        prepare_stencil<<<volume / 256, 256>>>(input.data, output.data,
                                             sigma.data);
        if (i == 4)
          profile.begin(target);
        stencil3d<<<grid, block>>>(
            input.data, output.data, sigma.data, sigma.data + 3 * volume,
            sigma.data + 6 * volume, StencilSize, StencilSize, StencilSize);
      }

      auto result = output.read();
      for (auto value : result)
        verify(std::isfinite(value), "nonfinite stencil output");

      // Include interior points on tile boundaries in every dimension.
      for (int x : {1, 19, 20, 21, 125, 126})
        for (int y : {1, 14, 15, 125, 126})
          for (int z : {1, 14, 15, 125, 126})
            verify(result[z + StencilSize * (y + StencilSize * x)] ==
                       stencil_reference(x, y, z),
                   "stencil reference mismatch");
    } else if (name == "nbody") {
      Buffer<Particle> particles(Particles);
      Buffer<float> energy(Particles);
      grid = dim3((Particles + 255) / 256);
      block = dim3(256);
      prepare_particles<<<grid, block>>>(particles.data, energy.data);

      for (int i = 0; i < 5; ++i) {
        if (i == 4)
          profile.begin(target);
        accelerate_particles<<<grid, block>>>(particles.data, Particles, 1.e-3f,
                                              6.67259e-11f);
        // Evolve four steps, then time only the fifth acceleration launch.
        if (i != 4) {
          update_particles<<<grid, block>>>(particles.data, energy.data,
                                           Particles, 0.1f);
          accumulate_energy<<<1, 1>>>(energy.data, Particles);
        }
      }

      auto result = particles.read();
      double norm = 0;
      for (const auto &p : result)
        for (float a : p.acc) {
          verify(std::isfinite(a), "nonfinite acceleration");
          norm += std::abs(a);
        }
      verify(norm > 0, "zero acceleration");
    } else if (name == "consumer") {
      Buffer<uint32_t> input(Elements), output(Elements);
      grid = dim3(Elements / 256);
      block = dim3(256);
      uint32_t seed = 0;

      for (int i = 1; i <= 5; ++i) {
        seed = 0x12345678u + uint32_t(i) * 257u;
        fidelity_prepare<<<grid, block>>>(input.data, output.data, seed);
        if (i == 5)
          profile.begin(target);
        fidelity_consumer<<<grid, block>>>(input.data, output.data, Elements,
                                          MixSteps);
      }

      // Input repeats every 256 elements; a small CPU oracle checks all output.
      std::array<uint32_t, 256> expected{};
      for (size_t i = 0; i < expected.size(); ++i) {
        auto value = seed + uint32_t(i);
        for (int j = 0; j < MixSteps; ++j)
          value = mix(value);
        expected[i] = value | 1u;
      }

      auto result = output.read();
      for (size_t i = 0; i < result.size(); ++i)
        verify(result[i] == expected[i & 255], "consumer reference mismatch");
    } else {
      throw std::runtime_error("unknown case");
    }

    check(GPU(GetLastError)());
    check(GPU(DeviceSynchronize)());
    auto times = profile.finish();

    std::cout << "{\"case\":\"" << name
              << "\",\"verified\":true,\"launches\":5,\"sampled_invocation\":5,"
              << "\"grid\":[" << grid.x << ',' << grid.y << ',' << grid.z
              << "],"
              << "\"block\":[" << block.x << ',' << block.y << ',' << block.z
              << "],"
              << "\"gpu\":\"" << properties.name << "\",\"durations_ns\":[";
    for (size_t i = 0; i < times.size(); ++i)
      std::cout << (i ? "," : "") << times[i];
    std::cout << "]}\n";
    return 0;
  } catch (const VerificationError &error) {
    std::cerr << "fidelity verification: " << error.what() << '\n';
    return 3;
  } catch (const std::exception &error) {
    std::cerr << "fidelity: " << error.what() << '\n';
    return 1;
  }
}
