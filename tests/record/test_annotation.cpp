// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_DATA_DIR="%t.$$.mneme" %build/test_annotation%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

#include <array>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"

using namespace mneme;

static_assert(
    std::is_same_v<decltype(Span(std::declval<float *>(), 4)), Span<float>>);
static_assert(std::is_same_v<decltype(Span(std::declval<const double *>(), 4)),
                             Span<const double>>);
static_assert(
    std::is_same_v<decltype(Span(std::declval<std::vector<double> &>())),
                   Span<double>>);
static_assert(std::is_convertible_v<Span<float>, Span<const float>>);
static_assert(!std::is_convertible_v<Span<const float>, Span<float>>);

constexpr float HostArray[6] = {};
static_assert(std::is_same_v<decltype(Span(HostArray)), Span<const float>>);
static_assert(Span(HostArray).size_bytes() == 6 * sizeof(float));
static_assert(Span(HostArray).subspan(2, 3).data() == HostArray + 2);
static_assert(Span(HostArray).subspan(2, 3).size_bytes() == 3 * sizeof(float));

constexpr std::array<std::int32_t, 5> HostStdArray = {};
static_assert(Span(HostStdArray).size_bytes() == 5 * sizeof(std::int32_t));

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

// A simple kernel that takes a float pointer argument so the preload captures
// at least one annotated blob in the prologue/epilogue snapshots.
__global__ void annotation_kernel(float *data, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    data[idx] = static_cast<float>(idx);
}

int main() {
  const int N = 64;
  const int AliasOffset = 16;
  const int AliasLen = 24;

  float *d_ptr = nullptr;
  MnemeDeviceRT::DeviceMalloc(reinterpret_cast<void **>(&d_ptr),
                              N * sizeof(float));
  float *d_alias = d_ptr + AliasOffset;

  // Annotate the device pointer before the kernel launch so the preload can
  // attach metadata to the recorded snapshot.
  mneme::annotate(d_ptr, mneme::Metadata{
                             .builtin = mneme::BuiltinDType::F32,
                             .threshold = 0.001,
                             .threshold_kind = mneme::ThresholdKind::Relative,
                             .norm = mneme::Norm::Linf,
                             .tag = std::string("test_ptr"),
                         });

  mneme::annotate(mneme::Span(d_alias, AliasLen),
                  mneme::Metadata{
                      .threshold = 0.002,
                      .threshold_kind = mneme::ThresholdKind::Relative,
                      .norm = mneme::Norm::L2,
                      .tag = std::string("test_region"),
                  });

  annotation_kernel<<<1, AliasLen>>>(d_alias, AliasLen);

  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Device error: " << EC.value() << "\n";
    MnemeDeviceRT::DeviceFree(d_ptr);
    return -1;
  }

  std::cout << "OK\n";
  MnemeDeviceRT::DeviceFree(d_ptr);
  return 0;
}

// clang-format off
// CHECK: OK
// CHECK-RR: DemangledName: annotation_kernel(float*, int)
// CHECK-RR: NumModules: 1
// CHECK-RR: NumInstances: 1
// CHECK-RR: {{^}}Annotation: offset=0 extent=64 threshold=0.001 threshold_kind=1 builtin=9 norm=3 tag=test_ptr
// CHECK-RR-NEXT: {{^}}Annotation: offset=64 extent=96 threshold=0.002 threshold_kind=1 builtin=9 norm=2 tag=test_region
// CHECK-RR-NEXT: {{^}}Annotation: offset=160 extent=96 threshold=0.001 threshold_kind=1 builtin=9 norm=3 tag=test_ptr
// clang-format on
