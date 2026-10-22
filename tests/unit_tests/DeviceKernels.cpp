#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeMemory.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

using namespace mneme;

#ifdef MNEME_ENABLE_HIP
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#include <hip/hip_fp16.h>
#elif defined(MNEME_ENABLE_CUDA)
constexpr DeviceVendors Vendor = DeviceVendors::CUDA;
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#include <cuda_fp16.h>
#else
#error "Define MNEME_ENABLE_HIP or MNEME_ENABLE_CUDA"
#endif

static bool approx(double a, double b, double eps = 1e-9) {
  return std::fabs(a - b) <= eps * (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

// Compute expected error per-element for Absolute/Relative (your device code
// does NOT implement ULP)
template <class T> static double host_element_error(T a, T b, ThresholdKind k) {
  const double da = (double)a;
  const double db = (double)b;
  const double abs_err = std::fabs(da - db);
  if (k == ThresholdKind::Absolute)
    return abs_err;
  // Relative: |a-b| / max(|b|, 1e-12)
  const double denom = std::max(std::fabs(db), 1e-12);
  return abs_err / denom;
}

template <>
double host_element_error<__half>(__half a, __half b, ThresholdKind k) {
#if defined(MNEME_ENABLE_HIP)
  float fa = __half2float(a);
  float fb = __half2float(b);
#else
  float fa = __half2float(a);
  float fb = __half2float(b);
#endif
  return host_element_error<float>(fa, fb, k);
}

template <class T>
static void fill_equal(std::vector<T> &x, std::vector<T> &y) {
  for (size_t i = 0; i < x.size(); ++i) {
    // pick a pattern that is safe for ints and floats
    if constexpr (std::is_integral_v<T>) {
      x[i] = (T)((i * 7) % 251);
      y[i] = x[i];
    } else {
      x[i] = (T)(0.25 + (double)i * 0.01);
      y[i] = x[i];
    }
  }
}

template <>
void fill_equal<__half>(std::vector<__half> &x, std::vector<__half> &y) {
  for (size_t i = 0; i < x.size(); ++i) {
    float v = 0.25f + (float)i * 0.01f;
#if defined(MNEME_ENABLE_HIP)
    x[i] = __float2half(v);
    y[i] = __float2half(v);
#else
    x[i] = __float2half(v);
    y[i] = __float2half(v);
#endif
  }
}

// Introduce one mismatch at idx with delta
template <class T>
static void introduce_mismatch(std::vector<T> &y, size_t idx, double delta) {
  if constexpr (std::is_integral_v<T>) {
    long long v = (long long)y[idx];
    y[idx] = (T)(v + (long long)delta);
  } else {
    y[idx] = (T)((double)y[idx] + delta);
  }
}

template <>
void introduce_mismatch<__half>(std::vector<__half> &y, size_t idx,
                                double delta) {
#if defined(MNEME_ENABLE_HIP)
  float v = __half2float(y[idx]);
  y[idx] = __float2half(v + (float)delta);
#else
  float v = __half2float(y[idx]);
  y[idx] = __float2half(v + (float)delta);
#endif
}

template <class T>
static void run_case(BuiltinDType dtype, ThresholdKind tk, Norm norm,
                     double threshold, bool expect_fail) {
  constexpr size_t N = 2048;
  std::vector<T> hA(N), hB(N);
  fill_equal(hA, hB);

  const size_t bad_idx = 777;
  const double delta = std::is_integral_v<T> ? 3.0 : 0.05; // sizeable diff
  if (expect_fail)
    introduce_mismatch<T>(hB, bad_idx, delta);

  // Allocate device buffers
  T *dA = nullptr;
  T *dB = nullptr;
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceMalloc(
      reinterpret_cast<void **>(&dA), N * sizeof(T)));
  if (EC)
    LOG_FATAL("DeviceMalloc dA failed: " + EC.value());

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceMalloc(
      reinterpret_cast<void **>(&dB), N * sizeof(T)));
  if (EC)
    LOG_FATAL("DeviceMalloc dB failed: " + EC.value());

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
      dA, hA.data(), N * sizeof(T), MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Memcpy H2D dA failed: " + EC.value());

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
      dB, hB.data(), N * sizeof(T), MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Memcpy H2D dB failed: " + EC.value());

  Metadata md;
  md.builtin = dtype;
  md.threshold_kind = tk;
  md.norm = norm;
  md.threshold = threshold;

  // Call Mneme comparator
  CompareResult r = compareDeviceBlobs(reinterpret_cast<const char *>(dA),
                                       reinterpret_cast<const char *>(dB),
                                       (uint64_t)(N * sizeof(T)), md);

  // Compute expected aggregates (matching your kernel semantics)
  double exp_max = 0.0;
  double exp_l1 = 0.0;
  double exp_l2_sum_sq = 0.0;
  double exp_linf = 0.0;

  int exp_any_fail = 0;
  uint64_t exp_first_bad = N;

  for (size_t i = 0; i < N; ++i) {
    double e = host_element_error<T>(hA[i], hB[i], tk);
    exp_max = std::max(exp_max, e);

    if (norm == Norm::None) {
      if (e > threshold) {
        exp_any_fail = 1;
        exp_first_bad = std::min<uint64_t>(exp_first_bad, i);
      }
    } else {
      exp_l1 += e;
      exp_l2_sum_sq += std::sqrt(e * e);
      exp_linf = std::max(exp_linf, e);
    }
  }

  // Assertions
  if (norm == Norm::None) {
    if (expect_fail) {
      assert(r.AnyFail == 1);
      assert(r.FirstBadIdx == exp_first_bad);
      assert(r.MaxErr >= exp_max - 1e-12);
    } else {
      assert(r.AnyFail == 0);
      assert(r.FirstBadIdx >= N);
      assert(approx(r.MaxErr, 0.0) || r.MaxErr <= 1e-12);
    }
  } else if (norm == Norm::Linf) {
    assert(r.AnyFail == 0); // your kernel only sets AnyFail for Norm::None
    assert(approx(r.Agg, exp_linf, 1e-7));
    assert(r.MaxErr >= exp_max - 1e-12);
  } else if (norm == Norm::L1) {
    assert(r.AnyFail == 0);
    assert(approx(r.Agg, exp_l1, 1e-6));
    assert(r.MaxErr >= exp_max - 1e-12);
  } else if (norm == Norm::L2) {
    assert(r.AnyFail == 0);
    assert(approx(r.Agg, exp_l2_sum_sq, 1e-6));
    assert(r.MaxErr >= exp_max - 1e-12);
  } else {
    assert(false && "Unhandled norm");
  }

  // Free
  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(dA));
  if (EC)
    LOG_FATAL("DeviceFree dA failed: " + EC.value());
  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(dB));
  if (EC)
    LOG_FATAL("DeviceFree dB failed: " + EC.value());
}

static void run_all_for_dtype(BuiltinDType dt) {
  // Absolute, exact equality should pass with threshold 0
  auto run_abs_pass = [&](auto tag_type) {
    using T = decltype(tag_type);
    run_case<T>(dt, ThresholdKind::Absolute, Norm::None, 0.0,
                /*expect_fail=*/false);
    run_case<T>(dt, ThresholdKind::Absolute, Norm::Linf, 0.0,
                /*expect_fail=*/false);
    run_case<T>(dt, ThresholdKind::Absolute, Norm::L1, 0.0,
                /*expect_fail=*/false);
    run_case<T>(dt, ThresholdKind::Absolute, Norm::L2, 0.0,
                /*expect_fail=*/false);
  };

  auto run_abs_fail = [&](auto tag_type) {
    using T = decltype(tag_type);
    // With a mismatch, Norm::None should fail for small threshold
    run_case<T>(dt, ThresholdKind::Absolute, Norm::None, 0.0,
                /*expect_fail=*/true);

    // Aggregates should reflect mismatch; we don't "fail" in kernel for
    // aggregates.
    run_case<T>(dt, ThresholdKind::Absolute, Norm::Linf, 0.0,
                /*expect_fail=*/true);
    run_case<T>(dt, ThresholdKind::Absolute, Norm::L1, 0.0,
                /*expect_fail=*/true);
    run_case<T>(dt, ThresholdKind::Absolute, Norm::L2, 0.0,
                /*expect_fail=*/true);
  };

  auto run_rel_pass = [&](auto tag_type) {
    using T = decltype(tag_type);
    // Equal buffers => relative errors 0
    run_case<T>(dt, ThresholdKind::Relative, Norm::None, 0.0,
                /*expect_fail=*/false);
    run_case<T>(dt, ThresholdKind::Relative, Norm::Linf, 0.0,
                /*expect_fail=*/false);
    run_case<T>(dt, ThresholdKind::Relative, Norm::L1, 0.0,
                /*expect_fail=*/false);
    run_case<T>(dt, ThresholdKind::Relative, Norm::L2, 0.0,
                /*expect_fail=*/false);
  };

  auto run_rel_fail = [&](auto tag_type) {
    using T = decltype(tag_type);
    // Make mismatch; relative error should be nonzero => fail at threshold 0 in
    // Norm::None
    run_case<T>(dt, ThresholdKind::Relative, Norm::None, 0.0,
                /*expect_fail=*/true);

    // Aggregates
    run_case<T>(dt, ThresholdKind::Relative, Norm::Linf, 0.0,
                /*expect_fail=*/true);
    run_case<T>(dt, ThresholdKind::Relative, Norm::L1, 0.0,
                /*expect_fail=*/true);
    run_case<T>(dt, ThresholdKind::Relative, Norm::L2, 0.0,
                /*expect_fail=*/true);
  };

  switch (dt) {
  case BuiltinDType::U8:
    run_abs_pass(uint8_t{});
    run_abs_fail(uint8_t{});
    run_rel_pass(uint8_t{});
    run_rel_fail(uint8_t{});
    break;
  case BuiltinDType::I8:
    run_abs_pass(int8_t{});
    run_abs_fail(int8_t{});
    run_rel_pass(int8_t{});
    run_rel_fail(int8_t{});
    break;
  case BuiltinDType::U16:
    run_abs_pass(uint16_t{});
    run_abs_fail(uint16_t{});
    run_rel_pass(uint16_t{});
    run_rel_fail(uint16_t{});
    break;
  case BuiltinDType::I16:
    run_abs_pass(int16_t{});
    run_abs_fail(int16_t{});
    run_rel_pass(int16_t{});
    run_rel_fail(int16_t{});
    break;
  case BuiltinDType::U32:
    run_abs_pass(uint32_t{});
    run_abs_fail(uint32_t{});
    run_rel_pass(uint32_t{});
    run_rel_fail(uint32_t{});
    break;
  case BuiltinDType::I32:
    run_abs_pass(int32_t{});
    run_abs_fail(int32_t{});
    run_rel_pass(int32_t{});
    run_rel_fail(int32_t{});
    break;
  case BuiltinDType::U64:
    run_abs_pass(uint64_t{});
    run_abs_fail(uint64_t{});
    run_rel_pass(uint64_t{});
    run_rel_fail(uint64_t{});
    break;
  case BuiltinDType::I64:
    run_abs_pass(int64_t{});
    run_abs_fail(int64_t{});
    run_rel_pass(int64_t{});
    run_rel_fail(int64_t{});
    break;
  case BuiltinDType::F16:
    run_abs_pass(__half{});
    run_abs_fail(__half{});
    run_rel_pass(__half{});
    run_rel_fail(__half{});
    break;
  case BuiltinDType::F32:
    run_abs_pass(float{});
    run_abs_fail(float{});
    run_rel_pass(float{});
    run_rel_fail(float{});
    break;
  case BuiltinDType::F64:
    run_abs_pass(double{});
    run_abs_fail(double{});
    run_rel_pass(double{});
    run_rel_fail(double{});
    break;
  default:
    assert(false && "unknown dtype");
  }
}

template <class T, class Bits> static T value_from_bits(Bits bits) {
  // helper to construct a float/double/__half from its bit pattern
  static_assert(sizeof(T) == sizeof(Bits));
  T value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// check the inf/nan pairs are treated properly in correctness checking
template <class T>
static void run_pair_case(BuiltinDType dtype, T a, T b, ThresholdKind tk,
                          Norm norm, double threshold, double expected_error,
                          bool expect_pass, size_t N = 2048,
                          size_t pair_idx = 777) {
  std::vector<T> hA(N, (T)1.0), hB(N, (T)1.0);
  hA[pair_idx] = a;
  hB[pair_idx] = b;
  const size_t NumBytes = N * sizeof(T);
  auto check = [](auto status) {
    auto EC = MnemeDeviceRT::DeviceErrorCheck(status);
    if (EC)
      LOG_FATAL("Pair comparison test failed: " + EC.value());
  };

  MnemeMemoryBlob<Vendor> A(NumBytes, nullptr, NumBytes);
  MnemeMemoryBlob<Vendor> B(NumBytes, nullptr, NumBytes);
  check(A.allocate(NumBytes));
  check(B.allocate(NumBytes));
  check(MnemeDeviceRT::DeviceCopy(A.getBlobAddr(), hA.data(), NumBytes,
                                  MnemeDeviceRT::MemcpyHostToDeviceKind()));
  check(MnemeDeviceRT::DeviceCopy(B.getBlobAddr(), hB.data(), NumBytes,
                                  MnemeDeviceRT::MemcpyHostToDeviceKind()));

  Metadata md;
  md.builtin = dtype;
  md.threshold_kind = tk;
  md.norm = norm;
  md.threshold = threshold;
  A.setMetadata(md);
  B.setMetadata(md);

  CompareResult r = compareDeviceBlobs(
      reinterpret_cast<const char *>(A.getBlobAddr()),
      reinterpret_cast<const char *>(B.getBlobAddr()), NumBytes, md);
  if (std::isfinite(expected_error))
    assert(std::isfinite(r.MaxErr) && approx(r.MaxErr, expected_error));
  else
    assert(r.MaxErr == expected_error);

  assert(r.AnyFail == (norm == Norm::None && !expect_pass));
  if (norm == Norm::None) {
    if (expect_pass)
      assert(r.FirstBadIdx >= N);
    else
      assert(r.FirstBadIdx == pair_idx);
  } else if (std::isfinite(expected_error)) {
    assert(std::isfinite(r.Agg) && approx(r.Agg, expected_error));
  } else {
    assert(r.Agg == expected_error);
  }
  assert((A == B) == expect_pass);
  assert((B == A) == expect_pass);

  check(A.release());
  check(B.release());
}

template <class T>
static void run_nonfinite_cases(BuiltinDType dtype, T nan_a, T nan_b) {
  const double inf = std::numeric_limits<double>::infinity();
  struct PairCase {
    T a, b;
    bool equal;
  };

  // nan_a and nan_b have different signs and payloads in each tested format.
  const PairCase Cases[] = {
      {nan_a, nan_a, true},    {nan_a, nan_b, true},
      {(T)inf, (T)inf, true},  {(T)-inf, (T)-inf, true},
      {nan_a, (T)1.0, false},  {nan_a, (T)inf, false},
      {nan_a, (T)-inf, false}, {(T)inf, (T)-inf, false},
      {(T)inf, (T)1.0, false}, {(T)-inf, (T)1.0, false},
  };

  for (auto tk : {ThresholdKind::Absolute, ThresholdKind::Relative}) {
    for (auto norm : {Norm::None, Norm::L1, Norm::L2, Norm::Linf}) {
      for (const auto &c : Cases) {
        for (double threshold : {0.0, inf}) {
          run_pair_case<T>(dtype, c.a, c.b, tk, norm, threshold,
                           c.equal ? 0.0 : inf, c.equal);
        }
      }


      const double error = tk == ThresholdKind::Absolute ? 0.25 : 0.2;
      run_pair_case<T>(dtype, (T)1.0, (T)1.25, tk, norm, 0.1, error,
                       false);
      run_pair_case<T>(dtype, (T)1.0, (T)1.25, tk, norm, 0.5, error,
                       true);
    }
  }
}

// Byte comparison of buffers larger than 2 GiB must still see a difference
// past byte 2^31.
static void run_large_u8_case() {
  constexpr uint64_t NumBytes = (1ULL << 31) + 4096;
  constexpr uint64_t BadIdx = (1ULL << 31) + 7;
  char *dA = nullptr;
  char *dB = nullptr;
  if (MnemeDeviceRT::DeviceMalloc(reinterpret_cast<void **>(&dA), NumBytes) !=
          MnemeDeviceRT::DeviceSuccess ||
      MnemeDeviceRT::DeviceMalloc(reinterpret_cast<void **>(&dB), NumBytes) !=
          MnemeDeviceRT::DeviceSuccess) {
    std::cerr << "Large comparison test needs 4 GiB of device memory\n";
    std::exit(1);
  }
  MnemeDeviceRT::DeviceMemset(dA, 0, NumBytes);
  MnemeDeviceRT::DeviceMemset(dB, 0, NumBytes);
  MnemeDeviceRT::DeviceMemset(dB + BadIdx, 1, 1);

  Metadata md;
  md.builtin = BuiltinDType::U8;
  CompareResult r = compareDeviceBlobs(dA, dB, NumBytes, md);
  if (r.AnyFail != 1 || r.FirstBadIdx != BadIdx) {
    std::cerr << "Large comparison missed byte " << BadIdx << ": AnyFail "
              << r.AnyFail << ", FirstBadIdx " << r.FirstBadIdx << "\n";
    std::exit(1);
  }
  MnemeDeviceRT::DeviceFree(dA);
  MnemeDeviceRT::DeviceFree(dB);
}

int main() {
  std::cout << "Running Mneme compareDeviceBlobs thorough tests...\n";

  run_large_u8_case();

  run_all_for_dtype(BuiltinDType::U8);
  run_all_for_dtype(BuiltinDType::I8);
  run_all_for_dtype(BuiltinDType::U16);
  run_all_for_dtype(BuiltinDType::I16);
  run_all_for_dtype(BuiltinDType::U32);
  run_all_for_dtype(BuiltinDType::I32);
  run_all_for_dtype(BuiltinDType::U64);
  run_all_for_dtype(BuiltinDType::I64);
  run_all_for_dtype(BuiltinDType::F16);
  run_all_for_dtype(BuiltinDType::F32);
  run_all_for_dtype(BuiltinDType::F64);

  run_nonfinite_cases<__half>(BuiltinDType::F16,
                              value_from_bits<__half>(uint16_t{0x7e01}),
                              value_from_bits<__half>(uint16_t{0xfe02}));
  run_nonfinite_cases<float>(BuiltinDType::F32,
                             value_from_bits<float>(uint32_t{0x7fc00001}),
                             value_from_bits<float>(uint32_t{0xffc00002}));
  run_nonfinite_cases<double>(
      BuiltinDType::F64, value_from_bits<double>(uint64_t{0x7ff8000000000001}),
      value_from_bits<double>(uint64_t{0xfff8000000000002}));

  // go beyond 2^31 threads to ensure the reduction sees the NaN and the matching pair.
  run_pair_case<float>(BuiltinDType::F32,
                       std::numeric_limits<float>::quiet_NaN(), 1.0f,
                       ThresholdKind::Absolute, Norm::Linf, 0.0,
                       std::numeric_limits<double>::infinity(),
                       false, 8192 * 256 + 1, 0);

  std::cout << "All tests passed.\n";
  return 0;
}
