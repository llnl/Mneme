#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeComparators.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mneme/MnemeLogger.hpp>
#include <type_traits>

using namespace mneme;

#define FULL_MASK 0xffffffff

#ifdef MNEME_ENABLE_HIP
#include <hip/hip_fp16.h>
using DeviceVendorTraits = DeviceTraits<DeviceVendors::HIP>;
#define MNEME_DEV __device__ __forceinline__
#elif defined(MNEME_ENABLE_CUDA)
#include <cstdint>
#include <cuda_fp16.h> // or hip/hip_fp16.h for HIP
using DeviceVendorTraits = DeviceTraits<DeviceVendors::CUDA>;
#define MNEME_DEV __device__ __forceinline__
#endif

namespace mneme {
namespace dev {

template <class T> MNEME_DEV T mneme_abs(T x) {
  if constexpr (std::is_floating_point_v<T>)
    return x < T(0) ? -x : x;
  else
    return x < T(0) ? -x : x; // for signed ints
}

template <class T> MNEME_DEV T mneme_max(T a, T b) { return a > b ? a : b; }

template <class T> MNEME_DEV T mneme_min(T a, T b) { return a < b ? a : b; }

// Promote integer diffs to double for aggregation.
template <class T> MNEME_DEV double diff_as_double(T a, T b) {
  if constexpr (std::is_same_v<T, __half>) {
    return (double)(__half2float(a) - __half2float(b));
  } else if constexpr (std::is_floating_point_v<T>) {
    return (double)(a - b);
  } else {
    // avoid overflow for small ints by widening first
    long long da = (long long)a;
    long long db = (long long)b;
    return (double)(da - db);
  }
}

MNEME_DEV double dabs(double x) { return x < 0.0 ? -x : x; }

// ---- ULP distance for float/double (monotonic ordering trick) ----

MNEME_DEV uint32_t float_to_ordered_uint(float f) {
  uint32_t u;
  // bitcast
  memcpy(&u, &f, sizeof(u));
  // make lexicographically ordered as signed magnitude
  if (u & 0x80000000u)
    u = 0x80000000u - u;
  return u;
}

MNEME_DEV uint64_t double_to_ordered_uint(double d) {
  uint64_t u;
  memcpy(&u, &d, sizeof(u));
  if (u & 0x8000000000000000ull)
    u = 0x8000000000000000ull - u;
  return u;
}

template <class T> MNEME_DEV uint64_t ulp_distance(T a, T b) {
  if constexpr (std::is_same_v<T, float>) {
    uint32_t ua = float_to_ordered_uint(a);
    uint32_t ub = float_to_ordered_uint(b);
    return (ua > ub) ? (uint64_t)(ua - ub) : (uint64_t)(ub - ua);
  } else if constexpr (std::is_same_v<T, double>) {
    uint64_t ua = double_to_ordered_uint(a);
    uint64_t ub = double_to_ordered_uint(b);
    return (ua > ub) ? (ua - ub) : (ub - ua);
  } else {
    return 0; // ULP not meaningful for ints
  }
}

// ---- Element error metric (returns non-negative "error") ----
//
// For Absolute/Relative: returns |a-b| (or |a-b|/max(|b|,eps) for relative)
// For ULP: returns ULP distance as double
//
template <class T>
MNEME_DEV double element_error(T a, T b, ThresholdKind kind,
                               double rel_eps = 1e-12) {
  if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, __half>) {
    const double da = (double)a;
    const double db = (double)b;
    if (!std::isfinite(da) || !std::isfinite(db)) {
      // allow nans and same-sign infinities
      return (da == db || (std::isnan(da) && std::isnan(db)))
                 ? 0.0
                 : std::numeric_limits<double>::infinity();
    }
  }

  const double abs_err = dabs(diff_as_double(a, b));
  if (kind == ThresholdKind::Absolute)
    return abs_err;

  // Relative: |a-b| / max(|b|, eps)
  const double denom = mneme_max(dabs((double)b), rel_eps);
  return abs_err / denom;
}

// Threshold check (per-element)
MNEME_DEV bool element_pass(double err, double thr) {
  return std::isfinite(err) && err <= thr;
}

MNEME_DEV double atomicMaxDouble(double *addr, double val) {
#if MNEME_ENABLE_CUDA
  unsigned long long *ull = (unsigned long long *)addr;
  unsigned long long old = *ull, assumed;
  do {
    assumed = old;
    double cur;
    memcpy(&cur, &assumed, sizeof(cur));
    if (cur >= val)
      break;
    unsigned long long next;
    memcpy(&next, &val, sizeof(next));
    old = atomicCAS(ull, assumed, next);
  } while (assumed != old);
  double out;
  memcpy(&out, &old, sizeof(out));
  return out;
#elif defined(MNEME_ENABLE_HIP)
  unsigned long long *ull = (unsigned long long *)addr;
  unsigned long long old = *ull, assumed;
  do {
    assumed = old;
    double cur;
    memcpy(&cur, &assumed, sizeof(cur));
    if (cur >= val)
      break;
    unsigned long long next;
    memcpy(&next, &val, sizeof(next));
    old =
        (unsigned long long)atomicCAS((unsigned long long *)ull, assumed, next);
  } while (assumed != old);
  double out;
  memcpy(&out, &old, sizeof(out));
  return out;
#endif
}

MNEME_DEV void atomicAddDouble(double *addr, double val) {
  atomicAdd(addr, val);
}

MNEME_DEV void atomicMinU64(uint64_t *addr, uint64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
  atomicMin(reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(val));
}

template <class T>
__global__ void
compare_builtin_kernel(const T *__restrict__ a, const T *__restrict__ b,
                       std::size_t NumElements, ThresholdKind thr_kind,
                       Norm norm, double threshold, CompareResult *out) {

  CompareResult LOut;
  LOut.AnyFail = 0;
  LOut.FirstBadIdx = NumElements;
  LOut.MaxErr = -1.0;
  LOut.Agg = 0.0;
  const std::size_t tid = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t GridSize = (uint64_t)blockDim.x * (uint64_t)gridDim.x;

  for (std::size_t i = tid; i < NumElements; i += GridSize) {

    const double err = element_error<T>(a[i], b[i], thr_kind);

    // Update max error
    atomicMaxDouble(&out->MaxErr, err);
    if (err > LOut.MaxErr)
      LOut.MaxErr = err;

    if (norm == Norm::None) {
      // Per-element threshold check: record first bad index + AnyFail
      if (!element_pass(err, threshold)) {
        LOut.AnyFail = 1;
        if (LOut.FirstBadIdx > i)
          LOut.FirstBadIdx = i;
      }
    } else if (norm == Norm::Linf) {
      if (LOut.Agg < err)
        LOut.Agg = err;
    } else if (norm == Norm::L1) {
      LOut.Agg += err;
    } else if (norm == Norm::L2) {
      LOut.Agg += err * err;
    }
  }

  // Update max error
  atomicMaxDouble(&out->MaxErr, LOut.MaxErr);

  if (norm == Norm::None) {
    // Per-element threshold check: record first bad index + AnyFail
    if (LOut.AnyFail) {
      out->AnyFail = 1;
      atomicMinU64(&out->FirstBadIdx, LOut.FirstBadIdx);
    }
    return;
  }

  // Aggregate error
  if (norm == Norm::Linf) {
    atomicMaxDouble(&out->Agg, LOut.Agg);
  } else if (norm == Norm::L1) {
    atomicAddDouble(&out->Agg, LOut.Agg);
  } else if (norm == Norm::L2) {
    atomicAddDouble(&out->Agg, LOut.Agg);
  }
}
} // namespace dev
} // namespace mneme

constexpr int NUM_THREADS_PER_BLOCK = 256;

namespace mneme {
CompareResult compareDeviceBlobs(const char *Blob1, const char *Blob2,
                                 uint64_t NumBytes, Metadata Md) {
  CompareResult Init{};
  Init.AnyFail = 0;
  Init.FirstBadIdx = NumBytes;
  Init.MaxErr = 0.0;
  Init.Agg = 0.0;

  CompareResult *Res;
  auto EC =
      DeviceVendorTraits::DeviceErrorCheck(DeviceVendorTraits::DeviceMalloc(
          reinterpret_cast<void **>(&Res), sizeof(CompareResult)));
  if (EC)
    LOG_FATAL("Error in allocating CompareResult " + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(DeviceVendorTraits::DeviceCopy(
      Res, &Init, sizeof(Init), DeviceVendorTraits::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Error in initializing CompareResult " + EC.value());

  auto NumElements = [](BuiltinDType DType, size_t NumBytes) {
    switch (DType) {
    case BuiltinDType::U8:
      return NumBytes / sizeof(uint8_t);
    case BuiltinDType::I8:
      return NumBytes / sizeof(int8_t);
    case BuiltinDType::U16:
      return NumBytes / sizeof(uint16_t);
    case BuiltinDType::I16:
      return NumBytes / sizeof(int16_t);
    case BuiltinDType::U32:
      return NumBytes / sizeof(uint32_t);
    case BuiltinDType::I32:
      return NumBytes / sizeof(int32_t);
    case BuiltinDType::U64:
      return NumBytes / sizeof(uint64_t);
    case BuiltinDType::I64:
      return NumBytes / sizeof(int64_t);
    case BuiltinDType::F16:
      return NumBytes / sizeof(__half);
    case BuiltinDType::F32:
      return NumBytes / sizeof(float);
    case BuiltinDType::F64:
      return NumBytes / sizeof(double);
    default:
      LOG_FATAL("Unsupported BuiltinDType in Mneme comparison dispatch");
    }
    LOG_FATAL("Unsupported BuiltinDType in Mneme comparison dispatch");
    return 0UL;
  }(Md.builtin, NumBytes);

  uint64_t NumBlocks = std::min(
      (NumElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK,
      (uint64_t)8192); // DN -- you could also set this to something like SMs*20
                       // to be more device dependent

  switch (Md.builtin) {
  case BuiltinDType::U8:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const uint8_t *>(Blob1),
        reinterpret_cast<const uint8_t *>(Blob2), NumElements,
        Md.threshold_kind, Md.norm, Md.threshold, Res);
    break;
  case BuiltinDType::I8:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const int8_t *>(Blob1),
        reinterpret_cast<const int8_t *>(Blob2), NumElements, Md.threshold_kind,
        Md.norm, Md.threshold, Res);
    break;

  case BuiltinDType::U16:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const uint16_t *>(Blob1),
        reinterpret_cast<const uint16_t *>(Blob2), NumElements,
        Md.threshold_kind, Md.norm, Md.threshold, Res);
    break;

  case BuiltinDType::I16:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const int16_t *>(Blob1),
        reinterpret_cast<const int16_t *>(Blob2), NumElements,
        Md.threshold_kind, Md.norm, Md.threshold, Res);
    break;

  case BuiltinDType::U32:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const uint32_t *>(Blob1),
        reinterpret_cast<const uint32_t *>(Blob2), NumElements,
        Md.threshold_kind, Md.norm, Md.threshold, Res);
    break;

  case BuiltinDType::I32:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const int32_t *>(Blob1),
        reinterpret_cast<const int32_t *>(Blob2), NumElements,
        Md.threshold_kind, Md.norm, Md.threshold, Res);
    break;

  case BuiltinDType::U64:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const uint64_t *>(Blob1),
        reinterpret_cast<const uint64_t *>(Blob2), NumElements,
        Md.threshold_kind, Md.norm, Md.threshold, Res);
    break;

  case BuiltinDType::I64:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const int64_t *>(Blob1),
        reinterpret_cast<const int64_t *>(Blob2), NumElements,
        Md.threshold_kind, Md.norm, Md.threshold, Res);
    break;

  case BuiltinDType::F16:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const __half *>(Blob1),
        reinterpret_cast<const __half *>(Blob2), NumElements, Md.threshold_kind,
        Md.norm, Md.threshold, Res);
    break;

  case BuiltinDType::F32:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const float *>(Blob1),
        reinterpret_cast<const float *>(Blob2), NumElements, Md.threshold_kind,
        Md.norm, Md.threshold, Res);
    break;

  case BuiltinDType::F64:
    mneme::dev::compare_builtin_kernel<<<NumBlocks, NUM_THREADS_PER_BLOCK>>>(
        reinterpret_cast<const double *>(Blob1),
        reinterpret_cast<const double *>(Blob2), NumElements, Md.threshold_kind,
        Md.norm, Md.threshold, Res);
    break;

  default:
    LOG_FATAL("Unsupported BuiltinDType in Mneme comparison dispatch");
  }

  CompareResult HRes;
  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceSynchronize());

  if (EC)
    LOG_FATAL("Error in comparing blobs " + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(DeviceVendorTraits::DeviceCopy(
      &HRes, Res, sizeof(CompareResult),
      DeviceVendorTraits::MemcpyDeviceToHostKind()));

  if (EC)
    LOG_FATAL("Error in copying back result" + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceSynchronize());

  if (EC)
    LOG_FATAL("Error in comparing blobs " + EC.value());

  if (Md.norm == Norm::L2)
    HRes.Agg = std::sqrt(HRes.Agg);

  EC =
      DeviceVendorTraits::DeviceErrorCheck(DeviceVendorTraits::DeviceFree(Res));

  if (EC)
    LOG_FATAL("Error in comparing blobs " + EC.value());
  return HRes;
}
} // namespace mneme
