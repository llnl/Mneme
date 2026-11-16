#pragma once
//===----------------------------------------------------------------------===//
// mneme_annotate.h - User-facing annotation API for Mneme
//
// This header declares mneme::annotate(ptr, Metadata{...}) and sub-region
// annotation overloads such as mneme::annotate(ptr, nbytes, Metadata{...}).
// It is intentionally *interface-only*: no lambdas / comparators yet.
//
// Example:
//   double* p = ...;
//   mneme::annotate(p, mneme::Metadata{
//     .dtype = mneme::Type::Builtin,
//     .builtin = mneme::BuiltinDType::F64,
//     .threshold = 0.1,
//     .threshold_kind = mneme::ThresholdKind::Absolute,
//     .norm = mneme::Norm::Linf,
//   });
//
// You can also use the typed helper:
//   mneme::annotate<double>(p, mneme::Metadata{ .threshold = 0.1 });
//
// Or annotate a sub-region explicitly:
//   mneme::annotate(p, 128 * sizeof(double),
//                   mneme::Metadata{ .threshold = 0.1 });
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

namespace mneme {

// Built-in scalar dtypes Mneme knows about today.
enum class BuiltinDType : std::uint8_t {
  U8 = 0,
  I8,
  U16,
  I16,
  U32,
  I32,
  U64,
  I64,
  F16,
  F32,
  F64,
};

// The size in bytes of one element of builtin, or 0 if builtin is not one of
// the enumerators, as when it was read from a corrupt file.
constexpr std::size_t builtinByteWidth(BuiltinDType builtin) {
  switch (builtin) {
  case BuiltinDType::U8:
  case BuiltinDType::I8:
    return 1;
  case BuiltinDType::U16:
  case BuiltinDType::I16:
  case BuiltinDType::F16:
    return 2;
  case BuiltinDType::U32:
  case BuiltinDType::I32:
  case BuiltinDType::F32:
    return 4;
  case BuiltinDType::U64:
  case BuiltinDType::I64:
  case BuiltinDType::F64:
    return 8;
  }
  return 0;
}

// How to interpret `threshold`.
enum class ThresholdKind : std::uint8_t {
  Absolute = 0,
  Relative = 1,
};

// If the pointer represents a collection, which norm to use for aggregation.
enum class Norm : std::uint8_t {
  None = 0, // interpret threshold per-element / per-scalar
  L1,
  L2,
  Linf,
};

// Core metadata attached to a pointer/region.
struct Metadata {
  // Builtin dtype (used when dtype == Builtin).
  BuiltinDType builtin = BuiltinDType::U8;

  // Error threshold semantics.
  double threshold = 0.0;
  ThresholdKind threshold_kind = ThresholdKind::Absolute;

  // Aggregation semantics (for arrays/regions). Optional.
  Norm norm = Norm::None;

  // Optional user tag (for reporting / grouping). Not interpreted by Mneme.
  std::optional<std::string> tag = std::nullopt;
};

inline bool operator==(const Metadata &LHS, const Metadata &RHS) {
  return LHS.builtin == RHS.builtin && LHS.threshold == RHS.threshold &&
         LHS.threshold_kind == RHS.threshold_kind && LHS.norm == RHS.norm &&
         LHS.tag == RHS.tag;
}

inline bool operator!=(const Metadata &LHS, const Metadata &RHS) {
  return !(LHS == RHS);
}

// --------- Builtin dtype mapping helpers (optional sugar) ------------------

template <class T> struct builtin_dtype_of {
  static constexpr BuiltinDType value = BuiltinDType::U8;
};

template <> struct builtin_dtype_of<float> {
  static constexpr BuiltinDType value = BuiltinDType::F32;
};
template <> struct builtin_dtype_of<double> {
  static constexpr BuiltinDType value = BuiltinDType::F64;
};

template <> struct builtin_dtype_of<std::int8_t> {
  static constexpr BuiltinDType value = BuiltinDType::I8;
};
template <> struct builtin_dtype_of<std::uint8_t> {
  static constexpr BuiltinDType value = BuiltinDType::U8;
};
template <> struct builtin_dtype_of<std::int16_t> {
  static constexpr BuiltinDType value = BuiltinDType::I16;
};
template <> struct builtin_dtype_of<std::uint16_t> {
  static constexpr BuiltinDType value = BuiltinDType::U16;
};
template <> struct builtin_dtype_of<std::int32_t> {
  static constexpr BuiltinDType value = BuiltinDType::I32;
};
template <> struct builtin_dtype_of<std::uint32_t> {
  static constexpr BuiltinDType value = BuiltinDType::U32;
};
template <> struct builtin_dtype_of<std::int64_t> {
  static constexpr BuiltinDType value = BuiltinDType::I64;
};
template <> struct builtin_dtype_of<std::uint64_t> {
  static constexpr BuiltinDType value = BuiltinDType::U64;
};

// --------- User-facing API ------------------------------------------------

// Primary user API: annotate a pointer with metadata.
void annotate(const void *ptr, Metadata md);

// Annotate a sub-region beginning at ptr and spanning bytes bytes.
void annotate(const void *ptr, std::size_t bytes, Metadata md);

// Typed helper: sets builtin dtype automatically when T maps to a known
// BuiltinDType. If T is unknown, this will leave builtin=Unknown (still useful
// if you set dtype=Custom later).
template <class T,
          std::enable_if_t<!std::is_void_v<std::remove_cv_t<T>>, int> = 0>
inline void annotate(T *ptr, Metadata md = {}) {
  // If the user didn't specify dtype explicitly, keep default Builtin.
  // If they *did* specify Custom, we don't override anything here.
  md.builtin = builtin_dtype_of<std::remove_cv_t<T>>::value;
  annotate(static_cast<const void *>(ptr), std::move(md));
}

} // namespace mneme
