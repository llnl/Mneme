#pragma once
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#if MNEME_ENABLE_DEBUG
#define MNEME_DBG(x) x;
#else
#define MNEME_DBG(x)
#endif

namespace mneme {
namespace util {
template <typename Ty> Ty roundUp(Ty Size, Ty Divider) {
  return (Size + Divider - 1) & ~(Divider - 1);
}
template <typename Ty> Ty extractScalar(const char *&Buffer) {
  Ty Value = *reinterpret_cast<const Ty *>(Buffer);
  Buffer += sizeof(Ty);
  return Value;
}
template <typename Ty> Ty extractScalar(char *&Buffer) {
  Ty Value = *reinterpret_cast<const Ty *>(Buffer);
  Buffer += sizeof(Ty);
  return Value;
}

inline std::string readSizedString(const char *&Buffer) {
  size_t StrLen = extractScalar<size_t>(Buffer);
  std::string Name{Buffer, StrLen};
  Buffer += StrLen;
  return Name;
}

} // namespace util
} // namespace mneme
