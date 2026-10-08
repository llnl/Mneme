#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace mneme {
namespace util {

struct VARange {
  uintptr_t Start;
  uintptr_t End;
  uint64_t size() const { return End - Start; }
};

constexpr uint64_t LargePageSize = 2ULL << 20;

// Large-page alignment when [Addr, Addr + Size) allows it, else PageSize.
inline uint64_t mapAlignment(uintptr_t Addr, uint64_t Size,
                             uint64_t PageSize) {
  if (Size >= LargePageSize && Addr % LargePageSize == 0)
    return LargePageSize;
  return PageSize;
}

// Calls F(Range, Line) for each entry of /proc/self/maps.
template <typename Fn> void forEachMapping(Fn F) {
  std::ifstream Maps("/proc/self/maps");
  std::string Line;
  while (std::getline(Maps, Line)) {
    size_t Dash = Line.find('-');
    if (Dash == std::string::npos)
      continue;
    uintptr_t S = std::stoull(Line.substr(0, Dash), nullptr, 16);
    uintptr_t E = std::stoull(Line.substr(Dash + 1), nullptr, 16);
    F(VARange{S, E}, Line);
  }
}

// /proc/self/maps lines overlapping [Lo, Hi).
inline std::string getMappingsIn(uintptr_t Lo, uintptr_t Hi) {
  std::string Out;
  forEachMapping([&](const VARange &R, const std::string &Line) {
    if (R.Start < Hi && R.End > Lo)
      Out += Line + "\n";
  });
  return Out;
}

// Unmapped user-space ranges of this process, from /proc/self/maps.
inline std::vector<VARange> getFreeVARanges(uintptr_t Lo, uintptr_t Hi) {
  std::vector<VARange> Mapped;
  forEachMapping(
      [&](const VARange &R, const std::string &) { Mapped.push_back(R); });
  std::sort(
      Mapped.begin(), Mapped.end(),
      [](const VARange &A, const VARange &B) { return A.Start < B.Start; });

  std::vector<VARange> Free;
  uintptr_t Cur = Lo;
  for (const auto &M : Mapped) {
    if (M.Start >= Hi)
      break;
    if (M.Start > Cur)
      Free.push_back({Cur, M.Start});
    Cur = std::max(Cur, M.End);
  }
  if (Cur < Hi)
    Free.push_back({Cur, Hi});
  return Free;
}

// Lowest Align-aligned address >= Lo where Size bytes fit unmapped below Hi.
inline std::optional<uintptr_t> firstFreeVAddr(uintptr_t Lo, uintptr_t Hi,
                                               uint64_t Size, uint64_t Align) {
  for (const auto &G : getFreeVARanges(Lo, Hi)) {
    uintptr_t Addr = (G.Start + Align - 1) & ~(Align - 1);
    if (Addr + Size <= G.End)
      return Addr;
  }
  return std::nullopt;
}

// Candidate addresses for a Size-byte reservation, centered in the largest
// free gaps first so neither the heap nor the mmap region grows into it.
inline std::vector<uintptr_t> suggestVAddrs(uint64_t Size, uint64_t Alignment) {
  constexpr uintptr_t Lo = 1ULL << 32;
  constexpr uintptr_t Hi = 1ULL << 47;
  constexpr uint64_t PreferredAlign = 1ULL << 30;

  auto Free = getFreeVARanges(Lo, Hi);
  std::sort(Free.begin(), Free.end(), [](const VARange &A, const VARange &B) {
    return A.size() > B.size();
  });

  std::vector<uintptr_t> Addrs;
  for (const auto &G : Free) {
    if (G.size() < Size + 2 * Alignment)
      continue;
    uintptr_t Mid = G.Start + (G.size() - Size) / 2;
    uint64_t Align =
        G.size() >= Size + 2 * PreferredAlign ? PreferredAlign : Alignment;
    uintptr_t Addr = Mid & ~(Align - 1);
    if (Addr < G.Start)
      Addr += Align;
    if (Addr + Size <= G.End)
      Addrs.push_back(Addr);
  }
  return Addrs;
}

} // namespace util
} // namespace mneme
