#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"
#include "mneme/MnemeVASpace.hpp"

// Picks device addresses for recorded allocations. Regions are nominal, never
// reserved with the driver; each allocation maps its own range.
template <mneme::DeviceVendors VendorTypes> class PageManager {
  using DT = mneme::DeviceTraits<VendorTypes>;
  static constexpr uint64_t MinRegionSize = 1ULL << 40;

  uint64_t PageSize;
  // Start -> size.
  std::map<uintptr_t, uint64_t> Regions;
  std::map<uintptr_t, uint64_t> FreeRanges;

  bool overlapsRegion(uintptr_t Start, uint64_t Size) const {
    auto It = Regions.upper_bound(Start);
    if (It != Regions.end() && It->first < Start + Size)
      return true;
    return It != Regions.begin() &&
           std::prev(It)->first + std::prev(It)->second > Start;
  }

  void grow(uint64_t Size) {
    uint64_t RegionSize = std::max(
        MinRegionSize, mneme::util::roundUp(Size, mneme::util::LargePageSize));
    for (uintptr_t Addr :
         mneme::util::suggestVAddrs(RegionSize, mneme::util::LargePageSize)) {
      if (overlapsRegion(Addr, RegionSize))
        continue;
      LOG_INFO("New device address region {} size {}",
               reinterpret_cast<void *>(Addr), RegionSize);
      Regions.emplace(Addr, RegionSize);
      releaseAddr(reinterpret_cast<void *>(Addr), RegionSize);
      return;
    }
    LOG_FATAL("No device address region for {} bytes", Size);
  }

  // Best fit: the smallest free range that holds Size bytes at Align.
  std::map<uintptr_t, uint64_t>::iterator findFit(uint64_t Size,
                                                  uint64_t Align,
                                                  uintptr_t &Addr) {
    auto Best = FreeRanges.end();
    for (auto It = FreeRanges.begin(); It != FreeRanges.end(); ++It) {
      uintptr_t A = mneme::util::roundUp(It->first, Align);
      if (A + Size > It->first + It->second)
        continue;
      if (Best == FreeRanges.end() || It->second < Best->second) {
        Best = It;
        Addr = A;
      }
    }
    return Best;
  }

public:
  struct AddrRange {
    void *Addr;
    uint64_t Size;
    uint64_t Align;
  };

  explicit PageManager(uint64_t PageSize) : PageSize(PageSize) {}

  // Large sizes get large-page alignment so they can use big GPU fragments.
  AddrRange allocateAddr(uint64_t Size) {
    uint64_t Align = Size >= mneme::util::LargePageSize
                         ? mneme::util::LargePageSize
                         : PageSize;
    uint64_t ActualSize = std::max(mneme::util::roundUp(Size, Align), PageSize);

    uintptr_t Addr = 0;
    auto Fit = findFit(ActualSize, Align, Addr);
    if (Fit == FreeRanges.end()) {
      grow(ActualSize);
      Fit = findFit(ActualSize, Align, Addr);
    }

    uintptr_t Start = Fit->first;
    uintptr_t End = Start + Fit->second;
    FreeRanges.erase(Fit);
    if (Addr > Start)
      FreeRanges.emplace(Start, Addr - Start);
    if (Addr + ActualSize < End)
      FreeRanges.emplace(Addr + ActualSize, End - Addr - ActualSize);
    return {reinterpret_cast<void *>(Addr), ActualSize, Align};
  }

  void releaseAddr(void *Addr, uint64_t Size) {
    uintptr_t Start = reinterpret_cast<uintptr_t>(Addr);
    uintptr_t End = Start + Size;
    auto Next = FreeRanges.lower_bound(Start);
    if (Next != FreeRanges.end() && Next->first == End) {
      End += Next->second;
      Next = FreeRanges.erase(Next);
    }
    if (Next != FreeRanges.begin()) {
      auto Prev = std::prev(Next);
      if (Prev->first + Prev->second == Start) {
        Prev->second = End - Prev->first;
        return;
      }
    }
    FreeRanges.emplace_hint(Next, Start, End - Start);
  }
};
