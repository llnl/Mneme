#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"
#include "mneme/MnemeVASpace.hpp"

// Free address ranges with best-fit allocation and coalescing release.
class FreeRanges {
  // Start -> size.
  std::map<uintptr_t, uint64_t> Ranges;

public:
  std::optional<uintptr_t> allocate(uint64_t Size, uint64_t Align) {
    auto Best = Ranges.end();
    uintptr_t Addr = 0;
    for (auto It = Ranges.begin(); It != Ranges.end(); ++It) {
      uintptr_t A = mneme::util::roundUp(It->first, Align);
      if (A + Size > It->first + It->second)
        continue;
      if (Best == Ranges.end() || It->second < Best->second) {
        Best = It;
        Addr = A;
      }
    }
    if (Best == Ranges.end())
      return std::nullopt;

    uintptr_t Start = Best->first;
    uintptr_t End = Start + Best->second;
    Ranges.erase(Best);
    if (Addr > Start)
      Ranges.emplace(Start, Addr - Start);
    if (Addr + Size < End)
      Ranges.emplace(Addr + Size, End - Addr - Size);
    return Addr;
  }

  void release(uintptr_t Start, uint64_t Size) {
    uintptr_t End = Start + Size;
    auto Next = Ranges.lower_bound(Start);
    if (Next != Ranges.end() && Next->first == End) {
      End += Next->second;
      Next = Ranges.erase(Next);
    }
    if (Next != Ranges.begin()) {
      auto Prev = std::prev(Next);
      if (Prev->first + Prev->second == Start) {
        Prev->second = End - Prev->first;
        return;
      }
    }
    Ranges.emplace_hint(Next, Start, End - Start);
  }
};

// Picks device addresses for recorded allocations. Regions are nominal, never
// reserved with the driver; each allocation maps its own range.
template <mneme::DeviceVendors VendorTypes> class PageManager {
  using DT = mneme::DeviceTraits<VendorTypes>;
  static constexpr uint64_t MinRegionSize = 1ULL << 40;
  static constexpr int MaxMapTries = 16;

  uint64_t PageSize;
  // Start -> size.
  std::map<uintptr_t, uint64_t> Regions;
  FreeRanges Free;

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
      Free.release(Addr, RegionSize);
      return;
    }
    LOG_FATAL("No device address region for {} bytes", Size);
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

    auto Addr = Free.allocate(ActualSize, Align);
    if (!Addr) {
      grow(ActualSize);
      Addr = Free.allocate(ActualSize, Align);
    }
    return {reinterpret_cast<void *>(*Addr), ActualSize, Align};
  }

  void releaseAddr(void *Addr, uint64_t Size) {
    Free.release(reinterpret_cast<uintptr_t>(Addr), Size);
  }

  // Allocates a range and maps it with Map(Range), skipping occupied
  // addresses. Nullopt when the device is out of memory.
  template <typename MapFn>
  std::optional<AddrRange> mapAddr(uint64_t Size, MapFn Map) {
    for (int Try = 0; Try < MaxMapTries; ++Try) {
      auto R = allocateAddr(Size);
      switch (Map(R)) {
      case mneme::MapStatus::Mapped:
        return R;
      case mneme::MapStatus::Occupied:
        // Something else lives there; keep the range out of the free set.
        LOG_DEBUG("Device address {} is occupied, retrying", R.Addr);
        break;
      case mneme::MapStatus::OutOfMemory:
        releaseAddr(R.Addr, R.Size);
        return std::nullopt;
      }
    }
    LOG_FATAL("Cannot map {} bytes of device memory", Size);
  }
};

// Packs allocations smaller than a large page into shared large-page chunks,
// since mapping each one separately is slow.
template <mneme::DeviceVendors VendorTypes> class SmallAllocator {
  using DT = mneme::DeviceTraits<VendorTypes>;
  using Handle_t = typename DT::MemoryAllocationHandle_t;
  static constexpr uint64_t ChunkSize = mneme::util::LargePageSize;

  struct Chunk {
    Handle_t Handle{};
    FreeRanges Free;
    uint64_t Used = 0;
  };

  PageManager<VendorTypes> &PM;
  int DeviceID;
  // Start -> chunk.
  std::map<uintptr_t, Chunk> Chunks;
  // One empty chunk stays mapped so malloc/free loops don't remap.
  bool HasSpare = false;

public:
  // Matches the device malloc alignment guarantee.
  static constexpr uint64_t Alignment = 256;

  static bool isSmall(uint64_t Size) { return Size < ChunkSize; }

  static uint64_t actualSize(uint64_t Size) {
    return std::max(mneme::util::roundUp(Size, Alignment), Alignment);
  }

  SmallAllocator(PageManager<VendorTypes> &PM, int DeviceID)
      : PM(PM), DeviceID(DeviceID) {}

  // Size must come from actualSize(). Nullptr when out of device memory.
  void *allocate(uint64_t Size) {
    for (auto &[Start, C] : Chunks) {
      auto Addr = C.Free.allocate(Size, Alignment);
      if (!Addr)
        continue;
      if (C.Used == 0)
        HasSpare = false;
      C.Used += Size;
      return reinterpret_cast<void *>(*Addr);
    }

    Handle_t H{};
    auto R = PM.mapAddr(ChunkSize, [&](const auto &R) {
      return DT::mapFixed(R.Addr, R.Size, R.Align, DeviceID, H);
    });
    if (!R)
      return nullptr;
    uintptr_t Start = reinterpret_cast<uintptr_t>(R->Addr);
    LOG_DEBUG("New small allocation chunk {}", R->Addr);
    auto &C = Chunks[Start];
    C.Handle = H;
    if (Size < ChunkSize)
      C.Free.release(Start + Size, ChunkSize - Size);
    C.Used = Size;
    return R->Addr;
  }

  void release(void *Addr, uint64_t Size) {
    uintptr_t A = reinterpret_cast<uintptr_t>(Addr);
    auto It = std::prev(Chunks.upper_bound(A));
    auto &C = It->second;
    C.Free.release(A, Size);
    C.Used -= Size;
    if (C.Used)
      return;
    if (!HasSpare) {
      HasSpare = true;
      return;
    }
    DT::unmapFixed(reinterpret_cast<void *>(It->first), ChunkSize, C.Handle);
    PM.releaseAddr(reinterpret_cast<void *>(It->first), ChunkSize);
    Chunks.erase(It);
  }

  ~SmallAllocator() {
    for (auto &[Start, C] : Chunks)
      DT::unmapFixed(reinterpret_cast<void *>(Start), ChunkSize, C.Handle);
  }

  SmallAllocator(const SmallAllocator &) = delete;
  SmallAllocator &operator=(const SmallAllocator &) = delete;
};
