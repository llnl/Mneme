#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <utility>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"
#include "mneme/MnemeVASpace.hpp"

// Free address ranges with best-fit allocation and coalescing release.
class FreeRanges {
  using AddrMap = std::map<uintptr_t, uint64_t>;
  // Start -> size.
  AddrMap ByAddr;
  // (Size, start) of the same ranges.
  std::set<std::pair<uint64_t, uintptr_t>> BySize;

  void insert(uintptr_t Start, uint64_t Size) {
    ByAddr.emplace(Start, Size);
    BySize.emplace(Size, Start);
  }

  AddrMap::iterator erase(AddrMap::iterator It) {
    BySize.erase({It->second, It->first});
    return ByAddr.erase(It);
  }

public:
  std::optional<uintptr_t> allocate(uint64_t Size, uint64_t Align) {
    for (auto It = BySize.lower_bound({Size, 0}); It != BySize.end(); ++It) {
      auto [RangeSize, Start] = *It;
      uintptr_t Addr = mneme::util::roundUp(Start, Align);
      uintptr_t End = Start + RangeSize;
      if (Addr + Size > End)
        continue;
      BySize.erase(It);
      ByAddr.erase(Start);
      if (Addr > Start)
        insert(Start, Addr - Start);
      if (Addr + Size < End)
        insert(Addr + Size, End - Addr - Size);
      return Addr;
    }
    return std::nullopt;
  }

  // Merges only with free neighbors inside [Lo, Hi).
  void release(uintptr_t Start, uint64_t Size, uintptr_t Lo = 0,
               uintptr_t Hi = UINTPTR_MAX) {
    uintptr_t End = Start + Size;
    auto Next = ByAddr.lower_bound(Start);
    if (Next != ByAddr.end() && Next->first == End && End < Hi) {
      End += Next->second;
      Next = erase(Next);
    }
    if (Next != ByAddr.begin()) {
      auto Prev = std::prev(Next);
      if (Prev->first + Prev->second == Start && Prev->first >= Lo) {
        Start = Prev->first;
        erase(Prev);
      }
    }
    insert(Start, End - Start);
  }

  // Drops the free range that starts at Start.
  void remove(uintptr_t Start) { erase(ByAddr.find(Start)); }
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

// Packs allocations into shared ChunkSize-byte chunks, each one device mapping,
// since mapping each allocation separately is slow.
template <mneme::DeviceVendors VendorTypes> class ChunkAllocator {
  using DT = mneme::DeviceTraits<VendorTypes>;
  using Handle_t = typename DT::MemoryAllocationHandle_t;

  struct Chunk {
    Handle_t Handle{};
    uint64_t Used = 0;
  };
  using ChunkMap = std::map<uintptr_t, Chunk>;

  PageManager<VendorTypes> &PM;
  int DeviceID;
  uint64_t ChunkSize;
  // Start -> chunk.
  ChunkMap Chunks;
  // Chunks may be adjacent but are separate mappings, so ranges never span two.
  FreeRanges Free;
  // One empty chunk stays mapped so malloc/free loops don't remap.
  bool HasSpare = false;

  typename ChunkMap::iterator chunkOf(uintptr_t Addr) {
    return std::prev(Chunks.upper_bound(Addr));
  }

  bool mapChunk() {
    Handle_t H{};
    auto R = PM.mapAddr(ChunkSize, [&](const auto &R) {
      return DT::mapFixed(R.Addr, R.Size, R.Align, DeviceID, H);
    });
    if (!R)
      return false;
    uintptr_t Start = reinterpret_cast<uintptr_t>(R->Addr);
    LOG_DEBUG("New {}-byte allocation chunk {}", ChunkSize, R->Addr);
    Chunks[Start].Handle = H;
    Free.release(Start, ChunkSize, Start, Start + ChunkSize);
    return true;
  }

public:
  // Matches the device malloc alignment guarantee.
  static constexpr uint64_t Alignment = 256;

  static uint64_t actualSize(uint64_t Size) {
    return std::max(mneme::util::roundUp(Size, Alignment), Alignment);
  }

  ChunkAllocator(PageManager<VendorTypes> &PM, int DeviceID, uint64_t ChunkSize)
      : PM(PM), DeviceID(DeviceID), ChunkSize(ChunkSize) {}

  bool packs(uint64_t Size) const { return Size < ChunkSize; }

  // Size must come from actualSize(). Nullptr when out of device memory.
  void *allocate(uint64_t Size) {
    auto Addr = Free.allocate(Size, Alignment);
    if (!Addr) {
      if (!mapChunk())
        return nullptr;
      Addr = Free.allocate(Size, Alignment);
    }
    auto &C = chunkOf(*Addr)->second;
    if (C.Used == 0)
      HasSpare = false;
    C.Used += Size;
    return reinterpret_cast<void *>(*Addr);
  }

  void release(void *Addr, uint64_t Size) {
    uintptr_t A = reinterpret_cast<uintptr_t>(Addr);
    auto It = chunkOf(A);
    auto &C = It->second;
    uintptr_t Start = It->first;
    Free.release(A, Size, Start, Start + ChunkSize);
    C.Used -= Size;
    if (C.Used)
      return;
    if (!HasSpare) {
      HasSpare = true;
      return;
    }
    Free.remove(Start);
    DT::unmapFixed(reinterpret_cast<void *>(Start), ChunkSize, C.Handle);
    PM.releaseAddr(reinterpret_cast<void *>(Start), ChunkSize);
    Chunks.erase(It);
  }

  ~ChunkAllocator() {
    for (auto &[Start, C] : Chunks)
      DT::unmapFixed(reinterpret_cast<void *>(Start), ChunkSize, C.Handle);
  }

  ChunkAllocator(const ChunkAllocator &) = delete;
  ChunkAllocator &operator=(const ChunkAllocator &) = delete;
};
