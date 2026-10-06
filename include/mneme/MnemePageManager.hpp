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

// Picks device addresses for recorded allocations: first fit above one anchor
// address, using /proc/self/maps as the free list.
template <mneme::DeviceVendors VendorTypes> class PageManager {
  using DT = mneme::DeviceTraits<VendorTypes>;
  static constexpr uintptr_t MaxAddr = 1ULL << 47;
  static constexpr int MaxMapTries = 16;

  uint64_t PageSize;
  uintptr_t Anchor = 0;
  // End of the free gap the anchor was picked in.
  uintptr_t Limit = 0;

  void pickAnchor(uint64_t Size) {
    auto Addrs = DT::getCandidateAddrs(Size, mneme::util::LargePageSize);
    if (Addrs.empty())
      LOG_FATAL("No device address range for {} bytes", Size);
    Anchor = Addrs.front();
    auto Free = mneme::util::getFreeVARanges(Anchor, MaxAddr);
    Limit = !Free.empty() && Free.front().Start == Anchor ? Free.front().End
                                                          : MaxAddr;
    LOG_INFO("Device address anchor {} limit {}",
             reinterpret_cast<void *>(Anchor), reinterpret_cast<void *>(Limit));
  }

public:
  struct AddrRange {
    void *Addr;
    uint64_t Size;
    uint64_t Align;
  };

  explicit PageManager(uint64_t PageSize) : PageSize(PageSize) {}

  // Maps a range with Map(Range), skipping occupied addresses. Large sizes get
  // large-page alignment so they can use big GPU fragments. Nullopt when the
  // device is out of memory.
  template <typename MapFn>
  std::optional<AddrRange> mapAddr(uint64_t Size, MapFn Map) {
    uint64_t Align = Size >= mneme::util::LargePageSize
                         ? mneme::util::LargePageSize
                         : PageSize;
    uint64_t ActualSize = std::max(mneme::util::roundUp(Size, Align), PageSize);

    if (!Anchor)
      pickAnchor(ActualSize);
    uintptr_t From = Anchor;
    bool Reanchored = false;
    int Tries = 0;
    while (true) {
      auto Addr = mneme::util::firstFreeVAddr(From, Limit, ActualSize, Align);
      if (!Addr) {
        if (Reanchored)
          LOG_FATAL("No device address range for {} bytes", Size);
        pickAnchor(ActualSize);
        From = Anchor;
        Reanchored = true;
        continue;
      }
      AddrRange R{reinterpret_cast<void *>(*Addr), ActualSize, Align};
      switch (Map(R)) {
      case mneme::MapStatus::Mapped:
        return R;
      case mneme::MapStatus::OutOfMemory:
        return std::nullopt;
      case mneme::MapStatus::Occupied:
        // Blocked by something /proc/self/maps does not show.
        if (++Tries == MaxMapTries)
          LOG_FATAL("Cannot map {} bytes at {}:\n{}", Size, R.Addr,
                    mneme::util::getMappingsIn(*Addr - ActualSize,
                                               *Addr + 2 * ActualSize));
        LOG_DEBUG("Device address {} is occupied, retrying", R.Addr);
        From = *Addr + ActualSize;
        break;
      }
    }
  }
};

// Packs allocations into shared ChunkSize-byte chunks, each one device mapping,
// since mapping each allocation separately is slow. Chunks shrink when the
// device is short on memory.
template <mneme::DeviceVendors VendorTypes> class ChunkAllocator {
  using DT = mneme::DeviceTraits<VendorTypes>;
  using Handle_t = typename DT::MemoryAllocationHandle_t;

  struct Chunk {
    Handle_t Handle{};
    uint64_t Size = 0;
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

  typename ChunkMap::iterator chunkOf(uintptr_t Addr) {
    return std::prev(Chunks.upper_bound(Addr));
  }

  bool mapChunk(uint64_t Size) {
    Handle_t H{};
    auto R = PM.mapAddr(Size, [&](const auto &R) {
      return DT::mapFixed(R.Addr, R.Size, R.Align, DeviceID, H);
    });
    if (!R)
      return false;
    uintptr_t Start = reinterpret_cast<uintptr_t>(R->Addr);
    LOG_DEBUG("New {}-byte allocation chunk {}", Size, R->Addr);
    Chunks[Start] = {H, Size, 0};
    Free.release(Start, Size, Start, Start + Size);
    return true;
  }

  // Halves the chunk while the device lacks memory, down to MinSize.
  bool growFor(uint64_t MinSize) {
    uint64_t Min = mneme::util::roundUp(MinSize, mneme::util::LargePageSize);
    for (uint64_t Size = ChunkSize;;
         Size = std::max(Min, mneme::util::roundUp(
                                  Size / 2, mneme::util::LargePageSize))) {
      if (mapChunk(Size))
        return true;
      if (Size == Min)
        return false;
    }
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
      if (!growFor(Size))
        return nullptr;
      Addr = Free.allocate(Size, Alignment);
    }
    chunkOf(*Addr)->second.Used += Size;
    return reinterpret_cast<void *>(*Addr);
  }

  void release(void *Addr, uint64_t Size) {
    uintptr_t A = reinterpret_cast<uintptr_t>(Addr);
    auto It = chunkOf(A);
    auto &C = It->second;
    uintptr_t Start = It->first;
    Free.release(A, Size, Start, Start + C.Size);
    C.Used -= Size;
    // The last chunk stays mapped so malloc/free loops don't remap.
    if (C.Used || Chunks.size() == 1)
      return;
    Free.remove(Start);
    DT::unmapFixed(reinterpret_cast<void *>(Start), C.Size, C.Handle);
    Chunks.erase(It);
  }

  ~ChunkAllocator() {
    for (auto &[Start, C] : Chunks)
      DT::unmapFixed(reinterpret_cast<void *>(Start), C.Size, C.Handle);
  }

  ChunkAllocator(const ChunkAllocator &) = delete;
  ChunkAllocator &operator=(const ChunkAllocator &) = delete;
};
