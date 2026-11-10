#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <sys/mman.h>
#include <unistd.h>
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

  uintptr_t Anchor = 0;

  void pickAnchor(uint64_t Size) {
    auto Addrs = mneme::util::suggestVAddrs(Size, mneme::util::LargePageSize);
    if (Addrs.empty())
      LOG_FATAL("No device address range for {} bytes", Size);
    Anchor = Addrs.front();
    LOG_INFO("Device address anchor {}", reinterpret_cast<void *>(Anchor));
  }

public:
  // Mapping granularity: large pages, and whole driver VA blocks on CUDA.
  static constexpr uint64_t Align =
      std::max(mneme::util::LargePageSize, DT::VABlockSize);

  struct AddrRange {
    void *Addr;
    uint64_t Size;
    uint64_t Align;
  };

  // Maps a range with Map(Range) at the first free address above the anchor.
  // Ranges are large-page aligned so they can use big GPU fragments, and cover
  // whole driver VA blocks. Nullopt when the device is out of memory.
  template <typename MapFn>
  std::optional<AddrRange> mapAddr(uint64_t Size, MapFn Map) {
    uint64_t ActualSize = mneme::util::roundUp(Size, Align);

    if (!Anchor)
      pickAnchor(ActualSize);
    auto Addr = mneme::util::firstFreeVAddr(Anchor, MaxAddr, ActualSize, Align);
    if (!Addr)
      LOG_FATAL("No device address range for {} bytes", Size);
    AddrRange R{reinterpret_cast<void *>(*Addr), ActualSize, Align};
    switch (Map(R)) {
    case mneme::MapStatus::Mapped:
      return R;
    case mneme::MapStatus::OutOfMemory:
      return std::nullopt;
    case mneme::MapStatus::Occupied:
      break;
    }
    // Blocked by something /proc/self/maps does not show.
    LOG_FATAL(
        "Driver placed {} bytes away from {}:\n{}", Size, R.Addr,
        mneme::util::getMappingsIn(*Addr - ActualSize, *Addr + 2 * ActualSize));
  }
};

// Backing memory for the chunks of a ChunkAllocator.
class ChunkSource {
public:
  struct Range {
    uintptr_t Start;
    uint64_t Size;
  };

  // Chunk sizes are multiples of this.
  virtual uint64_t unit() const = 0;
  // Alignment and size granularity of allocations inside chunks.
  virtual uint64_t alignment() const = 0;
  // Maps one chunk of at least Size bytes. Nullopt when out of memory.
  virtual std::optional<Range> map(uint64_t Size) = 0;
  virtual void unmap(uintptr_t Start, uint64_t Size) = 0;
  // Called before a failed map is retried with a smaller chunk.
  virtual void discardFailure() {}

  virtual ~ChunkSource() = default;
};

// Chunks of device memory, each one device mapping.
template <mneme::DeviceVendors VendorTypes>
class DeviceChunkSource final : public ChunkSource {
  using DT = mneme::DeviceTraits<VendorTypes>;
  using Handle_t = typename DT::MemoryAllocationHandle_t;

  PageManager<VendorTypes> &PM;
  int DeviceID;
  // Chunk start -> its device memory.
  std::map<uintptr_t, Handle_t> Handles;

public:
  DeviceChunkSource(PageManager<VendorTypes> &PM, int DeviceID)
      : PM(PM), DeviceID(DeviceID) {}

  // Chunks are whole mapping units, so CUDA chunks are 32 MiB multiples.
  uint64_t unit() const override { return PageManager<VendorTypes>::Align; }

  // Matches the device malloc alignment guarantee.
  uint64_t alignment() const override { return 256; }

  std::optional<Range> map(uint64_t Size) override {
    Handle_t H{};
    auto R = PM.mapAddr(Size, [&](const auto &R) {
      return DT::mapFixed(R.Addr, R.Size, R.Align, DeviceID, H);
    });
    if (!R)
      return std::nullopt;
    uintptr_t Start = reinterpret_cast<uintptr_t>(R->Addr);
    Handles[Start] = H;
    return Range{Start, R->Size};
  }

  void unmap(uintptr_t Start, uint64_t Size) override {
    auto It = Handles.find(Start);
    DT::unmapFixed(reinterpret_cast<void *>(Start), Size, DeviceID, It->second);
    Handles.erase(It);
  }

  // The app sees only the final failure.
  void discardFailure() override { DT::clearLastError(); }
};

namespace mneme {
inline constexpr int ReserveFlags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
} // namespace mneme

// Chunks of reserved address space with no memory behind them. The vendor
// runtime's own mappings are placed inside them.
template <mneme::DeviceVendors VendorTypes>
class ReservedChunkSource final : public ChunkSource {
  PageManager<VendorTypes> &PM;
  uint64_t PageSize = sysconf(_SC_PAGESIZE);

  static mneme::MapStatus reserve(void *Addr, uint64_t Size) {
    void *Got = mmap(Addr, Size, PROT_NONE,
                     mneme::ReserveFlags | MAP_FIXED_NOREPLACE, -1, 0);
    if (Got == Addr)
      return mneme::MapStatus::Mapped;
    // Kernels before 4.17 treat the address as a hint.
    if (Got != MAP_FAILED) {
      munmap(Got, Size);
      return mneme::MapStatus::Occupied;
    }
    if (errno == EEXIST)
      return mneme::MapStatus::Occupied;
    if (errno == ENOMEM)
      return mneme::MapStatus::OutOfMemory;
    LOG_FATAL("Cannot reserve {} bytes at {}: {}", Size, Addr,
              std::strerror(errno));
  }

public:
  explicit ReservedChunkSource(PageManager<VendorTypes> &PM) : PM(PM) {}

  uint64_t unit() const override { return PageManager<VendorTypes>::Align; }

  uint64_t alignment() const override { return PageSize; }

  std::optional<Range> map(uint64_t Size) override {
    auto R =
        PM.mapAddr(Size, [](const auto &R) { return reserve(R.Addr, R.Size); });
    if (!R)
      return std::nullopt;
    return Range{reinterpret_cast<uintptr_t>(R->Addr), R->Size};
  }

  void unmap(uintptr_t Start, uint64_t Size) override {
    if (munmap(reinterpret_cast<void *>(Start), Size))
      LOG_WARN("Could not unmap {}: {}", reinterpret_cast<void *>(Start),
               std::strerror(errno));
  }
};

// Packs allocations into shared ChunkSize-byte chunks mapped by a ChunkSource,
// since mapping each allocation separately is slow. Chunks shrink when the
// source is short on memory. An allocation larger than ChunkSize gets a
// dedicated chunk that nothing else shares and that is unmapped on release.
class ChunkAllocator {
  struct Chunk {
    uint64_t Size = 0;
    uint64_t Used = 0;
  };
  using ChunkMap = std::map<uintptr_t, Chunk>;

  ChunkSource &Source;
  uint64_t ChunkSize;
  // Start -> chunk.
  ChunkMap Chunks;
  // Chunks may be adjacent but are separate mappings, so ranges never span two.
  FreeRanges Free;

  ChunkMap::iterator chunkOf(uintptr_t Addr) {
    return std::prev(Chunks.upper_bound(Addr));
  }

  // Shared chunks only shrink from ChunkSize, and a dedicated chunk holds more
  // than ChunkSize bytes.
  bool isDedicated(const Chunk &C) const { return C.Size > ChunkSize; }

  void *allocateDedicated(uint64_t Size) {
    auto R = Source.map(Size);
    if (!R)
      return nullptr;
    LOG_DEBUG("New {}-byte dedicated chunk {}", R->Size,
              reinterpret_cast<void *>(R->Start));
    Chunks[R->Start] = {R->Size, Size};
    return reinterpret_cast<void *>(R->Start);
  }

  bool mapChunk(uint64_t Size) {
    auto R = Source.map(Size);
    if (!R)
      return false;
    LOG_DEBUG("New {}-byte allocation chunk {}", R->Size,
              reinterpret_cast<void *>(R->Start));
    Chunks[R->Start] = {R->Size, 0};
    Free.release(R->Start, R->Size, R->Start, R->Start + R->Size);
    return true;
  }

  // Halves the chunk while the source lacks memory, down to MinSize.
  bool growFor(uint64_t MinSize) {
    uint64_t Unit = Source.unit();
    uint64_t Min = mneme::util::roundUp(MinSize, Unit);
    for (uint64_t Size = ChunkSize;;
         Size = std::max(Min, mneme::util::roundUp(Size / 2, Unit))) {
      if (mapChunk(Size))
        return true;
      if (Size == Min)
        return false;
      Source.discardFailure();
    }
  }

public:
  ChunkAllocator(ChunkSource &Source, uint64_t ChunkSize)
      : Source(Source),
        ChunkSize(mneme::util::roundUp(ChunkSize, Source.unit())) {}

  uint64_t actualSize(uint64_t Size) const {
    uint64_t Alignment = Source.alignment();
    return std::max(mneme::util::roundUp(Size, Alignment), Alignment);
  }

  // Size must come from actualSize().
  bool packs(uint64_t Size) const { return Size <= ChunkSize; }

  // Size must come from actualSize(). Nullptr when out of memory.
  void *allocate(uint64_t Size) {
    if (!packs(Size))
      return allocateDedicated(Size);
    auto Addr = Free.allocate(Size, Source.alignment());
    if (!Addr) {
      if (!growFor(Size))
        return nullptr;
      Addr = Free.allocate(Size, Source.alignment());
    }
    chunkOf(*Addr)->second.Used += Size;
    return reinterpret_cast<void *>(*Addr);
  }

  void release(void *Addr, uint64_t Size) {
    uintptr_t A = reinterpret_cast<uintptr_t>(Addr);
    auto It = chunkOf(A);
    auto &C = It->second;
    uintptr_t Start = It->first;
    if (isDedicated(C)) {
      Source.unmap(Start, C.Size);
      Chunks.erase(It);
      return;
    }
    Free.release(A, Size, Start, Start + C.Size);
    C.Used -= Size;
    // One empty chunk stays mapped so malloc/free loops don't remap.
    if (C.Used || std::none_of(Chunks.begin(), Chunks.end(), [&](auto &E) {
          return E.first != Start && !E.second.Used;
        }))
      return;
    Free.remove(Start);
    Source.unmap(Start, C.Size);
    Chunks.erase(It);
  }

  ~ChunkAllocator() {
    for (auto &[Start, C] : Chunks)
      Source.unmap(Start, C.Size);
  }

  ChunkAllocator(const ChunkAllocator &) = delete;
  ChunkAllocator &operator=(const ChunkAllocator &) = delete;
};
