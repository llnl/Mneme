#pragma once
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemePageManager.hpp"

namespace mneme {

// The mapping the vendor runtime made for a managed allocation at an address
// Mneme chose; empty when none was placed.
struct ManagedMapping {
  uintptr_t Start = 0;
  uint64_t Size = 0;

  bool contains(const void *Ptr, uint64_t Bytes) const {
    auto P = reinterpret_cast<uintptr_t>(Ptr);
    return Size && P >= Start && P + Bytes <= Start + Size;
  }
};

class PlaceManagedMmap;
class KeepManagedRangeReserved;

namespace detail {
inline thread_local PlaceManagedMmap *ActivePlacement = nullptr;
inline thread_local KeepManagedRangeReserved *ActiveRelease = nullptr;

inline uint64_t roundUpToPage(uint64_t Size) {
  static const uint64_t PageSize = sysconf(_SC_PAGESIZE);
  return util::roundUp(Size, PageSize);
}

// Replaces whatever is mapped at [Addr, Addr + Len) with reserved address
// space, which Mneme owns.
inline void reserveFixed(void *Addr, size_t Len) {
  void *Got = mmap(Addr, Len, PROT_NONE, ReserveFlags | MAP_FIXED, -1, 0);
  if (Got == MAP_FAILED)
    LOG_FATAL("Cannot keep {} bytes at {} reserved: {}", Len, Addr,
              std::strerror(errno));
}
} // namespace detail

// While alive, the first anonymous mmap that this thread makes without an
// address and that spans the same pages as an allocation of Size bytes is
// placed in address space that Allocator reserved. Replay maps recorded memory
// at its recorded address, so that address must be free in the replay process.
// The runtime's own choice lies in the mmap region, which the replay process
// fills with its libraries. The object must go out of scope before the caller
// does anything else that may map memory, such as logging, so that only the
// runtime's mapping is placed.
class PlaceManagedMmap {
  ChunkAllocator &Allocator;
  // The runtime maps the allocation with its size, which some versions round
  // to whole pages first. Its other mappings are bookkeeping.
  uint64_t PageRoundedSize;
  ManagedMapping Placed;

  template <typename MmapFn>
  friend void *placeOrForwardMmap(MmapFn Mmap, void *Addr, size_t Len, int Prot,
                                  int Flags, int Fd, off_t Off);

public:
  PlaceManagedMmap(ChunkAllocator &Allocator, uint64_t Size)
      : Allocator(Allocator), PageRoundedSize(detail::roundUpToPage(Size)) {
    detail::ActivePlacement = this;
  }
  ~PlaceManagedMmap() { detail::ActivePlacement = nullptr; }

  PlaceManagedMmap(const PlaceManagedMmap &) = delete;
  PlaceManagedMmap &operator=(const PlaceManagedMmap &) = delete;

  const ManagedMapping &placed() const { return Placed; }
};

// While alive, this thread's munmap of exactly Mapping leaves the range
// reserved instead, so no other mapping can take its addresses before Mneme
// hands them out again.
class KeepManagedRangeReserved {
  ManagedMapping Mapping;
  bool WasUnmapped = false;

  template <typename MunmapFn>
  friend int keepOrForwardMunmap(MunmapFn Munmap, void *Addr, size_t Len);

public:
  explicit KeepManagedRangeReserved(const ManagedMapping &Mapping)
      : Mapping(Mapping) {
    detail::ActiveRelease = this;
  }
  ~KeepManagedRangeReserved() { detail::ActiveRelease = nullptr; }

  KeepManagedRangeReserved(const KeepManagedRangeReserved &) = delete;
  KeepManagedRangeReserved &
  operator=(const KeepManagedRangeReserved &) = delete;

  bool wasUnmapped() const { return WasUnmapped; }
};

// The body of the mmap hook. Forwards to Mmap, placing the mapping when a
// PlaceManagedMmap is active on this thread and the call qualifies. A mapping
// that cannot be placed is made where the kernel chooses and leaves the
// placement empty, so the caller reports it.
template <typename MmapFn>
void *placeOrForwardMmap(MmapFn Mmap, void *Addr, size_t Len, int Prot,
                         int Flags, int Fd, off_t Off) {
  PlaceManagedMmap *P = detail::ActivePlacement;
  if (!P || Addr || !(Flags & MAP_ANONYMOUS) || (Flags & MAP_FIXED) ||
      detail::roundUpToPage(Len) != P->PageRoundedSize)
    return Mmap(Addr, Len, Prot, Flags, Fd, Off);

  // Only one mapping is placed, and growing the allocator maps memory itself.
  detail::ActivePlacement = nullptr;
  uint64_t Size = P->Allocator.actualSize(Len);
  void *Want = P->Allocator.allocate(Size);
  if (!Want)
    return Mmap(Addr, Len, Prot, Flags, Fd, Off);
  // Mneme owns the reservation at Want, so the mapping may replace it.
  void *Got = Mmap(Want, Len, Prot, Flags | MAP_FIXED, Fd, Off);
  if (Got == MAP_FAILED) {
    int Err = errno;
    // A failed fixed mmap may already have unmapped the reservation.
    detail::reserveFixed(Want, Size);
    P->Allocator.release(Want, Size);
    errno = Err;
    return MAP_FAILED;
  }
  P->Placed = {reinterpret_cast<uintptr_t>(Got), Len};
  return Got;
}

// The body of the munmap hook. Forwards to Munmap, except for the mapping of a
// KeepManagedRangeReserved active on this thread, which stays reserved.
template <typename MunmapFn>
int keepOrForwardMunmap(MunmapFn Munmap, void *Addr, size_t Len) {
  KeepManagedRangeReserved *R = detail::ActiveRelease;
  if (!R || reinterpret_cast<uintptr_t>(Addr) != R->Mapping.Start ||
      Len != R->Mapping.Size)
    return Munmap(Addr, Len);

  detail::ActiveRelease = nullptr;
  detail::reserveFixed(Addr, Len);
  R->WasUnmapped = true;
  return 0;
}

} // namespace mneme
