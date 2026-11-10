#pragma once
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>

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

// Where the mapping behind a managed allocation should go. Replay maps recorded
// memory at its recorded address, so that address must be free in the replay
// process. The runtime's own choice lies in the mmap region, which the
// replay process fills with its libraries, so the mapping is placed in address
// space that Allocator reserved instead.
struct ManagedMmapPlacement {
  ChunkAllocator &Allocator;
  // Smaller mappings are runtime bookkeeping, not the allocation.
  uint64_t MinSize;
  ManagedMapping Placed;
};

// The runtime's unmap of a placed mapping while it frees the managed
// allocation.
struct ManagedMunmapRelease {
  ManagedMapping Mapping;
  bool WasUnmapped = false;
};

namespace detail {
inline thread_local ManagedMmapPlacement *ActivePlacement = nullptr;
inline thread_local ManagedMunmapRelease *ActiveRelease = nullptr;

// Replaces whatever is mapped at [Addr, Addr + Len) with reserved address
// space, which Mneme owns.
inline void reserveFixed(void *Addr, size_t Len) {
  void *Got = mmap(Addr, Len, PROT_NONE, ReserveFlags | MAP_FIXED, -1, 0);
  if (Got == MAP_FAILED)
    LOG_FATAL("Cannot keep {} bytes at {} reserved: {}", Len, Addr,
              std::strerror(errno));
}
} // namespace detail

// While in scope, the first anonymous mmap of at least MinSize bytes that this
// thread makes without an address is placed by Placement.
class PlaceManagedMmap {
public:
  explicit PlaceManagedMmap(ManagedMmapPlacement &Placement) {
    detail::ActivePlacement = &Placement;
  }
  ~PlaceManagedMmap() { detail::ActivePlacement = nullptr; }

  PlaceManagedMmap(const PlaceManagedMmap &) = delete;
  PlaceManagedMmap &operator=(const PlaceManagedMmap &) = delete;
};

// While in scope, this thread's munmap of exactly Release's mapping leaves the
// range reserved instead, so no other mapping can take its addresses before
// Mneme hands them out again.
class KeepManagedRangeReserved {
public:
  explicit KeepManagedRangeReserved(ManagedMunmapRelease &Release) {
    detail::ActiveRelease = &Release;
  }
  ~KeepManagedRangeReserved() { detail::ActiveRelease = nullptr; }

  KeepManagedRangeReserved(const KeepManagedRangeReserved &) = delete;
  KeepManagedRangeReserved &
  operator=(const KeepManagedRangeReserved &) = delete;
};

// The body of the mmap hook. Forwards to Mmap, placing the mapping when a
// PlaceManagedMmap is active on this thread and the call qualifies. A mapping
// that cannot be placed is made where the kernel chooses and leaves the
// placement empty, so the caller reports it.
template <typename MmapFn>
void *placeOrForwardMmap(MmapFn Mmap, void *Addr, size_t Len, int Prot,
                         int Flags, int Fd, off_t Off) {
  ManagedMmapPlacement *P = detail::ActivePlacement;
  if (!P || Addr || !(Flags & MAP_ANONYMOUS) || (Flags & MAP_FIXED) ||
      Len < P->MinSize)
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
  ManagedMunmapRelease *R = detail::ActiveRelease;
  if (!R || reinterpret_cast<uintptr_t>(Addr) != R->Mapping.Start ||
      Len != R->Mapping.Size)
    return Munmap(Addr, Len);

  detail::ActiveRelease = nullptr;
  detail::reserveFixed(Addr, Len);
  R->WasUnmapped = true;
  return 0;
}

} // namespace mneme
