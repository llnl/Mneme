#pragma once
#include <cerrno>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>

#include "mneme/MnemeVASpace.hpp"

namespace mneme {

// Where the mapping behind a vendor runtime allocation should go. Replay maps
// recorded memory at its recorded address, so that address must be free in the
// replay process. The runtime's own choice lies in the mmap region, which the
// replay process fills with its libraries, so the mapping is placed in [Lo, Hi)
// instead.
struct RuntimeMmapPlacement {
  uintptr_t Lo;
  uintptr_t Hi;
  // Smaller mappings are runtime bookkeeping, not the allocation.
  uint64_t MinSize;
  // The placed mapping; empty until one is placed.
  uintptr_t Start = 0;
  uint64_t Size = 0;

  bool contains(const void *Ptr, uint64_t Bytes) const {
    auto P = reinterpret_cast<uintptr_t>(Ptr);
    return Size && P >= Start && P + Bytes <= Start + Size;
  }
};

namespace detail {
inline thread_local RuntimeMmapPlacement *ActivePlacement = nullptr;
} // namespace detail

// While in scope, the first anonymous mmap of at least MinSize bytes that this
// thread makes without an address is placed by Placement.
class PlaceRuntimeMmap {
public:
  explicit PlaceRuntimeMmap(RuntimeMmapPlacement &Placement) {
    detail::ActivePlacement = &Placement;
  }
  ~PlaceRuntimeMmap() { detail::ActivePlacement = nullptr; }

  PlaceRuntimeMmap(const PlaceRuntimeMmap &) = delete;
  PlaceRuntimeMmap &operator=(const PlaceRuntimeMmap &) = delete;
};

// The body of the mmap hook. Forwards to Mmap, placing the mapping when a
// PlaceRuntimeMmap is active on this thread and the call qualifies. A mapping
// that cannot be placed is made where the kernel chooses and leaves the
// placement empty, so the caller reports it.
template <typename MmapFn>
void *placeOrForwardMmap(MmapFn Mmap, void *Addr, size_t Len, int Prot,
                         int Flags, int Fd, off_t Off) {
  RuntimeMmapPlacement *P = detail::ActivePlacement;
  if (!P || Addr || !(Flags & MAP_ANONYMOUS) || (Flags & MAP_FIXED) ||
      Len < P->MinSize)
    return Mmap(Addr, Len, Prot, Flags, Fd, Off);

  // Only one mapping is placed, and the search below may itself map memory.
  detail::ActivePlacement = nullptr;
  static const uint64_t PageSize = sysconf(_SC_PAGESIZE);
  // Another thread may take the free range between the search and the mmap.
  for (int Attempt = 0; Attempt < 8; ++Attempt) {
    auto Free = util::firstFreeVAddr(P->Lo, P->Hi, Len, PageSize);
    if (!Free)
      break;
    void *Want = reinterpret_cast<void *>(*Free);
    void *Got = Mmap(Want, Len, Prot, Flags | MAP_FIXED_NOREPLACE, Fd, Off);
    if (Got == Want) {
      P->Start = *Free;
      P->Size = Len;
      return Got;
    }
    // Kernels before 4.17 treat the address as a hint.
    if (Got != MAP_FAILED)
      return Got;
    if (errno != EEXIST)
      return MAP_FAILED;
  }
  return Mmap(Addr, Len, Prot, Flags, Fd, Off);
}

} // namespace mneme
