#include "mneme/MnemeManagedPlacement.hpp"
#include "mneme/MnemePageManager.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

using namespace mneme;

#ifdef MNEME_ENABLE_HIP
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
#elif defined(MNEME_ENABLE_CUDA)
constexpr DeviceVendors Vendor = DeviceVendors::CUDA;
#endif

namespace {

void expect(bool Condition, const std::string &Message) {
  if (!Condition) {
    std::cerr << Message << "\n";
    std::exit(1);
  }
}

// Whether [Addr, Addr + Size) is mapped, which a fresh mapping must not
// replace.
bool isReserved(void *Addr, uint64_t Size) {
  void *Got = mmap(Addr, Size, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (Got == MAP_FAILED)
    return errno == EEXIST;
  munmap(Got, Size);
  return false;
}

void *forwardMmap(void *Addr, size_t Len) {
  auto Mmap = [](void *Addr, size_t Len, int Prot, int Flags, int Fd,
                 off_t Off) { return mmap(Addr, Len, Prot, Flags, Fd, Off); };
  void *Got = placeOrForwardMmap(Mmap, Addr, Len, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  expect(Got != MAP_FAILED, "mmap should succeed");
  return Got;
}

// Only the mapping that spans the pages of an allocation of Size bytes is
// placed. The runtime maps it with MappedLen, which is Size or Size rounded to
// whole pages.
void expectPlacesOnlyAllocation(ChunkAllocator &Allocator, uint64_t Size,
                                uint64_t MappedLen) {
  uint64_t Page = sysconf(_SC_PAGESIZE);
  uint64_t Rounded = util::roundUp(Size, Page);
  PlaceManagedMmap Placement(Allocator, Size);

  for (uint64_t Other : {Rounded - Page, Rounded + Page, 16 * Rounded}) {
    void *Unplaced = forwardMmap(nullptr, Other);
    expect(Placement.placed().Size == 0,
           "a mapping of another size should not be placed");
    munmap(Unplaced, Other);
  }

  void *Hinted = forwardMmap(&Page, MappedLen);
  expect(Placement.placed().Size == 0,
         "a mapping with an address should not be placed");
  munmap(Hinted, MappedLen);

  void *Got = forwardMmap(nullptr, MappedLen);
  expect(Placement.placed().contains(Got, Size) &&
             Placement.placed().Size == MappedLen,
         "the allocation's mapping should be placed with its length");
  std::memset(Got, 1, Size);

  void *Second = forwardMmap(nullptr, MappedLen);
  expect(Placement.placed().Start == reinterpret_cast<uintptr_t>(Got),
         "only one mapping should be placed");
  munmap(Second, MappedLen);

  detail::reserveFixed(Got, Rounded);
  Allocator.release(Got, Rounded);
  expect(Allocator.allocate(Rounded) == Got,
         "the placed mapping should lie in the allocator's address space");
  Allocator.release(Got, Rounded);
}

} // namespace

int main() {
  uint64_t Page = sysconf(_SC_PAGESIZE);
  uint64_t ChunkSize = 2 * PageManager<Vendor>::Align;
  PageManager<Vendor> PM;
  ReservedChunkSource<Vendor> Source(PM);
  ChunkAllocator Allocator(Source, ChunkSize);

  expect(Allocator.actualSize(1) == Page, "should round to whole pages");
  void *A = Allocator.allocate(Page);
  void *B = Allocator.allocate(Page);
  expect(A && B && A != B, "should allocate distinct addresses");
  expect(reinterpret_cast<uintptr_t>(A) % Page == 0, "should align to pages");
  expect(isReserved(A, Page) && isReserved(B, Page),
         "allocations should lie in reserved address space");

  // The runtime maps its memory over the reservation.
  void *Mapped = mmap(A, Page, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  expect(Mapped == A, "should map over the reservation");
  std::memset(A, 1, Page);
  detail::reserveFixed(A, Page);

  Allocator.release(A, Page);
  expect(Allocator.allocate(Page) == A, "should reuse a released address");

  uint64_t BigSize = Allocator.actualSize(3 * ChunkSize + 1);
  void *Big = Allocator.allocate(BigSize);
  expect(Big != nullptr, "should allocate more than a chunk");
  expect(isReserved(Big, BigSize), "a large allocation should be reserved");
  auto *C = static_cast<char *>(Allocator.allocate(Page));
  uint64_t BigChunkSize = util::roundUp(BigSize, PageManager<Vendor>::Align);
  expect(C && (C < Big || C >= static_cast<char *>(Big) + BigChunkSize),
         "a large allocation should not share its chunk");
  Allocator.release(C, Page);

  Allocator.release(Big, BigSize);
  expect(!isReserved(Big, BigSize),
         "a released large allocation should be unmapped");

  uint64_t Unrounded = 2 * Page + 100;
  expectPlacesOnlyAllocation(Allocator, Unrounded, Unrounded);
  expectPlacesOnlyAllocation(Allocator, Unrounded, 3 * Page);

  Allocator.release(B, Page);
  Allocator.release(A, Page);
  return 0;
}
