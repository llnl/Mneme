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
  mmap(A, Page, PROT_NONE,
       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);

  Allocator.release(A, Page);
  expect(Allocator.allocate(Page) == A, "should reuse a released address");

  uint64_t BigSize = Allocator.actualSize(3 * ChunkSize + 1);
  void *Big = Allocator.allocate(BigSize);
  expect(Big != nullptr, "should allocate more than a chunk");
  expect(isReserved(Big, BigSize), "a large allocation should be reserved");

  Allocator.release(Big, BigSize);
  Allocator.release(B, Page);
  Allocator.release(A, Page);
  return 0;
}
