#include "mneme/MnemeVASpace.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

using namespace mneme::util;

namespace {

void expect(bool Condition, const std::string &Message) {
  if (!Condition) {
    std::cerr << Message << "\n";
    std::exit(1);
  }
}

} // namespace

int main() {
  uint64_t Page = sysconf(_SC_PAGESIZE);
  // Pages [0, 2) and [5, 6) stay mapped; [2, 5) and [6, 16) are free.
  void *Base =
      mmap(nullptr, 16 * Page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  expect(Base != MAP_FAILED, "mmap failed");
  uintptr_t B = reinterpret_cast<uintptr_t>(Base);
  munmap(reinterpret_cast<void *>(B + 2 * Page), 3 * Page);
  munmap(reinterpret_cast<void *>(B + 6 * Page), 10 * Page);
  uintptr_t Hi = B + 16 * Page;

  expect(firstFreeVAddr(B, Hi, Page, Page) == B + 2 * Page,
         "should skip mapped pages");
  expect(firstFreeVAddr(B, Hi, 3 * Page, Page) == B + 2 * Page,
         "should take an exact fit");
  expect(firstFreeVAddr(B, Hi, 4 * Page, Page) == B + 6 * Page,
         "should skip a small gap");
  expect(firstFreeVAddr(B + 3 * Page, Hi, 3 * Page, Page) == B + 6 * Page,
         "should start at Lo");
  uint64_t Align = 4 * Page;
  expect(firstFreeVAddr(B, Hi, Align, Align) ==
             ((B + 6 * Page + Align - 1) & ~(Align - 1)),
         "should align");
  expect(!firstFreeVAddr(B, Hi, 11 * Page, Page), "should not pass Hi");

  munmap(Base, 2 * Page);
  munmap(reinterpret_cast<void *>(B + 5 * Page), Page);
  return 0;
}
