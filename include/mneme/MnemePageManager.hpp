#pragma once
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <set>
#include <sys/types.h>
#include <vector>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"
#include "mneme/MnemeVASpace.hpp"

struct ContiguousAddrBlock {
  // Starting address of the free block
  uintptr_t PageAddr;
  // Size of the free block
  uint64_t Size;

  ContiguousAddrBlock(uintptr_t start, uint64_t sz);

  // Comparison operators for sorting blocks by address and size
  bool operator<(const ContiguousAddrBlock &other) const;
};

template <mneme::DeviceVendors VendorTypes> class PageManager {

protected:
  std::set<ContiguousAddrBlock> FreeVARanges;
  using MnemeDeviceRT = typename mneme::DeviceTraits<VendorTypes>;
  typename MnemeDeviceRT::MemoryAllocationHandle_t MemHandle;
  uintptr_t ReservedVA;
  uint64_t TotalVASize;
  uint64_t PageSize;
  int32_t DeviceID;

  // Erase a block from the set
  bool EraseVirtualAddress(uintptr_t addr, size_t size);
  // Function to coalesce contiguous blocks

  // Function to coalesce contiguous blocks
  void coalesce() {
    if (FreeVARanges.size() < 2)
      return; // Nothing to coalesce if less than two blocks

    auto it = FreeVARanges.begin();
    while (it != FreeVARanges.end()) {
      auto next_it = std::next(it);

      if (next_it == FreeVARanges.end())
        break;

      // Check if the current block is contiguous with the next block
      if (it->PageAddr + it->Size == next_it->PageAddr) {
        // Merge the two blocks
        ContiguousAddrBlock mergedBlock = {it->PageAddr,
                                           it->Size + next_it->Size};

        // Remove the original two blocks
        it = FreeVARanges.erase(it);
        next_it = FreeVARanges.erase(next_it);

        // Insert the merged block
        FreeVARanges.insert(mergedBlock);

        // Start over from the merged block (to ensure we handle multiple
        // contiguous blocks)
        it = FreeVARanges.find(mergedBlock);
      } else {
        ++it;
      }
    }
  }
  // Find a block that has a size >= requestedSize
  std::set<ContiguousAddrBlock>::iterator findFreeBlock(size_t requestedSize) {
    for (auto it = FreeVARanges.begin(); it != FreeVARanges.end(); ++it) {
      auto size = it->Size;
      if (size >= requestedSize) {
        return it;
      }
    }
    return FreeVARanges.end();
  }
  // Find a block that includes the range [Addr, Addr + size)

  std::set<ContiguousAddrBlock>::iterator findInclusivePage(uintptr_t Addr,
                                                            size_t Size) {
    uintptr_t request_end = Addr + Size;
    for (auto it = FreeVARanges.begin(); it != FreeVARanges.end(); ++it) {
      uintptr_t block_end = it->PageAddr + it->Size;
      if (it->PageAddr <= Addr && block_end >= request_end) {
        return it;
      }
    }
    return FreeVARanges.end();
  }

  std::pair<void *, uint64_t> reserveBestFitPage(uint64_t VASize) {
    // We need to always reserve at least a single page
    uint64_t ReqSize = mneme::util::roundUp(VASize, PageSize);
    LOG_DEBUG("The requsted size is rounded up from {} to {}", VASize, ReqSize);
    auto FreeNode = findFreeBlock(ReqSize);
    if (FreeNode == FreeVARanges.end()) {
      LOG_FATAL("We do not have any memory to give");
      return std::make_pair(nullptr, 0);
    }

    auto Ptr = FreeNode->PageAddr;
    auto NodePageSize = FreeNode->Size;

    FreeVARanges.erase(FreeNode);

    if (ReqSize == NodePageSize)
      return std::make_pair((void *)Ptr, ReqSize);

    auto NewPtr = Ptr + ReqSize;
    auto RemainingSize = NodePageSize - ReqSize;

    ContiguousAddrBlock block{NewPtr, RemainingSize};
    FreeVARanges.insert(block);

    // This can be expensive. Currently we coalesce in every request that
    // modifies our free-pages.
    coalesce();

    return std::make_pair((void *)Ptr, ReqSize);
  }

  std::pair<void *, uint64_t> requestExactPage(uint64_t VASize, void *VA) {
    uint64_t ReqSize = mneme::util::roundUp(VASize, PageSize);
    auto FreeNode = findInclusivePage((uintptr_t)VA, ReqSize);
    if (FreeNode == FreeVARanges.end())
      return std::make_pair(nullptr, 0);

    auto Ptr = FreeNode->PageAddr;
    auto NodePageSize = FreeNode->Size;

    FreeVARanges.erase(FreeNode);

    // We found exactly the requested page.
    if (ReqSize == NodePageSize && (uintptr_t)VA == Ptr)
      return std::make_pair((void *)Ptr, ReqSize);

    if (VA != nullptr) {
      if ((uintptr_t)VA < Ptr ||
          ((uintptr_t)VA + VASize) > (Ptr + NodePageSize)) {
        std::ostringstream oss;
        oss << "Unable to return requested address: " << std::hex
            << reinterpret_cast<uintptr_t>(VA)
            << " instead the returned address is " << std::hex
            << reinterpret_cast<uintptr_t>(Ptr) << std::dec << "\n";
        LOG_FATAL(oss.str());
      }
    }

    // There are 'unused' addresses left from the requested one
    // We add them back to the page manager
    auto NewNodePageSize = (uintptr_t)VA - Ptr;
    ContiguousAddrBlock block{Ptr, NewNodePageSize};
    FreeVARanges.insert(block);

    // There are 'unused' addresses right/higher than the end of
    // the requested page addresses
    auto NewPtr = (uintptr_t)VA + ReqSize;
    if (NewPtr < Ptr + NodePageSize) {
      ContiguousAddrBlock block{NewPtr, Ptr + NodePageSize - NewPtr};
      FreeVARanges.insert(block);
    }
    // This can be expensive. Currently we coalesce in every request that
    // modifies our free-pages.
    coalesce();

    return std::make_pair(VA, ReqSize);
  }

public:
  PageManager() = default;
  PageManager(uint64_t VASize, uint64_t PageSize, void *VA, int32_t DeviceID)
      : TotalVASize(VASize), ReservedVA(reinterpret_cast<uintptr_t>(VA)),
        PageSize(PageSize), DeviceID(DeviceID) {
    FreeVARanges.insert(ContiguousAddrBlock{ReservedVA, TotalVASize});
    MnemeDeviceRT::mmap(MemHandle, (void *)ReservedVA, VASize, DeviceID);
  }

  ~PageManager() {
    MnemeDeviceRT::unmap(MemHandle, (void *)ReservedVA, TotalVASize);
    MnemeDeviceRT::freeVirtualAddress((void *)ReservedVA, TotalVASize);
  }

  std::pair<void *, uint64_t> allocateAddr(uint64_t VASize, void *VA) {
    if (VA == nullptr)
      return reserveBestFitPage(VASize);
    return requestExactPage(VASize, VA);
  }

  void releaseAddr(uint64_t VASize, void *VA) {
    ContiguousAddrBlock AddrBlock{reinterpret_cast<uint64_t>(VA), VASize};
    FreeVARanges.insert(AddrBlock);
    coalesce();
  }
  void *getVAStart() const {
    LOG_DEBUG("Returning address {}", ReservedVA);
    return reinterpret_cast<void *>(ReservedVA);
  }
  uint64_t getTotalVASize() const { return TotalVASize; }
  uint64_t getNumPages() const { return FreeVARanges.size(); }
  uint64_t getUniqueNumPages() const {
    std::set<ContiguousAddrBlock> s(FreeVARanges.begin(), FreeVARanges.end());
    return s.size();
  }

  void dump() const {
    for (auto V : FreeVARanges) {
      LOG_INFO("Page Manager Start: {} Size {}", (void *)V.PageAddr, V.Size);
    }
  }

  typename MnemeDeviceRT::MemoryAllocationHandle_t &getMemHandle() {
    return MemHandle;
  }
};

template <mneme::DeviceVendors VendorTypes>
std::unique_ptr<PageManager<VendorTypes>>
initializePageManager(int DeviceID, void *ReqAddr = nullptr,
                      uint64_t ActualSize = -1) {
  using DT = mneme::DeviceTraits<VendorTypes>;
  auto MinPageSize = DT::getMinPageSize(DeviceID);
  if (ActualSize == -1)
    ActualSize = mneme::util::roundUp(DT::getFixedMemorySize(), MinPageSize);

  // Replay must get the recorded address; record picks one.
  std::vector<uintptr_t> Candidates;
  if (ReqAddr)
    Candidates.push_back(reinterpret_cast<uintptr_t>(ReqAddr));
  else
    Candidates = mneme::util::suggestVAddrs(ActualSize, MinPageSize);
  if (Candidates.empty())
    Candidates.push_back(0);

  void *VA = nullptr;
  for (size_t I = 0; I < Candidates.size(); I++) {
    void *Want = reinterpret_cast<void *>(Candidates[I]);
    LOG_INFO("Trying {}/{} to Reserve Virtual Address {} space of size {}...",
             I + 1, Candidates.size(), Want, ActualSize);
    VA = DT::getVirtualAddress(ActualSize, Want, MinPageSize);
    if (VA == Want || !Want || I + 1 == Candidates.size())
      break;
    LOG_INFO("... got {} instead", VA);
    DT::freeVirtualAddress(VA, ActualSize);
  }
  LOG_INFO("... Reserved Virtual Address {}", VA);
  return std::make_unique<PageManager<VendorTypes>>(ActualSize, MinPageSize, VA,
                                                    DeviceID);
}
