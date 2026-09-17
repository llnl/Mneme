#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <proteus/KernelMetadata.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"

namespace mneme {

// The base addresses of every tracked allocation, sorted ascending.
template <DeviceVendors VendorTypes>
llvm::SmallVector<void *> allBlobKeys(
    const llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory) {
  llvm::SmallVector<void *> Keys;
  Keys.reserve(DeviceMemory.size());
  for (const auto &[Ptr, Blob] : DeviceMemory)
    Keys.push_back(Ptr);
  llvm::sort(Keys);
  return Keys;
}

// The base addresses of the tracked allocations that the kernel's pointer
// arguments point into, sorted ascending. Pointers into globals select
// nothing because globals are always captured.
template <DeviceVendors VendorTypes>
llvm::SmallVector<void *> selectReachableBlobs(
    const llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
    llvm::ArrayRef<llvm::SmallVector<size_t>> PointerOffsetsByArg, void **Args,
    const proteus::runtime::GlobalMetadataMap &GlobalVars,
    llvm::StringRef KernelName) {
  using Extent = std::pair<uintptr_t, uintptr_t>;
  std::vector<Extent> Extents;
  Extents.reserve(DeviceMemory.size());
  for (const auto &[Ptr, Blob] : DeviceMemory)
    Extents.emplace_back(reinterpret_cast<uintptr_t>(Ptr),
                         Blob.getActualSize());
  llvm::sort(Extents);

  // A one-past-the-end pointer selects the allocation it came from. If that
  // address is also the next allocation's base, both are selected.
  auto selectBlobs = [&](uintptr_t P, llvm::SmallVectorImpl<void *> &Out) {
    auto It = std::upper_bound(
        Extents.begin(), Extents.end(), P,
        [](uintptr_t Value, const Extent &E) { return Value < E.first; });
    bool Found = false;
    for (int Back = 0; Back < 2 && It != Extents.begin(); ++Back) {
      --It;
      if (P > It->first + It->second)
        break;
      Out.push_back(reinterpret_cast<void *>(It->first));
      Found = true;
    }
    return Found;
  };

  auto insideGlobal = [&](uintptr_t P) {
    for (const auto &[Name, GV] : GlobalVars) {
      auto Base = reinterpret_cast<uintptr_t>(GV.DevAddr);
      if (P >= Base && P < Base + GV.VarSize)
        return true;
    }
    return false;
  };

  llvm::SmallVector<void *> Selected;
  for (size_t ArgIndex = 0; ArgIndex < PointerOffsetsByArg.size(); ++ArgIndex) {
    for (size_t Offset : PointerOffsetsByArg[ArgIndex]) {
      uintptr_t P;
      std::memcpy(&P, static_cast<const char *>(Args[ArgIndex]) + Offset,
                  sizeof(P));
      if (P == 0)
        continue;

      if (selectBlobs(P, Selected))
        continue;

      if (insideGlobal(P))
        continue;

      // Printed directly so the warning is visible even with logging off.
      std::cerr << "[mneme] Kernel " << KernelName.str() << " argument "
                << ArgIndex << " offset " << Offset << " points to "
                << reinterpret_cast<void *>(P)
                << " which is not a tracked allocation or registered global; "
                   "it will not be captured\n";
    }
  }

  llvm::sort(Selected);
  Selected.erase(std::unique(Selected.begin(), Selected.end()), Selected.end());
  return Selected;
}

} // namespace mneme
