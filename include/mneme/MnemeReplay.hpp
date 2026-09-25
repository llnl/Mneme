#include <algorithm>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeUtils.hpp"
#include "mneme/MnemeVASpace.hpp"

namespace mneme {

template <DeviceVendors VendorTypes> class PrologueState;
template <DeviceVendors VendorTypes> class EpilogueState;

// Abstract base for a replay memory state. A concrete state is either a
// prologue (kernel input state) or an epilogue (expected kernel output state).
// Subclasses provide load(), which materializes the state onto the device.
template <DeviceVendors VendorTypes> class ReplayMemState {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  using DeviceModule_t = typename DeviceTraits<VendorTypes>::DeviceModule_t;

protected:
  std::shared_ptr<KernelInfo> KInfo;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> DeviceMemoryState;
  std::unordered_map<std::string, ReplayGlobalVar> GlobalVars;
  std::unique_ptr<void *[]> Args;

  explicit ReplayMemState(Snapshot<VendorTypes> SnapshotState)
      : KInfo(std::move(SnapshotState.KInfo)),
        DeviceMemoryState(std::move(SnapshotState.DeviceMemory)),
        GlobalVars(std::move(SnapshotState.GlobalVars)) {
    LOG_DEBUG("Initialized replay memory state for kernel {}",
              KInfo->getName());
    Args = copyOutArgs();
  }

  void copyToDevice() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      LOG_DEBUG("Copying {} from Address {} to device address {} {}",
                isPrologue() ? "Prologue" : "Epilogue",
                (void *)MemBlob.getHostData().get(), MemBlob.getBlobAddr(),
                MemBlob.getSize());
      auto CEC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
          MemBlob.getBlobAddr(), MemBlob.getHostData().get(), MemBlob.getSize(),
          MnemeDeviceRT::MemcpyHostToDeviceKind()));
      if (CEC)
        LOG_FATAL("Could not copy Memory Blob to device EC: " + CEC.value() +
                  "\n");
    }
  }

  void copyGlobals() {
    for (auto &[GVName, GVI] : GlobalVars) {
      LOG_DEBUG("Copying data of variable {} to device addr {} and of size {}",
                GVName, GVI.DevAddr, GVI.VarSize);
      auto CEC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceCopy(GVI.DevAddr, GVI.HostAddr, GVI.VarSize,
                                    MnemeDeviceRT::MemcpyHostToDeviceKind()));
      if (CEC)
        LOG_FATAL("Could not copy global " + GVName +
                  " to device EC: " + CEC.value() + "\n");
    }
  }

  // Distinguishes the two concrete roles for diagnostic logging.
  virtual bool isPrologue() const = 0;

private:
  std::unique_ptr<void *[]> copyOutArgs() const {
    void **Args = new void *[KInfo->getNumArgs()];
    auto ArgData = KInfo->getArgData();
    for (int I = 0; I < getNumArgs(); I++) {
      Args[I] = ArgData[I].get();
    }
    std::unique_ptr<void *[]> ArgUniquePtr{Args};
    return ArgUniquePtr;
  }

public:
  ReplayMemState() = delete;
  virtual ~ReplayMemState() { release(); }

  virtual void load() = 0;

  void reset() {
    copyToDevice();
    copyGlobals();
  }

  void release() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(MemBlob.release());
      if (EC)
        LOG_WARN("Could not release replay memory blob: {}", EC.value());
    }
    DeviceMemoryState.clear();
  }

  const llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &
  getDeviceMemory() const {
    return DeviceMemoryState;
  }

  const std::unordered_map<std::string, ReplayGlobalVar> &getGlobalVars() const {
    return GlobalVars;
  }

  // RTTI-free downcasts to a concrete role.
  virtual PrologueState<VendorTypes> *asPrologue() { return nullptr; }
  virtual EpilogueState<VendorTypes> *asEpilogue() { return nullptr; }

  void **getArgs() const { return reinterpret_cast<void **>(Args.get()); }

  uint64_t getNumArgs() const { return KInfo->getNumArgs(); }

  void initializeGlobals(DeviceModule_t VendorMod) {
    LOG_INFO("Initializing {} Globals", GlobalVars.size());
    for (auto &KV : GlobalVars) {
      auto [LoadedAddr, LoadedSize] =
          DeviceTraits<VendorTypes>::getGlobalAddrFromModule(VendorMod,
                                                             KV.first);
      if (KV.second.DevAddr != LoadedAddr) {
        LOG_WARN("Global : {} was loaded on different addresses Record:{} vs "
                 "Replay:{}",
                 KV.first, KV.second.DevAddr, LoadedAddr);
        KV.second.DevAddr = LoadedAddr;
      }

      if (KV.second.VarSize != LoadedSize)
        LOG_FATAL("Global :" + KV.first +
                  "has a different size between record and replay\n" +
                  "Record Size:" + std::to_string(KV.second.VarSize) +
                  "\nReplay Size:" + std::to_string(LoadedSize));

      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              KV.second.DevAddr, KV.second.HostAddr, KV.second.VarSize,
              DeviceTraits<VendorTypes>::MemcpyHostToDeviceKind()));
      if (EC)
        LOG_FATAL("Copying Global :" + KV.first +
                  " from host to device raised error\nEC: " + EC.value());
      LOG_INFO("Successfully loaded global variable: {}", KV.first);
    }
  }
};

// Replay state for the recorded kernel input.
template <DeviceVendors VendorTypes>
class PrologueState : public ReplayMemState<VendorTypes> {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using Blob = MnemeMemoryBlob<VendorTypes>;

  static void mapBlob(uintptr_t Addr, Blob &B, uint64_t PageSize,
                      int DeviceID) {
    uint64_t Size = util::roundUp(B.getActualSize(), PageSize);
    if (Size == 0) {
      B.mapInto(reinterpret_cast<void *>(Addr));
      return;
    }
    auto Status = B.mapFixed(reinterpret_cast<void *>(Addr), Size,
                             util::mapAlignment(Addr, Size, PageSize),
                             DeviceID);
    if (Status == MapStatus::Mapped)
      return;
    LOG_FATAL("Cannot map recorded address {} size {}: {}\n{}",
              reinterpret_cast<void *>(Addr), Size,
              Status == MapStatus::Occupied ? "occupied" : "out of memory",
              util::getMappingsIn(Addr, Addr + Size));
  }

public:
  PrologueState(const std::string &KernelName, const std::string &SnapshotFile)
      : ReplayMemState<VendorTypes>(
            BaseSnapshotSource<VendorTypes>(SnapshotFile).load(KernelName)) {}

  // Maps every blob at its recorded address. Blobs closer than a large page
  // share one mapping, owned by the lowest blob.
  void load() override {
    std::vector<std::pair<uintptr_t, Blob *>> Blobs;
    for (auto &[DevAddr, MemBlob] : this->DeviceMemoryState)
      Blobs.emplace_back(reinterpret_cast<uintptr_t>(DevAddr), &MemBlob);
    llvm::sort(Blobs, llvm::less_first());

    int DeviceID = 0;
    MnemeDeviceRT::getDevice(DeviceID);
    uint64_t PageSize = MnemeDeviceRT::getMinPageSize(DeviceID);
    auto EndOf = [&](size_t I) {
      return Blobs[I].first +
             util::roundUp(Blobs[I].second->getActualSize(), PageSize);
    };

    for (size_t I = 0, N = Blobs.size(); I < N;) {
      uintptr_t Start = Blobs[I].first;
      uintptr_t End = EndOf(I);
      size_t J = I + 1;
      for (; J < N && Blobs[J].first < End + util::LargePageSize; ++J)
        End = std::max(End, EndOf(J));

      uint64_t Size = End - Start;
      if (J - I > 1 && Size &&
          Blobs[I].second->mapFixed(reinterpret_cast<void *>(Start), Size,
                                    util::mapAlignment(Start, Size, PageSize),
                                    DeviceID) == MapStatus::Mapped) {
        for (size_t K = I + 1; K < J; ++K)
          Blobs[K].second->mapInto(reinterpret_cast<void *>(Blobs[K].first));
      } else {
        for (size_t K = I; K < J; ++K)
          mapBlob(Blobs[K].first, *Blobs[K].second, PageSize, DeviceID);
      }
      I = J;
    }

    this->copyToDevice();
  }

  PrologueState<VendorTypes> *asPrologue() override { return this; }

protected:
  bool isPrologue() const override { return true; }
};

// Replay state for the expected kernel output. Unlike the prologue, load()
// allocates fresh device memory rather than mapping to recorded addresses.
template <DeviceVendors VendorTypes>
class EpilogueState : public ReplayMemState<VendorTypes> {
public:
  explicit EpilogueState(Snapshot<VendorTypes> SnapshotState)
      : ReplayMemState<VendorTypes>(std::move(SnapshotState)) {}

  void load() override {
    for (auto &[DevAddr, MemBlob] : this->DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.allocate(MemBlob.getSize()));
      if (EC)
        LOG_FATAL("Error raised during mapping prologue memeory:" + EC.value());
    }
    this->copyToDevice();
  }

  EpilogueState<VendorTypes> *asEpilogue() override { return this; }

  // Verifies a replayed prologue against this expected-output epilogue. At call
  // time the prologue's device buffers hold the kernel's actual output.
  virtual bool matches(const PrologueState<VendorTypes> &Prologue) const {
    return compare(Prologue, /*GlobalsOnDevice=*/true);
  }

  // True if the prologue's recorded input already satisfies this epilogue, so
  // a kernel that writes nothing would verify. Must be called before any launch
  // writes to the prologue's device buffers.
  bool matchesUnlaunched(const PrologueState<VendorTypes> &Prologue) const {
    return compare(Prologue, /*GlobalsOnDevice=*/false);
  }

protected:
  bool isPrologue() const override { return false; }

private:
  // Globals reach the device only when a replay module is loaded, so before any
  // launch the prologue's input globals are read from their host copies.
  bool compare(const PrologueState<VendorTypes> &Prologue,
               bool GlobalsOnDevice) const {
    LOG_DEBUG("Comparing memory states");
    bool Correct = true;

    // Device memory blobs: both states are device-resident, so the blob
    // comparator reads both device addresses directly.
    for (auto &[DevAddr, ProBlob] : Prologue.getDeviceMemory()) {
      auto It = this->DeviceMemoryState.find(DevAddr);
      if (It == this->DeviceMemoryState.end()) {
        LOG_WARN("Cannot find {} in comparators", DevAddr);
        return false;
      }
      auto &EpiBlob = It->second;
      if (EpiBlob.getSize() != ProBlob.getSize()) {
        LOG_WARN("Sizes Differ {} vs {}", ProBlob.getSize(), EpiBlob.getSize());
        return false;
      }
      if (EpiBlob != ProBlob)
        Correct = false;
    }

    // Global variables: only the prologue's globals are device-resident; the
    // epilogue never loads globals.
    for (auto &[GVName, ProGV] : Prologue.getGlobalVars()) {
      auto It = this->GlobalVars.find(GVName);
      if (It == this->GlobalVars.end()) {
        LOG_WARN("comparing with global var {} that exists only on one of the "
                 "comparators",
                 GVName);
        Correct = false;
        continue;
      }

      auto &EpiGV = It->second;
      if (!GlobalsOnDevice) {
        if (memcmp(EpiGV.HostAddr, ProGV.HostAddr, ProGV.VarSize) != 0)
          Correct = false;
        continue;
      }

      std::unique_ptr<uint8_t[]> ProData(new uint8_t[ProGV.VarSize]);
      auto CEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              ProData.get(), ProGV.DevAddr, ProGV.VarSize,
              DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
      if (CEC)
        LOG_FATAL("Could not copy global from device EC: " + CEC.value() +
                  "\n");

      if (memcmp(EpiGV.HostAddr, ProData.get(), ProGV.VarSize) != 0)
        Correct = false;
    }

    LOG_DEBUG("Memory States {}", Correct ? "are the same" : "differ");
    return Correct;
  }
};

template <DeviceVendors VendorTypes>
std::unique_ptr<ReplayMemState<VendorTypes>>
makeReplayPrologueState(const std::string &KernelName,
                        const std::string &SnapshotFile) {
  return std::make_unique<PrologueState<VendorTypes>>(KernelName, SnapshotFile);
}

template <DeviceVendors VendorTypes>
std::unique_ptr<ReplayMemState<VendorTypes>>
makeReplayEpilogueState(const std::string &KernelName,
                        const std::string &SnapshotFile,
                        const std::string &BasePrologueFile) {
  Snapshot<VendorTypes> Snap =
      SnapshotFormatRegistry<VendorTypes>::open(SnapshotFile)
          ->read(KernelName, BaseSnapshotSource<VendorTypes>(BasePrologueFile));
  return std::make_unique<EpilogueState<VendorTypes>>(std::move(Snap));
}

} // namespace mneme
