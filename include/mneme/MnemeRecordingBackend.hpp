#pragma once

#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLaunchTimer.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeManagedPlacement.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeRecordScope.hpp"
#include "mneme/MnemeRecorderBackend.hpp"
#include "mneme/MnemeSnapshot.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>

namespace mneme {

template <DeviceVendors VendorTypes>
class RecordingBackend final : public RecorderBackend<VendorTypes> {
  using Base = RecorderBackend<VendorTypes>;
  using MnemeDeviceRT = typename Base::MnemeDeviceRT;
  using DeviceError_t = typename Base::DeviceError_t;
  using DeviceStream_t = typename Base::DeviceStream_t;
  using StartedTiming = typename LaunchTimer<VendorTypes>::StartedTiming;

  RecorderRuntimeFunctions<VendorTypes> Runtime;
  std::unique_ptr<LaunchTimer<VendorTypes>> Timer;
  RecordDatabase DB;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> AllocatedBlobs;
  // Allocations served by the vendor runtime (pinned, pitched, stream-ordered,
  // managed where Mneme cannot place it, ...). They are not captured, but the
  // application may release them with the device free, which must then be
  // forwarded.
  llvm::DenseSet<void *> PassthroughAllocs;
  // Managed allocations that the vendor runtime made at an address Mneme
  // chose, with the mapping the runtime made for each. They are captured like
  // Mneme's own blobs, but the runtime keeps owning and releasing their memory.
  llvm::DenseMap<void *, ManagedMapping> ManagedBlobs;
  std::unique_ptr<PageManager<VendorTypes>> PM;
  std::unique_ptr<DeviceChunkSource<VendorTypes>> DeviceChunks;
  std::unique_ptr<ChunkAllocator> DevicePacker;
  // Address space for the mappings behind managed allocations; null unless
  // managed allocations can be placed.
  std::unique_ptr<ReservedChunkSource<VendorTypes>> ReservedChunks;
  std::unique_ptr<ChunkAllocator> ReservedPacker;
  bool WarnedUnpacked = false;

  // The exit handler that drains the timer is registered on the first timed
  // launch, after the device runtime has initialized, so it runs before the
  // runtime's own exit handlers tear it down.
  inline static RecordingBackend *TimedBackend = nullptr;

  // NOTE: We only keep track of the first time we set the device id. Once we
  // create the allocator we assume that the allocations go to the same device.
  int DeviceID = -1;

  PageManager<VendorTypes> &getOrCreatePageManager() {
    if (PM)
      return *PM;

    // NOTE: We need this arch cause internally we initialize the device.
    // FIXME: We need to have a DeviceTrait function to initialize the GPU
    // and call it separately here. Let's do this on a separate PR.
    auto arch = MnemeDeviceRT::GetDeviceArch();
    LOG_DEBUG("Initializing system {}", arch);
    if (DeviceID == -1)
      Runtime.origGetDeviceID(&DeviceID);
    PM = std::make_unique<PageManager<VendorTypes>>();
    DeviceChunks =
        std::make_unique<DeviceChunkSource<VendorTypes>>(*PM, DeviceID);
    DevicePacker = std::make_unique<ChunkAllocator>(*DeviceChunks,
                                                    Config::get().ChunkSize);
    if (MnemeDeviceRT::PlacesManagedMemory) {
      ReservedChunks = std::make_unique<ReservedChunkSource<VendorTypes>>(*PM);
      ReservedPacker = std::make_unique<ChunkAllocator>(
          *ReservedChunks, Config::get().ChunkSize);
    }
    return *PM;
  }

  // Captures the memory at Ptr from now on. Managed is the runtime's mapping
  // when Ptr is a managed allocation.
  void addBlob(void *Ptr, uint64_t ActualSize, uint64_t Size,
               std::optional<ManagedMapping> Managed = std::nullopt) {
    MnemeMemoryBlob<VendorTypes> Blob(ActualSize, nullptr, Size);
    Blob.mapInto(Ptr);
    AllocatedBlobs.insert({Ptr, std::move(Blob)});
    if (Managed)
      ManagedBlobs.insert({Ptr, *Managed});
  }

  // Releases the blob at Ptr and stops capturing its memory.
  DeviceError_t dropBlob(void *Ptr) {
    auto It = AllocatedBlobs.find(Ptr);
    auto ret = It->second.release();
    AllocatedBlobs.erase(It);
    ManagedBlobs.erase(Ptr);
    return ret;
  }

  // The caller must first wait for work that may still use the memory.
  DeviceError_t releaseBlob(void *ptr) {
    const auto &Blob = AllocatedBlobs.find(ptr)->second;
    uint64_t ActualSize = Blob.getActualSize();
    LOG_DEBUG("Intercepted device Free PTR:{} SIZE:{} ACTUALSIZE:{}", ptr,
              Blob.getSize(), ActualSize);
    auto ret = dropBlob(ptr);
    DevicePacker->release(ptr, ActualSize);
    return ret;
  }

  void registerExitDrain() {
    if (TimedBackend)
      return;
    TimedBackend = this;
    std::atexit([] {
      if (TimedBackend)
        TimedBackend->drainTimer();
    });
  }

  // Launches after the drain are still counted but no longer timed.
  void drainTimer() {
    if (!Timer)
      return;
    Timer->drain(DB);
    Timer.reset();
  }

  DeviceError_t launchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                             void **Args, size_t SharedMem,
                             DeviceStream_t Stream) {
    return Runtime.origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem,
                                    Stream);
  }

  // Requires Timer; Launch names the record the duration is added to.
  DeviceError_t timeLaunchKernel(const void *func, dim3 &GridDim,
                                 dim3 &BlockDim, void **Args, size_t SharedMem,
                                 DeviceStream_t Stream,
                                 const LaunchHandle &Launch) {
    registerExitDrain();
    Timer->collect(DB);
    std::optional<StartedTiming> Started = Timer->begin(Launch, Stream);
    auto ret = launchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    if (Started)
      Timer->end(*Started, ret);
    return ret;
  }

  // Makes a managed allocation of size bytes at *ptr with the runtime's own
  // managed malloc, with its mapping placed where replay can map it again, and
  // captures it.
  DeviceError_t captureManagedAlloc(void **ptr, size_t size,
                                    unsigned int flags) {
    const char *api = MnemeDeviceRT::getManagedMallocFnName();
    getOrCreatePageManager();
    DeviceError_t ret;
    ManagedMapping Placed;
    {
      PlaceManagedMmap Placement(*ReservedPacker, size);
      ret = Runtime.origMallocManaged(ptr, size, flags);
      Placed = Placement.placed();
    }
    if (ret != MnemeDeviceRT::DeviceSuccess)
      return ret;
    if (!Placed.contains(*ptr, size))
      LOG_FATAL("{} returned PTR:{} SIZE:{} outside the mapping Mneme placed "
                "at {} SIZE:{}; this runtime does not map the allocation with "
                "its own anonymous mmap, so Mneme cannot record it",
                api, *ptr, size, reinterpret_cast<void *>(Placed.Start),
                Placed.Size);

    addBlob(*ptr, DevicePacker->actualSize(size), size, Placed);
    LOG_DEBUG("Intercepted {} PTR:{} SIZE:{} MAPPING:{} MAPPINGSIZE:{}", api,
              *ptr, size, reinterpret_cast<void *>(Placed.Start), Placed.Size);
    return ret;
  }

  // Frees a captured managed allocation with Free, the runtime's own free
  // named api, which must unmap its mapping before returning so that Mneme can
  // keep the addresses reserved and reuse them.
  template <typename FreeFn>
  DeviceError_t releaseManagedBlob(void *ptr, const char *api, FreeFn Free) {
    ManagedMapping Mapping = ManagedBlobs.lookup(ptr);
    DeviceError_t ret;
    bool WasUnmapped;
    {
      ReserveOnManagedMunmap Reserve(Mapping);
      ret = Free();
      WasUnmapped = Reserve.wasUnmapped();
    }
    if (ret != MnemeDeviceRT::DeviceSuccess) {
      if (WasUnmapped)
        LOG_FATAL("{} failed after unmapping PTR:{} MAPPING:{} SIZE:{}, so "
                  "Mneme can no longer record it",
                  api, ptr, reinterpret_cast<void *>(Mapping.Start),
                  Mapping.Size);
      return ret;
    }
    if (!WasUnmapped)
      LOG_FATAL("{} did not unmap the mapping at {} SIZE:{} of PTR:{} before "
                "returning, so Mneme cannot keep its addresses reserved and "
                "cannot record managed memory with this runtime",
                api, reinterpret_cast<void *>(Mapping.Start), Mapping.Size,
                ptr);
    LOG_DEBUG("Forwarded free of captured managed allocation PTR:{}", ptr);
    dropBlob(ptr);
    ReservedPacker->release(reinterpret_cast<void *>(Mapping.Start),
                            ReservedPacker->actualSize(Mapping.Size));
    return ret;
  }

public:
  RecordingBackend()
      : Timer(createLaunchTimer<VendorTypes>(Config::get().KernelTiming)),
        DB(Config::get().getDataDirectory(), Config::get().EpilogueType,
           Config::get().CopySource, Config::get().KernelTiming) {}

  bool setMetadataForPointer(const void *ptr, Metadata md) override {
    if (!ptr)
      return false;

    auto It = AllocatedBlobs.find(const_cast<void *>(ptr));
    if (It == AllocatedBlobs.end())
      return false;

    It->second.setMetadata(std::move(md));
    return true;
  }

  bool getMetadataForPointer(const void *ptr, Metadata &md) const override {
    if (!ptr)
      return false;

    auto It = AllocatedBlobs.find(const_cast<void *>(ptr));
    if (It == AllocatedBlobs.end())
      return false;

    md = It->second.getMetadata();
    return true;
  }

  bool eraseMetadataForPointer(const void *ptr) override {
    if (!ptr)
      return false;

    auto It = AllocatedBlobs.find(const_cast<void *>(ptr));
    if (It == AllocatedBlobs.end())
      return false;

    It->second.setMetadata(Metadata{});
    return true;
  }

  DeviceError_t rtMalloc(void **ptr, size_t size) override {
    // The runtime still initializes the device for an empty allocation.
    if (size == 0) {
      auto ret = Runtime.origMallocDevice(ptr, 0);
      if (ret == MnemeDeviceRT::DeviceSuccess)
        trackPassthroughAlloc(*ptr, 0, MnemeDeviceRT::getDeviceMallocFnName());
      return ret;
    }

    getOrCreatePageManager();

    uint64_t ActualSize = DevicePacker->actualSize(size);
    if (!DevicePacker->packs(ActualSize) && !WarnedUnpacked) {
      LOG_WARN("Mapping {}-byte allocation separately; raise "
               "MNEME_CHUNK_SIZE to pack it",
               size);
      WarnedUnpacked = true;
    }
    *ptr = DevicePacker->allocate(ActualSize);
    if (!*ptr)
      return MnemeDeviceRT::DeviceOutOfMemory;
    addBlob(*ptr, ActualSize, size);
    LOG_DEBUG("Intercepted Device Malloc PTR:{} SIZE:{} ACTUALSIZE:{}", *ptr,
              size, ActualSize);
    return MnemeDeviceRT::DeviceSuccess;
  }

  DeviceError_t rtManagedMalloc(void **ptr, size_t size,
                                unsigned int flags) override {
    if (MnemeDeviceRT::PlacesManagedMemory && size != 0)
      return captureManagedAlloc(ptr, size, flags);

    auto ret = Runtime.origMallocManaged(ptr, size, flags);
    if (ret == MnemeDeviceRT::DeviceSuccess)
      trackPassthroughAlloc(*ptr, size,
                            MnemeDeviceRT::getManagedMallocFnName());
    return ret;
  }

  DeviceError_t rtHostMalloc(void **ptr, size_t size,
                             unsigned int flags) override {
    auto ret = Runtime.origMallocPinned(ptr, size, flags);
    if (ret == MnemeDeviceRT::DeviceSuccess)
      trackPassthroughAlloc(*ptr, size, MnemeDeviceRT::getPinnedMallocFnName());
    return ret;
  }

  void trackPassthroughAlloc(void *ptr, size_t size, const char *api) override {
    if (!ptr)
      return;
    PassthroughAllocs.insert(ptr);
    LOG_DEBUG("Intercepted {} PTR:{} SIZE:{}; will not be able to replay "
              "Kernels accessing it",
              api, ptr, size);
  }

  DeviceError_t rtFree(void *ptr) override {
    if (ptr == nullptr) {
      LOG_WARN("Mneme was instructed to de-allocate nullptr..., skipping");
      return MnemeDeviceRT::DeviceSuccess;
    }
    if (ManagedBlobs.contains(ptr))
      return releaseManagedBlob(ptr, MnemeDeviceRT::getDeviceFreeFnName(),
                                [&] { return Runtime.origFreeDevice(ptr); });
    if (!AllocatedBlobs.contains(ptr)) {
      if (PassthroughAllocs.erase(ptr)) {
        LOG_DEBUG("Forwarding free of vendor runtime allocation PTR:{}", ptr);
        return Runtime.origFreeDevice(ptr);
      }
      LOG_CRITICAL("Free address that is not being allocated through Mneme {}",
                   ptr);
      LOG_FATAL("Free address that is not being allocated through Mneme\n");
    }
    // Like hipFree, wait for kernels that may still use the memory, and report
    // their errors.
    auto ret = MnemeDeviceRT::DeviceSynchronize();
    if (ret != MnemeDeviceRT::DeviceSuccess) {
      LOG_WARN("Not freeing PTR:{}: device synchronize failed", ptr);
      return ret;
    }
    return releaseBlob(ptr);
  }

  DeviceError_t rtFreeAsync(void *ptr, DeviceStream_t stream) override {
    if (ManagedBlobs.contains(ptr))
      return releaseManagedBlob(ptr, MnemeDeviceRT::getAsyncFreeFnName(), [&] {
        return Runtime.origFreeAsync(ptr, stream);
      });
    if (AllocatedBlobs.contains(ptr)) {
      // Mneme releases its memory right away, so first wait for the work
      // queued on the stream that may still use it. The sync fails during
      // stream capture; keep the memory then.
      auto ret = MnemeDeviceRT::DeviceStreamSynchronize(stream);
      if (ret != MnemeDeviceRT::DeviceSuccess) {
        LOG_WARN("Not freeing PTR:{}: stream synchronize failed", ptr);
        return ret;
      }
      return releaseBlob(ptr);
    }
    if (ptr == nullptr || PassthroughAllocs.erase(ptr))
      LOG_DEBUG("Forwarding async free of vendor runtime allocation PTR:{}",
                ptr);
    else
      // Allocated by a function Mneme does not hook; let the runtime decide.
      LOG_WARN("Forwarding async free of unknown address PTR:{}", ptr);
    return Runtime.origFreeAsync(ptr, stream);
  }

  DeviceError_t rtHostFree(void *ptr) override {
    if (ManagedBlobs.contains(ptr))
      return releaseManagedBlob(ptr, MnemeDeviceRT::getPinnedFreeFnName(),
                                [&] { return Runtime.origFreeHost(ptr); });
    PassthroughAllocs.erase(ptr);
    auto ret = Runtime.origFreeHost(ptr);
    LOG_DEBUG("Free pinned address:{}", ptr);
    return ret;
  }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) override {
    getOrCreatePageManager();

    auto Launch = DB.registerLaunch(func, GridDim, BlockDim, SharedMem);
    if (!Launch) {
      LOG_DEBUG("Kernel {} has no record", func);
      return launchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    }
    const std::string &KernelName = DB.kernelName(Launch->StaticHash);

    bool InRecordingScope =
        Config::get().RecordingFilterMode != FilterMode::Scoped ||
        detail::scopeAllowsRecording(func);
    auto CaptureEpilogue =
        InRecordingScope ? DB.takeSnapshot<VendorTypes>(
                               func, *Launch, AllocatedBlobs, Args, Stream)
                         : std::nullopt;
    if (CaptureEpilogue)
      LOG_INFO("Successfully Recorded Prologue of Kernel {} NAME:{} GRID:({}, "
               "{}, {}) "
               "BLOCK:({}, {}, "
               "{}) SHM_SIZE:{}",
               func, KernelName, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
               BlockDim.y, BlockDim.z, SharedMem);
    auto ret =
        Timer ? timeLaunchKernel(func, GridDim, BlockDim, Args, SharedMem,
                                 Stream, *Launch)
              : launchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    if (CaptureEpilogue) {
      (*CaptureEpilogue)();
      LOG_INFO("Successfully Recorded Epilogue of Kernel {} NAME:{} GRID:({}, "
               "{}, {}) "
               "BLOCK:({}, {}, "
               "{}) SHM_SIZE:{}",
               func, KernelName, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
               BlockDim.y, BlockDim.z, SharedMem);
      // write kernel record incrementally after epilogue is recorded
      DB.writeKernelJSON(Launch->StaticHash);
    }
    return ret;
  }

  DeviceError_t rtSetDevice(int deviceID) override {
    auto ret = Runtime.origSetDeviceID(deviceID);
    if (DeviceID == -1) {
      DeviceID = deviceID;
    } else if (PM == nullptr) {
      DeviceID = deviceID;
    } else if (deviceID == DeviceID) {
      // no change; ignore
    } else if (DeviceID != -1 && PM != nullptr) {
      LOG_CRITICAL("Setting Device ID although it already "
                   "set and memory is "
                   "allocated");
    }
    return ret;
  }

  DeviceError_t rtGetDevice(int *deviceID) override {
    return Runtime.origGetDeviceID(deviceID);
  }

  ~RecordingBackend() {
    if (TimedBackend == this)
      TimedBackend = nullptr;
    DB.flush();

    LOG_DEBUG("RecordingBackend destructor: releasing {} blobs",
              AllocatedBlobs.size());

    // Release all unreleased blobs before the DenseMap destructs
    for (auto &Entry : AllocatedBlobs) {
      auto &Blob = Entry.second;
      if (Blob.getBlobAddr() != nullptr) {
        LOG_DEBUG("Releasing blob at addr={} during recorder cleanup",
                  Blob.getBlobAddr());
        Blob.release();
      }
    }

    DevicePacker.reset();
    DeviceChunks.reset();
    // The runtime still owns these allocations and may touch or unmap them
    // after Mneme is gone, so their reserved chunks must stay mapped.
    if (!ManagedBlobs.empty()) {
      (void)ReservedPacker.release();
      (void)ReservedChunks.release();
    }
    ReservedPacker.reset();
    ReservedChunks.reset();
    PM.reset();

    LOG_DEBUG("RecordingBackend destructor complete");
  }
};

} // namespace mneme
