#pragma once

#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLaunchTimer.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeRecordScope.hpp"
#include "mneme/MnemeRecorderBackend.hpp"
#include "mneme/MnemeRuntimeAllocPlacement.hpp"
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
  // Allocations served by the vendor runtime (managed, pinned, pitched,
  // stream-ordered, ...). They are not captured, but the application may
  // release them with the device free, which must then be forwarded.
  llvm::DenseSet<void *> PassthroughAllocs;
  // Managed allocations that the vendor runtime made at an address Mneme
  // chose, with the mapping the runtime made for each. They are captured like
  // Mneme's own blobs, but the runtime keeps owning and releasing their memory.
  llvm::DenseMap<void *, RuntimeMapping> RuntimeBlobs;
  std::unique_ptr<PageManager<VendorTypes>> PM;
  std::unique_ptr<DeviceChunkSource<VendorTypes>> DeviceChunks;
  std::unique_ptr<ChunkAllocator> DevicePacker;
  // Address space for the mappings behind runtime allocations; null unless
  // the runtime's allocations can be placed.
  std::unique_ptr<ReservedChunkSource<VendorTypes>> ReservedChunks;
  std::unique_ptr<ChunkAllocator> ReservedPacker;
  bool WarnedUnpacked = false;
  bool WarnedUnseenUnmap = false;

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
    if (MnemeDeviceRT::PlacesRuntimeAllocs) {
      ReservedChunks = std::make_unique<ReservedChunkSource<VendorTypes>>(*PM);
      ReservedPacker = std::make_unique<ChunkAllocator>(
          *ReservedChunks, Config::get().ChunkSize);
    }
    return *PM;
  }

  // The caller must first wait for work that may still use the memory.
  DeviceError_t releaseBlob(void *ptr) {
    auto It = AllocatedBlobs.find(ptr);
    auto &Blob = It->second;
    LOG_DEBUG("Intercepted device Free PTR:{} SIZE:{} ACTUALSIZE:{}", ptr,
              Blob.getSize(), Blob.getActualSize());
    auto ret = Blob.release();
    if (DevicePacker->packs(Blob.getSize()))
      DevicePacker->release(ptr, Blob.getActualSize());
    AllocatedBlobs.erase(It);
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

  // Runs Alloc, which makes a vendor runtime allocation of size bytes at *ptr,
  // with its mapping placed where replay can map it again, and captures it.
  template <typename AllocFn>
  DeviceError_t captureRuntimeAlloc(void **ptr, size_t size, const char *api,
                                    AllocFn Alloc) {
    getOrCreatePageManager();
    RuntimeMmapPlacement Placement{*ReservedPacker, size};
    DeviceError_t ret;
    {
      PlaceRuntimeMmap Scope(Placement);
      ret = Alloc();
    }
    if (ret != MnemeDeviceRT::DeviceSuccess)
      return ret;
    const RuntimeMapping &Placed = Placement.Placed;
    if (!Placed.contains(*ptr, size))
      LOG_FATAL("{} returned PTR:{} SIZE:{} outside the mapping Mneme placed "
                "at {} SIZE:{}; this runtime does not map the allocation with "
                "its own anonymous mmap, so Mneme cannot record it",
                api, *ptr, size, reinterpret_cast<void *>(Placed.Start),
                Placed.Size);

    MnemeMemoryBlob<VendorTypes> MemBlob(DevicePacker->actualSize(size),
                                         nullptr, size);
    MemBlob.mapInto(*ptr);
    LOG_DEBUG("Intercepted {} PTR:{} SIZE:{} MAPPING:{} MAPPINGSIZE:{}", api,
              *ptr, size, reinterpret_cast<void *>(Placed.Start), Placed.Size);
    AllocatedBlobs.insert({*ptr, std::move(MemBlob)});
    RuntimeBlobs.insert({*ptr, Placed});
    return ret;
  }

  // Frees a captured runtime allocation with Free, the runtime's own free
  // named api, and reuses its addresses once the runtime has unmapped them.
  template <typename FreeFn>
  DeviceError_t releaseRuntimeBlob(void *ptr, const char *api, FreeFn Free) {
    RuntimeMunmapRelease Release{RuntimeBlobs.lookup(ptr)};
    const RuntimeMapping &Mapping = Release.Mapping;
    DeviceError_t ret;
    {
      KeepRuntimeMunmap Scope(Release);
      ret = Free();
    }
    if (ret != MnemeDeviceRT::DeviceSuccess) {
      if (Release.IsSeen)
        LOG_FATAL("{} failed after unmapping PTR:{} MAPPING:{} SIZE:{}, so "
                  "Mneme can no longer record it",
                  api, ptr, reinterpret_cast<void *>(Mapping.Start),
                  Mapping.Size);
      return ret;
    }
    LOG_DEBUG("Forwarded free of captured runtime allocation PTR:{}", ptr);
    auto It = AllocatedBlobs.find(ptr);
    It->second.release();
    AllocatedBlobs.erase(It);
    RuntimeBlobs.erase(ptr);

    if (Release.IsSeen) {
      ReservedPacker->release(reinterpret_cast<void *>(Mapping.Start),
                              ReservedPacker->actualSize(Mapping.Size));
    } else if (!WarnedUnseenUnmap) {
      // The runtime may unmap the range later, which would destroy a new
      // allocation placed there.
      LOG_WARN("{} did not unmap the mapping at {} SIZE:{} of PTR:{} before "
               "returning; Mneme will not reuse the addresses of such "
               "allocations",
               api, reinterpret_cast<void *>(Mapping.Start), Mapping.Size, ptr);
      WarnedUnseenUnmap = true;
    }
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

    auto &Pages = getOrCreatePageManager();

    MnemeMemoryBlob<VendorTypes> MemBlob;
    if (DevicePacker->packs(size)) {
      uint64_t ActualSize = DevicePacker->actualSize(size);
      void *Addr = DevicePacker->allocate(ActualSize);
      if (Addr) {
        MemBlob = MnemeMemoryBlob<VendorTypes>(ActualSize, nullptr, size);
        MemBlob.mapInto(Addr);
      }
    } else {
      if (!WarnedUnpacked) {
        LOG_WARN("Mapping {}-byte allocation separately; raise "
                 "MNEME_CHUNK_SIZE to pack it",
                 size);
        WarnedUnpacked = true;
      }
      Pages.mapAddr(size, [&](const auto &R) {
        MemBlob = MnemeMemoryBlob<VendorTypes>(R.Size, nullptr, size);
        return MemBlob.mapFixed(R.Addr, R.Addr, R.Size, R.Align, DeviceID);
      });
    }

    *ptr = MemBlob.getBlobAddr();
    if (!*ptr)
      return MnemeDeviceRT::DeviceOutOfMemory;
    LOG_DEBUG("Intercepted Device Malloc PTR:{} SIZE:{} ACTUALSIZE:{}", *ptr,
              size, MemBlob.getActualSize());
    AllocatedBlobs.insert({*ptr, std::move(MemBlob)});
    return MnemeDeviceRT::DeviceSuccess;
  }

  DeviceError_t rtManagedMalloc(void **ptr, size_t size,
                                unsigned int flags) override {
    const char *Api = MnemeDeviceRT::getManagedMallocFnName();
    auto Alloc = [&] { return Runtime.origMallocManaged(ptr, size, flags); };
    if (MnemeDeviceRT::PlacesRuntimeAllocs && size != 0)
      return captureRuntimeAlloc(ptr, size, Api, Alloc);

    auto ret = Alloc();
    if (ret == MnemeDeviceRT::DeviceSuccess)
      trackPassthroughAlloc(*ptr, size, Api);
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
    if (RuntimeBlobs.contains(ptr))
      return releaseRuntimeBlob(ptr, MnemeDeviceRT::getDeviceFreeFnName(),
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
    if (RuntimeBlobs.contains(ptr))
      return releaseRuntimeBlob(ptr, MnemeDeviceRT::getAsyncFreeFnName(), [&] {
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
    if (!RuntimeBlobs.empty()) {
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
