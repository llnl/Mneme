#pragma once

#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeRecorderBackend.hpp"
#include "mneme/MnemeSnapshot.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include <llvm/ADT/DenseMap.h>

#include <proteus/KernelMetadata.h>

namespace mneme {

template <DeviceVendors VendorTypes>
class RecordingBackend final : public RecorderBackend<VendorTypes> {
  using Base = RecorderBackend<VendorTypes>;
  using MnemeDeviceRT = typename Base::MnemeDeviceRT;
  using DeviceError_t = typename Base::DeviceError_t;
  using DeviceStream_t = typename Base::DeviceStream_t;

  RecorderRuntimeFunctions<VendorTypes> Runtime;
  RecordDatabase DB;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> AllocatedBlobs;
  std::unique_ptr<PageManager<VendorTypes>> PM;
  std::unique_ptr<SmallAllocator<VendorTypes>> Small;

  // NOTE: We only keep track of the first time we set the device id. Once we
  // create the allocator we assume that the allocations go to the same device.
  int DeviceID = -1;

  void initializePageManagerIfNeeded() {
    if (PM)
      return;

    // NOTE: We need this arch cause internally we initialize the device.
    // FIXME: We need to have a DeviceTrait function to initialize the GPU
    // and call it separately here. Let's do this on a separate PR.
    auto arch = MnemeDeviceRT::GetDeviceArch();
    LOG_DEBUG("Initializing system {}", arch);
    if (DeviceID == -1)
      Runtime.origGetDeviceID(&DeviceID);
    PM = std::make_unique<PageManager<VendorTypes>>(
        MnemeDeviceRT::getMinPageSize(DeviceID));
    Small = std::make_unique<SmallAllocator<VendorTypes>>(*PM, DeviceID);
  }

public:
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
    initializePageManagerIfNeeded();

    MnemeMemoryBlob<VendorTypes> MemBlob;
    if (Small->isSmall(size)) {
      uint64_t ActualSize = Small->actualSize(size);
      void *Addr = Small->allocate(ActualSize);
      if (Addr) {
        MemBlob = MnemeMemoryBlob<VendorTypes>(ActualSize, nullptr, size);
        MemBlob.mapInto(Addr);
      }
    } else {
      PM->mapAddr(size, [&](const auto &R) {
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
    auto ret = Runtime.origMallocManaged(ptr, size, flags);
    LOG_DEBUG("Intercepted Managed Malloc PTR:{} SIZE:{}", *ptr, size);
    LOG_WARN("Will not be able to replay Kernels acessing:{}", *ptr);
    return ret;
  }

  DeviceError_t rtHostMalloc(void **ptr, size_t size,
                             unsigned int flags) override {
    auto ret = Runtime.origMallocPinned(ptr, size, flags);
    LOG_WARN("Intercepted Pinned|Host Malloc PTR:{} SIZE:{}", *ptr, size);
    return ret;
  }

  DeviceError_t rtFree(void *ptr) override {
    if (ptr == nullptr) {
      LOG_WARN("Mneme was instructed to de-allocate nullptr..., skipping");
      return MnemeDeviceRT::DeviceSuccess;
    }
    auto It = AllocatedBlobs.find(ptr);
    if (It == AllocatedBlobs.end()) {
      LOG_CRITICAL("Free address that is not being allocated through Mneme {}",
                   ptr);
      LOG_FATAL("Free address that is not being allocated through Mneme\n");
    }
    auto &Blob = It->second;
    LOG_DEBUG("Intercepted device Free PTR:{} SIZE:{} ACTUALSIZE:{}", ptr,
              Blob.getSize(), Blob.getActualSize());
    // Like hipFree, wait for kernels that may still use the memory.
    MnemeDeviceRT::DeviceSynchronize();
    auto ret = Blob.release();
    if (Small->isSmall(Blob.getSize()))
      Small->release(ptr, Blob.getActualSize());
    else
      PM->releaseAddr(ptr, Blob.getActualSize());
    AllocatedBlobs.erase(It);
    return ret;
  }

  DeviceError_t rtHostFree(void *ptr) override {
    auto ret = Runtime.origFreeHost(ptr);
    LOG_DEBUG("Free pinned address:{}", ptr);
    return ret;
  }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) override {
    initializePageManagerIfNeeded();

    // NOTE: Here we do something conceptually different. We no longer go
    // through proteus. We call immediately the vendor launcher. Thus we avoid
    // overheads from caching etc.
    LOG_DEBUG("Received OptionalKernel Info {}",
              (void *)Runtime.origLaunchKernel);
    auto OptionalKernelInfo = proteus::runtime::captureKernelMetadata(func);
    if (!OptionalKernelInfo) {
      LOG_DEBUG("Information for kernel  {} is not included", func);
      return Runtime.origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem,
                                      Stream);
    }
    auto &KInfo = OptionalKernelInfo.value();
    LOG_DEBUG("Continue with {}", KInfo.getName());
    LOG_INFO("Hash value is {}", KInfo.getStaticHash());

    auto RecordAction = DB.takeSnapshot<VendorTypes>(
        KInfo, AllocatedBlobs, GridDim, BlockDim, Args, SharedMem, Stream);
    if (RecordAction)
      LOG_INFO("Successfully Recorded Prologue of Kernel {} NAME:{} GRID:({}, "
               "{}, {}) "
               "BLOCK:({}, {}, "
               "{}) SHM_SIZE:{}",
               func, KInfo.getName(), GridDim.x, GridDim.y, GridDim.z,
               BlockDim.x, BlockDim.y, BlockDim.z, SharedMem);
    auto ret = Runtime.origLaunchKernel(func, GridDim, BlockDim, Args,
                                        SharedMem, Stream);
    if (RecordAction) {
      (*RecordAction)(AllocatedBlobs, Args, Stream);
      LOG_INFO("Successfully Recorded Epilogue of Kernel {} NAME:{} GRID:({}, "
               "{}, {}) "
               "BLOCK:({}, {}, "
               "{}) SHM_SIZE:{}",
               func, KInfo.getName(), GridDim.x, GridDim.y, GridDim.z,
               BlockDim.x, BlockDim.y, BlockDim.z, SharedMem);
      // write kernel record incrementally after epilogue is recorded
      DB.writeKernelJSON(KInfo.getStaticHash());
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
    LOG_DEBUG("RecordingBackend destructor: releasing {} blobs", AllocatedBlobs.size());

    // Release all unreleased blobs before the DenseMap destructs
    for (auto &Entry : AllocatedBlobs) {
      auto &Blob = Entry.second;
      if (Blob.getBlobAddr() != nullptr) {
        LOG_DEBUG("Releasing blob at addr={} during recorder cleanup",
                  Blob.getBlobAddr());
        Blob.release();
      }
    }

    Small.reset();
    PM.reset();

    LOG_DEBUG("RecordingBackend destructor complete");
  }
};

} // namespace mneme
