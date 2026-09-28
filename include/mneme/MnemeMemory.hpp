#pragma once

#include <cstdint>
#include <cstring>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <sys/types.h>
#include <utility>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeAnnotationInternal.hpp"
#include "mneme/MnemeComparators.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSnapshotRecords.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {
template <DeviceVendors VendorTypes> class MnemeMemoryBlob {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  using MemoryAllocationHandle_t =
      typename MnemeDeviceRT::MemoryAllocationHandle_t;

protected:
  Metadata PtrMD;
  uint64_t ActualSize;
  void *BlobAddr;
  uint64_t Size;
  std::unique_ptr<uint8_t[]> HostData;
  bool IsMapped;
  // Non-zero when this blob owns the MappedSize-byte mapping at MapAddr.
  uint64_t MappedSize = 0;
  void *MapAddr = nullptr;
  MemoryAllocationHandle_t Handle{};

public:
  MnemeMemoryBlob(uint64_t ActualSize = 0, void *BlobAddr = nullptr,
                  uint64_t Size = 0)
      : ActualSize(ActualSize), BlobAddr(BlobAddr), Size(Size),
        HostData(new uint8_t[Size]), IsMapped(false) {}

  // Maps MapSize bytes of new device memory at exactly MapVA, owned by this
  // blob, and places the blob at VA inside it.
  MapStatus mapFixed(void *VA, void *MapVA, uint64_t MapSize,
                     uint64_t Alignment, int DeviceID) {
    auto Status =
        MnemeDeviceRT::mapFixed(MapVA, MapSize, Alignment, DeviceID, Handle);
    if (Status == MapStatus::Mapped) {
      BlobAddr = VA;
      IsMapped = true;
      MappedSize = MapSize;
      MapAddr = MapVA;
    }
    return Status;
  }

  // Points this blob at VA inside a mapping it does not own.
  void mapInto(void *VA) {
    BlobAddr = VA;
    IsMapped = true;
    MappedSize = 0;
  }

  DeviceError_t allocate(size_t Size) {
    int _device;
    auto ret = MnemeDeviceRT::DeviceMalloc(&(this->BlobAddr), Size);
    MnemeDeviceRT::getDevice(_device);
    LOG_DEBUG("Loading epilogue for device {}", _device);
    this->ActualSize = Size;
    this->Size = Size;
    this->IsMapped = false;
    return ret;
  }

  DeviceError_t release() {
    if (!BlobAddr)
      return MnemeDeviceRT::DeviceSuccess;

    if (!IsMapped) {
      auto ret = MnemeDeviceRT::DeviceFree(BlobAddr);
      BlobAddr = 0;
      return ret;
    }
    if (MappedSize)
      MnemeDeviceRT::unmapFixed(MapAddr, MappedSize, Handle);
    MappedSize = 0;
    BlobAddr = 0;
    return MnemeDeviceRT::DeviceSuccess;
  }

  void setHostData(std::unique_ptr<uint8_t[]> HostData) {
    this->HostData = std::move(HostData);
  }

  ~MnemeMemoryBlob() {
    if (BlobAddr != 0 && IsMapped) {
      LOG_FATAL("Destroying memory descriptor without releasing device memory at addr={}, size={}",
                (void*)BlobAddr, Size);
    }
  }

  static std::pair<void *, MnemeMemoryBlob<VendorTypes>>
  fromBuffer(const char *&Buffer) {
    BlobHeader Header = BlobHeader::read(Buffer);
    auto Blob = MnemeMemoryBlob<VendorTypes>(Header.ActualSize, 0, Header.Size);
    std::memcpy(Blob.getHostData().get(), Buffer, Header.Size);
    Buffer += Header.Size;
    LOG_DEBUG("Read memory blob at address {} SIZE: {} ActualSize:{}",
              Header.DevAddr, Header.Size, Header.ActualSize);
    Blob.PtrMD = metadata::fromBuffer(Buffer);
    return std::make_pair(Header.DevAddr, std::move(Blob));
  }

  void *ptr() { return reinterpret_cast<void *>(BlobAddr); }

  MnemeMemoryBlob(const MnemeMemoryBlob &) = delete;
  MnemeMemoryBlob &operator=(const MnemeMemoryBlob &) = delete;

  MnemeMemoryBlob &operator=(MnemeMemoryBlob &&other) noexcept {
    if (this != &other) {
      BlobAddr = other.BlobAddr;
      Size = other.Size;
      ActualSize = other.ActualSize;
      HostData = std::move(other.HostData);
      IsMapped = other.IsMapped;
      MappedSize = other.MappedSize;
      MapAddr = other.MapAddr;
      Handle = other.Handle;
      PtrMD = other.PtrMD;
      other.BlobAddr = 0;
      other.MappedSize = 0;
      other.HostData = nullptr;
    }
    return *this;
  }

  MnemeMemoryBlob(MnemeMemoryBlob &&other) noexcept
      : PtrMD(other.PtrMD), ActualSize(other.ActualSize),
        BlobAddr(other.BlobAddr), Size(other.Size),
        HostData(std::move(other.HostData)), IsMapped(other.IsMapped),
        MappedSize(other.MappedSize), MapAddr(other.MapAddr),
        Handle(other.Handle) {
    other.BlobAddr = 0;
    other.MappedSize = 0;
    other.HostData = nullptr;
  }

  template <DeviceVendors VendorTypes_>
  friend llvm::raw_ostream &
  operator<<(llvm::raw_ostream &OS, const MnemeMemoryBlob<VendorTypes_> &Blob);

  void *getBlobAddr() const { return BlobAddr; }
  uint64_t getActualSize() const { return ActualSize; }
  uint64_t getSize() const { return Size; }
  const std::unique_ptr<uint8_t[]> &getHostData() const { return HostData; }

  void setMetadata(Metadata Md) { PtrMD = Md; }

  Metadata getMetadata() const { return PtrMD; }

  bool operator==(const MnemeMemoryBlob<VendorTypes> &other) const {
    if (getSize() != other.getSize()) {
      LOG_WARN("Sizes Differ {} vs {}", getSize(), other.getSize());
      return false;
    }

    auto Md = getMetadata();
    auto Compare = compareDeviceBlobs((const char *)other.getBlobAddr(),
                                      (const char *)getBlobAddr(), getSize(),
                                      Md);

    // Comparator semantics:
    // - Norm::None  => per-element thresholding reports AnyFail/FirstBadIdx
    // - Norm::L1/L2/Linf => aggregated error is reported in Agg
    if (Md.norm == Norm::None)
      return Compare.AnyFail == 0;

    return Compare.Agg <= Md.threshold;
  }
  bool operator!=(const MnemeMemoryBlob<VendorTypes> &other) const {
    return !(*this == other);
  }
};

template <DeviceVendors VendorTypes>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const MnemeMemoryBlob<VendorTypes> &Blob) {
  // The format in the binary is the following:
  // | Var Actual Size | Var-Size | Device Address | Var Data | Metadata
  BlobHeader{Blob.ActualSize, Blob.Size, Blob.BlobAddr}.write(OS);

  LOG_DEBUG("Serializing MemoryBlob, DevAddr:{} MirroredHostAddr:{} Size:{} "
            "ActualSize:{}",
            (void *)Blob.BlobAddr,
            reinterpret_cast<void *>(Blob.getHostData().get()), Blob.getSize(),
            Blob.getActualSize());

  auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
      DeviceTraits<VendorTypes>::DeviceCopy(
          static_cast<void *>(Blob.getHostData().get()),
          reinterpret_cast<void *>(Blob.BlobAddr), Blob.getSize(),
          DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
  if (EC)
    LOG_FATAL("Error in copying data from device when serializing context on "
              "output stream\nDevice Error Msg: " +
              EC.value() + "\n");
  OS << llvm::StringRef(
      reinterpret_cast<const char *>(Blob.getHostData().get()), Blob.Size);
  auto MD = Blob.getMetadata();
  mneme::metadata::serialize(OS, MD);

  return OS;
}
} // namespace mneme
