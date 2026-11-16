#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <sys/types.h>
#include <utility>
#include <vector>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeAnnotationInternal.hpp"
#include "mneme/MnemeComparators.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSnapshotRecords.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {
struct ByteSpan {
  uint64_t Offset = 0;
  uint64_t Extent = 0;

  uint64_t endOffset() const { return Offset + Extent; }

  bool contains(uint64_t ByteOffset) const {
    return Offset <= ByteOffset && ByteOffset < endOffset();
  }

  bool hasSameRange(uint64_t OtherOffset, uint64_t OtherExtent) const {
    return Offset == OtherOffset && Extent == OtherExtent;
  }

  bool hasSameRange(const ByteSpan &Other) const {
    return hasSameRange(Other.Offset, Other.Extent);
  }

  bool overlaps(uint64_t OtherOffset, uint64_t OtherExtent) const {
    return Offset < OtherOffset + OtherExtent && OtherOffset < endOffset();
  }

  bool overlaps(const ByteSpan &Other) const {
    return overlaps(Other.Offset, Other.Extent);
  }
};

inline bool operator==(const ByteSpan &LHS, const ByteSpan &RHS) {
  return LHS.Offset == RHS.Offset && LHS.Extent == RHS.Extent;
}

inline bool operator!=(const ByteSpan &LHS, const ByteSpan &RHS) {
  return !(LHS == RHS);
}

// Metadata attached to a byte range inside one owning blob.
struct MemoryAnnotation {
  ByteSpan Range;
  Metadata MD;
};

inline bool operator==(const MemoryAnnotation &LHS,
                       const MemoryAnnotation &RHS) {
  return LHS.Range == RHS.Range && LHS.MD == RHS.MD;
}

inline bool operator!=(const MemoryAnnotation &LHS,
                       const MemoryAnnotation &RHS) {
  return !(LHS == RHS);
}

inline std::vector<MemoryAnnotation>
readAnnotationsFromBuffer(const char *&Buffer, size_t NumAnnotations) {
  std::vector<MemoryAnnotation> LoadedAnnotations;
  LoadedAnnotations.reserve(NumAnnotations);
  for (size_t I = 0; I < NumAnnotations; ++I) {
    MemoryAnnotation Annotation;
    Annotation.Range.Offset = util::extractScalar<uint64_t>(Buffer);
    Annotation.Range.Extent = util::extractScalar<uint64_t>(Buffer);
    Annotation.MD = metadata::fromBuffer(Buffer);
    LoadedAnnotations.push_back(std::move(Annotation));
  }
  return LoadedAnnotations;
}

// Writes | NumAnnotations | (Offset | Extent | Metadata) * NumAnnotations |.
inline void
writeAnnotationsToStream(llvm::raw_ostream &OS,
                         const std::vector<MemoryAnnotation> &Annotations) {
  util::writeScalar(OS, Annotations.size());
  for (const auto &Annotation : Annotations) {
    util::writeScalar(OS, Annotation.Range.Offset);
    util::writeScalar(OS, Annotation.Range.Extent);
    metadata::serialize(OS, Annotation.MD);
  }
}

// The number of bytes writeAnnotationsToStream() writes for Annotations.
inline size_t
annotationsSerializedSize(const std::vector<MemoryAnnotation> &Annotations) {
  size_t Size = sizeof(size_t);
  for (const auto &Annotation : Annotations)
    Size += sizeof(Annotation.Range.Offset) + sizeof(Annotation.Range.Extent) +
            metadata::serializedSize(Annotation.MD);
  return Size;
}

enum class MemoryLocation { Host, Device };

// Owns a buffer of bytes that lives in either host or device memory.
template <DeviceVendors VendorTypes> class MnemeBuffer {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;

  void *Ptr = nullptr;
  uint64_t Size = 0;
  MemoryLocation Location = MemoryLocation::Host;

  static void *allocate(uint64_t Size, MemoryLocation Where) {
    if (Where == MemoryLocation::Host)
      return new uint8_t[Size];
    void *DevPtr = nullptr;
    if (!Size)
      return DevPtr;
    auto EC = MnemeDeviceRT::DeviceErrorCheck(
        MnemeDeviceRT::DeviceMalloc(&DevPtr, Size));
    if (EC)
      LOG_FATAL("Could not allocate {} bytes of device memory: {}", Size,
                EC.value());
    return DevPtr;
  }

  static void release(void *P, MemoryLocation Where) {
    if (!P)
      return;
    if (Where == MemoryLocation::Host) {
      delete[] static_cast<uint8_t *>(P);
      return;
    }
    auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(P));
    if (EC)
      LOG_WARN("Could not free the device buffer at {}: {}", P, EC.value());
  }

  void free() { release(std::exchange(Ptr, nullptr), Location); }

  // Copies the bytes into new memory at Dest and frees the old memory.
  void moveTo(MemoryLocation Dest) {
    if (Location == Dest)
      return;
    void *NewPtr = allocate(Size, Dest);
    if (Size) {
      auto Kind = Dest == MemoryLocation::Device
                      ? MnemeDeviceRT::MemcpyHostToDeviceKind()
                      : MnemeDeviceRT::MemcpyDeviceToHostKind();
      auto EC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceCopy(NewPtr, Ptr, Size, Kind));
      if (EC) {
        release(NewPtr, Dest);
        LOG_FATAL("Could not copy {} bytes to the {}: {}", Size,
                  Dest == MemoryLocation::Device ? "device" : "host",
                  EC.value());
      }
    }
    free();
    Ptr = NewPtr;
    Location = Dest;
  }

public:
  explicit MnemeBuffer(uint64_t Size = 0)
      : Ptr(new uint8_t[Size]), Size(Size) {}
  MnemeBuffer(std::unique_ptr<uint8_t[]> HostData, uint64_t Size)
      : Ptr(HostData.release()), Size(Size) {}
  ~MnemeBuffer() { free(); }

  MnemeBuffer(const MnemeBuffer &) = delete;
  MnemeBuffer &operator=(const MnemeBuffer &) = delete;
  MnemeBuffer(MnemeBuffer &&Other) noexcept
      : Ptr(std::exchange(Other.Ptr, nullptr)), Size(Other.Size),
        Location(Other.Location) {}
  MnemeBuffer &operator=(MnemeBuffer &&Other) noexcept {
    if (this != &Other) {
      free();
      Ptr = std::exchange(Other.Ptr, nullptr);
      Size = Other.Size;
      Location = Other.Location;
    }
    return *this;
  }

  MemoryLocation location() const { return Location; }

  uint8_t *hostData() const {
    if (Location != MemoryLocation::Host)
      LOG_FATAL("Reading the host data of a buffer that was moved to the "
                "device");
    return static_cast<uint8_t *>(Ptr);
  }

  void *deviceData() const {
    if (Location != MemoryLocation::Device)
      LOG_FATAL("Reading the device data of a buffer that is on the host");
    return Ptr;
  }

  void moveToDevice() { moveTo(MemoryLocation::Device); }
  void moveToHost() { moveTo(MemoryLocation::Host); }
};

template <DeviceVendors VendorTypes> class MnemeMemoryBlob {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  using MemoryAllocationHandle_t =
      typename MnemeDeviceRT::MemoryAllocationHandle_t;

protected:
  static bool compareRange(const char *Actual, const char *Expected,
                           uint64_t NumBytes, const Metadata &Md) {
    if (NumBytes == 0)
      return true;

    auto Compare = compareDeviceBlobs(Actual, Expected, NumBytes, Md);

    // Comparator semantics:
    // - Norm::None  => per-element thresholding reports AnyFail/FirstBadIdx
    // - Norm::L1/L2/Linf => aggregated error is reported in Agg
    if (Md.norm == Norm::None)
      return Compare.AnyFail == 0;

    return std::isfinite(Compare.Agg) && Compare.Agg <= Md.threshold;
  }

  std::vector<MemoryAnnotation> Annotations;
  uint64_t ActualSize;
  void *BlobAddr;
  uint64_t Size;
  // The recorded bytes, on the host until replay moves them to the device.
  MnemeBuffer<VendorTypes> Data;
  bool IsMapped;
  // Non-zero when this blob owns the MappedSize-byte mapping at MapAddr.
  uint64_t MappedSize = 0;
  void *MapAddr = nullptr;
  int DeviceID = 0;
  MemoryAllocationHandle_t Handle{};

  bool isValidAnnotationRange(const ByteSpan &Range) const {
    if (Range.Offset > Size)
      return false;
    if (Range.Extent == 0)
      return Size == 0 && Range.Offset == 0;
    return Range.Extent <= Size - Range.Offset;
  }

  bool isWholeBlobRange(const ByteSpan &Range) const {
    return Range.Offset == 0 && Range.Extent == Size;
  }

  static bool annotationSortLess(const MemoryAnnotation &LHS,
                                 const MemoryAnnotation &RHS) {
    if (LHS.Range.Offset != RHS.Range.Offset)
      return LHS.Range.Offset < RHS.Range.Offset;
    return LHS.Range.Extent < RHS.Range.Extent;
  }

  void resetDefaultAnnotation() { Annotations = {{{0, Size}, {}}}; }

  const MemoryAnnotation *findWholeBlobAnnotation() const {
    for (const auto &Annotation : Annotations) {
      if (isWholeBlobRange(Annotation.Range))
        return &Annotation;
    }
    return nullptr;
  }

  static std::vector<MemoryAnnotation>
  regionAnnotations(const std::vector<MemoryAnnotation> &Annotations,
                    uint64_t BlobSize) {
    std::vector<MemoryAnnotation> Regions;
    Regions.reserve(Annotations.size());
    for (const auto &Annotation : Annotations) {
      if (Annotation.Range.Offset == 0 && Annotation.Range.Extent == BlobSize)
        continue;
      Regions.push_back(Annotation);
    }
    return Regions;
  }

  // Reads | ActualSize | Size | DevAddr | Data |, the start of a blob record.
  static std::pair<void *, MnemeMemoryBlob<VendorTypes>>
  readHeaderAndData(const char *&Buffer) {
    BlobHeader Header = BlobHeader::read(Buffer);
    auto Blob = MnemeMemoryBlob<VendorTypes>(Header.ActualSize, 0, Header.Size);
    std::memcpy(Blob.getHostData(), Buffer, Header.Size);
    Buffer += Header.Size;
    LOG_DEBUG("Read memory blob at address {} SIZE: {} ActualSize:{}",
              Header.DevAddr, Header.Size, Header.ActualSize);
    return std::make_pair(Header.DevAddr, std::move(Blob));
  }

public:
  bool registerAnnotation(ByteSpan Range, Metadata Md) {
    if (!isValidAnnotationRange(Range))
      return false;

    for (auto &Annotation : Annotations) {
      if (Annotation.Range.hasSameRange(Range)) {
        Annotation.MD = std::move(Md);
        return true;
      }

      if (!Annotation.Range.overlaps(Range))
        continue;

      // The full-range annotation is the blob's default policy, so narrower
      // region annotations may overlap it. Overlaps between two region
      // annotations remain invalid.
      if (isWholeBlobRange(Annotation.Range) || isWholeBlobRange(Range))
        continue;

      return false;
    }

    Annotations.push_back({Range, std::move(Md)});
    std::sort(Annotations.begin(), Annotations.end(), annotationSortLess);
    return true;
  }

  MnemeMemoryBlob(uint64_t ActualSize = 0, void *BlobAddr = nullptr,
                  uint64_t Size = 0)
      : ActualSize(ActualSize), BlobAddr(BlobAddr), Size(Size), Data(Size),
        IsMapped(false) {
    resetDefaultAnnotation();
  }

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
      this->DeviceID = DeviceID;
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
    // Preserve any annotations already loaded from a snapshot. Fresh blobs get
    // their default whole-blob annotation from the constructor.
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
      MnemeDeviceRT::unmapFixed(MapAddr, MappedSize, DeviceID, Handle);
    MappedSize = 0;
    BlobAddr = 0;
    return MnemeDeviceRT::DeviceSuccess;
  }

  void setHostData(std::unique_ptr<uint8_t[]> HostData) {
    Data = MnemeBuffer<VendorTypes>(std::move(HostData), Size);
  }

  // Moves the blob's recorded bytes from host to device memory.
  void moveDataToDevice() { Data.moveToDevice(); }

  // Moves the blob's recorded bytes from device to host memory.
  void moveDataToHost() { Data.moveToHost(); }

  ~MnemeMemoryBlob() {
    if (BlobAddr != 0 && IsMapped) {
      LOG_FATAL("Destroying memory descriptor without releasing device memory at addr={}, size={}",
                (void*)BlobAddr, Size);
    }
  }

  // Reads a blob record that carries a single whole-blob metadata:
  // | ActualSize | Size | DevAddr | Data | Metadata |.
  static std::pair<void *, MnemeMemoryBlob<VendorTypes>>
  readWithMetadata(const char *&Buffer) {
    auto Entry = readHeaderAndData(Buffer);
    auto &Blob = Entry.second;
    Blob.registerAnnotation({0, Blob.getSize()}, metadata::fromBuffer(Buffer));
    return Entry;
  }

  // Reads a blob record that carries an annotation list:
  // | ActualSize | Size | DevAddr | Data |
  // | NumAnnotations | (Offset | Extent | Metadata) * NumAnnotations |.
  static std::pair<void *, MnemeMemoryBlob<VendorTypes>>
  readWithAnnotations(const char *&Buffer) {
    auto Entry = readHeaderAndData(Buffer);
    size_t NumAnnotations = util::extractScalar<size_t>(Buffer);
    auto LoadedAnnotations = readAnnotationsFromBuffer(Buffer, NumAnnotations);
    if (!Entry.second.replaceAnnotations(std::move(LoadedAnnotations)))
      LOG_FATAL("Invalid annotation set serialized for blob");
    return Entry;
  }

  // Writes the blob record that readWithAnnotations() reads, copying the data
  // from the blob's device memory.
  void writeWithAnnotations(llvm::raw_ostream &OS) const {
    BlobHeader{ActualSize, Size, BlobAddr}.write(OS);

    LOG_DEBUG("Serializing MemoryBlob, DevAddr:{} MirroredHostAddr:{} Size:{} "
              "ActualSize:{}",
              (void *)BlobAddr, reinterpret_cast<void *>(getHostData()),
              getSize(), getActualSize());

    auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
        static_cast<void *>(getHostData()), reinterpret_cast<void *>(BlobAddr),
        getSize(), MnemeDeviceRT::MemcpyDeviceToHostKind()));
    if (EC)
      LOG_FATAL("Error in copying data from device when serializing context on "
                "output stream\nDevice Error Msg: " +
                EC.value() + "\n");
    OS << llvm::StringRef(reinterpret_cast<const char *>(getHostData()), Size);
    writeAnnotationsToStream(OS, Annotations);
  }

  void *ptr() { return reinterpret_cast<void *>(BlobAddr); }

  MnemeMemoryBlob(const MnemeMemoryBlob &) = delete;
  MnemeMemoryBlob &operator=(const MnemeMemoryBlob &) = delete;

  MnemeMemoryBlob &operator=(MnemeMemoryBlob &&other) noexcept {
    if (this != &other) {
      BlobAddr = other.BlobAddr;
      Size = other.Size;
      ActualSize = other.ActualSize;
      Data = std::move(other.Data);
      IsMapped = other.IsMapped;
      MappedSize = other.MappedSize;
      MapAddr = other.MapAddr;
      Handle = other.Handle;
      Annotations = std::move(other.Annotations);
      other.BlobAddr = 0;
      other.MappedSize = 0;
    }
    return *this;
  }

  MnemeMemoryBlob(MnemeMemoryBlob &&other) noexcept
      : Annotations(std::move(other.Annotations)), ActualSize(other.ActualSize),
        BlobAddr(other.BlobAddr), Size(other.Size), Data(std::move(other.Data)),
        IsMapped(other.IsMapped), MappedSize(other.MappedSize),
        MapAddr(other.MapAddr), Handle(other.Handle) {
    other.BlobAddr = 0;
    other.MappedSize = 0;
  }

  void *getBlobAddr() const { return BlobAddr; }
  uint64_t getActualSize() const { return ActualSize; }
  uint64_t getSize() const { return Size; }
  MemoryLocation getDataLocation() const { return Data.location(); }
  uint8_t *getHostData() const { return Data.hostData(); }
  void *getDeviceData() const { return Data.deviceData(); }

  bool replaceAnnotations(std::vector<MemoryAnnotation> CandidateAnnotations) {
    if (CandidateAnnotations.empty())
      return false;

    // Install a complete blob annotation set, typically after deserializing a
    // snapshot. The set must contain exactly one full-range default annotation.
    size_t WholeBlobCount = 0;
    for (const auto &Annotation : CandidateAnnotations) {
      if (!isValidAnnotationRange(Annotation.Range))
        return false;
      if (isWholeBlobRange(Annotation.Range))
        ++WholeBlobCount;
    }
    if (WholeBlobCount != 1)
      return false;

    std::sort(CandidateAnnotations.begin(), CandidateAnnotations.end(),
              annotationSortLess);
    for (size_t I = 1; I < CandidateAnnotations.size(); ++I) {
      if (CandidateAnnotations[I - 1].Range.hasSameRange(
              CandidateAnnotations[I].Range))
        continue;
      if (CandidateAnnotations[I - 1].Range.overlaps(
              CandidateAnnotations[I].Range) &&
          !isWholeBlobRange(CandidateAnnotations[I - 1].Range) &&
          !isWholeBlobRange(CandidateAnnotations[I].Range)) {
        return false;
      }
    }

    Annotations = std::move(CandidateAnnotations);
    return true;
  }

  const std::vector<MemoryAnnotation> &getAnnotations() const {
    return Annotations;
  }

  const MemoryAnnotation *getWholeBlobAnnotation() const {
    return findWholeBlobAnnotation();
  }

  std::vector<MemoryAnnotation> getRegionAnnotations() const {
    // Region annotations are the narrower ranges, excluding the
    // full-range default annotation for the blob.
    return regionAnnotations(Annotations, Size);
  }

  bool operator==(const MnemeMemoryBlob<VendorTypes> &other) const {
    if (getSize() != other.getSize()) {
      LOG_WARN("Sizes Differ {} vs {}", getSize(), other.getSize());
      return false;
    }
    if (!hasSameAnnotations(other))
      return false;
    return matchesWithinTolerance(other.getBlobAddr(), getBlobAddr());
  }

  // True if the memory of Replayed matches this blob's device-resident
  // recorded data within this blob's tolerance.
  bool recordedDataMatches(const MnemeMemoryBlob<VendorTypes> &Replayed) const {
    if (getSize() != Replayed.getSize()) {
      LOG_WARN("Sizes Differ {} vs {}", getSize(), Replayed.getSize());
      return false;
    }
    if (!hasSameAnnotations(Replayed))
      return false;
    return matchesWithinTolerance(Replayed.getBlobAddr(), getDeviceData());
  }

  bool operator!=(const MnemeMemoryBlob<VendorTypes> &other) const {
    return !(*this == other);
  }

private:
  bool hasSameAnnotations(const MnemeMemoryBlob<VendorTypes> &Other) const {
    auto *WholeBlob = getWholeBlobAnnotation();
    auto *OtherWholeBlob = Other.getWholeBlobAnnotation();
    if (!WholeBlob || !OtherWholeBlob || WholeBlob->MD != OtherWholeBlob->MD) {
      LOG_WARN("Whole-blob metadata differs for blob {}", Other.getBlobAddr());
      return false;
    }

    auto Regions = getRegionAnnotations();
    auto OtherRegions = Other.getRegionAnnotations();
    if (Regions.size() != OtherRegions.size()) {
      LOG_WARN("Region metadata count differs for blob {}",
               Other.getBlobAddr());
      return false;
    }

    for (size_t I = 0; I < Regions.size(); ++I) {
      if (Regions[I] != OtherRegions[I]) {
        LOG_WARN("Region metadata differs for blob {} at region index {}",
                 Other.getBlobAddr(), I);
        return false;
      }
    }
    return true;
  }

  // Compares each region annotation's bytes with its own metadata and every
  // byte no region covers with the whole-blob metadata. Requires a whole-blob
  // annotation, which hasSameAnnotations() checks.
  bool matchesWithinTolerance(const void *Actual, const void *Expected) const {
    const auto *ActualBytes = static_cast<const char *>(Actual);
    const auto *ExpectedBytes = static_cast<const char *>(Expected);
    const auto &WholeMD = getWholeBlobAnnotation()->MD;
    auto Regions = getRegionAnnotations();

    if (Regions.empty()) {
      LOG_DEBUG(
          "Comparing blob {} full range [0, {}) with whole-blob metadata "
          "builtin={} threshold={} threshold_kind={} norm={} tag={}",
          Actual, getSize(), static_cast<unsigned>(WholeMD.builtin),
          WholeMD.threshold, static_cast<unsigned>(WholeMD.threshold_kind),
          static_cast<unsigned>(WholeMD.norm), WholeMD.tag.value_or("no_tag"));
      return compareRange(ActualBytes, ExpectedBytes, getSize(), WholeMD);
    }

    uint64_t CurrentOffset = 0;
    for (const auto &Region : Regions) {
      if (Region.Range.Offset > CurrentOffset) {
        LOG_DEBUG("Comparing blob {} gap [{}, {}) with whole-blob metadata "
                  "builtin={} threshold={} threshold_kind={} norm={} tag={}",
                  Actual, CurrentOffset, Region.Range.Offset,
                  static_cast<unsigned>(WholeMD.builtin), WholeMD.threshold,
                  static_cast<unsigned>(WholeMD.threshold_kind),
                  static_cast<unsigned>(WholeMD.norm),
                  WholeMD.tag.value_or("no_tag"));
      }
      if (!compareRange(ActualBytes + CurrentOffset,
                        ExpectedBytes + CurrentOffset,
                        Region.Range.Offset - CurrentOffset, WholeMD))
        return false;

      LOG_DEBUG("Comparing blob {} region [{}, {}) with region metadata "
                "builtin={} threshold={} threshold_kind={} norm={} tag={}",
                Actual, Region.Range.Offset, Region.Range.endOffset(),
                static_cast<unsigned>(Region.MD.builtin), Region.MD.threshold,
                static_cast<unsigned>(Region.MD.threshold_kind),
                static_cast<unsigned>(Region.MD.norm),
                Region.MD.tag.value_or("no_tag"));
      if (!compareRange(ActualBytes + Region.Range.Offset,
                        ExpectedBytes + Region.Range.Offset,
                        Region.Range.Extent, Region.MD))
        return false;

      CurrentOffset = Region.Range.endOffset();
    }

    if (CurrentOffset < getSize()) {
      LOG_DEBUG("Comparing blob {} tail [{}, {}) with whole-blob metadata "
                "builtin={} threshold={} threshold_kind={} norm={} tag={}",
                Actual, CurrentOffset, getSize(),
                static_cast<unsigned>(WholeMD.builtin), WholeMD.threshold,
                static_cast<unsigned>(WholeMD.threshold_kind),
                static_cast<unsigned>(WholeMD.norm),
                WholeMD.tag.value_or("no_tag"));
    }
    return compareRange(ActualBytes + CurrentOffset,
                        ExpectedBytes + CurrentOffset,
                        getSize() - CurrentOffset, WholeMD);
  }
};

} // namespace mneme
