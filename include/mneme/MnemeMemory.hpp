#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
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

// Reads what writeAnnotationsToStream() writes.
inline std::vector<MemoryAnnotation>
readAnnotationsFromBuffer(const char *&Buffer) {
  size_t NumAnnotations = util::extractScalar<size_t>(Buffer);
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

// Whether a blob gave a range the metadata, or else the rule the range breaks.
enum class AnnotationStatus {
  Ok,
  OutOfBounds,
  Empty,
  UnknownElementType,
  StartsInsideElement,
  EndsInsideElement,
};

inline const char *describeAnnotationStatus(AnnotationStatus Status) {
  switch (Status) {
  case AnnotationStatus::Ok:
    return "is valid";
  case AnnotationStatus::OutOfBounds:
    return "extends past the end of its allocation";
  case AnnotationStatus::Empty:
    return "is empty";
  case AnnotationStatus::UnknownElementType:
    return "has a builtin type with no element size";
  case AnnotationStatus::StartsInsideElement:
    return "starts inside an element of its type or of the annotation it "
           "overlaps";
  case AnnotationStatus::EndsInsideElement:
    return "ends inside an element of its type or of the annotation it "
           "overlaps";
  }
  return "has an unknown annotation status";
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
  explicit MnemeBuffer(uint64_t Size = 0,
                       MemoryLocation Where = MemoryLocation::Host)
      : Ptr(allocate(Size, Where)), Size(Size), Location(Where) {}
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
    auto Compare = compareDeviceBlobs(Actual, Expected, NumBytes, Md);

    // Comparator semantics:
    // - Norm::None  => per-element thresholding reports AnyFail/FirstBadIdx
    // - Norm::L1/L2/Linf => aggregated error is reported in Agg
    if (Md.norm == Norm::None)
      return Compare.AnyFail == 0;

    return std::isfinite(Compare.Agg) && Compare.Agg <= Md.threshold;
  }

  // Sorted subregions that cover the blob. A later annotation overrides the
  // bytes it covers, so it can split an earlier subregion in two.
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

  static bool
  areValidSubregions(const std::vector<MemoryAnnotation> &Subregions,
                     uint64_t Size) {
    if (Subregions.empty())
      return false;
    uint64_t End = 0;
    for (const auto &Subregion : Subregions) {
      const auto &Range = Subregion.Range;
      if (Range.Offset != End || Range.Extent > Size - Range.Offset)
        return false;
      // An empty blob keeps one empty subregion so that annotationAt() works.
      if (Range.Extent == 0 && !(Size == 0 && Subregions.size() == 1))
        return false;
      if (!isElementBoundary(Range.Offset, Subregion.MD) ||
          !isElementBoundary(Range.endOffset(), Subregion.MD))
        return false;
      End = Range.endOffset();
    }
    return End == Size;
  }

  static bool isElementBoundaryOfBoth(uint64_t Offset, const Metadata &Before,
                                      const Metadata &After) {
    return isElementBoundary(Offset, Before) &&
           isElementBoundary(Offset, After);
  }

  static bool isElementBoundary(uint64_t Offset, const Metadata &Md) {
    uint64_t Width = builtinByteWidth(Md.builtin);
    return Width != 0 && Offset % Width == 0;
  }

  AnnotationStatus checkAnnotation(ByteSpan Range, const Metadata &Md) const {
    uint64_t Begin = Range.Offset;
    uint64_t End = Range.endOffset();
    if (Begin > Size || Range.Extent > Size - Begin)
      return AnnotationStatus::OutOfBounds;
    if (Range.Extent == 0 && Size != 0)
      return AnnotationStatus::Empty;
    if (builtinByteWidth(Md.builtin) == 0)
      return AnnotationStatus::UnknownElementType;
    if (!isElementBoundaryOfBoth(Begin, annotationAt(Begin).MD, Md))
      return AnnotationStatus::StartsInsideElement;
    if (!isElementBoundaryOfBoth(End, Md, annotationAt(End).MD))
      return AnnotationStatus::EndsInsideElement;
    return AnnotationStatus::Ok;
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
  AnnotationStatus setAnnotation(ByteSpan Range, Metadata Md) {
    auto Status = checkAnnotation(Range, Md);
    if (Status != AnnotationStatus::Ok)
      return Status;

    uint64_t Begin = Range.Offset;
    uint64_t End = Range.endOffset();
    std::vector<MemoryAnnotation> Subregions;
    Subregions.reserve(Annotations.size() + 2);
    for (const auto &Old : Annotations) {
      if (Old.Range.Offset >= Begin)
        break;
      uint64_t OldEnd = std::min(Old.Range.endOffset(), Begin);
      Subregions.push_back(
          {{Old.Range.Offset, OldEnd - Old.Range.Offset}, Old.MD});
    }
    Subregions.push_back({Range, std::move(Md)});
    for (const auto &Old : Annotations) {
      if (Old.Range.endOffset() <= End)
        continue;
      uint64_t OldBegin = std::max(Old.Range.Offset, End);
      Subregions.push_back(
          {{OldBegin, Old.Range.endOffset() - OldBegin}, Old.MD});
    }
    Annotations = std::move(Subregions);
    return AnnotationStatus::Ok;
  }

  void resetAnnotations() { Annotations = {{{0, Size}, Metadata{}}}; }

  bool replaceAnnotations(std::vector<MemoryAnnotation> Subregions) {
    if (!areValidSubregions(Subregions, Size))
      return false;
    Annotations = std::move(Subregions);
    return true;
  }

  // Replaces the annotations with the list writeAnnotationsToStream() wrote at
  // Buffer.
  void readAnnotations(const char *&Buffer) {
    if (!replaceAnnotations(readAnnotationsFromBuffer(Buffer)))
      LOG_FATAL("Invalid annotation set serialized for blob");
  }

  void readWholeBlobMetadata(const char *&Buffer) {
    auto Status = setAnnotation({0, Size}, metadata::fromBuffer(Buffer));
    if (Status != AnnotationStatus::Ok)
      LOG_FATAL("Invalid metadata serialized for blob: annotation {}",
                describeAnnotationStatus(Status));
  }

  // Past the end of the blob, this is the last subregion.
  const MemoryAnnotation &annotationAt(uint64_t ByteOffset) const {
    auto It =
        std::upper_bound(Annotations.begin(), Annotations.end(), ByteOffset,
                         [](uint64_t Byte, const MemoryAnnotation &Subregion) {
                           return Byte < Subregion.Range.Offset;
                         });
    return *std::prev(It);
  }

  const std::vector<MemoryAnnotation> &getAnnotations() const {
    return Annotations;
  }

  MnemeMemoryBlob(uint64_t ActualSize = 0, void *BlobAddr = nullptr,
                  uint64_t Size = 0,
                  MemoryLocation Where = MemoryLocation::Host)
      : ActualSize(ActualSize), BlobAddr(BlobAddr), Size(Size),
        Data(Size, Where), IsMapped(false) {
    resetAnnotations();
  }

  // Delay backing allocation until the recorded VA ranges have been mapped.
  explicit MnemeMemoryBlob(BlobHeader H)
      : ActualSize(H.ActualSize), BlobAddr(nullptr), Size(H.Size),
        Data(0, MemoryLocation::Device), IsMapped(false) {
    resetAnnotations();
  }
  
  void allocateDeviceData() {
    Data = MnemeBuffer<VendorTypes>(Size, MemoryLocation::Device);
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
    Blob.readWholeBlobMetadata(Buffer);
    return Entry;
  }

  // Reads a blob record that carries an annotation list:
  // | ActualSize | Size | DevAddr | Data |
  // | NumAnnotations | (Offset | Extent | Metadata) * NumAnnotations |.
  static std::pair<void *, MnemeMemoryBlob<VendorTypes>>
  readWithAnnotations(const char *&Buffer) {
    auto Entry = readHeaderAndData(Buffer);
    Entry.second.readAnnotations(Buffer);
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

  // The number of bytes writeWithAnnotations() writes.
  size_t serializedSize() const {
    return BlobHeader::serializedSize() + Size +
           annotationsSerializedSize(Annotations);
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

  bool operator==(const MnemeMemoryBlob<VendorTypes> &other) const {
    if (getSize() != other.getSize()) {
      LOG_WARN("Sizes Differ {} vs {}", getSize(), other.getSize());
      return false;
    }
    requireSameAnnotations(other);
    return matchesWithinTolerance(other.getBlobAddr(), getBlobAddr());
  }

  // True if the memory of Replayed matches this blob's device-resident
  // recorded data within this blob's tolerance.
  bool recordedDataMatches(const MnemeMemoryBlob<VendorTypes> &Replayed) const {
    if (getSize() != Replayed.getSize()) {
      LOG_WARN("Sizes Differ {} vs {}", getSize(), Replayed.getSize());
      return false;
    }
    requireSameAnnotations(Replayed);
    return matchesWithinTolerance(Replayed.getBlobAddr(), getDeviceData());
  }

  bool operator!=(const MnemeMemoryBlob<VendorTypes> &other) const {
    return !(*this == other);
  }

private:
  void requireSameAnnotations(const MnemeMemoryBlob<VendorTypes> &Other) const {
    if (Annotations != Other.Annotations)
      LOG_FATAL("Blob {} has different annotations from the blob it is "
                "compared with",
                Other.getBlobAddr());
  }

  bool matchesWithinTolerance(const void *Actual, const void *Expected) const {
    const auto *ActualBytes = static_cast<const char *>(Actual);
    const auto *ExpectedBytes = static_cast<const char *>(Expected);
    for (const auto &Subregion : Annotations) {
      const auto &Range = Subregion.Range;
      const auto &Md = Subregion.MD;
      LOG_DEBUG("Comparing blob {} bytes [{}, {}) with builtin={} threshold={} "
                "threshold_kind={} norm={} tag={}",
                Actual, Range.Offset, Range.endOffset(),
                static_cast<unsigned>(Md.builtin), Md.threshold,
                static_cast<unsigned>(Md.threshold_kind),
                static_cast<unsigned>(Md.norm), Md.tag.value_or("no_tag"));
      if (!compareRange(ActualBytes + Range.Offset,
                        ExpectedBytes + Range.Offset, Range.Extent, Md))
        return false;
    }
    return true;
  }
};

// Device allocations by base address, ordered so that an interior pointer
// resolves to the allocation that contains it.
template <DeviceVendors VendorTypes>
using BlobsByAddress = std::map<void *, MnemeMemoryBlob<VendorTypes>>;

} // namespace mneme
