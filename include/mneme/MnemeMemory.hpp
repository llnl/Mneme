#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
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

  // Pieces tile a blob of Size bytes when they are sorted, contiguous, and
  // cover [0, Size) with non-empty extents, or are the single piece {0, 0} of
  // an empty blob. Each piece must start on an element boundary of its own
  // builtin type and, except for the last, end on one. The last piece may end
  // in a partial element because a blob's size need not be a multiple of its
  // element size; comparison ignores those trailing bytes.
  static bool isTiling(const std::vector<MemoryAnnotation> &Pieces,
                       uint64_t Size) {
    if (Pieces.empty())
      return false;
    uint64_t End = 0;
    for (size_t I = 0; I < Pieces.size(); ++I) {
      const auto &Range = Pieces[I].Range;
      bool IsLast = I + 1 == Pieces.size();
      if (Range.Offset != End || Range.Extent > Size - Range.Offset)
        return false;
      if (Range.Extent == 0 && !(Size == 0 && Pieces.size() == 1))
        return false;
      uint64_t Width = builtinByteWidth(Pieces[I].MD.builtin);
      if (Width == 0 || Range.Offset % Width != 0)
        return false;
      if (!IsLast && Range.Extent % Width != 0)
        return false;
      End = Range.endOffset();
    }
    return End == Size;
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
  // Gives the bytes in Range the metadata Md, cutting the pieces Range
  // overlaps so that their bytes outside Range keep their metadata. Fails,
  // leaving the annotations unchanged, if the result would not tile the blob:
  // Range is empty in a non-empty blob or extends past the blob, or the new
  // piece or a cut remainder would not be aligned to its element size.
  bool setAnnotation(ByteSpan Range, Metadata Md) {
    if (Range.Offset > Size || Range.Extent > Size - Range.Offset)
      return false;

    std::vector<MemoryAnnotation> Pieces;
    Pieces.reserve(Annotations.size() + 2);
    for (const auto &Piece : Annotations) {
      if (Piece.Range.Offset >= Range.Offset)
        break;
      uint64_t End = std::min(Piece.Range.endOffset(), Range.Offset);
      Pieces.push_back(
          {{Piece.Range.Offset, End - Piece.Range.Offset}, Piece.MD});
    }
    Pieces.push_back({Range, std::move(Md)});
    for (const auto &Piece : Annotations) {
      if (Piece.Range.endOffset() <= Range.endOffset())
        continue;
      uint64_t Start = std::max(Piece.Range.Offset, Range.endOffset());
      Pieces.push_back({{Start, Piece.Range.endOffset() - Start}, Piece.MD});
    }

    if (!isTiling(Pieces, Size))
      return false;
    Annotations = std::move(Pieces);
    return true;
  }

  // Replaces every piece with one piece that covers the blob.
  void resetAnnotations(Metadata Md = {}) {
    Annotations = {{{0, Size}, std::move(Md)}};
  }

  // Installs Pieces, typically after deserializing a snapshot. Fails, leaving
  // the annotations unchanged, if Pieces does not tile the blob.
  bool replaceAnnotations(std::vector<MemoryAnnotation> Pieces) {
    if (!isTiling(Pieces, Size))
      return false;
    Annotations = std::move(Pieces);
    return true;
  }

  // The piece that holds the byte at ByteOffset, or the last piece if
  // ByteOffset is past the end of the blob.
  const MemoryAnnotation &annotationAt(uint64_t ByteOffset) const {
    auto It =
        std::upper_bound(Annotations.begin(), Annotations.end(), ByteOffset,
                         [](uint64_t Byte, const MemoryAnnotation &Piece) {
                           return Byte < Piece.Range.Offset;
                         });
    return *std::prev(It);
  }

  const std::vector<MemoryAnnotation> &getAnnotations() const {
    return Annotations;
  }

  MnemeMemoryBlob(uint64_t ActualSize = 0, void *BlobAddr = nullptr,
                  uint64_t Size = 0)
      : ActualSize(ActualSize), BlobAddr(BlobAddr), Size(Size), Data(Size),
        IsMapped(false) {
    resetAnnotations();
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
    Blob.resetAnnotations(metadata::fromBuffer(Buffer));
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
    if (Annotations != Other.Annotations) {
      LOG_WARN("Annotations differ for blob {}", Other.getBlobAddr());
      return false;
    }
    return true;
  }

  // Compares the bytes of each piece with that piece's metadata.
  bool matchesWithinTolerance(const void *Actual, const void *Expected) const {
    const auto *ActualBytes = static_cast<const char *>(Actual);
    const auto *ExpectedBytes = static_cast<const char *>(Expected);
    for (const auto &Piece : Annotations) {
      const auto &Range = Piece.Range;
      const auto &Md = Piece.MD;
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

} // namespace mneme
