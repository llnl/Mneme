#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemeSnapshotRecords.hpp"
#include "mneme/MnemeUtils.hpp"
#include <proteus/KernelMetadata.h>

namespace mneme {

struct ReplayGlobalVar {
  void *HostAddr;
  void *DevAddr;
  uint64_t VarSize;
  ReplayGlobalVar(void *DevAddr, uint64_t VarSize)
      : HostAddr(new uint8_t[VarSize]), DevAddr(DevAddr), VarSize(VarSize) {}
  ReplayGlobalVar(void *HostAddr, void *DevAddr, uint64_t VarSize)
      : HostAddr(HostAddr), DevAddr(DevAddr), VarSize(VarSize) {}
  ReplayGlobalVar() = delete;
  ~ReplayGlobalVar() {
    if (HostAddr)
      delete[] static_cast<uint8_t *>(HostAddr);
  }

  ReplayGlobalVar(const ReplayGlobalVar &) = delete;
  ReplayGlobalVar &operator=(const ReplayGlobalVar &) = delete;

  ReplayGlobalVar(ReplayGlobalVar &&Other)
      : HostAddr(Other.HostAddr), DevAddr(Other.DevAddr),
        VarSize(Other.VarSize) {
    Other.HostAddr = nullptr;
  }

  ReplayGlobalVar &operator=(ReplayGlobalVar &&Other) {
    if (this != &Other) {
      this->HostAddr = Other.HostAddr;
      this->DevAddr = Other.DevAddr;
      this->VarSize = Other.VarSize;
      Other.HostAddr = nullptr;
    }
    return *this;
  }
};

// The in-memory contents of a snapshot, as produced by the snapshot readers
// and consumed by the replay memory state constructors.
template <DeviceVendors VendorTypes> struct Snapshot {
  std::shared_ptr<KernelInfo> KInfo;
  std::unordered_map<std::string, ReplayGlobalVar> GlobalVars;
  BlobsByAddress<VendorTypes> DeviceMemory;
};

inline std::pair<std::string, ReplayGlobalVar>
readGlobalVarRecord(const char *&Buffer) {
  GlobalVarHeader Header = GlobalVarHeader::read(Buffer);
  ReplayGlobalVar RGV(Header.DevAddr, Header.Size);
  std::memcpy(const_cast<void *>(RGV.HostAddr), Buffer, Header.Size);
  Buffer += Header.Size;
  LOG_DEBUG("Loaded from buffer Global, Name:{}, VarSize:{}, RecoredAddr:{}",
            Header.Name, Header.Size, Header.DevAddr);
  return std::pair<std::string, ReplayGlobalVar>(std::move(Header.Name),
                                                 std::move(RGV));
}

// Host copies of the globals a prologue captured; the diff writer's base.
using GlobalSnapshotData =
    std::unordered_map<std::string, std::vector<uint8_t>>;

enum class SnapshotKind : uint32_t { Bytes = 1, Diff = 2 };

// The only code in Mneme that knows a snapshot magic string.
struct SnapshotHeader {
  SnapshotKind Kind;
  uint32_t Version;

  static constexpr char Magic[8] = {'M', 'N', 'E', 'M', 'E', 'S', 'N', 'P'};
  static constexpr size_t Size = 16;

  // Returns the header and the number of prefix bytes the payload follows.
  static std::pair<SnapshotHeader, size_t> parse(llvm::StringRef Buffer);
  void write(llvm::raw_ostream &OS) const;

private:
  // Diff files written before the container header carried this instead.
  static constexpr char LegacyDiffMagic[] = "MNEME_DIFF_V1";
  static constexpr size_t LegacyDiffMagicSize = sizeof(LegacyDiffMagic) - 1;
};

inline std::pair<SnapshotHeader, size_t>
SnapshotHeader::parse(llvm::StringRef Buffer) {
  if (Buffer.size() >= Size &&
      Buffer.starts_with(llvm::StringRef(Magic, sizeof(Magic)))) {
    SnapshotKind Kind;
    uint32_t Version;
    std::memcpy(&Kind, Buffer.data() + sizeof(Magic), sizeof(Kind));
    std::memcpy(&Version, Buffer.data() + sizeof(Magic) + sizeof(Kind),
                sizeof(Version));
    return {SnapshotHeader{Kind, Version}, Size};
  }

  if (Buffer.starts_with(llvm::StringRef(LegacyDiffMagic, LegacyDiffMagicSize)))
    return {SnapshotHeader{SnapshotKind::Diff, 1}, LegacyDiffMagicSize};

  return {SnapshotHeader{SnapshotKind::Bytes, 0}, 0};
}

// Bytes files carry the prefix too so that they can be versioned.
inline void SnapshotHeader::write(llvm::raw_ostream &OS) const {
  util::writeBytes(OS, llvm::StringRef(Magic, sizeof(Magic)));
  util::writeScalar(OS, Kind);
  util::writeScalar(OS, Version);
}

namespace snapshot_layout {
inline constexpr SnapshotHeader BytesV0{SnapshotKind::Bytes, 0};
inline constexpr SnapshotHeader BytesV1{SnapshotKind::Bytes, 1};
inline constexpr SnapshotHeader DiffV1{SnapshotKind::Diff, 1};
inline constexpr SnapshotHeader DiffV2{SnapshotKind::Diff, 2};
} // namespace snapshot_layout

// Views into the mapped file stay valid for the lifetime of its reader.
class SnapshotCursor {
  llvm::StringRef Remaining;
  bool Diff, AnnotationList;

  template <typename F> void ranges(size_t Extent, F &&Visit) {
    size_t Count = scalar<size_t>();
    for (size_t I = 0; I < Count; ++I) {
      size_t Offset = scalar<size_t>(), Size = scalar<size_t>();
      if (Offset > Extent || Size > Extent - Offset)
        LOG_FATAL("Snapshot diff range exceeds its allocation");
      Visit(Offset, Size);
    }
  }

  Metadata metadataRecord() {
    // Validate the variable-length tag before calling the metadata decoder.
    size_t Prefix = metadata::serializedSize(Metadata{}) - sizeof(size_t);
    const char *Start = take(Prefix).data();
    take(scalar<size_t>());
    return metadata::fromBuffer(Start);
  }

  std::vector<MemoryAnnotation> annotations(size_t Size) {
    if (!AnnotationList)
      return {{{0, Size}, metadataRecord()}};

    std::vector<MemoryAnnotation> Result;
    size_t Count = scalar<size_t>();
    for (size_t I = 0; I < Count; ++I) {
      auto Offset = scalar<uint64_t>(), Extent = scalar<uint64_t>();
      Result.push_back({{Offset, Extent}, metadataRecord()});
    }

    return Result;
  }

public:
  SnapshotCursor(llvm::StringRef Payload, SnapshotHeader Header)
      : Remaining(Payload), Diff(Header.Kind == SnapshotKind::Diff),
        AnnotationList(Header.Version ==
                       (Diff ? snapshot_layout::DiffV2.Version
                             : snapshot_layout::BytesV1.Version)) {}

  llvm::StringRef take(size_t Size) {
    if (Size > Remaining.size())
      LOG_FATAL("Truncated Mneme snapshot");

    auto Bytes = Remaining.take_front(Size);
    Remaining = Remaining.drop_front(Size);
    return Bytes;
  }
  template <typename T> T scalar() {
    const char *P = take(sizeof(T)).data();
    return util::extractScalar<T>(P);
  }

  template <typename F> void global(F &&Destination) {
    auto Name = take(scalar<size_t>()).str();
    auto Size = scalar<size_t>();
    auto Addr = scalar<void *>();
    auto *Dst = Destination(GlobalVarHeader{std::move(Name), Size, Addr});

    auto Copy = [&](size_t Offset, size_t N) {
      auto Data = take(N);
      if (N)
        std::memcpy(Dst + Offset, Data.data(), N);
    };

    if (Diff)
      ranges(Size, Copy);
    else
      Copy(0, Size);
  }

  BlobHeader blobHeader() {
    const char *P = take(BlobHeader::serializedSize()).data();
    return BlobHeader::read(P);
  }

  template <typename F>
  std::vector<MemoryAnnotation> blob(size_t Size, F &&Visit) {
    auto Payload = [&](size_t Offset, size_t N) { Visit(Offset, take(N)); };
    if (Diff) {
      auto Annotations = annotations(Size);
      ranges(Size, Payload);
      return Annotations;
    }

    Payload(0, Size);
    return annotations(Size);
  }

  template <typename F> void args(F &&Visit) {
    if (!Diff) {
      size_t Count = scalar<size_t>();
      for (size_t I = 0; I < Count; ++I)
        Visit(take(scalar<size_t>()));
    }

    if (!Remaining.empty() && AnnotationList)
      LOG_FATAL("Unexpected trailing bytes in snapshot");
  }
};

// Loads a base prologue snapshot on demand for readers that need one.
template <DeviceVendors VendorTypes> class BaseSnapshotSource {
public:
  // An empty Filename means no base is available.
  explicit BaseSnapshotSource(std::string Filename = "")
      : Filename(std::move(Filename)) {}

  bool empty() const { return Filename.empty(); }

  Snapshot<VendorTypes> load(const std::string &KernelName) const;

private:
  std::string Filename;
};

// Device replay and host-side inspection use the same versioned decoder.
template <DeviceVendors VendorTypes> class SnapshotReader {
  using Blob = MnemeMemoryBlob<VendorTypes>;
  std::unique_ptr<llvm::MemoryBuffer> Buffer;
  size_t PayloadOffset;
  SnapshotHeader Header;

public:
  SnapshotReader(std::unique_ptr<llvm::MemoryBuffer> Buffer, size_t Offset,
                 SnapshotHeader Header)
      : Buffer(std::move(Buffer)), PayloadOffset(Offset), Header(Header) {}
  bool requiresBaseSnapshot() const {
    return Header.Kind == SnapshotKind::Diff;
  }

  // A diff starts with a private copy of its base. The caller owns the storage
  // policy for blob backing and consumes file payloads while File is alive.
  template <typename CreateBlob, typename LoadPayload>
  void readInto(Snapshot<VendorTypes> &Snap, CreateBlob Create,
                LoadPayload Load) const {
    SnapshotCursor In(Buffer->getBuffer().drop_front(PayloadOffset), Header);
    const bool Diff = requiresBaseSnapshot();
    
    auto Count = In.scalar<size_t>();
    if (Diff && Count != Snap.GlobalVars.size())
      LOG_FATAL("Diff global count differs from prologue");
    for (size_t I = 0; I < Count; ++I)
      In.global([&](const GlobalVarHeader &H) {
        if (Diff) {
          auto It = Snap.GlobalVars.find(H.Name);
          if (It == Snap.GlobalVars.end())
            LOG_FATAL("Diff global {} missing from prologue", H.Name);
          auto &G = It->second;
          if (G.VarSize != H.Size)
            LOG_FATAL("Diff size mismatch for global {}", H.Name);
          G.DevAddr = H.DevAddr;
          return static_cast<uint8_t *>(G.HostAddr);
        }
        auto [It, New] = Snap.GlobalVars.try_emplace(H.Name, H.DevAddr, H.Size);
        if (!New)
          LOG_FATAL("Duplicate global {}", H.Name);
        return static_cast<uint8_t *>(It->second.HostAddr);
      });

    Count = In.scalar<size_t>();
    if (Diff && Count != Snap.DeviceMemory.size())
      LOG_FATAL("Diff allocation count differs from prologue");
    for (size_t I = 0; I < Count; ++I) {
      auto H = In.blobHeader();
      auto It = Snap.DeviceMemory.find(H.DevAddr);

      if (Diff && It == Snap.DeviceMemory.end())
        LOG_FATAL("Diff allocation {} missing from prologue", H.DevAddr);
      if (!Diff && It != Snap.DeviceMemory.end())
        LOG_FATAL("Duplicate snapshot allocation");
      if (H.Size > H.ActualSize)
        LOG_FATAL("Snapshot payload exceeds its allocation");

      auto &B = Diff ? It->second : Create(H);
      if (B.getSize() != H.Size || B.getActualSize() != H.ActualSize)
        LOG_FATAL("Diff allocation size differs from prologue");

      auto Annotations =
          In.blob(H.Size, [&](size_t Offset, llvm::StringRef Data) {
            if (!Data.empty())
              Load(B, Offset, Data);
          });
      if (!B.replaceAnnotations(std::move(Annotations)))
        LOG_FATAL("Invalid allocation annotations");
    }


    In.args([&](llvm::StringRef Arg) {
      auto &K = *Snap.KInfo;
      K.KernelArgSizes.push_back(Arg.size());
      K.ArgData.emplace_back();
      const char *P = Arg.data();
      K.setArgData(P, K.ArgData.size() - 1);
    });
  }

  Snapshot<VendorTypes>
  read(const std::string &KernelName,
       const BaseSnapshotSource<VendorTypes> &Base) const {
    Snapshot<VendorTypes> Snap;
    if (requiresBaseSnapshot())
      Snap = Base.load(KernelName);
    else
      Snap.KInfo = std::make_shared<KernelInfo>(KernelName);

    readInto(
        Snap,
        [&](BlobHeader H) -> Blob & {
          return Snap.DeviceMemory
              .try_emplace(H.DevAddr, H.ActualSize, nullptr, H.Size)
              .first->second;
        },
        [](Blob &B, size_t Offset, llvm::StringRef Data) {
          std::memcpy(B.getHostData() + Offset, Data.data(), Data.size());
        });
    return Snap;
  }
};

template <DeviceVendors VendorTypes> class SnapshotFormatRegistry {
public:
  static std::unique_ptr<SnapshotReader<VendorTypes>>
  open(const std::string &Filename) {
    // Binary snapshot layouts do not need a trailing NUL. Allow a direct file
    // mapping even when the file ends exactly on a page boundary.
    auto File = llvm::MemoryBuffer::getFile(Filename, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/false);
    if (!File)
      LOG_FATAL("Opening snapshot {}: {}", Filename, File.getError().message());
    auto [Header, Offset] = SnapshotHeader::parse((*File)->getBuffer());
    for (auto Layout : {snapshot_layout::BytesV0, snapshot_layout::BytesV1,
                        snapshot_layout::DiffV1, snapshot_layout::DiffV2})
      if (Layout.Kind == Header.Kind && Layout.Version == Header.Version)
        return std::make_unique<SnapshotReader<VendorTypes>>(std::move(*File),
                                                             Offset, Header);
    LOG_FATAL(
        "Unsupported Mneme snapshot format in {}: kind {} version {}",
        Filename, static_cast<uint32_t>(Header.Kind), Header.Version);
  }
};

template <DeviceVendors VendorTypes>
Snapshot<VendorTypes>
BaseSnapshotSource<VendorTypes>::load(const std::string &KernelName) const {
  if (empty())
    LOG_FATAL("No base snapshot was provided");

  auto Reader = SnapshotFormatRegistry<VendorTypes>::open(Filename);
  if (Reader->requiresBaseSnapshot())
    LOG_FATAL(
        "Snapshot {} cannot be used as a base because it itself requires a base",
        Filename);

  return Reader->read(KernelName, BaseSnapshotSource<VendorTypes>());
}

template <DeviceVendors VendorTypes> struct SnapshotInput {
  const proteus::runtime::GlobalMetadataMap &GlobalVars;
  BlobsByAddress<VendorTypes> &DeviceMemory;
  llvm::ArrayRef<size_t> KernelArgSizes;
  void **Args;
  typename DeviceTraits<VendorTypes>::DeviceStream_t Stream;
};

// The on-disk record prefix describing a captured global variable.
inline GlobalVarHeader
globalVarHeader(const std::string &Name,
                const proteus::runtime::GlobalMetadata &GV) {
  return GlobalVarHeader{Name, GV.VarSize, const_cast<void *>(GV.DevAddr)};
}

// The global's current contents, copied off the device.
template <DeviceVendors VendorTypes>
std::vector<uint8_t>
readGlobalFromDevice(const proteus::runtime::GlobalMetadata &GV) {
  std::vector<uint8_t> HostData(GV.VarSize);
  auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
      DeviceTraits<VendorTypes>::DeviceCopy(
          HostData.data(), const_cast<void *>(GV.DevAddr), GV.VarSize,
          DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
  if (DEC)
    LOG_FATAL("Copying from device to host for global variable failed");

  return HostData;
}

// Which writer is used is a config choice, not a property of any file.
template <DeviceVendors VendorTypes> class SnapshotWriter {
public:
  virtual ~SnapshotWriter() = default;

  // Returns the basename of the written file.
  virtual std::filesystem::path
  write(const std::filesystem::path &Filename,
        const SnapshotInput<VendorTypes> &In) const = 0;

  // The exact size write() would produce. Must not mutate any diff base.
  virtual size_t measure(const SnapshotInput<VendorTypes> &In) const = 0;
};

// One on-disk format: a SnapshotHeader followed by a payload.
template <DeviceVendors VendorTypes>
class FormatWriter : public SnapshotWriter<VendorTypes> {
public:
  std::filesystem::path
  write(const std::filesystem::path &Filename,
        const SnapshotInput<VendorTypes> &In) const final {
    LOG_DEBUG("Storing mneme snapshot: {}", Filename.string());
    synchronize(In.Stream);

    std::error_code EC;
    llvm::raw_fd_ostream OS(Filename.string(), EC);
    if (EC)
      LOG_FATAL("Cannot write Mneme snapshot: " + EC.message());

    header().write(OS);
    writePayload(OS, In, /*UpdateBase=*/true);
    return Filename.filename();
  }

  size_t measure(const SnapshotInput<VendorTypes> &In) const override {
    synchronize(In.Stream);

    CountingRawOStream Counter;
    header().write(Counter);
    writePayload(Counter, In, /*UpdateBase=*/false);
    return Counter.bytesWritten();
  }

protected:
  virtual SnapshotHeader header() const = 0;

  // measure() passes UpdateBase=false so measuring never advances a diff base.
  virtual void writePayload(llvm::raw_ostream &OS,
                            const SnapshotInput<VendorTypes> &In,
                            bool UpdateBase) const = 0;

  static void
  synchronize(typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
    // Synchronize because we need a consistent GPU state. We may want to do a
    // DeviceSynchronize().
    auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
        DeviceTraits<VendorTypes>::DeviceStreamSynchronize(Stream));
    if (DEC)
      LOG_FATAL("Synchronizing stream failed before snapshot");
  }

  class CountingRawOStream : public llvm::raw_ostream {
    uint64_t Pos = 0;

    void write_impl(const char *, size_t Size) override { Pos += Size; }
    uint64_t current_pos() const override { return Pos; }

  public:
    CountingRawOStream() : llvm::raw_ostream(/*unbuffered=*/true) {}
    uint64_t bytesWritten() const { return tell(); }
  };
};

template <DeviceVendors VendorTypes>
class BytesWriter : public FormatWriter<VendorTypes> {
public:
  // CaptureGlobals, when set, receives the written globals as a diff base.
  explicit BytesWriter(
      std::shared_ptr<GlobalSnapshotData> CaptureGlobals = nullptr)
      : CaptureGlobals(std::move(CaptureGlobals)) {}

  // Computed from the layout because a counting pass would copy every blob to
  // the host just to measure it.
  size_t measure(const SnapshotInput<VendorTypes> &In) const override {
    typename FormatWriter<VendorTypes>::CountingRawOStream Counter;
    this->header().write(Counter);
    size_t Size = Counter.bytesWritten();

    Size += sizeof(size_t);
    for (const auto &[VarName, GV] : In.GlobalVars) {
      Size += globalVarHeader(VarName, GV).serializedSize();
      Size += GV.VarSize;
    }

    Size += sizeof(size_t);
    for (const auto &[Ptr, Blob] : In.DeviceMemory)
      Size += Blob.serializedSize();

    Size += sizeof(size_t);
    for (size_t ArgSize : In.KernelArgSizes) {
      Size += sizeof(size_t);
      Size += ArgSize;
    }
    return Size;
  }

protected:
  SnapshotHeader header() const override { return snapshot_layout::BytesV1; }

  void writePayload(llvm::raw_ostream &OutBC,
                    const SnapshotInput<VendorTypes> &In, bool) const override {
    const auto &GlobalVars = In.GlobalVars;
    auto &DeviceMemory = In.DeviceMemory;
    auto KernelArgSizes = In.KernelArgSizes;
    void **Args = In.Args;

    // First write Global Variables.
    size_t TotalGlobals = GlobalVars.size();
    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&TotalGlobals),
                             sizeof(size_t));

    LOG_DEBUG("Number of Globals in snapshot:{} stored at position:{}",
              TotalGlobals, OutBC.tell());

    for (const auto &[VarName, GV] : GlobalVars) {
      std::vector<uint8_t> HostData = readGlobalFromDevice<VendorTypes>(GV);

      globalVarHeader(VarName, GV).write(OutBC);
      util::writeBytes(OutBC, llvm::ArrayRef<uint8_t>(HostData));
      if (CaptureGlobals)
        (*CaptureGlobals)[VarName] = std::move(HostData);
    }

    size_t TotalBlobs = DeviceMemory.size();
    LOG_DEBUG("Number of Memory Blobs in snapshot:{} stored at position:{}",
              TotalBlobs, OutBC.tell());

    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&TotalBlobs),
                             sizeof(size_t));

    // Write the Device Memory
    for (auto &[Ptr, Blob] : DeviceMemory)
      Blob.writeWithAnnotations(OutBC);
    // Lastly write the arguments
    size_t NumArgs = KernelArgSizes.size();
    LOG_DEBUG("Number of Kernel Arguments in snapshot:{} stored at position:{}",
              NumArgs, OutBC.tell());

    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&NumArgs),
                             sizeof(NumArgs));

    for (int I = 0; I < NumArgs; I++) {
      OutBC << llvm::StringRef(
          reinterpret_cast<const char *>(&KernelArgSizes[I]), sizeof(size_t));
      OutBC << llvm::StringRef(reinterpret_cast<const char *>(Args[I]),
                               KernelArgSizes[I]);
    }
  }

private:
  std::shared_ptr<GlobalSnapshotData> CaptureGlobals;
};

template <DeviceVendors VendorTypes>
class DiffWriter : public FormatWriter<VendorTypes> {
public:
  explicit DiffWriter(std::shared_ptr<const GlobalSnapshotData> PrologueGlobals)
      : PrologueGlobals(std::move(PrologueGlobals)) {
    if (!this->PrologueGlobals)
      LOG_FATAL("A Mneme diff snapshot needs the prologue globals to diff "
                "against");
  }

protected:
  SnapshotHeader header() const override { return snapshot_layout::DiffV2; }

  void writePayload(llvm::raw_ostream &OutBC,
                    const SnapshotInput<VendorTypes> &In,
                    bool UpdateBaseData) const override {
    const auto &GlobalVars = In.GlobalVars;
    auto &DeviceMemory = In.DeviceMemory;

    size_t TotalGlobals = GlobalVars.size();
    util::writeScalar(OutBC, TotalGlobals);
    for (const auto &[VarName, GV] : GlobalVars) {
      auto BaseIt = PrologueGlobals->find(VarName);
      if (BaseIt == PrologueGlobals->end())
        LOG_FATAL("Cannot diff global missing from prologue: " + VarName);
      if (BaseIt->second.size() != GV.VarSize)
        LOG_FATAL("Cannot diff global with size mismatch: " + VarName);

      std::vector<uint8_t> Current = readGlobalFromDevice<VendorTypes>(GV);

      globalVarHeader(VarName, GV).write(OutBC);

      llvm::SmallVector<char, 0> DiffBytes;
      llvm::raw_svector_ostream DiffOS(DiffBytes);
      size_t NumRanges = writeChangedRanges(DiffOS, BaseIt->second, Current, 0);
      emitRanges(OutBC, NumRanges, DiffBytes);
    }

    size_t TotalBlobs = DeviceMemory.size();
    util::writeScalar(OutBC, TotalBlobs);
    for (auto &[Ptr, Blob] : DeviceMemory) {
      BlobHeader{Blob.getActualSize(), Blob.getSize(), Blob.getBlobAddr()}
          .write(OutBC);
      writeAnnotationsToStream(OutBC, Blob.getAnnotations());

      writeCountAndWriteChangedRanges(OutBC, Blob, UpdateBaseData);
    }
  }

private:
  static constexpr size_t DiffChunkSize = 1 << 20;

  // The reader needs the range count before the ranges themselves, so the
  // ranges are scanned into a scratch buffer first.
  static void emitRanges(llvm::raw_ostream &OS, size_t NumRanges,
                         const llvm::SmallVectorImpl<char> &DiffBytes) {
    util::writeScalar(OS, NumRanges);
    util::writeBytes(OS, llvm::StringRef(DiffBytes.data(), DiffBytes.size()));
  }

  static size_t
  writeChangedRanges(llvm::raw_ostream &OS, llvm::ArrayRef<uint8_t> Base,
                     llvm::ArrayRef<uint8_t> Current, size_t BaseOffset,
                     llvm::MutableArrayRef<uint8_t> UpdateBase = {}) {
    if (Base.size() != Current.size())
      LOG_FATAL("Cannot diff buffers with different sizes");
    if (!UpdateBase.empty() && UpdateBase.size() != Base.size())
      LOG_FATAL("Cannot update diff base with mismatched buffer size");

    // Write out the contiguous ranges that have changed between Base
    // and Current.
    size_t Count = 0;
    size_t I = 0;
    while (I < Base.size()) {
      while (I < Base.size() && Base[I] == Current[I])
        ++I;
      if (I == Base.size())
        break;

      size_t Start = I;
      while (I < Base.size() && Base[I] != Current[I])
        ++I;

      size_t Offset = BaseOffset + Start;
      size_t Len = I - Start;
      util::writeScalar(OS, Offset);
      util::writeScalar(OS, Len);
      util::writeBytes(OS, Current.slice(Start, Len));
      if (!UpdateBase.empty())
        std::memcpy(UpdateBase.data() + Start, Current.data() + Start, Len);
      ++Count;
    }
    return Count;
  }

  static void
  writeCountAndWriteChangedRanges(llvm::raw_ostream &OS,
                                  MnemeMemoryBlob<VendorTypes> &Blob,
                                  bool UpdateBaseData = true) {
    auto Size = Blob.getSize();

    // early exit
    if (Size == 0) {
      size_t NumRanges = 0;
      util::writeScalar(OS, NumRanges);
      return;
    }

    std::unique_ptr<uint8_t[]> Scratch(new uint8_t[DiffChunkSize]);
    llvm::SmallVector<char, 0> DiffBytes;
    llvm::raw_svector_ostream DiffOS(DiffBytes);
    auto *Base = Blob.getHostData();
    auto *DevAddr = static_cast<uint8_t *>(Blob.getBlobAddr());
    size_t NumRanges = 0;

    for (size_t Offset = 0; Offset < Size; Offset += DiffChunkSize) {
      size_t ChunkSize = std::min(DiffChunkSize, Size - Offset);
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              Scratch.get(), DevAddr + Offset, ChunkSize,
              DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
      if (EC)
        LOG_FATAL("Error in copying data from device when writing diff\n"
                  "Device Error Msg: " +
                  EC.value() + "\n");

      llvm::ArrayRef<uint8_t> ChunkBase(Base + Offset, ChunkSize);
      llvm::ArrayRef<uint8_t> ChunkCurrent(Scratch.get(), ChunkSize);
      if (UpdateBaseData) {
        llvm::MutableArrayRef<uint8_t> UpdateBase(Base + Offset, ChunkSize);
        NumRanges += writeChangedRanges(DiffOS, ChunkBase, ChunkCurrent, Offset,
                                        UpdateBase);
      } else {
        NumRanges +=
            writeChangedRanges(DiffOS, ChunkBase, ChunkCurrent, Offset);
      }
    }

    emitRanges(OS, NumRanges, DiffBytes);
  }

  std::shared_ptr<const GlobalSnapshotData> PrologueGlobals;
};

// Writes whichever of bytes or diff is smaller for this input.
template <DeviceVendors VendorTypes>
class BestWriter : public SnapshotWriter<VendorTypes> {
public:
  explicit BestWriter(std::shared_ptr<const GlobalSnapshotData> PrologueGlobals)
      : Diff(std::move(PrologueGlobals)) {}

  std::filesystem::path
  write(const std::filesystem::path &Filename,
        const SnapshotInput<VendorTypes> &In) const override {
    size_t DiffSize = Diff.measure(In);
    size_t BytesSize = Bytes.measure(In);

    if (DiffSize <= BytesSize)
      return Diff.write(Filename, In);

    return Bytes.write(Filename, In);
  }

  size_t measure(const SnapshotInput<VendorTypes> &In) const override {
    return std::min(Diff.measure(In), Bytes.measure(In));
  }

private:
  BytesWriter<VendorTypes> Bytes;
  DiffWriter<VendorTypes> Diff;
};

// The only place EpilogueSnapshotType is consumed.
template <DeviceVendors VendorTypes>
std::unique_ptr<SnapshotWriter<VendorTypes>>
makeEpilogueWriter(EpilogueSnapshotType Type,
                   std::shared_ptr<const GlobalSnapshotData> PrologueGlobals) {
  switch (Type) {
  case EpilogueSnapshotType::Bytes:
    return std::make_unique<BytesWriter<VendorTypes>>();
  case EpilogueSnapshotType::Diff:
    return std::make_unique<DiffWriter<VendorTypes>>(
        std::move(PrologueGlobals));
  case EpilogueSnapshotType::Best:
    return std::make_unique<BestWriter<VendorTypes>>(
        std::move(PrologueGlobals));
  }

  LOG_FATAL("Unknown Mneme epilogue snapshot type");
}

} // namespace mneme
