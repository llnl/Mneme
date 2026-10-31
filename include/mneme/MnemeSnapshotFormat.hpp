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
#include <llvm/ADT/DenseMap.h>
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
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> DeviceMemory;
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

template <DeviceVendors VendorTypes> class BaseSnapshotSource;

// An opened snapshot file. Each subclass decodes one on-disk layout.
template <DeviceVendors VendorTypes> class SnapshotReader {
public:
  SnapshotReader(std::string Filename,
                 std::unique_ptr<llvm::MemoryBuffer> Buffer,
                 size_t PayloadOffset)
      : Filename(std::move(Filename)), Buffer(std::move(Buffer)),
        PayloadOffset(PayloadOffset) {}
  virtual ~SnapshotReader() = default;

  // True if read() needs a base snapshot to reconstruct the state.
  virtual bool requiresBaseSnapshot() const = 0;

  virtual Snapshot<VendorTypes>
  read(const std::string &KernelName,
       const BaseSnapshotSource<VendorTypes> &Base) const = 0;

protected:
  const char *payload() const {
    return Buffer->getBufferStart() + PayloadOffset;
  }

  std::string Filename;
  std::unique_ptr<llvm::MemoryBuffer> Buffer;
  size_t PayloadOffset;
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

// Section decoders shared by the readers. A helper decodes a section as it
// appears in every layout that calls it. When a new layout changes a section,
// add a new helper instead of changing one an older reader calls.

inline void
readGlobals(const char *&Buffer,
            std::unordered_map<std::string, ReplayGlobalVar> &GlobalVars) {
  size_t TotalGlobals = util::extractScalar<size_t>(Buffer);
  LOG_DEBUG("Snapshot contains {} Globals", TotalGlobals);
  for (size_t I = 0; I < TotalGlobals; ++I) {
    auto [Name, RGV] = readGlobalVarRecord(Buffer);
    GlobalVars.try_emplace(Name, std::move(RGV));
  }
}

inline void readArgs(const char *&Buffer, KernelInfo &KInfo) {
  size_t TotalArguments = util::extractScalar<size_t>(Buffer);
  LOG_DEBUG("Snapshot contains {} total arguments", TotalArguments);
  KInfo.KernelArgSizes.resize(TotalArguments);
  KInfo.ArgData.resize(TotalArguments);
  for (size_t A = 0; A < TotalArguments; ++A) {
    KInfo.KernelArgSizes[A] = util::extractScalar<size_t>(Buffer);
    KInfo.setArgData(Buffer, A);
  }
}

template <DeviceVendors VendorTypes>
Snapshot<VendorTypes> loadDiffBase(const BaseSnapshotSource<VendorTypes> &Base,
                                   const std::string &KernelName,
                                   const std::string &Filename) {
  if (Base.empty())
    LOG_FATAL("Mneme diff snapshot " + Filename +
              " requires a base prologue snapshot");
  return Base.load(KernelName);
}

inline void expectCount(size_t Actual, size_t Expected,
                        const std::string &Filename, const char *What) {
  if (Actual != Expected)
    LOG_FATAL("Mneme diff " + Filename + " does not match prologue " + What +
              " count");
}

// A range count followed by that many (offset, size, bytes) ranges.
inline void applyDiffRanges(const char *&Buffer,
                            llvm::MutableArrayRef<uint8_t> Target) {
  size_t NumRanges = util::extractScalar<size_t>(Buffer);
  for (size_t R = 0; R < NumRanges; ++R) {
    size_t Offset = util::extractScalar<size_t>(Buffer);
    size_t Size = util::extractScalar<size_t>(Buffer);
    if (Offset > Target.size() || Size > Target.size() - Offset)
      LOG_FATAL("Malformed Mneme diff range: offset " + std::to_string(Offset) +
                " size " + std::to_string(Size) + " exceeds target size " +
                std::to_string(Target.size()));
    std::memcpy(Target.data() + Offset, Buffer, Size);
    Buffer += Size;
  }
}

inline void
applyGlobalDiffs(const char *&Buffer,
                 std::unordered_map<std::string, ReplayGlobalVar> &GlobalVars,
                 const std::string &Filename) {
  size_t TotalGlobals = util::extractScalar<size_t>(Buffer);
  expectCount(TotalGlobals, GlobalVars.size(), Filename, "global");
  for (size_t I = 0; I < TotalGlobals; ++I) {
    GlobalVarHeader GVH = GlobalVarHeader::read(Buffer);
    auto It = GlobalVars.find(GVH.Name);
    if (It == GlobalVars.end())
      LOG_FATAL("Mneme diff references global missing from prologue: " +
                GVH.Name);
    if (It->second.VarSize != GVH.Size)
      LOG_FATAL("Mneme diff global size mismatch for: " + GVH.Name);
    It->second.DevAddr = GVH.DevAddr;
    applyDiffRanges(Buffer, llvm::MutableArrayRef<uint8_t>(
                                static_cast<uint8_t *>(It->second.HostAddr),
                                It->second.VarSize));
  }
}

// Reads a diff blob header and returns the base blob it refers to.
template <DeviceVendors VendorTypes>
MnemeMemoryBlob<VendorTypes> &readDiffBlobHeader(
    const char *&Buffer,
    llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory) {
  BlobHeader BH = BlobHeader::read(Buffer);
  auto It = DeviceMemory.find(BH.DevAddr);
  if (It == DeviceMemory.end())
    LOG_FATAL("Mneme diff references device allocation missing from "
              "prologue");
  auto &Blob = It->second;
  if (Blob.getActualSize() != BH.ActualSize || Blob.getSize() != BH.Size)
    LOG_FATAL("Mneme diff memory blob size mismatch");
  return Blob;
}

template <DeviceVendors VendorTypes>
llvm::MutableArrayRef<uint8_t> hostBytes(MnemeMemoryBlob<VendorTypes> &Blob) {
  return llvm::MutableArrayRef<uint8_t>(Blob.getHostData().get(),
                                        Blob.getSize());
}

template <DeviceVendors VendorTypes>
class BytesReaderV0 : public SnapshotReader<VendorTypes> {
public:
  using SnapshotReader<VendorTypes>::SnapshotReader;

  static constexpr SnapshotHeader Layout{SnapshotKind::Bytes, 0};

  bool requiresBaseSnapshot() const override { return false; }

  Snapshot<VendorTypes>
  read(const std::string &KernelName,
       const BaseSnapshotSource<VendorTypes> &) const override {
    Snapshot<VendorTypes> Snap;
    Snap.KInfo = std::make_shared<KernelInfo>(KernelName);
    const char *CurrentPtr = this->payload();

    readGlobals(CurrentPtr, Snap.GlobalVars);
    size_t TotalMemBlobs = util::extractScalar<size_t>(CurrentPtr);
    for (size_t I = 0; I < TotalMemBlobs; ++I)
      Snap.DeviceMemory.insert(
          MnemeMemoryBlob<VendorTypes>::fromBuffer(CurrentPtr));
    readArgs(CurrentPtr, *Snap.KInfo);
    return Snap;
  }
};

// A diff stores only changed ranges, so the diff readers reconstruct the full
// base prologue first and then overlay the diff onto it.
template <DeviceVendors VendorTypes>
class DiffReaderV1 : public SnapshotReader<VendorTypes> {
public:
  using SnapshotReader<VendorTypes>::SnapshotReader;

  static constexpr SnapshotHeader Layout{SnapshotKind::Diff, 1};

  bool requiresBaseSnapshot() const override { return true; }

  Snapshot<VendorTypes>
  read(const std::string &KernelName,
       const BaseSnapshotSource<VendorTypes> &Base) const override {
    Snapshot<VendorTypes> Snap = loadDiffBase(Base, KernelName, this->Filename);
    const char *CurrentPtr = this->payload();

    applyGlobalDiffs(CurrentPtr, Snap.GlobalVars, this->Filename);
    size_t TotalMemBlobs = util::extractScalar<size_t>(CurrentPtr);
    expectCount(TotalMemBlobs, Snap.DeviceMemory.size(), this->Filename,
                "memory blob");
    for (size_t I = 0; I < TotalMemBlobs; ++I) {
      auto &Blob = readDiffBlobHeader(CurrentPtr, Snap.DeviceMemory);
      Blob.setMetadata(metadata::fromBuffer(CurrentPtr));
      applyDiffRanges(CurrentPtr, hostBytes(Blob));
    }
    return Snap;
  }
};

// Every layout Mneme has ever written needs a row here. A reader owns its
// (kind, version) pair through its Layout member; the writer that produces
// the current layout returns the same member from header().
template <DeviceVendors VendorTypes> class SnapshotFormatRegistry {
public:
  static std::unique_ptr<SnapshotReader<VendorTypes>>
  open(const std::string &Filename) {
    LOG_DEBUG("Opening snapshot file {}", Filename);
    auto Buffer = openSnapshotFile(Filename);
    auto [Header, PayloadOffset] = SnapshotHeader::parse(Buffer->getBuffer());

    size_t Count = 0;
    const Entry *Table = table(Count);
    for (size_t I = 0; I < Count; ++I)
      if (Table[I].Kind == Header.Kind && Table[I].Version == Header.Version)
        return Table[I].Make(Filename, std::move(Buffer), PayloadOffset);

    LOG_FATAL("Unsupported Mneme snapshot format in " + Filename + ": kind " +
              std::to_string(static_cast<uint32_t>(Header.Kind)) + " version " +
              std::to_string(Header.Version));
  }

private:
  using Factory = std::unique_ptr<SnapshotReader<VendorTypes>> (*)(
      std::string, std::unique_ptr<llvm::MemoryBuffer>, size_t);

  struct Entry {
    SnapshotKind Kind;
    uint32_t Version;
    Factory Make;
  };

  template <typename ReaderT>
  static std::unique_ptr<SnapshotReader<VendorTypes>>
  make(std::string Filename, std::unique_ptr<llvm::MemoryBuffer> Buffer,
       size_t PayloadOffset) {
    return std::make_unique<ReaderT>(std::move(Filename), std::move(Buffer),
                                     PayloadOffset);
  }

  template <typename ReaderT> static constexpr Entry entry() {
    return {ReaderT::Layout.Kind, ReaderT::Layout.Version, &make<ReaderT>};
  }

  static const Entry *table(size_t &Count) {
    static const Entry Table[] = {
        entry<BytesReaderV0<VendorTypes>>(),
        entry<DiffReaderV1<VendorTypes>>(),
    };
    Count = sizeof(Table) / sizeof(Table[0]);
    return Table;
  }

  static std::unique_ptr<llvm::MemoryBuffer>
  openSnapshotFile(const std::string &Filename) {
    if (!std::filesystem::exists(Filename))
      LOG_FATAL("Mneme Snapshot file does not exist");

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrErr =
        llvm::MemoryBuffer::getFile(Filename);
    if (std::error_code EC = BufferOrErr.getError())
      LOG_FATAL("Error when opening file " + EC.message());

    return std::move(BufferOrErr.get());
  }
};

template <DeviceVendors VendorTypes>
Snapshot<VendorTypes>
BaseSnapshotSource<VendorTypes>::load(const std::string &KernelName) const {
  if (empty())
    LOG_FATAL("No base snapshot was provided");

  auto Reader = SnapshotFormatRegistry<VendorTypes>::open(Filename);
  if (Reader->requiresBaseSnapshot())
    LOG_FATAL("Snapshot " + Filename +
              " cannot be used as a base because it itself requires a base");

  return Reader->read(KernelName, BaseSnapshotSource<VendorTypes>());
}

template <DeviceVendors VendorTypes> struct SnapshotInput {
  const proteus::runtime::GlobalMetadataMap &GlobalVars;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory;
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
    for (const auto &[Ptr, Blob] : In.DeviceMemory) {
      Size += BlobHeader::serializedSize();
      Size += Blob.getSize();
      Size += metadata::serializedSize(Blob.getMetadata());
    }

    Size += sizeof(size_t);
    for (size_t ArgSize : In.KernelArgSizes) {
      Size += sizeof(size_t);
      Size += ArgSize;
    }
    return Size;
  }

protected:
  SnapshotHeader header() const override {
    return BytesReaderV0<VendorTypes>::Layout;
  }

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
      OutBC << Blob;
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
  SnapshotHeader header() const override {
    return DiffReaderV1<VendorTypes>::Layout;
  }

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
      auto MD = Blob.getMetadata();
      mneme::metadata::serialize(OutBC, MD);

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
    auto *Base = Blob.getHostData().get();
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
