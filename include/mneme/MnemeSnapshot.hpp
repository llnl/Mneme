#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MD5.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <string>
#include <sys/types.h>
#include <type_traits>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemeRecordingPolicy.hpp"
#include "mneme/MnemeSnapshotFormat.hpp"
#include "mneme/MnemeSnapshotRecords.hpp"
#include "mneme/MnemeUtils.hpp"
#include <proteus/KernelMetadata.h>

namespace mneme {

// Identifies a counted launch by its kernel, its launch configuration when that
// has an instance entry, and its position among the kernel's launches.
struct LaunchHandle {
  uint64_t StaticHash;
  std::optional<uint64_t> DynamicHash;
  uint64_t LaunchIndex;
};

// Durations of an instance's timed launches, written as its TimeNs.
struct LaunchTimes {
  uint64_t Sum;
  uint64_t Min;
  uint64_t Max;

  void add(uint64_t Ns) {
    Sum += Ns;
    Min = std::min(Min, Ns);
    Max = std::max(Max, Ns);
  }
  llvm::json::Object toJSON() const {
    return llvm::json::Object{{"Sum", Sum}, {"Min", Min}, {"Max", Max}};
  }
};

struct KernelInstance {
  std::string PrologueFn;
  std::string EpilogueFn;
  dim3 BlockDim;
  dim3 GridDim;
  llvm::SmallVector<double> ArgValues;
  uint64_t NumOccurrences = 0;
  uint64_t SharedMem;
  uint64_t TimedLaunches = 0;
  std::optional<LaunchTimes> TimeNs;
  std::optional<uint64_t> RecordedLaunchIndex;
  std::optional<uint64_t> RecordedLaunchTimeNs;
  static llvm::json::Object toJSON(const dim3 &Dim) {
    llvm::json::Object JSONDim;
    JSONDim["x"] = Dim.x;
    JSONDim["y"] = Dim.y;
    JSONDim["z"] = Dim.z;
    return JSONDim;
  }
  bool isRecorded() const { return RecordedLaunchIndex.has_value(); }
  void addLaunchTime(uint64_t Ns, uint64_t LaunchIndex) {
    if (TimeNs)
      TimeNs->add(Ns);
    else
      TimeNs = LaunchTimes{Ns, Ns, Ns};
    TimedLaunches++;
    if (RecordedLaunchIndex == LaunchIndex)
      RecordedLaunchTimeNs = Ns;
  }
  llvm::json::Object toJSON(bool KernelTiming) const {
    llvm::json::Object instance;
    instance["Prologue"] = PrologueFn;
    instance["Epilogue"] = EpilogueFn;
    instance["BlockDims"] = KernelInstance::toJSON(BlockDim);
    instance["GridDims"] = KernelInstance::toJSON(GridDim);
    instance["SharedMem"] = SharedMem;
    instance["Args"] = llvm::json::Array(ArgValues);
    instance["Occurrences"] = NumOccurrences;
    if (KernelTiming) {
      instance["TimedLaunches"] = TimedLaunches;
      if (TimeNs)
        instance["TimeNs"] = TimeNs->toJSON();
      if (RecordedLaunchTimeNs)
        instance["RecordedLaunchTimeNs"] = *RecordedLaunchTimeNs;
    }
    return instance;
  }
  KernelInstance(const dim3 &GridDim, const dim3 &BlockDim, uint64_t SharedMem)
      : BlockDim(BlockDim), GridDim(GridDim), SharedMem(SharedMem) {}
  KernelInstance() = default;
};

// Locates a recorded kernel's source from its line-table debug info, which
// gives the file, the line range that generated code, and the MD5 checksum the
// file had at compile time.
class SourceFileInfo {
  std::string Path;
  std::string CopyName;
  std::string MD5;
  unsigned Line = 0;
  unsigned EndLine = 0;
  std::string CompileMD5;

public:
  SourceFileInfo() = default;

  explicit SourceFileInfo(const llvm::Module &Mod, llvm::StringRef KernelName) {
    const llvm::Function *F = Mod.getFunction(KernelName);
    const llvm::DISubprogram *SP = F ? F->getSubprogram() : nullptr;
    if (!SP)
      return;

    initFromSubprogram(*F, *SP);
  }

  bool isKnown() const { return !Path.empty(); }

  // The copy is named by content hash so kernels from the same file share one.
  void copyTo(const std::string &Dir) {
    auto BufOrErr = llvm::MemoryBuffer::getFile(Path);
    if (!BufOrErr) {
      LOG_WARN("Cannot read source file {}: {}", Path,
               BufOrErr.getError().message());
      return;
    }
    llvm::StringRef Contents = (*BufOrErr)->getBuffer();

    llvm::MD5 Hash;
    Hash.update(Contents);
    llvm::MD5::MD5Result Result;
    Hash.final(Result);
    std::string Digest = Result.digest().str().str();

    // A copy that no longer matches what was compiled would misattribute the
    // recorded line range, so keep the compile-time checksum instead.
    if (!CompileMD5.empty() && CompileMD5 != Digest) {
      LOG_WARN("Source file {} changed since it was compiled (compiled MD5 {}, "
               "current MD5 {}); not copying it",
               Path, CompileMD5, Digest);
      return;
    }

    std::string Basename = std::filesystem::path(Path).filename().string();
    std::string Filename = "RecordedSource_" + Digest + "_" + Basename;
    std::filesystem::path Dest = std::filesystem::path(Dir) / Filename;
    if (!std::filesystem::exists(Dest)) {
      std::error_code EC;
      llvm::raw_fd_ostream Out(Dest.string(), EC);
      if (EC) {
        LOG_WARN("Cannot write source copy {}: {}", Dest.string(),
                 EC.message());
        return;
      }
      Out << Contents;
    }

    CopyName = Filename;
    MD5 = Digest;
  }

  void addToJSON(llvm::json::Object &Obj) const {
    if (!isKnown())
      return;

    Obj["SourceFile"] = Path;
    Obj["SourceLine"] = Line;
    Obj["SourceEndLine"] = EndLine;
    if (!CopyName.empty())
      Obj["SourceCopy"] = CopyName;
    const std::string &Checksum = CompileMD5.empty() ? MD5 : CompileMD5;
    if (!Checksum.empty())
      Obj["SourceMD5"] = Checksum;
  }

private:
  void initFromSubprogram(const llvm::Function &F,
                          const llvm::DISubprogram &SP) {
    const llvm::DIFile *File = SP.getFile();
    llvm::SmallString<256> Joined{File->getFilename()};
    if (!llvm::sys::path::is_absolute(Joined)) {
      Joined = File->getDirectory();
      llvm::sys::path::append(Joined, File->getFilename());
    }

    llvm::SmallString<256> Real;
    if (llvm::sys::fs::real_path(Joined, Real))
      Path = Joined.str().str();
    else
      Path = Real.str().str();

    Line = SP.getLine();
    EndLine = findLastCodeLine(F, SP);

    if (auto Checksum = File->getChecksum())
      if (Checksum->Kind == llvm::DIFile::CSK_MD5)
        CompileMD5 = Checksum->Value.str();
  }

  // The last line of the kernel that generated code, which for Clang is
  // normally its closing brace. Proteus captures the bitcode before inlining,
  // so every location in the function belongs to the kernel itself.
  static unsigned findLastCodeLine(const llvm::Function &F,
                                   const llvm::DISubprogram &SP) {
    unsigned Last = SP.getScopeLine();
    for (const llvm::BasicBlock &BB : F)
      for (const llvm::Instruction &I : BB)
        if (const llvm::DILocation *DL = I.getDebugLoc().get())
          Last = std::max(Last, DL->getLine());
    return Last;
  }
};

class KernelInstancesCollection {
  llvm::DenseMap<uint64_t, KernelInstance> Instances;
  uint64_t NumRecords;
  uint64_t TotalLaunches = 0;
  uint64_t TotalTimeNs = 0;
  KernelRecordingPolicy Policy;
  llvm::SmallVector<size_t> KernelArgSizes;
  llvm::SmallVector<std::string> KernelArgNames;
  llvm::SmallVector<bool> KernelSpecializations;
  llvm::SmallVector<std::function<double(void *)>> ConvertArgToDouble;
  llvm::SmallVector<std::string> ModuleFiles;
  const uint64_t StaticHash;
  const std::string KName;
  SourceFileInfo Source;
  bool CopySource;

private:
  // Parse Proteus's serialized bitcode in a Mneme-owned LLVMContext and
  // extract per-argument metadata. Operating on a Mneme-owned Module
  // keeps Mneme's LLVM runtime from touching any
  // Proteus-owned LLVM C++ object across the DSO boundary.
  void extractArgInfoFromBitcode(llvm::StringRef Bitcode) {
    auto Ctx = std::make_unique<llvm::LLVMContext>();
    llvm::MemoryBufferRef BufRef(Bitcode, KName);
    auto ModOrErr = llvm::parseBitcodeFile(BufRef, *Ctx);
    if (!ModOrErr)
      LOG_FATAL("parseBitcodeFile failed for kernel " + KName + ": " +
                llvm::toString(ModOrErr.takeError()));
    std::unique_ptr<llvm::Module> Mod = std::move(*ModOrErr);

    llvm::Function *F = Mod->getFunction(KName);
    if (!F)
      LOG_FATAL("Function " + KName + " not found in parsed bitcode");

    KernelArgSizes = mneme::getFuncDescr(*F);
    KernelArgNames = mneme::getArgNames(*F);
    KernelSpecializations = mneme::canSpecialize(*F);
    ConvertArgToDouble = mneme::convertToDouble(*F);
    Source = SourceFileInfo(*Mod, KName);
  }

  std::string StoreModuleBytes(llvm::StringRef Bytes,
                               const std::string &RecordReplayDir) {
    std::string Filename(
        std::filesystem::path(llvm::Twine(RecordReplayDir + "/RecordedIR_" +
                                          std::to_string(StaticHash) + ".bc")
                                  .str())
            .string());

    std::error_code EC;
    llvm::raw_fd_ostream OutBC(Filename, EC);
    if (EC)
      LOG_FATAL("Cannot write module ir file");
    OutBC << Bytes;
    OutBC.close();

    LOG_DEBUG("Stored Blob with StaticHash:{} to file {}", StaticHash,
              std::filesystem::canonical(Filename).string());
    return std::filesystem::path(Filename).filename().string();
  }

public:
  llvm::json::Object toJSON(KernelTimingType Timing) const {
    llvm::json::Object Collection;
    Collection["StaticHash"] = StaticHash;
    Collection["KernelName"] = KName;
    Collection["DemangledName"] = util::demangleKernelName(KName);
    Collection["Modules"] = llvm::json::Array(ModuleFiles);
    Collection["BinaryBlobs"] = llvm::json::Array();
    Collection["ArgNames"] = llvm::json::Array(KernelArgNames);
    Collection["Specializations"] = llvm::json::Array(KernelSpecializations);
    Source.addToJSON(Collection);
    Collection["TotalLaunches"] = TotalLaunches;
    bool KernelTiming = Timing != KernelTimingType::Off;
    if (KernelTiming) {
      Collection["TimingSource"] = getKernelTimingName(Timing);
      Collection["TotalTimeNs"] = TotalTimeNs;
    }
    llvm::json::Object JSONInstances;
    for (auto &[hash, KI] : Instances)
      if (KI.isRecorded())
        JSONInstances[std::to_string(hash)] = KI.toJSON(KernelTiming);
    Collection["instances"] = std::move(JSONInstances);
    return Collection;
  }

  KernelInstancesCollection(const proteus::runtime::KernelMetadata &KInfo,
                            const KernelRecordingPolicy &Policy,
                            bool CopySource)
      : NumRecords(0), Policy(Policy), StaticHash(KInfo.getStaticHash()),
        KName(KInfo.getName()), CopySource(CopySource) {}

  bool hasRecords() const { return NumRecords > 0; }
  const std::string &getName() const { return KName; }

  void addLaunchTime(const LaunchHandle &Launch, uint64_t Ns) {
    TotalTimeNs += Ns;
    if (!Launch.DynamicHash)
      return;
    Instances.find(*Launch.DynamicHash)
        ->second.addLaunchTime(Ns, Launch.LaunchIndex);
  }

  // Deferred to the first recording so unrecorded kernels write nothing.
  void storeModule(const std::string &MnemeDirectory,
                   const proteus::runtime::KernelMetadata &KInfo) {
    const auto &BitcodeBytes = KInfo.getBitcode();
    llvm::StringRef Bitcode(BitcodeBytes.data(), BitcodeBytes.size());
    if (Bitcode.empty())
      LOG_FATAL("Empty bitcode for kernel " + KName);

    extractArgInfoFromBitcode(Bitcode);
    if (CopySource) {
      if (Source.isKnown()) {
        Source.copyTo(MnemeDirectory);
      } else {
        // Printed without the logger so that default builds still tell the
        // user why the record has no source fields. One notice suffices since
        // the fix is the same for every kernel.
        static std::once_flag Warned;
        std::call_once(Warned, [this] {
          std::cerr << "[mneme] Kernel " << KName
                    << " has no line-table debug info; compile with "
                       "-gline-tables-only to record source locations "
                       "(reported once)\n";
        });
      }
    }
    ModuleFiles.emplace_back(StoreModuleBytes(Bitcode, MnemeDirectory));
  }

  LaunchHandle registerLaunch(const dim3 &GridDim, const dim3 &BlockDim,
                              uint64_t SharedMem) {
    TotalLaunches++;
    LaunchHandle Launch{StaticHash, std::nullopt, TotalLaunches};

    auto DynamicHash = computeHash(GridDim, BlockDim, SharedMem);
    auto It = Instances.find(DynamicHash);
    if (It == Instances.end()) {
      // A new launch configuration past the limit can never be recorded, so
      // it is only counted in TotalLaunches and the map stays bounded.
      if (NumRecords >= Policy.MaxRecords)
        return Launch;
      It = Instances.try_emplace(DynamicHash, GridDim, BlockDim, SharedMem)
               .first;
    }
    It->second.NumOccurrences++;
    Launch.DynamicHash = DynamicHash;
    return Launch;
  }

  static llvm::stable_hash
  computeHash(const dim3 &GridDim, const dim3 &BlockDim, uint64_t SharedMem) {
    auto BlockHash = llvm::stable_hash_combine((llvm::stable_hash)BlockDim.x,
                                               (llvm::stable_hash)BlockDim.y,
                                               (llvm::stable_hash)BlockDim.z);
    auto GridHash = llvm::stable_hash_combine((llvm::stable_hash)GridDim.x,
                                              (llvm::stable_hash)GridDim.y,
                                              (llvm::stable_hash)GridDim.z);
    auto DHash = llvm::stable_hash_combine(GridHash, BlockHash, SharedMem);
    return DHash;
  }

  // Returns the epilogue capture to run after the launch completes.
  template <DeviceVendors VendorTypes>
  std::optional<std::function<void()>> takeSnapshot(
      std::filesystem::path &MnemeDir, const void *Kernel,
      const LaunchHandle &Launch,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      void **Args, typename DeviceTraits<VendorTypes>::DeviceStream_t Stream,
      EpilogueSnapshotType EpilogueType) {
    if (!Launch.DynamicHash)
      return std::nullopt;
    uint64_t DynamicHash = *Launch.DynamicHash;

    if (!Policy.selects(Launch.LaunchIndex)) {
      LOG_DEBUG("Launch {} of kernel {} is not selected for recording",
                Launch.LaunchIndex, KName);
      return std::nullopt;
    }

    KernelInstance *Instance = &Instances.find(DynamicHash)->second;
    if (Instance->isRecorded()) {
      LOG_DEBUG(
          "Kernel {} with DynamicHash {} is already recorded, skipping ...",
          StaticHash, DynamicHash);
      return std::nullopt;
    }

    if (NumRecords >= Policy.MaxRecords)
      return std::nullopt;

    // Captured only for recorded launches, because it copies the bitcode.
    auto KInfo = proteus::runtime::captureKernelMetadata(Kernel);
    if (!KInfo)
      LOG_FATAL("Kernel metadata disappeared for counted kernel " + KName);
    const auto &GlobalVars = KInfo->getGlobals();

    if (NumRecords == 0)
      storeModule(MnemeDir.string(), *KInfo);
    NumRecords++;
    Instance->RecordedLaunchIndex = Launch.LaunchIndex;

    LOG_DEBUG("First Instance of Kernel {} with DynamicHash {}, recording ...",
              StaticHash, DynamicHash);

    std::filesystem::path Filename(MnemeDir /
                                   (std::string("DeviceState.prologue.") +
                                    std::to_string(StaticHash) + "." +
                                    std::to_string(DynamicHash) + ".mneme"));

    auto PrologueGlobals = std::make_shared<GlobalSnapshotData>();
    SnapshotInput<VendorTypes> In{GlobalVars, DeviceMemory, KernelArgSizes,
                                  Args, Stream};
    Instance->PrologueFn =
        BytesWriter<VendorTypes>(PrologueGlobals).write(Filename, In).string();

    // std::function requires a copyable callable, so the writer is shared.
    std::shared_ptr<SnapshotWriter<VendorTypes>> Writer =
        makeEpilogueWriter<VendorTypes>(EpilogueType, PrologueGlobals);

    return [this, DynamicHash, MnemeDir, GlobalVars, Writer, &DeviceMemory,
            Args, Stream] {
      std::filesystem::path Filename(MnemeDir /
                                     (std::string("DeviceState.epilogue.") +
                                      std::to_string(StaticHash) + "." +
                                      std::to_string(DynamicHash) + ".mneme"));

      SnapshotInput<VendorTypes> In{GlobalVars, DeviceMemory, KernelArgSizes,
                                    Args, Stream};
      Instances[DynamicHash].EpilogueFn = Writer->write(Filename, In).string();
    };
  }
};

class RecordDatabase {
  std::filesystem::path MnemeDirectory;
  llvm::DenseMap<uint64_t, KernelInstancesCollection> KernelRecords;
  llvm::DenseSet<uint64_t> FilteredKernels;
  // Launches are counted by kernel address, so only a kernel's first launch
  // pays for capturing its metadata.
  llvm::DenseMap<const void *, std::optional<uint64_t>> CountedKernels;
  EpilogueSnapshotType EpilogueType;
  bool CopySource;
  KernelTimingType Timing;

public:
  RecordDatabase(std::filesystem::path MnemeDirectory,
                 EpilogueSnapshotType EpilogueType, bool CopySource,
                 KernelTimingType Timing)
      : MnemeDirectory(std::move(MnemeDirectory)), EpilogueType(EpilogueType),
        CopySource(CopySource), Timing(Timing) {
    // Logging here constructs the logger first so it outlives flush() at exit.
    LOG_DEBUG("Recording into {}", MnemeDirectory.string());
  }

  void writeKernelJSON(uint64_t StaticHash) {
    auto It = KernelRecords.find(StaticHash);
    if (It == KernelRecords.end()) {
      LOG_WARN("Attempted to write JSON for unrecorded kernel hash {}",
               StaticHash);
      return;
    }
    auto *RecordPtr = &It->second;

    auto JsonFilename = MnemeDirectory / (std::to_string(StaticHash) + ".json");
    auto JSONRecord = RecordPtr->toJSON(Timing);

    std::error_code EC;
    llvm::raw_fd_ostream JsonOS(JsonFilename.string(), EC);
    if (EC) {
      LOG_WARN("Failed to open JSON file for kernel {}: {}", StaticHash,
               EC.message());
      return;
    }

    JsonOS << llvm::json::Value(std::move(JSONRecord));
    JsonOS.close();
    if (JsonOS.has_error()) {
      LOG_WARN("Failed to write JSON for kernel {}: {}", StaticHash,
               JsonOS.error().message());
      return;
    }
  }

  void flush() {
    for (const auto &Entry : KernelRecords)
      if (Entry.second.hasRecords())
        writeKernelJSON(Entry.first);
  }

  // Only a kernel's first launch resolves its policy, because kernels that can
  // record get a record and kernels that cannot are remembered here.
  std::optional<KernelRecordingPolicy>
  filterLaunch(uint64_t StaticHash, const std::string &KernelName) {
    if (FilteredKernels.contains(StaticHash))
      return std::nullopt;

    // Match the same original name displayed in the recording database.
    auto Policy = RecordingPolicy::get().resolve(
        util::demangleKernelName(KernelName), KernelName);
    if (Policy.MaxRecords == 0) {
      LOG_INFO("Skip record of Kernel {}", KernelName);
      FilteredKernels.insert(StaticHash);
      return std::nullopt;
    }
    return Policy;
  }

  KernelInstancesCollection *
  findOrCreateRecord(const proteus::runtime::KernelMetadata &KInfo) {
    auto StaticHash = KInfo.getStaticHash();
    auto It = KernelRecords.find(StaticHash);
    if (It == KernelRecords.end()) {
      auto Policy = filterLaunch(StaticHash, KInfo.getName());
      if (!Policy)
        return nullptr;
      It = KernelRecords.try_emplace(StaticHash, KInfo, *Policy, CopySource)
               .first;
      LOG_INFO("Created instance");
    }
    return &It->second;
  }

  // Returns nothing when Kernel has no record, because the recording filter
  // rejects it or Proteus has no metadata for it.
  std::optional<LaunchHandle> registerLaunch(const void *Kernel,
                                             const dim3 &GridDim,
                                             const dim3 &BlockDim,
                                             size_t SharedMem) {
    auto [It, Inserted] = CountedKernels.try_emplace(Kernel);
    if (Inserted) {
      auto KInfo = proteus::runtime::captureKernelMetadata(Kernel);
      if (KInfo && findOrCreateRecord(*KInfo))
        It->second = KInfo->getStaticHash();
    }
    if (!It->second)
      return std::nullopt;
    return KernelRecords.find(*It->second)
        ->second.registerLaunch(GridDim, BlockDim, SharedMem);
  }

  template <DeviceVendors VendorTypes>
  std::optional<std::function<void()>> takeSnapshot(
      const void *Kernel, const LaunchHandle &Launch,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      void **Args, typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
    return KernelRecords.find(Launch.StaticHash)
        ->second.takeSnapshot<VendorTypes>(MnemeDirectory, Kernel, Launch,
                                           DeviceMemory, Args, Stream,
                                           EpilogueType);
  }

  const std::string &kernelName(uint64_t StaticHash) const {
    return KernelRecords.find(StaticHash)->second.getName();
  }

  void addLaunchTime(const LaunchHandle &Launch, uint64_t Ns) {
    KernelRecords.find(Launch.StaticHash)->second.addLaunchTime(Launch, Ns);
  }
};

} // namespace mneme
