#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MD5.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>
#include <regex>
#include <unordered_map>
#include <vector>

#include "llvm/Demangle/Demangle.h"
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

struct KernelInstance {
  std::string PrologueFn;
  std::string EpilogueFn;
  dim3 BlockDim;
  dim3 GridDim;
  llvm::SmallVector<double> ArgValues;
  int NumOccurrences;
  uint64_t SharedMem;
  static llvm::json::Object toJSON(const dim3 &Dim) {
    llvm::json::Object JSONDim;
    JSONDim["x"] = Dim.x;
    JSONDim["y"] = Dim.y;
    JSONDim["z"] = Dim.z;
    return JSONDim;
  }
  llvm::json::Object toJSON() const {
    llvm::json::Object instance;
    instance["Prologue"] = PrologueFn;
    instance["Epilogue"] = EpilogueFn;
    instance["BlockDims"] = KernelInstance::toJSON(BlockDim);
    instance["GridDims"] = KernelInstance::toJSON(GridDim);
    instance["SharedMem"] = SharedMem;
    instance["Args"] = llvm::json::Array(ArgValues);
    instance["Occurrences"] = NumOccurrences;
    return instance;
  }
  KernelInstance(dim3 &GridDim, dim3 &BlockDim, uint64_t SharedMem, void **Args)
      : GridDim(GridDim), BlockDim(BlockDim), SharedMem(SharedMem),
        NumOccurrences(1) {}
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
    if (!SP) {
      // Printed without the logger so that default builds still tell the user
      // why the record has no source fields.
      std::cerr << "[mneme] Kernel " << KernelName.str()
                << " has no line-table debug info; compile with "
                   "-gline-tables-only to record its source location\n";
      return;
    }

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
  void *VAddr;
  uint64_t VASize;
  llvm::DenseMap<uint64_t, KernelInstance> Instances;
  uint64_t NumRecords;
  uint64_t MaxRecordings;
  llvm::SmallVector<size_t> KernelArgSizes;
  llvm::SmallVector<std::string> KernelArgNames;
  llvm::SmallVector<bool> KernelSpecializations;
  llvm::SmallVector<std::function<double(void *)>> ConvertArgToDouble;
  llvm::SmallVector<std::string> ModuleFiles;
  const std::string KName;
  SourceFileInfo Source;

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
                               const std::string &RecordReplayDir,
                               uint64_t StaticHash) {
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
  llvm::json::Object toJSON(uint64_t StaticHash) const {
    llvm::json::Object Collection;
    Collection["StaticHash"] = StaticHash;
    Collection["VAddr"] =
        util::pointerToHexString(reinterpret_cast<uint8_t *>(VAddr));
    Collection["VASize"] = VASize;
    Collection["KernelName"] = KName;
    std::size_t pos = KName.find("__intern__");
    std::string Orig =
        (pos != std::string::npos) ? KName.substr(0, pos) : KName;
    Collection["DemangledName"] = llvm::demangle(Orig);
    Collection["Modules"] = llvm::json::Array(ModuleFiles);
    Collection["BinaryBlobs"] = llvm::json::Array();
    Collection["ArgNames"] = llvm::json::Array(KernelArgNames);
    Collection["Specializations"] = llvm::json::Array(KernelSpecializations);
    Source.addToJSON(Collection);
    llvm::json::Object JSONInstances;
    for (auto &[hash, KI] : Instances) {
      JSONInstances[std::to_string(hash)] = KI.toJSON();
    }
    Collection["instances"] = std::move(JSONInstances);
    return Collection;
  }

  KernelInstancesCollection(const std::string &MnemeDirectory, void *VAddr,
                            uint64_t VASize,
                            const proteus::runtime::KernelMetadata &KInfo,
                            uint64_t MaxRecordings, bool CopySource)
      : VAddr(VAddr), VASize(VASize), MaxRecordings(MaxRecordings),
        NumRecords(0), KName(KInfo.getName()) {
    const auto &BitcodeBytes = KInfo.getBitcode();
    llvm::StringRef Bitcode(BitcodeBytes.data(), BitcodeBytes.size());
    if (Bitcode.empty())
      LOG_FATAL("Empty bitcode for kernel " + KName);

    extractArgInfoFromBitcode(Bitcode);
    if (CopySource && Source.isKnown())
      Source.copyTo(MnemeDirectory);
    ModuleFiles.emplace_back(StoreModuleBytes(
        Bitcode, MnemeDirectory, KInfo.getStaticHash()));
  }

  llvm::stable_hash computeHash(dim3 &GridDim, dim3 &BlockDim,
                                uint64_t SharedMem, void **Args) {
    auto BlockHash = llvm::stable_hash_combine((llvm::stable_hash)BlockDim.x,
                                               (llvm::stable_hash)BlockDim.y,
                                               (llvm::stable_hash)BlockDim.z);
    auto GridHash = llvm::stable_hash_combine((llvm::stable_hash)GridDim.x,
                                              (llvm::stable_hash)GridDim.y,
                                              (llvm::stable_hash)GridDim.z);
    auto DHash = llvm::stable_hash_combine(GridHash, BlockHash, SharedMem);
    return DHash;
  }

  template <DeviceVendors VendorTypes>
  std::optional<std::function<
      void(llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      std::filesystem::path &MnemeDir,
      const proteus::runtime::GlobalMetadataMap &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream,
      uint64_t StaticHash, EpilogueSnapshotType EpilogueType,
      uint64_t LaunchNumber, Deduplication Dedup) {

    if (NumRecords >= MaxRecordings)
      return std::nullopt;

    auto InstanceID = Dedup == Deduplication::Invocations
                           ? LaunchNumber
                           : computeHash(GridDim, BlockDim, SharedMem, Args);

    if (Instances.contains(InstanceID)) {
      Instances[InstanceID].NumOccurrences++;
      LOG_DEBUG(
          "Kernel {} with InstanceID {} is already recorded, skipping ...",
          StaticHash, InstanceID);
      return std::nullopt;
    }

    NumRecords++;

    LOG_DEBUG("First Instance of Kernel {} with InstanceID {}, recording ...",
              StaticHash, InstanceID);

    Instances.insert(
        {InstanceID, KernelInstance(GridDim, BlockDim, SharedMem, Args)});
    std::filesystem::path Filename(MnemeDir /
                                   (std::string("DeviceState.prologue.") +
                                    std::to_string(StaticHash) + "." +
                                    std::to_string(InstanceID) + ".mneme"));

    auto PrologueGlobals = std::make_shared<GlobalSnapshotData>();
    SnapshotInput<VendorTypes> In{GlobalVars, DeviceMemory, KernelArgSizes,
                                  Args, Stream};
    Instances[InstanceID].PrologueFn =
        BytesWriter<VendorTypes>(PrologueGlobals).write(Filename, In).string();

    // std::function requires a copyable callable, so the writer is shared.
    std::shared_ptr<SnapshotWriter<VendorTypes>> Writer =
        makeEpilogueWriter<VendorTypes>(EpilogueType, PrologueGlobals);

    std::function<void(llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &,
                       void **,
                       typename DeviceTraits<VendorTypes>::DeviceStream_t)>
        CaptureEpilogue =
            [this, InstanceID, StaticHash, MnemeDir, GlobalVars, Writer](
                llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>>
                    &DeviceMemory,
                void **Args,
                typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
              std::filesystem::path Filename(
                  MnemeDir / (std::string("DeviceState.epilogue.") +
                              std::to_string(StaticHash) + "." +
                              std::to_string(InstanceID) + ".mneme"));

              SnapshotInput<VendorTypes> In{GlobalVars, DeviceMemory,
                                            KernelArgSizes, Args, Stream};
              Instances[InstanceID].EpilogueFn =
                  Writer->write(Filename, In).string();
            };
    return CaptureEpilogue;
  }
};

class RecordDatabase {
  std::filesystem::path MnemeDirectory;
  llvm::DenseMap<uint64_t, KernelInstancesCollection> KernelRecords;
  struct KernelLaunchState {
    KernelRecordingPolicy Policy;
    uint64_t Count = 0;
  };
  llvm::DenseMap<uint64_t, KernelLaunchState> KernelLaunches;
  EpilogueSnapshotType EpilogueType;
  bool CopySource;

public:
  RecordDatabase() {
    const auto &Conf = Config::get();

    MnemeDirectory = Conf.getDataDirectory();
    EpilogueType = Conf.EpilogueType;
    CopySource = Conf.CopySource;
  }

  void writeKernelJSON(uint64_t StaticHash) {
    auto It = KernelRecords.find(StaticHash);
    if (It == KernelRecords.end()) {
      LOG_WARN("Attempted to write JSON for unrecorded kernel hash {}", StaticHash);
      return;
    }
    auto* RecordPtr = &It->second;

    auto JsonFilename = MnemeDirectory / (std::to_string(StaticHash) + ".json");
    auto JSONRecord = RecordPtr->toJSON(StaticHash);

    std::error_code EC;
    llvm::raw_fd_ostream JsonOS(JsonFilename.string(), EC);
    if (EC) {
      LOG_WARN("Failed to open JSON file for kernel {}: {}", StaticHash, EC.message());
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

  template <DeviceVendors VendorTypes>
  std::optional<std::function<
      void(llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      void *VAddr, uint64_t VASize,
      const proteus::runtime::KernelMetadata &KInfo,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
    
    auto StaticHash = KInfo.getStaticHash();
    auto [StateIt, Inserted] = KernelLaunches.try_emplace(StaticHash);
    auto &State = StateIt->second;
    if (Inserted) {
      // Match the same original name displayed in the recording database.
      auto Name = KInfo.getName();
      auto Pos = Name.find("__intern__");
      State.Policy = RecordingPolicy::get().resolve(
          llvm::demangle(Name.substr(0, Pos)));
    }
    
    ++State.Count;
    if (!State.Policy.selects(State.Count))
      return std::nullopt;

    auto IT = KernelRecords.try_emplace(StaticHash, getDir(), VAddr, VASize,
                                        KInfo, State.Policy.MaxRecords, CopySource);
    if (IT.second)
      LOG_INFO("Created instance");
    return IT.first->second.takeSnapshot<VendorTypes>(
        MnemeDirectory, KInfo.getGlobals(), DeviceMemory, GridDim, BlockDim,
        Args, SharedMem, Stream, StaticHash, EpilogueType, State.Count,
        RecordingPolicy::get().deduplication());
  }

  const std::string getDir() const { return MnemeDirectory.string(); }
};

} // namespace mneme
