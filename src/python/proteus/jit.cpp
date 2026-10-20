#include "../llvm/core.h"
#include "mneme/MnemeUtils.hpp"
#include "llvm-c/Core.h"
#include "llvm/IR/Module.h"
#include <chrono>
#include <iostream>
#include <llvm-c/Types.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <mneme/MnemeLogger.hpp>
#include <optional>
#include <proteus/CompilerInterfaceTypes.h>
#include <proteus/Init.h>
#include <proteus/impl/CompilerInterfaceRuntimeConstantInfo.h>
#include <proteus/impl/CoreLLVM.h>
#include <proteus/impl/CoreLLVMDevice.h>
#include <proteus/impl/Hashing.h>
#include <string>

#ifdef MNEME_ENABLE_HIP
constexpr const char *getRTCMethod() { return "serial"; }
#else
constexpr const char *getRTCMethod() { return "rtc"; }
#endif

using namespace proteus;

namespace {
inline std::optional<CodegenOption> fromString(std::string str) {
  if (str.compare("rtc") == 0) {
    return CodegenOption::RTC;
  } else if (str.compare("serial") == 0) {
    return CodegenOption::Serial;
  } else if (str.compare("parallel") == 0) {
    return CodegenOption::Parallel;
  }
  return std::nullopt;
}

OptimizationPipelineConfig makeOptimizationConfig(const char *OptLevel,
                                                  unsigned CodegenOptLevel) {
  llvm::StringRef OptLevelRef(OptLevel ? OptLevel : "");
  if (OptLevelRef.size() == 2 && OptLevelRef[0] == 'O') {
    char SimpleOptLevel = OptLevelRef[1];
    if (SimpleOptLevel == '0' || SimpleOptLevel == '1' ||
        SimpleOptLevel == '2' || SimpleOptLevel == '3' ||
        SimpleOptLevel == 's' || SimpleOptLevel == 'z')
      return OptimizationPipelineConfig(std::nullopt, SimpleOptLevel,
                                        CodegenOptLevel);
  }

  return OptimizationPipelineConfig(OptLevelRef.str(), '3', CodegenOptLevel);
}

// Clang marks every host-visible __device__/__constant__ variable
// externally_initialized. Replay writes their recorded values through
// hipModuleGetGlobal/cuModuleGetGlobal, so they must keep the flag (or loads
// from __constant__ fold to the initializer) and stay exported (or the lookup
// fails).
void pruneKeepingHostGlobals(llvm::Module &M) {
  pruneIR(M, /*UnsetExternallyInitialized=*/false);
}

void internalizeKeepingHostGlobals(llvm::Module &M, llvm::StringRef KernelSym) {
  auto *F = M.getFunction(KernelSym);
  llvm::internalizeModule(M, [F](const llvm::GlobalValue &GV) {
    if (&GV == F)
      return true;
    auto *GVar = llvm::dyn_cast<llvm::GlobalVariable>(&GV);
    return GVar && GVar->isExternallyInitialized();
  });
}

} // namespace

extern "C" {
API_EXPORT(void) ProteusPY_pruneIR(LLVMModuleRef Mod) {
  pruneKeepingHostGlobals(*llvm::unwrap(Mod));
}

API_EXPORT(void)
ProteusPY_internalize(LLVMModuleRef Mod, const char *KernelSym) {
  auto *M = llvm::unwrap(Mod);
  internalizeKeepingHostGlobals(*M, KernelSym);
}

API_EXPORT(void)
ProteusPY_optimize(LLVMModuleRef Mod, const char *DeviceArch,
                   const char *OptLevel, unsigned CodegenOptLevel) {
  auto *M = llvm::unwrap(Mod);
  auto start = std::chrono::high_resolution_clock::now();
  optimizeIR(*M, DeviceArch, makeOptimizationConfig(OptLevel, CodegenOptLevel));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> duration = end - start;
  float seconds = duration.count();
  LOG_DEBUG("Middle end compilation took {}", seconds);
}

API_EXPORT(LLVMMemoryBufferRef)
ProteusPY_codeGenObject(LLVMModuleRef Mod, const char *DeviceArch, unsigned CodegenOptLevel) {
  auto proteus_rtc = fromString(getRTCMethod());
  if (!proteus_rtc)
    LOG_FATAL("Unknown RTC value of {}", getRTCMethod());

  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  auto *M = llvm::unwrap(Mod);
  auto start = std::chrono::high_resolution_clock::now();
#ifdef MNEME_ENABLE_HIP
  auto OptConfig =
      OptimizationPipelineConfig(std::nullopt, '3', CodegenOptLevel);
  auto DeviceObject = proteus::codegenObject(
      *M, DeviceArch, GlobalLinkedBinaries, proteus_rtc.value(), OptConfig);
#else
  auto DeviceObject = proteus::codegenObject(
      *M, DeviceArch, GlobalLinkedBinaries, proteus_rtc.value());
#endif
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate duration and convert to seconds as float
  std::chrono::duration<float> duration = end - start;
  float seconds = duration.count();
  LOG_DEBUG("Backend compilation took {}", seconds);
  if (!DeviceObject) {
    LOG_WARN("Device Object is nullptr");
    return nullptr;
  }
  auto *ptr = DeviceObject.release();
  return wrap(ptr);
}

API_EXPORT(LLVMModuleRef)
ProteusPY_linkModules(const char **LLVMIRFiles, int size,
                      LLVMContextRef context, const char *KernelSym,
                      bool prune_flag = true, bool internalize_flag = true) {
  auto Ctx = unwrap(context);
  llvm::SmallVector<std::unique_ptr<llvm::Module>> RecordedModules;
  for (int i = 0; i < size; i++) {
    auto Fn = LLVMIRFiles[i];
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(Fn);
    if (!Buffer)
      LOG_FATAL("Error with loading file {}\n Error Code:", Fn,
                Buffer.getError().message());

    llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
        llvm::parseBitcodeFile(Buffer->get()->getMemBufferRef(), *Ctx);

    if (!ModuleOrErr)
      LOG_FATAL("Error parsing bitcode: {}",
                llvm::toString(ModuleOrErr.takeError()));

    if (prune_flag) {
      pruneKeepingHostGlobals(*ModuleOrErr->get());
    }

    RecordedModules.emplace_back(std::move(ModuleOrErr.get()));
  }

  auto Mod = proteus::linkModules(*unwrap(context), std::move(RecordedModules));

  if (internalize_flag) {
    internalizeKeepingHostGlobals(*Mod.get(), KernelSym);
  }

  return wrap(Mod.release());
}

API_EXPORT(uint64_t)
ProteusPY_specializeArguments(LLVMModuleRef Mod, const uint64_t StaticHash,
                              const char *KernelName, void **KernelArgs,
                              int NumArgs, int *SpecializeIndexes,
                              int NumSpecializations) {
  auto *M = llvm::unwrap(Mod);
  auto *F = M->getFunction(KernelName);
  SmallVector<int32_t> RCTypes(NumSpecializations);
  SmallVector<std::unique_ptr<RuntimeConstantInfo>, 64> RCStorage;

  for (int i = 0; i < NumSpecializations; i++) {
    RCStorage.emplace_back(std::make_unique<RuntimeConstantInfo>(
        proteus::convertTypeToRuntimeConstantType(
            F->getArg(SpecializeIndexes[i])->getType()),
        SpecializeIndexes[i]));
  }

  SmallVector<RuntimeConstant> RCVec;
  RCVec.reserve(RCStorage.size());

  for (const auto &RCInfo : RCStorage) {
    PROTEUS_DBG(Logger::logs("proteus")
                << "RC Index " << RCInfo->ArgInfo.Pos << " Type "
                << toString(RCInfo->ArgInfo.Type) << " ");

    RCVec.emplace_back(dispatchGetRuntimeConstantValue(KernelArgs, *RCInfo));
  }

  TransformArgumentSpecialization::transform(*M, *F, RCVec);

  auto Hash = hash(StaticHash, StringRef(KernelName), RCVec);
  return Hash.getValue();
}

API_EXPORT(uint64_t)
ProteusPY_specializeDims(LLVMModuleRef Mod, uint64_t CurrentHash,
                         const char *KernelName, dim3 GridDim, dim3 BlockDim) {
  auto *M = llvm::unwrap(Mod);
  auto *F = M->getFunction(KernelName);
  proteus::setKernelDims(*M, GridDim, BlockDim);
  auto Hash = hash(CurrentHash, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                   BlockDim.y, BlockDim.z);
  return Hash.getValue();
}

API_EXPORT(uint64_t)
ProteusPY_specializeDimsAssume(LLVMModuleRef Mod, uint64_t CurrentHash,
                         const char *KernelName, dim3 GridDim, dim3 BlockDim) {
  auto *M = llvm::unwrap(Mod);
  auto *F = M->getFunction(KernelName);
  proteus::setKernelDimsRange(*M, GridDim, BlockDim);
  auto Hash = hash(CurrentHash, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                   BlockDim.y, BlockDim.z);
  return Hash.getValue();
}


API_EXPORT(uint64_t)
ProteusPY_setLaunchBounds(LLVMModuleRef Mod, uint64_t CurrentHash,
                          const char *KernelName, int MaxThreadsPerBlock,
                          int MinBlocksPerSM) {
  auto *M = llvm::unwrap(Mod);
  auto *F = M->getFunction(KernelName);
  auto Hash = hash(CurrentHash, MaxThreadsPerBlock, MinBlocksPerSM);
  LOG_INFO("Was called with {} and {}", MaxThreadsPerBlock, MinBlocksPerSM);
  proteus::setLaunchBoundsForKernel(*F, MaxThreadsPerBlock, MinBlocksPerSM);
  return Hash.getValue();
}

API_EXPORT(const char*) ProteusPY_getCodegenMethod(){
  return getRTCMethod();
}

API_EXPORT(void)
ProteusPY_registerJITPassPlugin(const char *Path) {
  proteus::registerJITPassPlugin(Path);
}
}
