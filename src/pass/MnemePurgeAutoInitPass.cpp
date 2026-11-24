#include "MnemePurgeAutoInit.hpp"
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#if __has_include(<llvm/Plugins/PassPlugin.h>)
#include <llvm/Plugins/PassPlugin.h>
#elif __has_include(<llvm/Passes/PassPlugin.h>)
#include <llvm/Passes/PassPlugin.h>
#else
#error "Cannot find LLVM PassPlugin.h"
#endif

using namespace llvm;

namespace {

class PurgeAutoInitPass : public PassInfoMixin<PurgeAutoInitPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    return mneme::purgeAutoInit(M) ? PreservedAnalyses::none()
                                   : PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MnemePurgeAutoInit", "0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "purge-autoinit")
                    return false;
                  MPM.addPass(PurgeAutoInitPass());
                  return true;
                });
          }};
}
