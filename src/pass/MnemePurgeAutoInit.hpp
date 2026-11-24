#pragma once

#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>

namespace mneme {

// Clang's !annotation can contain strings directly or group them in tuples.
inline bool hasAutoInitAnnotation(const llvm::Instruction &I) {
  const auto *Annotations = I.getMetadata(llvm::LLVMContext::MD_annotation);
  if (!Annotations)
    return false;

  auto IsAutoInit = [](const llvm::Metadata *MD) {
    const auto *Str = llvm::dyn_cast_or_null<llvm::MDString>(MD);
    return Str && Str->getString() == "auto-init";
  };

  for (const llvm::MDOperand &Op : Annotations->operands()) {
    if (IsAutoInit(Op.get()))
      return true;

    if (const auto *Group = llvm::dyn_cast_or_null<llvm::MDNode>(Op.get()))
      for (const llvm::MDOperand &GroupOp : Group->operands())
        if (IsAutoInit(GroupOp.get()))
          return true;
  }
  return false;
}

// Strip only the memsets emitted for -ftrivial-auto-var-init.
inline bool purgeAutoInit(llvm::Module &M) {
  bool Changed = false;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : llvm::make_early_inc_range(BB)) {
        auto *II = llvm::dyn_cast<llvm::IntrinsicInst>(&I);
        if (II && II->getIntrinsicID() == llvm::Intrinsic::memset &&
            hasAutoInitAnnotation(*II)) {
          II->eraseFromParent();
          Changed = true;
        }
      }
  return Changed;
}

} // namespace mneme
