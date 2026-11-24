#include "../../pass/MnemePurgeAutoInit.hpp"
#include "llvm/IR/Module.h"
#include "core.h"
#include "llvm-c/Analysis.h"
#include "llvm-c/Core.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <clocale>
#include <mneme/MnemeLogger.hpp>
#include <string>

/* An iterator around a module's globals, including the stop condition */
struct GlobalsIterator {
  typedef llvm::Module::global_iterator iterator;
  iterator cur;
  iterator end;

  GlobalsIterator(iterator cur, iterator end) : cur(cur), end(end) {}
};

struct OpaqueGlobalsIterator;
typedef OpaqueGlobalsIterator *LLVMGlobalsIteratorRef;

/* An iterator around a module's functions, including the stop condition */
struct FunctionsIterator {
  typedef llvm::Module::const_iterator const_iterator;
  const_iterator cur;
  const_iterator end;

  FunctionsIterator(const_iterator cur, const_iterator end)
      : cur(cur), end(end) {}
};

struct OpaqueFunctionsIterator;
typedef OpaqueFunctionsIterator *LLVMFunctionsIteratorRef;

/* module types iterator */
class TypesIterator {
private:
  llvm::TypeFinder finder;
  using const_iterator = llvm::TypeFinder::const_iterator;
  const_iterator cur;

public:
  TypesIterator(llvm::Module &m, bool namedOnly) : finder(llvm::TypeFinder()) {
    finder.run(m, namedOnly);
    cur = finder.begin();
  }
  const llvm::Type *next() {
    if (cur != finder.end()) {
      return *cur++;
    }
    return nullptr;
  }
};

typedef TypesIterator *LLVMTypesIteratorRef;

/* An iterator around a module's aliases, including the stop condition */
struct AliasesIterator {
  typedef llvm::Module::alias_iterator iterator;
  iterator cur;
  iterator end;

  AliasesIterator(iterator cur, iterator end) : cur(cur), end(end) {}
};

struct OpaqueAliasesIterator;
typedef OpaqueAliasesIterator *LLVMAliasesIteratorRef;

/* An iterator around a module's ifuncs, including the stop condition */
struct IFuncsIterator {
  typedef llvm::Module::ifunc_iterator iterator;
  iterator cur;
  iterator end;

  IFuncsIterator(iterator cur, iterator end) : cur(cur), end(end) {}
};

struct OpaqueIFuncsIterator;
typedef OpaqueIFuncsIterator *LLVMIFuncsIteratorRef;

//
// Local helper functions
//

namespace llvm {

static LLVMGlobalsIteratorRef wrap(GlobalsIterator *GI) {
  return reinterpret_cast<LLVMGlobalsIteratorRef>(GI);
}

static GlobalsIterator *unwrap(LLVMGlobalsIteratorRef GI) {
  return reinterpret_cast<GlobalsIterator *>(GI);
}

static LLVMFunctionsIteratorRef wrap(FunctionsIterator *GI) {
  return reinterpret_cast<LLVMFunctionsIteratorRef>(GI);
}

static FunctionsIterator *unwrap(LLVMFunctionsIteratorRef GI) {
  return reinterpret_cast<FunctionsIterator *>(GI);
}

static LLVMTypesIteratorRef wrap(TypesIterator *TyI) {
  return reinterpret_cast<LLVMTypesIteratorRef>(TyI);
}

static TypesIterator *unwrap(LLVMTypesIteratorRef TyI) {
  return reinterpret_cast<TypesIterator *>(TyI);
}

static LLVMAliasesIteratorRef wrap(AliasesIterator *TyI) {
  return reinterpret_cast<LLVMAliasesIteratorRef>(TyI);
}

static AliasesIterator *unwrap(LLVMAliasesIteratorRef TyI) {
  return reinterpret_cast<AliasesIterator *>(TyI);
}

static LLVMIFuncsIteratorRef wrap(IFuncsIterator *TyI) {
  return reinterpret_cast<LLVMIFuncsIteratorRef>(TyI);
}

static IFuncsIterator *unwrap(LLVMIFuncsIteratorRef TyI) {
  return reinterpret_cast<IFuncsIterator *>(TyI);
}

} // end namespace llvm

//
// Exported API
//

extern "C" {

API_EXPORT(void)
LLVMPY_DisposeModule(LLVMModuleRef m) { return LLVMDisposeModule(m); }

API_EXPORT(void)
LLVMPY_PrintModuleToString(LLVMModuleRef M, const char **outstr) {
  // Change the locale to en_US before calling LLVM to print the module
  // due to a LLVM bug https://llvm.org/bugs/show_bug.cgi?id=12906
  char *old_locale = strdup(setlocale(LC_ALL, NULL));
  setlocale(LC_ALL, "C");

  *outstr = LLVMPrintModuleToString(M);

  // Revert locale
  setlocale(LC_ALL, old_locale);
  free(old_locale);
}

API_EXPORT(const char *)
LLVMPY_GetModuleSourceFileName(LLVMModuleRef M) {
  return llvm::unwrap(M)->getSourceFileName().c_str();
}

API_EXPORT(const char *)
LLVMPY_GetModuleName(LLVMModuleRef M) {
  return llvm::unwrap(M)->getModuleIdentifier().c_str();
}

API_EXPORT(void)
LLVMPY_SetModuleName(LLVMModuleRef M, const char *Name) {
  llvm::unwrap(M)->setModuleIdentifier(Name);
}

API_EXPORT(LLVMValueRef)
LLVMPY_GetNamedFunction(LLVMModuleRef M, const char *Name) {
  return LLVMGetNamedFunction(M, Name);
}

API_EXPORT(LLVMValueRef)
LLVMPY_GetNamedGlobalVariable(LLVMModuleRef M, const char *Name) {
  using namespace llvm;
  return wrap(unwrap(M)->getGlobalVariable(Name));
}

API_EXPORT(LLVMTypeRef)
LLVMPY_GetNamedStructType(LLVMModuleRef M, const char *Name) {
  return LLVMGetTypeByName(M, Name);
}

API_EXPORT(int)
LLVMPY_VerifyModule(LLVMModuleRef M, char **OutMsg) {
  return LLVMVerifyModule(M, LLVMReturnStatusAction, OutMsg);
}

API_EXPORT(void)
LLVMPY_GetDataLayout(LLVMModuleRef M, const char **DL) {
  *DL = LLVMGetDataLayoutStr(M);
}

API_EXPORT(void)
LLVMPY_SetDataLayout(LLVMModuleRef M, const char *DL) {
  LLVMSetDataLayout(M, DL);
}

API_EXPORT(void)
LLVMPY_GetTarget(LLVMModuleRef M, const char **Triple) {
  *Triple = LLVMGetTarget(M);
}

API_EXPORT(void)
LLVMPY_SetTarget(LLVMModuleRef M, const char *Triple) {
  LLVMSetTarget(M, Triple);
}

// Iteration APIs

API_EXPORT(LLVMGlobalsIteratorRef)
LLVMPY_ModuleGlobalsIter(LLVMModuleRef M) {
  using namespace llvm;
  Module *mod = unwrap(M);
  return wrap(new GlobalsIterator(mod->global_begin(), mod->global_end()));
}

API_EXPORT(LLVMFunctionsIteratorRef)
LLVMPY_ModuleFunctionsIter(LLVMModuleRef M) {
  using namespace llvm;
  Module *mod = unwrap(M);
  return wrap(new FunctionsIterator(mod->begin(), mod->end()));
}

API_EXPORT(LLVMTypesIteratorRef)
LLVMPY_ModuleTypesIter(LLVMModuleRef M) {
  llvm::Module *mod = llvm::unwrap(M);
  auto *iter = new TypesIterator(*mod, false);
  return llvm::wrap(iter);
}

API_EXPORT(LLVMAliasesIteratorRef)
LLVMPY_ModuleAliasesIter(LLVMModuleRef M) {
  llvm::Module *mod = llvm::unwrap(M);
  auto *iter = new AliasesIterator(mod->alias_begin(), mod->alias_end());
  return llvm::wrap(iter);
}

API_EXPORT(LLVMIFuncsIteratorRef)
LLVMPY_ModuleIFuncsIter(LLVMModuleRef M) {
  llvm::Module *mod = llvm::unwrap(M);
  auto *iter = new IFuncsIterator(mod->ifunc_begin(), mod->ifunc_end());
  return llvm::wrap(iter);
}

/*
  These functions return NULL if we are at the end
*/
API_EXPORT(LLVMValueRef)
LLVMPY_GlobalsIterNext(LLVMGlobalsIteratorRef GI) {
  using namespace llvm;
  GlobalsIterator *iter = unwrap(GI);
  if (iter->cur != iter->end) {
    return wrap(&*iter->cur++);
  } else {
    return NULL;
  }
}

API_EXPORT(LLVMValueRef)
LLVMPY_FunctionsIterNext(LLVMFunctionsIteratorRef GI) {
  using namespace llvm;
  FunctionsIterator *iter = unwrap(GI);
  if (iter->cur != iter->end) {
    return wrap(&*iter->cur++);
  } else {
    return NULL;
  }
}

API_EXPORT(LLVMTypeRef)
LLVMPY_TypesIterNext(LLVMTypesIteratorRef TyI) {
  return llvm::wrap(llvm::unwrap(TyI)->next());
}

API_EXPORT(LLVMValueRef)
LLVMPY_AliasesIterNext(LLVMAliasesIteratorRef GI) {
  using namespace llvm;
  AliasesIterator *iter = unwrap(GI);
  if (iter->cur != iter->end) {
    return wrap(&*iter->cur++);
  } else {
    return NULL;
  }
}

API_EXPORT(LLVMValueRef)
LLVMPY_IFuncsIterNext(LLVMIFuncsIteratorRef GI) {
  using namespace llvm;
  IFuncsIterator *iter = unwrap(GI);
  if (iter->cur != iter->end) {
    return wrap(&*iter->cur++);
  } else {
    return NULL;
  }
}

API_EXPORT(void)
LLVMPY_WriteBitcode(LLVMModuleRef ModRef, const char *Filename) {

  // Open the file for writing.
  std::error_code EC;
  auto *Mod = llvm::unwrap(ModRef);
  llvm::raw_fd_ostream Out(Filename, EC, llvm::sys::fs::OF_None);
  if (EC)
    LOG_FATAL("Got Error when storing Module\nEC:{}", EC.message());

  // Emit the module.
  llvm::WriteBitcodeToFile(*Mod, Out);
  Out.flush(); // Ensure everything is written.

  return;
}

API_EXPORT(void)
LLVMPY_DisposeGlobalsIter(LLVMGlobalsIteratorRef GI) {
  delete llvm::unwrap(GI);
}

API_EXPORT(void)
LLVMPY_DisposeFunctionsIter(LLVMFunctionsIteratorRef GI) {
  delete llvm::unwrap(GI);
}

API_EXPORT(void)
LLVMPY_DisposeTypesIter(LLVMTypesIteratorRef TyI) { delete llvm::unwrap(TyI); }

API_EXPORT(void)
LLVMPY_DisposeAliasesIter(LLVMAliasesIteratorRef GI) {
  delete llvm::unwrap(GI);
}

API_EXPORT(void)
LLVMPY_DisposeIFuncsIter(LLVMIFuncsIteratorRef GI) { delete llvm::unwrap(GI); }

API_EXPORT(LLVMModuleRef)
LLVMPY_CloneModule(LLVMModuleRef M) { return LLVMCloneModule(M); }

API_EXPORT(bool)
LLVMPY_PurgeAutoInit(LLVMModuleRef M) {
  return mneme::purgeAutoInit(*llvm::unwrap(M));
}

} // end extern "C"
