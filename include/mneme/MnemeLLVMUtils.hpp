#pragma once
#include "llvm/ADT/DenseMapInfo.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <string>

#include "mneme/MnemeLogger.hpp"

namespace llvm {
template <> struct DenseMapInfo<std::string> {
  static inline std::string getEmptyKey() {
    return std::string(); // Define an empty key
  }
  static inline std::string getTombstoneKey() {
    return std::string("<TOMBSTONE_KEY>");
  }
  static unsigned getHashValue(const std::string &Key) {
    // Use std::hash for hashing the string
    return std::hash<std::string>{}(Key);
  }
  static bool isEqual(const std::string &LHS, const std::string &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm
//
namespace mneme {
namespace util {
inline std::string demangleKernelName(const std::string &Name) {
  return llvm::demangle(Name.substr(0, Name.find("__intern__")));
}

template <typename Ty>
void writeScalar(llvm::raw_ostream &OS, const Ty &Value) {
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Value), sizeof(Value));
}

inline void writeBytes(llvm::raw_ostream &OS, llvm::StringRef Data) {
  if (!Data.empty())
    OS << Data;
}

inline void writeBytes(llvm::raw_ostream &OS, llvm::ArrayRef<uint8_t> Data) {
  writeBytes(OS, llvm::StringRef(reinterpret_cast<const char *>(Data.data()),
                                 Data.size()));
}
} // namespace util

inline static llvm::SmallVector<std::string> getArgNames(llvm::Function &F) {
  llvm::SmallVector<std::string> Info;
  for (auto &A : F.args()) {
    auto name = A.getName().str();
    if (name.size() > 0)
      Info.push_back(name);
    else
      Info.push_back("");
  }
  return Info;
}

inline static llvm::SmallVector<bool> canSpecialize(llvm::Function &F) {
  llvm::SmallVector<bool> RRInfo;
  for (auto &A : F.args()) {
    llvm::Type *ArgType = A.getType();
    if (ArgType->isIntegerTy(1)) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isIntegerTy(8)) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isIntegerTy(32)) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isIntegerTy(64)) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isFloatTy()) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isDoubleTy()) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
               ArgType->isFP128Ty()) {
      RRInfo.emplace_back(true);
    } else {
      RRInfo.emplace_back(false);
    }
  }
  return RRInfo;
}

inline static llvm::SmallVector<std::function<double(void *)>>
convertToDouble(llvm::Function &F) {
  llvm::SmallVector<std::function<double(void *)>> RRInfo;
  for (auto &A : F.args()) {
    llvm::Type *ArgType = A.getType();
    if (ArgType->isIntegerTy(1)) {
      RRInfo.emplace_back([](void *data) {
        bool val = *(bool *)data;
        return (double)val;
      });
    } else if (ArgType->isIntegerTy(8)) {
      RRInfo.emplace_back([](void *data) {
        int8_t val = *(int8_t *)data;
        return (double)val;
      });
    } else if (ArgType->isIntegerTy(32)) {
      RRInfo.emplace_back([](void *data) {
        int32_t val = *(int32_t *)data;
        return (double)val;
      });
    } else if (ArgType->isIntegerTy(64)) {
      RRInfo.emplace_back([](void *data) {
        int64_t val = *(int64_t *)data;
        return (double)val;
      });
    } else if (ArgType->isFloatTy()) {
      RRInfo.emplace_back([](void *data) {
        float val = *(float *)data;
        return (double)val;
      });
    } else if (ArgType->isDoubleTy()) {
      RRInfo.emplace_back([](void *data) {
        double val = *(double *)data;
        return (double)val;
      });
    } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
               ArgType->isFP128Ty()) {
      RRInfo.emplace_back([](void *data) {
        double val = *(double *)data;
        return (double)val;
      });
    } else {
      RRInfo.emplace_back([](void *data) {
        double val = *(double *)data;
        return -1.0;
      });
    }
  }
  return RRInfo;
}

inline static llvm::SmallVector<uint64_t, 8> getFuncDescr(llvm::Function &F) {
  llvm::SmallVector<uint64_t, 8> RRInfo;
  auto DL = F.getParent()->getDataLayout();
  for (auto &A : F.args()) {
    // Datatypes such as structs passed by value to kernels are copied
    // into a parameter vector. Over here we test whether an argument is
    // byval, if it is we know on the host side this invocation forwards
    // the arguments by value
    if (A.hasByRefAttr() || A.hasByValAttr()) {
      RRInfo.emplace_back(DL.getTypeStoreSize(A.getPointeeInMemoryValueType()));
    } else {
      RRInfo.emplace_back(DL.getTypeStoreSize(A.getType()));
    }
  }
  return RRInfo;
}
} // namespace mneme
