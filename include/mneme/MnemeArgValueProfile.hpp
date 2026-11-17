#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

#include <proteus/impl/CoreLLVM.h>

namespace mneme {

enum class ArgLeafKind { Integer, Float, Pointer };

// A scalar inside a kernel argument, found at Offset bytes past Args[ArgIdx].
struct ArgLeaf {
  unsigned ArgIdx;
  uint64_t Offset;
  unsigned Size;
  ArgLeafKind Kind;
  std::string Path;
  std::string Type;
  // The optimized kernel may read the leaf. Unread leaves, such as the byte of
  // an empty functor, cannot change the generated code and hold whatever the
  // host left there, so they are not part of any specialization key.
  bool IsRead = true;
};

struct KernelArgLayout {
  llvm::SmallVector<std::string> ArgNames;
  llvm::SmallVector<std::string> ArgTypes;
  // The bytes each argument reads were found exactly; otherwise every leaf of
  // the argument counts as read.
  llvm::SmallVector<bool> ArgReadsKnown;
  llvm::SmallVector<ArgLeaf> Leaves;
  // Pieces of arguments without leaves, such as long double, i128 or padding.
  llvm::SmallVector<std::string> Unsupported;
};

namespace arg_layout_detail {

inline std::string typeName(llvm::Type *T) {
  std::string Name;
  llvm::raw_string_ostream OS(Name);
  T->print(OS);
  return Name;
}

inline void flattenType(llvm::Type *T, unsigned ArgIdx, uint64_t Offset,
                        const std::string &Path, const llvm::DataLayout &DL,
                        KernelArgLayout &Layout) {
  if (auto *ST = llvm::dyn_cast<llvm::StructType>(T)) {
    // Walking the fields through the struct layout skips padding bytes, which
    // hold whatever the host left there.
    const llvm::StructLayout *SL = DL.getStructLayout(ST);
    for (unsigned I = 0; I < ST->getNumElements(); I++)
      flattenType(ST->getElementType(I), ArgIdx,
                  Offset + SL->getElementOffset(I),
                  Path + "." + std::to_string(I), DL, Layout);
    return;
  }

  llvm::Type *ElemTy = nullptr;
  uint64_t NumElems = 0;
  if (auto *AT = llvm::dyn_cast<llvm::ArrayType>(T)) {
    ElemTy = AT->getElementType();
    NumElems = AT->getNumElements();
  } else if (auto *VT = llvm::dyn_cast<llvm::FixedVectorType>(T)) {
    ElemTy = VT->getElementType();
    NumElems = VT->getNumElements();
  }
  // Clang pads some structs with explicit [N x i8] fields, which hold whatever
  // the host left there, and IR cannot tell them from a char array member.
  if (ElemTy && ElemTy->isIntegerTy(8)) {
    Layout.Unsupported.push_back(Path + ": " + typeName(T) +
                                 " (padding or char array)");
    return;
  }
  if (ElemTy) {
    uint64_t Stride = DL.getTypeAllocSize(ElemTy);
    for (uint64_t I = 0; I < NumElems; I++)
      flattenType(ElemTy, ArgIdx, Offset + I * Stride,
                  Path + "[" + std::to_string(I) + "]", DL, Layout);
    return;
  }

  uint64_t Size = DL.getTypeStoreSize(T);
  std::optional<ArgLeafKind> Kind;
  if (T->isPointerTy())
    Kind = ArgLeafKind::Pointer;
  else if (T->isIntegerTy())
    Kind = ArgLeafKind::Integer;
  else if (T->isHalfTy() || T->isBFloatTy() || T->isFloatTy() ||
           T->isDoubleTy())
    Kind = ArgLeafKind::Float;

  if (!Kind || Size == 0 || Size > sizeof(uint64_t)) {
    Layout.Unsupported.push_back(Path + ": " + typeName(T));
    return;
  }
  Layout.Leaves.push_back(ArgLeaf{ArgIdx, Offset, static_cast<unsigned>(Size),
                                  *Kind, Path, typeName(T)});
}

// Half-open byte ranges of an argument that a kernel reads.
using ByteRanges = llvm::SmallVector<std::pair<int64_t, int64_t>>;

// Adds the bytes read through Ptr, which points Offset bytes into an in-memory
// argument. Returns false when a use is anything but a load, a memcpy source
// or a constant-offset address computation, so any byte may be read.
inline bool collectPointerReads(const llvm::Value *Ptr, int64_t Offset,
                                const llvm::DataLayout &DL, ByteRanges &Reads) {
  for (const llvm::User *U : Ptr->users()) {
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
      Reads.emplace_back(Offset, Offset + DL.getTypeStoreSize(LI->getType()));
      continue;
    }
    if (auto *MT = llvm::dyn_cast<llvm::MemTransferInst>(U)) {
      auto *Len = llvm::dyn_cast<llvm::ConstantInt>(MT->getLength());
      if (MT->getDest() == Ptr || !Len)
        return false;
      Reads.emplace_back(Offset, Offset + Len->getSExtValue());
      continue;
    }
    if (auto *GEP = llvm::dyn_cast<llvm::GEPOperator>(U)) {
      llvm::APInt Delta(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      if (GEP->getPointerOperand() != Ptr ||
          !GEP->accumulateConstantOffset(DL, Delta) ||
          !collectPointerReads(GEP, Offset + Delta.getSExtValue(), DL, Reads))
        return false;
      continue;
    }
    if (llvm::isa<llvm::BitCastOperator, llvm::AddrSpaceCastOperator>(U)) {
      if (!collectPointerReads(U, Offset, DL, Reads))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

// Adds the bytes of a by-value argument that are read: the extracted members,
// or the whole value for any other use. The reads are always known exactly.
inline bool collectValueReads(const llvm::Argument &A,
                              const llvm::DataLayout &DL, ByteRanges &Reads) {
  for (const llvm::User *U : A.users()) {
    auto *EV = llvm::dyn_cast<llvm::ExtractValueInst>(U);
    if (!EV) {
      Reads.emplace_back(0, DL.getTypeStoreSize(A.getType()));
      continue;
    }
    llvm::Type *T = A.getType();
    int64_t Offset = 0;
    for (unsigned Idx : EV->getIndices()) {
      if (auto *ST = llvm::dyn_cast<llvm::StructType>(T)) {
        Offset += DL.getStructLayout(ST)->getElementOffset(Idx);
        T = ST->getElementType(Idx);
      } else {
        T = T->getArrayElementType();
        Offset += Idx * DL.getTypeAllocSize(T);
      }
    }
    Reads.emplace_back(Offset, Offset + DL.getTypeStoreSize(T));
  }
  return true;
}

} // namespace arg_layout_detail

// Host launch arguments hold each source-level argument in its in-memory
// layout, so a byref or byval argument is flattened from its pointee type.
inline KernelArgLayout getKernelArgLayout(const llvm::Function &F) {
  const llvm::DataLayout &DL = F.getParent()->getDataLayout();
  KernelArgLayout Layout;
  for (const llvm::Argument &A : F.args()) {
    std::string Name =
        A.hasName() ? A.getName().str() : "arg" + std::to_string(A.getArgNo());
    llvm::Type *InMemoryTy = A.getPointeeInMemoryValueType();
    llvm::Type *ArgTy = InMemoryTy ? InMemoryTy : A.getType();
    Layout.ArgNames.push_back(Name);
    Layout.ArgTypes.push_back(arg_layout_detail::typeName(ArgTy) +
                              (InMemoryTy ? " (in memory)" : ""));
    Layout.ArgReadsKnown.push_back(false);
    arg_layout_detail::flattenType(ArgTy, A.getArgNo(), 0, Name, DL, Layout);
  }
  return Layout;
}

// Optimizes the module at O3, with a target machine for the kernel's
// architecture. A byte the optimized kernel never reads cannot affect the
// kernel under any pipeline, so the strongest fixed pipeline gives the most
// precise answer; the record run's PROTEUS_* settings and JIT pass plugins are
// deliberately ignored so profiles do not depend on them.
inline void optimizeAtO3(llvm::Module &M, const std::string &KernelName) {
  static proteus::InitLLVMTargets InitTargets;
  llvm::Function *F = M.getFunction(KernelName);
  if (!F || !F->hasFnAttribute("target-cpu"))
    LOG_FATAL("Kernel " + KernelName +
              " has no target-cpu attribute to optimize its arguments for");
  std::string Arch = F->getFnAttribute("target-cpu").getValueAsString().str();
  proteus::detail::runOptimizationPassPipeline(M, Arch, '3', 3);
}

// Marks the leaves that the kernel cannot read. The captured bitcode is not yet
// optimized, so every argument still has uses that copy it to the stack; the
// module is optimized first, at O3, which removes the
// copies of unread bytes. Uses the analysis cannot follow, such as passing the
// argument to a call that was not inlined, keep every leaf of that argument as
// read.
inline void markUnreadLeaves(llvm::Module &M, const std::string &KernelName,
                             KernelArgLayout &Layout) {
  optimizeAtO3(M, KernelName);
  llvm::Function *F = M.getFunction(KernelName);
  if (!F || F->arg_size() != Layout.ArgNames.size())
    return;
  const llvm::DataLayout &DL = M.getDataLayout();
  for (const llvm::Argument &A : F->args()) {
    arg_layout_detail::ByteRanges Reads;
    bool Known = A.getPointeeInMemoryValueType()
                     ? arg_layout_detail::collectPointerReads(&A, 0, DL, Reads)
                     : arg_layout_detail::collectValueReads(A, DL, Reads);
    Layout.ArgReadsKnown[A.getArgNo()] = Known;
    if (!Known)
      continue;
    for (ArgLeaf &Leaf : Layout.Leaves) {
      if (Leaf.ArgIdx != A.getArgNo())
        continue;
      int64_t Begin = Leaf.Offset, End = Leaf.Offset + Leaf.Size;
      Leaf.IsRead = llvm::any_of(Reads, [&](const auto &R) {
        return R.first < End && Begin < R.second;
      });
    }
  }
}

// Counts launches per value of one leaf, up to Cap distinct values.
class LeafValueStats {
  std::unordered_map<uint64_t, uint64_t> Launches;
  uint64_t UntrackedLaunches = 0;

public:
  void observe(uint64_t Bits, size_t Cap) {
    auto It = Launches.find(Bits);
    if (It != Launches.end())
      It->second++;
    else if (Launches.size() < Cap)
      Launches.emplace(Bits, 1);
    else
      UntrackedLaunches++;
  }

  size_t distinct() const { return Launches.size(); }
  uint64_t untrackedLaunches() const { return UntrackedLaunches; }

  std::vector<std::pair<uint64_t, uint64_t>> top(size_t K) const {
    std::vector<std::pair<uint64_t, uint64_t>> Sorted(Launches.begin(),
                                                      Launches.end());
    std::sort(Sorted.begin(), Sorted.end(), [](const auto &A, const auto &B) {
      return A.second != B.second ? A.second > B.second : A.first < B.first;
    });
    if (Sorted.size() > K)
      Sorted.resize(K);
    return Sorted;
  }
};

class KernelArgProfile {
  static constexpr size_t TopValues = 8;

  KernelArgLayout Layout;
  std::vector<LeafValueStats> Stats;
  size_t Cap;
  uint64_t Launches = 0;

  static uint64_t readLeaf(void **Args, const ArgLeaf &Leaf) {
    uint64_t Bits = 0;
    std::memcpy(&Bits,
                static_cast<const char *>(Args[Leaf.ArgIdx]) + Leaf.Offset,
                Leaf.Size);
    return Bits;
  }

  static int64_t signExtend(uint64_t Bits, unsigned Size) {
    unsigned Shift = 64 - 8 * Size;
    return static_cast<int64_t>(Bits << Shift) >> Shift;
  }

  // JSON has no infinity or NaN, so those are written as strings.
  static llvm::json::Value decodeFloat(double D) {
    if (std::isnan(D))
      return "nan";
    if (std::isinf(D))
      return D > 0 ? "inf" : "-inf";
    return D;
  }

  static llvm::json::Value decode(const ArgLeaf &Leaf, uint64_t Bits) {
    switch (Leaf.Kind) {
    case ArgLeafKind::Integer:
      return signExtend(Bits, Leaf.Size);
    case ArgLeafKind::Float:
      if (Leaf.Size == sizeof(double)) {
        double D;
        std::memcpy(&D, &Bits, sizeof(D));
        return decodeFloat(D);
      }
      if (Leaf.Size == sizeof(float)) {
        float F;
        std::memcpy(&F, &Bits, sizeof(F));
        return decodeFloat(F);
      }
      break;
    case ArgLeafKind::Pointer:
      break;
    }
    std::string Hex;
    llvm::raw_string_ostream(Hex) << llvm::format_hex(Bits, 2 + 2 * Leaf.Size);
    return Hex;
  }

public:
  KernelArgProfile(KernelArgLayout Layout, size_t Cap)
      : Layout(std::move(Layout)), Stats(this->Layout.Leaves.size()), Cap(Cap) {
  }

  void observe(void **Args) {
    Launches++;
    for (size_t I = 0; I < Layout.Leaves.size(); I++)
      Stats[I].observe(readLeaf(Args, Layout.Leaves[I]), Cap);
  }

  llvm::json::Object toJSON() const {
    llvm::json::Array Args;
    for (size_t I = 0; I < Layout.ArgNames.size(); I++)
      Args.push_back(
          llvm::json::Object{{"Name", Layout.ArgNames[I]},
                             {"Type", Layout.ArgTypes[I]},
                             {"ReadsAreKnown", Layout.ArgReadsKnown[I]}});

    llvm::json::Array Leaves;
    for (size_t I = 0; I < Layout.Leaves.size(); I++) {
      const ArgLeaf &Leaf = Layout.Leaves[I];
      const LeafValueStats &S = Stats[I];
      llvm::json::Array Top;
      for (const auto &[Bits, Count] : S.top(TopValues))
        Top.push_back(llvm::json::Object{{"Value", decode(Leaf, Bits)},
                                         {"Launches", Count}});
      Leaves.push_back(
          llvm::json::Object{{"Label", Leaf.Path},
                             {"ArgIndex", Leaf.ArgIdx},
                             {"OffsetBytes", Leaf.Offset},
                             {"SizeBytes", Leaf.Size},
                             {"Type", Leaf.Type},
                             {"IsReadByKernel", Leaf.IsRead},
                             {"NumDistinctValues", S.distinct()},
                             {"UntrackedLaunches", S.untrackedLaunches()},
                             {"TopValues", std::move(Top)}});
    }

    return llvm::json::Object{
        {"Launches", Launches},
        {"Cap", Cap},
        {"Args", std::move(Args)},
        {"Leaves", std::move(Leaves)},
        {"Unsupported", llvm::json::Array(Layout.Unsupported)}};
  }
};

} // namespace mneme
