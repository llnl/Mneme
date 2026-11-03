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
#include <llvm/IR/Module.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/xxhash.h>

#include "mneme/DeviceTraits.hpp"

namespace mneme {

enum class ArgLeafKind { Integer, Float, Pointer };

inline const char *getArgLeafKindName(ArgLeafKind Kind) {
  switch (Kind) {
  case ArgLeafKind::Integer:
    return "int";
  case ArgLeafKind::Float:
    return "float";
  case ArgLeafKind::Pointer:
    return "ptr";
  }
  return "int";
}

// A scalar inside a kernel argument, found at Offset bytes past Args[ArgIdx].
struct ArgLeaf {
  unsigned ArgIdx;
  uint64_t Offset;
  unsigned Size;
  ArgLeafKind Kind;
  std::string Path;
  std::string Type;
};

struct KernelArgLayout {
  llvm::SmallVector<std::string> ArgNames;
  llvm::SmallVector<std::string> ArgTypes;
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
    arg_layout_detail::flattenType(ArgTy, A.getArgNo(), 0, Name, DL, Layout);
  }
  return Layout;
}

// Counts launches per value of one leaf, up to Cap distinct values.
class LeafValueStats {
  std::unordered_map<uint64_t, uint64_t> Launches;
  uint64_t UntrackedLaunches = 0;
  uint64_t Changes = 0;
  std::optional<uint64_t> Previous;

public:
  void observe(uint64_t Bits, size_t Cap) {
    if (Previous && *Previous != Bits)
      Changes++;
    Previous = Bits;

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
  uint64_t changes() const { return Changes; }

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

// Counts launches per distinct tuple of values, up to Cap distinct tuples.
class TupleStats {
  std::unordered_map<uint64_t, uint64_t> Launches;
  uint64_t UntrackedLaunches = 0;

public:
  void observe(uint64_t Hash, size_t Cap) {
    auto It = Launches.find(Hash);
    if (It != Launches.end())
      It->second++;
    else if (Launches.size() < Cap)
      Launches.emplace(Hash, 1);
    else
      UntrackedLaunches++;
  }

  llvm::json::Object toJSON(size_t Cap) const {
    uint64_t MostLaunches = 0;
    for (const auto &Entry : Launches)
      MostLaunches = std::max(MostLaunches, Entry.second);
    return llvm::json::Object{{"Distinct", Launches.size()},
                              {"Saturated", Launches.size() >= Cap},
                              {"UntrackedLaunches", UntrackedLaunches},
                              {"MostCommonLaunches", MostLaunches}};
  }
};

class KernelArgProfile {
  static constexpr size_t TopValues = 8;

  KernelArgLayout Layout;
  std::vector<LeafValueStats> Stats;
  TupleStats ScalarTuples;
  TupleStats ScalarBlockTuples;
  size_t Cap;
  uint64_t Launches = 0;

  static uint64_t readLeaf(void **Args, const ArgLeaf &Leaf) {
    uint64_t Bits = 0;
    std::memcpy(&Bits,
                static_cast<const char *>(Args[Leaf.ArgIdx]) + Leaf.Offset,
                Leaf.Size);
    return Bits;
  }

  static uint64_t hashWords(llvm::ArrayRef<uint64_t> Words) {
    return llvm::xxh3_64bits(
        llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Words.data()),
                                Words.size() * sizeof(uint64_t)));
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

  void observe(void **Args, const dim3 &BlockDim) {
    Launches++;
    llvm::SmallVector<uint64_t, 32> Scalars;
    for (size_t I = 0; I < Layout.Leaves.size(); I++) {
      const ArgLeaf &Leaf = Layout.Leaves[I];
      uint64_t Bits = readLeaf(Args, Leaf);
      Stats[I].observe(Bits, Cap);
      if (Leaf.Kind != ArgLeafKind::Pointer)
        Scalars.push_back(Bits);
    }
    ScalarTuples.observe(hashWords(Scalars), Cap);
    Scalars.append({BlockDim.x, BlockDim.y, BlockDim.z});
    ScalarBlockTuples.observe(hashWords(Scalars), Cap);
  }

  llvm::json::Object toJSON() const {
    llvm::json::Array Args;
    for (size_t I = 0; I < Layout.ArgNames.size(); I++)
      Args.push_back(llvm::json::Object{{"Name", Layout.ArgNames[I]},
                                        {"Type", Layout.ArgTypes[I]}});

    llvm::json::Array Leaves;
    for (size_t I = 0; I < Layout.Leaves.size(); I++) {
      const ArgLeaf &Leaf = Layout.Leaves[I];
      const LeafValueStats &S = Stats[I];
      llvm::json::Array Top;
      for (const auto &[Bits, Count] : S.top(TopValues))
        Top.push_back(llvm::json::Object{{"Value", decode(Leaf, Bits)},
                                         {"Launches", Count}});
      Leaves.push_back(
          llvm::json::Object{{"Path", Leaf.Path},
                             {"Arg", Leaf.ArgIdx},
                             {"Offset", Leaf.Offset},
                             {"Size", Leaf.Size},
                             {"Kind", getArgLeafKindName(Leaf.Kind)},
                             {"Type", Leaf.Type},
                             {"Distinct", S.distinct()},
                             {"Saturated", S.distinct() >= Cap},
                             {"UntrackedLaunches", S.untrackedLaunches()},
                             {"Changes", S.changes()},
                             {"Top", std::move(Top)}});
    }

    return llvm::json::Object{
        {"Launches", Launches},
        {"Cap", Cap},
        {"Args", std::move(Args)},
        {"Leaves", std::move(Leaves)},
        {"Unsupported", llvm::json::Array(Layout.Unsupported)},
        {"ScalarTuples", ScalarTuples.toJSON(Cap)},
        {"ScalarBlockTuples", ScalarBlockTuples.toJSON(Cap)}};
  }
};

} // namespace mneme
