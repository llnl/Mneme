#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/ADT/Hashing.h>
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

enum class ArgLeafKind { Integer, Float };

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
  if (ElemTy && ElemTy->isIntegerTy(8))
    return;
  if (ElemTy) {
    uint64_t Stride = DL.getTypeAllocSize(ElemTy);
    for (uint64_t I = 0; I < NumElems; I++)
      flattenType(ElemTy, ArgIdx, Offset + I * Stride,
                  Path + "[" + std::to_string(I) + "]", DL, Layout);
    return;
  }

  uint64_t Size = DL.getTypeStoreSize(T);
  std::optional<ArgLeafKind> Kind;
  if (T->isIntegerTy())
    Kind = ArgLeafKind::Integer;
  else if (T->isHalfTy() || T->isBFloatTy() || T->isFloatTy() ||
           T->isDoubleTy())
    Kind = ArgLeafKind::Float;

  if (!Kind || Size == 0 || Size > sizeof(uint64_t))
    return;
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

// Removes the leaves that the kernel cannot read. Unread leaves, such as the
// byte of an empty functor, cannot change the generated code and hold whatever
// the host left there, so profiling them only adds noise. The captured bitcode
// is not yet optimized, so every argument still has uses that copy it to the
// stack; the module is optimized first, at O3, which removes the copies of
// unread bytes. Uses the analysis cannot follow, such as passing the argument
// to a call that was not inlined, keep every leaf of that argument.
inline void removeUnreadLeaves(llvm::Module &M, const std::string &KernelName,
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
    if (!Known) {
      bool HasLeaves = llvm::any_of(Layout.Leaves, [&](const ArgLeaf &Leaf) {
        return Leaf.ArgIdx == A.getArgNo();
      });
      // Printed without the logger so that default builds still tell the user
      // that this argument's leaves may include bytes the kernel never reads.
      if (HasLeaves)
        std::cerr << "[mneme] Argument value profile keeps every leaf of "
                     "argument "
                  << Layout.ArgNames[A.getArgNo()] << " of kernel "
                  << KernelName << " because its reads cannot be followed\n";
      continue;
    }
    llvm::erase_if(Layout.Leaves, [&](const ArgLeaf &Leaf) {
      if (Leaf.ArgIdx != A.getArgNo())
        return false;
      int64_t Begin = Leaf.Offset, End = Leaf.Offset + Leaf.Size;
      return llvm::none_of(Reads, [&](const auto &R) {
        return R.first < End && Begin < R.second;
      });
    });
  }
}

// The launch configuration is profiled as leaves that follow the argument
// leaves, so signatures tie argument values to the launch they came with.
struct LaunchLeaf {
  const char *Label;
  unsigned Size;
  const char *Type;
};

inline constexpr std::array<LaunchLeaf, 7> LaunchLeaves = {{
    {"GridDims.x", 4, "i32"},
    {"GridDims.y", 4, "i32"},
    {"GridDims.z", 4, "i32"},
    {"BlockDims.x", 4, "i32"},
    {"BlockDims.y", 4, "i32"},
    {"BlockDims.z", 4, "i32"},
    {"SharedMem", 8, "i64"},
}};

using LaunchLeafValues = std::array<uint64_t, LaunchLeaves.size()>;

// Maps the values of one leaf to indices in admission order, up to Cap values.
class LeafValueTable {
  std::unordered_map<uint64_t, int32_t> IndexOf;
  std::vector<uint64_t> Values;
  std::vector<uint64_t> Launches;
  uint64_t OverflowLaunches = 0;

public:
  static constexpr int32_t Overflow = -1;

  int32_t observe(uint64_t Bits, size_t Cap) {
    auto It = IndexOf.find(Bits);
    if (It != IndexOf.end()) {
      Launches[It->second]++;
      return It->second;
    }
    if (Values.size() >= Cap) {
      OverflowLaunches++;
      return Overflow;
    }
    auto Index = static_cast<int32_t>(Values.size());
    IndexOf.emplace(Bits, Index);
    Values.push_back(Bits);
    Launches.push_back(1);
    return Index;
  }

  size_t size() const { return Values.size(); }
  uint64_t value(size_t Index) const { return Values[Index]; }
  uint64_t launches(size_t Index) const { return Launches[Index]; }
  uint64_t overflowLaunches() const { return OverflowLaunches; }
};

class KernelArgProfile {
  static constexpr size_t MaxSignatures = 4096;

  struct SignatureHash {
    size_t operator()(const std::vector<int32_t> &Indices) const {
      return llvm::hash_combine_range(Indices.begin(), Indices.end());
    }
  };

  KernelArgLayout Layout;
  // One table per argument leaf, then one per launch leaf.
  std::vector<LeafValueTable> Tables;
  // Maps a signature to its row; rows are numbered in first-seen order.
  std::unordered_map<std::vector<int32_t>, size_t, SignatureHash> SignatureRows;
  std::vector<uint64_t> SignatureLaunches;
  // Reused across launches so a launch with a known signature allocates
  // nothing.
  std::vector<int32_t> Signature;
  size_t Cap;
  uint64_t Launches = 0;
  uint64_t UntrackedLaunches = 0;

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

  static void writeValue(llvm::raw_ostream &OS, ArgLeafKind Kind, unsigned Size,
                         uint64_t Bits) {
    if (Kind == ArgLeafKind::Float)
      OS << '"' << llvm::format_hex(Bits, 2 + 2 * Size) << '"';
    else
      OS << signExtend(Bits, Size);
  }

  template <typename WriteValueT>
  static void writeValues(llvm::json::OStream &J, const LeafValueTable &Table,
                          WriteValueT WriteValue) {
    J.attribute("OverflowLaunches", Table.overflowLaunches());
    J.attributeArray("Values", [&] {
      for (size_t I = 0; I < Table.size(); I++)
        J.rawValue([&](llvm::raw_ostream &OS) {
          OS << '[';
          WriteValue(OS, Table.value(I));
          OS << ',' << Table.launches(I) << ']';
        });
    });
  }

  void writeLeaves(llvm::json::OStream &J) const {
    size_t NumArgLeaves = Layout.Leaves.size();
    for (size_t I = 0; I < NumArgLeaves; I++) {
      const ArgLeaf &Leaf = Layout.Leaves[I];
      J.object([&] {
        J.attribute("Label", Leaf.Path);
        J.attribute("ArgIndex", Leaf.ArgIdx);
        J.attribute("OffsetBytes", Leaf.Offset);
        J.attribute("SizeBytes", Leaf.Size);
        J.attribute("Type", Leaf.Type);
        writeValues(J, Tables[I], [&](llvm::raw_ostream &OS, uint64_t Bits) {
          writeValue(OS, Leaf.Kind, Leaf.Size, Bits);
        });
      });
    }
    for (size_t I = 0; I < LaunchLeaves.size(); I++) {
      const LaunchLeaf &Leaf = LaunchLeaves[I];
      J.object([&] {
        J.attribute("Label", Leaf.Label);
        J.attribute("ArgIndex", nullptr);
        J.attribute("OffsetBytes", nullptr);
        J.attribute("SizeBytes", Leaf.Size);
        J.attribute("Type", Leaf.Type);
        writeValues(J, Tables[NumArgLeaves + I],
                    [](llvm::raw_ostream &OS, uint64_t Bits) { OS << Bits; });
      });
    }
  }

  void writeSignatures(llvm::json::OStream &J) const {
    std::vector<const std::vector<int32_t> *> Rows(SignatureRows.size());
    for (const auto &Entry : SignatureRows)
      Rows[Entry.second] = &Entry.first;
    for (size_t Row = 0; Row < Rows.size(); Row++)
      J.rawValue([&](llvm::raw_ostream &OS) {
        OS << "[[";
        for (size_t I = 0; I < Rows[Row]->size(); I++)
          OS << (I ? "," : "") << (*Rows[Row])[I];
        OS << "]," << SignatureLaunches[Row] << ']';
      });
  }

public:
  KernelArgProfile(KernelArgLayout Layout, size_t Cap)
      : Layout(std::move(Layout)),
        Tables(this->Layout.Leaves.size() + LaunchLeaves.size()),
        Signature(Tables.size()), Cap(Cap) {}

  void observe(void **Args, const LaunchLeafValues &LaunchValues) {
    Launches++;
    size_t NumArgLeaves = Layout.Leaves.size();
    for (size_t I = 0; I < NumArgLeaves; I++)
      Signature[I] = Tables[I].observe(readLeaf(Args, Layout.Leaves[I]), Cap);
    for (size_t I = 0; I < LaunchValues.size(); I++)
      Signature[NumArgLeaves + I] =
          Tables[NumArgLeaves + I].observe(LaunchValues[I], Cap);

    auto It = SignatureRows.find(Signature);
    if (It != SignatureRows.end()) {
      SignatureLaunches[It->second]++;
      return;
    }
    if (SignatureRows.size() >= MaxSignatures) {
      UntrackedLaunches++;
      return;
    }
    SignatureRows.emplace(Signature, SignatureLaunches.size());
    SignatureLaunches.push_back(1);
  }

  // Writes the launch counts into the JSON object that is open in J.
  void writeCounts(llvm::json::OStream &J) const {
    J.attribute("Launches", Launches);
    J.attribute("UntrackedLaunches", UntrackedLaunches);
  }

  // Writes the arguments, leaves and signatures into the JSON object that is
  // open in J.
  void writeTables(llvm::json::OStream &J) const {
    J.attributeArray("Args", [&] {
      for (size_t I = 0; I < Layout.ArgNames.size(); I++)
        J.object([&] {
          J.attribute("Name", Layout.ArgNames[I]);
          J.attribute("Type", Layout.ArgTypes[I]);
        });
    });
    J.attributeArray("Leaves", [&] { writeLeaves(J); });
    J.attributeArray("Signatures", [&] { writeSignatures(J); });
  }
};

} // namespace mneme
