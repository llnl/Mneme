#pragma once
#include <cstddef>
#include <string>
#include <utility>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {

// Global-variable record prefix: | Name length | Name | Size | DevAddr |.
struct GlobalVarHeader {
  std::string Name;
  size_t Size;
  void *DevAddr;

  void write(llvm::raw_ostream &OS) const {
    util::writeScalar(OS, Name.size());
    util::writeBytes(OS, llvm::StringRef(Name));
    util::writeScalar(OS, Size);
    util::writeScalar(OS, DevAddr);
  }

  static GlobalVarHeader read(const char *&Buffer) {
    std::string Name = util::readSizedString(Buffer);
    size_t Size = util::extractScalar<size_t>(Buffer);
    void *DevAddr = util::extractScalar<void *>(Buffer);
    return GlobalVarHeader{std::move(Name), Size, DevAddr};
  }

  size_t serializedSize() const {
    return sizeof(size_t) + Name.size() + sizeof(Size) + sizeof(DevAddr);
  }
};

// Device-memory blob record prefix: | ActualSize | Size | DevAddr |.
struct BlobHeader {
  size_t ActualSize;
  size_t Size;
  void *DevAddr;

  void write(llvm::raw_ostream &OS) const {
    util::writeScalar(OS, ActualSize);
    util::writeScalar(OS, Size);
    util::writeScalar(OS, DevAddr);
  }

  static BlobHeader read(const char *&Buffer) {
    size_t ActualSize = util::extractScalar<size_t>(Buffer);
    size_t Size = util::extractScalar<size_t>(Buffer);
    void *DevAddr = util::extractScalar<void *>(Buffer);
    return BlobHeader{ActualSize, Size, DevAddr};
  }

  static constexpr size_t serializedSize() {
    return 2 * sizeof(size_t) + sizeof(void *);
  }
};

} // namespace mneme
