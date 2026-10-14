#pragma once
#include "mneme/MnemeAnnotation.hpp"
#include <llvm/Support/JSON.h>

namespace mneme {
namespace metadata {
// Reads the metadata that older snapshot layouts stored after each blob.
Metadata fromBuffer(const char *&Buffer);
bool isDefault(const Metadata &Md);
// An entry of a kernel instance's "Compare" list in the record database.
llvm::json::Object toJSON(void *DevAddr, const Metadata &Md);
} // namespace metadata
} // namespace mneme
