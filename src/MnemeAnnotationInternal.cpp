#include "mneme/MnemeAnnotationInternal.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {
namespace metadata {
// Indexed by enum value. recorded_execution.py uses the same names.
static const char *const DTypeNames[] = {
    "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f16", "f32", "f64"};
static const char *const ThresholdKindNames[] = {"absolute", "relative"};
static const char *const NormNames[] = {"none", "l1", "l2", "linf"};

Metadata fromBuffer(const char *&Buffer) {
  Metadata Md;
  Md.builtin = util::extractScalar<BuiltinDType>(Buffer);
  Md.threshold = util::extractScalar<double>(Buffer);
  Md.threshold_kind = util::extractScalar<ThresholdKind>(Buffer);
  Md.norm = util::extractScalar<Norm>(Buffer);
  size_t Size = util::extractScalar<size_t>(Buffer);
  if (Size) {
    Md.tag = std::string(Buffer, Size);
    Buffer += Size;
  }
  return Md;
}

bool isDefault(const Metadata &Md) {
  const Metadata Default;
  return Md.builtin == Default.builtin && Md.threshold == Default.threshold &&
         Md.threshold_kind == Default.threshold_kind &&
         Md.norm == Default.norm && Md.tag == Default.tag;
}

llvm::json::Object toJSON(void *DevAddr, const Metadata &Md) {
  llvm::json::Object Entry;
  Entry["DevAddr"] = util::pointerToHexString(DevAddr);
  if (Md.tag)
    Entry["Tag"] = *Md.tag;
  Entry["DType"] = DTypeNames[static_cast<size_t>(Md.builtin)];
  Entry["Threshold"] = Md.threshold;
  Entry["ThresholdKind"] =
      ThresholdKindNames[static_cast<size_t>(Md.threshold_kind)];
  Entry["Norm"] = NormNames[static_cast<size_t>(Md.norm)];
  return Entry;
}

} // namespace metadata
} // namespace mneme
