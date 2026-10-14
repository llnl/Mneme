#include <cstdint>

#include "mneme/MnemeAnnotation.hpp"

namespace mneme {

struct CompareResult {
  int AnyFail;     // 0/1
  uint64_t FirstBadIdx; // NumBytes if none
  double MaxErr;   // max element error observed
  double Agg;      // L1/L2/Linf aggregation (meaning depends on norm)
};

CompareResult compareDeviceBlobs(const char *Blob1, const char *Blob2,
                                 uint64_t NumBytes, Metadata Md);
} // namespace mneme
