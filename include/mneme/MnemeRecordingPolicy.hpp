#pragma once

#include <cstdint>
#include <regex>
#include <string>
#include <vector>

namespace llvm::json {
class Value;
}

namespace mneme {

class Config;
namespace config_detail {
struct RecordingRankPolicy;
}

struct KernelRecordingPolicy {
  uint64_t Skip = 0;
  uint64_t Every = 1;
  uint64_t MaxRecords = 4;

  bool selects(uint64_t LaunchNumber) const {
    return MaxRecords != 0 && LaunchNumber > Skip &&
           (LaunchNumber - Skip - 1) % Every == 0;
  }
};

// immutable after loading.
class RecordingPolicy {
public:
  static const RecordingPolicy &get();
  static RecordingPolicy createFromEnvironment();
  static RecordingPolicy fromJSON(const std::string &Text);

  KernelRecordingPolicy resolve(const std::string &DemangledName,
                                const std::string &MangledName = "") const;
  bool isRecordingEnabledForCurrentRank() const { return EnabledThisRank; }

private:
  struct Rule {
    std::regex Pattern;
    KernelRecordingPolicy Policy;
  };

  KernelRecordingPolicy Defaults;
  std::vector<Rule> Rules;
  bool EnabledThisRank = true;
  bool MatchMangledNames = false;

  static RecordingPolicy load(const Config &Conf);
  static RecordingPolicy
  fromJSONValue(const llvm::json::Value &Value,
                const config_detail::RecordingRankPolicy &Ranks);
};

} // namespace mneme
