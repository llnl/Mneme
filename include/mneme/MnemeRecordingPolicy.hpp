#pragma once

#include <cstdint>
#include <regex>
#include <string>
#include <vector>

namespace mneme {

class Config;

/* for skips and max recordings how do we determine what a unique launch is */
enum class Deduplication { Configuration, Invocations };

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

  KernelRecordingPolicy resolve(const std::string &DemangledName) const;
  Deduplication deduplication() const { return Dedup; }
  bool isRecordingEnabledForCurrentRank() const { return EnabledThisRank; }

private:
  struct Rule {
    std::regex Pattern;
    KernelRecordingPolicy Policy;
  };

  KernelRecordingPolicy Defaults;
  std::vector<Rule> Rules;
  Deduplication Dedup = Deduplication::Configuration;
  bool EnabledThisRank = true;

  static RecordingPolicy load(const Config &Conf);
};

} // namespace mneme
