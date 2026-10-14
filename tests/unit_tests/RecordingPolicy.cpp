#include "mneme/MnemeRecordingPolicy.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

using namespace mneme;

void expect(bool Condition, const std::string &Message) {
  if (!Condition) {
    std::cerr << Message << '\n';
    std::exit(1);
  }
}

void invalid(const std::string &JSON, const std::string &Field) {
  try {
    RecordingPolicy::fromJSON(JSON);
  } catch (const std::runtime_error &Error) {
    expect(std::string(Error.what()).find(Field) != std::string::npos,
           Error.what());
    return;
  }
  expect(false, "Expected rejection: " + JSON);
}

int main() {
  unsetenv("MNEME_FILTER_MODE");
  unsetenv("MNEME_FILTER_CONFIG");
  unsetenv("MNEME_RECORD_RANKS");
  unsetenv("MNEME_SKIP_RECORDINGS");
  unsetenv("MNEME_MAX_RECORDINGS");
  unsetenv("MNEME_RR_KERNELS");
  setenv("FLUX_TASK_RANK", "0", 1);

  auto Builtin = RecordingPolicy::createFromEnvironment();
  auto Default = Builtin.resolve("foo()");
  expect(Default.Skip == 0 && Default.Every == 1 && Default.MaxRecords == 4,
         "Built-in defaults");

  auto Large = RecordingPolicy::fromJSON(R"({"recording": {
    "defaults": {
      "skip": 4294967296,
      "every": 4294967297,
      "max_records": 4294967298
    }
  }})")
                   .resolve("foo()");
  expect(Large.Skip == 4294967296ULL && Large.Every == 4294967297ULL &&
             Large.MaxRecords == 4294967298ULL && Large.selects(4294967297ULL),
         "Counts and limits retain 64-bit values");

  setenv("MNEME_SKIP_RECORDINGS", "2", 1);
  setenv("MNEME_MAX_RECORDINGS", "3", 1);
  setenv("MNEME_RR_KERNELS", "^_Z3foov$|bar", 1);
  auto Environment = RecordingPolicy::createFromEnvironment();
  auto Selected = Environment.resolve("foo()", "_Z3foov");
  expect(
      Selected.Skip == 2 && Selected.MaxRecords == 3 && Selected.selects(3),
      "Environment controls use the policy parser and support mangled names");
  expect(Environment.resolve("bar()").MaxRecords == 3 &&
             Environment.resolve("other()").MaxRecords == 0,
         "Environment regex allowlist also matches demangled names");

  setenv("MNEME_SKIP_RECORDINGS", "invalid", 1);
  bool BadCount = false;
  try {
    RecordingPolicy::createFromEnvironment();
  } catch (const std::runtime_error &Error) {
    BadCount = std::string(Error.what()).find("MNEME_SKIP_RECORDINGS") !=
               std::string::npos;
  }
  expect(BadCount, "Invalid active environment count fails");

  setenv("MNEME_FILTER_MODE", "scoped", 1);
  setenv("MNEME_RR_KERNELS", "[", 1);
  std::ostringstream Warning;
  auto *Saved = std::cerr.rdbuf(Warning.rdbuf());
  auto Scoped = RecordingPolicy::createFromEnvironment();
  std::cerr.rdbuf(Saved);
  expect(Scoped.resolve("foo()").Skip == 0 &&
             Scoped.resolve("foo()").MaxRecords == 4,
         "Scoped mode ignores even malformed controls");
  expect(Warning.str().find("ignored") != std::string::npos &&
             Warning.str().find("MNEME_RR_KERNELS") != std::string::npos &&
             Warning.str().find("MNEME_SKIP_RECORDINGS") != std::string::npos &&
             Warning.str().find("MNEME_MAX_RECORDINGS") != std::string::npos,
         "One aggregated warning names ignored controls");

  unsetenv("MNEME_FILTER_MODE");
  unsetenv("MNEME_SKIP_RECORDINGS");
  unsetenv("MNEME_MAX_RECORDINGS");
  unsetenv("MNEME_RR_KERNELS");

  auto Policy = RecordingPolicy::fromJSON(R"({"recording": {
    "defaults": {"skip": 1, "every": 10, "max_records": 3, "ranks": [0, 2]},
    "kernel_overrides": [
      {"match": {"demangled_regex": "^foo\\("}, "skip": 5},
      {"match": {"demangled_regex": "foo"}, "max_records": 99},
      {"match": {"demangled_regex": "bar"}, "max_records": 0}
    ]}})");
  auto Foo = Policy.resolve("foo()");
  expect(Foo.Skip == 5 && Foo.Every == 10 && Foo.MaxRecords == 3,
         "First match inherits defaults without cascading");
  expect(Policy.resolve("other()").Skip == 1, "Unmatched kernel uses defaults");
  expect(!Policy.resolve("bar()").selects(2), "Zero max disables captures");

  auto Other = Policy.resolve("other()");
  expect(!Other.selects(1) && Other.selects(2) && !Other.selects(3) &&
             Other.selects(12) && Other.selects(22),
         "Sampling is one-based");
  expect(Policy.isRecordingEnabledForCurrentRank(), "Selected rank");

  invalid("{", "document");
  invalid("[]", "document");
  invalid("{}", "recording");
  invalid(R"({"recording": null})", "recording");
  invalid(R"({"recording": {}, "extra": 1})", "extra");
  invalid(R"({"recording": {"deduplication": "configuration"}})",
          "recording.deduplication");
  invalid(R"({"recording": {"deduplication": "invocations"}})",
          "recording.deduplication");
  invalid(R"({"recording": {"kernel_overrides": {}}})", "kernel_overrides");
  invalid(R"({"recording": {"defaults": {"typo": 1}}})", "defaults.typo");

  for (const auto &Value :
       {"true", "false", "null", "-1", "1.0", "1.5", "\"1\""}) {
    for (const auto &Field : {"skip", "every", "max_records"})
      invalid(std::string("{\"recording\":{\"defaults\":{\"") + Field +
                  "\":" + Value + "}}}",
              Field);
  }
  invalid(R"({"recording": {"defaults": {"every": 0}}})", "every");

  for (const auto &Value :
       {"\"0,2\"", "[true]", "[-1]", "[1.0]", "[2147483648]"})
    invalid(std::string("{\"recording\":{\"defaults\":{\"ranks\":") + Value +
                "}}}",
            "ranks");

  for (const auto &Override :
       {R"({"ranks": [0]})", R"({"rank": 0})",
        R"({"deduplication": "invocations"})", R"({"skip": 1})",
        R"({"match": {}})", R"({"match": {"demangled_regex": 3}})",
        R"({"match": {"demangled_regex": "["}})",
        R"({"match": {"mangled_regex": "foo"}})",
        R"({"match": {"demangled_regex": "foo"}, "every": 0})"})
    invalid(std::string("{\"recording\":{\"kernel_overrides\":[") + Override +
                "]}}",
            "kernel_overrides[0]");

  const std::string RankPolicy =
      R"({"recording":{"defaults":{"ranks":[0,2]}}})";
  setenv("FLUX_TASK_RANK", "1", 1);
  expect(
      !RecordingPolicy::fromJSON(RankPolicy).isRecordingEnabledForCurrentRank(),
      "Excluded rank");
  expect(!RecordingPolicy::fromJSON(R"({"recording":{}})")
              .isRecordingEnabledForCurrentRank(),
         "Default rank exclusion");
  expect(
      RecordingPolicy::fromJSON(R"({"recording":{"defaults":{"ranks":"all"}}})")
          .isRecordingEnabledForCurrentRank(),
      "All ranks");
  expect(
      !RecordingPolicy::fromJSON(R"({"recording":{"defaults":{"ranks":[]}}})")
           .isRecordingEnabledForCurrentRank(),
      "No ranks");

  setenv("MNEME_RECORD_RANKS", "", 1);
  setenv("FLUX_TASK_RANK", "0", 1);
  expect(
      !RecordingPolicy::fromJSON(R"({"recording":{"defaults":{"ranks":[]}}})")
           .isRecordingEnabledForCurrentRank(),
      "Empty environment rank setting preserves disabled config ranks");
  setenv("FLUX_TASK_RANK", "1", 1);
  expect(
      RecordingPolicy::fromJSON(R"({"recording":{"defaults":{"ranks":[1]}}})")
          .isRecordingEnabledForCurrentRank(),
      "Empty environment rank setting preserves included config rank");

  setenv("MNEME_RECORD_RANKS", "1", 1);
  expect(
      RecordingPolicy::fromJSON(RankPolicy).isRecordingEnabledForCurrentRank(),
      "Environment overrides YAML ranks");

  setenv("MNEME_RECORD_RANKS", "0", 1);
  expect(!RecordingPolicy::fromJSON(
              R"({"recording":{"defaults":{"ranks":"all"}}})")
              .isRecordingEnabledForCurrentRank(),
         "Environment can exclude YAML rank");

  // Validate the whole document even when the rank is excluded.
  invalid(
      R"({"recording":{"defaults":{"ranks":[]},"kernel_overrides":[{"ranks":[1]}]}})",
      "process-wide");

  auto Path = std::filesystem::temp_directory_path() /
              ("mneme-policy-" + std::to_string(getpid()) + ".json");
  {
    std::ofstream File(Path);
    File << R"({"recording":{"defaults":{"skip":7}}})";
  }
  setenv("MNEME_FILTER_MODE", "config", 1);
  setenv("MNEME_FILTER_CONFIG", Path.c_str(), 1);
  expect(RecordingPolicy::createFromEnvironment().resolve("foo()").Skip == 7,
         "JSON file transport");

  setenv("MNEME_SKIP_RECORDINGS", "invalid", 1);
  setenv("MNEME_RR_KERNELS", "[", 1);
  Warning.str("");
  Saved = std::cerr.rdbuf(Warning.rdbuf());
  auto Configured = RecordingPolicy::createFromEnvironment();
  std::cerr.rdbuf(Saved);
  expect(Configured.resolve("foo()").Skip == 7 &&
             Warning.str().find("ignored") != std::string::npos,
         "Config ignores environment controls and keeps file settings");

  unsetenv("MNEME_SKIP_RECORDINGS");
  unsetenv("MNEME_RR_KERNELS");
  setenv("MNEME_FILTER_MODE", "default", 1);
  expect(RecordingPolicy::createFromEnvironment().resolve("foo()").Skip == 7,
         "Default mode loads CLI-generated JSON through the same parser");

  std::filesystem::remove(Path);
  bool Threw = false;
  try {
    RecordingPolicy::createFromEnvironment();
  } catch (const std::runtime_error &Error) {
    Threw = std::string(Error.what()).find("MNEME_FILTER_CONFIG") !=
            std::string::npos;
  }
  expect(Threw, "Missing JSON file must fail");
}
