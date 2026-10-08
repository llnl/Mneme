#pragma once

#include "mneme/MnemeRank.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mneme {

enum class LogLevel { Trace, Debug, Info, Warn, Error, Critical, Off };
enum class EpilogueSnapshotType { Bytes, Diff, Best };
enum class FilterMode { Default, Scoped, Config };

namespace config_detail {

inline std::optional<int> parseInteger(std::string_view Text) {
  if (Text.empty())
    return std::nullopt;

  std::string NullTerminated(Text);
  errno = 0;
  char *End = nullptr;
  long Parsed = std::strtol(NullTerminated.c_str(), &End, 10);
  if (errno == ERANGE || End == NullTerminated.c_str() || *End != '\0' ||
      Parsed > INT_MAX || Parsed < INT_MIN)
    return std::nullopt;

  return static_cast<int>(Parsed);
}

inline void warnMalformedEnvironmentValue(const char *Description,
                                          const char *Name,
                                          std::string_view Value,
                                          const char *Suffix = "") {
  std::ostringstream Message;
  Message << "[mneme] Ignoring malformed " << Description << " " << Name << "='"
          << Value << "'";
  if (Suffix && Suffix[0] != '\0')
    Message << " " << Suffix;

  std::cerr << Message.str() << "\n";
}

inline std::optional<std::string> getEnvOrDefaultString(const char *VarName) {
  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return std::nullopt;

  return std::string(EnvValue);
}

inline std::optional<long> getEnvOrDefaultPageSizeGiB(const char *VarName) {
  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return std::nullopt;

  return std::atol(EnvValue);
}

inline uint64_t pageSizeGiBToBytes(long PageSizeGiB) {
  return static_cast<uint64_t>(PageSizeGiB * 1024L * 1024L * 1024L);
}

inline LogLevel getEnvOrDefaultLogLevel(const char *VarName, LogLevel Default) {
  auto EnvValue = getEnvOrDefaultString(VarName);
  if (!EnvValue)
    return Default;

  if (*EnvValue == "trace")
    return LogLevel::Trace;
  if (*EnvValue == "debug")
    return LogLevel::Debug;
  if (*EnvValue == "info")
    return LogLevel::Info;
  if (*EnvValue == "warn")
    return LogLevel::Warn;
  if (*EnvValue == "error")
    return LogLevel::Error;
  if (*EnvValue == "critical")
    return LogLevel::Critical;
  if (*EnvValue == "off")
    return LogLevel::Off;

  return LogLevel::Info;
}

inline bool getEnvOrDefaultBool(const char *VarName, bool Default) {
  auto EnvValue = getEnvOrDefaultString(VarName);
  if (!EnvValue)
    return Default;

  if (*EnvValue == "1")
    return true;
  if (*EnvValue == "0")
    return false;

  warnMalformedEnvironmentValue("environment variable", VarName, *EnvValue,
                                "; expected 0 or 1");
  return Default;
}

inline EpilogueSnapshotType
getEnvOrDefaultEpilogueSnapshotType(const char *VarName,
                                    EpilogueSnapshotType Default) {
  auto EnvValue = getEnvOrDefaultString(VarName);
  if (!EnvValue)
    return Default;

  if (*EnvValue == "bytes")
    return EpilogueSnapshotType::Bytes;
  if (*EnvValue == "diff")
    return EpilogueSnapshotType::Diff;
  if (*EnvValue == "best")
    return EpilogueSnapshotType::Best;

  throw std::runtime_error("Invalid MNEME_EPILOGUE_TYPE value '" + *EnvValue +
                           "'. Expected 'bytes', 'diff', or 'best'.");
}

inline FilterMode getFilterMode() {
  auto Value = getEnvOrDefaultString("MNEME_FILTER_MODE");
  if (!Value || *Value == "default")
    return FilterMode::Default;
  if (*Value == "scoped")
    return FilterMode::Scoped;
  if (*Value == "config")
    return FilterMode::Config;

  throw std::runtime_error("Invalid MNEME_FILTER_MODE value '" + *Value +
                           "'. Expected 'default', 'scoped', or 'config'.");
}

inline void rejectDeprecatedRecordingControls() {
  std::string Controls;
  for (const char *Name : {"MNEME_RR_KERNELS", "MNEME_SKIP_RECORDINGS",
                           "MNEME_MAX_RECORDINGS"}) {
    if (std::getenv(Name)) {
      if (!Controls.empty())
        Controls += ", ";
      Controls += Name;
    }
  }
  
  if (!Controls.empty())
    throw std::runtime_error(
        Controls + " are deprecated and no longer supported. Unset them and "
        "use --filter-config with recording.defaults.skip/max_records and "
        "kernel_overrides (or MNEME_FILTER_MODE=config and MNEME_FILTER_CONFIG "
        "for direct preloading).");
}

inline bool defaultRecordingPolicy(const std::optional<int> &DistributedRank) {
  if (!DistributedRank)
    return true;
  return *DistributedRank == 0;
}

inline std::optional<std::set<int>>
parseRecordRanksList(const std::string &Value) {
  std::set<int> Parsed;
  size_t Start = 0;
  while (Start <= Value.size()) {
    size_t Comma = Value.find(',', Start);
    std::string Token = Value.substr(
        Start, Comma == std::string::npos ? std::string::npos : Comma - Start);

    auto ParsedRank = parseInteger(Token);
    if (!ParsedRank || *ParsedRank < 0)
      return std::nullopt;

    Parsed.insert(*ParsedRank);
    if (Comma == std::string::npos)
      break;
    Start = Comma + 1;
  }

  return Parsed;
}

inline bool computeRecordingEnabledForCurrentRank() {
  auto DistributedRank = detectDistributedRank();
  const char *EnvValue = std::getenv("MNEME_RECORD_RANKS");
  if (!EnvValue || EnvValue[0] == '\0')
    return defaultRecordingPolicy(DistributedRank);

  std::string Value(EnvValue);
  std::string Lower = Value;
  std::transform(Lower.begin(), Lower.end(), Lower.begin(),
                 [](unsigned char C) { return std::tolower(C); });
  if (Lower == "all")
    return true;

  auto ParsedRanks = parseRecordRanksList(Value);
  if (!ParsedRanks) {
    warnMalformedEnvironmentValue("environment variable", "MNEME_RECORD_RANKS",
                                  Value,
                                  "; falling back to default rank policy");
    return defaultRecordingPolicy(DistributedRank);
  }

  return ParsedRanks->count(DistributedRank.value_or(0)) > 0;
}

} // namespace config_detail

class Config {
public:
  static Config &get() {
    static Config Conf;
    return Conf;
  }

  static Config createFromEnvironment() { return Config(); }

  const FilterMode RecordingFilterMode;
  const std::optional<std::string> FilterConfigPath;
  const std::optional<long> PageSizeGiB;
  const LogLevel MnemeLogLevel;
  const EpilogueSnapshotType EpilogueType;
  const bool CopySource;

  bool isRecordingEnabledForCurrentRank() const {
    return RecordingEnabledThisRank;
  }

  std::filesystem::path getDataDirectory() const {
    auto Dir = MnemeDataDir.value_or(std::filesystem::current_path().string());
    std::filesystem::path Path(Dir);
    if (!std::filesystem::is_directory(Path)) {
      throw std::runtime_error("Path :" + Path.string() + " does not exist.\n");
    }
    return std::filesystem::absolute(Path);
  }

  uint64_t getPageSizeBytesOrDefault(long DefaultPageSizeGiB) const {
    return config_detail::pageSizeGiBToBytes(
        PageSizeGiB.value_or(DefaultPageSizeGiB));
  }

  std::optional<std::string> getLogDirectory() const {
    if (MnemeLogDir && !std::filesystem::exists(*MnemeLogDir)) {
      throw std::runtime_error("'MNEME_LOG_DIR' directory does not exist\n");
    }
    return MnemeLogDir;
  }

private:
  const std::optional<std::string> MnemeDataDir;
  const std::optional<std::string> MnemeLogDir;
  const bool RecordingEnabledThisRank;

  Config()
      : RecordingFilterMode(config_detail::getFilterMode()),
        FilterConfigPath(config_detail::getEnvOrDefaultString("MNEME_FILTER_CONFIG")),
        PageSizeGiB(
            config_detail::getEnvOrDefaultPageSizeGiB("MNEME_PAGE_SIZE")),
        MnemeLogLevel(config_detail::getEnvOrDefaultLogLevel(
            "MNEME_LOG_LEVEL", LogLevel::Critical)),
        EpilogueType(config_detail::getEnvOrDefaultEpilogueSnapshotType(
            "MNEME_EPILOGUE_TYPE", EpilogueSnapshotType::Diff)),
        CopySource(
            config_detail::getEnvOrDefaultBool("MNEME_COPY_SOURCE", false)),
        MnemeDataDir(config_detail::getEnvOrDefaultString("MNEME_DATA_DIR")),
        MnemeLogDir(config_detail::getEnvOrDefaultString("MNEME_LOG_DIR")),
        RecordingEnabledThisRank(
            config_detail::computeRecordingEnabledForCurrentRank()) {
    
    config_detail::rejectDeprecatedRecordingControls();
    if (RecordingFilterMode == FilterMode::Config &&
        (!FilterConfigPath || FilterConfigPath->empty()))
      throw std::runtime_error("MNEME_FILTER_MODE=config requires MNEME_FILTER_CONFIG");
    
    if (RecordingFilterMode != FilterMode::Config && FilterConfigPath)
      throw std::runtime_error("MNEME_FILTER_CONFIG requires MNEME_FILTER_MODE=config");
  }
};

} // namespace mneme
