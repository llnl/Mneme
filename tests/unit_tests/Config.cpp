#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeRank.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace mneme;

namespace {

constexpr uint64_t GiB = 1024L * 1024L * 1024L;

void expect(bool Condition, const std::string &Message) {
  if (!Condition) {
    std::cerr << Message << "\n";
    std::exit(1);
  }
}

void clearMnemeEnv() {
  unsetenv("MNEME_RR_KERNELS");
  unsetenv("MNEME_FILTER_MODE");
  unsetenv("MNEME_FILTER_CONFIG");
  unsetenv("MNEME_DATA_DIR");
  unsetenv("MNEME_MAX_RECORDINGS");
  unsetenv("MNEME_SKIP_RECORDINGS");
  unsetenv("MNEME_PAGE_SIZE");
  unsetenv("MNEME_LOG_LEVEL");
  unsetenv("MNEME_LOG_DIR");
  unsetenv("MNEME_EPILOGUE_TYPE");
  unsetenv("MNEME_COPY_SOURCE");
  unsetenv("MNEME_RECORD_RANKS");
  unsetenv("FLUX_TASK_RANK");
  unsetenv("OMPI_COMM_WORLD_RANK");
  unsetenv("PMI_RANK");
  unsetenv("MPI_RANK");
  unsetenv("SLURM_PROCID");
  unsetenv("JSM_NAMESPACE_RANK");
  unsetenv("PMIX_RANK");
  unsetenv("PBS_TASKNUM");
  unsetenv("FLUX_JOB_SIZE");
  unsetenv("FLUX_NTASKS");
  unsetenv("OMPI_COMM_WORLD_SIZE");
  unsetenv("PMI_SIZE");
  unsetenv("SLURM_NTASKS");
  unsetenv("PMIX_SIZE");
}

std::filesystem::path makeTempDir() {
  auto Path = std::filesystem::temp_directory_path() /
              ("mneme-config-test-" + std::to_string(getpid()));
  std::filesystem::create_directories(Path);
  return Path;
}

} // namespace

int main() {
  clearMnemeEnv();
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.RecordingFilterMode == FilterMode::Default,
           "Recording filter mode should default to default");
    expect(Conf.getDataDirectory() == std::filesystem::current_path(),
           "MNEME_DATA_DIR should default to current path");
    expect(!Conf.PageSizeGiB, "MNEME_PAGE_SIZE should default to unset");
    expect(Conf.getPageSizeBytesOrDefault(64) == 64 * GiB,
           "HIP page size fallback should be 64 GiB");
    expect(Conf.getPageSizeBytesOrDefault(2) == 2 * GiB,
           "CUDA page size fallback should be 2 GiB");
    expect(Conf.MnemeLogLevel == LogLevel::Critical,
           "MNEME_LOG_LEVEL should default to critical");
    expect(!Conf.getLogDirectory(), "MNEME_LOG_DIR should default to unset");
    expect(Conf.EpilogueType == EpilogueSnapshotType::Diff,
           "MNEME_EPILOGUE_TYPE should default to diff");
    expect(!Conf.CopySource, "MNEME_COPY_SOURCE should default to false");
  }

  auto TempDir = makeTempDir();
  setenv("MNEME_DATA_DIR", TempDir.c_str(), 1);
  setenv("MNEME_PAGE_SIZE", "3", 1);
  setenv("MNEME_LOG_LEVEL", "debug", 1);
  setenv("MNEME_LOG_DIR", TempDir.c_str(), 1);
  setenv("MNEME_EPILOGUE_TYPE", "diff", 1);
  setenv("MNEME_COPY_SOURCE", "1", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.getDataDirectory() == std::filesystem::absolute(TempDir),
           "MNEME_DATA_DIR should use the configured directory");
    expect(Conf.PageSizeGiB && *Conf.PageSizeGiB == 3,
           "MNEME_PAGE_SIZE should parse as GiB");
    expect(Conf.getPageSizeBytesOrDefault(64) == 3 * GiB,
           "MNEME_PAGE_SIZE should override backend defaults");
    expect(Conf.MnemeLogLevel == LogLevel::Debug,
           "MNEME_LOG_LEVEL should map debug");
    expect(Conf.getLogDirectory() && *Conf.getLogDirectory() == TempDir,
           "MNEME_LOG_DIR should use the configured directory");
    expect(Conf.EpilogueType == EpilogueSnapshotType::Diff,
           "MNEME_EPILOGUE_TYPE should map diff");
    expect(Conf.CopySource, "MNEME_COPY_SOURCE should map 1 to true");
  }

  setenv("MNEME_PAGE_SIZE", "5abc", 1);
  setenv("MNEME_LOG_LEVEL", "verbose", 1);
  setenv("MNEME_COPY_SOURCE", "0", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.PageSizeGiB && *Conf.PageSizeGiB == 5,
           "MNEME_PAGE_SIZE should keep atol-style parsing");
    expect(Conf.MnemeLogLevel == LogLevel::Info,
           "invalid MNEME_LOG_LEVEL should fall back to info");
    expect(!Conf.CopySource, "MNEME_COPY_SOURCE should map 0 to false");
  }

  setenv("MNEME_PAGE_SIZE", "abc", 1);
  setenv("MNEME_EPILOGUE_TYPE", "bytes", 1);
  setenv("MNEME_COPY_SOURCE", "maybe", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.PageSizeGiB && *Conf.PageSizeGiB == 0,
           "invalid MNEME_PAGE_SIZE should parse to 0");
    expect(Conf.EpilogueType == EpilogueSnapshotType::Bytes,
           "MNEME_EPILOGUE_TYPE should map bytes");
    expect(!Conf.CopySource,
           "invalid MNEME_COPY_SOURCE should fall back to false");
  }

  setenv("MNEME_EPILOGUE_TYPE", "best", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.EpilogueType == EpilogueSnapshotType::Best,
           "MNEME_EPILOGUE_TYPE should map best");
  }

  setenv("MNEME_EPILOGUE_TYPE", "delta", 1);
  {
    bool Threw = false;
    try {
      auto Conf = Config::createFromEnvironment();
      (void)Conf;
    } catch (const std::runtime_error &) {
      Threw = true;
    }
    expect(Threw, "invalid MNEME_EPILOGUE_TYPE should throw");
  }

  clearMnemeEnv();

  // MNEME_RECORD_RANKS: default policy, no rank env -> single-process records.
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.isRecordingEnabledForCurrentRank(),
           "Single-process run (no rank env) should record by default");
  }

  // MNEME_RECORD_RANKS: default policy, rank 0 in MPI run.
  setenv("OMPI_COMM_WORLD_RANK", "0", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.isRecordingEnabledForCurrentRank(),
           "Default policy should record on rank 0");
  }

  // MNEME_RECORD_RANKS: default policy, rank 1 in MPI run -> excluded.
  setenv("OMPI_COMM_WORLD_RANK", "1", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(!Conf.isRecordingEnabledForCurrentRank(),
           "Default policy should exclude rank 1");
  }

  // MNEME_RECORD_RANKS=all overrides default policy.
  setenv("OMPI_COMM_WORLD_RANK", "2", 1);
  setenv("MNEME_RECORD_RANKS", "all", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.isRecordingEnabledForCurrentRank(),
           "Explicit 'all' should record on every rank");
  }

  // MNEME_RECORD_RANKS subset hit / miss.
  setenv("MNEME_RECORD_RANKS", "0,1,3", 1);
  setenv("OMPI_COMM_WORLD_RANK", "1", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.isRecordingEnabledForCurrentRank(),
           "Rank 1 should record when listed in MNEME_RECORD_RANKS=0,1,3");
  }
  setenv("OMPI_COMM_WORLD_RANK", "2", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(
        !Conf.isRecordingEnabledForCurrentRank(),
        "Rank 2 should not record when not listed in MNEME_RECORD_RANKS=0,1,3");
  }

  // Single-rank explicit list.
  setenv("MNEME_RECORD_RANKS", "0", 1);
  setenv("OMPI_COMM_WORLD_RANK", "0", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.isRecordingEnabledForCurrentRank(),
           "Rank 0 should record when MNEME_RECORD_RANKS=0");
  }

  // Empty MNEME_RECORD_RANKS behaves like unset.
  setenv("MNEME_RECORD_RANKS", "", 1);
  setenv("OMPI_COMM_WORLD_RANK", "1", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(!Conf.isRecordingEnabledForCurrentRank(),
           "Empty MNEME_RECORD_RANKS should use default rank policy");
  }

  // Rank-detection precedence: Open MPI takes priority over SLURM.
  unsetenv("MNEME_RECORD_RANKS");
  setenv("OMPI_COMM_WORLD_RANK", "2", 1);
  setenv("SLURM_PROCID", "0", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(!Conf.isRecordingEnabledForCurrentRank(),
           "OMPI_COMM_WORLD_RANK=2 should win over SLURM_PROCID=0 (rank 2 "
           "excluded)");
  }

  // Rank-detection precedence: first configured rank variable wins.
  clearMnemeEnv();
  setenv("FLUX_TASK_RANK", "3", 1);
  setenv("OMPI_COMM_WORLD_RANK", "0", 1);
  {
    auto Rank = detectDistributedRank();
    expect(Rank && *Rank == 3,
           "FLUX_TASK_RANK should win over later configured rank variables");
  }

  // Malformed rank variables fall through to later valid launcher values.
  setenv("FLUX_TASK_RANK", "abc", 1);
  setenv("OMPI_COMM_WORLD_RANK", "0", 1);
  {
    auto Rank = detectDistributedRank();
    expect(Rank && *Rank == 0,
           "Malformed rank variable should fall through to later valid value");
  }

  clearMnemeEnv();
  setenv("MNEME_RECORD_RANKS", "0,abc,2", 1);
  setenv("OMPI_COMM_WORLD_RANK", "0", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.isRecordingEnabledForCurrentRank(),
           "Malformed MNEME_RECORD_RANKS should fall back to default policy "
           "(rank 0 records)");
  }
  setenv("OMPI_COMM_WORLD_RANK", "1", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(!Conf.isRecordingEnabledForCurrentRank(),
           "Malformed MNEME_RECORD_RANKS should fall back to default policy "
           "(non-zero rank excluded)");
  }

  clearMnemeEnv();
  std::filesystem::remove_all(TempDir);

  for (const char *Mode : {"default", "scoped", "config"}) {
    clearMnemeEnv();
    setenv("MNEME_FILTER_MODE", Mode, 1);
    if (std::string(Mode) == "config")
      setenv("MNEME_FILTER_CONFIG", "policy.json", 1);
    auto Conf = Config::createFromEnvironment();
    expect(Conf.RecordingFilterMode ==
               (std::string(Mode) == "config"   ? FilterMode::Config
                : std::string(Mode) == "scoped" ? FilterMode::Scoped
                                                : FilterMode::Default),
           "Filter mode should use the configured value");
  }

  for (const char *Mode : {"automatic", "", "SCOPED", "config"}) {
    clearMnemeEnv();
    setenv("MNEME_FILTER_MODE", Mode, 1);
    bool Threw = false;
    try {
      Config::createFromEnvironment();
    } catch (const std::runtime_error &) {
      Threw = true;
    }
    expect(Threw, "Invalid modes and config without a path must fail");
  }

  clearMnemeEnv();
  setenv("MNEME_FILTER_CONFIG", "policy.json", 1);
  
  // this still takes json though in default
  Config::createFromEnvironment();
  setenv("MNEME_FILTER_MODE", "scoped", 1);
  
  bool Threw = false;
  try {
    Config::createFromEnvironment();
  } catch (const std::runtime_error &) {
    Threw = true;
  }
  expect(Threw, "Scoped mode does not accept a config path");
  clearMnemeEnv();
  return 0;
}
