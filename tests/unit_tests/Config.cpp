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

void expect(bool Condition, const std::string &Message) {
  if (!Condition) {
    std::cerr << Message << "\n";
    std::exit(1);
  }
}

void clearMnemeEnv() {
  unsetenv("MNEME_RR_KERNELS");
  unsetenv("MNEME_DATA_DIR");
  unsetenv("MNEME_MAX_RECORDINGS");
  unsetenv("MNEME_SKIP_RECORDINGS");
  unsetenv("MNEME_LOG_LEVEL");
  unsetenv("MNEME_LOG_DIR");
  unsetenv("MNEME_EPILOGUE_TYPE");
  unsetenv("MNEME_CHUNK_SIZE");
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
    expect(!Conf.KernelRegex, "MNEME_RR_KERNELS should default to unset");
    expect(Conf.getDataDirectory() == std::filesystem::current_path(),
           "MNEME_DATA_DIR should default to current path");
    expect(Conf.MaxRecordings == 4, "MNEME_MAX_RECORDINGS should default to 4");
    expect(Conf.SkipRecordings == 0,
           "MNEME_SKIP_RECORDINGS should default to 0");
    expect(Conf.MnemeLogLevel == LogLevel::Critical,
           "MNEME_LOG_LEVEL should default to critical");
    expect(!Conf.getLogDirectory(), "MNEME_LOG_DIR should default to unset");
    expect(Conf.EpilogueType == EpilogueSnapshotType::Diff,
           "MNEME_EPILOGUE_TYPE should default to diff");
    expect(Conf.ChunkSize == 256ULL << 20,
           "MNEME_CHUNK_SIZE should default to 256 MiB");
  }

  auto TempDir = makeTempDir();
  setenv("MNEME_RR_KERNELS", "_two", 1);
  setenv("MNEME_DATA_DIR", TempDir.c_str(), 1);
  setenv("MNEME_MAX_RECORDINGS", "8", 1);
  setenv("MNEME_SKIP_RECORDINGS", "2", 1);
  setenv("MNEME_LOG_LEVEL", "debug", 1);
  setenv("MNEME_LOG_DIR", TempDir.c_str(), 1);
  setenv("MNEME_EPILOGUE_TYPE", "diff", 1);
  setenv("MNEME_CHUNK_SIZE", "8388608", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.KernelRegex && *Conf.KernelRegex == "_two",
           "MNEME_RR_KERNELS should preserve the regex string");
    expect(Conf.getDataDirectory() == std::filesystem::absolute(TempDir),
           "MNEME_DATA_DIR should use the configured directory");
    expect(Conf.MaxRecordings == 8,
           "MNEME_MAX_RECORDINGS should use the configured value");
    expect(Conf.SkipRecordings == 2,
           "MNEME_SKIP_RECORDINGS should use the configured value");
    expect(Conf.MnemeLogLevel == LogLevel::Debug,
           "MNEME_LOG_LEVEL should map debug");
    expect(Conf.getLogDirectory() && *Conf.getLogDirectory() == TempDir,
           "MNEME_LOG_DIR should use the configured directory");
    expect(Conf.EpilogueType == EpilogueSnapshotType::Diff,
           "MNEME_EPILOGUE_TYPE should map diff");
    expect(Conf.ChunkSize == 8ULL << 20,
           "MNEME_CHUNK_SIZE should use the configured value");
  }

  for (const char *Bad :
       {"", "2097152", "3145728", "-4194304", "8M", "18446744073709551616"}) {
    setenv("MNEME_CHUNK_SIZE", Bad, 1);
    auto Conf = Config::createFromEnvironment();
    expect(Conf.ChunkSize == 256ULL << 20,
           std::string("invalid MNEME_CHUNK_SIZE '") + Bad +
               "' should fall back to 256 MiB");
  }
  unsetenv("MNEME_CHUNK_SIZE");

  setenv("MNEME_MAX_RECORDINGS", "12abc", 1);
  setenv("MNEME_SKIP_RECORDINGS", "6abc", 1);
  setenv("MNEME_LOG_LEVEL", "verbose", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.MaxRecordings == 12,
           "MNEME_MAX_RECORDINGS should keep atoi-style parsing");
    expect(Conf.SkipRecordings == 6,
           "MNEME_SKIP_RECORDINGS should keep atoi-style parsing");
    expect(Conf.MnemeLogLevel == LogLevel::Info,
           "invalid MNEME_LOG_LEVEL should fall back to info");
  }

  setenv("MNEME_MAX_RECORDINGS", "abc", 1);
  setenv("MNEME_SKIP_RECORDINGS", "abc", 1);
  setenv("MNEME_EPILOGUE_TYPE", "bytes", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.MaxRecordings == 0,
           "invalid MNEME_MAX_RECORDINGS should parse to 0");
    expect(Conf.SkipRecordings == 0,
           "invalid MNEME_SKIP_RECORDINGS should parse to 0");
    expect(Conf.EpilogueType == EpilogueSnapshotType::Bytes,
           "MNEME_EPILOGUE_TYPE should map bytes");
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
  return 0;
}
