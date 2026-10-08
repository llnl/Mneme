#include "mneme/MnemeRecordingPolicy.hpp"
#include "mneme/MnemeConfig.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>

namespace mneme {
namespace {

[[noreturn]] void invalid(const std::string &Path, const std::string &Message) {
  throw std::runtime_error("Invalid recording filter " + Path + ": " + Message);
}

const llvm::json::Object &object(const llvm::json::Value &Value,
                                 const std::string &Path) {
  auto *Object = Value.getAsObject();
  if (!Object)
    invalid(Path, "expected a mapping");
  return *Object;
}

void keys(const llvm::json::Object &Object, const std::string &Path,
          std::initializer_list<llvm::StringRef> Allowed) {
  for (const auto &Entry : Object) {
    llvm::StringRef Key = Entry.first;
    if (std::find(Allowed.begin(), Allowed.end(), Key) == Allowed.end())
      invalid(Path + "." + Key.str(), "unknown field");
  }
}

void counts(const llvm::json::Object &Object, const std::string &Path,
            KernelRecordingPolicy &Policy) {
  for (auto Field :
       {std::pair{"skip", &Policy.Skip}, std::pair{"every", &Policy.Every},
        std::pair{"max_records", &Policy.MaxRecords}}) {

    if (const auto *Value = Object.get(Field.first)) {
      // rejects floating-point values and booleans.
      auto Number = Value->getAsUINT64();
      if (!Number || (std::string(Field.first) == "every" && *Number == 0)) {
        invalid(Path + "." + Field.first,
                std::string(Field.first) == "every"
                    ? "expected a positive integer"
                    : "expected a nonnegative integer");
      }
        
      *Field.second = static_cast<uint64_t>(*Number);
    }
  }
}

} // namespace

RecordingPolicy RecordingPolicy::fromJSON(const std::string &Text) {
  auto Parsed = llvm::json::parse(Text);
  if (!Parsed)
    invalid("document", llvm::toString(Parsed.takeError()));

  const auto &Root = object(*Parsed, "document");
  keys(Root, "document", {"recording"});

  const auto *Recording = Root.get("recording");
  if (!Recording)
    invalid("recording", "required mapping is missing");

  const auto &Body = object(*Recording, "recording");
  keys(Body, "recording", {"defaults", "kernel_overrides", "deduplication"});

  RecordingPolicy Result;
  Result.EnabledThisRank =
      config_detail::computeRecordingEnabledForCurrentRank();

  if (const auto *Value = Body.get("deduplication")) {
    auto Name = Value->getAsString();
    if (!Name || (*Name != "configuration" && *Name != "invocations"))
      invalid("recording.deduplication",
              "expected configuration or invocations");
    Result.Dedup = *Name == "invocations" ? Deduplication::Invocations
                                          : Deduplication::Configuration;
  }

  if (const auto *Value = Body.get("defaults")) {
    const auto &Defaults = object(*Value, "recording.defaults");
    keys(Defaults, "recording.defaults",
         {"skip", "every", "max_records", "ranks"});
    counts(Defaults, "recording.defaults", Result.Defaults);

    if (const auto *Ranks = Defaults.get("ranks")) {
      bool Enabled = false;

      if (auto All = Ranks->getAsString(); All && *All == "all") {
        Enabled = true;
      } else if (auto *List = Ranks->getAsArray()) {
        // check if the current rank is included in the list of ranks
        int Rank = detectDistributedRank().value_or(0);
        for (const auto &Item : *List) {
          auto Number = Item.getAsUINT64();
          if (!Number || *Number > INT_MAX)
            invalid("recording.defaults.ranks",
                    "expected nonnegative integer ranks");
          Enabled |= *Number == Rank;
        }
      } else {
        invalid("recording.defaults.ranks", "expected a list of ranks or all");
      }

      if (!std::getenv("MNEME_RECORD_RANKS"))
        Result.EnabledThisRank = Enabled;
    }
  }


  if (const auto *Value = Body.get("kernel_overrides")) {

    const auto *Overrides = Value->getAsArray();
    if (!Overrides)
      invalid("recording.kernel_overrides", "expected a list");

    for (size_t I = 0; I < Overrides->size(); ++I) {
      std::string Path =
          "recording.kernel_overrides[" + std::to_string(I) + "]";

      const auto &Override = object((*Overrides)[I], Path);
      if (Override.get("ranks") || Override.get("rank")) {
        invalid(Path + ".ranks",
                "rank selection is process-wide; enable the rank "
                "in recording.defaults.ranks or --record-ranks before "
                "filtering its kernels");

      }

      keys(Override, Path, {"match", "skip", "every", "max_records"});
      const auto *Match = Override.get("match");
      if (!Match)
        invalid(Path + ".match", "required mapping is missing");

      const auto &Matcher = object(*Match, Path + ".match");
      keys(Matcher, Path + ".match", {"demangled_regex"});
      auto Pattern = Matcher.getString("demangled_regex");
      if (!Pattern)
        invalid(Path + ".match.demangled_regex", "expected a string");

      auto Policy = Result.Defaults;
      counts(Override, Path, Policy);
      try {
        Result.Rules.push_back({std::regex(Pattern->str()), Policy});
      } catch (const std::regex_error &Error) {
        invalid(Path + ".match.demangled_regex", Error.what());
      }
    }
  }

  return Result;
}

RecordingPolicy RecordingPolicy::load(const Config &Conf) {
  if (Conf.RecordingFilterMode != FilterMode::Config) {
    // If the recording filter mode is not Config, we create a default
    // RecordingPolicy with the current rank's recording status.
    RecordingPolicy Result;
    Result.EnabledThisRank = Conf.isRecordingEnabledForCurrentRank();
    return Result;
  }

  // otherwise read teh config file
  auto Buffer = llvm::MemoryBuffer::getFile(*Conf.FilterConfigPath);
  if (!Buffer)
    throw std::runtime_error("Cannot read MNEME_FILTER_CONFIG '" +
                             *Conf.FilterConfigPath +
                             "': " + Buffer.getError().message());

  return fromJSON((*Buffer)->getBuffer().str());
}

const RecordingPolicy &RecordingPolicy::get() {
  static const RecordingPolicy Policy = load(Config::get());
  return Policy;
}

RecordingPolicy RecordingPolicy::createFromEnvironment() {
  return load(Config::createFromEnvironment());
}

KernelRecordingPolicy RecordingPolicy::resolve(const std::string &Name) const {
  for (const auto &Rule : Rules)
    if (std::regex_search(Name, Rule.Pattern))
      return Rule.Policy;
  return Defaults;
}

} // namespace mneme
