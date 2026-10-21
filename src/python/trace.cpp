#include "trace.hpp"
#include "llvm/core.h"

#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace mneme::trace {

namespace {

uint64_t now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool requested() {
  static const bool Enabled = [] {
    const char *Dir = std::getenv("MNEME_TRACE_DIR");

    return Dir && *Dir;
  }();

  return Enabled;
}

struct Event {
  std::string Name;
  uint64_t Start, End;
  long Thread;
  std::string Args;
  int64_t Iteration;
};

struct State {
  const int Owner = getpid();
  const uint64_t Start = now();

  std::atomic<bool> Failed{false};
  std::mutex Mutex;
  std::vector<Event> Pending;
  std::unordered_set<long> Threads;

  int File = -1;
  off_t Tail = 2;
  bool HasEvents = false;
};


std::atomic<State *> Current{nullptr};
std::atomic<int> WarnedProcess{0};

void warn(const char *Reason) noexcept {
  int Pid = getpid();

  if (WarnedProcess.exchange(Pid) != Pid)
    std::fprintf(stderr, "mneme: tracing disabled: %s\n", Reason);
}

State *state() {
  State *S = Current.load();

  if (!S || S->Owner != getpid()) {
    auto *Fresh = new State;

    if (Current.compare_exchange_strong(S, Fresh)) {
      S = Fresh;
      std::atexit(flush);
    } else {
      delete Fresh;
    }
  }

  return S;
}

long threadID() {
  thread_local int Owner = 0;
  thread_local long Thread = 0;

  if (Owner != getpid()) {
    Owner = getpid();
    Thread = syscall(SYS_gettid);
  }

  return Thread;
}

void writeAll(int File, const std::string &Data) {
  size_t Done = 0;

  while (Done < Data.size()) {
    ssize_t N = write(File, Data.data() + Done, Data.size() - Done);

    if (N < 0 && errno == EINTR)
      continue;

    if (N <= 0)
      throw std::runtime_error("could not write trace file");

    Done += N;
  }
}

void openFile(State &S) {
  const char *Dir = std::getenv("MNEME_TRACE_DIR");
  if (!Dir || !*Dir)
    throw std::runtime_error("MNEME_TRACE_DIR is empty");

  std::filesystem::create_directories(Dir);

  char Host[256] = {};
  if (gethostname(Host, sizeof(Host) - 1) != 0)
    throw std::runtime_error("could not determine trace hostname");

  std::string Name = "mneme-" + std::string(Host) + "-" +
                     std::to_string(S.Owner) + "-" + std::to_string(S.Start) +
                     ".json";
  auto Path = std::filesystem::path(Dir) / Name;
  S.File = open(Path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (S.File < 0)
    throw std::runtime_error("could not open trace file: " +
                             std::string(std::strerror(errno)));

  writeAll(S.File, "[\n]\n");
}

} // namespace

void disable(const char *Reason) noexcept {
  try {
    if (requested())
      state()->Failed = true;
  } catch (...) {
  }

  warn(Reason);
}

uint64_t timestamp() noexcept {
  if (!requested())
    return 0;

  try {
    return state()->Failed ? 0 : now();
  } catch (...) {
    disable("could not initialize trace buffer");
    return 0;
  }
}

void complete(const char *Name, uint64_t Start, uint64_t End, const char *Args,
              int64_t Iteration) noexcept {
  if (!Start || End < Start || !requested())
    return;

  try {
    auto *S = state();
    if (S->Failed)
      return;

    std::lock_guard<std::mutex> Lock(S->Mutex);
    S->Pending.push_back(
        {Name, Start, End, threadID(), Args ? Args : "", Iteration});
  } catch (...) {
    disable("could not buffer trace event");
  }
}

void flush() noexcept {
  if (!requested())
    return;

  try {
    auto *S = state();
    if (S->Failed)
      return;

    std::lock_guard<std::mutex> Lock(S->Mutex);

    if (S->Pending.empty())
      return;

    if (S->File < 0)
      openFile(*S);

    std::string Batch;
    llvm::raw_string_ostream Out(Batch);

    auto Emit = [&](llvm::json::Object Object) {
      if (S->HasEvents)
        Out << ",\n";

      Out << llvm::json::Value(std::move(Object));
      S->HasEvents = true;
    };

    if (S->Tail == 2) {
#if defined(MNEME_ENABLE_HIP)
      const char *Backend = "hip";
#elif defined(MNEME_ENABLE_CUDA)
      const char *Backend = "cuda";
#else
      const char *Backend = "unknown";
#endif
      Emit(llvm::json::Object{
          {"ph", "M"},
          {"name", "process_name"},
          {"pid", S->Owner},
          {"args", llvm::json::Object{{"name", "Mneme replay"}}}});

      Emit(llvm::json::Object{
          {"ph", "X"},
          {"name", "metadata.process"},
          {"cat", "mneme.metadata"},
          {"pid", S->Owner},
          {"tid", S->Pending.front().Thread},
          {"ts", S->Start / 1000.0},
          {"dur", 0},
          {"args", llvm::json::Object{{"backend", Backend}}}});
    }

    for (const auto &E : S->Pending) {
      if (S->Threads.insert(E.Thread).second)
        Emit(llvm::json::Object{
            {"ph", "M"},
            {"name", "thread_name"},
            {"pid", S->Owner},
            {"tid", E.Thread},
            {"args", llvm::json::Object{{"name", "Mneme replay"}}}});

      llvm::json::Object Args;
      if (!E.Args.empty()) {
        auto Parsed = llvm::json::parse(E.Args);
        if (!Parsed) {
          llvm::consumeError(Parsed.takeError());
          throw std::runtime_error("invalid trace event arguments");
        }

        auto *Object = Parsed->getAsObject();
        if (!Object)
          throw std::runtime_error("trace event arguments must be an object");

        Args = std::move(*Object);
      }

      if (E.Iteration >= 0)
        Args["iteration"] = E.Iteration;

      Emit(llvm::json::Object{
          {"ph", "X"},
          {"name", E.Name},
          {"cat", "mneme." + E.Name.substr(0, E.Name.find('.'))},
          {"pid", S->Owner},
          {"tid", E.Thread},
          {"ts", E.Start / 1000.0},
          {"dur", (E.End - E.Start) / 1000.0},
          {"args", std::move(Args)}});
    }

    Out.flush();

    // only replace the last closing bracket so we can still read
    // completed batches in the file
    if (lseek(S->File, S->Tail, SEEK_SET) < 0)
      throw std::runtime_error("could not seek trace file");

    writeAll(S->File, Batch + "]\n");
    S->Tail += Batch.size();
    std::vector<Event>().swap(S->Pending);
  } catch (const std::exception &Error) {
    disable(Error.what());
  } catch (...) {
    disable("could not flush trace buffer");
  }
}

Scope::Scope(const char *Name, int64_t Iteration) noexcept
    : Name(Name), Iteration(Iteration), Start(timestamp()),
      Owner(Start ? getpid() : 0) {}

Scope::~Scope() noexcept {
  if (Start && Owner == getpid())
    complete(Name, Start, now(), nullptr, Iteration);
}

} // namespace mneme::trace

extern "C" {

API_EXPORT(uint64_t) MnemePy_TraceTimestamp() {
  return mneme::trace::timestamp();
}

API_EXPORT(void)
MnemePy_TraceComplete(const char *Name, uint64_t Start, uint64_t End,
                      const char *Args) {
  mneme::trace::complete(Name, Start, End, Args);
}

API_EXPORT(void) MnemePy_TraceFlush() {
  mneme::trace::flush();
}

API_EXPORT(void) MnemePy_TraceDisable(const char *Reason) {
  mneme::trace::disable(Reason);
}

}
