#include "mneme/MnemeKernelTimer.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>

namespace mneme {

namespace {

// The tracer library is installed, and built, next to this one.
std::string tracerLibraryPath() {
  Dl_info Info;
  if (!dladdr(reinterpret_cast<void *>(&tracerLibraryPath), &Info) ||
      !Info.dli_fname)
    return DispatchTracerLibrary;
  std::string Path = Info.dli_fname;
  return Path.substr(0, Path.rfind('/') + 1) + DispatchTracerLibrary;
}

[[noreturn]] void failToLoad(const std::string &Path) {
  std::cerr << "[mneme] Kernel timing requires " << Path << ": " << dlerror()
            << "\n";
  std::abort();
}

template <typename Fn>
Fn loadSymbol(void *Lib, const std::string &Path, const char *Name) {
  auto *Symbol = reinterpret_cast<Fn>(dlsym(Lib, Name));
  if (!Symbol)
    failToLoad(Path);
  return Symbol;
}

} // namespace

void KernelTimer::configure() {
  auto &Timer = instance();
  if (Timer.Configured)
    return;
  Timer.Configured = true;

  std::string Path = tracerLibraryPath();
  void *Lib = dlopen(Path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!Lib)
    failToLoad(Path);
  auto *Create = loadSymbol<decltype(&mnemeCreateDispatchTracer)>(
      Lib, Path, "mnemeCreateDispatchTracer");
  auto *Configure = loadSymbol<decltype(&mnemeConfigureDispatchTool)>(
      Lib, Path, "mnemeConfigureDispatchTool");

  Timer.Tracer.reset(Create());
  Configure(&KernelTimer::initialize, &KernelTimer::deinitialize);
}

KernelTimer::KernelTimer() = default;

KernelTimer::~KernelTimer() = default;

KernelTimer &KernelTimer::instance() {
  // Never destroyed, because the profiler may finalize after static
  // destructors have run.
  static KernelTimer *Timer = new KernelTimer();
  return *Timer;
}

KernelTimer &KernelTimer::get() {
  configure();
  auto &Timer = instance();
  if (!Timer.Running) {
    std::cerr << "[mneme] Kernel timing cannot start: "
              << Timer.Tracer->unavailableReason() << "\n";
    std::abort();
  }
  return Timer;
}

void KernelTimer::initialize() {
  auto &Timer = instance();
  Timer.Running = Timer.Tracer->start(&KernelTimer::onDispatch, &Timer);
}

void KernelTimer::deinitialize() {
  auto &Timer = instance();
  Timer.Running = false;
  Timer.Tracer->stop();
}

void KernelTimer::onDispatch(void *Data, const DispatchRecord &Record) {
  auto *Timer = static_cast<KernelTimer *>(Data);
  std::lock_guard<std::mutex> Lock(Timer->Mutex);
  Timer->Measured.push_back(Record);
  Timer->Arrived.notify_all();
}

std::optional<uint64_t> KernelTimer::begin(const LaunchHandle &Launch) {
  if (!Running)
    return std::nullopt;
  uint64_t Token = ++LastToken;
  Pending.try_emplace(Token, Launch);
  Tracer->push(Token);
  return Token;
}

void KernelTimer::end(uint64_t Token) {
  if (!Running)
    return;
  uint64_t Popped = Tracer->pop();
  if (Popped != Token) {
    std::cerr << "[mneme] Kernel timing popped token " << Popped
              << " but the launch pushed " << Token << "\n";
    std::abort();
  }
}

void KernelTimer::abandon(uint64_t Token) {
  end(Token);
  Pending.erase(Token);
}

void KernelTimer::addMeasured(RecordDatabase &DB) {
  for (const DispatchRecord &Record : Measured) {
    auto It = Pending.find(Record.Token);
    if (It == Pending.end() ||
        !Tracer->isDispatchOf(Record, DB.kernelName(It->second.StaticHash)))
      continue;
    const LaunchHandle &Launch = It->second;
    if (std::optional<uint64_t> Ns = Record.duration())
      DB.addLaunchTime(Launch, *Ns);
    else
      std::cerr << "[mneme] The profiler reported launch " << Launch.LaunchIndex
                << " of kernel " << DB.kernelName(Launch.StaticHash)
                << " ending before it started; it is left untimed\n";
    Pending.erase(It);
  }
  Measured.clear();
}

void KernelTimer::collect(RecordDatabase &DB) {
  std::lock_guard<std::mutex> Lock(Mutex);
  addMeasured(DB);
}

void KernelTimer::drain(RecordDatabase &DB) {
  std::unique_lock<std::mutex> Lock(Mutex);
  waitForRecords(
      Lock, Arrived, *Tracer,
      [&] {
        addMeasured(DB);
        return Pending.empty() || !Running;
      },
      RecordWaitTimeout);

  if (!Pending.empty()) {
    std::cerr << "[mneme] No duration arrived for " << Pending.size()
              << " kernel launches; they are left untimed\n";
    Pending.clear();
  }
}

} // namespace mneme
