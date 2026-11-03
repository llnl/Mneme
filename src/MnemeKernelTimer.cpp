#include "mneme/MnemeKernelTimer.hpp"
#include "mneme/MnemeDispatchTracer.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>

namespace mneme {

namespace {

#if defined(MNEME_ENABLE_HIP)
constexpr const char *UnavailableMessage =
    "[mneme] Kernel timing disabled: rocprofiler-sdk is unavailable, likely in "
    "use by another tool such as rocprofv3.\n";
#else
constexpr const char *UnavailableMessage =
    "[mneme] Kernel timing disabled: CUPTI is unavailable, likely in use by "
    "another tool such as Nsight Compute.\n";
#endif

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

void initializeTool() { KernelTimer::instance().start(); }

void finalizeTool() { KernelTimer::instance().stop(); }

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
  Configure(&initializeTool, &finalizeTool);
}

KernelTimer::KernelTimer() = default;

KernelTimer::~KernelTimer() = default;

KernelTimer &KernelTimer::instance() {
  // Never destroyed, because the profiler may finalize after static
  // destructors have run.
  static KernelTimer *Timer = new KernelTimer();
  return *Timer;
}

KernelTimer *KernelTimer::get() {
  configure();
  auto &Timer = instance();
  if (Timer.Running)
    return &Timer;
  std::cerr << UnavailableMessage;
  return nullptr;
}

void KernelTimer::start() {
  Running = Tracer->start(&KernelTimer::onDispatch, this);
}

void KernelTimer::stop() {
  Running = false;
  Tracer->stop();
}

void KernelTimer::onDispatch(void *Data, const DispatchRecord &Record) {
  // Token 0 marks dispatches that are not timed, such as runtime copies.
  if (Record.Token == 0)
    return;
  auto *Timer = static_cast<KernelTimer *>(Data);
  std::lock_guard<std::mutex> Lock(Timer->Mutex);
  Timer->Measured.push_back(
      {Record.Token, Record.KernelId, Record.Start, Record.End});
  Timer->Arrived.notify_all();
}

bool KernelTimer::begin(const TimedLaunch &Launch, llvm::StringRef KernelName) {
  if (!Running)
    return false;
  // rocprofiler names a kernel by its descriptor symbol, which ends in ".kd".
  KernelNames.try_emplace(Launch.StaticHash, (KernelName + ".kd").str());
  Pending.try_emplace(++LastToken, Launch);
  Tracer->push(LastToken);
  return true;
}

void KernelTimer::end() {
  if (!Running)
    return;
  Tracer->pop();
}

void KernelTimer::abandon() {
  if (!Running)
    return;
  Tracer->pop();
  Pending.erase(LastToken);
}

// Setup kernels the runtime dispatches during a launch, such as heap
// initialization, carry the launch's token, so the token alone does not
// identify the launched kernel's record.
bool KernelTimer::dispatchedLaunchedKernel(const Measurement &M,
                                           const TimedLaunch &Launch) {
#if defined(MNEME_ENABLE_HIP)
  auto Name = Tracer->kernelName(M.KernelId);
  return Name && *Name == KernelNames.find(Launch.StaticHash)->second;
#else
  // CUPTI's dispatch records carry no kernel id to check.
  return true;
#endif
}

void KernelTimer::addMeasured(AddFn Add) {
  for (const Measurement &M : Measured) {
    auto It = Pending.find(M.Token);
    if (It == Pending.end() || !dispatchedLaunchedKernel(M, It->second))
      continue;
    if (M.End >= M.Start)
      Add(It->second, M.End - M.Start);
    Pending.erase(It);
  }
  Measured.clear();
}

void KernelTimer::collect(AddFn Add) {
  std::lock_guard<std::mutex> Lock(Mutex);
  addMeasured(Add);
}

void KernelTimer::drain(AddFn Add) {
  std::unique_lock<std::mutex> Lock(Mutex);
  waitForRecords(
      Lock, Arrived, [&] { (void)Tracer->flush(); },
      [&] {
        addMeasured(Add);
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
