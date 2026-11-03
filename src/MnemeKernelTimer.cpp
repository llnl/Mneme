#include "mneme/MnemeKernelTimer.hpp"
#include "mneme/MnemeDispatchTracer.hpp"

#include <chrono>
#include <iostream>

#if defined(MNEME_ENABLE_HIP)
#include <rocprofiler-sdk/registration.h>
#endif

namespace mneme {

#if defined(MNEME_ENABLE_HIP)
namespace {

constexpr const char *UnavailableMessage =
    "[mneme] Kernel timing disabled: rocprofiler-sdk is unavailable, likely in "
    "use by another tool such as rocprofv3.\n";

int initializeTool(rocprofiler_client_finalize_t, void *) {
  KernelTimer::instance().start();
  return 0;
}

void finalizeTool(void *) { KernelTimer::instance().stop(); }

rocprofiler_tool_configure_result_t *
configureTool(uint32_t, const char *, uint32_t,
              rocprofiler_client_id_t *ClientId) {
  ClientId->name = "mneme-record";
  static rocprofiler_tool_configure_result_t Result{
      sizeof(Result), &initializeTool, &finalizeTool, nullptr};
  return &Result;
}

} // namespace

void KernelTimer::configure() {
  (void)rocprofiler_force_configure(&configureTool);
}
#else
namespace {

constexpr const char *UnavailableMessage =
    "[mneme] Kernel timing disabled: CUPTI is unavailable, likely in use by "
    "another tool such as Nsight Compute.\n";

} // namespace

void KernelTimer::configure() {}
#endif

KernelTimer::KernelTimer() : Tracer(std::make_unique<DispatchTracer>()) {}

KernelTimer::~KernelTimer() = default;

KernelTimer &KernelTimer::instance() {
  // Never destroyed, because the profiler may finalize after static
  // destructors have run.
  static KernelTimer *Timer = new KernelTimer();
  return *Timer;
}

KernelTimer *KernelTimer::get() {
  auto &Timer = instance();
#if defined(MNEME_ENABLE_CUDA)
  if (!Timer.Running)
    Timer.start();
#endif
  if (!Timer.Running) {
    std::cerr << UnavailableMessage;
    return nullptr;
  }
  return &Timer;
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

void KernelTimer::begin(const TimedLaunch &Launch, llvm::StringRef KernelName) {
  if (!Running)
    return;
  // rocprofiler names a kernel by its descriptor symbol, which ends in ".kd".
  KernelNames.try_emplace(Launch.StaticHash, (KernelName + ".kd").str());
  Pending.try_emplace(++LastToken, Launch);
  Tracer->push(LastToken);
}

void KernelTimer::end(bool Launched) {
  if (!Running)
    return;
  Tracer->pop();
  // A launch that failed never produces a dispatch record.
  if (!Launched)
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
