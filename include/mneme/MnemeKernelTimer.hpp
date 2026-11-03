#pragma once

#include "mneme/MnemeDispatchTracer.hpp"
#include "mneme/MnemeSnapshot.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <llvm/ADT/DenseMap.h>

namespace mneme {

// Times recorded kernels' launches with the vendor profiler. Durations arrive
// on a profiler thread, which only queues them; the recording thread folds
// them into the record database on later launches and in drain().
class KernelTimer {
public:
  // Loads the vendor profiler and registers the timer with it. HIP must do
  // this before the runtime initializes; get() does it otherwise.
  static void configure();

  // Aborts when the profiler is unavailable.
  static KernelTimer &get();

  // Returns the launch's token, which end() or abandon() must receive, or
  // nothing once the profiler has stopped.
  std::optional<uint64_t> begin(const LaunchHandle &Launch);
  void end(uint64_t Token);
  // Ends a launch that failed, so it never produces a dispatch record.
  void abandon(uint64_t Token);
  void collect(RecordDatabase &DB);
  // Waits for the durations of all launches, which must have completed.
  void drain(RecordDatabase &DB);

  // Used by the profiler's tool callbacks.
  static KernelTimer &instance();
  void start();
  void stop();

private:
  bool Configured = false;
  std::unique_ptr<DispatchTracer> Tracer;
  std::atomic<bool> Running{false};
  std::mutex Mutex;
  std::condition_variable Arrived;
  std::vector<DispatchRecord> Measured;
  llvm::DenseMap<uint64_t, LaunchHandle> Pending;
  uint64_t LastToken = 0;

  KernelTimer();
  ~KernelTimer();
  static void onDispatch(void *Data, const DispatchRecord &Record);
  // The caller must hold Mutex.
  void addMeasured(RecordDatabase &DB);
};

} // namespace mneme
