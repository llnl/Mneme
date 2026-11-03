#pragma once

#include "mneme/MnemeSnapshot.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringRef.h>

namespace mneme {

class DispatchTracer;
struct DispatchRecord;

// Times recorded kernels' launches with the vendor profiler. Durations arrive
// on a profiler thread, which only queues them; the recording thread folds
// them into the record database on later launches and in drain().
class KernelTimer {
public:
  using AddFn = llvm::function_ref<void(const LaunchHandle &, uint64_t)>;

  // Loads the vendor profiler and registers the timer with it. HIP must do
  // this before the runtime initializes; get() does it otherwise.
  static void configure();

  // Returns nullptr when the profiler is unavailable.
  static KernelTimer *get();

  // Returns whether the launch is timed, in which case end() or abandon() must
  // follow it.
  bool begin(const LaunchHandle &Launch, llvm::StringRef KernelName);
  void end();
  // Ends a launch that failed, so it never produces a dispatch record.
  void abandon();
  void collect(AddFn Add);
  // Waits for the durations of all launches, which must have completed.
  void drain(AddFn Add);

  // Used by the profiler's tool callbacks.
  static KernelTimer &instance();
  void start();
  void stop();

private:
  struct Measurement {
    uint64_t Token;
    uint64_t KernelId;
    uint64_t Start;
    uint64_t End;
  };

  bool Configured = false;
  std::unique_ptr<DispatchTracer> Tracer;
  std::atomic<bool> Running{false};
  std::mutex Mutex;
  std::condition_variable Arrived;
  std::vector<Measurement> Measured;
  llvm::DenseMap<uint64_t, LaunchHandle> Pending;
  llvm::DenseMap<uint64_t, std::string> KernelNames;
  uint64_t LastToken = 0;

  KernelTimer();
  ~KernelTimer();
  static void onDispatch(void *Data, const DispatchRecord &Record);
  bool dispatchedLaunchedKernel(const Measurement &M,
                                const LaunchHandle &Launch);
  // The caller must hold Mutex.
  void addMeasured(AddFn Add);
};

} // namespace mneme
