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
  using AddFn = llvm::function_ref<void(const TimedLaunch &, uint64_t)>;

  // HIP only: rocprofiler-sdk must be configured before the runtime
  // initializes.
  static void configure();

  // Returns nullptr when the profiler is unavailable.
  static KernelTimer *get();

  void begin(const TimedLaunch &Launch, llvm::StringRef KernelName);
  void end(bool Launched);
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

  std::unique_ptr<DispatchTracer> Tracer;
  std::atomic<bool> Running{false};
  std::mutex Mutex;
  std::condition_variable Arrived;
  std::vector<Measurement> Measured;
  llvm::DenseMap<uint64_t, TimedLaunch> Pending;
  llvm::DenseMap<uint64_t, std::string> KernelNames;
  uint64_t LastToken = 0;

  KernelTimer();
  ~KernelTimer();
  static void onDispatch(void *Data, const DispatchRecord &Record);
  bool dispatchedLaunchedKernel(const Measurement &M,
                                const TimedLaunch &Launch);
  // The caller must hold Mutex.
  void addMeasured(AddFn Add);
};

} // namespace mneme
