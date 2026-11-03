#pragma once

// Kernel dispatch tracing with the vendor profiler, shared by recording and
// replay so that both measure kernel durations the same way. Each dispatch is
// tagged with the external correlation token pushed on the launching thread.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace mneme {

// Timestamps are in nanoseconds. Token is 0 for dispatches launched without a
// pushed token, and KernelId is only known on HIP.
struct DispatchRecord {
  uint64_t Token;
  uint64_t KernelId;
  uint64_t Start;
  uint64_t End;
};

// Called on a profiler thread, or on the thread that flushes the profiler.
using DispatchSink = void (*)(void *Data, const DispatchRecord &Record);

inline constexpr auto RecordWaitTimeout = std::chrono::seconds(10);

// Flushes the profiler until Done() holds or Timeout passes and returns Done().
// Lock guards the state Done() reads and is the lock the sink takes.
template <typename FlushFn, typename DoneFn>
bool waitForRecords(std::unique_lock<std::mutex> &Lock,
                    std::condition_variable &Arrived, FlushFn Flush,
                    DoneFn Done, std::chrono::steady_clock::duration Timeout) {
  auto Deadline = std::chrono::steady_clock::now() + Timeout;
  while (!Done() && std::chrono::steady_clock::now() < Deadline) {
    // Flushing may run the sink on this thread, which takes Lock.
    Lock.unlock();
    Flush();
    Lock.lock();
    Arrived.wait_for(Lock, std::chrono::milliseconds(10), Done);
  }
  return Done();
}

// Errors that the caller can act on are printed and returned as false; any
// other profiler error aborts.
class DispatchTracer {
public:
  DispatchTracer();
  ~DispatchTracer();
  DispatchTracer(const DispatchTracer &) = delete;
  DispatchTracer &operator=(const DispatchTracer &) = delete;

  // Returns false when the profiler is unavailable, e.g. because another tool
  // uses it. On HIP this must run in a rocprofiler tool's initialize callback;
  // on CUDA only one tracer per process may start.
  bool start(DispatchSink Sink, void *Data);
  // Runs in the HIP tool's finalize callback, so errors are ignored.
  void stop();

  void push(uint64_t Token);
  uint64_t pop();
  bool flush();

  // CUDA dispatch records carry no kernel id, so there it returns nullopt.
  std::optional<std::string> kernelName(uint64_t KernelId);

private:
  struct Impl;
  std::unique_ptr<Impl> State;
};

} // namespace mneme
