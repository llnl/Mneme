#pragma once

// Kernel dispatch tracing with the vendor profiler, shared by recording and
// replay so that both measure kernel durations the same way. A launch is one
// kernel launch call on the host; a dispatch is one kernel the GPU runs. Each
// dispatch is tagged with the external correlation token pushed on the
// launching thread.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string_view>

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
  virtual ~DispatchTracer() = default;

  // Returns false when the profiler is unavailable, e.g. because another tool
  // uses it. On HIP this must run in a rocprofiler tool's initialize callback;
  // on CUDA only one tracer per process may start.
  virtual bool start(DispatchSink Sink, void *Data) = 0;
  // Runs in the HIP tool's finalize callback, so errors are ignored.
  virtual void stop() = 0;

  virtual void push(uint64_t Token) = 0;
  virtual uint64_t pop() = 0;
  virtual bool flush() = 0;

  // Returns whether Record is a dispatch of the kernel whose source symbol is
  // KernelName. A launch can produce more than one dispatch with its token,
  // because the runtime may first dispatch its own kernels, such as heap
  // initialization.
  virtual bool isDispatchOf(const DispatchRecord &Record,
                            std::string_view KernelName) = 0;

  // Why start() can fail or never run, for the error that stops recording.
  virtual const char *unavailableReason() const = 0;
};

using ToolCallback = void (*)();

// Defined by the vendor tracer source, which links the vendor profiler.
// Recording loads them from the dispatch tracer library only when kernel
// timing is requested, so that it does not otherwise depend on the profiler.
extern "C" {
DispatchTracer *mnemeCreateDispatchTracer();
// Registers a profiler tool whose Initialize starts a tracer. HIP requires
// this before the runtime initializes; CUDA calls Initialize immediately.
void mnemeConfigureDispatchTool(ToolCallback Initialize, ToolCallback Finalize);
}

inline constexpr const char *DispatchTracerLibrary =
    "libmneme_dispatch_tracer.so";

} // namespace mneme
