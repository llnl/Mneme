#pragma once

// Kernel dispatch tracing with the vendor profiler, shared by recording and
// replay so that both measure kernel durations the same way. Each dispatch is
// tagged with the external correlation token pushed on the launching thread.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#if defined(MNEME_ENABLE_HIP)
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/rocprofiler.h>
#elif defined(MNEME_ENABLE_CUDA)
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cupti.h>
#include <cupti_activity.h>

#include "mneme/MnemeLogger.hpp"
#endif

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

#if defined(MNEME_ENABLE_HIP)

#define CHECK_ROCP(x)                                                          \
  do {                                                                         \
    auto _st = (x);                                                            \
    if (_st != ROCPROFILER_STATUS_SUCCESS) {                                   \
      const char *msg = rocprofiler_get_status_name(_st);                      \
      const char *str = rocprofiler_get_status_string(_st);                    \
      fprintf(stderr, "rocprof err %d status:'%s' string:'%s' at %s:%d: %s\n", \
              (int)_st, msg, str, __FILE__, __LINE__, #x);                     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

class DispatchTracer {
  rocprofiler_context_id_t Context{};
  rocprofiler_buffer_id_t Buffer{};
  DispatchSink Sink = nullptr;
  void *SinkData = nullptr;

  static void deliver(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                      rocprofiler_record_header_t **Headers, size_t NumHeaders,
                      void *Data, uint64_t) {
    auto *Tracer = static_cast<DispatchTracer *>(Data);
    for (size_t I = 0; I < NumHeaders; ++I) {
      if (Headers[I]->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
          Headers[I]->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        continue;
      auto *Record = static_cast<
          const rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
          Headers[I]->payload);
      Tracer->Sink(Tracer->SinkData,
                   {Record->correlation_id.external.value,
                    Record->dispatch_info.kernel_id, Record->start_timestamp,
                    Record->end_timestamp});
    }
  }

public:
  // Must run in a rocprofiler tool's initialize callback. The context is
  // started separately so that callers can add services to it first.
  void create(DispatchSink S, void *Data) {
    Sink = S;
    SinkData = Data;
    CHECK_ROCP(rocprofiler_create_context(&Context));
    CHECK_ROCP(rocprofiler_create_buffer(Context, 256 * 1024, 0,
                                         ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                         &deliver, this, &Buffer));
    CHECK_ROCP(rocprofiler_configure_buffer_tracing_service(
        Context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
        Buffer));
  }

  void start() { CHECK_ROCP(rocprofiler_start_context(Context)); }

  // Runs in the tool's finalize callback, so errors are ignored.
  void stop() {
    (void)rocprofiler_stop_context(Context);
    (void)rocprofiler_flush_buffer(Buffer);
    (void)rocprofiler_destroy_buffer(Buffer);
  }

  void push(uint64_t Token) {
    rocprofiler_thread_id_t Tid{};
    CHECK_ROCP(rocprofiler_get_thread_id(&Tid));
    rocprofiler_user_data_t Data{};
    Data.value = Token;
    CHECK_ROCP(rocprofiler_push_external_correlation_id(Context, Tid, Data));
  }

  void pop() {
    rocprofiler_thread_id_t Tid{};
    CHECK_ROCP(rocprofiler_get_thread_id(&Tid));
    rocprofiler_user_data_t Popped{};
    CHECK_ROCP(rocprofiler_pop_external_correlation_id(Context, Tid, &Popped));
  }

  rocprofiler_status_t flush() { return rocprofiler_flush_buffer(Buffer); }

  rocprofiler_context_id_t context() const { return Context; }
};

#elif defined(MNEME_ENABLE_CUDA)

inline const char *cuptiResultString(CUptiResult r) {
  const char *s = nullptr;
  cuptiGetResultString(r, &s);
  return s ? s : "<unknown>";
}

#define CHECK_CUPTI(call)                                                      \
  do {                                                                         \
    CUptiResult _r = (call);                                                   \
    if (_r != CUPTI_SUCCESS) {                                                 \
      fprintf(stderr, "CUPTI error %d (%s) at %s:%d: %s\n", (int)_r,           \
              mneme::cuptiResultString(_r), __FILE__, __LINE__, #call);        \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

class DispatchTracer {
#if defined(CUpti_ActivityKernel9)
  using KernelRec = CUpti_ActivityKernel9;
#elif defined(CUpti_ActivityKernel8)
  using KernelRec = CUpti_ActivityKernel8;
#elif defined(CUpti_ActivityKernel7)
  using KernelRec = CUpti_ActivityKernel7;
#elif defined(CUpti_ActivityKernel6)
  using KernelRec = CUpti_ActivityKernel6;
#elif defined(CUpti_ActivityKernel5)
  using KernelRec = CUpti_ActivityKernel5;
#elif defined(CUpti_ActivityKernel4)
  using KernelRec = CUpti_ActivityKernel4;
#else
  using KernelRec = CUpti_ActivityKernel9;
#endif

  static constexpr size_t BufferSize = 1 << 20;
  static constexpr size_t BufferAlign = 8;
  static constexpr size_t BufferPoolCount = 8;
  static constexpr CUpti_ExternalCorrelationKind ExternalKind =
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;

  // CUPTI delivers buffers to process-wide callbacks, so they reach the one
  // created tracer through this pointer.
  static inline DispatchTracer *Active = nullptr;

  std::mutex Mutex;
  DispatchSink Sink = nullptr;
  void *SinkData = nullptr;
  // Reused so that the callbacks do not allocate.
  std::vector<uint8_t *> BufferPool;
  size_t PoolIndex = 0;
  // Filled from EXTERNAL_CORRELATION records, which precede the kernel's.
  std::unordered_map<uint32_t, uint64_t> TokenByCorrelation;

  static void CUPTIAPI requestBuffer(uint8_t **Buffer, size_t *Size,
                                     size_t *MaxNumRecords) {
    auto &Tracer = *Active;
    std::lock_guard<std::mutex> Lock(Tracer.Mutex);

    if (Tracer.BufferPool.empty()) {
      Tracer.BufferPool.reserve(BufferPoolCount);
      for (size_t I = 0; I < BufferPoolCount; ++I) {
        void *P = nullptr;
        if (posix_memalign(&P, BufferAlign, BufferSize) != 0)
          std::abort();
        Tracer.BufferPool.push_back(reinterpret_cast<uint8_t *>(P));
      }
      Tracer.PoolIndex = 0;
    }

    *Buffer = Tracer.BufferPool[Tracer.PoolIndex++ % Tracer.BufferPool.size()];
    *Size = BufferSize;
    *MaxNumRecords = 0;
  }

  static void CUPTIAPI completeBuffer(CUcontext Ctx, uint32_t StreamId,
                                      uint8_t *Buffer, size_t,
                                      size_t ValidSize) {
    auto &Tracer = *Active;

    size_t Dropped = 0;
    CHECK_CUPTI(cuptiActivityGetNumDroppedRecords(Ctx, StreamId, &Dropped));
    if (Dropped)
      LOG_WARN("CUPTI dropped {} records", Dropped);

    std::vector<std::pair<uint32_t, uint64_t>> Correlations;
    std::vector<DispatchRecord> Kernels;
    std::vector<uint32_t> KernelCorrelations;

    CUpti_Activity *Rec = nullptr;
    CUptiResult St = CUPTI_SUCCESS;
    while ((St = cuptiActivityGetNextRecord(Buffer, ValidSize, &Rec)) ==
           CUPTI_SUCCESS) {
      if (!Rec)
        break;
      if (Rec->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        auto *E =
            reinterpret_cast<const CUpti_ActivityExternalCorrelation *>(Rec);
        Correlations.emplace_back(E->correlationId, E->externalId);
      } else if (Rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        auto *K = reinterpret_cast<const KernelRec *>(Rec);
        Kernels.push_back({0, 0, (uint64_t)K->start, (uint64_t)K->end});
        KernelCorrelations.push_back(K->correlationId);
      }
    }
    if (St != CUPTI_SUCCESS && St != CUPTI_ERROR_MAX_LIMIT_REACHED)
      LOG_WARN("cuptiActivityGetNextRecord returned {}", (int)St);

    {
      std::lock_guard<std::mutex> Lock(Tracer.Mutex);
      for (auto &[Correlation, Token] : Correlations)
        Tracer.TokenByCorrelation[Correlation] = Token;
      for (size_t I = 0; I < Kernels.size(); ++I) {
        auto It = Tracer.TokenByCorrelation.find(KernelCorrelations[I]);
        if (It == Tracer.TokenByCorrelation.end())
          continue;
        Kernels[I].Token = It->second;
        Tracer.TokenByCorrelation.erase(It);
      }
    }

    for (const DispatchRecord &Kernel : Kernels)
      if (Kernel.Token)
        Tracer.Sink(Tracer.SinkData, Kernel);
  }

public:
  // Only one tracer per process may be created.
  void create(DispatchSink S, void *Data) {
    Sink = S;
    SinkData = Data;
    Active = this;
    CHECK_CUPTI(
        cuptiActivityRegisterCallbacks(&requestBuffer, &completeBuffer));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  }

  void push(uint64_t Token) {
    CHECK_CUPTI(cuptiActivityPushExternalCorrelationId(ExternalKind, Token));
  }

  uint64_t pop() {
    uint64_t Popped = 0;
    CHECK_CUPTI(cuptiActivityPopExternalCorrelationId(ExternalKind, &Popped));
    return Popped;
  }

  CUptiResult flush() { return cuptiActivityFlushAll(0); }
};

#endif

} // namespace mneme
