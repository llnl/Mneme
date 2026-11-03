#include "mneme/MnemeDispatchTracer.hpp"
#include "mneme/MnemeLogger.hpp"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cupti.h>
#include <cupti_activity.h>

namespace mneme {

namespace {

using KernelRec = CUpti_ActivityKernel9;

constexpr size_t BufferSize = 1 << 20;
constexpr CUpti_ExternalCorrelationKind ExternalKind =
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;

bool succeeded(CUptiResult Result, const char *Call) {
  if (Result == CUPTI_SUCCESS)
    return true;
  const char *Message = nullptr;
  cuptiGetResultString(Result, &Message);
  std::fprintf(stderr, "[mneme] %s failed: %s\n", Call,
               Message ? Message : "<unknown>");
  return false;
}

void check(CUptiResult Result, const char *Call) {
  if (!succeeded(Result, Call))
    std::abort();
}

} // namespace

struct DispatchTracer::Impl {
  // CUPTI delivers buffers to process-wide callbacks, so they reach the one
  // started tracer through this pointer.
  static inline Impl *Active = nullptr;

  std::mutex Mutex;
  DispatchSink Sink = nullptr;
  void *SinkData = nullptr;
  // Filled from EXTERNAL_CORRELATION records, which precede the kernel's.
  std::unordered_map<uint32_t, uint64_t> TokenByCorrelation;

  static void CUPTIAPI requestBuffer(uint8_t **Buffer, size_t *Size,
                                     size_t *MaxNumRecords) {
    *Buffer = static_cast<uint8_t *>(std::malloc(BufferSize));
    *Size = BufferSize;
    *MaxNumRecords = 0;
  }

  static void CUPTIAPI completeBuffer(CUcontext Ctx, uint32_t StreamId,
                                      uint8_t *Buffer, size_t,
                                      size_t ValidSize) {
    Impl &Self = *Active;

    size_t Dropped = 0;
    check(cuptiActivityGetNumDroppedRecords(Ctx, StreamId, &Dropped),
          "cuptiActivityGetNumDroppedRecords");
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
    std::free(Buffer);

    {
      std::lock_guard<std::mutex> Lock(Self.Mutex);
      for (auto &[Correlation, Token] : Correlations)
        Self.TokenByCorrelation[Correlation] = Token;
      for (size_t I = 0; I < Kernels.size(); ++I) {
        auto It = Self.TokenByCorrelation.find(KernelCorrelations[I]);
        if (It == Self.TokenByCorrelation.end())
          continue;
        Kernels[I].Token = It->second;
        Self.TokenByCorrelation.erase(It);
      }
    }

    for (const DispatchRecord &Kernel : Kernels)
      if (Kernel.Token)
        Self.Sink(Self.SinkData, Kernel);
  }
};

DispatchTracer::DispatchTracer() : State(std::make_unique<Impl>()) {}

DispatchTracer::~DispatchTracer() = default;

bool DispatchTracer::start(DispatchSink Sink, void *Data) {
  State->Sink = Sink;
  State->SinkData = Data;
  Impl::Active = State.get();
  if (!succeeded(cuptiActivityRegisterCallbacks(&Impl::requestBuffer,
                                                &Impl::completeBuffer),
                 "cuptiActivityRegisterCallbacks"))
    return false;
  for (CUpti_ActivityKind Kind :
       {CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_DRIVER})
    if (!succeeded(cuptiActivityEnable(Kind), "cuptiActivityEnable"))
      return false;
  return true;
}

void DispatchTracer::stop() {}

void DispatchTracer::push(uint64_t Token) {
  check(cuptiActivityPushExternalCorrelationId(ExternalKind, Token),
        "cuptiActivityPushExternalCorrelationId");
}

uint64_t DispatchTracer::pop() {
  uint64_t Popped = 0;
  check(cuptiActivityPopExternalCorrelationId(ExternalKind, &Popped),
        "cuptiActivityPopExternalCorrelationId");
  return Popped;
}

bool DispatchTracer::flush() {
  return succeeded(cuptiActivityFlushAll(0), "cuptiActivityFlushAll");
}

std::optional<std::string> DispatchTracer::kernelName(uint64_t) {
  return std::nullopt;
}

} // namespace mneme
