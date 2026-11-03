#include "mneme/MnemeDispatchTracer.hpp"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace mneme {

namespace {

bool succeeded(rocprofiler_status_t Status, const char *Call) {
  if (Status == ROCPROFILER_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "[mneme] %s failed: %s (%s)\n", Call,
               rocprofiler_get_status_name(Status),
               rocprofiler_get_status_string(Status));
  return false;
}

void check(rocprofiler_status_t Status, const char *Call) {
  if (!succeeded(Status, Call))
    std::abort();
}

} // namespace

struct DispatchTracer::Impl {
  rocprofiler_context_id_t Context{};
  rocprofiler_buffer_id_t Buffer{};
  DispatchSink Sink = nullptr;
  void *SinkData = nullptr;
  std::mutex NamesMutex;
  std::unordered_map<uint64_t, std::string> KernelNames;

  static void registerKernel(rocprofiler_callback_tracing_record_t Record,
                             rocprofiler_user_data_t *, void *Data) {
    if (Record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
        Record.operation !=
            ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
        Record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
      return;
    auto *Symbol = static_cast<
        const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
            *>(Record.payload);
    if (!Symbol->kernel_name)
      return;
    auto *Self = static_cast<Impl *>(Data);
    std::lock_guard<std::mutex> Lock(Self->NamesMutex);
    Self->KernelNames[Symbol->kernel_id] = Symbol->kernel_name;
  }

  static void deliver(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                      rocprofiler_record_header_t **Headers, size_t NumHeaders,
                      void *Data, uint64_t) {
    auto *Self = static_cast<Impl *>(Data);
    for (size_t I = 0; I < NumHeaders; ++I) {
      if (Headers[I]->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
          Headers[I]->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        continue;
      auto *Record = static_cast<
          const rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
          Headers[I]->payload);
      Self->Sink(Self->SinkData,
                 {Record->correlation_id.external.value,
                  Record->dispatch_info.kernel_id, Record->start_timestamp,
                  Record->end_timestamp});
    }
  }
};

DispatchTracer::DispatchTracer() : State(std::make_unique<Impl>()) {}

DispatchTracer::~DispatchTracer() = default;

bool DispatchTracer::start(DispatchSink Sink, void *Data) {
  Impl &S = *State;
  S.Sink = Sink;
  S.SinkData = Data;
  rocprofiler_tracing_operation_t SymbolRegister =
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
  return succeeded(rocprofiler_create_context(&S.Context),
                   "rocprofiler_create_context") &&
         succeeded(rocprofiler_create_buffer(S.Context, 256 * 1024, 0,
                                             ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                             &Impl::deliver, &S, &S.Buffer),
                   "rocprofiler_create_buffer") &&
         succeeded(rocprofiler_configure_buffer_tracing_service(
                       S.Context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                       nullptr, 0, S.Buffer),
                   "rocprofiler_configure_buffer_tracing_service") &&
         succeeded(rocprofiler_configure_callback_tracing_service(
                       S.Context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                       &SymbolRegister, 1, &Impl::registerKernel, &S),
                   "rocprofiler_configure_callback_tracing_service") &&
         succeeded(rocprofiler_start_context(S.Context),
                   "rocprofiler_start_context");
}

void DispatchTracer::stop() {
  (void)rocprofiler_stop_context(State->Context);
  (void)rocprofiler_flush_buffer(State->Buffer);
  (void)rocprofiler_destroy_buffer(State->Buffer);
}

void DispatchTracer::push(uint64_t Token) {
  rocprofiler_thread_id_t Tid{};
  check(rocprofiler_get_thread_id(&Tid), "rocprofiler_get_thread_id");
  rocprofiler_user_data_t Data{};
  Data.value = Token;
  check(rocprofiler_push_external_correlation_id(State->Context, Tid, Data),
        "rocprofiler_push_external_correlation_id");
}

uint64_t DispatchTracer::pop() {
  rocprofiler_thread_id_t Tid{};
  check(rocprofiler_get_thread_id(&Tid), "rocprofiler_get_thread_id");
  rocprofiler_user_data_t Popped{};
  check(rocprofiler_pop_external_correlation_id(State->Context, Tid, &Popped),
        "rocprofiler_pop_external_correlation_id");
  return Popped.value;
}

bool DispatchTracer::flush() {
  return succeeded(rocprofiler_flush_buffer(State->Buffer),
                   "rocprofiler_flush_buffer");
}

std::optional<std::string> DispatchTracer::kernelName(uint64_t KernelId) {
  std::lock_guard<std::mutex> Lock(State->NamesMutex);
  auto It = State->KernelNames.find(KernelId);
  if (It == State->KernelNames.end())
    return std::nullopt;
  return It->second;
}

} // namespace mneme
