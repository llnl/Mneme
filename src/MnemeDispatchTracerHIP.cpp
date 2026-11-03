#include "mneme/MnemeDispatchTracer.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/registration.h>
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

class HIPDispatchTracer final : public DispatchTracer {
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
    auto *Self = static_cast<HIPDispatchTracer *>(Data);
    std::lock_guard<std::mutex> Lock(Self->NamesMutex);
    Self->KernelNames[Symbol->kernel_id] = Symbol->kernel_name;
  }

  static void deliver(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                      rocprofiler_record_header_t **Headers, size_t NumHeaders,
                      void *Data, uint64_t) {
    auto *Self = static_cast<HIPDispatchTracer *>(Data);
    for (size_t I = 0; I < NumHeaders; ++I) {
      if (Headers[I]->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
          Headers[I]->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        continue;
      auto *Record = static_cast<
          const rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
          Headers[I]->payload);
      // Dispatches launched without a pushed token, such as those of other
      // threads, have no external correlation id.
      if (Record->correlation_id.external.value == 0)
        continue;
      Self->Sink(Self->SinkData,
                 {Record->correlation_id.external.value,
                  Record->dispatch_info.kernel_id, Record->start_timestamp,
                  Record->end_timestamp});
    }
  }

public:
  bool start(DispatchSink OnDispatch, void *Data) override {
    Sink = OnDispatch;
    SinkData = Data;
    rocprofiler_tracing_operation_t SymbolRegister =
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
    return succeeded(rocprofiler_create_context(&Context),
                     "rocprofiler_create_context") &&
           succeeded(rocprofiler_create_buffer(
                         Context, 256 * 1024, 0,
                         ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                         &HIPDispatchTracer::deliver, this, &Buffer),
                     "rocprofiler_create_buffer") &&
           succeeded(rocprofiler_configure_buffer_tracing_service(
                         Context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                         nullptr, 0, Buffer),
                     "rocprofiler_configure_buffer_tracing_service") &&
           succeeded(rocprofiler_configure_callback_tracing_service(
                         Context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                         &SymbolRegister, 1, &HIPDispatchTracer::registerKernel,
                         this),
                     "rocprofiler_configure_callback_tracing_service") &&
           succeeded(rocprofiler_start_context(Context),
                     "rocprofiler_start_context");
  }

  void stop() override {
    (void)rocprofiler_stop_context(Context);
    (void)rocprofiler_flush_buffer(Buffer);
    (void)rocprofiler_destroy_buffer(Buffer);
  }

  void push(uint64_t Token) override {
    rocprofiler_thread_id_t Tid{};
    check(rocprofiler_get_thread_id(&Tid), "rocprofiler_get_thread_id");
    rocprofiler_user_data_t Data{};
    Data.value = Token;
    check(rocprofiler_push_external_correlation_id(Context, Tid, Data),
          "rocprofiler_push_external_correlation_id");
  }

  uint64_t pop() override {
    rocprofiler_thread_id_t Tid{};
    check(rocprofiler_get_thread_id(&Tid), "rocprofiler_get_thread_id");
    rocprofiler_user_data_t Popped{};
    check(rocprofiler_pop_external_correlation_id(Context, Tid, &Popped),
          "rocprofiler_pop_external_correlation_id");
    return Popped.value;
  }

  void flush() override {
    check(rocprofiler_flush_buffer(Buffer), "rocprofiler_flush_buffer");
  }

  bool isDispatchOf(const DispatchRecord &Record,
                    std::string_view KernelName) override {
    // rocprofiler names a kernel by its descriptor symbol, which ends in ".kd".
    constexpr std::string_view DescriptorSuffix = ".kd";
    std::lock_guard<std::mutex> Lock(NamesMutex);
    auto It = KernelNames.find(Record.KernelId);
    if (It == KernelNames.end())
      return false;
    std::string_view Symbol = It->second;
    return Symbol.size() == KernelName.size() + DescriptorSuffix.size() &&
           Symbol.substr(0, KernelName.size()) == KernelName &&
           Symbol.substr(KernelName.size()) == DescriptorSuffix;
  }

  const char *unavailableReason() const override {
    return "rocprofiler-sdk is unavailable, likely in use by another tool "
           "such as rocprofv3";
  }
};

ToolCallback ToolInitialize = nullptr;
ToolCallback ToolFinalize = nullptr;

int initializeTool(rocprofiler_client_finalize_t, void *) {
  ToolInitialize();
  return 0;
}

void finalizeTool(void *) { ToolFinalize(); }

rocprofiler_tool_configure_result_t *
configureTool(uint32_t, const char *, uint32_t,
              rocprofiler_client_id_t *ClientId) {
  ClientId->name = "mneme-record";
  static rocprofiler_tool_configure_result_t Result{
      sizeof(Result), &initializeTool, &finalizeTool, nullptr};
  return &Result;
}

} // namespace

extern "C" DispatchTracer *mnemeCreateDispatchTracer() {
  return new HIPDispatchTracer();
}

extern "C" void mnemeConfigureDispatchTool(ToolCallback Initialize,
                                           ToolCallback Finalize) {
  ToolInitialize = Initialize;
  ToolFinalize = Finalize;
  (void)rocprofiler_force_configure(&configureTool);
}

} // namespace mneme
