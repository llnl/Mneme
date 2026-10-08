#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <mutex>
#include <unordered_map>
#include <condition_variable>
#include <utility>
#include <vector>

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "mneme/MnemeLogger.hpp"
#include <mneme/DeviceTraits.hpp>

#ifdef MNEME_ENABLE_HIP
using DeviceVendorTraits = mneme::DeviceTraits<mneme::DeviceVendors::HIP>;
constexpr mneme::DeviceVendors Vendor = mneme::DeviceVendors::HIP;
#endif

#if defined(_MSC_VER)
#define HAVE_DECLSPEC_DLL
#endif

#if defined(HAVE_DECLSPEC_DLL)
#define API_EXPORT(RTYPE) __declspec(dllexport) RTYPE
#else
#define API_EXPORT(RTYPE) RTYPE
#endif

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

namespace mneme {
class MnemeRocProfiler {
private:
  rocprofiler_context_id_t rocrCtx{};
  rocprofiler_buffer_id_t rocrGBuf{};
  std::mutex rocrMutex;
  std::atomic<int64_t> rocrNextToken{0};
  std::unordered_map<int64_t, std::string> rocrTargetByToken;
  std::unordered_map<int64_t, std::vector<int64_t>> profilesByToken;
  std::unordered_map<rocprofiler_kernel_id_t, std::string> rocrKernelNames;

  // Signaled when a target kernel's duration arrives.
  std::condition_variable recordsCv;

  MnemeRocProfiler() {};

public:
  static MnemeRocProfiler &instance() {
    static MnemeRocProfiler Profiler;
    return Profiler;
  }

  // Callback to get the kernel-id->Name mapping
  void registerCodeObj(rocprofiler_callback_tracing_record_t Record) {
    if (Record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
      return;

    if (Record.operation ==
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
      auto *PayloadData = static_cast<
          const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
              *>(Record.payload);

      std::lock_guard<std::mutex> lk(rocrMutex);
      if (Record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
        rocrKernelNames[PayloadData->kernel_id] =
            PayloadData->kernel_name ? PayloadData->kernel_name : "<unknown>";
      }
    } else if (Record.operation == ROCPROFILER_CODE_OBJECT_LOAD &&
               Record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // ensure any deferred name lookups have been delivered
      (void)rocprofiler_flush_buffer(rocrGBuf);
    }
  }

  // Call back to pick duration of a kernel
  void logDuration(rocprofiler_record_header_t **headers, unsigned long n) {
    for (unsigned long i = 0; i < n; ++i) {
      auto *Record = static_cast<
          const rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
          headers[i]->payload);

      std::string KName{"<unknown>"};
      {
        std::lock_guard<std::mutex> lk(rocrMutex);
        if (auto it = rocrKernelNames.find(Record->dispatch_info.kernel_id);
            it != rocrKernelNames.end())
          KName = it->second;
      }

      int64_t token = Record->correlation_id.external.value;
      if (Record->end_timestamp >= Record->start_timestamp) {
        LOG_DEBUG("Received Log of {} associated with token {}", KName, token);
        int64_t dur = Record->end_timestamp - Record->start_timestamp;

        std::lock_guard<std::mutex> lk(rocrMutex);
        int64_t token2 = Record->correlation_id.external.value;
        LOG_DEBUG("dispatch: kernel_id={} name={} ext_token={}",
                  (uint64_t)Record->dispatch_info.kernel_id, KName, token2);

        auto tgt = rocrTargetByToken.find(token2);
        if (tgt == rocrTargetByToken.end())
          continue;

        const std::string &want = tgt->second;
        if (!want.empty() && KName != want)
          continue;

        LOG_DEBUG("Associating {} with token id {} and duration {}", KName, token2, dur);
        profilesByToken[token2].push_back(dur);
        recordsCv.notify_all();
      }
    }
  }

  int64_t start(const char *KernelName) {
    int64_t tok = rocrNextToken.fetch_add(1, std::memory_order_relaxed) + 1;
    rocprofiler_thread_id_t tid{};
    CHECK_ROCP(rocprofiler_get_thread_id(&tid));
    rocprofiler_user_data_t ud{};
    ud.value = tok;
    CHECK_ROCP(rocprofiler_push_external_correlation_id(rocrCtx, tid, ud));
    {
      std::lock_guard<std::mutex> lk(rocrMutex);
      rocrTargetByToken[tok] =
          KernelName ? std::string(KernelName) : std::string();
      profilesByToken.erase(tok);
    }
    LOG_DEBUG("Kernel {} matched with token {} assigned to thread id {}",
              KernelName, tok, tid);
    return tok;
  }

  // Waits until Expected durations arrived for Token; returns how many did.
  int64_t numRecords(int64_t Token, int64_t Expected) {
    auto EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::DeviceSynchronize());
    if (EC)
      LOG_FATAL("Error When Launching Kernel: " + EC.value());

    auto Deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::unique_lock<std::mutex> lk(rocrMutex);
    auto Have = [&]() -> int64_t {
      auto It = profilesByToken.find(Token);
      return It == profilesByToken.end() ? 0 : It->second.size();
    };
    while (Have() < Expected &&
           std::chrono::steady_clock::now() < Deadline) {
      // Flush may run buffer_cb on this thread, which takes rocrMutex.
      lk.unlock();
      CHECK_ROCP(rocprofiler_flush_buffer(rocrGBuf));
      lk.lock();
      recordsCv.wait_for(lk, std::chrono::milliseconds(10),
                         [&] { return Have() >= Expected; });
    }
    return Have();
  }

  std::vector<int64_t> stop(int64_t Token) {
    rocprofiler_thread_id_t tid{};
    CHECK_ROCP(rocprofiler_get_thread_id(&tid));
    rocprofiler_user_data_t popped{};
    CHECK_ROCP(rocprofiler_pop_external_correlation_id(rocrCtx, tid, &popped));

    auto EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::DeviceSynchronize());
    if (EC)
      LOG_FATAL("Error When Launching Kernel: " + EC.value());

    std::vector<int64_t> out;
    {
      std::lock_guard<std::mutex> lk(rocrMutex);
      if (auto it = profilesByToken.find(Token); it != profilesByToken.end()) {
        out = std::move(it->second);
        profilesByToken.erase(it);
      }
      rocrTargetByToken.erase(Token);
    }
    return out;
  }

  rocprofiler_buffer_id_t &getBuffer() { return rocrGBuf; }
  rocprofiler_context_id_t &getContext() { return rocrCtx; }
  MnemeRocProfiler(const MnemeRocProfiler &) = delete;
  MnemeRocProfiler &operator=(const MnemeRocProfiler &) = delete;
};
} // namespace mneme

static void buffer_cb(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                      rocprofiler_record_header_t **headers, unsigned long n,
                      void *, uint64_t) {
  auto &instance = mneme::MnemeRocProfiler::instance();
  LOG_DEBUG("Logging duration");
  instance.logDuration(headers, n);
}

static void codeobj_cb(rocprofiler_callback_tracing_record_t rec,
                       rocprofiler_user_data_t *data, void * /*data*/) {
  auto &instance = mneme::MnemeRocProfiler::instance();
  instance.registerCodeObj(rec);
}

namespace {
struct ToolData {};

int tool_init(rocprofiler_client_finalize_t /*fini*/, void *data_v) {
  // Create context/buffer/services during the configuration window
  LOG_DEBUG("Initializer ROCPrrofiler");
  auto &instance = mneme::MnemeRocProfiler::instance();
  CHECK_ROCP(rocprofiler_create_context(&instance.getContext()));
  CHECK_ROCP(rocprofiler_create_buffer(
      instance.getContext(), 256 * 1024, 0, ROCPROFILER_BUFFER_POLICY_LOSSLESS,
      buffer_cb, nullptr, &instance.getBuffer()));
  // Service to pick duration of kernel
  CHECK_ROCP(rocprofiler_configure_buffer_tracing_service(
      instance.getContext(), ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
      nullptr, 0, instance.getBuffer()));

  // Service to pick up kernel name -> kernel id and use it when profiling
  const rocprofiler_tracing_operation_t ops[] = {
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER,
      ROCPROFILER_CODE_OBJECT_LOAD};
  CHECK_ROCP(rocprofiler_configure_callback_tracing_service(
      instance.getContext(), ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, ops, 2,
      codeobj_cb, nullptr));

  CHECK_ROCP(rocprofiler_start_context(instance.getContext()));
  return 0;
}

void tool_fini(void * /*data_v*/) {
  LOG_DEBUG("Finalize ROCprofiler");
  auto &instance = mneme::MnemeRocProfiler::instance();
  (void)rocprofiler_stop_context(instance.getContext());
  (void)rocprofiler_flush_buffer(instance.getBuffer());
  (void)rocprofiler_destroy_buffer(instance.getBuffer());
}
} // namespace

extern "C" {
rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t version, const char *runtime_version,
                      uint32_t priority, rocprofiler_client_id_t *client_id) {
  static ToolData tool{};
  LOG_DEBUG("Configuring ROCPrrofiler");
  static rocprofiler_tool_configure_result_t result{};
  result.initialize = &tool_init; // creates context/buffer and starts it
  result.finalize = &tool_fini;   // cleanup at exit (or explicit finalize)
  result.tool_data = &tool;
  return &result;
}

API_EXPORT(int64_t) MnemePy_startProfile(const char *KernelName) {
  LOG_DEBUG("Requested to start profile {}", KernelName);
  auto &instance = mneme::MnemeRocProfiler::instance();
  return instance.start(KernelName);
}

API_EXPORT(void)
MnemePy_stopProfile(int64_t Token, int64_t *ProfileData, int64_t Size) {
  LOG_DEBUG("Requested to stop profile {}", Token);
  auto &instance = mneme::MnemeRocProfiler::instance();
  auto prof = instance.stop(Token);

  if (prof.size() != Size)
    LOG_WARN("Number of records differ between python container and C++ "
             "continaer size {} vs {}",
             Size, prof.size());

  LOG_DEBUG("Assigning measurements {} {}", Size, prof.size());
  for (int i = 0; i < std::min((size_t)Size, prof.size()); i++) {
    ProfileData[i] = prof[i];
  }
}

API_EXPORT(int64_t) MnemePy_getNumRecords(int64_t token, int64_t expected) {
  LOG_DEBUG("Requested to get number of profile records {}", token);
  auto &instance = mneme::MnemeRocProfiler::instance();
  auto records = instance.numRecords(token, expected);
  LOG_DEBUG("Record contains {} elements", records);
  return records;
}

API_EXPORT(void) MnemePy_initProfiler() {
  // Triggers rocprofiler to collect all rocprofiler_configure() symbols
  //  CHECK_ROCP(rocprofiler_force_configure(rocprofiler_configure));
}
}
