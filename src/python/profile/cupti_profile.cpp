// MnemeCuptiProfiler.cpp
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

#include "mneme/MnemeDispatchTracer.hpp"
#include "mneme/MnemeLogger.hpp"

#if defined(_MSC_VER)
#define HAVE_DECLSPEC_DLL
#endif

#if defined(HAVE_DECLSPEC_DLL)
#define API_EXPORT(RTYPE) __declspec(dllexport) RTYPE
#else
#define API_EXPORT(RTYPE) RTYPE
#endif

using u64 = unsigned long long;

// -----------------------------
// Error handling helpers
// -----------------------------
#define CHECK_CUDA(call)                                                      \
do {                                                                        \
  cudaError_t _e = (call);                                                  \
  if (_e != cudaSuccess) {                                                  \
    fprintf(stderr, "CUDA error %d (%s) at %s:%d: %s\n",                     \
            (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__, #call);    \
                                                             std::abort();                                                           \
  }                                                                         \
} while (0)

namespace mneme {

class MnemeCuptiProfiler {
private:
  DispatchTracer tracer_;
  std::mutex mtx_;
  std::atomic<u64> nextToken_{0};

  // Token -> *target* kernel name (empty means accept any)
  std::unordered_map<u64, std::string> targetByToken_;
  // Token -> durations (ns)
  std::unordered_map<u64, std::vector<int64_t>> profilesByToken_;
  // Signaled when a target kernel's duration arrives.
  std::condition_variable recordsCv_;

  bool initialized_ = false;

  MnemeCuptiProfiler() = default;

  static void recordDuration(void *, const DispatchRecord &Record) {
    auto& inst = instance();
    if (Record.End <= Record.Start)
      return;
    u64 dur = Record.End - Record.Start;

    std::lock_guard<std::mutex> lk(inst.mtx_);
    if (inst.targetByToken_.find(Record.Token) == inst.targetByToken_.end())
      return;

    inst.profilesByToken_[Record.Token].push_back(dur);
    inst.recordsCv_.notify_all();
    LOG_DEBUG("CUPTI Token: {} Received duration {}", Record.Token,
              (int64_t)dur);
  }

public:
  static MnemeCuptiProfiler& instance() {
    static MnemeCuptiProfiler inst;
    return inst;
  }

  void initIfNeeded() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (initialized_) return;

    // Make sure CUDA is initialized
    CHECK_CUDA(cudaFree(nullptr));

    tracer_.create(&recordDuration, nullptr);

    initialized_ = true;
    LOG_DEBUG("Initialized CUPTI activity profiling");
  }

  int64_t start(const char* kernelName) {
    initIfNeeded();
    u64 tok = nextToken_.fetch_add(1, std::memory_order_relaxed) + 1;

    // Push thread-scoped external correlation id
    tracer_.push(tok);

    {
      std::lock_guard<std::mutex> lk(mtx_);
      targetByToken_[tok] = kernelName ? std::string(kernelName) : std::string();
      profilesByToken_.erase(tok);
    }

    LOG_DEBUG("CUPTI startProfile kernel='{}' token={}", kernelName ? kernelName : "<any>", tok);
    return tok;
  }

  // Waits until expected durations arrived for token; returns how many did.
  int64_t numRecords(u64 token, int64_t expected) {
    initIfNeeded();
    CHECK_CUDA(cudaDeviceSynchronize());

    std::unique_lock<std::mutex> lk(mtx_);
    auto have = [&]() -> int64_t {
      auto it = profilesByToken_.find(token);
      return it == profilesByToken_.end() ? 0 : it->second.size();
    };
    waitForRecords(
        lk, recordsCv_, [&] { CHECK_CUPTI(tracer_.flush()); },
        [&] { return have() >= expected; }, RecordWaitTimeout);
    return have();
  }

  std::vector<int64_t> stop(u64 token) {
    initIfNeeded();

    // Pop the correlation id
    uint64_t popped = tracer_.pop();
    if (popped != token) {
      LOG_WARN("CUPTI popped token {} but expected {}", (u64)popped, token);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUPTI(tracer_.flush());

    std::vector<int64_t> out;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (auto it = profilesByToken_.find(token); it != profilesByToken_.end()) {
        out = std::move(it->second);
        profilesByToken_.erase(it);
      }
      targetByToken_.erase(token);
    }
    LOG_DEBUG("CUPTI stopProfile token={} records={}", token, out.size());
    return out;
  }

  MnemeCuptiProfiler(const MnemeCuptiProfiler&) = delete;
  MnemeCuptiProfiler& operator=(const MnemeCuptiProfiler&) = delete;
};

} // namespace mneme

// -----------------------------
// C ABI entrypoints (match your HIP ones)
// -----------------------------
extern "C" {

API_EXPORT(int64_t) MnemePy_startProfile(const char* KernelName) {
  auto& inst = mneme::MnemeCuptiProfiler::instance();
  return inst.start(KernelName);
}

API_EXPORT(void) MnemePy_stopProfile(u64 Token, int64_t* ProfileData, u64 Size) {
  auto& inst = mneme::MnemeCuptiProfiler::instance();
  auto prof = inst.stop(Token);

  if (prof.size() != Size) {
    LOG_WARN("Number of records differ between python container and C++ container size {} vs {}",
             Size, prof.size());
  }

  for (u64 i = 0; i < std::min<u64>(Size, static_cast<u64>(prof.size())); ++i) {
    ProfileData[i] = prof[i];
  }
}

API_EXPORT(int64_t) MnemePy_getNumRecords(u64 token, int64_t expected) {
  auto& inst = mneme::MnemeCuptiProfiler::instance();
  return inst.numRecords(token, expected);
}

API_EXPORT(void) MnemePy_initProfiler() {
  auto& inst = mneme::MnemeCuptiProfiler::instance();
  inst.initIfNeeded();
}

} // extern "C"

