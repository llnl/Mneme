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
#include <cupti.h>
#include <cupti_activity.h>

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

static inline const char* cuptiGetResultString(CUptiResult r) {
  const char* s = nullptr;
  cuptiGetResultString(r, &s);
  return s ? s : "<unknown>";
}

#define CHECK_CUPTI(call)                                                     \
do {                                                                        \
  CUptiResult _r = (call);                                                  \
  if (_r != CUPTI_SUCCESS) {                                                \
    fprintf(stderr, "CUPTI error %d (%s) at %s:%d: %s\n",                    \
            (int)_r, cuptiGetResultString(_r), __FILE__, __LINE__, #call);  \
                                                             std::abort();                                                           \
  }                                                                         \
} while (0)

// -----------------------------
// CUPTI buffer config
// -----------------------------
namespace {
constexpr size_t kBufSize      = 1 << 20;   // 1 MiB
constexpr size_t kBufAlign     = 8;
constexpr size_t kBufPoolCount = 8;         // small pool for async callbacks
constexpr CUpti_ExternalCorrelationKind kExtKind = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;
} // namespace

namespace mneme {

class MnemeCuptiProfiler {
private:
  std::mutex mtx_;
  std::atomic<u64> nextToken_{0};

  // Token -> *target* kernel name (empty means accept any)
  std::unordered_map<u64, std::string> targetByToken_;
  // Token -> durations (ns)
  std::unordered_map<u64, std::vector<int64_t>> profilesByToken_;
  // Signaled when a target kernel's duration arrives.
  std::condition_variable recordsCv_;

  // CUPTI correlationId (uint32) -> our token (u64)
  // Filled by CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION records.
  std::unordered_map<uint32_t, u64> corrIdToToken_;

  // simple pool of buffers to avoid malloc in callbacks
  std::vector<uint8_t*> bufPool_;
  size_t poolIdx_ = 0;

  bool initialized_ = false;

  MnemeCuptiProfiler() = default;

  static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    auto& inst = instance();
    std::lock_guard<std::mutex> lk(inst.mtx_);

    if (inst.bufPool_.empty()) {
      // Lazily create pool
      inst.bufPool_.reserve(kBufPoolCount);
      for (size_t i = 0; i < kBufPoolCount; ++i) {
        void* p = nullptr;
#if defined(_MSC_VER)
        p = _aligned_malloc(kBufSize, kBufAlign);
        if (!p) std::abort();
#else
        if (posix_memalign(&p, kBufAlign, kBufSize) != 0) std::abort();
#endif
        inst.bufPool_.push_back(reinterpret_cast<uint8_t*>(p));
      }
      inst.poolIdx_ = 0;
    }

    uint8_t* b = inst.bufPool_[inst.poolIdx_++ % inst.bufPool_.size()];
    *buffer = b;
    *size = kBufSize;
    *maxNumRecords = 0; // let CUPTI decide
  }

  static void CUPTIAPI bufferCompleted(CUcontext, uint32_t,
                                       uint8_t* buffer, size_t, size_t validSize)
  {
    auto& inst = instance();

    // dropped check
    size_t dropped = 0;
    CHECK_CUPTI(cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped));
    if (dropped) LOG_WARN("CUPTI dropped {} records", dropped);

    std::unordered_map<uint32_t, u64> localCorr;
    struct KernelSample { uint32_t corr; const char* name; uint64_t start; uint64_t end; bool ok; };
    std::vector<KernelSample> kernels;
    kernels.reserve(256);

    size_t nExt = 0, nKer = 0;

    CUpti_Activity* rec = nullptr;
    CUptiResult st = CUPTI_SUCCESS;

    while ((st = cuptiActivityGetNextRecord(buffer, validSize, &rec)) == CUPTI_SUCCESS) {
      if (!rec) break;

      switch (rec->kind) {
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
          auto* e = reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(rec);
          localCorr[e->correlationId] = (u64)e->externalId;
          nExt++;
          break;
        }

        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          auto* k = reinterpret_cast<const KernelRec*>(rec);
          kernels.push_back({k->correlationId, k->name, (uint64_t)k->start, (uint64_t)k->end, k->end >= k->start});
          nKer++;
          break;
        }

        default:
          break;
      }
    }

    if (st != CUPTI_SUCCESS && st != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      LOG_WARN("cuptiActivityGetNextRecord returned {}", (int)st);
    }


    // merge corr map
    if (!localCorr.empty()) {
      std::lock_guard<std::mutex> lk(inst.mtx_);
      for (auto& kv : localCorr) inst.corrIdToToken_[kv.first] = kv.second;
    }

    // associate kernels -> token
    for (const auto& s : kernels) {
      u64 token = 0;
      {
        std::lock_guard<std::mutex> lk(inst.mtx_);
        auto it = inst.corrIdToToken_.find(s.corr);
        if (it != inst.corrIdToToken_.end()) token = it->second;
      }

      if (!token) continue;

      u64 dur = (s.ok) ? (u64)(s.end - s.start) : 0;
      if (!dur) continue;

      // NOTE: don't construct std::string from s.name inside callback until you're sure it's safe.
      // For now, only do name filtering if want is empty.
      std::lock_guard<std::mutex> lk(inst.mtx_);

      auto tgtIt = inst.targetByToken_.find(token);
      if (tgtIt == inst.targetByToken_.end()) continue;

      const std::string& want = tgtIt->second;

      inst.profilesByToken_[token].push_back(dur);
      inst.recordsCv_.notify_all();
    }

    for (auto &PD : inst.profilesByToken_)
      for (auto &Dur : PD.second) 
        LOG_DEBUG("CUPTI Token: {} Received duration {}", PD.first, (int64_t) Dur);
    LOG_DEBUG("CUPTI Returning");
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

    // Activity setup
    CHECK_CUPTI(cuptiActivityRegisterCallbacks(&bufferRequested, &bufferCompleted));

    // Enable activities we need
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    // Some setups only emit KERNEL (non-concurrent); enabling doesn't hurt

    initialized_ = true;
    LOG_DEBUG("Initialized CUPTI activity profiling");
  }

  int64_t start(const char* kernelName) {
    initIfNeeded();
    u64 tok = nextToken_.fetch_add(1, std::memory_order_relaxed) + 1;

    // Push thread-scoped external correlation id
    CHECK_CUPTI(cuptiActivityPushExternalCorrelationId(kExtKind, static_cast<uint64_t>(tok)));

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

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::unique_lock<std::mutex> lk(mtx_);
    auto have = [&]() -> int64_t {
      auto it = profilesByToken_.find(token);
      return it == profilesByToken_.end() ? 0 : it->second.size();
    };
    while (have() < expected && std::chrono::steady_clock::now() < deadline) {
      // Flush runs bufferCompleted, which takes mtx_.
      lk.unlock();
      CHECK_CUPTI(cuptiActivityFlushAll(0));
      lk.lock();
      recordsCv_.wait_for(lk, std::chrono::milliseconds(10),
                          [&] { return have() >= expected; });
    }
    return have();
  }

  std::vector<int64_t> stop(u64 token) {
    initIfNeeded();

    // Pop the correlation id
    uint64_t popped = 0;
    CHECK_CUPTI(cuptiActivityPopExternalCorrelationId(kExtKind, &popped));
    if (popped != token) {
      LOG_WARN("CUPTI popped token {} but expected {}", (u64)popped, token);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUPTI(cuptiActivityFlushAll(0));

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

