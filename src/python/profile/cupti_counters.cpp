// cupti_counters.cpp
//
// Minimal hardware-counter collector for Mneme replay (initial version).
//
// Reads one hard-coded counter with the CUPTI Range Profiler API around a
// replay pass. It is separate from the Activity-based timing tracer
// (MnemeDispatchTracerCUDA.cpp): while a counter session is open the Activity
// API receives no kernel records, so counters are collected in their own pass.
//
// Toolkits whose CUPTI lacks the Range Profiler headers build the stubs at the
// bottom, which report the collector as unavailable.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#if defined(_MSC_VER)
#define HAVE_DECLSPEC_DLL
#endif

#if defined(HAVE_DECLSPEC_DLL)
#define API_EXPORT(RTYPE) __declspec(dllexport) RTYPE
#else
#define API_EXPORT(RTYPE) RTYPE
#endif

#if defined(__has_include)
#if __has_include(<cupti_range_profiler.h>) && __has_include(<cupti_profiler_host.h>)
#define MNEME_HAS_RANGE_PROFILER 1
#endif
#endif

#ifdef MNEME_HAS_RANGE_PROFILER

#include <cupti.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>

namespace {

// Hard-coded for the initial version.
constexpr const char *kMetricName = "sm__ctas_launched.sum";

bool succeeded(CUptiResult Result, const char *Call) {
  if (Result == CUPTI_SUCCESS)
    return true;
  const char *Message = nullptr;
  cuptiGetResultString(Result, &Message);
  std::fprintf(stderr, "[mneme] counters: %s failed: %s\n", Call,
               Message ? Message : "<unknown>");
  return false;
}

CUcontext currentContext() {
  CUcontext Ctx = nullptr;
  if (cuCtxGetCurrent(&Ctx) == CUDA_SUCCESS && Ctx)
    return Ctx;
  cudaFree(nullptr); // makes the primary context current
  if (cuCtxGetCurrent(&Ctx) == CUDA_SUCCESS)
    return Ctx;
  return nullptr;
}

struct CounterState {
  bool Initialized = false;
  std::string ChipName;
  std::vector<const char *> Metrics{kMetricName};
  std::vector<uint8_t> Availability;
  std::vector<uint8_t> ConfigImage;
  std::vector<uint8_t> CounterData;
  CUpti_Profiler_Host_Object *Host = nullptr;
  CUpti_RangeProfiler_Object *Range = nullptr;
};

CounterState &state() {
  static CounterState S;
  return S;
}

void disableRange(CounterState &S) {
  if (!S.Range)
    return;
  CUpti_RangeProfiler_Disable_Params Disable = {
      CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
  Disable.pRangeProfilerObject = S.Range;
  succeeded(cuptiRangeProfilerDisable(&Disable), "cuptiRangeProfilerDisable");
  S.Range = nullptr;
}

// One-time setup per process: profiler, chip name, counter availability, and
// the config image for the metric. Returns 0 on success.
int initCounters(int Device) {
  CounterState &S = state();
  if (S.Initialized)
    return 0;

  CUcontext Ctx = currentContext();
  if (!Ctx) {
    std::fprintf(stderr, "[mneme] counters: no current CUDA context\n");
    return 1;
  }

  CUpti_Profiler_Initialize_Params Init = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  if (!succeeded(cuptiProfilerInitialize(&Init), "cuptiProfilerInitialize"))
    return 1;

  CUpti_Device_GetChipName_Params Chip = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  Chip.deviceIndex = Device;
  if (!succeeded(cuptiDeviceGetChipName(&Chip), "cuptiDeviceGetChipName"))
    return 1;
  S.ChipName = Chip.pChipName;

  // The availability image takes two calls: one for its size, one for the data.
  CUpti_Profiler_GetCounterAvailability_Params Avail = {
      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  Avail.ctx = Ctx;
  if (!succeeded(cuptiProfilerGetCounterAvailability(&Avail),
                 "cuptiProfilerGetCounterAvailability (size)"))
    return 1;
  S.Availability.resize(Avail.counterAvailabilityImageSize);
  Avail.pCounterAvailabilityImage = S.Availability.data();
  if (!succeeded(cuptiProfilerGetCounterAvailability(&Avail),
                 "cuptiProfilerGetCounterAvailability"))
    return 1;

  CUpti_Profiler_Host_Initialize_Params HostInit = {
      CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
  HostInit.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
  HostInit.pChipName = S.ChipName.c_str();
  HostInit.pCounterAvailabilityImage = S.Availability.data();
  if (!succeeded(cuptiProfilerHostInitialize(&HostInit),
                 "cuptiProfilerHostInitialize"))
    return 1;
  S.Host = HostInit.pHostObject;

  CUpti_Profiler_Host_ConfigAddMetrics_Params Add = {
      CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
  Add.pHostObject = S.Host;
  Add.ppMetricNames = S.Metrics.data();
  Add.numMetrics = S.Metrics.size();
  if (!succeeded(cuptiProfilerHostConfigAddMetrics(&Add),
                 "cuptiProfilerHostConfigAddMetrics"))
    return 1;

  CUpti_Profiler_Host_GetConfigImageSize_Params Size = {
      CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
  Size.pHostObject = S.Host;
  if (!succeeded(cuptiProfilerHostGetConfigImageSize(&Size),
                 "cuptiProfilerHostGetConfigImageSize"))
    return 1;
  S.ConfigImage.resize(Size.configImageSize);

  CUpti_Profiler_Host_GetConfigImage_Params Image = {
      CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
  Image.pHostObject = S.Host;
  Image.pConfigImage = S.ConfigImage.data();
  Image.configImageSize = S.ConfigImage.size();
  if (!succeeded(cuptiProfilerHostGetConfigImage(&Image),
                 "cuptiProfilerHostGetConfigImage"))
    return 1;

  S.Initialized = true;
  return 0;
}

// Enables the range profiler on Ctx and sets it up for up to MaxRanges
// launches. On failure the caller disables the profiler again.
bool beginSession(CounterState &S, CUcontext Ctx, size_t MaxRanges) {
  CUpti_RangeProfiler_Enable_Params Enable = {
      CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
  Enable.ctx = Ctx;
  if (!succeeded(cuptiRangeProfilerEnable(&Enable), "cuptiRangeProfilerEnable"))
    return false;
  S.Range = Enable.pRangeProfilerObject;

  CUpti_RangeProfiler_GetCounterDataSize_Params Size = {
      CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
  Size.pRangeProfilerObject = S.Range;
  Size.pMetricNames = S.Metrics.data();
  Size.numMetrics = S.Metrics.size();
  Size.maxNumOfRanges = MaxRanges;
  Size.maxNumRangeTreeNodes = MaxRanges;
  if (!succeeded(cuptiRangeProfilerGetCounterDataSize(&Size),
                 "cuptiRangeProfilerGetCounterDataSize"))
    return false;
  S.CounterData.assign(Size.counterDataSize, 0);

  CUpti_RangeProfiler_CounterDataImage_Initialize_Params DataInit = {
      CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  DataInit.pRangeProfilerObject = S.Range;
  DataInit.pCounterData = S.CounterData.data();
  DataInit.counterDataSize = S.CounterData.size();
  if (!succeeded(cuptiRangeProfilerCounterDataImageInitialize(&DataInit),
                 "cuptiRangeProfilerCounterDataImageInitialize"))
    return false;

  CUpti_RangeProfiler_SetConfig_Params Config = {
      CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
  Config.pRangeProfilerObject = S.Range;
  Config.pConfig = S.ConfigImage.data();
  Config.configSize = S.ConfigImage.size();
  Config.pCounterDataImage = S.CounterData.data();
  Config.counterDataImageSize = S.CounterData.size();
  Config.maxRangesPerPass = MaxRanges;
  Config.numNestingLevels = 1;
  Config.minNestingLevel = 1;
  Config.passIndex = 0;
  Config.targetNestingLevel = 1;
  Config.range = CUPTI_AutoRange;
  Config.replayMode = CUPTI_KernelReplay;
  if (!succeeded(cuptiRangeProfilerSetConfig(&Config),
                 "cuptiRangeProfilerSetConfig"))
    return false;

  CUpti_RangeProfiler_Start_Params Start = {
      CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
  Start.pRangeProfilerObject = S.Range;
  if (!succeeded(cuptiRangeProfilerStart(&Start), "cuptiRangeProfilerStart"))
    return false;

  // Auto-range profiling records nothing unless the launches sit inside a range.
  CUpti_RangeProfiler_PushRange_Params Push = {
      CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
  Push.pRangeProfilerObject = S.Range;
  Push.pRangeName = "MnemeCounterPass";
  return succeeded(cuptiRangeProfilerPushRange(&Push),
                   "cuptiRangeProfilerPushRange");
}

// Opens a counter session for up to MaxRanges kernel launches (one range each).
// Returns 0 on success.
int startCounters(int64_t MaxRanges) {
  CounterState &S = state();
  if (!S.Initialized) {
    std::fprintf(stderr, "[mneme] counters: not initialized\n");
    return 1;
  }
  if (S.Range) {
    std::fprintf(stderr, "[mneme] counters: a session is already open\n");
    return 1;
  }
  CUcontext Ctx = currentContext();
  if (!Ctx) {
    std::fprintf(stderr, "[mneme] counters: no current CUDA context\n");
    return 1;
  }
  if (!beginSession(S, Ctx, static_cast<size_t>(MaxRanges))) {
    disableRange(S);
    return 1;
  }
  return 0;
}

// Closes the session and writes one value per recorded range into Values (at
// most Capacity). Returns the number of ranges CUPTI recorded, which can exceed
// Capacity, or -1 on failure.
int64_t stopCounters(double *Values, int64_t Capacity) {
  CounterState &S = state();
  if (!S.Range) {
    std::fprintf(stderr, "[mneme] counters: no session is open\n");
    return -1;
  }

  cudaDeviceSynchronize();

  CUpti_RangeProfiler_PopRange_Params Pop = {
      CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
  Pop.pRangeProfilerObject = S.Range;
  if (!succeeded(cuptiRangeProfilerPopRange(&Pop), "cuptiRangeProfilerPopRange")) {
    disableRange(S);
    return -1;
  }

  CUpti_RangeProfiler_Stop_Params Stop = {
      CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
  Stop.pRangeProfilerObject = S.Range;
  if (!succeeded(cuptiRangeProfilerStop(&Stop), "cuptiRangeProfilerStop")) {
    disableRange(S);
    return -1;
  }
  if (!Stop.isAllPassSubmitted)
    std::fprintf(stderr, "[mneme] counters: warning: not all passes were submitted\n");

  CUpti_RangeProfiler_DecodeData_Params Decode = {
      CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
  Decode.pRangeProfilerObject = S.Range;
  if (!succeeded(cuptiRangeProfilerDecodeData(&Decode),
                 "cuptiRangeProfilerDecodeData")) {
    disableRange(S);
    return -1;
  }

  CUpti_RangeProfiler_GetCounterDataInfo_Params Info = {
      CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
  Info.pCounterDataImage = S.CounterData.data();
  Info.counterDataImageSize = S.CounterData.size();
  if (!succeeded(cuptiRangeProfilerGetCounterDataInfo(&Info),
                 "cuptiRangeProfilerGetCounterDataInfo")) {
    disableRange(S);
    return -1;
  }

  for (size_t I = 0; I < Info.numTotalRanges; ++I) {
    double Value = 0;
    CUpti_Profiler_Host_EvaluateToGpuValues_Params Eval = {
        CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    Eval.pHostObject = S.Host;
    Eval.pCounterDataImage = S.CounterData.data();
    Eval.counterDataImageSize = S.CounterData.size();
    Eval.ppMetricNames = S.Metrics.data();
    Eval.numMetrics = S.Metrics.size();
    Eval.rangeIndex = I;
    Eval.pMetricValues = &Value;
    if (!succeeded(cuptiProfilerHostEvaluateToGpuValues(&Eval),
                   "cuptiProfilerHostEvaluateToGpuValues")) {
      disableRange(S);
      return -1;
    }
    if (static_cast<int64_t>(I) < Capacity)
      Values[I] = Value;
  }

  disableRange(S);
  return static_cast<int64_t>(Info.numTotalRanges);
}

} // namespace

extern "C" {

API_EXPORT(int) MnemePy_counterAvailable() { return 1; }

API_EXPORT(const char *) MnemePy_counterMetricName() { return kMetricName; }

API_EXPORT(int) MnemePy_counterInit(int Device) { return initCounters(Device); }

API_EXPORT(int) MnemePy_counterStart(int64_t MaxRanges) {
  return startCounters(MaxRanges);
}

API_EXPORT(int64_t) MnemePy_counterStop(double *Values, int64_t Capacity) {
  return stopCounters(Values, Capacity);
}

} // extern "C"

#else // !MNEME_HAS_RANGE_PROFILER

extern "C" {

API_EXPORT(int) MnemePy_counterAvailable() { return 0; }

API_EXPORT(const char *) MnemePy_counterMetricName() { return ""; }

API_EXPORT(int) MnemePy_counterInit(int) { return 1; }

API_EXPORT(int) MnemePy_counterStart(int64_t) { return 1; }

API_EXPORT(int64_t) MnemePy_counterStop(double *, int64_t) { return -1; }

} // extern "C"

#endif // MNEME_HAS_RANGE_PROFILER
