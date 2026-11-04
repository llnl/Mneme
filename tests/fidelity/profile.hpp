#pragma once

#include <cstdint>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

// Load before the first runtime call so ROCprofiler can discover its tool.
// Use the same GPU dispatch timestamps as Mneme replay.
class Profile {
  void *library = nullptr;
  int64_t token = 0;

  using Start = int64_t (*)(const char *);
  using Count = int64_t (*)(uint64_t, int64_t);
  using Stop = void (*)(uint64_t, int64_t *, uint64_t);

  Start start = nullptr;
  Count count = nullptr;
  Stop stop = nullptr;

  template <typename T> T symbol(const char *name) {
    auto ptr = dlsym(library, name);
    if (!ptr)
      throw std::runtime_error(std::string("Missing profiler symbol: ") + name);
    return reinterpret_cast<T>(ptr);
  }

public:
  Profile(const char *path) {
    // Recording runs use this adapter without enabling native profiling.
    if (!path)
      return;

    library = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!library)
      throw std::runtime_error(dlerror());

    start = symbol<Start>("MnemePy_startProfile");
    count = symbol<Count>("MnemePy_getNumRecords");
    stop = symbol<Stop>("MnemePy_stopProfile");
    symbol<void (*)()>("MnemePy_initProfiler")();
    // Do not dlclose: GPU runtime callbacks can outlive this object's scope.
  }

  void begin(const char *name) {
    if (start)
      token = start(name);
  }

  std::vector<int64_t> finish() {
    if (!start)
      return {};

    // The window contains only the fifth target dispatch. This also works when
    // the profiler backend does not filter dispatches by kernel name.
    const auto n = count(token, 1);
    if (n != 1)
      throw std::runtime_error(
          "Expected exactly one target dispatch, received " +
          std::to_string(n));

    std::vector<int64_t> times(n);
    stop(token, times.data(), times.size());
    for (auto value : times)
      if (value <= 0)
        throw std::runtime_error("Nonpositive dispatch duration");

    return times;
  }
};
