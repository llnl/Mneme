// clang-format off
// RUN: %PYTHON %S/check_record_scope.py %build/test_record_scope%ext MNEME_PRELOAD_LIB %t.scoped
// RUN: %PYTHON %S/check_record_config.py %build/test_record_scope%ext MNEME_PRELOAD_LIB %t.config
// clang-format on

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeRecordScope.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace mneme;
#ifdef MNEME_ENABLE_HIP
using RT = DeviceTraits<DeviceVendors::HIP>;
#else
using RT = DeviceTraits<DeviceVendors::CUDA>;
#endif

template <typename T> __global__ void foo(T *Data, int Iter) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    Data[0] += Iter + 1;
}

__global__ void bar(int *Data, int Iter) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    Data[1] += 2 * (Iter + 1);
}

__global__ void bar(float *Data, int Iter) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    Data[1] += 2 * (Iter + 1);
}

void check(RT::DeviceError_t Error) {
  if (RT::DeviceErrorCheck(Error))
    std::exit(1);
}

void advance_solution(int *Data, int Iter, int Grid, bool Indirect) {
  if (Indirect) {
    void *Args[] = {&Data, &Iter};
    check(RT::deviceLaunchKernel(reinterpret_cast<const void *>(foo<int>),
                                 dim3(Grid), dim3(1), Args, 0, nullptr));
  } else {
    foo<int><<<Grid, 1>>>(Data, Iter);
  }
  bar<<<Grid, 1>>>(Data, Iter);
}

int main(int Argc, char **Argv) {
  if (Argc != 2)
    return 1;

  std::string Scenario = Argv[1];
  int *Data = nullptr;
  check(RT::DeviceMalloc(reinterpret_cast<void **>(&Data), 2 * sizeof(int)));
  check(RT::DeviceMemset(Data, 0, 2 * sizeof(int)));

  for (int Iter = 0; Iter < 9; ++Iter) {
    const bool Selected =
        Scenario == "sampling" || Scenario == "limit" ? Iter >= 3 : Iter == 5;
    int Grid = 1;
    if (Scenario == "limit")
      Grid = Iter + 1;
    else if (Scenario == "mixed")
      Grid = Iter < 4 ? 1 : Iter < 6 ? 2 : 3;

    if (Scenario == "target" || Scenario == "indirect") {
      record_scope Scope(foo<int>, Selected);
      advance_solution(Data, Iter, Grid, Scenario == "indirect");
    } else if (Scenario == "overload") {
      record_scope Scope(static_cast<void (*)(int *, int)>(bar), Selected);
      advance_solution(Data, Iter, Grid, false);
    } else if (Scenario == "nested") {
      record_scope Outer(Selected);
      record_scope Suppress(foo<int>, false);
      advance_solution(Data, Iter, Grid, false);
    } else if (Scenario == "blocked") {
      record_scope Outer(true);
      record_scope Inner(false);
      advance_solution(Data, Iter, Grid, false);
    } else if (Scenario == "inner-wildcard") {
      record_scope Outer(false);
      record_scope Inner(Selected);
      advance_solution(Data, Iter, Grid, false);
    } else if (Scenario == "inner-kernel") {
      record_scope Outer(foo<int>, false);
      record_scope Inner(foo<int>, Selected);
      advance_solution(Data, Iter, Grid, false);
    } else if (Scenario == "none") {
      advance_solution(Data, Iter, Grid, false);
    } else {
      record_scope Scope(Selected);
      advance_solution(Data, Iter, Grid, false);
    }
    check(RT::DeviceSynchronize());
  }

  int Result[2] = {};
  check(RT::DeviceCopy(Result, Data, sizeof(Result),
                       RT::MemcpyDeviceToHostKind()));
  check(RT::DeviceFree(Data));
  if (Result[0] != 45 || Result[1] != 90) {
    std::cerr << "Filtered launches did not execute: " << Result[0] << ", "
              << Result[1] << '\n';
    return 1;
  }

  return 0;
}
