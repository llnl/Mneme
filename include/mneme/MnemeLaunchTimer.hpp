#pragma once

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeKernelTimer.hpp"
#include "mneme/MnemeSnapshot.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringRef.h>

namespace mneme {

// Measures the GPU durations of kernel launches. Durations may arrive after
// later launches, so they are handed out by collect() and drain().
template <DeviceVendors VendorTypes> class LaunchTimer {
public:
  using AddFn = llvm::function_ref<void(const TimedLaunch &, uint64_t Ns)>;
  using DeviceError_t = typename DeviceTraits<VendorTypes>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<VendorTypes>::DeviceStream_t;

  virtual ~LaunchTimer() = default;

  // Adds the durations that have arrived.
  virtual void collect(AddFn Add) = 0;
  // Returns whether the launch on Stream is timed, in which case end() must
  // follow it with the launch's result.
  virtual bool begin(const TimedLaunch &Launch, llvm::StringRef KernelName,
                     DeviceStream_t Stream) = 0;
  virtual void end(DeviceError_t LaunchResult) = 0;
  // Waits for the durations of all launches and adds them.
  virtual void drain(AddFn Add) = 0;
};

// The vendor profiler's callbacks reach the process-wide KernelTimer, so this
// only forwards to it.
template <DeviceVendors VendorTypes>
class ProfilerLaunchTimer final : public LaunchTimer<VendorTypes> {
  using DevTraits = DeviceTraits<VendorTypes>;
  using typename LaunchTimer<VendorTypes>::AddFn;
  using typename LaunchTimer<VendorTypes>::DeviceError_t;
  using typename LaunchTimer<VendorTypes>::DeviceStream_t;

  KernelTimer &Profiler;
  DeviceError_t CaptureQueryResult = DevTraits::DeviceSuccess;

public:
  explicit ProfilerLaunchTimer(KernelTimer &Profiler) : Profiler(Profiler) {}

  void collect(AddFn Add) override { Profiler.collect(Add); }

  bool begin(const TimedLaunch &Launch, llvm::StringRef KernelName,
             DeviceStream_t Stream) override {
    bool Capturing;
    CaptureQueryResult = DevTraits::deviceStreamIsCapturing(Stream, Capturing);
    // A captured launch dispatches nothing now, and its graph's later
    // dispatches carry no token, so it would never get a duration.
    if (Capturing)
      return false;
    return Profiler.begin(Launch, KernelName);
  }

  // The capture query fails only where the launch should fail the same way,
  // such as on the legacy stream while another stream captures, so a launch
  // that succeeds anyway may have been captured instead of dispatched.
  void end(DeviceError_t LaunchResult) override {
    bool Launched = LaunchResult == DevTraits::DeviceSuccess;
    if (Launched && CaptureQueryResult != DevTraits::DeviceSuccess) {
      std::cerr << "[mneme] Kernel timing cannot tell whether a launch was "
                   "captured into a graph: "
                << *DevTraits::DeviceErrorCheck(CaptureQueryResult) << "\n";
      std::abort();
    }
    Profiler.end(Launched);
  }

  // The profiler reports a dispatch only after it completes.
  void drain(AddFn Add) override {
    (void)DeviceTraits<VendorTypes>::DeviceSynchronize();
    Profiler.drain(Add);
  }
};

// Returns nullptr when timing is off or the selected timer is unavailable.
template <DeviceVendors VendorTypes>
std::unique_ptr<LaunchTimer<VendorTypes>>
createLaunchTimer(KernelTimingType Type) {
  switch (Type) {
  case KernelTimingType::Off:
    return nullptr;
  case KernelTimingType::Profiler:
    if (KernelTimer *Profiler = KernelTimer::get())
      return std::make_unique<ProfilerLaunchTimer<VendorTypes>>(*Profiler);
    return nullptr;
  }
  return nullptr;
}

} // namespace mneme
