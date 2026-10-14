#pragma once

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeKernelTimer.hpp"
#include "mneme/MnemeSnapshot.hpp"

#include <cstdint>
#include <memory>

#include <llvm/ADT/STLFunctionalExtras.h>

namespace mneme {

// Measures the GPU durations of kernel launches. Durations may arrive after
// later launches, so they are handed out by collect() and drain().
template <DeviceVendors VendorTypes> class LaunchTimer {
public:
  using DeviceStream_t = typename DeviceTraits<VendorTypes>::DeviceStream_t;
  using AddFn = llvm::function_ref<void(const TimedLaunch &, uint64_t Ns)>;

  virtual ~LaunchTimer() = default;

  // Adds the durations that have arrived.
  virtual void collect(AddFn Add) = 0;
  virtual void begin(DeviceStream_t Stream, const TimedLaunch &Launch) = 0;
  virtual void end(bool Launched) = 0;
  // Waits for the durations of all launches and adds them.
  virtual void drain(AddFn Add) = 0;
};

// The vendor profiler's callbacks reach the process-wide KernelTimer, so this
// only forwards to it.
template <DeviceVendors VendorTypes>
class ProfilerLaunchTimer final : public LaunchTimer<VendorTypes> {
  using typename LaunchTimer<VendorTypes>::DeviceStream_t;
  using typename LaunchTimer<VendorTypes>::AddFn;

  KernelTimer &Profiler;

public:
  explicit ProfilerLaunchTimer(KernelTimer &Profiler) : Profiler(Profiler) {}

  void collect(AddFn Add) override { Profiler.collect(Add); }

  void begin(DeviceStream_t, const TimedLaunch &Launch) override {
    Profiler.begin(Launch);
  }

  void end(bool Launched) override { Profiler.end(Launched); }

  // The profiler reports a dispatch only after it completes.
  void drain(AddFn Add) override {
    (void)DeviceTraits<VendorTypes>::DeviceSynchronize();
    Profiler.drain(Add);
  }
};

// Returns nullptr when timing is off or unavailable.
template <DeviceVendors VendorTypes>
std::unique_ptr<LaunchTimer<VendorTypes>> createLaunchTimer(bool KernelTiming) {
  if (!KernelTiming)
    return nullptr;
  KernelTimer *Profiler = KernelTimer::get();
  if (!Profiler)
    return nullptr;
  return std::make_unique<ProfilerLaunchTimer<VendorTypes>>(*Profiler);
}

} // namespace mneme
