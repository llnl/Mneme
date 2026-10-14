#pragma once

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeKernelTimer.hpp"
#include "mneme/MnemeSnapshot.hpp"

#include <cmath>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

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

// Times launches with events recorded on the launch stream around each one.
// Pending launches are queued per stream because events on one stream
// complete in order, and events are reused once their duration is added.
template <DeviceVendors VendorTypes>
class EventLaunchTimer final : public LaunchTimer<VendorTypes> {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceEvent_t = typename MnemeDeviceRT::DeviceEvent_t;
  using typename LaunchTimer<VendorTypes>::DeviceStream_t;
  using typename LaunchTimer<VendorTypes>::AddFn;

  struct PendingLaunch {
    TimedLaunch Launch;
    DeviceEvent_t Start;
    DeviceEvent_t Stop;
  };

  std::vector<DeviceEvent_t> FreeEvents;
  std::unordered_map<DeviceStream_t, std::deque<PendingLaunch>> Pending;
  // The launch between begin() and end(), when its start event was recorded.
  std::optional<std::pair<DeviceStream_t, PendingLaunch>> Started;

  std::optional<DeviceEvent_t> recordEvent(DeviceStream_t Stream) {
    DeviceEvent_t Event;
    if (FreeEvents.empty()) {
      if (MnemeDeviceRT::deviceTimingEventCreate(&Event) !=
          MnemeDeviceRT::DeviceSuccess)
        return std::nullopt;
    } else {
      Event = FreeEvents.back();
      FreeEvents.pop_back();
    }

    if (MnemeDeviceRT::deviceEventRecord(Event, Stream) !=
        MnemeDeviceRT::DeviceSuccess) {
      FreeEvents.push_back(Event);
      return std::nullopt;
    }
    return Event;
  }

  // A launch whose events cannot be read stays untimed.
  void fold(const PendingLaunch &Launch, AddFn Add) {
    float Ms;
    if (MnemeDeviceRT::deviceEventElapsedTime(&Ms, Launch.Start, Launch.Stop) ==
        MnemeDeviceRT::DeviceSuccess)
      Add(Launch.Launch, std::llround(Ms * 1e6));
    FreeEvents.push_back(Launch.Start);
    FreeEvents.push_back(Launch.Stop);
  }

public:
  void collect(AddFn Add) override {
    for (auto It = Pending.begin(); It != Pending.end();) {
      auto &Launches = It->second;
      while (!Launches.empty() &&
             MnemeDeviceRT::deviceEventIsComplete(Launches.front().Stop)) {
        fold(Launches.front(), Add);
        Launches.pop_front();
      }
      It = Launches.empty() ? Pending.erase(It) : std::next(It);
    }
  }

  void begin(DeviceStream_t Stream, const TimedLaunch &Launch) override {
    if (auto Start = recordEvent(Stream))
      Started.emplace(Stream, PendingLaunch{Launch, *Start, {}});
  }

  void end(bool Launched) override {
    if (!Started)
      return;
    auto [Stream, Launch] = *Started;
    Started.reset();
    std::optional<DeviceEvent_t> Stop;
    if (Launched)
      Stop = recordEvent(Stream);
    if (!Stop) {
      FreeEvents.push_back(Launch.Start);
      return;
    }
    Launch.Stop = *Stop;
    Pending[Stream].push_back(Launch);
  }

  void drain(AddFn Add) override {
    for (auto &[Stream, Launches] : Pending)
      for (const PendingLaunch &Launch : Launches) {
        (void)MnemeDeviceRT::deviceEventSynchronize(Launch.Stop);
        fold(Launch, Add);
      }
    Pending.clear();

    for (DeviceEvent_t Event : FreeEvents)
      (void)MnemeDeviceRT::deviceEventDestroy(Event);
    FreeEvents.clear();
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
  case KernelTimingType::Events:
    return std::make_unique<EventLaunchTimer<VendorTypes>>();
  }
  return nullptr;
}

} // namespace mneme
