#include "llvm/core.h"
#include <memory>
#include <mneme/MnemePython.hpp>
#include <mneme/MnemeUtils.hpp>
#include <string>
#include <utility>

using namespace mneme;
using namespace mneme::python;
using namespace llvm;
// namespace

// The prologue and epilogue may be passed in either argument order.
static std::pair<PrologueState<Vendor> *, EpilogueState<Vendor> *>
prologueAndEpilogue(MnemeDeviceMemStateRef v1, MnemeDeviceMemStateRef v2) {
  auto *S1 = unwrap(v1);
  auto *S2 = unwrap(v2);

  auto *Prologue = S1->asPrologue() ? S1->asPrologue() : S2->asPrologue();
  auto *Epilogue = S1->asEpilogue() ? S1->asEpilogue() : S2->asEpilogue();
  if (!Prologue || !Epilogue)
    LOG_FATAL("Comparing memory states expects one prologue and one epilogue");

  return {Prologue, Epilogue};
}

extern "C" {
API_EXPORT(MnemeDeviceMemStateRef)
MnemePy_initializeMemState(const char *KernelName, const char *fn,
                           const char *BasePrologueFn, bool isPrologue) {
  std::string BaseSnapshotName =
      BasePrologueFn == nullptr ? "" : std::string(BasePrologueFn);
  std::unique_ptr<DeviceMemState> state =
      isPrologue
          ? makeReplayPrologueState<Vendor>(KernelName, fn)
          : makeReplayEpilogueState<Vendor>(KernelName, fn, BaseSnapshotName);
  return wrap(state.release());
}

API_EXPORT(void) MnemePy_DisposeMemState(MnemeDeviceMemStateRef MemState) {
  if (MemState == nullptr)
    return;
  auto state = unwrap(MemState);
  delete state;
}

API_EXPORT(void) MnemePy_LoadMemState(MnemeDeviceMemStateRef MemState) {
  auto state = unwrap(MemState);
  state->load();
}

API_EXPORT(bool)
MnemePy_CompareMemState(MnemeDeviceMemStateRef v1, MnemeDeviceMemStateRef v2) {
  auto [Prologue, Epilogue] = prologueAndEpilogue(v1, v2);
  return Epilogue->matches(*Prologue);
}

API_EXPORT(bool)
MnemePy_MatchesUnlaunched(MnemeDeviceMemStateRef v1,
                          MnemeDeviceMemStateRef v2) {
  auto [Prologue, Epilogue] = prologueAndEpilogue(v1, v2);
  return Epilogue->matchesUnlaunched(*Prologue);
}

API_EXPORT(void)
MnemePy_ResetMemState(MnemeDeviceMemStateRef MemState) {
  auto state = unwrap(MemState);
  DeviceVendorTraits::DeviceStream_t DefaultStream{};
  state->reset(DefaultStream);
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceStreamSynchronize(DefaultStream));
  if (EC)
    LOG_FATAL("Could not finish resetting memory state: {}", EC.value());
}
}
