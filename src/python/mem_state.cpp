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

static EpilogueState<Vendor> *epilogue(MnemeDeviceMemStateRef State) {
  auto *Epilogue = unwrap(State)->asEpilogue();
  if (!Epilogue)
    LOG_FATAL("Expected an epilogue memory state");
  return Epilogue;
}

// Mirrors CompareMetadata in recorded_execution.py.
struct MnemePyCompareMetadata {
  void *DevAddr;
  uint8_t DType;
  double Threshold;
  uint8_t ThresholdKind;
  uint8_t Norm;
  const char *Tag;
};

static CompareRules compareRules(const MnemePyCompareMetadata *Rules,
                                 uint64_t NumRules) {
  CompareRules Out;
  for (uint64_t I = 0; I < NumRules; ++I) {
    const auto &In = Rules[I];
    mneme::Metadata Md;
    Md.builtin = static_cast<BuiltinDType>(In.DType);
    Md.threshold = In.Threshold;
    Md.threshold_kind = static_cast<ThresholdKind>(In.ThresholdKind);
    Md.norm = static_cast<Norm>(In.Norm);
    if (In.Tag)
      Md.tag = std::string(In.Tag);
    if (!Out.try_emplace(In.DevAddr, std::move(Md)).second)
      LOG_FATAL("Duplicate comparison rule for blob " +
                util::pointerToHexString(In.DevAddr));
  }
  return Out;
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
MnemePy_CompareMemState(MnemeDeviceMemStateRef v1, MnemeDeviceMemStateRef v2,
                        const MnemePyCompareMetadata *Rules,
                        uint64_t NumRules) {
  auto [Prologue, Epilogue] = prologueAndEpilogue(v1, v2);
  return Epilogue->matches(*Prologue, compareRules(Rules, NumRules));
}

API_EXPORT(bool)
MnemePy_MatchesUnlaunched(MnemeDeviceMemStateRef v1, MnemeDeviceMemStateRef v2,
                          const MnemePyCompareMetadata *Rules,
                          uint64_t NumRules) {
  auto [Prologue, Epilogue] = prologueAndEpilogue(v1, v2);
  return Epilogue->matchesUnlaunched(*Prologue, compareRules(Rules, NumRules));
}

API_EXPORT(void)
MnemePy_ResetMemState(MnemeDeviceMemStateRef MemState) {
  auto state = unwrap(MemState);
  state->reset();
}

// Fills at most Capacity entries and returns the number of blobs. Tags point
// into the state.
API_EXPORT(uint64_t)
MnemePy_getCompareMetadata(MnemeDeviceMemStateRef State,
                           MnemePyCompareMetadata *Out, uint64_t Capacity) {
  uint64_t Count = 0;
  for (const auto &[DevAddr, Blob] : epilogue(State)->getDeviceMemory()) {
    if (Count < Capacity) {
      const mneme::Metadata &Md = Blob.getMetadata();
      Out[Count] = {DevAddr,
                    static_cast<uint8_t>(Md.builtin),
                    Md.threshold,
                    static_cast<uint8_t>(Md.threshold_kind),
                    static_cast<uint8_t>(Md.norm),
                    Md.tag ? Md.tag->c_str() : nullptr};
    }
    ++Count;
  }
  return Count;
}
}
