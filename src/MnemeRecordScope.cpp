#include "mneme/MnemeRecordScope.hpp"

#include <cassert>
#include <stdexcept>

namespace mneme {

thread_local record_scope *record_scope::Top = nullptr;

record_scope::record_scope(bool Enabled) noexcept
    : Kernel(nullptr), Enabled(Enabled), Previous(Top) {
  Top = this;
}

record_scope::record_scope(const void *Kernel, bool Enabled)
    : Kernel(Kernel), Enabled(Enabled), Previous(Top) {
  if (!Kernel)
    throw std::invalid_argument(
        "mneme::record_scope requires a non-null kernel");
  Top = this;
}

record_scope::~record_scope() noexcept {
  assert(Top == this && "record_scope must unwind on its constructing thread");
  Top = Previous;
}

bool detail::scopeAllowsRecording(const void *Kernel) noexcept {
  bool Matched = false;
  for (auto *Scope = record_scope::Top; Scope; Scope = Scope->Previous) {
    if (Scope->Kernel && Scope->Kernel != Kernel)
      continue;
    if (!Scope->Enabled)
      return false;
    Matched = true;
  }
  return Matched;
}

} // namespace mneme
