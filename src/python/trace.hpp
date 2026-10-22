#pragma once

#include <cstdint>

namespace mneme::trace {

uint64_t timestamp() noexcept;

void complete(const char *Name, uint64_t Start, uint64_t End,
              const char *Args = nullptr, int64_t Iteration = -1) noexcept;

void flush() noexcept;
void disable(const char *Reason) noexcept;

// Native scopes only buffer events; serialization and I/O happen at flush().
class Scope {
  const char *Name;
  int64_t Iteration;
  uint64_t Start;
  int Owner;

public:
  explicit Scope(const char *Name, int64_t Iteration = -1) noexcept;
  ~Scope() noexcept;

  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
};

} // namespace mneme::trace
