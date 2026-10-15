#include "mneme/MnemeRecordScope.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <type_traits>

using mneme::record_scope;
using mneme::detail::scopeAllowsRecording;

namespace {
void foo(int) {}
void bar() {}
void overloaded(int) {}
void overloaded(double) {}
template <typename T> void specialized(T) {}

void expect(bool Condition, const char *Message) {
  if (!Condition) {
    std::cerr << Message << '\n';
    std::exit(1);
  }
}
} // namespace

int main() {
  static_assert(!std::is_copy_constructible_v<record_scope>);
  static_assert(!std::is_copy_assignable_v<record_scope>);
  static_assert(!std::is_move_constructible_v<record_scope>);
  static_assert(!std::is_move_assignable_v<record_scope>);
  static_assert(std::is_nothrow_destructible_v<record_scope>);

  auto Foo = reinterpret_cast<const void *>(foo);
  auto Bar = reinterpret_cast<const void *>(bar);
  expect(!scopeAllowsRecording(Foo), "No scope must reject launches");

  {
    bool Enabled = true;
    record_scope Outer(Enabled);
    Enabled = false;
    expect(scopeAllowsRecording(Foo) && scopeAllowsRecording(Bar),
           "Wildcard must cover both kernels and retain its initial predicate");

    {
      record_scope Suppress(foo, false);
      expect(!scopeAllowsRecording(Foo) && scopeAllowsRecording(Bar),
             "Kernel-specific false scope must suppress only its kernel");
      {
        record_scope Inner(foo, true);
        expect(scopeAllowsRecording(Foo),
               "Innermost matching scope must override outer false");
      }
      expect(!scopeAllowsRecording(Foo),
             "Leaving inner scope must restore outer false");
    }

    expect(scopeAllowsRecording(Foo), "Leaving inner scope must restore state");

    try {
      record_scope Inner(false);
      expect(!scopeAllowsRecording(Foo),
             "False wildcard must suppress launches");
      throw std::runtime_error("unwind");
    } catch (const std::runtime_error &) {
    }
    expect(scopeAllowsRecording(Foo), "Exception unwinding must restore state");

    std::thread Worker([&] {
      expect(!scopeAllowsRecording(Foo), "Worker must not inherit scopes");
      record_scope Local(bar, true);
      expect(!scopeAllowsRecording(Foo) && scopeAllowsRecording(Bar),
             "Worker scope must select its own kernel");
    });
    Worker.join();
    expect(scopeAllowsRecording(Foo), "Worker must not change parent state");

    void (*NullKernel)() = nullptr;
    bool Threw = false;
    try {
      record_scope Invalid(NullKernel, true);
    } catch (const std::invalid_argument &) {
      Threw = true;
    }
    expect(Threw && scopeAllowsRecording(Foo),
           "Null selector must throw without modifying the stack");
  }

  expect(!scopeAllowsRecording(Foo),
         "Leaving all scopes must restore rejection");

  {
    record_scope Outer(false);
    record_scope Inner(foo, true);
    expect(scopeAllowsRecording(Foo) && !scopeAllowsRecording(Bar),
           "Inner kernel scope must override wildcard only for its kernel");
  }

  {
    record_scope Outer(foo, false);
    record_scope Inner(true);
    expect(scopeAllowsRecording(Foo) && scopeAllowsRecording(Bar),
           "Inner wildcard must override outer kernel scope");
  }

  {
    record_scope Outer(false);
    record_scope Inner(true);
    expect(scopeAllowsRecording(Foo) && scopeAllowsRecording(Bar),
           "Inner wildcard must override outer wildcard");
  }

  {
    record_scope OnlyFoo(foo, true);
    record_scope Unrelated(bar, false);
    expect(scopeAllowsRecording(Foo) && !scopeAllowsRecording(Bar),
           "Only matching scopes must participate");
  }

  {
    record_scope Template(specialized<int>, true);
    record_scope Overload(static_cast<void (*)(int)>(overloaded), true);
    expect(
        scopeAllowsRecording(reinterpret_cast<const void *>(specialized<int>)),
        "Template specialization must be selectable");
    expect(scopeAllowsRecording(reinterpret_cast<const void *>(
               static_cast<void (*)(int)>(overloaded))),
           "Disambiguated overload must be selectable");
    expect(!scopeAllowsRecording(reinterpret_cast<const void *>(
               static_cast<void (*)(double)>(overloaded))),
           "Other overload must not match");
    expect(!scopeAllowsRecording(
               reinterpret_cast<const void *>(specialized<double>)),
           "Different template specialization must not match");
  }

  return 0;
}
