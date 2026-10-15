# Record

The `mneme record` command executes an application while transparently
capturing GPU kernel executions and the associated device memory state.
For full CLI details, see [CLI → mneme record](cli.md#mneme-record).

Recording does not require modifying application source code.  However,
if you want **tolerance-aware verification** during replay, you can
optionally annotate device pointers with comparison metadata before the
kernel launches that will be recorded.

## Selecting which launches are recorded

Mneme supports several ways to filter which kernel launches get recorded
and which do not.

_Scoped_ filtering uses a Mneme runtime API to enable/disable recording.
It is best used when application state determines which invocation
should be recorded.

```cpp
#include "mneme/MnemeRecordScope.hpp"

for (int iter = 0; iter < iterations; ++iter) {
    mneme::record_scope scope(iter == 5);
    advance_solution();  // All kernels submitted here are eligible at iteration 5.
}
```

Run the instrumented application with:

```bash
mneme record --filter-mode scoped -- ./app
```

The default mode, `default`, records normally and ignores scopes.
Invalid modes are rejected. Scoped mode records nothing without an enabled scope.

To select a particular kernel, pass its function instead:

```cpp
{
  mneme::record_scope scope(foo, iter == 5);
  foo<<<grid, block>>>(data);
  bar<<<grid, block>>>(data);  // This scope does not select bar.
}
foo<<<grid, block>>>(data); // This does not record, since `scope` is out-of-scope.
```

Kernel selectors use function addresses, not names. Template specializations
such as `foo<double>` are supported; overloaded functions require an explicit
function-pointer cast. Null pointers throw `std::invalid_argument`.
The **innermost matching scope** decides whether a launch is eligible for
recording. A scope without a kernel matches every kernel; a kernel-specific scope
matches only its selected kernel. Without a matching scope, the launch is not recorded.

```cpp
mneme::record_scope outer_scope(foo, false);
{
  mneme::record_scope inner_scope(foo, true);
  foo<<<grid, block>>>(data);  // recorded
}
foo<<<grid, block>>>(data);  // not recorded
```

Scopes cover host launch submission on the calling thread, including ordinary
helper calls. Worker threads need their own scopes. Scope entry and exit do
not synchronize the GPU. Memory allocated before a scope remains available
for recording. All kernels still execute whether or not they are recorded.

## Selecting launches with a configuration file

```bash
mneme record --filter-mode config --filter-config filter.yaml -- ./app
# Equivalent shorthand:
mneme record --filter-config filter.yaml -- ./app
```

Config mode ignores application scopes. It is an error to pass 
`--filter-config` if the mode is set to scope.
With neither `--filter-mode` nor `--filter-config`, the CLI selects `default`.

```yaml
recording:
  defaults:
    skip: 0
    every: 1
    max_records: 1
    ranks: [0, 2]              # or: all; [] disables all ranks
  kernel_overrides:
    - match:
        demangled_regex: "^foo\\("
      skip: 5
    - match:
        demangled_regex: "^bar\\("
      skip: 1
      every: 10
      max_records: 3
```

`recording` is required, but all fields inside it are optional.
The built-in defaults are `skip: 0`, `every: 1`, and `max_records: 4`.
Mneme uses these by default, unless `defaults` overrides them or
the fields supplied by the **first matching override** change them.

### Default-mode CLI and environment controls

Default mode accepts some overrides from CLI and/or environment variables:

| CLI option | Environment variable | Default |
| --- | --- | --- |
| `-sr` / `--per-kernel-skip-recordings` | `MNEME_SKIP_RECORDINGS` | `0` |
| `-mr` / `--per-kernel-max-recordings` | `MNEME_MAX_RECORDINGS` | `4` |
| — | `MNEME_RR_KERNELS` | All kernels |

They apply only in `--filter-mode default`. An explicit CLI count overrides its
environment variable, including explicit zero or four. The below example would
record the `foo` kernel, skip its first five invocations, and record a maximum
of two unique configurations of `foo`.

```bash
MNEME_RR_KERNELS='foo' mneme record -sr 5 -mr 2 -- ./app
```

In `scoped` and `config` modes, these count options and environment variables
are **ignored with a warning**.

## Annotating buffers for verification

By default, replay verification uses exact byte-wise comparison between
the replayed and recorded epilogue snapshots.  For kernels that produce
floating-point results, this is often too strict.

Mneme provides a lightweight C++ API — `mneme::annotate()` — that lets
you attach a data type, error threshold, aggregation norm, and an
optional tag to any device pointer.  The metadata is captured inside
the recorded snapshot and used automatically during every subsequent
replay.

```cpp
#include "mneme/MnemeAnnotation.hpp"

mneme::annotate(d_output, mneme::Metadata{
    .builtin        = mneme::BuiltinDType::F64,
    .threshold      = 1e-6,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
    .tag            = std::string("output_vector"),
});
```

Annotations must be applied **before** the kernel launch they should
affect.  You can update the annotation on the same pointer between
launches to record different tolerance policies for different dynamic
instances of the same kernel.

For the full API reference, supported data types, threshold semantics,
and a complete example, see **[Usage → Verification](verification.md)**.
