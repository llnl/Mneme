# Record

The `mneme record` command executes an application while transparently
capturing GPU kernel executions and the associated device memory state.
For full CLI details, see [CLI → mneme record](cli.md#mneme-record).

Recording does not require modifying application source code.  However,
if you want **tolerance-aware verification** during replay, you can
optionally annotate device pointers with comparison metadata before the
kernel launches that will be recorded.

## Annotating buffers for verification

By default, replay verification uses exact byte-wise comparison between
the replayed and recorded epilogue snapshots.  For kernels that produce
floating-point results, this is often too strict.

Mneme provides a lightweight C++ API — `mneme::annotate()` — that lets
you attach a data type, error threshold, aggregation norm, and an
optional tag to any device pointer.  The metadata is stored in the
record database and used automatically during every subsequent replay.

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
