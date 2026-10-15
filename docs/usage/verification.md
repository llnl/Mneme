# Verification

Mneme verifies replay correctness by comparing the device memory state
after kernel execution (the replay epilogue) against the recorded
epilogue snapshot.  By default this comparison is an exact byte-wise
check.

**Annotations** let you override this default on a per-buffer basis by
attaching typed comparison metadata — data type, error threshold, norm,
and an optional human-readable tag — to any device pointer *before* a
kernel is recorded.  The metadata is stored in the record database
and used automatically during every subsequent replay, and
`--compare-policy` can override it without re-recording.

---

## Why annotations?

Many GPU kernels produce floating-point results where bit-exact
reproduction is neither expected nor required.  Without annotations,
any recompilation or launch-parameter change that introduces
floating-point rounding differences will cause verification to fail,
even when the result is numerically acceptable.

Annotations solve this by letting you declare, at record time, what
"correct enough" means for each buffer.

---

## The `mneme::annotate` API

The annotation API is declared in `mneme/MnemeAnnotation.hpp`, which is
installed automatically when you build with Mneme.

### Include and link

```cpp
#include "mneme/MnemeAnnotation.hpp"
```

If your CMake target was created with `add_mneme(my_target)`, the
annotation runtime library (`mnemert`) is linked automatically and no
extra `target_link_libraries` call is required.

### Basic usage

```cpp
double *d_output = nullptr;
cudaMalloc(&d_output, N * sizeof(double));

// Annotate *before* the kernel launch that will be recorded.
mneme::annotate(d_output, mneme::Metadata{
    .builtin        = mneme::BuiltinDType::F64,
    .threshold      = 1e-6,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
    .tag            = std::string("output_vector"),
});

my_kernel<<<grid, block>>>(d_output, ...);
```

The typed convenience overload deduces `builtin` from the pointer type:

```cpp
mneme::annotate<double>(d_output, mneme::Metadata{
    .threshold      = 1e-6,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
});
```

### Updating annotations between launches

You can call `annotate` on the same pointer more than once.  Each kernel
launch captures the annotation state that was active when recording
began, so different dynamic instances of the same kernel can carry
different verification policies:

```cpp
mneme::annotate(d_out, mneme::Metadata{
    .builtin        = mneme::BuiltinDType::F64,
    .threshold      = 0.01,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
    .tag            = std::string("out_loose"),
});
vecAdd<<<4, 128>>>(d_in, d_out, N);

// Tighten the tolerance for the next invocation.
mneme::annotate(d_out, mneme::Metadata{
    .builtin        = mneme::BuiltinDType::F64,
    .threshold      = 0.005,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
    .tag            = std::string("out_tight"),
});
vecAdd<<<8, 64>>>(d_in, d_out, N);
```

---

## Metadata fields

### `BuiltinDType`

Tells Mneme how to interpret the raw bytes in the buffer.

| Value  | C++ type     |
| ------ | ------------ |
| `U8`   | `uint8_t`    |
| `I8`   | `int8_t`     |
| `U16`  | `uint16_t`   |
| `I16`  | `int16_t`    |
| `U32`  | `uint32_t`   |
| `I32`  | `int32_t`    |
| `U64`  | `uint64_t`   |
| `I64`  | `int64_t`    |
| `F16`  | `__half`     |
| `F32`  | `float`      |
| `F64`  | `double`     |

When you use the typed overload `mneme::annotate<T>(ptr, md)`, the
`builtin` field is set automatically from the pointer type.

### `threshold`

A non-negative `double` that defines the maximum acceptable error.
Its exact meaning depends on `threshold_kind` and `norm` (see below).

### `ThresholdKind`

Controls how the per-element error between the replayed value *r* and
the recorded value *e* is computed.

| Value      | Formula                                      |
| ---------- | -------------------------------------------- |
| `Absolute` | $\lvert r - e \rvert$                        |
| `Relative` | $\frac{\lvert r - e \rvert}{\lvert e \rvert}$ |

### `Norm`

Determines how per-element errors are aggregated across the buffer.

| Value  | Semantics |
| ------ | --------- |
| `None` | **Per-element mode.** Each element is individually compared against `threshold`. Verification fails if *any* element exceeds the threshold. |
| `L1`   | $\sum_i \text{err}(i) \leq \text{threshold}$ |
| `L2`   | $\sqrt{\sum_i \text{err}(i)^2} \leq \text{threshold}$ |
| `Linf` | $\max_i \text{err}(i) \leq \text{threshold}$ |

When `norm` is `L1`, `L2`, or `Linf`, the aggregated value is compared
against `threshold` to determine pass/fail.

### `tag` (optional)

A free-form string recorded with the metadata.  A compare policy can
select buffers by tag; Mneme does not otherwise interpret it.

---

## How verification works at replay time

1. Mneme restores the recorded **prologue** device memory state.
2. The kernel is recompiled and executed.
3. For every recorded memory blob, Mneme dispatches a GPU comparison
   kernel that applies the blob's `BuiltinDType`, `ThresholdKind`, and
   `Norm`, taken from the record database and any `--compare-policy`.
4. If *all* blobs pass, the replay result reports `"verified": true`.

Buffers that were **not** annotated fall back to the default byte-wise
comparison (equivalent to `threshold = 0`, `Norm::None`).

### Recorded metadata

Mneme records the metadata each annotated buffer had when the kernel
finished, as the `Compare` list of the kernel instance in
`<static-hash>.json`:

```json
"Compare": [
  {"DevAddr": "0x00004aaa40002000", "Tag": "out_loose", "DType": "f64",
   "Threshold": 0.01, "ThresholdKind": "relative", "Norm": "linf"}
]
```

`DType` is one of `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`,
`f16`, `f32`, `f64`; `ThresholdKind` is `absolute` or `relative`; `Norm`
is `none`, `l1`, `l2`, or `linf`.  `Tag` is present only when set.
Buffers without an entry are compared byte-wise.

Recordings made before the `Compare` list existed have none; their
metadata is read from the snapshots instead and verifies as before.

---

## Overriding tolerances at replay time

`mneme replay` and `mneme tune` accept `--compare-policy FILE`, a JSON
file that overrides the recorded metadata without re-recording:

```json
{"Default": {"Threshold": 1e-6},
 "Tags": {"out_loose": {"Threshold": 1e-3}},
 "Addresses": {"0x00004aaa40002000": {"Norm": "l2"}}}
```

Each section holds `Compare` fields (`DType`, `Threshold`,
`ThresholdKind`, `Norm`).  `Default` applies to every buffer, `Tags` to
the buffers recorded with that tag, and `Addresses` to the buffer at
that device address.  Fields are merged one at a time with precedence
`Addresses` > `Tags` > `Default` > recorded metadata > byte-wise
comparison.

An unknown key, an unknown enum name, a negative threshold, or a tag or
address that matches no buffer of the replayed instance is an error.
The Python API takes the same policy as
`ComparePolicy.from_file(path)`, passed as `compare_policy` to
`AsyncReplayExecutor` or `BaseExecutor`.  To compare snapshots
directly, pass `KernelInstance.compare_rules(policy)` to
`MemStateRef.matches`; a buffer without a rule is an error.

---

## Complete example

The following program annotates two device buffers with different
tolerances and records two kernel launches, each capturing a distinct
annotation state.

```cpp
#include <cstdlib>
#include <iostream>
#include "mneme/MnemeAnnotation.hpp"

// Use DeviceTraits or plain CUDA/HIP runtime calls.
#ifdef __ENABLE_CUDA__
#include <cuda_runtime.h>
#define gpuMalloc   cudaMalloc
#define gpuMemset   cudaMemset
#define gpuFree     cudaFree
#define gpuSync     cudaDeviceSynchronize
#elif defined(__ENABLE_HIP__)
#include <hip/hip_runtime.h>
#define gpuMalloc   hipMalloc
#define gpuMemset   hipMemset
#define gpuFree     hipFree
#define gpuSync     hipDeviceSynchronize
#endif

template <typename T>
__global__ void vecAdd(T *in, T *out, size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n)
    out[tid] += in[tid] + tid;
}

int main(int argc, const char *argv[]) {
  size_t N = std::atoi(argv[1]);
  double *in = nullptr, *out = nullptr;

  gpuMalloc((void **)&in,  N * sizeof(double));
  gpuMalloc((void **)&out, N * sizeof(double));
  gpuMemset(in,  0, N * sizeof(double));
  gpuMemset(out, 0, N * sizeof(double));

  // Annotate both pointers before the first recorded launch.
  mneme::annotate(in, mneme::Metadata{
      .builtin        = mneme::BuiltinDType::F64,
      .threshold      = 0.125,
      .threshold_kind = mneme::ThresholdKind::Absolute,
      .norm           = mneme::Norm::L2,
      .tag            = std::string("in_vec"),
  });

  mneme::annotate(out, mneme::Metadata{
      .builtin        = mneme::BuiltinDType::F64,
      .threshold      = 0.01,
      .threshold_kind = mneme::ThresholdKind::Relative,
      .norm           = mneme::Norm::Linf,
      .tag            = std::string("out_loose"),
  });

  vecAdd<<<4, 128>>>(in, out, N);
  gpuSync();

  // Update the annotation for the next launch.
  mneme::annotate(out, mneme::Metadata{
      .builtin        = mneme::BuiltinDType::F64,
      .threshold      = 0.005,
      .threshold_kind = mneme::ThresholdKind::Relative,
      .norm           = mneme::Norm::Linf,
      .tag            = std::string("out_tight"),
  });

  vecAdd<<<8, 64>>>(in, out, N);
  gpuSync();

  gpuFree(in);
  gpuFree(out);
  return 0;
}
```

Record, replay, and verify as usual:

```bash
mneme record  -rdb db/ -- ./vecAddAnnotations 1024
mneme replay  -rdb db/<static-hash>.json -rid <dynamic-hash> "default<O3>"
```

The replay JSON result will contain `"verified": true` when the
replayed epilogue satisfies every annotated buffer's tolerance.

---

## Summary

| Concept | Description |
| ------- | ----------- |
| `mneme::annotate(ptr, md)` | Attach verification metadata to a device pointer |
| `BuiltinDType` | Scalar type used to interpret buffer contents |
| `ThresholdKind` | Absolute or relative error formula |
| `Norm` | Per-element (`None`) or aggregate (`L1`, `L2`, `Linf`) comparison |
| `tag` | Optional label that compare policies can select |
| `--compare-policy` | Override the recorded metadata at replay time |

Annotations are recorded once and applied automatically on every replay.
Buffers without annotations continue to use the default byte-wise
comparison.
