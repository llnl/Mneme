# Verification

After a replay, Mneme compares the device memory the kernel left behind
with the recorded epilogue snapshot.
By default the comparison is byte-exact,
which fails on harmless floating-point rounding differences
after a recompilation or a launch-parameter change.

**Annotations** say what "correct enough" means for a buffer:
its data type, an error threshold, and how errors are aggregated.
You attach them at record time,
and every later replay of that recording uses them.

---

## Annotating a buffer

Include `mneme/MnemeAnnotation.hpp`.
Targets created with `add_mneme(my_target)` link the annotation runtime
(`mnemert`) automatically.

```cpp
#include "mneme/MnemeAnnotation.hpp"

double *d_out = nullptr;
cudaMalloc(&d_out, N * sizeof(double));

mneme::annotate(d_out, mneme::Metadata{
    .threshold      = 1e-6,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
    .tag            = std::string("output_vector"),
});

my_kernel<<<grid, block>>>(d_out, ...);
```

Annotate before the kernel launch the annotation should affect.
For a typed pointer such as `double *`, `builtin` is set from the pointer type;
for a `void *`, set `builtin` yourself.

To use different tolerances for different launches of the same kernel,
annotate again between them:

```cpp
mneme::annotate(d_out, mneme::Metadata{.threshold = 0.01,
                                       .threshold_kind = mneme::ThresholdKind::Relative,
                                       .norm = mneme::Norm::Linf});
vecAdd<<<4, 128>>>(d_in, d_out, N);

mneme::annotate(d_out, mneme::Metadata{.threshold = 0.005,
                                       .threshold_kind = mneme::ThresholdKind::Relative,
                                       .norm = mneme::Norm::Linf});
vecAdd<<<8, 64>>>(d_in, d_out, N);
```

Record and replay as usual.
The replay result reports `"verified": true`
when every buffer passes its comparison.
Buffers without annotations are compared byte for byte.

---

## Metadata fields

### `builtin`

The element type of the buffer.

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

### `threshold` and `threshold_kind`

`threshold` is the largest acceptable error, a non-negative `double`.
`threshold_kind` sets how the error between the replayed value *r*
and the recorded value *e* is computed:

| Value      | Error                                         |
| ---------- | --------------------------------------------- |
| `Absolute` | $\lvert r - e \rvert$                         |
| `Relative` | $\frac{\lvert r - e \rvert}{\lvert e \rvert}$ |

### `norm`

How the per-element errors are combined before comparing with `threshold`:

| Value  | Passes when |
| ------ | ----------- |
| `None` | every element's error is at most `threshold` |
| `L1`   | $\sum_i \text{err}(i) \leq \text{threshold}$ |
| `L2`   | $\sqrt{\sum_i \text{err}(i)^2} \leq \text{threshold}$ |
| `Linf` | $\max_i \text{err}(i) \leq \text{threshold}$ |

For `F16`, `F32` and `F64`, two NaNs match,
and two infinities match when their signs agree;
any other NaN or infinity fails the comparison.

### `tag`

An optional label for identifying the buffer. Mneme does not interpret it.

---

## Annotating part of an allocation

To annotate an interior pointer into a larger allocation,
or to give the fields of one allocation different tolerances,
pass the size of the region in bytes:

```cpp
double *d_field = base + offset;

mneme::annotate(d_field, count * sizeof(double), mneme::Metadata{
    .builtin        = mneme::BuiltinDType::F64,
    .threshold      = 1e-12,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::L2,
});
```

- Where annotations overlap, the later one wins,
  so annotate the whole allocation first and its regions after it.
- Each region is checked on its own, with its own norm.
- The region must lie inside the allocation and cover whole elements,
  and it cannot start or end inside an element of an earlier annotation.
  Recording aborts otherwise.
