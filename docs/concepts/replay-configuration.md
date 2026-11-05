# Replay Configuration

Mneme replay is controlled by an explicit **replay configuration**: a compact
set of parameters that determines **how a recorded kernel is recompiled and
executed** during `mneme replay` and during tuning.

Replay configurations are first-class objects in Mneme. They can be:
- created from recorded defaults,
- overridden from the CLI,
- generated programmatically (Python API), and
- hashed to obtain a stable identifier for experiments.

---

## Mental model

A replay configuration answers two questions:

1. **How should the kernel be launched?**  
   (grid/block dimensions, shared memory)

2. **How should the kernel be recompiled?**  
   (IR pipeline, specialization options, codegen settings)

Mneme uses the recorded execution as a baseline. When options are omitted,
the **recorded values are reused**.

---

## Configuration object

In the Python API, replay configurations are represented by:

- `mneme.mneme_types.ExperimentConfiguration`

This object is intentionally:
- **serializable** (`to_dict()` / `from_dict()`),
- **validatable** (`is_valid()`),
- **groundable** (`ground()`), and
- **hashable** (`hash()`), so configurations can be tracked persistently
  across runs.

---

## Launch configuration knobs

These fields control the concrete kernel launch:

### Grid and block dimensions

- `grid: dim3`  
  Grid dimensions (x, y, z).  
- `block: dim3`  
  Block dimensions (x, y, z).

By default, Mneme uses the recorded dimensions for the selected instance
(dynamic hash). Users may override them to explore alternative launch
geometries.

> In many kernels, `block.x` is the primary tuning dimension; Mneme supports
> overriding any of x/y/z.

### Dynamic shared memory

- `shared_mem: int`  
  Dynamic shared memory bytes for the launch.

If not set explicitly, Mneme defaults to the recorded shared memory size.

---

## Specialization knobs

Mneme can apply specialization when replaying a kernel. These options
control *what is treated as constant* during replay compilation.

### Specialize arguments

- `specialize: bool`  
  Enables specialization based on the recorded execution context.

Conceptually, this allows Mneme to propagate recorded values into the IR,
enabling constant folding and simplification. The exact specialization
surface is backend-dependent and evolves over time. Briefly, any custom specialization 
supported in Proteus is also exposed in this parameter.

### Specialize launch dimensions

- `specialize_dims: bool`  
  Specializes `ThreadID.*`, `BlockDim.*`, `GridDim.*` as constants.

This is useful when the kernel’s control flow depends on launch geometry
(e.g., bounds checks, tiling, unrolling decisions).

---

## Launch bounds controls

Launch bounds affect register allocation and occupancy. Mneme can
optionally inject launch bounds into the generated kernel.

- `set_launch_bounds: bool`  
  Enable launch bounds injection.
- `max_threads: int`  
  Launch bound `max_threads` parameter.
- `min_blocks_per_sm: int`  
  Launch bound `min_blocks_per_sm` parameter.

### Validity constraints

Mneme enforces a basic constraint:

> If launch bounds are enabled, `max_threads` must be ≥ total threads per block.

In other words:

`max_threads >= block.x * block.y * block.z`

The Python API exposes this via:

- `ExperimentConfiguration.is_valid()`

### Grounding unused values

When launch bounds are disabled, Mneme treats `max_threads` and
`min_blocks_per_sm` as unused knobs. To make hashing stable and reduce
configuration ambiguity, Mneme supports “grounding”:

- `ExperimentConfiguration.ground()`

This sets unused launch-bounds fields to zero when `set_launch_bounds=False`.

---

## Compilation and code generation knobs

These fields control how Mneme recompiles LLVM IR into a device object.

### Pass pipeline

- `passes: str`  
  The IR optimization pipeline specification, e.g. `default<O3>`.

This is the primary “compiler pipeline” knob for replay and tuning. It
controls the sequence of transformations applied to the recorded LLVM IR
before code generation.

### Pass plugins

Replay can also run passes from your own LLVM pass plugins. These options
belong to the replay session rather than to `ExperimentConfiguration`, so they
stay fixed while tuning varies `passes`:

- `--pass-plugin PATH` (repeatable) loads a plugin shared library, the same
  kind `opt --load-pass-plugin` accepts, into the JIT so its passes can be
  named in pipelines.
- `--prepend-passes PIPELINE` runs before `passes`.
- `--append-passes PIPELINE` runs after `passes`.

Every build of the kernel, including the verification build, runs
`<prepend>,<passes>,<append>`. The tracked build puts Mneme's own
`purge-autoinit` pass in front of that.

```bash
mneme replay -rdb kernel.json \
  --pass-plugin ./libMyPass.so --prepend-passes my-pass "default<O3>"

mneme tune -rdb kernel.json -rid <id> --space-preset compiler --sampler random \
  --trials 50 --pass-plugin ./libMyPass.so --prepend-passes my-pass
```

From Python, pass `PassPlugins(libraries=("./libMyPass.so",), prepend="my-pass")`
from `mneme.pass_plugins` as `pass_plugins=` to `AsyncReplayExecutor` or a
`BaseExecutor` subclass. Each tuning worker loads the plugins itself.

`mneme tune` exports only the tuned `passes` to the Proteus JSON. To get the
same code in the full application, register the plugin with Proteus there,
e.g. `proteus::registerJITPassPlugin(path, "my-pass", JITPassPluginPosition::Prepend)`.

### Codegen optimization level

- `codegen_opt: int`  
  Backend optimization level (typically 0–3).

This influences backend passes during machine code generation (register
allocation heuristics, scheduling, etc.), distinct from IR-level pipelines.

### Codegen method

- `codegen_method: str`  
  Code generation strategy. Mneme currently supports:
  - `"serial"`: compile in a single-threaded codegen path.

> Mneme exposes this knob because Proteus can support different backends
> over time; Mneme currently restricts it to what is supported and tested.

### Mandatory replay hygiene

Mneme currently enforces two IR hygiene options:

- `prune: bool` (currently mandatory `True`)  
  Enables IR pruning / dead-code elimination.
- `internalize: bool` (currently mandatory `True`)  
  Internalizes symbols to reduce visibility and improve replay isolation.

These are exposed as fields because Mneme intends to explore and document
their performance/correctness tradeoffs in the future, but today they are
treated as always-on.

---

## Configuration identity and hashing

Replay configurations are designed to be tracked and compared across runs.

- `ExperimentConfiguration.to_dict()` produces a JSON-serializable view.
- `ExperimentConfiguration.hash()` computes a stable SHA-256 digest over the
  normalized representation.

This enables:
- stable experiment IDs,
- caching of replay results,
- reproducing “best configs” from tuning runs.

---

## How this maps to the CLI

The `mneme replay` CLI flags correspond directly to configuration fields:

- `--grid-dim-*` → `grid.{x,y,z}`
- `--block-dim-*` → `block.{x,y,z}`
- `--shared-mem` → `shared_mem`
- `--specialize / --no-specialize` → `specialize`
- `--set-launch-bounds` + `--max-threads` + `--min-threads-per-block`
  → launch bounds fields
- `--specialize-dims` → `specialize_dims`
- `passes` positional argument → `passes`
- `--codegen-opt` → `codegen_opt`
- `--codegen-method` → `codegen_method`

`--pass-plugin`, `--prepend-passes` and `--append-passes` configure the
executor rather than `ExperimentConfiguration`; see [Pass plugins](#pass-plugins).

For CLI details, see **[Usage → CLI](../usage/cli.md)**.

---

## Recommended usage patterns

- **Start from a recorded instance** (dynamic hash) and replay with defaults.  
  This gives a correctness baseline.

- **Change one family of knobs at a time**:
  - launch geometry (block/grid),
  - specialization toggles,
  - pipeline (`passes`) and codegen opt.

- **Use hashing for reproducibility**:
  persist the config dict and hash alongside performance results.

---

## Summary

Replay configuration is Mneme’s “control surface” for:
- reproducing recorded kernel executions,
- enabling controlled experimentation,
- defining tuning search spaces.

A single recorded kernel instance can be replayed under many
configurations, but each configuration remains explicit, serializable,
and reproducible.

