# Mneme Execution Phases 


This page describes the three execution phases of Mneme and how they
work together to enable kernel record, replay, and tuning.

---

## Phase 1: Instrumentation (build time)

### Purpose

The purpose of the instrumentation phase is to produce a recordable executable:
a binary that self-describes its GPU kernels and can be observed by Mneme at runtime.

Recording and replay require a stable handle to the kernel code itself. Rather than
trying to reconstruct this information from source code—which is difficult due to
language semantics, translation units, and type-system complexity—Mneme operates
at the LLVM IR level, where:

- the type system is explicit and normalized,
- translation units are well defined, and
- kernel boundaries and dependencies are clear.

Instrumentation provides a single, unified analysis point—the executable—where
kernel code, metadata, and runtime hooks are consistently available.

### What happens

During compilation, Mneme (via Proteus) applies an LLVM instrumentation pass that performs two complementary actions:

#### 1. **Kernel code capture**

For each GPU kernel, the pass:

- identifies the kernel entry point,
- collects the kernel’s LLVM IR along with its entire transitive dependency graph, and
- embeds this information into a dedicated ELF section of the device object.

Each embedded data segment corresponds to a single kernel and contains everything
required to later recompile and specialize that kernel in isolation.

#### 2. **Kernel launch interception**

During host-side compilation, all GPU kernel launches are redirected to pass
through the Proteus runtime system. 

Instead of invoking the vendor runtime API directly (e.g., CUDA or HIP launch calls),
the application *enters the Proteus runtime* which provides a controlled interception point used by Mneme during recording and replay.
This redirection is transparent to the application and preserves original behavior
when Mneme is not actively recording.

#### 3. **Fatbinary and global variable registration**

During host-side compilation, the instrumentation pass performs registration of the fatbinary global variable and kernel stub handles to the
Proteus runtime system.

!!! note
    The instrumentation pass is not developed by Mneme. Instead, Mneme relies entirely on the Proteus instrumentation pass, maintained and developed for the
    [annotation-based JIT interface](https://olympus-hpc.github.io/proteus/user/interface/#code-annotations).

### What is produced

The instrumentation phase produces a recordable executable with the following properties:
- it contains embedded GPU kernel LLVM IR stored in ELF sections,
- it launches GPU kernels through the Proteus runtime rather than directly through
the vendor API, and
- it behaves identically to the original application when executed normally.

### What has **not happened** yet

At this stage:

- no recording occurs, and
- no device memory is captured.


### Summary
Instrumentation simply enables the subsequent runtime phases by making kernel code
and execution observable. This executable is the input to the record phase, where actual kernel executions
and device memory state are captured.

---

## Phase 2: Recording (runtime)

### Purpose

The purpose of the record phase is to capture everything required to
re-execute a GPU kernel as an independent executable, without the
original application. At runtime, Mneme records a concrete kernel invocation, consisting of:

- the kernel code (LLVM IR),
- the complete device memory state the kernel can observe, and
- the kernel launch configuration.

Together, these form a self-contained description of a single kernel
execution.

### Core idea: capture code and memory

To faithfully replay a kernel, Mneme must capture both:

1. the code being executed, and
2. the memory state the kernel operates on.

The recorded LLVM IR provides a stable representation of the kernel
code, while the device memory snapshots provide realistic and
reproducible inputs.

> Code + Memory = concrete kernel invocation

### Device memory tracking

During recording, Mneme intercepts all device memory allocations and
deallocations performed by the application.

To do this, Mneme provides its own device memory allocator, which:

1. tracks the size and device address of every allocation,
2. maps each allocation at a device address it chooses, packing small
   allocations into shared mappings, and
3. records allocation lifetimes across kernel launches.

Mneme places allocations far from addresses the vendor runtime and host
use, so replay can map the same addresses again.

This mechanism enables Mneme to precisely identify which regions of
device memory may be accessed by a kernel.

### Capturing kernel memory state

Before each kernel launch, Mneme captures the prologue memory state:
a snapshot of all relevant device memory regions.

After kernel execution completes, Mneme captures the epilogue memory
state.

These snapshots include:

- device heap allocations,
- device global variables, and
- kernel argument memory.

If the application has annotated any device pointer via
`mneme::annotate()`, the associated metadata (data type, error
threshold, norm, and optional tag) is captured alongside the memory
content.  This metadata governs how Mneme compares the replayed output
against the recorded epilogue during the replay phase.  See
[Usage → Verification](../usage/verification.md) for details.

The closure of kernel arguments, global variables and heap memory represents a
superset of the memory the kernel may access during execution, ensuring
that replay has all required inputs available.

### Kernel identity and code association

The instrumentation phase provides Mneme with:
access to the kernel’s LLVM IR (embedded in ELF sections), and
metadata describing kernel identity and global variables.

During recording, Mneme queries Proteus for LLVM IR and associates each observed kernel launch with:

- its static kernel identity (LLVM IR code),
- a dynamic execution identity (launch configuration and runtime parameters), and
- the corresponding memory snapshots.

This association is stored in the recording database, which serves as
the entry point for replay and tuning workflows.

### What is produced

The record phase produces a set of recording artifacts, including:

- a recording database (JSON) describing a single static kernel and all the dynamic instances being recorded
- device memory snapshots (prologue and epilogue) for every static kernel and dynamic instance being recorded
- references to the recorded LLVM IR modules.

These artifacts are immutable and can be reused indefinitely for replay,
verification, and tuning.

### Summary

The record phase transforms a live kernel execution into a portable,
replayable description by capturing:

- the kernel’s LLVM IR,
- the complete device memory state it observes, and
- its execution configuration.

Once recorded, kernels can be replayed and tuned without rerunning the
original application or re-recording inputs.

---

## Phase 3: Replay (runtime)

### Purpose

The purpose of the replay phase is to re-execute a recorded GPU kernel
in isolation, under controlled and configurable conditions, without
running the original application.

Replay allows users to:

- faithfully reproduce recorded kernel executions,
- validate correctness against recorded memory state,
- modify compilation and launch parameters, and
- systematically explore performance optimizations.

Replay operates entirely on the artifacts produced during recording.

### Core idea: restore, recompile, execute

Replay consists of three fundamental steps:

- Restore device memory
- Recompile the kernel using Proteus
- Execute and validate

Together, these steps reconstruct a concrete kernel execution that is
equivalent to the recorded one, while allowing controlled variation.

All of these steps are exposed as C-bindings to Python, hiding system-level
intricacies from the user and providing a fast prototyping environment.

### Restoring device memory state

At the beginning of replay, Mneme restores the prologue device memory
state recorded during the record phase.

This involves:

1. mapping device memory at the recorded allocation addresses, and
2. populating device memory with the recorded contents of:
    - heap allocations,
    - device global variables, and
    - kernel argument buffers.

Memory restoration ensures that the kernel observes the same initial
state as during recording, independent of the replay environment.

### Kernel recompilation from LLVM IR

Once memory is restored, Mneme recompiles the kernel from recorded LLVM
IR, not from the original application binary.

During recompilation, users may optionally apply:

- different optimization pipelines,
- LLVM transformation passes,
- code generation optimization levels,
- argument specialization, and
- launch-bound modifications.

LLVM IR serves as the stable transformation boundary, enabling replay-time
code modification without rebuilding the original application.

### Replay configuration

Kernel replay is driven by an explicit replay configuration, which
controls both compilation and execution parameters, including:

- grid and block dimensions,
- shared memory size,
- specialization flags,
- compilation pipelines and optimization levels, and
- execution iteration counts.

Replay configurations are first-class objects and can be constructed
manually or generated programmatically during tuning.

### Execution and validation

After recompilation, Mneme executes the kernel using the restored memory
state and selected replay configuration.

Every kernel execution is profiled through the vendor profiling tooling for example in AMD, 
Mneme uses the `rocprofiler-sdk` to provide highly accurate performance metrics that can be used as feedback
for future decisions.

Following execution, Mneme:

- captures the resulting device memory state, and
- compares it against the recorded epilogue snapshot.

When a recorded memory blob carries annotation metadata (set via
`mneme::annotate()` before the corresponding kernel launch), Mneme
uses the stored data type, threshold, and norm to perform a
tolerance-aware comparison instead of a byte-exact check.  See
[Usage → Verification](../usage/verification.md) for the full
semantics.

This comparison enables:

- correctness verification,
- detection of miscompilations or invalid transformations, and
- safe exploration of optimization spaces as replay executions that fail validation can be automatically discarded
during tuning workflows.

## Tuning and experimentation

Replay is the foundation of Mneme’s tuning model.

Because replay is:

- isolated from the original application,
- driven by explicit configurations, and
- validated against recorded memory state,

it can be executed repeatedly and efficiently across large configuration
spaces without requiring the entire application to be recompiled and re-executed.

Replay supports integration with automated tuning frameworks, such as
Optuna, and enables reproducible performance experimentation.

### What replay does not do

During replay:

- host-side application logic is not executed,
- CPU memory state is not restored,
- external side effects are not reproduced.
- Replay focuses exclusively on GPU kernel behavior.

### Summary

The replay phase reconstructs a recorded kernel execution by:

- restoring recorded device memory state,
- recompiling the kernel from LLVM IR, and
- executing it under controlled conditions.

This enables reproducible validation, optimization, and tuning of GPU
kernels without rerunning the original application.
