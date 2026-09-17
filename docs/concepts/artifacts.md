# Artifacts

Mneme’s record–replay workflow is built around a small set of immutable
artifacts produced during recording and consumed during replay and
tuning.

These artifacts form the contract between recording and replay.
Once created, they fully describe a kernel execution and can be reused
indefinitely without rerunning the original application.

## What are artifacts?

In Mneme, an artifact is any persistent data generated during recording
that is required to:

- reproduce a kernel execution,
- validate correctness during replay, and
- enable controlled experimentation and tuning.

Artifacts are created only during the record phase and are treated
as read-only inputs during replay.

## The artifact set

Recording a kernel execution produces three conceptual classes of
artifacts:

* Recording database,
* Device memory state, 
* Recorded LLVM IR.

Each artifact captures a different aspect of the execution.

## Recording database

The recording database, one for every recorded kernel uniquely identified by the static hash,  is the entry point to all recorded data.

Conceptually, it provides:

- a mapping from static kernel identity to dynamic execution instances,
- references to the recorded LLVM IR required for replay, and
- pointers to the associated device memory snapshots.

The database does not contain executable code or memory contents
itself. Instead, it describes how all recorded components relate to each
other.

From a conceptual standpoint, the recording database answers the
question:

> **What kernel was executed, how was it executed, and where are the artifacts needed to reproduce it?**

## Device memory state

The device memory state artifacts capture the concrete inputs and
outputs of a kernel execution.

For each recorded kernel instance, Mneme stores:

- a prologue state (device memory before execution), and
- an epilogue state (device memory after execution).

These snapshots include the closure of all device-visible memory that a
kernel may access, including:

- device heap allocations,
- device global variables, and
- kernel argument buffers.

With `mneme record --capture-mode reachable`, the heap allocations are
narrowed to those a pointer-typed kernel argument points into; globals and
argument buffers are captured as before.

Device heap allocations are identified by the:

- Device allocation address,
- Allocation size.

While global variables are identified by:

- Global variable device address,
- Global variable size, and
- Global variable name.

The memory state provides realistic and representative inputs for
replay and serves as the reference for correctness validation.

Each memory blob in the snapshot may also carry **annotation metadata**
(data type, error threshold, aggregation norm, and an optional tag)
attached by the application via `mneme::annotate()`.  This metadata
defines how Mneme judges equivalence between the replayed output and
the recorded epilogue during verification, making it part of the
artifact contract rather than optional debug information.
See [Usage → Verification](../usage/verification.md) for the full
API reference.

## Recorded LLVM IR

The recorded LLVM IR artifacts capture the kernel code itself.

Rather than replaying a binary kernel image, Mneme records the kernel in
LLVM IR form, along with its transitive dependencies.

LLVM IR serves as the transformation boundary for Mneme:

- kernels can be recompiled at replay time,
- compiler passes and optimizations can be applied,
- specialization decisions can be explored, and
- different code generation strategies can be evaluated.

This design decouples replay and tuning from the original application
binary.

## Artifacts as a contract

Together, these artifacts form a complete and portable description of
a kernel execution:

- Code → recorded LLVM IR
- Inputs → prologue device memory state
- Outputs → epilogue device memory state
- Structure → recording database

Once artifacts are created:

the original application is no longer needed,
kernels can be replayed arbitrarily many times, and
tuning workflows can operate safely and reproducibly.

Artifacts are intentionally immutable, ensuring that replay results
are deterministic with respect to the recorded state.

## Lifecycle of artifacts

From a conceptual perspective, artifacts follow a simple lifecycle:

1. Created during the record phase
2. Consumed during replay and tuning
3. Validated against recorded memory state
4. Managed (copied, moved, cleaned) by the user

**Artifacts are never modified in-place by Mneme.**

## Why this design matters

The artifact-based design enables Mneme to:

- scale to large applications with many kernel launches,
- decouple recording from replay and tuning,
- provide strong correctness guarantees, and
- support automated and reproducible performance experimentation.

Artifacts are the foundation that allows Mneme to operate as a kernel
record–replay system rather than a traditional profiler or tracer.

## Where to go next

For file formats and concrete examples, see [Usage → Artifacts](../usage/artifacts.md)

For how artifacts are produced, see [Execution Phases](mneme-phases.md)

For how artifacts are consumed, see [Usage → Artifacts](../usage/artifacts.md) 
