# Limitations

Mneme provides a practical record–replay and autotuning workflow for GPU kernels, but there are currently a few known limitations. 
These are not fundamental design blockers, but they may affect certain applications today.

---

## 1. No Support for Managed Memory

Mneme does **not** currently support CUDA Unified / Managed Memory (`cudaMallocManaged`).

**Implications:**
- Kernels that rely on managed memory allocations may fail during replay.
- Memory state reconstruction assumes explicit device memory allocations (`cudaMalloc`, `hipMalloc`).

**Workaround:**
- Replace managed memory with explicit host–device memory transfers.
- Use pinned host memory and explicit `cudaMemcpy` where possible.

---

## 2. Restrictions on Global Variables

Global variables are supported **as long as their addresses are not captured by another global variable**.

**Unsupported pattern:**
```cpp
__device__ int g_value;
__device__ int* g_ptr = &g_value;  // ❌ not supported
```

Supported pattern:
```cpp
__device__ int g_value;             // ✅ supported
```

Why this matters:

- Mneme records and reconstructs global memory symbols independently.
- Address aliasing between globals complicates relocation and replay correctness.

Workaround:

- Avoid global pointer aliasing.
- Initialize pointer relationships inside a kernel or host-side setup code instead.

## 3. CUDA RDC (Relocatable Device Code) Is Untested

CUDA Relocatable Device Code (RDC) is not tested and may not work reliably.

Implications:

- Multi-translation-unit device code, device-side linking, and dynamic device symbol resolution may fail.
- Kernel replay may break when kernels depend on symbols defined in separate device objects.

Current status:
- RDC-related issues have not yet been systematically evaluated.

If you need this:

- Please open a GitHub issue with a minimal reproducer.
- RDC support is planned but not yet prioritized.

## 4. GPU-Aware Cray MPICH on AMD APUs Needs `MPICH_SMP_SINGLE_COPY_MODE=NONE`

During recording, Mneme serves device allocations from its own virtual address
space, mapped with the HIP virtual memory API (`hipMemCreate`/`hipMemMap`).
Cray MPICH's GPU transport layer does not recognize these mappings as GPU memory
and treats them as host memory. On AMD APUs (e.g. MI300A), on-node messages
between ranks then go through MPICH's default XPMEM single-copy path, which
cannot attach these pages.

**Symptoms:**
- A recorded multi-rank run with `MPICH_GPU_SUPPORT_ENABLED=1` crashes with
  `SIGBUS` in an MPI call when GPU buffers are exchanged between ranks on the
  same node.
- The same application runs correctly without `mneme record`.

**Workaround:**
Set the following variable for the recorded run:
```bash
export MPICH_SMP_SINGLE_COPY_MODE=NONE
mneme record -rdb record-dir --record-ranks all -- flux run -n 2 ./app
```
On-node messages are then staged through MPICH's shared-memory buffers. They
are correct but slower, and the setting also applies to buffers that Mneme does
not manage. Replay does not use MPI and is unaffected.

## Reporting Issues or Requesting Support

If any of these limitations block your use case, please:


[Open a GitHub issue](https://github.com/LLNL/Mneme/issues)
Include:

- CUDA / HIP version
- LLVM version
- Mneme version
- Minimal reproducer
- Expected vs. actual behavior

Your feedback directly drives prioritization.
