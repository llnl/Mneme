---
name: mneme
description: Set up and use Mneme to instrument CUDA or HIP applications, record and replay GPU kernels, and tune verified recordings. Use for Mneme experiments, scheduler execution, and native toolchain troubleshooting.
---

# Mneme

Mneme records GPU kernel executions so they can be replayed independently
of the original application. It saves the kernel's LLVM IR, launch
configuration, and device memory snapshots. During replay, it restores
the recorded inputs, compiles and executes the kernel, and checks the
result against the recorded output.

Use the Mneme repo, its build files, and installed CLI help to determine supported toolchains and syntax. The [repository](https://github.com/llnl/Mneme) and [documentation](https://software.llnl.gov/Mneme/) may need a newer revision than the installed package.

Read [recording and replay](references/record-replay.md) when preparing an application, inspecting recordings, or changing replay settings. Read [tuning](references/tuning.md) before doing anything tuning related. These references are in-depth explanation about the concept in relation to Mneme and HPC.

## Setup

Keep Python environment management with uv when requested. LLVM and Clang development libraries are separate native dependencies; a Python LLVM binding does not replace them. Point `LLVM_INSTALL_DIR` at the compatible toolchain prefix and follow the checkout's backend-specific installation instructions.

Keep large caches, environments, toolchains, and temporary files under the user's chosen dir(s).

Load the required site modules before building. On nodes without GPUs, specify the intended architecture instead of relying on device detection. CUDA builds using scikit-build-core can accept `--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=<target>` through `uv pip install`; choose the target for the allocated GPU. Follow the separate HIP instructions for AMD GPUs.

Use the compilers and CMake integration required by the installed Mneme. Check `mneme config cc`, `mneme config cxx`, and `mneme config cmakedir` where available. `mneme_DIR` must contain `mnemeConfig.cmake`, sometimes inside a `mneme` subdirectory of the reported path. CLI imports can require GPU driver libraries, so a login-node `libcuda.so.1` error may require checking inside a GPU allocation.

## Examples

There is a CUDA example in `scripts/vec_add_ex/vec_add.cu` and the CMake file at `scripts/vec_add_ex/CMakeLists.txt` provide a small recording workload. This example does not define the workflow for other applications or HIP.

The [Python tuning example](scripts/tune_threads.py) shows the SearchSpace and executor APIs from the supplied example. This python file can be run with the vec_add_ex. Its failure scoring has been corrected and basic timing checks added. Read the tuning reference before adapting it; its broad search and one-dimensional block assumptions are not general defaults.


## System Software and Hardware

GPU recording, replay, and tuning require a supported NVIDIA or AMD GPU
with the corresponding drivers and runtime libraries. Follow the
checkout's CUDA or HIP installation instructions to select compatible
GPU architectures, operating systems, and toolkit versions.

LLVM and Clang development libraries are native dependencies. A Python
LLVM binding does not replace them. Point `LLVM_INSTALL_DIR` at the
compatible toolchain prefix when required by the build.

Importing Mneme can also require GPU driver libraries. If a CUDA build
cannot load `libcuda.so.1`, check that the execution environment has
access to the NVIDIA driver libraries. CUDA link stubs do not provide
a runtime driver.
