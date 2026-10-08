# Installation

Mneme is open-source and available on GitHub. We recommend using the
**latest `develop` branch**, which is well tested and includes the most
recent features.

---

## Dependencies and compatibility

Mneme depends on a small set of external components.
Compatibility is defined in terms of supported ROCm versions,
CUDA/LLVM versions, Python versions, and a specific Proteus commit.

### Compatibility matrix

The following table summarizes the configurations that are regularly tested
and known to work with Mneme. These combinations are validated in CI and/or
on internal test systems.

#### AMD Systems

| ROCm version | LLVM version | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|-------------|--------------|------------|-------------|-------------|-------------|
| **6.4.3**   | **19**       | ✅         | ✅          | ✅          | ✅          |
| **7.1.1**   | **20**       | ✅         | ✅          | ✅          | ✅          |
| **7.2.0**   | **22**       | ✅         | ✅          | ✅          | ✅          |

#### NVIDIA Systems

For NVIDIA systems,
Mneme follows Proteus CI and tests CUDA 12.2.2 with the LLVM versions below.
Newer CUDA versions may be functional,
but are not part of the tested matrix.

| CUDA version | LLVM version | Python 3.10 |
|-------------|--------------|-------------|
| **12.2.2**  | **19.1.7**   | ✅          |
| **12.2.2**  | **20.1.8**   | ✅          |
| **12.2.2**  | **22.1.0**   | ✅          |

#### Notes

- ✅ **Supported**: configuration is tested and fully supported.
- Mneme relies on the LLVM/Clang toolchain shipped with the corresponding
  ROCm release on AMD systems.
- Python support refers to the Python version used to run the Mneme CLI and
  Python API; it does not affect device compilation.

> Support for additional ROCm versions and CUDA backends is planned.

### Proteus dependency and compatibility

Mneme depends on the Proteus JIT and LLVM transformation infrastructure.
Compatibility between Mneme and Proteus is defined by the tested Proteus
commit.

Mneme is regularly tested against a specific Proteus commit known to be
compatible. Users building Mneme from source are strongly encouraged to
use that commit to avoid incompatibilities.

!!! note
    Mneme requires Proteus to be built and installed as a **shared library**.
    Mneme intercepts selected Proteus entry points to inject custom
    record and replay functionality, which is not possible with a
    static-only Proteus build.

#### Tested Proteus commit

- Repository: https://github.com/Olympus-HPC/Proteus
- Commit: the full SHA in the `PROTEUS_VERSION` file at the root of the
  Mneme source tree
- Tested with: Mneme `develop`

To check out the tested commit in an existing Proteus clone:

```bash
git -C /path/to/proteus fetch origin "$(cat /path/to/mneme/PROTEUS_VERSION)"
git -C /path/to/proteus checkout FETCH_HEAD
```

Proteus must be configured with:

```bash
-DBUILD_SHARED=On -DPROTEUS_INSTALL_IMPL_HEADERS=On
```

!!! note
    Mneme may not be compatible with the latest Proteus `main` branch at all times.
    Proteus is under active development, and changes to core components may
    temporarily break compatibility with Mneme.

    Users are strongly encouraged to use the tested Proteus commit listed above.

### spdlog dependency and compatibility

Mneme depends on [spdlog](https://github.com/gabime/spdlog) for emitting logging
messages during recording and when using the C++ bindings.

Compatibility between Mneme and spdlog is defined through **library versioning**.
Mneme is currently tested against and pins spdlog version **1.15.0**.

### LLVM and toolchain requirements

Mneme requires a working **Clang/LLVM toolchain** for IR instrumentation,
kernel replay, and specialization.

The following tools and libraries must be available:

- `clang` / `clang++`
- LLVM libraries

These tools are provided by the **LLVM distribution bundled with ROCm**
on AMD systems.
Mneme is currently tested with LLVM **19**, **20**, and **22** as shipped by
supported ROCm releases 6.4.3, 7.1.1, and 7.2.0 respectively.
On NVIDIA systems,
Mneme is tested with CUDA 12.2.2 and LLVM **19.1.7**, **20.1.8**,
and **22.1.0**.

!!! note
    Mneme expects the ROCm-provided LLVM toolchain to be used.
    System-installed LLVM versions may not be compatible.
    Users are encouraged to rely on `mneme config cc`, `mneme config cxx`,
    and `mneme config cmakedir` when building applications with Mneme.


## Installation steps

### User installation (recommended)

This installation method is recommended for users who want to use Mneme
to record and replay kernels.

### AMD Systems
```bash
git clone https://github.com/LLNL/Mneme.git
cd Mneme
export LLVM_INSTALL_DIR=${ROCM_PATH}
MNEME_GPU_BACKEND=hip pip install -e .
```

### NVIDIA Systems

NVIDIA systems do not provide a proper LLVM installation. You can install one LLVM installation by using conda:
```bash
export MINIFORGE_DIR=miniforge
export LLVM_VERSION=22.1.0
PYTHON_VERSION=3.10
mkdir -p ${MINIFORGE_DIR}
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh -O ${MINIFORGE_DIR}/miniforge.sh
bash ${MINIFORGE_DIR}/miniforge.sh -b -u -p ${MINIFORGE_DIR}
rm ${MINIFORGE_DIR}/miniforge.sh
source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
conda activate base
conda create -y -n mneme --override-channels -c conda-forge \
  python=${PYTHON_VERSION} clang=${LLVM_VERSION} clangxx=${LLVM_VERSION} \
  clangdev=${LLVM_VERSION} llvmdev=${LLVM_VERSION} lit=${LLVM_VERSION} \
  gcc=12 gxx=12
conda activate mneme
```

Once you have LLVM installed you can install Mneme as:
```bash
git clone https://github.com/LLNL/Mneme.git
cd Mneme
export LLVM_INSTALL_DIR=$(llvm-config --prefix)
MNEME_GPU_BACKEND=cuda pip install -e .
```

This installs the Mneme CLI (mneme) and Python bindings along with all
tested runtime dependencies.

### Developer installation

This installation method is recommended for contributors and developers
working on Mneme itself.

For Python development, use an editable install:

```bash
pip install -e .
```

Editable mode installs Mneme in-place, allowing local source changes to be
picked up without reinstallation.

For C++ and runtime development, Mneme also provides simple setup scripts
that configure an out-of-tree CMake build against an existing Proteus
installation.
These scripts are intended for developers who want a local Mneme build
and install prefix that mirrors the Proteus developer workflow.

On AMD systems:

```bash
export PROTEUS_DIR=/path/to/proteus/install-prefix
export MNEME_ENABLE_TESTS=On
source scripts/setup-rocm.sh 7.2.0
cd build-$(hostname | sed 's/[0-9]//g')-rocm-7.2.0
cmake --build . --parallel 10
ctest --output-on-failure
cmake --install .
```

On NVIDIA systems:

```bash
export PROTEUS_DIR=/path/to/proteus/install-prefix
export MNEME_ENABLE_TESTS=On
export MNEME_CUDA_ARCHITECTURES=90
source scripts/setup-cuda.sh /path/to/llvm 12.2.2
cd build-$(hostname | sed 's/[0-9]//g')-cuda-12.2.2-llvm-$(/path/to/llvm/bin/llvm-config --version)
cmake --build . --parallel 10
ctest --output-on-failure
cmake --install .
```

`PROTEUS_DIR` should point to a compatible Proteus installation prefix
built for the same backend.
If `PROTEUS_DIR` is not set, the scripts default to the sibling
`../proteus/install-*` prefixes used by the Proteus developer scripts.

The scripts can be configured with environment variables before sourcing:

- `MNEME_ENABLE_TESTS`: configure the C++ test targets (`Off` by default).
- `MNEME_ENABLE_LOGGER`: enable logging support (`Off` by default).
- `MNEME_ENABLE_PYTHON`: build Python bindings, profiling, and autotuning (`Off` by default).
- `MNEME_LINK_SHARED_LLVM`: link against shared LLVM libraries
  (`On` by default for CUDA, `Off` by default for ROCm).
- `MNEME_CUDA_ARCHITECTURES`: CUDA architecture list for NVIDIA builds
  (`90` by default).

Additional CMake arguments may be passed after the required script
arguments.
For example:

```bash
source scripts/setup-rocm.sh 7.2.0 -DCMAKE_BUILD_TYPE=RelWithDebInfo
source scripts/setup-cuda.sh /path/to/llvm 12.2.2 -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

!!! note
    The setup scripts configure and install the native CMake project.
    They do not replace `pip install -e .` for the Mneme Python package,
    CLI, or Python bindings.

### Optional: Using an external Proteus installation

Mneme can be configured to use an **existing Proteus source tree** or a
**pre-built Proteus installation** via environment variables.

This option is intended for advanced users and application developers who
already build or ship binaries linked against Proteus. In such cases,
applications do **not** need to be rebuilt specifically for Mneme:
recording and replay functionality can be enabled by preloading Mneme
at runtime.

This capability is not part of the default installation path and is
currently less extensively tested than the bundled Proteus workflow.

#### Environment variables

The following environment variables may be used to point Mneme to an
external Proteus installation:

- `PROTEUS_SRC`: Path to a Proteus source tree. When set, the Mneme
  installer will configure and build Proteus from this source.
- `PROTEUS_DIR`: Path to an existing Proteus installation prefix.
  This directory must allow `find_package(proteus)` to succeed.

External Proteus installations must use the commit in `PROTEUS_VERSION`
and must be built with `-DBUILD_SHARED=On -DPROTEUS_INSTALL_IMPL_HEADERS=On`.

When either of these variables is set, Mneme will use the specified
Proteus installation instead of the internally managed one.

When both of these variables are set, `PROTEUS_DIR` takes priority.

!!! note
    This workflow is intended for advanced use cases.
    Users are encouraged to start with the default installation unless
    they already have an existing Proteus-based application.

### Verifying the installation

Mneme uses pytest for its Python test suite.

First, install the test dependencies:

```bash
pip install -e ".[test]"
```

Then run the tests:

```bash
pytest python/tests
```

Successful completion of the test suite indicates that Mneme and its
Python bindings are correctly installed.

## Next steps

Once Mneme is installed and the test suite completes successfully,
proceed to **[Getting Started](getting-started.md)** for a guided, end-to-end example of
building, recording, and replaying a GPU kernel with Mneme.
