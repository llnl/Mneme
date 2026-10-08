# Getting Started

This guide walks through recording and replaying a simple GPU kernel
using Mneme.

## Prerequisites

- A capable system:
    - AMD supporting ROCm 6.4.3, 7.1.1, or 7.2.0 or,
    - NVIDIA supporting CUDA 12.2.2 with LLVM 19.1.7, 20.1.8, or 22.1.0.
- CMake and a C++ compiler
- Python 3.9+

For full installation details, see **[Usage → Install](install.md)**.

## Install Mneme (quick path)

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


## Execute Example Code

### Phase 1: Instrumentation (compile time) 

#### AMD Systems
Build the provided example code:

```bash
cmake -B build-example -S examples/hip_vec_add/ -DCMAKE_C_COMPILER=$(mneme config cc) -DCMAKE_CXX_COMPILER=$(mneme config cxx) -DCMAKE_PREFIX_PATH=$(mneme config cmakedir)
cmake --build build-example/
```

#### NVIDIA Systems
Build the provided example code:

```bash
cmake -B build-example -S examples/cuda_vec_add/ -DCMAKE_CUDA_FLAGS=-std=c++17 -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_C_COMPILER=$(mneme config cc) -DCMAKE_CUDA_COMPILER=$(mneme config cxx)  -DCMAKE_CXX_COMPILER=$(mneme config cxx) -DCMAKE_PREFIX_PATH=$(mneme config cmakedir)
cmake --build build-example/
```


!!! note
    The `cmake` build command will emit a warning `clang++: warning: argument unused during compilation: '-Xoffload-linker --load-pass-plugin=<path-to>/libProteusPass.so'`. It is safe to ignore the warning.

!!! note
    On CUDA systems you need to pass a couple of extra cmake flags `-DCMAKE_CUDA_FLAGS=-std=c++17`, `-DCMAKE_CUDA_ARCHITECTURES=90` and `-DCMAKE_CUDA_COMPILER=$(mneme config cxx)`. The `c++17` flags are necessary due to an existing LLVM@18 and cmake bug. 

### Phase 2: Record Executable (runtime)

To record the previously built example code you can run:

```bash
mkdir record-example-dir/
mneme record -rdb record-example-dir/ -- ./build-example/vecAdd 1024
```
After the execution of the example you can check the generated recorded artifacts under the `record-example-dir` directory. For example:

```bash
./record-example-dir/
├── 15941914485064662553.json
├── DeviceState.epilogue.15941914485064662553.18248140455151687155.mneme
├── DeviceState.prologue.15941914485064662553.18248140455151687155.mneme
└── RecordedIR_15941914485064662553.bc
```

!!! note
    The exact filenames will differ depending on your compiler versions and Mneme versions. However, there should be a single `*.json` file, `DeviceState.epilogue.*` and a `DeviceState.prologue.*` file and a `RecordedIR_*.bc` file.

### Phase 3: Replay kernel 

```bash
mneme replay -rdb record-example-dir/15941914485064662553.json "default<O3>"
```

This recording contains one kernel instance,
so Mneme selects its record ID automatically.
The command will execute the kernel and **replay** the exact same execution
as the recorded one.
If the command completes successfully and produces a JSON file,
recording worked.
The output will be similar to:

```json
{
  "Replay-config": {
    "grid": {
      "x": 40000,
      "y": 1,
      "z": 1
    },
    "block": {
      "x": 256,
      "y": 1,
      "z": 1
    },
    "shared_mem": 0,
    "specialize": false,
    "set_launch_bounds": false,
    "max_threads": null,
    "min_blocks_per_sm": 0,
    "specialize_dims": false,
    "passes": "default<O3>",
    "codegen_opt": 3,
    "codegen_method": "serial",
    "prune": true,
    "internalize": true
  },
  "Result": {
    "preprocess_ir_time": 9.2298723757267e-06,
    "opt_time": 0.006206092890352011,
    "codegen_time": 0.01226967596448958,
    "obj_size": 4792,
    "exec_time": [
      84040,
      82561,
      81761,
      83360,
      76520
    ],
    "verified": true,
    "executed": true,
    "failed": false,
    "start_time": "",
    "end_time": "",
    "gpu_id": 0,
    "const_mem_usage": -1,
    "local_mem_usage": 0,
    "reg_usage": 12,
    "error": ""
  }
}
```

!!! note
    The exact `-rdb` filename will differ.
    It points to the JSON file under `./record-example-dir/`.
    If that database contains multiple entries under `instances`,
    select one explicitly with `-rid <record-id>`.

!!! note
    If verified on the *Result* entry is true and executed is true, **Congratulations** you have replayed successfully a vector addition.

!!! tip "Tolerance-aware verification"
    By default, replay verification uses a byte-exact comparison.  If your kernel produces floating-point results and you want to accept small numerical differences across optimization pipelines, you can annotate device buffers with `mneme::annotate()` before recording. See **[Usage → Verification](verification.md)** for the full API and examples.
