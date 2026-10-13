#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


if [[ -f "$SCRIPT_DIR/mneme-common-ci.sh" ]]; then
  source "$SCRIPT_DIR/mneme-common-ci.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # Prefer GitLab project dir if available, else fall back to git to find repo root.
  if [[ -n "${CI_PROJECT_DIR:-}" && -f "${CI_PROJECT_DIR}/scripts/gitlab/mneme-common-ci.sh" ]]; then
    source "${CI_PROJECT_DIR}/scripts/gitlab/mneme-common-ci.sh"
  else
    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "${REPO_ROOT}" && -f "${REPO_ROOT}/scripts/gitlab/mneme-common-ci.sh" ]]; then
      source "${REPO_ROOT}/scripts/gitlab/mneme-common-ci.sh"
    else
      echo "ERROR: cannot find mneme-common-ci.sh (SCRIPT_DIR=${SCRIPT_DIR}, CI_PROJECT_DIR=${CI_PROJECT_DIR:-unset})"
      exit 1
    fi
  fi
fi

temp_dir=$(mktemp -d)
echo "Temporary directory created at: $temp_dir"
host=$(hostname)
host=${host//[0-9]/}
mkdir -p ${temp_dir}/build-${host};
build_dir=${temp_dir}/build-${host}
installDir=$(mktemp -d)

mneme_src=$(pwd)
pushd $build_dir

build_proteus() {
  echo "Building PROTEUS"
  PROTEUS_VERSION=$(cat "${mneme_src}/PROTEUS_VERSION")
  if [[ ! -d proteus ]]; then
    # fetch by ref so PROTEUS_VERSION can be a tag or a commit
    git init --quiet proteus
    git -C proteus fetch --quiet --depth 1 https://github.com/Olympus-HPC/proteus.git "${PROTEUS_VERSION}"
    git -C proteus checkout --quiet FETCH_HEAD
  fi
  pushd proteus
  PROTEUS_ENABLE_HIP=$1
  PROTEUS_ENABLE_CUDA=$2
  PROTEUS_INSTALL_DIR=$3
  echo "Proteus: ENABLE_HIP: $PROTEUS_ENABLE_HIP ENABLE_CUDA: $PROTEUS_ENABLE_CUDA"
  rm -rf build-proteus-${host}
  mkdir build-proteus-${host}
  pushd build-proteus-${host}
  cmake .. \
    -DBUILD_SHARED=On \
    -DPROTEUS_INSTALL_IMPL_HEADERS=On \
    -DCMAKE_POSITION_INDEPENDENT_CODE=On \
    -DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
    -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake" \
    -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
    -DPROTEUS_ENABLE_HIP=${PROTEUS_ENABLE_HIP} \
    -DPROTEUS_ENABLE_CUDA=${PROTEUS_ENABLE_CUDA} \
    -DCMAKE_CUDA_COMPILER=$LLVM_INSTALL_DIR/bin/clang++ \
    -DCMAKE_CUDA_FLAGS=-std=c++17 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DENABLE_TESTS=Off \
    -DCMAKE_INSTALL_PREFIX=${PROTEUS_INSTALL_DIR}
  make -j 10
  make install -j 10
  popd
  popd
}

build_spdlog() {
  echo "Building spdlog ${MNEME_CI_ENABLE_LOGGER} ${MNEME_CI_ENABLE_LOGGER^^}"
  if [[ "${MNEME_CI_ENABLE_LOGGER^^}" == "OFF" ]]; then
    echo "Early exit"
    return
  fi

  echo "Building SPDLOG"
  if [[ ! -d spdlog ]]; then
    git clone --depth 1 --branch v1.15.0  --single-branch https://github.com/gabime/spdlog.git
  fi
  pushd spdlog
  SPDLOG_INSTALL_DIR=$1
  rm -rf build-spdlog-${host}
  mkdir build-spdlog-${host}
  pushd build-spdlog-${host}
  cmake \
    -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
    -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
    -DCMAKE_INSTALL_PREFIX=${SPDLOG_INSTALL_DIR} \
    ..

  make -j 10
  make install -j 10
  popd
  popd
}

if [[ "$SYS_TYPE" == "toss_4_x86_64_ib" ]]; then
  unset CUDA_VISIBLE_DEVICES
  ml load cmake/3.30
  ml load cuda/12.2.2
  PYTHON_VERSION=3.12

# Install Clang/LLVM through conda.
setup_conda_env "miniforge3" "${MNEME_CI_LLVM_VERSION}" "$PYTHON_VERSION"

LLVM_INSTALL_DIR=$(llvm-config --prefix)
export LLVM_INSTALL_DIR=$(llvm-config --prefix)
cpp=$(which clang++)
cc=$(which clang)
echo "Setting root dir to be ${LLVM_INSTALL_DIR}"

build_proteus "OFF" "ON" $installDir
echo "After proteus Current directory is $(pwd)"
build_spdlog $installDir
echo "Current dir is $(pwd)"
cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_FLAGS=-std=c++17 \
  -DCMAKE_PREFIX_PATH="$installDir;$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake" \
  -Dproteus_DIR=$installDir \
  -DCMAKE_INSTALL_PREFIX=$installDir \
  -DCMAKE_CXX_COMPILER=${cpp} \
  -DCMAKE_C_COMPILER=${cc} \
  -DMNEME_LINK_SHARED_LLVM=ON \
  -DCMAKE_CUDA_COMPILER=${cpp} \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
  -DMNEME_GPU_BACKEND=cuda \
  -DMNEME_ENABLE_LOGGER=${MNEME_CI_ENABLE_LOGGER} \
  -DMNEME_ENABLE_TESTS=On \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=on ${mneme_src}


elif [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
  ml load cmake/3.29.2
  ml load rocm/${MNEME_CI_ROCM_VERSION}
  export LLVM_INSTALL_DIR=${ROCM_PATH}/llvm
  export CMAKE_HIP_ARCHITECTURES="gfx942;gfx90a"
  echo "LLVM INSTALL DIR is ${LLVM_INSTALL_DIR}"

  build_proteus "ON" "OFF" $installDir
  echo "After proteus Current directory is $(pwd)"
  build_spdlog $installDir
  echo "After spdlog Current directory is $(pwd)"


  cmake \
    -DCMAKE_BUILD_TYPE=Relwithdebinfo \
    -DCMAKE_PREFIX_PATH="$installDir" \
    -Dproteus_DIR=$installDir \
    -DCMAKE_INSTALL_PREFIX=$installDir \
    -DCMAKE_CXX_COMPILER=amdclang++ \
    -DCMAKE_C_COMPILER=amdclang \
    -DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES} \
    -DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
    -DMNEME_GPU_BACKEND=hip \
    -DMNEME_ENABLE_LOGGER=${MNEME_CI_ENABLE_LOGGER} \
    -DMNEME_ENABLE_TESTS=On \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=on ${mneme_src}
fi


make -j 10
echo "### TESTING ###"
ctest --output-on-failure
echo "### TESTING  ###"

make -j 10 install

popd
