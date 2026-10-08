#!/usr/bin/env bash

setup_conda_env(){
  MINIFORGE_DIR=$(realpath -m "$1")
  LLVM_VERSION=$2
  PYTHON_VERSION=$3
  # Create the env by prefix: a named env follows the user's condarc envs_dirs,
  # which concurrent jobs then share and clobber.
  CONDA_ENV_DIR="${MINIFORGE_DIR}/envs/mneme"
  echo "Requested python version ${PYTHON_VERSION}"
  echo "Requested LLVM VERSION ${LLVM_VERSION}"
  echo "MINIFORGE_DIR is ${MINIFORGE_DIR}"
  if [[ ! -d ${MINIFORGE_DIR} ]]; then
    mkdir -p ${MINIFORGE_DIR}
    wget -q --tries=5 --retry-connrefused --wait=5 \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
      -O ${MINIFORGE_DIR}/miniforge.sh
    bash ${MINIFORGE_DIR}/miniforge.sh -b -u -p ${MINIFORGE_DIR}
    rm ${MINIFORGE_DIR}/miniforge.sh
    source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
    conda activate base
    conda create -y -p "${CONDA_ENV_DIR}" --override-channels -c conda-forge \
      python=${PYTHON_VERSION} clang=${LLVM_VERSION} clangxx=${LLVM_VERSION} \
      clangdev=${LLVM_VERSION} llvmdev=${LLVM_VERSION} lit=${LLVM_VERSION} \
      gcc=12 gxx=12
  else
    source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
  fi
  conda activate "${CONDA_ENV_DIR}"
}
