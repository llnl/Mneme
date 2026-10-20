#!/bin/bash

set -e

ml load cuda/${MNEME_CI_CUDA_VERSION}
ml load clang/${MNEME_CI_LLVM_VERSION}

# Install spack.
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_USER_CACHE_PATH=/tmp/spack-${CI_JOB_ID}/user_cache
git clone --quiet --depth=2 https://github.com/spack/spack.git /tmp/spack-${CI_JOB_ID}
source /tmp/spack-${CI_JOB_ID}/share/spack/setup-env.sh

# Create environment.
spack env create -d /tmp/mneme-spack-env-${CI_JOB_ID}
spack env activate /tmp/mneme-spack-env-${CI_JOB_ID}

# Add external packages.
LLVM_PREFIX=$(llvm-config --prefix)
# We manually add llvm as an external package to avoid spack's detection logic
# which may return incompatible versions.
spack config add --file <(cat <<EOF
packages:
  llvm:
    buildable: false
    externals:
    - spec: "llvm@${MNEME_CI_LLVM_VERSION}+clang targets=all"
      prefix: ${LLVM_PREFIX}
      extra_attributes:
        compilers:
          c: ${LLVM_PREFIX}/bin/clang
          cxx: ${LLVM_PREFIX}/bin/clang++
EOF
)

spack external find
spack external find cuda

# Add repo and package.
# Use the proteus recipe and source pinned in PROTEUS_VERSION (tag or commit).
PROTEUS_SRC=/tmp/proteus-${CI_JOB_ID}
git init --quiet ${PROTEUS_SRC}
git -C ${PROTEUS_SRC} fetch --quiet --depth 1 https://github.com/Olympus-HPC/proteus.git "$(cat ${CI_PROJECT_DIR}/PROTEUS_VERSION)"
git -C ${PROTEUS_SRC} checkout --quiet FETCH_HEAD
PROTEUS_SHA=$(git -C ${PROTEUS_SRC} rev-parse HEAD)
spack repo add ${PROTEUS_SRC}/packaging/spack/spack_repo/proteus
spack repo add ${CI_PROJECT_DIR}/packaging/spack/spack_repo/mneme
# Fetch Mneme from the CI clone because a fork PR's commit is not on GitHub.
spack config add --file <(cat <<EOF
packages:
  mneme:
    package_attributes:
      git: file://${CI_PROJECT_DIR}
EOF
)
spack add mneme@git.${CI_COMMIT_SHA} ~python +cuda cuda_arch=${MNEME_CI_CUDA_ARCH} ^cuda@${MNEME_CI_CUDA_VERSION} ^llvm@${MNEME_CI_LLVM_VERSION} ^proteus@git.${PROTEUS_SHA}=main

# Concretize and install.
spack concretize -f
spack install -v

# Cleanup.
rm -rf ${SPACK_USER_CACHE_PATH}
rm -rf /tmp/mneme-spack-env-${CI_JOB_ID}
rm -rf /tmp/spack-${CI_JOB_ID}
rm -rf ${PROTEUS_SRC}
