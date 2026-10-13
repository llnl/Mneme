#!/usr/bin/bash

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

trap 'echo "[ERROR] line $LINENO" >&2' ERR

log() { echo " #### [$(date +%T)] $* ###" >&2; }

log "CI job $CI_JOB_ID starting"

if [[ -n "$CODECOV_TOKEN" ]]; then
  log "CODECOV_TOKEN is set"
else
  log "CODECOV_TOKEN is not set"
fi

log "CI JOB ID IS ${CI_JOB_ID}"

test_dir="$(mktemp -d)"
mkdir -p ${test_dir}

# Make a copy of src to local FS to accelerate building etc. 
log "Start copying src to temp"
mneme_orig_src=$(pwd)
mkdir -p $test_dir/mneme_src/
rsync -a --delete "$mneme_orig_src" "$test_dir/mneme_src/"
mneme_src=$test_dir/mneme_src/Mneme
log "End copying mneme src to ${mneme_src} from ${mneme_orig_src}"

if [[ "$SYS_TYPE" == "toss_4_x86_64_ib" ]]; then
  unset CUDA_VISIBLE_DEVICES
  ml load cmake/3.30
  ml load cuda/12.2.2
  setup_conda_env "${test_dir}/miniforge3" "${MNEME_CI_LLVM_VERSION}" "${MNEME_CI_PYTHON_VERSION}"
  export LLVM_INSTALL_DIR=$(llvm-config --prefix)
  export MNEME_GPU_BACKEND=cuda
elif [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
  ml load cmake/3.29.2
  ml load python/${MNEME_CI_PYTHON_VERSION}
  ml load rocm/${MNEME_CI_ROCM_VERSION}
  export LLVM_INSTALL_DIR=${ROCM_PATH}/
  export MNEME_GPU_BACKEND=hip
fi
log "Test dir is ${test_dir}"
log "Using LLVM under ${LLVM_INSTALL_DIR}"

# Make a copy of preinstalled deps to local FS for fast installation
#VENV_NAME="/usr/workspace/LExperts/ci/gitlab/venv-${LCSCHEDCLUSTER}-${MNEME_CI_ROCM_VERSION}-${MNEME_CI_PYTHON_VERSION}/"
#rm -rf ${VENV_NAME}/lib*/python*/site-packages/mneme

log "Starting making environment"
LOCAL_VENV_NAME="${test_dir}/venv/"
mkdir -p ${LOCAL_VENV_NAME}
python -m venv "${LOCAL_VENV_NAME}"
#rsync -a "$VENV_NAME/" "$LOCAL_VENV_NAME"
pushd ${test_dir}
source ${LOCAL_VENV_NAME}/bin/activate
echo "Environment is: ${LOCAL_VENV_NAME}"
echo "ENV will use python:"
which python

python -m pip uninstall -y mneme
log "Environment made"

log "Start installing mneme"
python -m pip  -v install ${mneme_src}
log "Finalized with installation"

log "Installing pytest"
python -m pip install pytest pytest-cov

log "Start running mneme tests"
pytest -v -s ${mneme_src}/python/tests/ || exit $? 
log "Done with testing"
pushd ${mneme_src}
pytest --cov-report=xml:coverage-${CI_JOB_ID}.xml --cov-config=.coveragerc python/tests/

# we only need to upload reports once and we only need to test editable installs once.
if [[ "${MNEME_CI_PYTHON_VERSION}" == "3.10" && "${MNEME_CI_ROCM_VERSION}" == "7.2.0" ]]; then
  # Upload to Codecov (only if token is available)
  if [[ -n "$CODECOV_TOKEN" ]]; then
    log "Uploading coverage to Codecov..."
    log "SHA is $CI_COMMIT_SHA"
    log "Branch is $CI_COMMIT_BRANCH"
    export NODE_TLS_REJECT_UNAUTHORIZED=0

    curl -k -Os https://uploader.codecov.io/latest/linux/codecov
    chmod +x codecov
    ./codecov \
      -t "$CODECOV_TOKEN" \
      -f coverage-${CI_JOB_ID}.xml \
      -C "$CI_COMMIT_SHA" \
      -B "$CI_COMMIT_BRANCH" \
      --insecure \
      --disable-ci-detection \
      --slug=gh/LLNL/Mneme || echo "Codecov upload failed"
  else
    echo "No CODECOV_TOKEN set, skipping upload."
  fi

  rm -f coverage-${CI_JOB_ID}.xml


  python -m pip uninstall -y mneme
  python -m pip install -U pip setuptools wheel

# Try a editable install
log "Removed mneme"
log "Mneme src is ${mneme_src}"
log "Start editable install"
python -m pip install -e ${mneme_src}
# If everything is properly installed this command will not fail
log "End editable install"

mneme config cxx || exit $?
fi

deactivate
rm -rf ${test_dir}
