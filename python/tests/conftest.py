import pathlib
import subprocess
from multiprocessing import Event
from queue import Queue as ThreadQueue

import pytest
from mneme.cli import main as mneme_main
from mneme.futures import EvalFuture
from mneme.recorded_execution import RecordedExecution


@pytest.fixture
def has_nvidia_gpu():
    try:
        subprocess.check_output("nvidia-smi", shell=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


@pytest.fixture
def has_amd_gpu():
    try:
        output = subprocess.check_output("rocminfo", shell=True, text=True)
        return "AMD" in output or "gfx" in output
    except subprocess.CalledProcessError:
        return False


def detect_local_sm_via_nvidia_smi() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        text=True,
    ).strip()
    # If multiple GPUs, take the first line
    cc = out.splitlines()[0].strip()  # e.g. "9.0"
    major, minor = cc.split(".")
    return int(major + minor)  # "9"+"0" -> 90


# This is correct, as any tests that uses "subprocess" to test the cli will not be affected by "autoreuse"
@pytest.fixture(autouse=True)
def disable_all_dispose(monkeypatch):
    monkeypatch.setattr("mneme.llvm.buffer.MemBufferRef._dispose", lambda self: None)


def run_checked(cmd, cwd=None):
    """Run subprocess and print full output on error."""
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print("\n=========================================")
        print(" COMMAND FAILED ")
        print("=========================================")
        print("Cmd:", " ".join(e.cmd))
        print("Return code:", e.returncode)
        print("---- STDOUT ----")
        print(e.stdout)
        print("---- STDERR ----")
        print(e.stderr)
        print("=========================================\n")
        raise


def capture_stdout(func, argv):
    import io
    import sys

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        func(argv)
    finally:
        sys.stdout = old
    return buf.getvalue().strip()


# ----------------------------
# Shared helper to call mneme config
# ----------------------------
def call_mneme_config(arg: str) -> str:
    """Call the CLI entry point directly and capture output."""
    return capture_stdout(mneme_main, ["config", arg])


# ----------------------------
# Build vecAdd only once per session per RDC setting
# ----------------------------
@pytest.fixture(scope="session")
def build_cache():
    return {}


def build_vecadd(
    rdc_value, tmp_path_factory, call_mneme_config, has_amd_gpu, has_nvidia_gpu
):
    has_nvidia = has_nvidia_gpu
    has_amd = has_amd_gpu
    cuda_arch = "native"

    if has_nvidia:
        cuda_arch = detect_local_sm_via_nvidia_smi()

    key = f"vecadd_{rdc_value.lower()}"
    tmpdir = tmp_path_factory.mktemp(key)

    src_dir = pathlib.Path(__file__).parents[0] / "c_src" / "cmake"

    mneme_cc = call_mneme_config("cc")
    mneme_cxx = call_mneme_config("cxx")
    mneme_cmake_dir = call_mneme_config("cmakedir")
    if has_nvidia and has_amd:
        raise RuntimeError("Cannot build test on system with 2 GPU architectures")
    if not has_nvidia and not has_amd:
        raise RuntimeError("Cannot build test on system with no GPU available")

    nv_flag = "On" if has_nvidia else "Off"
    amd_flag = "On" if has_amd else "Off"

    subprocess.run(
        [
            "cmake",
            f"-DCMAKE_C_COMPILER={mneme_cc}",
            f"-DENABLE_CUDA={nv_flag}",
            f"-DCMAKE_CUDA_COMPILER={mneme_cxx}",
            "-DCMAKE_CUDA_FLAGS=-std=c++17",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            f"-DENABLE_HIP={amd_flag}",
            f"-DCMAKE_CXX_COMPILER={mneme_cxx}",
            f"-DCMAKE_PREFIX_PATH={mneme_cmake_dir}",
            f"-DRDC={rdc_value}",
            str(src_dir),
        ],
        cwd=tmpdir,
        check=True,
    )

    run_checked(["make", "-j"], cwd=tmpdir)

    return {
        "src_dir": src_dir,
        "build_dir": tmpdir,
        "binary": tmpdir / "vecAdd",
        "rdc": rdc_value,
        "cc": mneme_cc,
        "cxx": mneme_cxx,
        "cmakedir": mneme_cmake_dir,
    }


def build_test_target(
    target, source, tmp_path_factory, call_mneme_config, has_amd_gpu, has_nvidia_gpu
):
    has_nvidia = has_nvidia_gpu
    has_amd = has_amd_gpu
    cuda_arch = "native"

    if has_nvidia:
        cuda_arch = detect_local_sm_via_nvidia_smi()

    tmpdir = tmp_path_factory.mktemp(target)
    src_dir = pathlib.Path(__file__).parents[0] / "c_src" / "cmake"

    mneme_cc = call_mneme_config("cc")
    mneme_cxx = call_mneme_config("cxx")
    mneme_cmake_dir = call_mneme_config("cmakedir")
    if has_nvidia and has_amd:
        raise RuntimeError("Cannot build test on system with 2 GPU architectures")
    if not has_nvidia and not has_amd:
        raise RuntimeError("Cannot build test on system with no GPU available")

    nv_flag = "On" if has_nvidia else "Off"
    amd_flag = "On" if has_amd else "Off"

    subprocess.run(
        [
            "cmake",
            f"-DCMAKE_C_COMPILER={mneme_cc}",
            f"-DENABLE_CUDA={nv_flag}",
            f"-DCMAKE_CUDA_COMPILER={mneme_cxx}",
            "-DCMAKE_CUDA_FLAGS=-std=c++17",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            f"-DENABLE_HIP={amd_flag}",
            f"-DCMAKE_CXX_COMPILER={mneme_cxx}",
            f"-DCMAKE_PREFIX_PATH={mneme_cmake_dir}",
            "-DRDC=Off",
            f"-DTEST_TARGET_NAME={target}",
            f"-DTEST_SOURCE_FILE={source}",
            str(src_dir),
        ],
        cwd=tmpdir,
        check=True,
    )

    run_checked(["make", "-j"], cwd=tmpdir)

    return {
        "src_dir": src_dir,
        "build_dir": tmpdir,
        "binary": tmpdir / target,
        "rdc": "Off",
        "cc": mneme_cc,
        "cxx": mneme_cxx,
        "cmakedir": mneme_cmake_dir,
    }


@pytest.fixture(params=["On", "Off"])
def build_test_program(
    request, tmp_path_factory, build_cache, has_amd_gpu, has_nvidia_gpu
):
    rdc = request.param
    key = f"vecadd_cached_{rdc}"

    if key not in build_cache:
        build_cache[key] = build_vecadd(
            rdc, tmp_path_factory, call_mneme_config, has_amd_gpu, has_nvidia_gpu
        )

    return build_cache[key]


@pytest.fixture
def build_annotation_test_program(
    tmp_path_factory, build_cache, has_amd_gpu, has_nvidia_gpu
):
    key = "vecadd_annotations_cached"
    if key not in build_cache:
        build_cache[key] = build_test_target(
            "vecAddAnnotations",
            "vec_add_annotations.cu",
            tmp_path_factory,
            call_mneme_config,
            has_amd_gpu,
            has_nvidia_gpu,
        )
    return build_cache[key]


@pytest.fixture
def build_small_allocs_program(
    tmp_path_factory, build_cache, has_amd_gpu, has_nvidia_gpu
):
    key = "small_allocs_cached"
    if key not in build_cache:
        build_cache[key] = build_test_target(
            "smallAllocs",
            "small_allocs.cu",
            tmp_path_factory,
            call_mneme_config,
            has_amd_gpu,
            has_nvidia_gpu,
        )
    return build_cache[key]


@pytest.fixture
def build_chunked_allocs_program(
    tmp_path_factory, build_cache, has_amd_gpu, has_nvidia_gpu
):
    key = "chunked_allocs_cached"
    if key not in build_cache:
        build_cache[key] = build_test_target(
            "chunkedAllocs",
            "chunked_allocs.cu",
            tmp_path_factory,
            call_mneme_config,
            has_amd_gpu,
            has_nvidia_gpu,
        )
    return build_cache[key]


@pytest.fixture
def recorded_execution(build_test_program, tmp_path):
    """
    Test that 'mneme record <binary>' correctly generates
    a JSON recording file and registers at least one kernel.
    """

    binary = build_test_program["binary"]
    assert binary.exists(), "vecAdd binary must exist"

    out_dir = tmp_path / "record_out"
    out_dir.mkdir()

    # -------------------------------------
    # Run: mneme record <binary> -o <out_json>
    # -------------------------------------
    args = [
        "record",
        "--record-db-dir",
        str(out_dir),
        "--",
        str(binary),
        "1024",
    ]

    result = mneme_main(args)
    assert result == 0, "mneme record returned non-zero exit code"
    records = list(out_dir.glob("*.json"))

    # -------------------------------------
    # Verify that the JSON file exists
    # -------------------------------------
    assert len(records) == 1, f"Expected to have a single record file"

    for r in records:
        text = r.read_text()
        assert len(text) > 0, f"JSON file: {r} is empty"
        recorded_execution = RecordedExecution.from_json(r)
        assert (
            len(recorded_execution.llvm_files) == 1
        ), "Mneme record should have 1 llvm file"
        assert (
            len(recorded_execution.kernel_instances.keys()) == 1
        ), "Recording should had recorded single kernel"
        kernel_instance = recorded_execution[
            list(recorded_execution.kernel_instances.keys())[0]
        ]
        assert (
            len(kernel_instance.available_specializations) == 2
        ), "vecAdd allows 2 specializations"
        return r
