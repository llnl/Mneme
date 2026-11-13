"""CPU-only policy/protocol checks; no Mneme import or GPU is required."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

spec = importlib.util.spec_from_file_location(
    "fidelity_runner", Path(__file__).with_name("run.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    "ratio,status",
    [
        (0.8499, "fidelity"),
        (0.85, "pass"),
        (1, "pass"),
        (1.15, "pass"),
        (1.1501, "fidelity"),
    ],
)
def test_two_sided_boundaries(ratio, status):
    assert runner.summarize([100] * 10, [ratio * 100] * 10)["status"] == status


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "100", None])
def test_invalid_samples(value):
    with pytest.raises(ValueError):
        runner.valid_samples([100] * 9 + [value], 10)


def test_sample_count():
    with pytest.raises(ValueError):
        runner.valid_samples([100] * 9, 10)


def replay_result():
    return {
        "executed": True,
        "verified": True,
        "noop_verifies": False,
        "failed": False,
        "exec_time": [99999, 99999] + [100] * 10,
    }


def test_warmups_excluded_exactly_once():
    result = replay_result()

    assert runner.measured_replays(result) == [100] * 10
    assert len(result["exec_time"]) == 12


@pytest.mark.parametrize(
    "key,value",
    [
        ("executed", False),
        ("verified", False),
        ("noop_verifies", True),
        ("failed", True),
    ],
)
def test_replay_must_execute_and_verify_meaningfully(key, value):
    result = replay_result()
    result[key] = value

    with pytest.raises(ValueError):
        runner.measured_replays(result)

    del result[key]
    if key != "failed":
        with pytest.raises(ValueError):
            runner.measured_replays(result)


def test_stable_fidelity_failure_has_no_retry():
    calls = []

    def collect():
        calls.append(1)
        return {"app_ns": [100] * 10, "replay_ns": [130] * 10}

    result = {}
    runner.collect_batches(collect, result, lambda: None)

    assert result["status"] == "fidelity"
    assert len(calls) == 1


def test_noise_recheck_pools_both_batches():
    batches = [
        {"app_ns": [80] * 5 + [120] * 5, "replay_ns": [130] * 10},
        {"app_ns": [100] * 10, "replay_ns": [100] * 10},
    ]
    result = {}
    runner.collect_batches(iter(batches).__next__, result, lambda: None)

    assert result["batches"] == batches

    # Choosing only the second batch would pass. Pooled replay remains unstable.
    assert result["status"] == "unstable"
    assert result["summary"]["replay"]["median_ns"] == 115


def test_one_recheck_can_resolve_noise():
    batches = [
        {"app_ns": [90] * 3 + [100] * 4 + [110] * 3, "replay_ns": [100] * 10},
        {"app_ns": [100] * 10, "replay_ns": [100] * 10},
    ]
    result = {}
    runner.collect_batches(iter(batches).__next__, result, lambda: None)

    assert len(result["batches"]) == 2
    assert result["status"] == "pass"


def record():
    return {
        "DemangledName": "stencil3d(double const*, double*)",
        "TotalLaunches": 5,
        "instances": {
            "abc": {
                "GridDims": {"x": 7, "y": 9, "z": 9},
                "BlockDims": {"x": 16, "y": 16, "z": 1},
                "SharedMem": 0,
            },
        },
    }


def test_record_identity_and_geometry():
    data = record()
    assert runner.validate_record(data, "stencil") == "abc"

    for key, value in [
        ("TotalLaunches", 4),
        ("DemangledName", "prepare_stencil()"),
        ("instances", {}),
        ("instances", {**data["instances"], "def": data["instances"]["abc"]}),
    ]:
        bad = {**data, key: value}
        with pytest.raises(ValueError):
            runner.validate_record(bad, "stencil")

    bad = copy.deepcopy(data)
    bad["instances"]["abc"]["GridDims"]["x"] = 8
    with pytest.raises(ValueError):
        runner.validate_record(bad, "stencil")


def test_application_requires_fifth_sample_and_correct_geometry():
    result = {
        "case": "rotate",
        "verified": True,
        "launches": 5,
        "sampled_invocation": 5,
        "grid": [4, 1, 1],
        "block": [256, 1, 1],
        "durations_ns": [100],
    }
    runner.validate_application(result, "rotate", True)

    result["durations_ns"] = [100, 100]
    with pytest.raises(ValueError):
        runner.validate_application(result, "rotate", True)


def test_reports_preserve_failure_categories(tmp_path):
    report = {
        "cases": [
            {"case": name, "status": status, "error": "detail <&>"}
            for name, status in zip(
                runner.CASES, ["pass", "fidelity", "unstable", "timeout"]
            )
        ],
    }
    runner.write_reports(tmp_path, report)

    assert json.loads((tmp_path / "results.json").read_text()) == report
    suite = ET.parse(tmp_path / "junit.xml").getroot()
    assert suite.attrib["failures"] == "2"
    assert suite.attrib["errors"] == "1"
    assert suite.findall("testcase")[3].find("error").attrib["type"] == "timeout"


def test_native_verification_failure_is_distinct(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kw: SimpleNamespace(returncode=3)
    )

    with pytest.raises(runner.CorrectnessError):
        runner.command(
            ["python", "-m", "mneme.cli", "record"],
            tmp_path / "record.log",
            application=True,
        )

    assert (tmp_path / "record.log").read_text().startswith("$ python")


def test_supervisor_timeout_writes_failing_reports(tmp_path, monkeypatch):
    child = tmp_path / "slow.py"
    child.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setattr(runner, "__file__", str(child))
    output = tmp_path / "results"
    args = SimpleNamespace(output_dir=output, case="all", timeout=0.1)

    assert runner.supervise(args) == 1

    report = json.loads((output / "results.json").read_text())
    assert all(r["status"] == "timeout" for r in report["cases"])
    assert ET.parse(output / "junit.xml").getroot().attrib["errors"] == "4"


def test_supervisor_preserves_fidelity_failure_and_exit_status(tmp_path, monkeypatch):
    child = tmp_path / "failure.py"
    child.write_text(
        "import json, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "report = json.loads((out / 'results.json').read_text())\n"
        "report['cases'][0]['status'] = 'fidelity'\n"
        "(out / 'results.json').write_text(json.dumps(report))\n"
        "sys.exit(1)\n"
    )
    monkeypatch.setattr(runner, "__file__", str(child))
    output = tmp_path / "results"
    assert runner.supervise(
        SimpleNamespace(output_dir=output, case="stencil", timeout=10)
    ) == 1

    report = json.loads((output / "results.json").read_text())
    assert report["cases"][0]["status"] == "fidelity"
    assert ET.parse(output / "junit.xml").getroot().attrib["failures"] == "1"


def test_ci_keeps_full_ordinary_matrix_and_separate_fidelity_matrix():
    import itertools
    import yaml

    repo = Path(__file__).resolve().parents[2]

    # These files use !reference only to refer to inherited launcher snippets.
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor(
        "!reference", lambda loader, node: loader.construct_sequence(node)
    )

    for machine in ("matrix", "tioga", "tuolumne"):
        data = yaml.load(
            (repo / f".gitlab/jobs/{machine}-python-cov.yml").read_text(),
            Loader=Loader,
        )
        compiler = "MNEME_CI_LLVM_VERSION" if machine == "matrix" else "MNEME_CI_ROCM_VERSION"
        ordinary = []

        for row in data[".build-variants"]["parallel"]["matrix"]:
            ordinary.extend(
                itertools.product(row[compiler], row["MNEME_CI_PYTHON_VERSION"])
            )

        job = data[f"{machine}-python-fidelity"]
        selected = job["variables"]
        fidelity = []
        for row in job["parallel"]["matrix"]:
            fidelity.extend(
                itertools.product(row[compiler], row["MNEME_CI_PYTHON_VERSION"])
            )

        versions = (
            ["19.1.7", "20.1.8", "22.1.0"]
            if machine == "matrix"
            else ["6.4.3", "7.1.1", "7.2.0"]
        )
        pythons = ["3.10"] if machine == "matrix" else ["3.9", "3.10", "3.11", "3.12"]
        assert sorted(ordinary) == sorted(itertools.product(versions, pythons))
        assert fidelity == [(versions[-1], "3.10")]
        assert selected["MNEME_CI_FIDELITY"] == "1"
        assert "fidelity" in selected["ALLOC_NAME"]


def test_worker_crash_after_pass_cannot_emit_passing_junit(tmp_path, monkeypatch):
    child = tmp_path / "crash.py"
    child.write_text(
        "import json, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "report = json.loads((out / 'results.json').read_text())\n"
        "report['cases'][0]['status'] = 'pass'\n"
        "(out / 'results.json').write_text(json.dumps(report))\n"
        "sys.exit(9)\n"
    )
    monkeypatch.setattr(runner, "__file__", str(child))
    output = tmp_path / "results"
    assert runner.supervise(
        SimpleNamespace(output_dir=output, case="stencil", timeout=10)
    ) == 1

    assert ET.parse(output / "junit.xml").getroot().attrib["errors"] == "1"


@pytest.mark.parametrize(
    "fidelity,backend,exit_code",
    [
        (False, "hip", 0),
        (True, "hip", 0),
        (True, "cuda", 0),
        (True, "hip", 1),
        (True, "cuda", 1),
    ],
)
def test_ci_dispatch_separates_fidelity_from_ordinary_tests(
    tmp_path, fidelity, backend, exit_code
):
    # Execute the real post-build shell flow with command stubs: no compiler,
    # scheduler, package installation, or GPU is needed to check job dispatch.
    repo = Path(__file__).resolve().parents[2]
    script = (repo / "scripts/gitlab/ci-python-test.sh").read_text()
    _, marker, post_build = script.partition('log "Installing pytest"')
    assert marker

    stubs = r'''
set -e
log() { :; }
python() {
  echo "python $*"
  for arg in "$@"; do
    if [[ "$arg" == */tests/fidelity/run.py ]]; then
      return "$TEST_FIDELITY_EXIT"
    fi
  done
}
pytest() { echo "pytest $*"; }
flux() {
  echo "flux $*"
  while [[ "$1" != python ]]; do shift; done
  "$@"
}
mneme() { echo "mneme $*"; }
deactivate() { echo deactivate; }
rm() { echo "rm $*"; }
'''
    result = subprocess.run(
        ["bash", "-c", stubs + marker + post_build],
        env={
            **os.environ,
            "MNEME_CI_FIDELITY": "1" if fidelity else "0",
            "MNEME_GPU_BACKEND": backend,
            "MNEME_CI_ROCM_VERSION": "7.2.0",
            "MNEME_CI_PYTHON_VERSION": "3.10",
            "CI_JOB_ID": "test",
            "CI_PROJECT_DIR": str(tmp_path),
            "CODECOV_TOKEN": "",
            "mneme_src": str(repo),
            "test_dir": str(tmp_path),
            "TEST_FIDELITY_EXIT": str(exit_code),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == exit_code, result.stderr
    assert "deactivate" in result.stdout
    assert f"rm -rf {tmp_path}" in result.stdout

    if fidelity:
        assert result.stdout.count("python -m pytest -o addopts=") == 1
        assert "tests/fidelity/run.py" in result.stdout
        assert ("flux run" in result.stdout) == (backend == "hip")
        assert "python/tests" not in result.stdout
        assert "--cov-report" not in result.stdout
        assert "pip install -e" not in result.stdout
    else:
        assert "tests/fidelity/" not in result.stdout
        assert "pytest -v -s" in result.stdout
        assert "--cov-report" in result.stdout
        assert "pip install -e" in result.stdout
