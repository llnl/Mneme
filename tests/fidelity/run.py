#!/usr/bin/env python3
"""Standalone fidelity gate. Only the supervised worker imports Mneme/GPU code."""

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET


SOURCE = Path(__file__).resolve().parent
CASES = {
    "rotate": {
        "kernel": "rotate_matrix_parallel",
        "grid": [4, 1, 1],
        "block": [256, 1, 1],
        "parameters": {"size": 2048},
    },
    "stencil": {
        "kernel": "stencil3d",
        "grid": [7, 9, 9],
        "block": [16, 16, 1],
        "parameters": {"size": 128},
    },
    "nbody": {
        "kernel": "accelerate_particles",
        "grid": [4, 1, 1],
        "block": [256, 1, 1],
        "parameters": {"particles": 1024},
    },
    "consumer": {
        "kernel": "fidelity_consumer",
        "grid": [1024, 1, 1],
        "block": [256, 1, 1],
        "parameters": {"elements": 262144, "steps": 256},
    },
}

SAMPLES = 10
WARMUPS = 2
SPREAD_LIMIT = 0.05
RATIO_MIN, RATIO_MAX = 0.85, 1.15


class CorrectnessError(RuntimeError):
    pass


def valid_samples(values, count):
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(
            isinstance(x, bool)
            or not isinstance(x, (float, int))
            or not math.isfinite(x)
            or x <= 0
            for x in values
        )
    ):
        raise ValueError(f"Expected {count} positive finite GPU durations")

    return values


def measured_replays(result):
    if result.get("failed"):
        raise ValueError("Replay did not execute successfully")

    if result.get("verified") is not True or result.get("noop_verifies") is not False:
        raise ValueError("Replay must verify and must not be a no-op verification")

    if result.get("executed") is not True:
        raise ValueError("Replay did not execute successfully")

    # Mneme returns warmups and measured dispatches together, in launch order.
    return valid_samples(result.get("exec_time"), SAMPLES + WARMUPS)[WARMUPS:]


def summarize(app, replay):
    """Apply the noise check before interpreting the two-sided fidelity ratio."""

    def stats(values):
        valid_samples(values, len(values))
        median = statistics.median(values)
        mad = statistics.median(abs(x - median) for x in values)

        # Scale MAD to a standard-deviation estimate without amplifying outliers.
        return {"median_ns": median, "relative_spread": 1.4826 * mad / median}

    a, r = stats(app), stats(replay)
    ratio = r["median_ns"] / a["median_ns"]
    noisy = max(a["relative_spread"], r["relative_spread"]) > SPREAD_LIMIT

    if noisy:
        status = "unstable"
    elif RATIO_MIN <= ratio <= RATIO_MAX:
        status = "pass"
    else:
        status = "fidelity"

    return {
        "app": a,
        "replay": r,
        "ratio": ratio,
        "signed_error_pct": (ratio - 1) * 100,
        "status": status,
    }


def collect_batches(collect, result, checkpoint):
    """Only noise triggers a second batch; all observations remain in the gate."""
    app, replay = [], []
    result["batches"] = []

    for _ in range(2):
        batch = collect()
        result["batches"].append(batch)

        # A recheck augments the first batch; it never replaces a worse result.
        app.extend(valid_samples(batch["app_ns"], SAMPLES))
        replay.extend(valid_samples(batch["replay_ns"], SAMPLES))
        result["summary"] = summarize(app, replay)
        checkpoint()

        if result["summary"]["status"] != "unstable":
            break

    result["status"] = result["summary"]["status"]


def write_reports(output, report):
    """Checkpoint both formats so interrupted runs retain completed case results."""
    output.mkdir(parents=True, exist_ok=True)

    # Replace complete files atomically; the supervisor may read after a timeout.
    path = output / "results.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

    suite = ET.Element("testsuite", name="mneme-fidelity", tests=str(len(report["cases"])))
    failures = errors = 0

    for result in report["cases"]:
        test = ET.SubElement(
            suite,
            "testcase",
            classname="fidelity",
            name=result["case"],
            time=str(result.get("elapsed_s", 0)),
        )
        status = result["status"]

        if status != "pass":
            if status in ("fidelity", "unstable", "correctness", "duration"):
                kind = "failure"
            else:
                kind = "error"

            failures += kind == "failure"
            errors += kind == "error"
            ET.SubElement(
                test,
                kind,
                type=status,
                message=result.get("error", status),
            ).text = json.dumps(result, indent=2)

        ET.SubElement(test, "system-out").text = json.dumps(result, indent=2)

    suite.set("failures", str(failures))
    suite.set("errors", str(errors))
    path = output / "junit.xml"
    temporary = path.with_suffix(".tmp")
    ET.ElementTree(suite).write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(path)


def command(args, log, cwd=None, application=False):
    # Stream logs to disk so compiler/application failures and timeouts retain output.
    started = time.monotonic()

    with log.open("w") as stream:
        stream.write("$ " + " ".join(map(str, args)) + "\n")
        stream.flush()
        result = subprocess.run(
            [str(x) for x in args],
            cwd=cwd,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )

    text = log.read_text().partition("\n")[2]
    with log.open("a") as stream:
        stream.write(f"\n# elapsed_seconds: {time.monotonic() - started:.3f}\n")

    # The native harness reserves exit code 3 for failed output verification.
    if application and result.returncode == 3:
        raise CorrectnessError(f"Native application verification failed; see {log.name}")

    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log.name}")

    return text


def application_json(text):
    lines = [line for line in text.splitlines() if line.startswith('{"case":')]
    if len(lines) != 1:
        raise ValueError("Expected one application result")

    return json.loads(lines[0])


def validate_application(result, name, timed):
    case = CASES[name]
    if (
        result.get("case") != name
        or result.get("verified") is not True
        or result.get("launches") != 5
        or result.get("sampled_invocation") != 5
        or result.get("grid") != case["grid"]
        or result.get("block") != case["block"]
    ):
        raise ValueError("Application identity, verification, or launch geometry mismatch")

    valid_samples(result.get("durations_ns"), 1 if timed else 0)


def validate_record(data, name):
    case = CASES[name]
    if not data.get("DemangledName", "").startswith(case["kernel"] + "("):
        raise ValueError("Unexpected recorded kernel")

    if data.get("TotalLaunches") != 5 or len(data.get("instances", {})) != 1:
        raise ValueError("Expected five launches and exactly one recorded instance")

    rid, instance = next(iter(data["instances"].items()))
    for key, expected in (("GridDims", case["grid"]), ("BlockDims", case["block"])):
        if [instance[key][axis] for axis in ("x", "y", "z")] != expected:
            raise ValueError("Recorded launch geometry mismatch")

    if instance["SharedMem"] != 0:
        raise ValueError("Unexpected dynamic shared memory")

    return rid


def build(work, output, report):
    from mneme.device import get_device_arch
    from mneme.llvm.utils import get_profile_library

    arch = get_device_arch()
    backend = "hip" if arch.startswith("gfx") else "cuda"

    # Compile against the same installation and toolchain used for replay.
    config = {}
    for key in ("cc", "cxx", "cmakedir", "llvmdir"):
        config[key] = command(
            [sys.executable, "-m", "mneme.cli", "config", key],
            output / f"config-{key}.log",
        ).strip()

    compiler_version = command([config["cxx"], "--version"], output / "compiler.log")
    report["environment"].update(
        backend=backend,
        arch=arch,
        compiler=compiler_version,
        config=config,
        profile_library=str(get_profile_library()),
    )

    args = [
        "cmake",
        "-S", SOURCE,
        "-B", work / "build",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DFIDELITY_BACKEND={backend}",
        f"-DCMAKE_C_COMPILER={config['cc']}",
        f"-DCMAKE_CXX_COMPILER={config['cxx']}",
        f"-DCMAKE_PREFIX_PATH={config['cmakedir']};{config['llvmdir']}",
    ]
    lang = backend.upper()
    cmake_arch = arch.split(":")[0] if backend == "hip" else arch.removeprefix("sm_")
    args += [
        f"-DCMAKE_{lang}_COMPILER={config['cxx']}",
        f"-DCMAKE_{lang}_ARCHITECTURES={cmake_arch}",
    ]

    command(args, output / "configure.log")
    command(["cmake", "--build", work / "build", "-j", "1"], output / "build.log")

    return backend, str(get_profile_library())


def run_case(name, work, output, report, result, backend, profile_library):
    from mneme.async_executor import AsyncReplayExecutor
    from mneme.mneme_types import ExperimentConfiguration

    case = CASES[name]
    directory = work / name
    directory.mkdir()
    logs = output / name
    logs.mkdir()
    result.update(configuration=case, invocation=5, stage="recording")

    def checkpoint():
        write_reports(output, report)

    checkpoint()

    # JSON is also accepted by the YAML filter loader. No ambient recording policy.
    # Only the fifth target launch is captured; all prefix kernels still execute.
    policy = {
        "recording": {
            "defaults": {"skip": 4, "every": 1, "max_records": 0},
            "kernel_overrides": [
                {
                    "match": {"demangled_regex": "^" + case["kernel"] + r"\("},
                    "max_records": 1,
                },
            ],
        },
    }
    policy_path = logs / "filter.json"
    policy_path.write_text(json.dumps(policy, indent=2))
    records = directory / "records"
    text = command(
        [
            sys.executable, "-m", "mneme.cli", "record",
            "--filter-config", policy_path,
            "-rdb", records,
            "--", work / "build/fidelity_record", name,
        ],
        logs / "record.log",
        directory,
        application=True,
    )
    record_app = application_json(text)
    validate_application(record_app, name, False)

    databases = list(records.glob("*.json"))
    if len(databases) != 1:
        raise ValueError("Expected exactly one recorded kernel database")

    database = json.loads(databases[0].read_text())
    (logs / "record-metadata.json").write_text(json.dumps(database, indent=2))
    rid = validate_record(database, name)
    result.update(record_id=rid, kernel=database["KernelName"], gpu=record_app["gpu"])

    config = ExperimentConfiguration.from_dict(
        {
            "grid": dict(zip(("x", "y", "z"), case["grid"])),
            "block": dict(zip(("x", "y", "z"), case["block"])),
            "shared_mem": 0,
        }
    )
    target = database["KernelName"] + (".kd" if backend == "hip" else "")
    executor = None
    batch_number = 0

    try:
        def collect():
            nonlocal executor, batch_number

            batch_number += 1
            result["stage"] = "profiling"
            checkpoint()

            # Each native sample comes from a fresh process with the same prefix.
            apps = []
            for i in range(SAMPLES):
                text = command(
                    [work / "build/fidelity_native", name, profile_library, target],
                    logs / f"app-{batch_number}-{i}.log",
                    directory,
                    application=True,
                )
                app = application_json(text)
                validate_application(app, name, True)
                if app["gpu"] != result["gpu"]:
                    raise ValueError("Application GPU changed")
                apps.append(app)

            result["stage"] = "replay"
            checkpoint()

            # Reuse one replay worker for a noise recheck, after native sampling.
            if executor is None:
                executor = AsyncReplayExecutor(
                    str(databases[0]),
                    rid,
                    SAMPLES,
                    str(logs),
                    num_workers=1,
                    warmup=WARMUPS,
                    short_circuit=True,
                )

            replay = executor.evaluate(config).to_dict()
            (logs / f"replay-{batch_number}.json").write_text(json.dumps(replay, indent=2))
            if not replay.get("failed") and (
                replay.get("verified") is not True
                or replay.get("noop_verifies") is not False
            ):
                raise CorrectnessError(
                    "Replay must verify and must not be a no-op verification"
                )

            result["stage"] = "profiling" if replay.get("executed") is True else "replay"
            times = measured_replays(replay)

            return {
                "app_ns": [app["durations_ns"][0] for app in apps],
                "replay_ns": times,
                "replay_warmup_ns": replay["exec_time"][:WARMUPS],
                "executed": replay["executed"],
                "verified": replay["verified"],
                "noop_verifies": replay["noop_verifies"],
            }

        collect_batches(collect, result, checkpoint)
        if name == "rotate" and result["summary"]["app"]["median_ns"] < 1_000_000:
            result.update(status="duration", error="MatrixRotate must take at least 1 ms")
    finally:
        if executor is not None:
            executor.shutdown()


def worker(args):
    """Run GPU work in a child process whose entire process group can be stopped."""
    multiprocessing.set_start_method("spawn", force=True)
    output = args.output_dir
    report = json.loads((output / "results.json").read_text())
    start = time.monotonic()

    try:
        backend, library = build(args.work_dir, output, report)
    except Exception:
        for result in report["cases"]:
            result.update(status="build", error=traceback.format_exc())
        write_reports(output, report)
        return 1

    for result in report["cases"]:
        case_start = time.monotonic()
        result.update(status="running", stage="recording")

        try:
            run_case(result["case"], args.work_dir, output, report, result, backend, library)
        except CorrectnessError:
            result.update(status="correctness", error=traceback.format_exc())
        except Exception:
            result.update(status=result.get("stage", "error"), error=traceback.format_exc())

        result["elapsed_s"] = time.monotonic() - case_start
        report["elapsed_s"] = time.monotonic() - start
        write_reports(output, report)

    return int(any(result["status"] != "pass" for result in report["cases"]))


def supervise(args):
    output = args.output_dir.resolve()

    # Refuse to mix a new run with artifacts from an earlier run.
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")

    names = list(CASES) if args.case == "all" else [args.case]
    report = {
        "schema_version": 1,
        "policy": {
            "ratio_range": [RATIO_MIN, RATIO_MAX],
            "spread_limit": SPREAD_LIMIT,
            "samples_per_batch": SAMPLES,
            "warmups": WARMUPS,
            "max_batches": 2,
            "timeout_s": args.timeout,
            "wall_time_target_s": 120,
        },
        "environment": {
            "host": socket.gethostname(),
            "python": sys.version,
            "variables": {
                key: os.environ[key]
                for key in (
                    "CI_COMMIT_SHA",
                    "CI_MACHINE",
                    "LOADEDMODULES",
                    "ROCM_PATH",
                    "CUDA_HOME",
                    "CUDA_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "HSA_XNACK",
                    "MPIBIND_RESTRICT",
                    "SLURM_JOB_ID",
                    "FLUX_JOB_ID",
                )
                if key in os.environ
            },
        },
        "cases": [{"case": name, "status": "pending"} for name in names],
    }
    revision = subprocess.run(
        ["git", "-C", str(SOURCE.parents[1]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    report["environment"]["mneme_revision"] = revision.stdout.strip()
    report["environment"]["proteus_version_requested"] = (
        (SOURCE.parents[1] / "PROTEUS_VERSION").read_text().strip()
    )
    report["environment"]["cpu_affinity"] = sorted(os.sched_getaffinity(0))

    # Save pending results before starting the worker, including for build timeouts.
    write_reports(output, report)
    start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="mneme-fidelity-") as work:
        with (output / "runner.log").open("w") as log:
            proc = subprocess.Popen(
                [
                    sys.executable, __file__, "--worker",
                    "--work-dir", work,
                    "--output-dir", str(output),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False

            try:
                proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            finally:
                # Reap replay/compiler grandchildren too, including on a worker crash.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()

        # Keep completed case results and classify anything the worker left pending.
        report = json.loads((output / "results.json").read_text())
        unfinished = any(r["status"] in ("pending", "running") for r in report["cases"])
        for result in report["cases"]:
            if result["status"] in ("pending", "running"):
                result.update(
                    status="timeout" if timed_out else "error",
                    error=(
                        "Suite deadline exceeded"
                        if timed_out
                        else f"Worker exited {proc.returncode}; see runner.log"
                    ),
                )

        # Exit 1 is expected for failed cases. A crash after all cases report still
        # needs its own failing JUnit entry, even if every individual case passed.
        if not unfinished and (
            timed_out
            or proc.returncode not in (0, 1)
            or (
                proc.returncode
                and all(r["status"] == "pass" for r in report["cases"])
            )
        ):
            report["cases"].append(
                {
                    "case": "suite",
                    "status": "timeout" if timed_out else "error",
                    "error": (
                        "Worker did not exit cleanly after reporting cases; "
                        "see runner.log"
                    ),
                }
            )

        report["elapsed_s"] = time.monotonic() - start
        write_reports(output, report)

    print("case       status        app us    replay us    error %")
    for result in report["cases"]:
        s = result.get("summary")
        if s:
            values = (
                f"{s['app']['median_ns'] / 1000:9.2f} "
                f"{s['replay']['median_ns'] / 1000:12.2f}"
                f" {s['signed_error_pct']:+9.2f}"
            )
        else:
            values = "       --           --         --"
        print(f"{result['case']:10} {result['status']:12} {values}")

    print(f"Elapsed: {report['elapsed_s']:.1f}s; reports: {output}")
    return int(
        timed_out
        or proc.returncode != 0
        or any(r["status"] != "pass" for r in report["cases"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")

    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    sys.exit(main())
