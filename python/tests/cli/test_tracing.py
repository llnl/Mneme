import json
import os
import subprocess
import sys


def run_python(args, trace_dir=None):
    # run helper for trace dir
    env = os.environ.copy()
    env.pop("MNEME_TRACE_DIR", None)

    if trace_dir is not None:
        env["MNEME_TRACE_DIR"] = str(trace_dir)
    
    result = subprocess.run(
        [sys.executable, *args], env=env, capture_output=True, text=True, timeout=90,
    )
    
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_replay_and_tuning_host_trace(recorded_execution, tmp_path):
    args = [
        "-m", "mneme.cli", "replay", "-rdb", str(recorded_execution),
        "--iterations", "2", "default<O0>",
    ]

    disabled = run_python(args)
    
    trace_dir = tmp_path / "replay-traces"
    enabled = run_python(args, trace_dir)
    for report in (disabled, enabled):
        assert report["Result"]["verified"]
        assert report["Result"]["executed"]
        assert len(report["Result"]["exec_time"]) == 4
    assert disabled["Replay-config"] == enabled["Replay-config"]
    
    path, = trace_dir.glob("*.json")
    events = json.loads(path.read_text())
    for name in ("launch", "state.restore", "sync.pre_launch", "sync.post_launch"):
        assert sum(event["name"] == name for event in events) == 5
    
    assert sum(event["name"] == "sync.device" for event in events) == 2
    
    comparison, = [e for e in events if e["name"] == "verify.compare"]
    assert comparison["args"]["verified"] is True

    # Spawn a real tuning worker and reuse it for two requests.
    logs = tmp_path / "workers"
    logs.mkdir()
    trace_dir = tmp_path / "tune-traces"
    report = run_python([
        "-c", """
import json
import sys
from pathlib import Path
from mneme import cli  # Select the CLI's spawn multiprocessing context.
from mneme.async_executor import AsyncReplayExecutor
from mneme.mneme_types import ExperimentConfiguration
from mneme.recorded_execution import RecordedExecution

if __name__ == "__main__":
    recording = RecordedExecution.from_json(sys.argv[1])
    record_id = next(iter(recording))
    kernel = recording[record_id]
    config = ExperimentConfiguration(
        grid=kernel.grid_dim, block=kernel.block_dim,
        shared_mem=kernel.shared_mem, passes="default<O0>",
    )
    executor = AsyncReplayExecutor(Path(sys.argv[1]), record_id, 2, sys.argv[2], 1, warmup=1)
    try:
        results = [executor.submit(config).result(timeout=30) for _ in range(2)]
        assert all(result is not None and result.verified for result in results)
        print(json.dumps([len(result.exec_time) for result in results]))
    finally:
        executor.shutdown()
        """, str(recorded_execution), str(logs),
    ], trace_dir)
    
    assert report == [3, 3]
    
    # make sure the trace actually gets written and contains the expected events
    path, = trace_dir.glob("*.json")
    events = json.loads(path.read_text())
    requests = [e["args"]["request_id"] for e in events if e["name"] == "tuning.request"]
    
    assert requests == [0, 1]
    assert sum(event["name"] == "experiment" for event in events) == 2
    assert sum(event["name"] == "launch" for event in events) == 8
