import json

import pytest

from mneme.recorded_execution import RecordedExecution
from mneme.mneme_types import ExperimentConfiguration
from mneme.async_executor import AsyncReplayExecutor
from mneme.cli import main as mneme_main


def test_tune(recorded_execution, has_amd_gpu, has_nvidia_gpu):
    recorded_kernel = RecordedExecution.from_json(str(recorded_execution))
    dynamic_hash = list(recorded_kernel.kernel_instances.keys())[0]
    kernel = recorded_kernel[dynamic_hash]
    baseline_config = {
        "block": {
            "x": kernel.block_dim.x,
            "y": kernel.block_dim.y,
            "z": kernel.block_dim.z,
        },
        "grid": {
            "x": kernel.grid_dim.x,
            "y": kernel.grid_dim.y,
            "z": kernel.grid_dim.z,
        },
        "shared_mem": kernel.shared_mem,
        "specialize": True,
        "set_launch_bounds": True,
        "specialize_dims": True,
        "passes": "default<O3>",
    }

    executor = AsyncReplayExecutor(
        record_db=recorded_execution,
        record_id=dynamic_hash,
        iterations=5,
        results_db_dir="./",
        num_workers=1,
    )

    baseline_result = executor.evaluate(
        ExperimentConfiguration.from_dict(baseline_config)
    )
    executor.shutdown()
    assert baseline_result.verified, "Replay run was not verified"
    assert not baseline_result.noop_verifies
    assert len(baseline_result.exec_time) == 7, "Did not execute 5 experiments"


@pytest.mark.parametrize(
    "recorded_execution, noop_verifies",
    [(1024, False), (1, True)],
    indirect=["recorded_execution"],
)
def test_replay(recorded_execution, noop_verifies, has_amd_gpu, has_nvidia_gpu, capsys):
    result = mneme_main(
        [
            "replay",
            "-rdb",
            str(recorded_execution),
            "default<O0>",
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    out = captured.out
    report, _ = json.JSONDecoder().raw_decode(out[out.index('{\n  "Replay-config"'):])
    assert report["Result"]["verified"]
    assert report["Result"]["noop_verifies"] is noop_verifies
    assert ("a kernel that does nothing would pass verification" in captured.err) is noop_verifies


def test_replay_small_allocations(build_small_allocs_program, tmp_path, capsys):
    binary = build_small_allocs_program["binary"]
    out_dir = tmp_path / "record_out"
    rc = mneme_main(["record", "--record-db-dir", str(out_dir), "--", str(binary)])
    assert rc == 0, "smallAllocs failed under mneme record"

    records = list(out_dir.glob("*.json"))
    assert len(records) == 1, "Expected one record JSON"

    capsys.readouterr()
    assert mneme_main(["replay", "-rdb", str(records[0]), "default<O0>"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["Result"]["verified"], "Replay of small allocations not verified"
