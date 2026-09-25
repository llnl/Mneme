"""Installed CLI recording and replay with dynamic-hash record IDs."""

import json
import os
import struct
import subprocess
import sys

from mneme.cli import main as mneme_main
from mneme.recorded_execution import RecordedExecution


def test_default_controls(build_test_program, tmp_path, monkeypatch):
    monkeypatch.setenv("MNEME_SKIP_RECORDINGS", "1")
    monkeypatch.setenv("MNEME_MAX_RECORDINGS", "0")
    monkeypatch.setenv("MNEME_RR_KERNELS", "^_Z11vecAdd_test")
    directory = tmp_path / "records"

    assert mneme_main(
        [
            "record",
            "-sr", "2",
            "-mr", "1",
            "-rdb", str(directory),
            "--", str(build_test_program["binary"]), "1024",
        ]
    ) == 0

    databases = list(directory.glob("*.json"))
    assert len(databases) == 1
    instances = json.loads(databases[0].read_text())["instances"]
    assert len(instances) == 1
    snapshot = directory / next(iter(instances.values()))["Prologue"]

    # last arg is i % 2; skipping two launches captures iteration 2
    assert struct.unpack("<Qi", snapshot.read_bytes()[-12:]) == (4, 0)
    assert not list(directory.glob(".mneme-filter-*"))


def test_config_record_and_replay(build_test_program, tmp_path):
    policy = tmp_path / "filter.yaml"
    policy.write_text(
        """recording:
  defaults:
    skip: 1
    every: 3
    max_records: 3
"""
    )
    directory = tmp_path / "records"

    assert mneme_main(
        [
            "record",
            "--filter-config", str(policy),
            "-rdb", str(directory),
            "--", str(build_test_program["binary"]), "1024",
        ]
    ) == 0

    databases = list(directory.glob("*.json"))
    assert len(databases) == 1
    assert not list(directory.glob(".mneme-filter-*"))
    records = RecordedExecution.from_json(str(databases[0]))
    instances = json.loads(databases[0].read_text())["instances"]
    assert len(instances) == 1
    record_id, instance = next(iter(instances.items()))

    assert instance["Occurrences"] == 1
    snapshot = directory / instance["Prologue"]

    # candidates are iterations 1, 4, and 7
    assert struct.unpack("<Qi", snapshot.read_bytes()[-12:]) == (4, 1)

    assert records[record_id].dynamic_hash == record_id
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MNEME_") and key != "LD_PRELOAD"
    }
    result = subprocess.run(
        [
            sys.executable, "-m", "mneme.cli", "replay",
            "-rdb", str(databases[0]),
            "-rid", record_id,
            "default<O0>",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)

    report, _ = json.JSONDecoder().raw_decode(
        result.stdout[result.stdout.index('{\n  "Replay-config"'):]
    )
    assert report["Result"]["verified"]
    assert not report["Result"]["noop_verifies"]
