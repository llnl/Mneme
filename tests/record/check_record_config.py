"""Exercise native filtering and optionally replay each dynamic-hash record."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from check_record_scope import check_case


def replay_records(database, records):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MNEME_") and key != "LD_PRELOAD"
    }

    for record_id in records:
        result = subprocess.run(
            [
                sys.executable, "-m", "mneme.cli", "replay",
                "-rdb", str(database),
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
        assert report["Result"]["verified"], report
        assert not report["Result"]["noop_verifies"], report
        print(f"PASS: replay {database.name} instance {record_id}", flush=True)


def main():
    binary, library, output, *options = sys.argv[1:]
    root = Path(output).resolve()

    try:
        for name, scenario, iterations, occurrences in (
            ("same-configuration", "blocked", [1], [1]),
            ("distinct-configurations", "limit", [1, 3, 5], [1, 1, 1]),
        ):
            directory = check_case(
                binary, library, root, name, scenario,
                {"foo": iterations, "bar": iterations},
                policy={
                    "kernel_overrides": [
                        {
                            "match": {"demangled_regex": "foo|bar"},
                            "skip": 1,
                            "every": 2,
                            "max_records": 3,
                        }
                    ]
                },
            )
            for database in directory.glob("*.json"):
                records = json.loads(database.read_text())["instances"]
                observed = sorted(r["Occurrences"] for r in records.values())
                assert observed == sorted(occurrences), (name, observed, occurrences)
                assert len({r["Prologue"] for r in records.values()}) == len(iterations)
                assert len({r["Epilogue"] for r in records.values()}) == len(iterations)
                if "--replay" in options and name == "distinct-configurations":
                    replay_records(database, records)

        check_case(
            binary, library, root, "config-ignored", "blocked",
            {"foo": [2], "bar": [2]},
            {
                "MNEME_SKIP_RECORDINGS": "invalid",
                "MNEME_MAX_RECORDINGS": "0",
                "MNEME_RR_KERNELS": "[",
            },
            policy={"defaults": {"skip": 2}},
        )
        check_case(
            binary, library, root, "disabled", "none", {},
            policy={"defaults": {"max_records": 0}},
        )
        check_case(
            binary, library, root, "first-match", "none", {"foo": [5]},
            policy={
                "defaults": {"skip": 1, "every": 2, "max_records": 0},
                "kernel_overrides": [
                    {"match": {"demangled_regex": "foo"}, "skip": 5, "max_records": 1},
                    {"match": {"demangled_regex": "foo"}, "skip": 0},
                ],
            },
        )

        directory = check_case(
            binary, library, root, "dedup-limit", "mixed",
            {"foo": [1, 5], "bar": [1, 5]},
            policy={"defaults": {"skip": 1, "every": 2, "max_records": 2}},
        )
        for database in directory.glob("*.json"):
            records = json.loads(database.read_text())["instances"]
            assert sorted(r["Occurrences"] for r in records.values()) == [1, 2]

        directory = check_case(
            binary, library, root, "full-limit", "none",
            {"foo": [0], "bar": [0]},
            policy={"defaults": {"max_records": 1}},
        )
        for database in directory.glob("*.json"):
            records = json.loads(database.read_text())["instances"]
            assert next(iter(records.values()))["Occurrences"] == 1

        for name, ranks, overrides, expected in (
            ("yaml-excluded", [], {}, {}),
            ("yaml-included", [1], {"FLUX_TASK_RANK": "1"}, {"foo": [0], "bar": [0]}),
            (
                "env-excluded", "all",
                {"FLUX_TASK_RANK": "1", "MNEME_RECORD_RANKS": "0"}, {},
            ),
            ("env-included", [], {"MNEME_RECORD_RANKS": "all"}, {"foo": [0], "bar": [0]}),
        ):
            check_case(
                binary, library, root, name, "none", expected, overrides,
                policy={"defaults": {"ranks": ranks}},
            )
    finally:
        if root.exists():
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
