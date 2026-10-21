"""Check selected invocation arguments without loading the replay runtime."""

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys


def check_case(
    binary,
    library,
    root,
    name,
    scenario,
    expected,
    overrides=None,
    preload=True,
    policy=None,
):
    directory = root / name
    directory.mkdir(parents=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MNEME_")
    }
    env.update(
        MNEME_FILTER_MODE="scoped",
        MNEME_DATA_DIR=str(directory),
        MNEME_LOG_LEVEL="off",
        MNEME_RECORD_RANKS="all",
    )
    if policy is not None:
        config_dir = directory / "config"
        config_dir.mkdir()
        config = config_dir / "filter.json"
        config.write_text(json.dumps({"recording": policy}))
        env["MNEME_FILTER_MODE"] = "config"
        env["MNEME_FILTER_CONFIG"] = str(config)
        env.pop("MNEME_RECORD_RANKS")

    env.pop("LD_PRELOAD", None)
    if preload:
        env["LD_PRELOAD"] = library
    env.update(overrides or {})

    result = subprocess.run([binary, scenario], env=env, capture_output=True, text=True)
    assert result.returncode == 0, (name, result.stdout, result.stderr)

    found = {}
    for path in directory.glob("*.json"):
        record = json.loads(path.read_text())
        kernel = "foo" if "foo<" in record["DemangledName"] else "bar"
        assert kernel not in found, (name, record["DemangledName"])
        # Every scenario launches each kernel nine times, recorded or not.
        assert record["TotalLaunches"] == 9, (name, record["TotalLaunches"])

        iterations = []
        for dynamic_hash, instance in record["instances"].items():
            prologue = (directory / instance["Prologue"]).read_bytes()
            # Bytes snapshots end with the arguments. Both test kernels end
            # with an int iteration, preceded by its uint64 size.
            size, iteration = struct.unpack("<Qi", prologue[-12:])
            assert size == 4, (name, size)
            assert (directory / instance["Epilogue"]).is_file()
            assert Path(instance["Prologue"]).name.endswith(f".{dynamic_hash}.mneme")
            assert Path(instance["Epilogue"]).name.endswith(f".{dynamic_hash}.mneme")
            iterations.append(iteration)

        found[kernel] = sorted(iterations)

    assert found == expected, (name, found, expected)

    assert "deprecated" not in result.stderr, (name, result.stderr)
    controls = [
        key
        for key in (
            "MNEME_SKIP_RECORDINGS",
            "MNEME_MAX_RECORDINGS",
            "MNEME_RR_KERNELS",
        )
        if key in env
    ]
    if controls and env["MNEME_FILTER_MODE"] != "default" and preload:
        assert result.stderr.count("ignored") == 1, (name, result.stderr)
        assert all(key in result.stderr for key in controls), (name, result.stderr)
    else:
        assert "ignored" not in result.stderr, (name, result.stderr)

    print(f"PASS: {name}", flush=True)
    return directory


def main():
    binary, library, output = sys.argv[1:]
    root = Path(output).resolve()
    both = {"foo": [5], "bar": [5]}
    cases = [
        ("wildcard", "wildcard", both, {}),
        ("target", "target", {"foo": [5]}, {}),
        ("indirect", "indirect", {"foo": [5]}, {}),
        ("overload", "overload", {"bar": [5]}, {}),
        ("nested", "nested", {"bar": [5]}, {}),
        ("blocked", "blocked", {}, {}),
        ("inner-wildcard", "inner-wildcard", both, {}),
        ("inner-kernel", "inner-kernel", {"foo": [5]}, {}),
        ("no-scopes", "none", {}, {}),
        ("dedup", "sampling", {"foo": [3], "bar": [3]}, {}),
        ("scoped-limit", "limit", {"foo": [3, 4, 5, 6], "bar": [3, 4, 5, 6]}, {}),
        (
            "default", "blocked", {"foo": [0], "bar": [0]},
            {"MNEME_FILTER_MODE": "default"},
        ),
        (
            "default-limit", "limit",
            {"foo": [0, 1, 2, 3], "bar": [0, 1, 2, 3]},
            {"MNEME_FILTER_MODE": "default"},
        ),
        (
            "default-controls", "limit", {"foo": [2, 3]},
            {
                "MNEME_FILTER_MODE": "default",
                "MNEME_SKIP_RECORDINGS": "2",
                "MNEME_MAX_RECORDINGS": "2",
                "MNEME_RR_KERNELS": "foo",
            },
        ),
        (
            "default-mangled", "none", {"foo": [0]},
            {"MNEME_FILTER_MODE": "default", "MNEME_RR_KERNELS": "^_Z3foo"},
        ),
        (
            "scoped-ignored", "wildcard", both,
            {
                "MNEME_SKIP_RECORDINGS": "invalid",
                "MNEME_MAX_RECORDINGS": "0",
                "MNEME_RR_KERNELS": "[",
            },
        ),
        (
            "excluded-rank", "wildcard", {},
            {"MNEME_RECORD_RANKS": "0", "FLUX_TASK_RANK": "1"},
        ),
        (
            "included-rank", "wildcard", both,
            {"MNEME_RECORD_RANKS": "all", "FLUX_TASK_RANK": "1"},
        ),
    ]

    try:
        for name, scenario, expected, overrides in cases:
            check_case(binary, library, root, name, scenario, expected, overrides)
        check_case(binary, library, root, "no-preload", "wildcard", {}, preload=False)
    finally:
        if root.exists():
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
