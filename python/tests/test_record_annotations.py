import json

import pytest

from mneme.cli import main as mneme_main


def _find_tagged(entries, tag):
    for entry in entries:
        if entry.get("Tag") == tag:
            return entry
    return None


def _approx_eq(a, b, eps=1e-12):
    return abs(a - b) <= eps


def _record(binary, out_dir):
    assert binary.exists(), "vecAddAnnotations binary must exist"
    out_dir.mkdir()

    policy = tmp_path / "filter.json"
    policy.write_text(json.dumps({"recording": {"defaults": {"max_records": 8}}}))

    rc = mneme_main(
        [
            "record",
            "--record-db-dir",
            str(out_dir),
            "-vass",
            "2",
            "--filter-config",
            str(policy),
            "--",
            str(binary),
            "1024",
        ]
    )
    assert rc == 0, "mneme record returned non-zero exit code"

    json_records = list(out_dir.glob("*.json"))
    assert len(json_records) == 1, "Expected one record JSON"
    return json_records[0]


def test_record_annotations_complex_cases(build_annotation_test_program, tmp_path):
    record = _record(
        build_annotation_test_program["binary"], tmp_path / "record_annotations_out"
    )

    rr = json.loads(record.read_text())
    instances = rr["instances"]
    assert len(instances) >= 2, "Expected at least two dynamic kernel instances"

    record_dir = record.resolve().parent

    saw_in_vec = False
    saw_out_loose = False
    saw_out_tight = False

    for instance in instances.values():
        prologue = record_dir / instance["Prologue"]
        epilogue = record_dir / instance["Epilogue"]
        assert prologue.exists(), f"Missing prologue snapshot: {prologue}"
        assert epilogue.exists(), f"Missing epilogue snapshot: {epilogue}"

        all_blob_md = instance["Compare"]

        in_md = _find_tagged(all_blob_md, "in_vec")
        if in_md is not None:
            assert in_md["DType"] == "f64"
            assert _approx_eq(in_md["Threshold"], 0.125)
            assert in_md["ThresholdKind"] == "absolute"
            assert in_md["Norm"] == "l2"
            saw_in_vec = True

        loose_md = _find_tagged(all_blob_md, "out_loose")
        if loose_md is not None:
            assert loose_md["DType"] == "f64"
            assert _approx_eq(loose_md["Threshold"], 0.01)
            assert loose_md["ThresholdKind"] == "relative"
            assert loose_md["Norm"] == "linf"
            saw_out_loose = True

        tight_md = _find_tagged(all_blob_md, "out_tight")
        if tight_md is not None:
            assert tight_md["DType"] == "f64"
            assert _approx_eq(tight_md["Threshold"], 0.005)
            assert tight_md["ThresholdKind"] == "relative"
            assert tight_md["Norm"] == "linf"
            saw_out_tight = True

    assert saw_in_vec, "Did not find in_vec annotation in any instance"
    assert saw_out_loose, "Did not find out_loose annotation in any instance"
    assert saw_out_tight, "Did not find out_tight annotation in any instance"


# A policy loose enough that the input of the first launch, whose output
# buffer is tagged out_loose, already matches its output.
LOOSE_POLICY = {"Tags": {"out_loose": {"ThresholdKind": "absolute", "Threshold": 1e9}}}
MISSING_TAG_POLICY = {"Tags": {"missing": {"Threshold": 1.0}}}


def _record_loose_instance(binary, out_dir):
    record = _record(binary, out_dir)
    instances = json.loads(record.read_text())["instances"]
    record_id = next(
        rid
        for rid, instance in instances.items()
        if _find_tagged(instance["Compare"], "out_loose") is not None
    )
    return record, record_id


def test_compare_policy_overrides_recorded_tolerance(
    build_annotation_test_program, tmp_path, capsys
):
    record, record_id = _record_loose_instance(
        build_annotation_test_program["binary"], tmp_path / "record_out"
    )
    policy = tmp_path / "policy.json"

    def replay(*args):
        rc = mneme_main(["replay", "-rdb", str(record), "-rid", record_id, *args, "default<O0>"])
        assert rc == 0
        out = capsys.readouterr().out
        report, _ = json.JSONDecoder().raw_decode(out[out.index('{\n  "Replay-config"'):])
        return report["Result"]

    result = replay()
    assert result["verified"]
    assert not result["noop_verifies"]

    policy.write_text(json.dumps(LOOSE_POLICY))
    result = replay("--compare-policy", str(policy))
    assert result["verified"]
    assert result["noop_verifies"]

    policy.write_text(json.dumps(MISSING_TAG_POLICY))
    with pytest.raises(ValueError, match="no blob has tag 'missing'"):
        replay("--compare-policy", str(policy))


def test_tune_verifies_with_compare_policy(
    build_annotation_test_program, tmp_path, capsys
):
    record, record_id = _record_loose_instance(
        build_annotation_test_program["binary"], tmp_path / "record_out"
    )
    policy = tmp_path / "policy.json"

    def tune(results_dir):
        return mneme_main(
            [
                "tune",
                "-rdb",
                str(record),
                "-rid",
                record_id,
                "--compare-policy",
                str(policy),
                "--baseline-only",
                "--trials",
                "1",
                "--results-dir",
                str(results_dir),
                "--no-proteus-output",
                "--quiet",
            ]
        )

    policy.write_text(json.dumps(LOOSE_POLICY))
    assert tune(tmp_path / "tune") == 0
    assert json.loads((tmp_path / "tune" / "summary.json").read_text())["noop_verifies"]

    policy.write_text(json.dumps(MISSING_TAG_POLICY))
    assert tune(tmp_path / "tune_missing") == 3
    assert "no blob has tag 'missing'" in capsys.readouterr().out
