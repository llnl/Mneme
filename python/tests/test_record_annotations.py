import json

from mneme.cli import main as mneme_main


def _find_tagged(entries, tag):
    for entry in entries:
        if entry.get("Tag") == tag:
            return entry
    return None


def _approx_eq(a, b, eps=1e-12):
    return abs(a - b) <= eps


def test_record_annotations_complex_cases(build_annotation_test_program, tmp_path):
    binary = build_annotation_test_program["binary"]
    assert binary.exists(), "vecAddAnnotations binary must exist"

    out_dir = tmp_path / "record_annotations_out"
    out_dir.mkdir()

    policy = tmp_path / "filter.json"
    policy.write_text(json.dumps({"recording": {"defaults": {"max_records": 8}}}))

    rc = mneme_main(
        [
            "record",
            "--record-db-dir",
            str(out_dir),
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

    rr = json.loads(json_records[0].read_text())
    instances = rr["instances"]
    assert len(instances) >= 2, "Expected at least two dynamic kernel instances"

    record_dir = json_records[0].resolve().parent

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
