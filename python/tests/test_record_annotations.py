import json
import struct
from pathlib import Path

from mneme.cli import main as mneme_main


# Enum values from include/mneme/MnemeAnnotation.hpp
BUILTIN_F64 = 10
THRESHOLD_ABSOLUTE = 0
THRESHOLD_RELATIVE = 1
NORM_L2 = 2
NORM_LINF = 3


# The container prefix SnapshotHeader writes. Older recordings have none.
CONTAINER_MAGIC = b"MNEMESNP"
CONTAINER_SIZE = 16


def _payload_offset(data: bytes) -> int:
    if len(data) >= CONTAINER_SIZE and data[:8] == CONTAINER_MAGIC:
        return CONTAINER_SIZE
    return 0


def _parse_prologue_blob_metadata(prologue_path: Path):
    data = prologue_path.read_bytes()
    off = _payload_offset(data)

    def read_u64():
        nonlocal off
        v = struct.unpack_from("<Q", data, off)[0]
        off += 8
        return v

    def read_u8():
        nonlocal off
        v = struct.unpack_from("B", data, off)[0]
        off += 1
        return v

    def read_f64():
        nonlocal off
        v = struct.unpack_from("<d", data, off)[0]
        off += 8
        return v

    def skip(n):
        nonlocal off
        off += n

    # globals
    for _ in range(read_u64()):
        name_len = read_u64()
        skip(name_len)
        var_size = read_u64()
        skip(8)  # dev addr
        skip(var_size)

    out = []
    # blobs
    for _ in range(read_u64()):
        read_u64()  # actual size
        blob_size = read_u64()
        skip(8)  # dev addr
        skip(blob_size)

        builtin = read_u8()
        threshold = read_f64()
        threshold_kind = read_u8()
        norm = read_u8()
        tag_len = read_u64()
        tag = None
        if tag_len:
            tag = data[off : off + tag_len].decode("utf-8", errors="replace")
            skip(tag_len)

        out.append(
            {
                "builtin": builtin,
                "threshold": threshold,
                "threshold_kind": threshold_kind,
                "norm": norm,
                "tag": tag,
            }
        )

    return out


def _find_tagged(mds, tag):
    for md in mds:
        if md["tag"] == tag:
            return md
    return None


def _approx_eq(a, b, eps=1e-12):
    return abs(a - b) <= eps


def test_record_annotations_complex_cases(build_annotation_test_program, tmp_path):
    binary = build_annotation_test_program["binary"]
    assert binary.exists(), "vecAddAnnotations binary must exist"

    out_dir = tmp_path / "record_annotations_out"
    out_dir.mkdir()

    rc = mneme_main(
        [
            "record",
            "--record-db-dir",
            str(out_dir),
            "--per-kernel-max-recordings",
            "8",
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

        all_blob_md = _parse_prologue_blob_metadata(prologue)

        in_md = _find_tagged(all_blob_md, "in_vec")
        if in_md is not None:
            assert in_md["builtin"] == BUILTIN_F64
            assert _approx_eq(in_md["threshold"], 0.125)
            assert in_md["threshold_kind"] == THRESHOLD_ABSOLUTE
            assert in_md["norm"] == NORM_L2
            saw_in_vec = True

        loose_md = _find_tagged(all_blob_md, "out_loose")
        if loose_md is not None:
            assert loose_md["builtin"] == BUILTIN_F64
            assert _approx_eq(loose_md["threshold"], 0.01)
            assert loose_md["threshold_kind"] == THRESHOLD_RELATIVE
            assert loose_md["norm"] == NORM_LINF
            saw_out_loose = True

        tight_md = _find_tagged(all_blob_md, "out_tight")
        if tight_md is not None:
            assert tight_md["builtin"] == BUILTIN_F64
            assert _approx_eq(tight_md["threshold"], 0.005)
            assert tight_md["threshold_kind"] == THRESHOLD_RELATIVE
            assert tight_md["norm"] == NORM_LINF
            saw_out_tight = True

    assert saw_in_vec, "Did not find in_vec annotation in any prologue blob"
    assert saw_out_loose, "Did not find out_loose annotation in any prologue blob"
    assert saw_out_tight, "Did not find out_tight annotation in any prologue blob"
