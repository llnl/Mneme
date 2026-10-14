import glob
import hashlib
import json
import os
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Binary .mneme prologue parser
#
# On-disk format (all integers little-endian, pointer-sized = 8 bytes):
#   SnapshotHeader: char[8] "MNEMESNP"; uint32 Kind; uint32 Version (absent in
#     older recordings)
#   uint64  TotalGlobals
#   for each global:
#     uint64 StrLen; char[StrLen] name; uint64 VarSize; void* DevAddr; char[VarSize] data
#   uint64  TotalBlobs
#   for each blob:
#     uint64 ActualSize; uint64 Size; void* DevAddr; char[Size] data
#     Metadata: uint8 builtin; double threshold; uint8 threshold_kind;
#               uint8 norm; uint64 tag_len; char[tag_len] tag
#   uint64  NumArgs
#   for each arg: uint64 ArgSize; char[ArgSize] data
# ---------------------------------------------------------------------------

_CONTAINER_MAGIC = b"MNEMESNP"
_CONTAINER_SIZE = 16


def _payload_offset(data):
    """Byte offset of the payload of a bytes snapshot."""
    if len(data) >= _CONTAINER_SIZE and data[:8] == _CONTAINER_MAGIC:
        return _CONTAINER_SIZE
    return 0


def _parse_prologue_metadata(filename):
    """Return a list of Metadata dicts for every blob in a prologue file."""
    with open(filename, "rb") as f:
        data = f.read()
    off = _payload_offset(data)

    def u64():
        nonlocal off
        v = struct.unpack_from("<Q", data, off)[0]
        off += 8
        return v

    def u8():
        nonlocal off
        v = struct.unpack_from("B", data, off)[0]
        off += 1
        return v

    def dbl():
        nonlocal off
        v = struct.unpack_from("<d", data, off)[0]
        off += 8
        return v

    def skip(n):
        nonlocal off
        off += n

    # globals
    for _ in range(u64()):
        str_len = u64()
        skip(str_len)           # name
        var_size = u64()
        skip(8)                 # dev_addr
        skip(var_size)          # data

    # blobs
    results = []
    for _ in range(u64()):
        u64()                   # actual_size
        size = u64()
        skip(8)                 # dev_addr
        skip(size)              # blob data
        builtin       = u8()
        threshold     = dbl()
        threshold_kind = u8()
        norm          = u8()
        tag_len       = u64()
        tag = None
        if tag_len:
            tag = data[off:off + tag_len].decode("utf-8", errors="replace")
            skip(tag_len)
        results.append(
            dict(builtin=builtin, threshold=threshold,
                 threshold_kind=threshold_kind, norm=norm, tag=tag)
        )
    return results

data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
for fn in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
    with open(fn, "r") as fd:
        rr_data = json.load(fd)
    base_dir = Path(fn).resolve().parent

    d_name = rr_data["DemangledName"]
    print("DemangledName:", rr_data["DemangledName"])
    print("NumModules:", len(rr_data["Modules"]))
    print("NumInstances:", len(rr_data["instances"]))
    if "TotalLaunches" in rr_data:
        print("TotalLaunches:", rr_data["TotalLaunches"])
        recorded_launches = sum(i["Occurrences"] for i in rr_data["instances"].values())
        print("UnrecordedLaunches:", rr_data["TotalLaunches"] - recorded_launches)
    if "TotalTimeNs" in rr_data:
        recorded_time = sum(
            i["TimeNs"]["Sum"] for i in rr_data["instances"].values() if "TimeNs" in i
        )
        print("TimingSource:", rr_data["TimingSource"])
        print("TotalTimeNs:", rr_data["TotalTimeNs"])
        print("UnrecordedTimeNs:", rr_data["TotalTimeNs"] - recorded_time)
    if "SourceFile" in rr_data:
        print("SourceFile:", rr_data["SourceFile"])
    if "SourceLine" in rr_data:
        print("SourceLine:", rr_data["SourceLine"])
    if "SourceEndLine" in rr_data:
        print("SourceEndLine:", rr_data["SourceEndLine"])
    if "SourceCopy" in rr_data:
        copy_path = base_dir / rr_data["SourceCopy"]
        if not copy_path.exists():
            print("Expected source copy to exist")
            sys.exit(-1)
        digest = hashlib.md5(copy_path.read_bytes()).hexdigest()
        print("SourceCopy:", rr_data["SourceCopy"])
        print("SourceMD5:", "ok" if digest == rr_data.get("SourceMD5") else "mismatch")
        source_lines = copy_path.read_text().splitlines()
        if "SourceLine" in rr_data:
            print("SourceText:", source_lines[rr_data["SourceLine"] - 1].strip())
        if "SourceEndLine" in rr_data:
            print("SourceEndText:", source_lines[rr_data["SourceEndLine"] - 1].strip())
    dims = lambda d: (d["x"], d["y"], d["z"])
    instances = sorted(
        rr_data["instances"].values(),
        key=lambda i: (dims(i["GridDims"]), dims(i["BlockDims"]), i["SharedMem"]),
    )
    for instance in instances:
        print(
            "BlockDims:({0}, {1}, {2})".format(
                instance["BlockDims"]["x"],
                instance["BlockDims"]["y"],
                instance["BlockDims"]["z"],
            )
        )
        print(
            "GridDims:({0}, {1}, {2})".format(
                instance["GridDims"]["x"],
                instance["GridDims"]["y"],
                instance["GridDims"]["z"],
            )
        )
        print(
            "SharedMem:{0} Occurrences:{1}".format(
                instance["SharedMem"], instance["Occurrences"]
            )
        )
        if "TimedLaunches" in instance:
            line = "Timing: TimedLaunches:{0}".format(instance["TimedLaunches"])
            t = instance.get("TimeNs")
            if t is not None:
                line += " SumNs:{0} MinNs:{1} MaxNs:{2}".format(t["Sum"], t["Min"], t["Max"])
            recorded = instance.get("RecordedLaunchTimeNs")
            if recorded is not None:
                line += " RecordedNs:{0}".format(recorded)
            print(line)
        prologue_path = base_dir / instance["Prologue"]
        epilogue_path = base_dir / instance["Epilogue"]
        if not prologue_path.exists():
            print("Expected prologue file to exist")
            sys.exit(-1)
        if not epilogue_path.exists():
            print("Expected epilogue file to exist")
            sys.exit(-1)

        # Parse the prologue binary to report any non-default blob metadata.
        try:
            mds = _parse_prologue_metadata(prologue_path)
            for md in mds:
                if (md["threshold"] != 0.0 or md["builtin"] != 0
                        or md["norm"] != 0 or md["threshold_kind"] != 0
                        or md["tag"] is not None):
                    parts = [
                        f"threshold={md['threshold']}",
                        f"threshold_kind={md['threshold_kind']}",
                        f"builtin={md['builtin']}",
                        f"norm={md['norm']}",
                    ]
                    if md["tag"] is not None:
                        parts.append(f"tag={md['tag']}")
                    print("BlobAnnotation:", " ".join(parts))
        except Exception as e:
            print(f"Warning: could not parse prologue metadata: {e}",
                  file=sys.stderr)


sys.exit(0)
