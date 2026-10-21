import glob
import hashlib
import json
import os
import sys
from pathlib import Path

data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
for fn in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
    with open(fn, "r") as fd:
        rr_data = json.load(fd)
    base_dir = Path(fn).resolve().parent

    d_name = rr_data["DemangledName"]
    print("DemangledName:", rr_data["DemangledName"])
    print("NumModules:", len(rr_data["Modules"]))
    print("NumInstances:", len(rr_data["instances"]))
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
    for k, instance in rr_data["instances"].items():
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
        prologue_path = base_dir / instance["Prologue"]
        epilogue_path = base_dir / instance["Epilogue"]
        if not prologue_path.exists():
            print("Expected prologue file to exist")
            sys.exit(-1)
        if not epilogue_path.exists():
            print("Expected epilogue file to exist")
            sys.exit(-1)

        for entry in instance["Compare"]:
            parts = [
                f"{key}={entry[key]}"
                for key in ("DType", "Threshold", "ThresholdKind", "Norm", "Tag")
                if key in entry
            ]
            print("Compare:", " ".join(parts))


sys.exit(0)
