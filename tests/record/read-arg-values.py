#!/usr/bin/env python3
"""Print the argument value profile that MNEME_PROFILE_ARG_VALUES=1 records."""

import argparse
import collections
import json
import math
import pathlib
import struct
import sys

OVERFLOW = -1


def format_float(text, leaf):
    raw = int(text, 16).to_bytes(leaf["SizeBytes"], "little")
    if leaf["Type"] == "double":
        return repr(struct.unpack("<d", raw)[0])
    if leaf["Type"] == "half":
        return repr(struct.unpack("<e", raw)[0])
    if leaf["Type"] == "float":
        value = struct.unpack("<f", raw)[0]
        if not math.isfinite(value):
            return repr(value)
        # The shortest decimal that rounds back to the same float, so 0.1f
        # prints as 0.1 rather than as its double expansion.
        for digits in range(1, 10):
            short = float(f"{value:.{digits}g}")
            if struct.pack("<f", short) == raw:
                return repr(short)
        return repr(value)
    return text


def format_value(value, leaf):
    if isinstance(value, str):
        return format_float(value, leaf)
    return str(value)


def format_top(leaf, limit):
    by_launches = sorted(leaf["Values"], key=lambda pair: -pair[1])
    return ",".join(
        f"{format_value(value, leaf)}({launches})"
        for value, launches in by_launches[:limit]
    )


def format_signature(indices, leaves):
    return ",".join(
        "*" if index == OVERFLOW else format_value(leaf["Values"][index][0], leaf)
        for index, leaf in zip(indices, leaves)
    )


def check(kernel):
    name = kernel["DemangledName"]
    leaves = kernel["Leaves"]
    launches = kernel["Launches"]
    untracked = kernel["UntrackedLaunches"]
    signatures = kernel["Signatures"]

    for row, (indices, _) in enumerate(signatures):
        if len(indices) != len(leaves):
            sys.exit(
                f"{name}: signature {row} has {len(indices)} indices "
                f"for {len(leaves)} leaves"
            )
        for index, leaf in zip(indices, leaves):
            if not OVERFLOW <= index < len(leaf["Values"]):
                sys.exit(
                    f"{name}: signature {row} has index {index} for leaf "
                    f"{leaf['Label']} with {len(leaf['Values'])} values"
                )

    signature_launches = sum(count for _, count in signatures)
    if signature_launches + untracked != launches:
        sys.exit(
            f"{name}: signature launches {signature_launches} + untracked "
            f"{untracked} != launches {launches}"
        )

    for position, leaf in enumerate(leaves):
        label = leaf["Label"]
        stored = collections.Counter(
            {index: count for index, (_, count) in enumerate(leaf["Values"])}
        )
        stored[OVERFLOW] = leaf["OverflowLaunches"]
        leaf_launches = sum(stored.values())
        if leaf_launches != launches:
            sys.exit(
                f"{name}: leaf {label} counts {leaf_launches} launches, "
                f"kernel has {launches}"
            )

        derived = collections.Counter()
        for indices, count in signatures:
            derived[indices[position]] += count
        for index in set(stored) | set(derived):
            if untracked == 0:
                failed = derived[index] != stored[index]
            else:
                failed = derived[index] > stored[index]
            if failed:
                sys.exit(
                    f"{name}: leaf {label} index {index} has {derived[index]} "
                    f"launches in signatures and {stored[index]} stored"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=pathlib.Path, help="Mneme data directory")
    parser.add_argument("--top", type=int, default=3, help="values shown per leaf")
    parser.add_argument(
        "--signatures", action="store_true", help="print every signature"
    )
    args = parser.parse_args()

    profile = json.loads((args.db / "profile" / "ArgValues.json").read_text())
    kernels = sorted(profile["Kernels"].values(), key=lambda k: k["DemangledName"])
    for kernel in kernels:
        check(kernel)
        print(
            f"Kernel: {kernel['DemangledName']} Launches: {kernel['Launches']} "
            f"Signatures: {len(kernel['Signatures'])} "
            f"Untracked: {kernel['UntrackedLaunches']}"
        )
        if "TimedLaunches" in kernel:
            print(
                f"  Timing: TimedLaunches:{kernel['TimedLaunches']} "
                f"TotalTimeNs:{kernel['TotalTimeNs']}"
            )
        for leaf in kernel["Leaves"]:
            print(
                f"  Leaf:{leaf['Label']} Type:{leaf['Type']} "
                f"NumDistinctValues:{len(leaf['Values'])} "
                f"Overflow:{leaf['OverflowLaunches']} "
                f"TopValues:{format_top(leaf, args.top)}"
            )
        if args.signatures:
            for indices, launches in kernel["Signatures"]:
                print(
                    f"  Signature:{format_signature(indices, kernel['Leaves'])} "
                    f"Launches:{launches}"
                )


if __name__ == "__main__":
    main()
