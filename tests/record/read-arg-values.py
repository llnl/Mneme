#!/usr/bin/env python3
"""Print the argument value profile that MNEME_PROFILE_ARG_VALUES=1 records."""

import argparse
import json
import pathlib


def format_top(leaf, limit):
    return ",".join(f"{t['Value']}({t['Launches']})" for t in leaf["TopValues"][:limit])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=pathlib.Path, help="Mneme data directory")
    parser.add_argument("--top", type=int, default=3, help="values shown per leaf")
    parser.add_argument(
        "--no-pointers", action="store_true", help="hide pointer leaves"
    )
    args = parser.parse_args()

    profile = json.loads((args.db / "profile" / "ArgValues.json").read_text())
    kernels = sorted(profile["Kernels"].values(), key=lambda k: k["DemangledName"])
    for kernel in kernels:
        print(f"Kernel: {kernel['DemangledName']} Launches: {kernel['Launches']}")
        if "TimedLaunches" in kernel:
            print(
                f"  Timing: TimedLaunches:{kernel['TimedLaunches']} "
                f"TotalTimeNs:{kernel['TotalTimeNs']}"
            )
        for leaf in kernel["Leaves"]:
            if args.no_pointers and leaf["Type"].startswith("ptr"):
                continue
            print(
                f"  Leaf:{leaf['Label']} Type:{leaf['Type']} "
                f"IsReadByKernel:{leaf['IsReadByKernel']} "
                f"NumDistinctValues:{leaf['NumDistinctValues']} "
                f"Untracked:{leaf['UntrackedLaunches']} "
                f"TopValues:{format_top(leaf, args.top)}"
            )
        for unsupported in kernel["Unsupported"]:
            print(f"  Unsupported:{unsupported}")


if __name__ == "__main__":
    main()
