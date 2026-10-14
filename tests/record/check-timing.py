import glob
import json
import os
import sys

# Checks the invariants between the timing fields of every record database in
# a directory, which FileCheck cannot compare numerically.
errors = []
data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
for fn in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
    with open(fn, "r") as fd:
        rr_data = json.load(fd)
    name = rr_data["DemangledName"]
    recorded_time = 0
    for dhash, instance in rr_data["instances"].items():
        where = f"{name} instance {dhash}"
        if instance["TimedLaunches"] > instance["Occurrences"]:
            errors.append(f"{where}: TimedLaunches exceeds Occurrences")
        times = instance.get("TimeNs")
        if (times is None) != (instance["TimedLaunches"] == 0):
            errors.append(f"{where}: TimeNs must be present iff launches were timed")
            continue
        if times is None:
            continue
        recorded_time += times["Sum"]
        if not times["Min"] <= times["Max"] <= times["Sum"]:
            errors.append(f"{where}: expected Min <= Max <= Sum")
        recorded = instance.get("RecordedLaunchTimeNs")
        if recorded is not None and not times["Min"] <= recorded <= times["Max"]:
            errors.append(f"{where}: RecordedLaunchTimeNs outside [Min, Max]")
    if rr_data["TotalTimeNs"] < recorded_time:
        errors.append(f"{name}: TotalTimeNs below the sum of instance sums")

for error in errors:
    print("TimingError:", error)
print("TimingInvariants:", "ok" if not errors else "failed")
sys.exit(1 if errors else 0)
