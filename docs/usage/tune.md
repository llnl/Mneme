# Tuning a Recorded Kernel

This page shows how to tune a **previously recorded** kernel instance by exploring:
- the **compilation pipeline** (`passes`), and
- the kernel **launch parameter** `blockDim.x`

using Mneme’s Python API and an exhaustive sampling strategy.

> This example uses the HIP vector-add example from *Getting Started* and assumes you already have a recording database (`*.json`) produced by `mneme record`.

---

## Prerequisites

You need:

- A valid Mneme recording database file (`<static-hash>.json`)
- A recorded kernel **instance id** (the *dynamic hash*) present in the database under `instances`
- A working Mneme installation (see **[Usage → Install](install.md)**)

---

## What this tuning does

The tuning workflow is:

1. Load a recorded execution database (`--record-db`) and select an instance (`--record-id`)
2. Run a **baseline replay** using the recorded launch configuration
3. Enumerate candidate configurations:
   - `blockDim.x` in `[64..1024]` stepping by 64
   - compilation pipeline among `default<O1>, default<O2>, default<O3>, default<Os>, default<Oz>`
4. Replay each candidate, verify correctness, and measure execution time
5. Report the best configuration and speedup vs baseline

---

## Example: exhaustive tuning of `passes` and `blockDim.x` with the Python API

Save the following as `examples/tuning_vecadd_exhaustive.py`:

```python
"""
Mneme tuning example

This example demonstrates how to run a tuning session on a
previously recorded kernel execution.

Workflow:
  1) Load a recorded execution (record-db) and select a kernel (record-id).
  2) Define a SearchSpace that exposes tunable parameters.
  3) Run a baseline configuration to verify replay correctness and measure baseline time.
  4) Enumerate the search space and report the best configuration.

Notes:
  - This example keeps API usage explicit and minimal.
  - The search strategy is exhaustive over a small space (pipelines + blockDim.x).
"""

import argparse
import json
import statistics
import sys
import time

from mneme.async_executor import AsyncReplayExecutor
from mneme.mneme_types import ExperimentConfiguration
from mneme.recorded_execution import RecordedExecution
from mneme.tuning.sample_strategy import ExhaustiveSamplingStrategy
from mneme.tuning.search_space import (
    CategoricalParam,
    IntRangeParam,
    SearchSpace,
)


class EntireSpace(SearchSpace):
    """
    Example SearchSpace for tuning a recorded kernel.

    This SearchSpace:
      - Uses the recorded grid/block dims as fixed reference values where needed.
      - Exposes tunable parameters (block_dim_x and pass pipeline).
      - Produces an ExperimentConfiguration via derived(params).
    """

    def __init__(self, recorded_kernel: RecordedExecution.KernelInstance):
        self.grid_dim_x = recorded_kernel.grid_dim.x
        self.grid_dim_y = recorded_kernel.grid_dim.y
        self.grid_dim_z = recorded_kernel.grid_dim.z

        self.block_dim_x = recorded_kernel.block_dim.x
        self.block_dim_y = recorded_kernel.block_dim.y
        self.block_dim_z = recorded_kernel.block_dim.z

        self.shared_mem = recorded_kernel.shared_mem

        self._search_space = {
            "block_dim_x": IntRangeParam("block_dim_x", low=64, high=1024, step=64),
            "passes": CategoricalParam(
                "passes",
                [
                    "default<O1>",
                    "default<O2>",
                    "default<O3>",
                    "default<Os>",
                    "default<Oz>",
                ],
            ),
        }

    def dimensions(self):
        return self._search_space

    def derived(self, params) -> ExperimentConfiguration:
        derived_config = {
            "block": {
                "x": params["block_dim_x"],
                "y": self.block_dim_y,
                "z": self.block_dim_z,
            },
            "grid": {"x": self.grid_dim_x, "y": self.grid_dim_y, "z": self.grid_dim_z},
            "shared_mem": self.shared_mem,
            "passes": params["passes"],
        }
        return ExperimentConfiguration.from_dict(derived_config)

    def constraints(self, params):
        return True

    def baseline(self) -> ExperimentConfiguration:
        return ExperimentConfiguration.from_dict(
            {
                "block": {
                    "x": self.block_dim_x,
                    "y": self.block_dim_y,
                    "z": self.block_dim_z,
                },
                "grid": {
                    "x": self.grid_dim_x,
                    "y": self.grid_dim_y,
                    "z": self.grid_dim_z,
                },
                "shared_mem": self.shared_mem,
            }
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mneme-tune-example",
        description="Run exhaustive tuning on a Mneme recorded kernel.",
    )
    p.add_argument(
        "--record-db",
        required=True,
        help="Path to the Mneme recording database JSON file.",
    )
    p.add_argument(
        "--record-id",
        required=True,
        help="Kernel instance id (dynamic hash) to operate on.",
    )
    return p.parse_args(argv)


def tune(executor: AsyncReplayExecutor, record_db: str, record_id: str) -> int:
    rec = RecordedExecution.from_json(record_db)
    kernel = rec[record_id]
    space = EntireSpace(kernel)

    # Baseline execution (verify correctness first).
    baseline_config = space.baseline()
    print("Baseline configuration:")
    print(json.dumps(baseline_config.to_dict(), indent=2))

    baseline_result = executor.evaluate(baseline_config)
    if not baseline_result.verified:
        print("Baseline replay did not verify; aborting tuning.")
        return 1

    baseline_time = statistics.mean(baseline_result.exec_time)
    print(f"Baseline mean exec_time = {baseline_time} (samples = {baseline_result.exec_time})")

    sampler = ExhaustiveSamplingStrategy(space)

    start = time.time()
    results = []

    for i, config in enumerate(sampler):
        if not config.is_valid():
            continue

        val = executor.evaluate(config)
        if val.verified:
            avg_time = statistics.mean(val.exec_time)
            speedup = baseline_time / avg_time
            results.append((config, val))
            print(
                f"[{i}] passes={config.passes} blockDim.x={config.block.x} "
                f"mean={avg_time} speedup={speedup}"
            )
        else:
            print(
                f"[{i}] passes={config.passes} blockDim.x={config.block.x} "
                f"FAILED error={val.error}"
            )

    end = time.time()
    print(f"Total tuning time: {end - start:.2f}s")

    if not results:
        print("No successful configurations verified.")
        return 2

    best = min(results, key=lambda x: statistics.mean(x[1].exec_time))
    best_time = statistics.mean(best[1].exec_time)
    best_speedup = baseline_time / best_time

    print(
        f"BEST: passes={best[0].passes} blockDim.x={best[0].block.x} "
        f"mean={best_time} speedup={best_speedup}"
    )
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    executor = AsyncReplayExecutor(
        record_db=args.record_db,
        record_id=args.record_id,
        iterations=5,
        results_db_dir="./results",
        num_workers=1,
    )
    try:
        return tune(executor, args.record_db, args.record_id)
    finally:
        executor.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

## How to run

Pick the recording database JSON (<static-hash>.json) and the instance id from its instances map (the dynamic hash)

Example:

```bash
python tuning_vecadd_exhaustive.py \
  --record-db record-example-dir/15941914485064662553.json \
  --record-id 16313427880266313990
```

## Interpreting the output

The output will be similar to:
```bash
Baseline configuration:
{
  "grid": {
    "x": 40000,
    "y": 1,
    "z": 1
  },
  "block": {
    "x": 256,
    "y": 1,
    "z": 1
  },
  "shared_mem": 0,
  "specialize": false,
  "set_launch_bounds": false,
  "max_threads": 1024,
  "min_blocks_per_sm": 1,
  "specialize_dims": false,
  "passes": "default<O3>",
  "codegen_opt": 3,
  "codegen_method": "serial",
  "prune": true,
  "internalize": true
}
Baseline mean exec_time = 82468.85714285714 (samples = [86280, 84600, 88561, 81040, 81360, 80721, 74720])
[0] passes=default<O1> blockDim.x=64 mean=78359 speedup=1.0524490759562672
[1] passes=default<O2> blockDim.x=64 mean=76611.42857142857 speedup=1.0764563287834714
[2] passes=default<O3> blockDim.x=64 mean=77006.28571428571 speedup=1.0709366953347037
[3] passes=default<Os> blockDim.x=64 mean=76863.14285714286 speedup=1.0729311094672906
[4] passes=default<Oz> blockDim.x=64 mean=74229.28571428571 speedup=1.1110016262353135
[5] passes=default<O1> blockDim.x=128 mean=73486.14285714286 speedup=1.1222368454305283
[6] passes=default<O2> blockDim.x=128 mean=76394.57142857143 speedup=1.0795120072106843
[7] passes=default<O3> blockDim.x=128 mean=74806 speedup=1.1024363973860003
[8] passes=default<Os> blockDim.x=128 mean=77109.14285714286 speedup=1.0695081554160708
...
[78] passes=default<Os> blockDim.x=1024 mean=123389.14285714286 speedup=0.6683639681194455
[79] passes=default<Oz> blockDim.x=1024 mean=122823.71428571429 speedup=0.6714408339013174
Total tuning time: 7.09s
BEST: passes=default<O1> blockDim.x=128 mean=73486.14285714286 speedup=1.1222368454305283
```

From this you should be able to identify:
- The baseline configuration and its exec_time samples
- One line per candidate configuration
- A final BEST line with the best passes, best `blockDim.x`, mean time, and
  speedup vs baseline

!!! note
    This example uses `ExhaustiveSamplingStrategy`, which is ideal for small search spaces. The run above evaluates 80 configurations in 7.09 seconds, with each configuration performing 7 benchmark executions, for a throughput of approximately 78 replays/second. For larger spaces, switch to a sampling-based strategy such as Optuna and persist results to a dedicated tuning database.


## Example: exhaustive tuning of `passes` and `blockDim.x` with the CLI API

The `mneme tune` command can run the same search space through the CLI by loading
the `EntireSpace` class from the Python example above. This keeps the tuning
problem identical: all combinations of `passes` and `blockDim.x` are enumerated,
the recorded launch is used as the baseline, and candidates are ranked by mean
execution time.

```bash
mneme tune \
  --record-database record-example-dir/15941914485064662553.json \
  --record-id 16313427880266313990 \
  --space-module examples/tuning_vecadd_exhaustive.py:EntireSpace \
  --sampler exhaustive \
  --iterations 5 \
  --warmup 2 \
  --workers 1 \
  --metric mean \
  --objective time \
  --results-dir mneme-tune-results/vecadd-exhaustive
```

The same run can be captured in a configuration file, for example
`examples/hip_vec_add/config.yml`:

```yaml
record_database: record-example-dir/15941914485064662553.json
record_id: "16313427880266313990"
space_module: examples/tuning_vecadd_exhaustive.py:EntireSpace

sampler: exhaustive
iterations: 5
warmup: 2
workers: 1
metric: mean
objective: time

results_dir: mneme-tune-results/vecadd-exhaustive
```

Run it with:

```bash
mneme tune --config examples/hip_vec_add/config.yml
```

CLI arguments override values from the config file, so a common pattern is to
keep stable search settings in YAML and pass record-specific values on the
command line.

You can also use the built-in search space instead of providing a custom
`SearchSpace` class. This does not reproduce the exact `blockDim.x = 64..1024`
range from the Python example; the built-in space derives valid `block_shape`
candidates from the recorded launch and the selected launch axes. It is the
recommended path when the generated launch space is sufficient.

```bash
mneme tune \
  --record-database record-example-dir/15941914485064662553.json \
  --record-id 16313427880266313990 \
  --space-preset launch \
  --launch-dim x \
  --launch-safety aggressive \
  --passes 'default<O1>' 'default<O2>' 'default<O3>' 'default<Os>' 'default<Oz>' \
  --fixed-codegen-opt 3 \
  --sampler exhaustive \
  --iterations 5 \
  --warmup 2 \
  --workers 1 \
  --metric mean \
  --objective time \
  --results-dir mneme-tune-results/vecadd-builtin
```

The equivalent config file is:

```yaml
record_database: record-example-dir/15941914485064662553.json
record_id: "16313427880266313990"

space_preset: launch
launch_dim: x
launch_safety: aggressive
passes:
  - default<O1>
  - default<O2>
  - default<O3>
  - default<Os>
  - default<Oz>
fixed_codegen_opt: 3

sampler: exhaustive
iterations: 5
warmup: 2
workers: 1
metric: mean
objective: time

results_dir: mneme-tune-results/vecadd-builtin
```

## Tuning CLI Specification

`mneme tune` tunes one recorded kernel instance. The required inputs are:

- `--record-database`: path to the Mneme recording database JSON file
- `--record-id`: dynamic kernel instance id within that database

### Search spaces

Use `--space-module MODULE_OR_PATH:CLASS` to load a custom `SearchSpace` class.
The class is constructed with the recorded kernel instance. Extra constructor
arguments may be passed with repeated `--space-arg KEY=VALUE` options; values are
decoded as JSON when possible. The instance's `execution` attribute is the owning
`RecordedExecution`, so `recorded_kernel.execution.kernel_source()` returns the
kernel's source text and location when the record carries source information.

Without `--space-module`, the built-in search space is used. Select its scope
with `--space-preset`:

- `quick`: small exploratory space
- `standard`: default balanced space
- `launch`: focus on launch/block-shape parameters
- `compiler`: focus on compiler pipeline and codegen choices
- `full`: broader launch, compiler, specialization, and launch-bounds space

Built-in launch controls include `--launch-dim`, `--launch-safety`, and
`--adaptive-invalid-ban` / `--no-adaptive-invalid-ban`. Compiler controls include
`--passes`, `--fixed-passes`, `--pipeline-file`, `--codegen-opt-range`, and
`--fixed-codegen-opt`. Boolean configuration dimensions such as specialization
and launch bounds use `fixed`, `on`, `off`, or `on-off` modes.

### Sampling and evaluation

`--sampler` selects how candidates are generated:

- `random`: random sampling; requires `--trials`
- `tpe`: Optuna TPE sampling; requires `--trials`
- `grid` / `exhaustive`: enumerate the finite search space

Use `--seed` for reproducible random or TPE sampling, and `--timeout` to stop
submitting new trials after a fixed number of seconds. Each candidate is replayed
with `--warmup` warmup iterations and `--iterations` measured iterations.
`--workers` controls concurrent replay workers. `--compare-policy FILE`
overrides the recorded comparison metadata used to verify every candidate
(see [Verification](verification.md#overriding-tolerances-at-replay-time)).

The ranking metric is selected with `--metric` (`mean`, `median`, `min`, or
`max`). `--objective time` minimizes the selected time metric, while
`--objective speedup` maximizes speedup relative to the baseline.

The tuner will run the baseline version of a kernel first, so it 
can compute speedups later. By default it will early exit if the baseline
execution fails execution or verification. Use `--require-verified-baseline` 
(default) and `--no-require-verified-baseline` to toggle this behavior.

### Outputs and resuming

If `--results-dir` is not provided, Mneme creates a timestamped directory under
`mneme-tune-results/`. A tuning run writes:

- `config.json`: resolved configuration
- `search_space.json`: resolved search-space description
- `baseline.json`: baseline replay result
- `trials.jsonl`: one record per evaluated candidate
- `best.json`: best verified configuration
- `summary.json`: aggregate trial counts and best speedup
- `logs/tune.log`: tuning log file

Use `--resume` with an existing results directory to skip completed
configurations. Use `--rerun-baseline` to recompute the baseline while resuming.

Additional run modes are available for inspection and validation:

- `--print-space`: print the resolved search space and exit
- `--dry-run`: print the baseline configuration and search space without replay
- `--baseline-only`: run only the baseline replay
- `--dump-config PATH`: write the resolved configuration and exit
- `--quiet`: suppress progress output

### Optuna and Proteus integration

For Optuna-backed runs, use `--sampler tpe` with optional `--study-name`,
`--optuna-storage`, and `--pruner`. Persistent Optuna storage is useful when
resuming or inspecting studies outside Mneme.

By default, successful tuning writes Proteus tuned-kernel metadata to
`proteus_tuned_kernels.json` in the results directory. Override the path with
`--proteus-output`, or disable export with `--no-proteus-output`.
