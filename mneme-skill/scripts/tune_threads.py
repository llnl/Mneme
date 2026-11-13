#!/usr/bin/env python3
"""
Mneme tuning example (Optuna)

This example demonstrates how to run an Optuna-driven tuning session on a
previously recorded kernel execution.

Workflow:
  1) Load a recorded execution (record-db) and select a kernel (record-id).
  2) Define a SearchSpace that exposes tunable parameters.
  3) Run a baseline configuration to verify replay correctness and measure baseline time.
  4) Run N Optuna trials, evaluate each config, and report the result back to Optuna.
  5) Print the best configuration and its result.

Notes:
  - This example intentionally keeps the API usage explicit and minimal.
  - The Optuna objective is configured as direction="maximize" and the script
    reports a speedup value to Optuna, exactly as shown in the original example.
"""

import argparse
import json
import statistics
import sys
import time

import optuna

from mneme.async_executor import AsyncReplayExecutor
from mneme.mneme_types import ExperimentConfiguration
from mneme.recorded_execution import RecordedExecution
from mneme.tuning.sample_strategy import OptunaSamplingStrategy
from mneme.tuning.search_space import (
    BoolParam,
    CategoricalParam,
    IntRangeParam,
    PipelineParam,
    RealRangeParam,
    SearchSpace,
)


class EntireSpace(SearchSpace):
    """
    Example SearchSpace for tuning a recorded kernel.

    This SearchSpace:
      - Uses the recorded grid/block dims as fixed reference values where needed.
      - Exposes a set of tunable parameters (block_dim_x, specialization toggles,
        min_blocks_per_sm, launch bounds, codegen choices, and pass pipeline).
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
            "specialize": BoolParam("specialize"),
            "specialize_dims": BoolParam("specialize_dims"),
            "max_threads_fraction": RealRangeParam(
                "max_threads_fraction", low=0.0, high=1.0
            ),
            "min_blocks_per_sm": IntRangeParam("min_blocks_per_sm", low=1, high=8),
            "set_launch_bounds": BoolParam("set_launch_bounds"),
            "codegen_opt": IntRangeParam("codegen_opt", low=1, high=3, step=1),
            "codegen_method": CategoricalParam("codegen_method", ["serial"]),
            "passes": PipelineParam("passes", 120),
        }

    def dimensions(self):
        return self._search_space

    def derived(self, params) -> ExperimentConfiguration:
        """
        Convert sampled parameters into a concrete ExperimentConfiguration.

        If launch bounds are enabled, this example maps max_threads_fraction into a
        proper max_threads integer in [block_dim_x, 1024] (rounded to multiples of 64).
        """
        derived_config = {
            "block": {
                "x": params["block_dim_x"],
                "y": self.block_dim_y,
                "z": self.block_dim_z,
            },
            "grid": {"x": self.grid_dim_x, "y": self.grid_dim_y, "z": self.grid_dim_z},
            "shared_mem": self.shared_mem,
            "specialize": params["specialize"],
            "specialize_dims": params["specialize_dims"],
            "min_blocks_per_sm": params["min_blocks_per_sm"],
            "set_launch_bounds": params["set_launch_bounds"],
            "max_threads": params["max_threads_fraction"],
            "codegen_opt": params["codegen_opt"],
            "codegen_method": params["codegen_method"],
            "passes": params["passes"],
        }

        if not params["set_launch_bounds"]:
            return ExperimentConfiguration.from_dict(derived_config)

        # NOTE: max_threads value is a real number. In reality max threads can take values
        # from "block_dim_x*block_dim_y*block_dim_z" until 1024 (1024 is the maximum threads
        # within a block in GPUs both (AMD and NVIDIA)
        proper_range = (
            params["block_dim_x"]
            + round(
                (1024 - params["block_dim_x"]) * params["max_threads_fraction"] / 64
            )
            * 64
        )
        derived_config["max_threads"] = proper_range
        return ExperimentConfiguration.from_dict(derived_config)

    def constraints(self, params):
        # No constraint for now
        return True

    def baseline(self) -> ExperimentConfiguration:
        """
        Baseline configuration: replay with the recorded launch dims and shared memory.
        """
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
    """
    Parse CLI arguments.

    Required:
      --record-db   Path to the recording database (or recording JSON, depending on your setup).
      --record-id   Identifier of the recorded kernel instance within the recording.
      --num-trials  Number of Optuna trials to run.
    """
    p = argparse.ArgumentParser(
        prog="mneme-tune-example",
        description="Run Optuna tuning on a Mneme recorded kernel.",
    )
    p.add_argument(
        "--record-db",
        required=True,
        help="Path to the recorded execution database/file.",
    )
    p.add_argument(
        "--record-id",
        required=True,
        help="Kernel record identifier within the recorded execution.",
    )
    p.add_argument(
        "--num-trials",
        type=int,
        required=True,
        help="Number of Optuna trials to execute.",
    )
    return p.parse_args(argv)


def tune(executor, record_db, record_id, num_trials):
    recorded_kernel = RecordedExecution.from_json(record_db)
    kernel = recorded_kernel[record_id]
    space = EntireSpace(kernel)

    # Baseline execution (verify correctness first).
    baseline_config = space.baseline()
    print(json.dumps(baseline_config.to_dict()))

    baseline_result = executor.evaluate(baseline_config)
    if not baseline_result.verified:
        print("Cannot verify baseline execution exiting")
        return 1

    baseline_time = statistics.mean(baseline_result.exec_time)
    print(f"Average baseline time {baseline_time}, {baseline_result.exec_time}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(),
    )

    SS = OptunaSamplingStrategy(space, study, num_trials)

    start = time.time()
    for i, (config, ctrial) in enumerate(SS):
        if not config.is_valid():
            study.tell(ctrial, state=optuna.trial.TrialState.FAIL)
            continue

        val = executor.evaluate(config)
        if val.verified:
            avg_time = statistics.mean(val.exec_time)
            speedup = baseline_time / avg_time

            ctrial.set_user_attr("mneme.config", config.to_dict())
            ctrial.set_user_attr("mneme.result", val.to_dict())

            study.tell(ctrial, speedup)
            print(
                f" Experiment {i} with hash:{config.hash()} has speedup of {speedup} and total time is {avg_time}"
            )
        else:
            study.tell(ctrial, state=optuna.trial.TrialState.FAIL)
            print(i, config.hash(), f"Experiment failed with {val.error}")

    best = study.best_trial
    print(json.dumps(best.user_attrs.get("mneme.config"), indent=2))
    print(json.dumps(best.user_attrs.get("mneme.result"), indent=2))

    end = time.time()
    print("total process time is ", end - start)
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
        return tune(executor, args.record_db, args.record_id, args.num_trials)
    except KeyboardInterrupt:
        print("Interrupted by user")
        raise
    except Exception as e:
        print("Fatal error during tuning:")
        print(f"{type(e).__name__}: {e}")
        raise
    finally:
        executor.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))