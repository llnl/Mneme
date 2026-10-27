""" The main tuning class; defines a tuning Session that manages the search and replay workers.

"""

import hashlib
import json
import logging
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import optuna

from mneme.async_executor import AsyncReplayExecutor
from mneme.convert import export_proteus_tuned_kernel
from mneme.futures import EvalFuture
from mneme.mneme_types import ExperimentConfiguration, ExperimentResult, dim3
from mneme.recorded_execution import RecordedExecution
from mneme.replay_executor import noop_verifies_warning
from mneme.tuning.builtin_spaces import (
    BuiltinTuneSearchSpace,
    describe_search_space,
    iter_finite_params,
    load_custom_space,
    parse_key_value_args,
)
from mneme.tuning.launch_candidates import launch_layout_key, normalize_launch_candidates
from mneme.tuning.result_store import ResultStore
from mneme.tuning.search_space import (
    BaseParam,
    BoolParam,
    CategoricalParam,
    FixedParam,
    IntRangeParam,
    RealRangeParam,
    sample_optuna_param,
)


# Exit codes:
EXIT_BASELINE_FAILED = 1
EXIT_INVALID_CONFIGURATION = 3
EXIT_RECORD_LOAD_FAILED = 4
EXIT_INTERNAL_ERROR = 5


@dataclass
class TuneOptions:
    # TODO -- consider a 'profile' option that expands into space_preset, sampler, and trials defaults.

    # Record selection: identifies the kernel to tune in the Mneme database.
    record_database: str
    record_id: str

    # Search control.
    space_preset: str = "standard"
    sampler: str = "random"
    trials: Optional[int] = None
    timeout: Optional[float] = None

    # Replay execution.
    iterations: int = 5
    warmup: int = 2
    short_circuit: bool = False
    workers: int = 1

    # Output location, scoring, and reproducibility.
    results_dir: Optional[str] = None
    metric: str = "mean"
    objective: str = "time"
    seed: Optional[int] = None

    # Run mode.
    resume: bool = False
    rerun_baseline: bool = False
    fail_fast: bool = False
    print_space: bool = False
    dry_run: bool = False
    baseline_only: bool = False
    proteus_output: Optional[str] = None
    proteus_enabled: bool = True
    quiet: bool = False

    # Optuna integration.
    study_name: Optional[str] = None
    optuna_storage: Optional[str] = None
    pruner: str = "none"

    # Launch-space controls.
    launch_dim: str = "auto"
    launch_safety: Optional[str] = None
    adaptive_invalid_ban: bool = True
    launch_candidates: list[dict] = field(default_factory=list)
    require_verified_baseline: bool = True

    # Compiler-space controls.
    passes: Optional[List[str]] = None
    fixed_passes: Optional[str] = None
    pipeline_file: Optional[str] = None
    codegen_opt_range: Optional[str] = None
    fixed_codegen_opt: Optional[int] = None
    codegen_method: Optional[str] = None

    # Configuration-space controls.
    specialize_space: Optional[str] = None
    specialize_dims_space: Optional[str] = None
    launch_bounds_space: Optional[str] = None
    min_blocks_per_sm_range: Optional[str] = None
    fixed_min_blocks_per_sm: Optional[int] = None
    max_threads_range: Optional[str] = None
    max_threads_policy: str = "block-threads"

    # Custom search space.
    space_module: Optional[str] = None
    space_arg: List[str] = field(default_factory=list)

    # Configuration files.
    config: Optional[str] = None
    dump_config: Optional[str] = None

    def resolved_results_dir(self) -> str:
        """ Determine the results directory to use. Uses results_dir if provided, otherwise 
            generates one based on the current time.
        """
        if self.results_dir:
            return self.results_dir
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_record = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(self.record_id))
        return str(Path("mneme-tune-results") / f"{safe_record}-{timestamp}")

    def to_config_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["results_dir"] = self.resolved_results_dir()
        if self.proteus_enabled and self.proteus_output is None:
            data["proteus_output"] = str(Path(data["results_dir"]) / "proteus_tuned_kernels.json")
        return data

    def _validate(self) -> None:
        """ an early, quick check to validate some settings before any tuning is started """

        self.launch_candidates = normalize_launch_candidates(self.launch_candidates)
        if self.launch_candidates:
            if self.space_module:
                raise ValueError("launch_candidates are not supported with a custom search space")
            if self.sampler not in {"random", "tpe"}:
                raise ValueError("launch_candidates require the random or tpe sampler")

        if not self.require_verified_baseline and self.objective != "time":
            raise ValueError("require_verified_baseline=False requires objective='time'")

        # check trials is positive and compatible with sampler choice
        if self.trials is not None and self.trials <= 0:
            raise ValueError("--trials must be a positive integer")
        
        if self.sampler in {"grid", "exhaustive"} and self.trials is not None:
            raise ValueError("--trials cannot be used with grid or exhaustive sampler since the search space size is determined by the grid")

        if self.sampler in {"random", "tpe"} and self.trials is None:
            raise ValueError("--trials is required for random and tpe samplers")

        if self.workers <= 0:
            raise ValueError("--workers must be a positive integer")

        if self.iterations <= 0:
            raise ValueError("--iterations must be a positive integer")
        
        if self.warmup < 0:
            raise ValueError("--warmup must be a non-negative integer")

    def __post_init__(self):
        self._validate()


@dataclass
class Candidate:
    trial: int
    params: Dict[str, Any]
    config: ExperimentConfiguration
    optuna_trial: Any = None
    optuna_study: Any = None
    error: Optional[str] = None


@dataclass
class CompletedCandidate:
    # TODO -- perhaps fold this into Candidate and add a completed() -> bool function

    trial: int
    params: Dict[str, Any]
    config: ExperimentConfiguration
    result: ExperimentResult
    status: str
    metric: Optional[float]
    speedup: Optional[float]


def metric_value(result: ExperimentResult, metric: str, iterations: int) -> Optional[float]:
    """ Extract the desired metric value from the experiment result.
        Supports metrics: mean, median, min, max of the recorded execution times.
    """

    values = list(result.exec_time or [])
    if not values:
        return None
    
    if iterations > 0 and len(values) >= iterations:
        values = values[-iterations:]
    
    if metric == "mean":
        return float(statistics.mean(values))
    if metric == "median":
        return float(statistics.median(values))
    if metric == "min":
        return float(min(values))
    if metric == "max":
        return float(max(values))
    raise ValueError(f"Unknown metric {metric!r}")


def stored_result_and_metric(data: Mapping[str, Any]) -> Tuple[ExperimentResult, Optional[float]]:
    """Load a stored result while keeping tuning-only metadata separate."""

    metric = data.get("metric")
    result = ExperimentResult.from_dict({k: v for k, v in data.items() if k != "metric"})
    return result, metric


def classify_result(result: ExperimentResult) -> str:
    """ Classify the result of an experiment into categories for easier analysis and reporting.
        Categories include:
        - verified: The replay executed successfully and produced correct results.
        - invalid_launch: The replay failed due to an invalid launch configuration (eg too many threads).
        - invalid_config: The replay failed due to an invalid configuration that violates constraints.
        - compile_error: The replay failed to compile or JIT due to a codegen or compilation error.
        - runtime_error: The replay compiled but failed during execution.
        - failed_verification: The replay executed but did not produce correct results.
    """
    error = (result.error or "").lower()

    if result.verified:
        return "verified"
    
    if "invalid" in error and "launch" in error:
        return "invalid_launch"
    
    if "invalid configuration" in error:
        return "invalid_config"
    
    if result.failed:
        if any(token in error for token in ("compile", "codegen", "llvm", "jit", "link")):
            return "compile_error"
        return "runtime_error"
    
    if result.executed:
        return "failed_verification"
    
    if result.error:
        return "runtime_error"
    
    return "failed_verification"


def _objective_value(
    status: str,
    candidate_metric: Optional[float],
    speedup: Optional[float],
    objective: str,
) -> float:
    if status != "verified" or candidate_metric is None:
        return math.inf if objective == "time" else 0.0
    if objective == "speedup":
        return float(speedup or 0.0)
    return float(candidate_metric)


def _sample_random_param(param: BaseParam, rng: random.Random) -> Any:
    """ helper for handling mneme space random params """
    if isinstance(param, FixedParam):
        return param.value
    if isinstance(param, BoolParam):
        return rng.choice(list(param.choices))
    if isinstance(param, CategoricalParam):
        return rng.choice(list(param.choices))
    if isinstance(param, IntRangeParam):
        values = list(range(param.low, param.high + 1, param.step))
        return rng.choice(values)
    if isinstance(param, RealRangeParam):
        return rng.uniform(param.low, param.high)
    raise TypeError(f"Unsupported random parameter type: {type(param).__name__}")


def _constraints_ok(space: Any, params: Dict[str, Any], config: ExperimentConfiguration) -> bool:
    try:
        if not bool(space.constraints(params)):
            return False
    except Exception:
        # keep this as a fallback, since Mneme spaces in docs use this format
        # TODO -- we could remove this
        if not bool(space.constraints(config)):
            return False
    return bool(config.is_valid())


def _load_config_file(path: str) -> Dict[str, Any]:
    """ read config from a JSON or YAML file """
    with open(path, "r") as fd:
        text = fd.read()

    if path.endswith(".json"):
        return json.loads(text)
    
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML config files require PyYAML to be installed") from exc
    return yaml.safe_load(text) or {}


def merge_config_and_args(args: Any) -> Dict[str, Any]:
    """ we want to allow reading from _both_ a config file and CLI args, with CLI args taking precedence over config file options.
        This function merges the config file into the CLI options.
    """

    cli = vars(args).copy()

    config_path = cli.get("config")
    merged: Dict[str, Any] = {}
    if config_path:
        merged.update(_load_config_file(config_path))
        search = merged.pop("search", None)
        if isinstance(search, Mapping):
            merged.update(search)

    for key, value in cli.items():
        if key in {"func", "parser", "command"}:
            continue
        if value is not None:
            merged[key] = value

    return merged


class TuningSession:
    """ The main wrapping class for tuning behavior. This manages the tuning experiments and 
        is controlled by TuneOptions.
    """

    def __init__(
        self,
        options: TuneOptions,
        *,
        executor: Optional[AsyncReplayExecutor] = None,
        ir: Optional[Union[str, Path]] = None,
    ):
        """Use ``ir`` for this session, optionally on a caller-owned executor.

        A borrowed executor must target the same record, instance, worker count,
        iterations, and warmup as ``options``. The caller remains responsible for
        shutting it down.
        """
        self.options = options
        self._borrowed_executor = executor
        self._ir = ir
        self.options.results_dir = self.options.resolved_results_dir()
        if self.options.proteus_enabled and self.options.proteus_output is None:
            self.options.proteus_output = str(Path(self.options.results_dir) / "proteus_tuned_kernels.json")
        
        self.store = ResultStore(self.options.results_dir)
        self.rng = random.Random(self.options.seed)
        self._completed_hashes = set()
        self._banned_layouts = set()

    def _print(self, message: str) -> None:
        if not self.options.quiet:
            print(message)

    def _load_record(self) -> Tuple[RecordedExecution, RecordedExecution.KernelInstance]:
        try:
            recorded = RecordedExecution.from_json(self.options.record_database)
            return recorded, recorded[self.options.record_id]
        except KeyError as exc:
            raise RuntimeError(f"Record id {self.options.record_id!r} was not found") from exc

    def _build_space(self, kernel: RecordedExecution.KernelInstance) -> Tuple[Any, Dict[str, Any]]:
        """ helper -- if user passes a custom search space module, then use that to define search space,
            otherwise build one from builtin_spaces
        """

        if self.options.space_module:
            kwargs = parse_key_value_args(self.options.space_arg)
            space = load_custom_space(self.options.space_module, kernel, kwargs)
            return space, describe_search_space(space, custom=True)

        space = BuiltinTuneSearchSpace(
            kernel,
            preset=self.options.space_preset,
            launch_dim=self.options.launch_dim,
            launch_safety=self.options.launch_safety,
            passes=self.options.passes,
            fixed_passes=self.options.fixed_passes,
            pipeline_file=self.options.pipeline_file,
            codegen_opt_range=self.options.codegen_opt_range,
            fixed_codegen_opt=self.options.fixed_codegen_opt,
            codegen_method=self.options.codegen_method,
            specialize_space=self.options.specialize_space,
            specialize_dims_space=self.options.specialize_dims_space,
            launch_bounds_space=self.options.launch_bounds_space,
            min_blocks_per_sm_range=self.options.min_blocks_per_sm_range,
            fixed_min_blocks_per_sm=self.options.fixed_min_blocks_per_sm,
            max_threads_range=self.options.max_threads_range,
            max_threads_policy=self.options.max_threads_policy,
            launch_candidates=self.options.launch_candidates,
        )
        return space, describe_search_space(space, preset=self.options.space_preset)

    def _baseline_config(self, space: Any, kernel: RecordedExecution.KernelInstance) -> ExperimentConfiguration:
        if hasattr(space, "baseline"):
            return space.baseline()
        
        # TODO -- space defines no baseline run config; should this be an error?
        return ExperimentConfiguration(
            grid=kernel.grid_dim,
            block=kernel.block_dim,
            shared_mem=kernel.shared_mem,
            passes="default<O3>",
            max_threads=0,
            min_blocks_per_sm=0,
        )

    def _validate_resume(self, resolved_config: Dict[str, Any]) -> None:
        """ check if loading from a previous tuning session is valid """

        if not self.options.resume:
            return
        existing = self.store.load_config()
        if existing is None:
            raise RuntimeError("--resume requires an existing results directory with config.json")
        
        # fix here -- we cant change some things from a previous tuning session
        # be safe and error if they're different
        immutable = [
            "ir_hash",
            "record_database",
            "record_id",
            "space_preset",
            "space_module",
            "metric",
            "launch_dim",
            "launch_safety",
            "passes",
            "fixed_passes",
            "pipeline_file",
            "codegen_opt_range",
            "fixed_codegen_opt",
            "specialize_space",
            "specialize_dims_space",
            "launch_bounds_space",
            "launch_candidates",
            "require_verified_baseline",
        ]
        existing.setdefault("launch_candidates", [])
        existing.setdefault("require_verified_baseline", True)
        mismatched = [
            key for key in immutable if existing.get(key) != resolved_config.get(key)
        ]
        if mismatched:
            raise RuntimeError(f"--resume option mismatch for: {', '.join(mismatched)}")

        # finally save any check any completed trial hashes to avoid re-running them
        self._completed_hashes = self.store.completed_hashes()

    def _resolve_candidate(self, space: Any, trial: int, params: Dict[str, Any]) -> Optional[Candidate]:
        config = ExperimentConfiguration()
        try:
            config = space.derived(params)
            if self._layout_key(config) in self._banned_layouts or not _constraints_ok(space, params, config):
                return None
        except Exception as exc:
            if self.options.fail_fast:
                raise
            return Candidate(trial, params, config, error=str(exc))
            
        return Candidate(trial, params, config)

    def _iter_random_candidates(self, space: Any) -> Iterator[Candidate]:
        """ helper generator for looping over search space """
        if self.options.trials is None:
            raise RuntimeError("--trials is required for random sampling")
        
        dims = space.dimensions()
        produced = 0
        attempts = 0
        while produced < self.options.trials:
            attempts += 1
            if attempts > max(1000, self.options.trials * 100):
                # fail safe to prevent infinite loops
                # TODO -- perhaps there's a safer way to check this or we should make the scaling
                # parameter on trials cap configurable
                raise RuntimeError("Could not sample enough valid random candidates")
            
            params = {
                name: _sample_random_param(param, self.rng)
                for name, param in dims.items()
            }
            candidate = self._resolve_candidate(space, produced, params)
            if candidate is None:
                continue

            yield candidate
            produced += 1

    def _iter_grid_candidates(self, space: Any) -> Iterator[Candidate]:
        """ helper for iterating over all grid or exhaustive candidates """
        limit = self.options.trials
        produced = 0
        for params in iter_finite_params(space):
            candidate = self._resolve_candidate(space, produced, params)
            if candidate is None:
                continue

            yield candidate
            produced += 1

            if limit is not None and produced >= limit:
                # TODO consider warning here; we're ending early and this might not be expected
                # behavior if user passed exhaustive sampling
                break

    def _make_study(self) -> optuna.Study:
        sampler: Optional[optuna.samplers.BaseSampler]
        if self.options.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(seed=self.options.seed)
        else:
            # I'm ignoring the GridSampler, since we define the random sampler search space
            # based on the grid. In the future GridSampler can be added if we want to be more
            # 1-1 with optuna's samplers
            sampler = optuna.samplers.RandomSampler(seed=self.options.seed)

        pruner = optuna.pruners.NopPruner()
        if self.options.pruner == "median":
            pruner = optuna.pruners.MedianPruner()

        return optuna.create_study(
            direction="maximize" if self.options.objective == "speedup" else "minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=self.options.study_name,
            storage=self.options.optuna_storage,
            load_if_exists=bool(self.options.resume or self.options.optuna_storage),
        )

    def _iter_optuna_candidates(self, space: Any) -> Iterator[Candidate]:
        if self.options.trials is None:
            raise RuntimeError("--trials is required for Optuna-backed sampling")
        
        study = self._make_study()
        dims = space.dimensions()
        produced = 0
        attempts = 0
        while produced < self.options.trials:
            attempts += 1
            if attempts > max(1000, self.options.trials * 100):
                # TODO -- perhaps there's a safer way to check this or we should make the scaling
                # parameter on trials cap configurable
                raise RuntimeError("Could not sample enough valid candidates with Optuna")

            trial = study.ask()
            params = {
                name: sample_optuna_param(trial, param)
                for name, param in dims.items()
            }

            candidate = self._resolve_candidate(space, produced, params)
            if candidate is None:
                study.tell(trial, _objective_value("invalid_config", None, None, self.options.objective))
                continue

            candidate.optuna_trial = trial
            candidate.optuna_study = study
            yield candidate
            produced += 1

    def _candidate_iter(self, space: Any) -> Iterator[Candidate]:
        if self.options.sampler == "random":
            return self._iter_random_candidates(space)
        if self.options.sampler in {"grid", "exhaustive"}:
            return self._iter_grid_candidates(space)
        if self.options.sampler == "tpe":
            return self._iter_optuna_candidates(space)
        raise RuntimeError(f"Unknown sampler {self.options.sampler!r}")

    @staticmethod
    def _layout_key(config: ExperimentConfiguration) -> str:
        return launch_layout_key(
            [config.block.x, config.block.y, config.block.z],
            [config.grid.x, config.grid.y, config.grid.z],
        )

    def _initial_candidates(self, space: Any, baseline: ExperimentConfiguration) -> Iterator[Candidate]:
        for trial, layout in enumerate(space.launch_candidates):
            config = replace(baseline, block=dim3(*layout["block"]), grid=dim3(*layout["grid"]))
            params = config.to_dict()
            params.pop("block")
            params.pop("grid")
            params["launch_layout"] = self._layout_key(config)
            yield Candidate(trial, params, config)

    def _tell_optuna(self, candidate: Candidate, status: str, metric: Optional[float], speedup: Optional[float]) -> None:
        if candidate.optuna_trial is None:
            return
        value = _objective_value(status, metric, speedup, self.options.objective)
        candidate.optuna_study.tell(candidate.optuna_trial, value)

    def _trial_record(
        self,
        candidate: Candidate,
        result: ExperimentResult,
        status: str,
        candidate_metric: Optional[float],
        baseline_metric: Optional[float],
    ) -> Dict[str, Any]:
        speedup = (
            baseline_metric / candidate_metric
            if baseline_metric is not None and candidate_metric and candidate_metric > 0
            else None
        )
        return {
            "trial": candidate.trial,
            "status": status,
            "config_hash": candidate.config.hash(),
            "params": candidate.params,
            "config": candidate.config.to_dict(),
            "result": dict(result.to_dict(), metric=candidate_metric),
            "baseline_metric": baseline_metric,
            "speedup": speedup,
        }

    def _record_invalid_candidate(
        self,
        candidate: Candidate,
        baseline_metric: Optional[float],
        error: str,
    ) -> CompletedCandidate:
        result = ExperimentResult(error=error, failed=True, executed=False)
        status = "invalid_config"
        record = self._trial_record(candidate, result, status, None, baseline_metric)
        self.store.append_trial(record)
        self._tell_optuna(candidate, status, None, None)
        return CompletedCandidate(candidate.trial, candidate.params, candidate.config, result, status, None, None)

    def _load_resume_best(
        self,
        baseline_config: ExperimentConfiguration,
        baseline_result: ExperimentResult,
        baseline_metric: Optional[float],
    ) -> Tuple[Optional[CompletedCandidate], Dict[str, int], int]:
        """ helper for loading the best candidate from previously saved session
            Also counts the number of completed trials and their status for reporting purposes.
        """

        best = CompletedCandidate(
            -1,
            {"baseline": True},
            baseline_config,
            baseline_result,
            "verified",
            baseline_metric,
            1.0,
        ) if baseline_result.verified and baseline_metric is not None else None
        counts: Dict[str, int] = {}
        completed = 0

        for trial in self.store.load_trials():
            completed += 1

            status = trial.get("status", "internal_error")
            counts[status] = counts.get(status, 0) + 1
            if status != "verified":
                continue

            result, metric = stored_result_and_metric(trial.get("result", {}))
            config = ExperimentConfiguration.from_dict(trial["config"])
            speedup = baseline_metric / metric if baseline_metric is not None and metric and metric > 0 else None

            current = CompletedCandidate(
                int(trial["trial"]),
                dict(trial.get("params", {})),
                config,
                result,
                status,
                metric,
                speedup,
            )
            if metric is not None and (best is None or best.metric is None or metric < best.metric):
                best = current
        
        return best, counts, completed

    def _write_final_outputs(
        self,
        recorded: RecordedExecution,
        kernel: RecordedExecution.KernelInstance,
        baseline_metric: Optional[float],
        best: Optional[CompletedCandidate],
        counts: Dict[str, int],
        completed_trials: int,
        num_requested: Optional[int],
        noop_verifies: bool,
    ) -> Dict[str, Any]:
        """ helper to write out the final results """

        if best is None:
            summary = {
                "status": "failed",
                "num_trials_requested": num_requested,
                "num_trials_completed": completed_trials,
                "baseline_metric": baseline_metric,
                "best_metric": None,
                "best_speedup": None,
                "best_trial": None,
                "error": "No configuration verified",
                "noop_verifies": noop_verifies,
                **{f"num_{status}": counts.get(status, 0) for status in (
                    "verified", "failed_verification", "invalid_launch", "compile_error",
                    "runtime_error", "invalid_config", "internal_error",
                )},
            }
            self.store.write_summary(summary)
            return summary

        speedup = None
        if baseline_metric is not None:
            speedup = baseline_metric / best.metric if best.metric and best.metric > 0 else 1.0
        best_doc = {
            "status": "success",
            "trial": best.trial,
            "speedup": speedup,
            "baseline_metric": baseline_metric,
            "best_metric": best.metric,
            "metric": self.options.metric,
            "params": best.params,
            "config": best.config.to_dict(),
            "result": best.result.to_dict(),
            "record": {
                "record_database": self.options.record_database,
                "record_id": self.options.record_id,
                "kernel_name": kernel.kernel_name,
            },
        }
        self.store.write_best(best_doc)

        summary = {
            "status": "success",
            "num_trials_requested": num_requested,
            "num_trials_completed": completed_trials,
            "num_verified": counts.get("verified", 0),
            "num_failed_verification": counts.get("failed_verification", 0),
            "num_invalid_launch": counts.get("invalid_launch", 0),
            "num_compile_error": counts.get("compile_error", 0),
            "num_runtime_error": counts.get("runtime_error", 0),
            "num_invalid_config": counts.get("invalid_config", 0),
            "num_internal_error": counts.get("internal_error", 0),
            "baseline_metric": baseline_metric,
            "best_metric": best.metric,
            "best_speedup": speedup,
            "best_trial": best.trial,
            "noop_verifies": noop_verifies,
        }
        self.store.write_summary(summary)

        if self.options.proteus_enabled and self.options.proteus_output:
            Path(self.options.proteus_output).parent.mkdir(parents=True, exist_ok=True)
            export_proteus_tuned_kernel(
                self.options.proteus_output,
                recorded,
                kernel,
                best.config,
                usage_filename=str(self.store.path("proteus_usage.txt")),
            )
        
        return summary

    def _evaluate_baseline(
        self,
        executor: AsyncReplayExecutor,
        baseline_config: ExperimentConfiguration,
    ) -> Tuple[ExperimentResult, Optional[float]]:
        """ helper for the baseline since sometimes we want to 
            load it from a previous session.    
        """

        if self.options.resume and not self.options.rerun_baseline:
            loaded = self.store.load_baseline()
            if loaded is not None:
                result, metric = stored_result_and_metric(loaded.get("result", {}))

                if metric is None:
                    metric = metric_value(result, self.options.metric, self.options.iterations)
                return self._checked_baseline(result, metric)

        try:
            result = executor.evaluate(baseline_config)
        except Exception as exc:
            if self.options.fail_fast:
                raise
            result = ExperimentResult(error=str(exc), failed=True, executed=False)
        baseline_metric = metric_value(result, self.options.metric, self.options.iterations)
        self.store.write_baseline(
            {
                "config": baseline_config.to_dict(),
                "result": dict(result.to_dict(), metric=baseline_metric),
            }
        )

        return self._checked_baseline(result, baseline_metric)

    def _checked_baseline(
        self, result: ExperimentResult, metric: Optional[float]
    ) -> Tuple[ExperimentResult, Optional[float]]:
        if not result.verified or metric is None:
            if self.options.require_verified_baseline:
                raise BaselineVerificationError("Baseline replay did not verify")
            return result, None
        return result, metric

    def run(self) -> int:
        """ Main entry point for tuning. Returns an exit code indicating success or failure.
        """

        # validate and save out the config
        try:
            resolved_config = self.options.to_config_dict()
            if self._ir is not None:
                ir = self._ir
                if isinstance(ir, Path) or ir.endswith((".ll", ".bc")):
                    ir = Path(ir).read_bytes()
                else:
                    ir = ir.encode()
                resolved_config["ir_hash"] = hashlib.sha256(ir).hexdigest()
            self._validate_resume(resolved_config)
            self.store.write_config(resolved_config)
        except Exception as exc:
            print(f"Invalid tuning configuration: {exc}")
            return EXIT_INVALID_CONFIGURATION

        # setup mneme from db for replay
        try:
            recorded, kernel = self._load_record()
        except Exception as exc:
            print(f"Failed to load record database: {exc}")
            return EXIT_RECORD_LOAD_FAILED

        # create, save, and print out search space
        try:
            space, space_description = self._build_space(kernel)
        except Exception as exc:
            print(f"Invalid tuning configuration: {exc}")
            return EXIT_INVALID_CONFIGURATION
        self.store.write_search_space(space_description)

        if self.options.print_space:
            print(json.dumps(space_description, indent=2))
            return 0

        # determine baseline config and metric for comparison; this is either loaded from a previous session or evaluated fresh
        baseline_config = self._baseline_config(space, kernel)
        if self.options.dry_run:
            print(json.dumps({"baseline": baseline_config.to_dict(), "search_space": space_description}, indent=2))
            return 0

        self._print("Mneme tune")
        self._print(f"  record database: {self.options.record_database}")
        self._print(f"  record id:       {self.options.record_id}")
        self._print(f"  kernel:          {kernel.kernel_name}")
        source = recorded.kernel_source()
        if source is not None:
            self._print(f"  source:          {source.location}")
        self._print(f"  space preset:    {self.options.space_preset if not self.options.space_module else 'custom'}")
        self._print(f"  sampler:         {self.options.sampler}")
        self._print(f"  trials:          {self.options.trials}")
        self._print(f"  workers:         {self.options.workers}")

        # A caller may reuse an executor across tuning sessions with different IR.
        owns_executor = self._borrowed_executor is None
        executor = self._borrowed_executor
        if executor is None:
            executor = AsyncReplayExecutor(
                record_db=self.options.record_database,
                record_id=self.options.record_id,
                iterations=self.options.iterations,
                results_db_dir=self.options.results_dir,
                num_workers=self.options.workers,
                warmup=self.options.warmup,
                short_circuit=self.options.short_circuit,
            )
        # the main tuning phase
        try:
            if self._ir is not None:
                executor.set_ir(self._ir)
            # ensure baseline passes; collect its performance
            try:
                baseline_result, baseline_metric = self._evaluate_baseline(executor, baseline_config)
            except BaselineVerificationError as exc:
                print(str(exc))
                return EXIT_BASELINE_FAILED

            self._print("")
            if baseline_metric is not None:
                self._print("Baseline verified:")
                self._print(f"  {self.options.metric} time: {baseline_metric:.6g}")
            else:
                self._print("Baseline did not verify; continuing search.")
            if baseline_result.noop_verifies:
                print(noop_verifies_warning(self.options.record_id), file=sys.stderr)

            best, counts, completed_trials = self._load_resume_best(
                baseline_config, baseline_result, baseline_metric
            )
            if self.options.baseline_only:
                # early exit
                summary = self._write_final_outputs(
                    recorded,
                    kernel,
                    baseline_metric,
                    best,
                    counts,
                    completed_trials,
                    self.options.trials,
                    baseline_result.noop_verifies,
                )
                return 0 if best is not None else EXIT_BASELINE_FAILED

            initial_phase = bool(self.options.launch_candidates)
            candidates = (
                self._initial_candidates(space, baseline_config)
                if initial_phase else self._candidate_iter(space)
            )
            in_flight: List[Tuple[Candidate, EvalFuture]] = []
            start_time = time.monotonic()
            submitted = 0
            exhausted = False

            # outer tuning loop -- continue until we've exhausted the candidate iterator and all in-flight trials have completed, or we've hit the timeout 
            while in_flight or not exhausted:
                timed_out = (
                    self.options.timeout is not None
                    and time.monotonic() - start_time >= self.options.timeout
                )

                # inner loop -- submit new candidates until we run out or hit the worker limit
                while not exhausted and not timed_out and len(in_flight) < self.options.workers:

                    try:
                        candidate = next(candidates)
                    except StopIteration:
                        if initial_phase:
                            # Finish supplied layouts before asking the sampler for trials.
                            if in_flight:
                                break
                            initial_phase = False
                            candidates = self._candidate_iter(space)
                            continue
                        exhausted = True
                        break
                    except Exception as exc:
                        if self.options.fail_fast:
                            raise
                        dummy = Candidate(submitted, {}, baseline_config)
                        self._record_invalid_candidate(dummy, baseline_metric, str(exc))
                        counts["invalid_config"] = counts.get("invalid_config", 0) + 1
                        completed_trials += 1
                        submitted += 1
                        continue

                    candidate.trial = submitted
                    config_hash = candidate.config.hash()
                    if self.options.resume and config_hash in self._completed_hashes:
                        continue

                    if candidate.error is not None or not candidate.config.is_valid():
                        completed = self._record_invalid_candidate(
                            candidate, baseline_metric, candidate.error or "ExperimentConfiguration.is_valid() returned false"
                        )
                        counts[completed.status] = counts.get(completed.status, 0) + 1
                        completed_trials += 1
                        submitted += 1
                        continue
                    try:
                        future = executor.submit(candidate.config)
                    except Exception as exc:
                        if self.options.fail_fast:
                            raise
                        future = EvalFuture(candidate.trial, candidate.config)
                        future.set_error(str(exc))
                    in_flight.append((candidate, future))
                    submitted += 1

                # check on in-flight trials and record any that have completed
                made_progress = False
                remaining: List[Tuple[Candidate, EvalFuture]] = []
                for candidate, future in in_flight:
                    if not future.done():
                        remaining.append((candidate, future))
                        continue

                    made_progress = True
                    try:
                        result = future.result()
                        if result is None:
                            result = ExperimentResult(error="Future completed without result", failed=True)
                    except Exception as exc:
                        result = ExperimentResult(error=str(exc), failed=True, executed=False)
                        if self.options.fail_fast:
                            raise

                    # process results
                    status = classify_result(result)
                    if (
                        self.options.adaptive_invalid_ban
                        and status == "invalid_launch"
                    ):
                        self._banned_layouts.add(self._layout_key(candidate.config))
                    
                    candidate_metric = (
                        metric_value(result, self.options.metric, self.options.iterations)
                        if status == "verified"
                        else None
                    )
                    speedup = (
                        baseline_metric / candidate_metric
                        if baseline_metric is not None and candidate_metric and candidate_metric > 0
                        else None
                    )
                    record = self._trial_record(
                        candidate, result, status, candidate_metric, baseline_metric
                    )
                    self.store.append_trial(record)
                    self._tell_optuna(candidate, status, candidate_metric, speedup)

                    counts[status] = counts.get(status, 0) + 1
                    completed_trials += 1

                    if status == "verified" and candidate_metric is not None:
                        relative = f", speedup {speedup:.3f}x" if speedup is not None else ""
                        self._print(f"Trial {candidate.trial} verified: {candidate_metric:.6g}{relative}")
                        current = CompletedCandidate(
                            candidate.trial,
                            candidate.params,
                            candidate.config,
                            result,
                            status,
                            candidate_metric,
                            speedup,
                        )
                        if best is None or best.metric is None or candidate_metric < best.metric:
                            best = current
                    else:
                        self._print(f"Trial {candidate.trial} {status}")

                # update the in-flight list for the next loop iteration
                in_flight = remaining
                if timed_out and not in_flight:
                    break
                if in_flight and not made_progress:
                    time.sleep(0.05)

            summary = self._write_final_outputs(
                recorded,
                kernel,
                baseline_metric,
                best,
                counts,
                completed_trials,
                self.options.trials,
                baseline_result.noop_verifies,
            )
        except Exception as exc:
            if self.options.fail_fast:
                raise
            print(f"Internal tuner error: {type(exc).__name__}: {exc}")
            return EXIT_INTERNAL_ERROR
        finally:
            if owns_executor:
                executor.shutdown()

        self._print("")
        if best is None:
            self._print("No configuration verified")
            return EXIT_BASELINE_FAILED
        self._print("Best verified configuration:")
        self._print(f"  trial:        {best.trial}")
        self._print(f"  {self.options.metric} time: {best.metric:.6g}")
        if baseline_metric is not None:
            self._print(f"  baseline:     {baseline_metric:.6g}")
            self._print(f"  speedup:      {summary['best_speedup']:.3f}x")
        self._print("")
        self._print("Wrote:")
        self._print(f"  results:       {self.options.results_dir}")
        self._print("  best:          best.json")
        if self.options.proteus_enabled and self.options.proteus_output:
            self._print(f"  Proteus JSON:  {self.options.proteus_output}")
            self._print("")
            self._print("Use with full application:")
            self._print(f"  export PROTEUS_TUNED_KERNELS={Path(self.options.proteus_output).resolve()}")
        
        return 0


class BaselineVerificationError(RuntimeError):
    pass
