import argparse
import json
from dataclasses import fields
from typing import Any, Dict

from mneme.tuning.session import TuneOptions, TuningSession, merge_config_and_args


def _none_bool_action() -> Any:
    return argparse.BooleanOptionalAction


def add_tune_args(parser: argparse.ArgumentParser) -> None:
    record = parser.add_argument_group("record selection")
    record.add_argument(
        "-rdb",
        "--record-database",
        dest="record_database",
        default=None,
        help="Path to the Mneme record database containing the kernel replay.",
    )
    record.add_argument(
        "-rid",
        "--record-id",
        dest="record_id",
        default=None,
        help="Record id within the database to tune.",
    )

    search = parser.add_argument_group("search control")
    search.add_argument(
        "--space-preset",
        choices=["quick", "standard", "launch", "compiler", "full"],
        default=None,
        help=(
            "Built-in search-space preset. quick searches a small space, launch "
            "focuses on launch parameters, compiler focuses on compiler choices, "
            "and full enables the broadest built-in space."
        ),
    )
    search.add_argument(
        "--sampler",
        choices=["random", "tpe", "grid", "exhaustive"],
        default=None,
        help="Candidate selection strategy. random and tpe require --trials.",
    )
    search.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of candidate configurations to evaluate for random or TPE search.",
    )
    search.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Stop submitting new trials after this many seconds.",
    )
    search.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used by the random and TPE samplers.",
    )
    search.add_argument(
        "--metric",
        choices=["mean", "median", "min", "max"],
        default=None,
        help="Timing statistic used to summarize multiple evaluations of the same candidate configuration.",
    )
    search.add_argument(
        "--objective",
        choices=["time", "speedup"],
        default=None,
        help="Optimize raw execution time or speedup relative to the baseline.",
    )

    replay = parser.add_argument_group("replay execution")
    replay.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Measured replay iterations per candidate. The selected metric is computed from these samples.",
    )
    replay.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup replay iterations to run before collecting timing samples.",
    )
    replay.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of replay workers used to evaluate candidates concurrently.",
    )
    replay.add_argument(
        "--compare-policy",
        default=None,
        help="JSON file that overrides the recorded comparison metadata used to verify candidates.",
    )
    replay.add_argument(
        "--fail-fast",
        action="store_true",
        default=None,
        help="Stop the tuning run after the first candidate evaluation error.",
    )
    replay.add_argument(
        "--keep-going",
        dest="fail_fast",
        action="store_false",
        default=None,
        help="Continue evaluating candidates after errors. Default behavior.",
    )

    output = parser.add_argument_group("output and run mode")
    output.add_argument(
        "--results-dir",
        default=None,
        help="Directory for config, search-space, trial, baseline, summary, and best-result files.",
    )
    output.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume from an existing results directory and skip completed trial configurations.",
    )
    output.add_argument(
        "--require-verified-baseline",
        action=_none_bool_action(),
        default=None,
        help=(
            "Require the recorded baseline to verify (default: enabled). "
            "Use --no-require-verified-baseline to continue after failure with --objective time."
        ),
    )
    output.add_argument(
        "--rerun-baseline",
        action="store_true",
        default=None,
        help="Re-evaluate the baseline when resuming instead of loading the saved baseline result.",
    )
    output.add_argument(
        "--print-space",
        action="store_true",
        default=None,
        help="Print the resolved search-space description and exit.",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Print the baseline configuration and search-space description without replaying candidates.",
    )
    output.add_argument(
        "--baseline-only",
        action="store_true",
        default=None,
        help="Evaluate only the baseline configuration, write outputs, and exit.",
    )
    output.add_argument(
        "--quiet",
        action="store_true",
        default=None,
        help="Suppress progress output.",
    )
    output.add_argument(
        "--proteus-output",
        default=None,
        help="Path for exported Proteus tuned-kernel metadata.",
    )
    output.add_argument(
        "--no-proteus-output",
        dest="proteus_enabled",
        action="store_false",
        default=None,
        help="Disable Proteus tuned-kernel metadata export.",
    )

    optuna = parser.add_argument_group("Optuna")
    optuna.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name. Useful with --optuna-storage or --resume.",
    )
    optuna.add_argument(
        "--optuna-storage",
        default=None,
        help="Optuna storage URL, such as sqlite:///tune.db, for persistent studies.",
    )
    optuna.add_argument(
        "--pruner",
        choices=["none", "median"],
        default=None,
        help="Optuna pruner used for Optuna-backed search.",
    )

    launch = parser.add_argument_group("launch-space controls")
    launch.add_argument(
        "--launch-dim",
        choices=["none", "auto", "x", "xy", "xyz"],
        default=None,
        help="Launch axes whose block sizes may vary. none keeps the recorded launch shape.",
    )
    launch.add_argument(
        "--launch-safety",
        choices=["conservative", "balanced", "aggressive"],
        default=None,
        help="How broadly to generate candidate block shapes for the selected launch axes.",
    )
    launch.add_argument(
        "--adaptive-invalid-ban",
        dest="adaptive_invalid_ban",
        action=_none_bool_action(),
        default=None,
        help="Adaptively avoid block shapes that produce invalid launch results.",
    )

    compiler = parser.add_argument_group("compiler-space controls")
    compiler.add_argument(
        "--passes",
        nargs="+",
        default=None,
        help="Candidate compiler pass pipelines to include in the search space.",
    )
    compiler.add_argument(
        "--fixed-passes",
        default=None,
        help="Compiler pass pipeline to use for every candidate.",
    )
    compiler.add_argument(
        "--pipeline-file",
        default=None,
        help="File containing compiler pass pipelines to include as candidate choices.",
    )
    compiler.add_argument(
        "--codegen-opt-range",
        default=None,
        help="Integer range for codegen optimization level, formatted as LOW:HIGH[:STEP].",
    )
    compiler.add_argument(
        "--fixed-codegen-opt",
        type=int,
        default=None,
        help="Codegen optimization level to use for every candidate.",
    )
    compiler.add_argument(
        "--codegen-method",
        default=None,
        help="Codegen method metadata to attach to the built-in search space.",
    )

    config_space = parser.add_argument_group("configuration-space controls")
    config_space.add_argument(
        "--specialize-space",
        choices=["fixed", "on", "off", "on-off"],
        default=None,
        help="Whether kernel specialization is fixed, forced on, forced off, or searched.",
    )
    config_space.add_argument(
        "--specialize-dims-space",
        choices=["fixed", "on", "off", "on-off"],
        default=None,
        help="Whether launch-dimension specialization is fixed, forced on, forced off, or searched.",
    )
    config_space.add_argument(
        "--launch-bounds-space",
        choices=["fixed", "on", "off", "on-off"],
        default=None,
        help="Whether launch bounds are fixed, forced on, forced off, or searched.",
    )
    config_space.add_argument(
        "--min-blocks-per-sm-range",
        default=None,
        help="Integer range for launch-bounds min_blocks_per_sm, formatted as LOW:HIGH[:STEP].",
    )
    config_space.add_argument(
        "--fixed-min-blocks-per-sm",
        type=int,
        default=None,
        help="Launch-bounds min_blocks_per_sm value to use for every candidate.",
    )
    config_space.add_argument(
        "--max-threads-range",
        default=None,
        help="Integer range for launch-bounds max_threads, formatted as LOW:HIGH[:STEP].",
    )
    config_space.add_argument(
        "--max-threads-policy",
        choices=["recorded", "block-threads", "powers-of-two", "hardware"],
        default=None,
        help=(
            "How to derive max_threads when launch bounds are enabled and "
            "--max-threads-range is not set."
        ),
    )

    custom = parser.add_argument_group("custom search space")
    custom.add_argument(
        "--space-module",
        default=None,
        help="Custom search-space class, formatted as MODULE_OR_PATH:CLASS.",
    )
    custom.add_argument(
        "--space-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Argument passed to the custom search-space constructor. May be repeated.",
    )

    config = parser.add_argument_group("configuration files")
    config.add_argument(
        "--config",
        default=None,
        help="JSON or YAML tune configuration file. CLI options override file values.",
    )
    config.add_argument(
        "--dump-config",
        default=None,
        help="Write the resolved tune configuration to this path and exit.",
    )
    parser.set_defaults(func=run_tune)


def _options_from_mapping(data: Dict[str, Any]) -> TuneOptions:
    """ Incoming options includes both CLI args and config file options
        Merge back into TuneOptions config obj.
    """
    valid = {f.name for f in fields(TuneOptions)}
    kwargs = {key: value for key, value in data.items() if key in valid}

    missing = [key for key in ("record_database", "record_id") if not kwargs.get(key)]
    if missing:
        raise ValueError(f"Missing required tune option(s): {', '.join(missing)}")
    
    if kwargs.get("space_arg") is None:
        kwargs["space_arg"] = []
    return TuneOptions(**kwargs)


def run_tune(args: argparse.Namespace, verbosity) -> int:
    try:
        # Create TuneOptions from cli and config file
        merged = merge_config_and_args(args)
        options = _options_from_mapping(merged)

        if options.dump_config:
            resolved = options.to_config_dict()
            with open(options.dump_config, "w") as fd:
                json.dump(resolved, fd, indent=2)
                fd.write("\n")
            return 0
        return TuningSession(options).run()
    except ValueError as exc:
        parser = getattr(args, "parser", None)
        if parser is not None:
            parser.error(str(exc))
        print(f"Invalid tuning configuration: {exc}")
        return 3
