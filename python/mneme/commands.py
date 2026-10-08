"""
Mneme command-line interface implementation.

This module defines the concrete subcommands exposed by the ``mneme`` CLI.
Each command is implemented as a small class with two static entry points:

  - ``set_cli_args(parser)``: declares command-specific arguments
  - ``run(args, verbosity)``: executes the command

The CLI supports workflows for:
  * recording GPU executions (``record``)
  * cleaning, copying, and moving recorded databases (``clean``, ``copy``, ``move``)
  * querying Mneme configuration (``config``)
  * replaying and executing recorded kernels with custom configurations (``execute``)

This module is intentionally procedural and orchestration-focused.
It delegates all heavy lifting to lower-level Mneme components such as
``RecordedExecution``, ``BaseExecutor``, and ``PipelineManager``.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mneme.llvm import utils
from mneme.mneme_logging import logger
from mneme.mneme_types import ExperimentConfiguration, ExperimentResult, dim3
from mneme.pipeline import PipelineManager
from mneme.profile import init_profiler
from mneme.recorded_execution import RecordedExecution
from mneme.replay_executor import BaseExecutor, noop_verifies_warning
from mneme.tuning.cli import add_tune_args, run_tune
from mneme.utils import MnemeEncoder


def _copy_or_move(sources, dest, move=False):
    for s in sources:
        if not s.exists():
            continue

        exec = RecordedExecution.from_json(str(s))
        new_ir_files = []
        for ll in exec.llvm_files:
            dest_ll = dest / Path(ll).name
            if move:
                shutil.move(ll, dest_ll)
            else:
                shutil.copy(ll, dest_ll)
            new_ir_files.append(str(dest_ll))

        exec.llvm_files = new_ir_files

        for _, kernel in exec.items():
            kfn = Path(kernel.prologue.fn)
            dest_kfn = dest / kfn.name
            if move:
                shutil.move(kfn, dest_kfn)
            else:
                shutil.copy(kfn, dest_kfn)
            kernel.prologue.fn = str(dest_kfn)

            kfn = Path(kernel.epilogue.fn)
            dest_kfn = dest / kfn.name
            if move:
                shutil.move(kfn, dest_kfn)
            else:
                shutil.copy(kfn, dest_kfn)
            kernel.epilogue.fn = str(dest_kfn)

        json_fn = dest / s.name
        exec.to_json(str(json_fn))
        if move:
            s.unlink()
        return 0


class Clean:
    """
    Remove recorded Mneme databases and associated artifacts.

    This command deletes:
      - the specified Mneme JSON database files
      - all referenced LLVM IR files
      - all prologue/epilogue snapshot files

    Use with care: this operation is destructive.
    """

    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "record_database", help="Recorded Mneme DB files", nargs="+"
        )
        parser.set_defaults(func=Clean.run)

    @staticmethod
    def run(args, verbosity):
        dbs = args.record_database
        for db in dbs:
            if not Path(db).exists():
                raise FileNotFoundError(f"Mneme Database '{db}' does not exist")

        ir_records = set()
        mneme_db_files = set()
        for db in dbs:
            exec = RecordedExecution.from_json(db)
            ir_records |= {Path(e) for e in exec.llvm_files}
            for _, kernel in exec.items():
                mneme_db_files.add(Path(kernel.prologue.fn))
                mneme_db_files.add(Path(kernel.epilogue.fn))
            Path(db).unlink()

        for fn in ir_records:
            if fn.exists():
                fn.unlink()

        for fn in mneme_db_files:
            if fn.exists():
                fn.unlink()
        return 0


class Copy:
    """
    Copy one or more Mneme recording databases to a new directory.

    All referenced artifacts (LLVM IR, prologue, epilogue snapshots)
    are duplicated and the database is rewritten to point to the new locations.
    """

    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "record_database",
            nargs="+",
            help="Source paths of Mneme DB records followed by a destination path to which data will be copied to",
        )
        parser.set_defaults(func=Copy.run)

    @staticmethod
    def run(args, verbosity):
        paths = args.record_database
        if len(paths) < 2:
            raise ValueError(
                f"Please provide both source and destination directories {paths}"
            )

        *_sources, _dest = paths
        dest = Path(_dest).absolute()
        if not dest.exists():
            raise RuntimeError(f"Destination target '{str(dest)}' does not exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"Destination target '{str(dest)}' is not a directory"
            )

        sources = [Path(s) for s in _sources]
        return _copy_or_move(sources, dest)


class Move:
    """
    Move one or more Mneme recording databases to a new directory.

    This is equivalent to ``copy`` followed by deletion of the original files.
    All internal paths in the database are rewritten accordingly.
    """

    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "record_database",
            nargs="+",
            help="Source paths of Mneme DB records followed by a destination path to which data will be moved to",
        )
        parser.set_defaults(func=Move.run)

    @staticmethod
    def run(args, verbosity):
        paths = args.record_database
        if len(paths) < 2:
            raise ValueError(
                f"Please provide both source and destination directories {paths}"
            )

        *_sources, _dest = paths
        dest = Path(_dest).absolute()
        if not dest.exists():
            raise RuntimeError(f"Destination target '{str(dest)}' does not exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"Destination target '{str(dest)}' is not a directory"
            )

        sources = [Path(s) for s in _sources]

        return _copy_or_move(sources, dest, move=True)


class Record:
    """
    Record GPU kernel executions using Mneme.

    This command runs a user-provided executable under ``LD_PRELOAD`` with
    the Mneme recording runtime enabled. Kernel launches, memory state,
    and execution metadata are captured into one or more Mneme databases.

    The command expects ``--`` to separate Mneme arguments from the target
    executable and its arguments.
    """

    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "-rdb",
            "--record-db-dir",
            dest="record_db_dir",
            default=os.getcwd(),
            help="Path to directory to store the recorded database(s) and memory snapshots",
        )
        parser.add_argument(
            "--chunk-size",
            default=None,
            help=(
                "Device memory Mneme maps at once and packs allocations into, "
                "as bytes or with a suffix such as 64MB or 4GB, where MB "
                "means MiB (a multiple of 2 MiB; default 4GB). Sets MNEME_CHUNK_SIZE."
            ),
        )
        # Ignored; kept so existing scripts still parse.
        parser.add_argument(
            "-vass",
            "--virtual-address-space-size",
            type=int,
            default=None,
            help=argparse.SUPPRESS,
        )

        parser.add_argument(
            "-mr",
            "--per-kernel-max-recordings",
            type=int,
            default=4,
            help="The maximum number of times to record the same GPU kernel (function) with different dynamic hashes",
        )
        parser.add_argument(
            "-sr",
            "--per-kernel-skip-recordings",
            type=int,
            default=0,
            help="The number of matching GPU kernel launches to skip before recording each kernel",
        )
        parser.add_argument(
            "--epilogue-format",
            choices=["bytes", "diff", "best"],
            default="diff",
            help="The format to use when saving epilogue snapshots: full bytes, diffs from the prologue, or the smaller of the two",
        )
        parser.add_argument(
            "-rr",
            "--record-ranks",
            dest="record_ranks",
            default=None,
            help=(
                "Restrict recording to a comma-separated set of MPI ranks "
                "(e.g. '0', '0,1,3'), or 'all' for every rank. "
                "When omitted, distributed runs default to recording on rank 0 only; "
                "single-process runs always record."
            ),
        )
        parser.add_argument(
            "--copy-source",
            dest="copy_source",
            action="store_true",
            help="Copy the source file that defines each recorded kernel into the record directory",
        )
        parser.add_argument("cmd", nargs=argparse.REMAINDER)
        parser.set_defaults(func=Record.run, parser=parser)

    @staticmethod
    def run(args, verbosity):
        parser = args.parser
        idx = 0
        try:
            idx = args.cmd.index("--")
        except ValueError:
            idx = -1

        if idx != 0:
            parser.error(f"Unrecognized options are passed to mneme {args.cmd[:idx]}")

        cmd = args.cmd[idx + 1 :]
        record_env = os.environ.copy()
        librecord_path = utils.get_mneme_record_library_name()
        logger.debug(f"LD_PRELOAD={librecord_path}")
        record_env["LD_PRELOAD"] = librecord_path
        if args.virtual_address_space_size is not None:
            print(
                "mneme: warning: -vass is deprecated and ignored. Mneme no "
                "longer reserves a fixed address space; it maps device memory "
                "on demand in chunks. Use --chunk-size to set the chunk size.",
                file=sys.stderr,
            )
        if args.chunk_size is not None:
            logger.debug(f"MNEME_CHUNK_SIZE={args.chunk_size}")
            record_env["MNEME_CHUNK_SIZE"] = args.chunk_size
        logger.debug(f"MNEME_MAX_RECORDINGS={args.per_kernel_max_recordings}")
        record_env["MNEME_MAX_RECORDINGS"] = str(args.per_kernel_max_recordings)
        logger.debug(f"MNEME_SKIP_RECORDINGS={args.per_kernel_skip_recordings}")
        record_env["MNEME_SKIP_RECORDINGS"] = str(args.per_kernel_skip_recordings)
        record_db_dir = Path(args.record_db_dir).resolve()
        if record_db_dir.exists() and not record_db_dir.is_dir():
            raise NotADirectoryError(f"Path '{args.record_db_dir}' is not a directory")
        record_db_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"MNEME_DATA_DIR={str(record_db_dir)}")
        record_env["MNEME_DATA_DIR"] = str(record_db_dir)

        record_env["MNEME_EPILOGUE_TYPE"] = args.epilogue_format.lower()

        if args.record_ranks is not None:
            logger.debug(f"MNEME_RECORD_RANKS={args.record_ranks}")
            record_env["MNEME_RECORD_RANKS"] = str(args.record_ranks)

        if args.copy_source:
            logger.debug("MNEME_COPY_SOURCE=1")
            record_env["MNEME_COPY_SOURCE"] = "1"

        if verbosity is not None:
            logger.debug(f"MNEME_LOG_LEVEL={verbosity}")
            record_env["MNEME_LOG_LEVEL"] = verbosity

        try:
            result = subprocess.run(cmd, env=record_env)
            return result.returncode
        except FileNotFoundError:
            parser.error(f"Executable '{cmd[0]}' not found")
        except PermissionError:
            parser.error(f"Executable '{cmd[0]}' is not executable")
        except OSError as e:
            parser.error(f"Failed to execute '{cmd[0]}': {e.strerror}")


class Config:
    """
    Query Mneme build-time configuration.

    This command prints values from Mneme’s installation-time configuration
    file, similar in spirit to ``llvm-config``.
    """

    @staticmethod
    def set_cli_args(parser):
        cfg_file = Path(utils.get_config_file())
        if not cfg_file.exists():
            raise RuntimeError("mneme config file not found — installation broken.")

        with open(cfg_file) as fd:
            cfg = json.load(fd)

        runtime_prefix = str(cfg_file.parent.resolve())

        # Replace @PREFIX@ placeholder with actual mneme installation path
        for k, v in cfg.items():
            if isinstance(v, str):
                cfg[k] = v.replace("@PREFIX@", runtime_prefix)

        parser.add_argument("key", choices=list(cfg.keys()), help="Config key to query")
        parser.set_defaults(func=Config.run, mneme_config=cfg)

    @staticmethod
    def run(args, verbosity):
        key = args.key
        cfg = args.mneme_config

        if key not in cfg:
            raise ValueError(
                f"Unknown config key '{key}'. Available: {list(cfg.keys())}"
            )

        value = cfg[key]

        # Pretty print lists as space-separated, like llvm-config
        if isinstance(value, list):
            print(" ".join(value))
        else:
            print(value)


class Tune:
    """
    Tune a recorded kernel using Mneme's in-process replay executor.
    """

    @staticmethod
    def set_cli_args(parser):
        add_tune_args(parser)
        parser.set_defaults(func=Tune.run, parser=parser)

    @staticmethod
    def run(args, verbosity):
        return run_tune(args, verbosity)


class Replay(BaseExecutor):
    """
    Replay and execute a recorded kernel with a user-defined configuration.

    This command allows:
      - overriding grid/block dimensions
      - enabling specialization and launch bounds
      - selecting an LLVM optimization pipeline
      - controlling code generation parameters
      - executing the kernel multiple times for measurement

    The execution reuses the recorded prologue/epilogue state to ensure
    correctness and reproducibility.
    """

    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "-rdb",
            "--record-database",
            dest="record_db",
            required=True,
            help="Path to Mneme JSON/db file",
        )

        parser.add_argument(
            "-record-id",
            "-rid",
            dest="record_id",
            default=None,
            help="Kernel ID to operate on; inferred when the database contains exactly one instance",
        )

        parser.add_argument(
            "--grid-dim-x",
            "-gidx",
            dest="grid_dim_x",
            type=int,
            default=None,
            help="Value of GridDim.x during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--grid-dim-y",
            "-gidy",
            dest="grid_dim_y",
            type=int,
            default=None,
            help="Value of GridDim.y during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--grid-dim-z",
            "-gidz",
            dest="grid_dim_z",
            type=int,
            default=None,
            help="Value of GridDim.z during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--block-dim-x",
            "-bidx",
            dest="block_dim_x",
            type=int,
            default=None,
            help="Value of BlockDim.x during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--block-dim-y",
            "-bidy",
            dest="block_dim_y",
            type=int,
            default=None,
            help="Value of BlockDim.y during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--block-dim-z",
            "-bidz",
            dest="block_dim_z",
            type=int,
            default=None,
            help="Value of BlockDim.z during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--shared-mem",
            "-shem",
            dest="shared_mem",
            type=int,
            default=None,
            help="Size of shared memory, if not set we default to recorded value",
        )

        parser.add_argument(
            "--specialize",
            default=False,
            required=False,
            action=argparse.BooleanOptionalAction,
            dest="specialize",
            help="Apply argument specialization on the kernel",
        )

        parser.add_argument(
            "--set-launch-bounds",
            "-slb",
            dest="set_launch_bounds",
            default=False,
            required=False,
            action=argparse.BooleanOptionalAction,
            help="Set the launch bounds of the execution",
        )

        parser.add_argument(
            "--max-threads",
            default=None,
            required=False,
            type=int,
            dest="max_threads",
            help="Set launch bound 'max_threads' parameter to the provided value",
        )

        parser.add_argument(
            "--min-threads-per-block",
            default=0,
            type=int,
            dest="min_blocks_per_sm",
            help="Set launch bound 'min_blocks_per_sm' of kernel to the provided value",
        )

        parser.add_argument(
            "--specialize-dims",
            "-sdims",
            dest="specialize_dims",
            default=False,
            required=False,
            action=argparse.BooleanOptionalAction,
            help="Specialize ThreadID.*, BlockDim.* and GridDim.* with constants",
        )

        parser.add_argument(
            "passes",
            help="Compilation pipeline of the kernel to execute",
        )

        parser.add_argument(
            "--codegen-opt",
            "-co",
            dest="codegen_opt",
            type=int,
            default=3,
            help="Optimization level to be used when generating machine code (back end optimizations)",
        )

        parser.add_argument(
            "--iterations",
            "-it",
            required=False,
            type=int,
            help="The number of iterations to run every execution, used to get statistical meaningful results",
            default=3,
        )

        parser.add_argument(
            "--output-ll",
            "-ol",
            dest="output_ll",
            required=False,
            default=None,
            help="Store the output LLVM IR to this file",
        )

        parser.set_defaults(func=Replay.run)

    def __init__(self, *args, **kwargs):
        # NOTE: We need to instantiate the profiler here so
        # that upcoming calls are going to be robust
        init_profiler()
        self.grid_dim_x = kwargs.pop("grid_dim_x", None)
        self.grid_dim_y = kwargs.pop("grid_dim_y", None)
        self.grid_dim_z = kwargs.pop("grid_dim_z", None)

        self.block_dim_x = kwargs.pop("block_dim_x", None)
        self.block_dim_y = kwargs.pop("block_dim_y", None)
        self.block_dim_z = kwargs.pop("block_dim_z", None)

        self.shared_mem = kwargs.pop("shared_mem", None)
        self.specialize = kwargs.pop("specialize", False)
        self.set_launch_bounds = kwargs.pop("set_launch_bounds", False)
        self.max_threads = kwargs.pop("max_threads", None)
        self.min_blocks_per_sm = kwargs.pop("min_blocks_per_sm", 0)
        self.specialize_dims = kwargs.pop("specialize_dims", False)
        self.passes = kwargs.pop("passes", None)
        self.codegen_opt = kwargs.pop("codegen_opt", 3)

        self.output_ll = kwargs.pop("output_ll", None)

        super().__init__(*args, **kwargs)
        self.pass_manager = PipelineManager()
        if self.passes not in (
            "default<O3>",
            "default<O2>",
            "default<O1>",
            "default<O0>",
            "default<Os>",
            "default<Oz>",
        ):
            self.passes = self.pass_manager.to_string(
                self.pass_manager.from_string(self.passes)
            )
        else:
            self.passes = self.passes
        self._db = None

    def get_mneme_config(self, passes):
        """
        Construct an ExperimentConfiguration from CLI arguments and recorded defaults.

        CLI-provided values override recorded values where present.
        Missing values are filled from the original recorded execution.

        Parameters
        ----------
        passes
            Either a parsed pipeline (list of concrete passes) or a default pipeline string.

        Returns
        -------
        ExperimentConfiguration
            Fully specified configuration ready for execution.
        """
        self.block_dim_x = (
            self.kernel_descr.block_dim.x
            if (self.block_dim_x is None)
            else self.block_dim_x
        )
        self.block_dim_y = (
            self.kernel_descr.block_dim.y
            if (self.block_dim_y is None)
            else self.block_dim_y
        )
        self.block_dim_z = (
            self.kernel_descr.block_dim.z
            if (self.block_dim_z is None)
            else self.block_dim_z
        )

        self.grid_dim_x = (
            self.kernel_descr.grid_dim.x
            if (self.grid_dim_x is None)
            else self.grid_dim_x
        )
        self.grid_dim_y = (
            self.kernel_descr.grid_dim.y
            if (self.grid_dim_y is None)
            else self.grid_dim_y
        )
        self.grid_dim_z = (
            self.kernel_descr.grid_dim.z
            if (self.grid_dim_z is None)
            else self.grid_dim_z
        )

        max_threads = self.max_threads
        if self.set_launch_bounds:
            if self.max_threads == -1 or self.max_threads is None:
                max_threads = self.block_dim_x * self.block_dim_y * self.block_dim_z

        self.shared_mem = (
            self.kernel_descr.shared_mem if self.shared_mem is None else self.shared_mem
        )

        return ExperimentConfiguration(
            grid=dim3(self.grid_dim_x, self.grid_dim_y, self.grid_dim_z),
            block=dim3(self.block_dim_x, self.block_dim_y, self.block_dim_z),
            shared_mem=self.shared_mem,
            specialize=self.specialize,
            set_launch_bounds=self.set_launch_bounds,
            max_threads=max_threads,
            min_blocks_per_sm=self.min_blocks_per_sm,
            specialize_dims=self.specialize_dims,
            passes=passes,
            codegen_opt=self.codegen_opt,
            prune=True,
            internalize=True,
        )

    def __str__(self):
        return f"{self.__class__.__name__}"

    def execute(self, config, ir_module, clone=False, orig=""):
        result = ExperimentResult()
        generated_ir = super()._execute(result, config, ir_module)
        if self.output_ll is not None:
            with open(self.output_ll, "w") as fd:
                fd.write(str(generated_ir))
        return result

    @staticmethod
    def run(args, verbosity):
        kwargs = vars(args)
        kwargs.pop("command")
        kwargs.pop("func")
        executor = Replay(**kwargs)

        # We currently link all LLVM IR modules together
        # NOTE: Does this break with externals on CUDA?
        root_ir = executor.link_ir()

        with executor as Memory:
            if executor.noop_verifies:
                print(noop_verifies_warning(executor.record_id), file=sys.stderr)
            exp = executor.get_mneme_config(executor.passes)
            res = executor.execute(
                exp,
                root_ir.clone(),
                True,
            )
            out = {"Replay-config": exp.to_dict(), "Result": res.to_dict()}
            print(
                json.dumps(out, cls=MnemeEncoder, indent=2),
            )

        return 0
