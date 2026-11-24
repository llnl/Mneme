"""
mneme.replay_executor

Core record–replay execution pipeline for Mneme.

This module provides the execution backbone used by both:
  * the synchronous CLI execution path (via BaseExecutor subclasses), and
  * the asynchronous tuning engine (via TuneWorker).

At a high level, an "experiment" in Mneme is:
  1) Load a recorded kernel execution (RecordedExecution + KernelInstance).
  2) Reconstruct the recorded GPU memory state (prologue/epilogue snapshots),
     mapping the prologue at its recorded device addresses.
  3) Link recorded LLVM IR modules into a single IR module suitable for replay.
  4) Apply optional IR specializations (arguments, launch dims, launch bounds).
  5) Run an optimization pipeline and generate a device object.
  6) Execute a chain of runs (see :mod:`mneme.replay_runs`): by default a
     verification run that compares the device memory with the recorded
     epilogue, then a timing run that collects timing/resource metrics.

Worker processes amortize initialization costs by reusing a single executor.

Public API
----------
BaseExecutor:
  Base class that owns GPU affinity, recorded state, and the build/run pipeline.

TuneWorker:
  Worker-process implementation used by the async tuning infrastructure.
"""

import os
import time
from datetime import datetime, timezone
from multiprocessing import Event, Queue
from pathlib import Path
from typing import Optional, Sequence, Tuple

from mneme import tracing
from mneme.device import get_device_arch, get_device_count, set_device
from mneme.llvm._lib_path_config import MNEME_PASS_PLUGIN_LIB
from mneme.llvm.buffer import MemBufferRef
from mneme.llvm.module import ModuleRef, parse_assembly, parse_bitcode
from mneme.mneme_logging import logger
from mneme.mneme_types import BuildStats, ExperimentConfiguration, ExperimentResult
from mneme.pass_plugins import resolve_pass_plugins
from mneme.proteus import jit
from mneme.recorded_execution import RecordedExecution
from mneme.replay_runs import (
    BuildArtifact,
    BuildVariant,
    ChainOutcome,
    ReplayExperiment,
    RunChain,
    default_run_chain,
)


def noop_verifies_warning(record_id: str) -> str:
    """Warning shown to users when a do-nothing kernel would pass verification."""
    return (
        f"Warning: the recorded input of record {record_id} already matches its "
        "recorded output, so a kernel that does nothing would pass verification. "
        "Consider re-recording with --filter-config and a skip policy, or using a "
        "different record id."
    )


class BaseExecutor:
    """
    Base class for executing Mneme record–replay experiments.

    A BaseExecutor instance is bound to:
      * one recorded database file (record_db),
      * one kernel instance inside that database (record_id), inferred when the
        database contains exactly one instance,
      * one GPU device (device_id),
      * and the run chain every experiment executes (run_chain).

    Responsibilities
    ---------------
    * Load the recorded execution metadata (RecordedExecution) and select the
      target KernelInstance (kernel_descr).
    * Pin the current OS process to a specific GPU device (set_device()).
    * Manage the recorded snapshots:
        - the prologue is mapped at its recorded device addresses.
        - prologue/epilogue snapshots are opened and later compared.
    * Provide a structured pipeline that takes IR -> object -> execution:
        - _preprocess_ir(): apply specialization transforms and compute a variant hash
        - _optimize(): run pass pipeline / O-level selection
        - _codegen(): lower to a device object (MemBufferRef)
        - build(): run the three stages on a clone of the IR and measure them
        - execute(): execute the run chain on one configuration

    Lifecycle
    ---------
    BaseExecutor is designed to be used as a context manager:

        executor = MyExecutor(record_db=..., record_id=..., device_id=...)
        root_ir = executor.link_ir()
        with executor:
            res = executor.execute(...)

    The context manager ensures GPU memory state (snapshots) is
    opened exactly once and released even when execution raises.

    Notes / invariants
    ------------------
    * A BaseExecutor instance is intended to be used within a single process.
      (Workers should construct one executor per worker process.)
    * open() must be called before any execution; execute() assumes prologue and
      epilogue states are loaded.
    * link_ir() returns a linked IR module representing the recorded kernel; callers
      should clone before mutation if reusing across experiments.
    """

    @tracing.traced("executor.initialize", flush_after=True)
    def __init__(
        self,
        record_db: str = "",
        record_id: Optional[str] = None,
        run_chain: Optional[RunChain] = None,
        device_id: int = 0,
        pass_plugins: Sequence[str] = (),
    ):
        """
        Parameters
        ----------
        run_chain : RunChain, optional
            Runs every experiment executes; ``default_run_chain()`` when omitted.
            The chain prepares the process before any device call.
        """
        self.run_chain = run_chain if run_chain is not None else default_run_chain()
        self.run_chain.prepare_process()
        self.record_db = record_db
        self.device_id = device_id
        with tracing.span(
            "metadata.load", lambda: {"record_database": str(self.record_db)}
        ):
            self.records = RecordedExecution.from_json(self.record_db)
        if record_id is None:
            num_records = len(self.records)
            if num_records != 1:
                raise ValueError(
                    f"Cannot infer record ID from '{self.record_db}': expected "
                    f"exactly one recorded instance, found {num_records}. "
                    "Specify -rid/-record-id."
                )
            record_id = next(iter(self.records))
        self.record_id = record_id
        logger.debug(
            f"BaseExecutor Got {self.record_db} and {self.record_id} and will run on device:{self.device_id}"
        )
        self.kernel_descr = self.records[self.record_id]
        self.device_arch = get_device_arch()
        self._epilogue = None
        self._prologue = None
        self.noop_verifies = False
        self.num_devices = get_device_count()
        set_device(device_id)
        logger.debug(
            f"GPU Affinity of process was set to device:{self.device_id} out of {self.num_devices}"
        )
        jit.register_pass_plugin(MNEME_PASS_PLUGIN_LIB)
        for library in resolve_pass_plugins(pass_plugins):
            jit.register_pass_plugin(library)
        with tracing.span(
            "metadata.record",
            lambda: {
                "record_database": str(self.record_db),
                "record_id": self.record_id,
                "kernel": self.kernel_descr.kernel_name,
                "device_id": self.device_id,
                "device_arch": self.device_arch,
            },
        ):
            pass

    @tracing.traced("snapshot.setup", flush_after=True)
    def open(self):
        with tracing.span(
            "snapshot.prologue.open",
            lambda: {"path": self.kernel_descr.prologue.fn},
        ):
            self._prologue = self.kernel_descr.prologue.open()

        with tracing.span(
            "snapshot.epilogue.open",
            lambda: {"path": self.kernel_descr.epilogue.fn},
        ):
            self._epilogue = self.kernel_descr.epilogue.open()

        # No kernel has run yet, so this asks whether doing nothing would verify.
        with tracing.span("verify.noop") as event:
            self.noop_verifies = self._prologue.matches_unlaunched(self._epilogue)
            event.annotate(noop_verifies=self.noop_verifies)

        return self

    @property
    def prologue(self):
        return self._prologue

    @property
    def epilogue(self):
        return self._epilogue

    @tracing.traced("snapshot.cleanup", flush_after=True)
    def close(self):
        if self._epilogue is not None:
            with tracing.span("snapshot.epilogue.close"):
                self._epilogue.close()
            self._epilogue = None
        if self._prologue is not None:
            with tracing.span("snapshot.prologue.close"):
                self._prologue.close()
            self._prologue = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @tracing.traced("ir.link", flush_after=True)
    def link_ir(self):
        return self.records.link_llvm_modules(prune=True, internalize=True)

    @tracing.traced("ir.replace", flush_after=True)
    def set_new_ir(self, ir_path_or_asm: str):
        if isinstance(ir_path_or_asm, str) and (ir_path_or_asm.endswith('.ll') or ir_path_or_asm.endswith('.bc')):
            ir_path = Path(ir_path_or_asm)
            if ir_path.suffix == '.bc':
                with open(ir_path, 'rb') as f:
                    new_ir = parse_bitcode(f.read())
            else: # .ll
                with open(ir_path, 'r') as f:
                    new_ir = parse_assembly(f.read())
        else:
            # assume str is text IR
            new_ir = parse_assembly(ir_path_or_asm)

        # apply internalization and pruning
        jit.internalize(new_ir, self.kernel_descr.kernel_name)
        jit.pruneIR(new_ir)

        return new_ir

    @tracing.traced("ir.preprocess")
    def _preprocess_ir(
        self,
        config: ExperimentConfiguration,
        llvm_ir: ModuleRef,
    ) -> Tuple[str, ModuleRef]:
        """
        Apply IR-level preprocessing and specialization transformations prior to
        kernel code generation.

        This method computes a deterministic code hash reflecting all applied
        specializations and transformations. The input IR module may be modified
        in-place depending on the selected configuration options. The resulting
        hash is used to uniquely identify the transformed kernel and ensure
        reproducibility across record/replay runs.

        The preprocessing pipeline consists of the following conditional steps:

        1. **Argument specialization** (``config.specialize``)
           Specializes the kernel based on recorded argument values from the
           prologue. This may produce more optimized IR for kernels whose behavior
           depends on constant parameters.

        2. **Launch-dimension specialization** (``config.specialize_dims``)
           Specializes the kernel based on the provided grid and block dimensions,
           enabling IR simplification or elimination of dimension-dependent logic.

        3. **Launch-bounds insertion** (``config.set_launch_bounds``)
           Applies explicit CUDA/HIP launch bounds using the maximum threads per block
           and minimum blocks per SM provided in the experiment configuration.

        Each transformation updates the evolving code hash to reflect the applied
        change, ensuring that semantically distinct IR variants map to unique
        identifiers.

        Parameters
        ----------
        config : ExperimentConfiguration
            Configuration controlling which IR specializations are applied.
        llvm_ir : ModuleRef
            Intermediate representation (LLVM-like) to be specialized. The module
            may be modified during preprocessing.

        Returns
        -------
        (str, ModuleRef)
            A tuple containing:

            * **str** – The updated code hash after all applicable transformations.
            * **ModuleRef** – The (potentially modified) IR module.

        Notes
        -----
        * IR-specialization routines are delegated to the ``proteus`` subsystem.

        """

        code_hash = self.kernel_descr.static_hash

        if config.specialize:
            code_hash = jit.specialize_args(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                self.prologue.args,
                self.prologue.num_args,
                self.kernel_descr.available_specializations,
            )

        if config.specialize_dims:
            code_hash = jit.specialize_dims(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                config.grid,
                config.block,
            )
        if config.set_launch_bounds:
            code_hash = jit.set_launch_bounds(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                config.max_threads,
                config.min_blocks_per_sm,
            )
        return code_hash, llvm_ir

    @tracing.traced("ir.optimize")
    def _optimize(
        self,
        config: ExperimentConfiguration,
        ir_module: ModuleRef,
        purge_autoinit: bool,
    ):
        """
        Apply optimization passes to the IR module prior to code generation.

        This method invokes the JIT optimization pipeline configured for the current
        device architecture. The pipeline typically includes both generic compiler
        optimizations (e.g., ``O1–O3``) and Mneme-specific IR transformations
        specified in the experiment configuration. Optimization operates in-place on
        the provided IR module.

        Parameters
        ----------
        config : ExperimentConfiguration
            Configuration controlling the optimization pipeline. Relevant fields
            include:
            * ``passes`` – Name or specification of the optimization pass pipeline.
            * ``codegen_opt`` – Code generation optimization level.
        ir_module : ModuleRef
            The intermediate representation to be optimized. The module is mutated
            in-place by the underlying JIT subsystem.
        purge_autoinit : bool
            Whether to run the ``purge-autoinit`` pass before ``config.passes``.

        Notes
        -----
        * Optimization routines are delegated to ``jit.optimize``.
        * The optimization phase typically precedes code generation and may
          significantly affect both performance and final code size.
        * This method does not return a value; the IR module is modified directly.

        """
        # purge-autoinit runs as its own pipeline because prefixing it to
        # config.passes would change how LLVM infers the nesting of a flat list.
        if purge_autoinit:
            jit.optimize(ir_module, self.device_arch, "purge-autoinit", config.codegen_opt)
        jit.optimize(ir_module, self.device_arch, config.passes, config.codegen_opt)

    @tracing.traced("ir.codegen")
    def _codegen(
        self,
        config: ExperimentConfiguration,
        ir_module: ModuleRef,
    ) -> MemBufferRef:
        """
        Generate a device-executable object from the optimized IR module.

        This method invokes the JIT backend to lower the intermediate representation
        into a binary object suitable for loading and execution on the target device.
        The resulting artifact is returned as a :class:`MemBufferRef`. Code generation
        behavior—including backend choice and optimization level—is controlled by the
        experiment configuration.

        Parameters
        ----------
        config : ExperimentConfiguration
            Experiment configuration specifying code-generation parameters:
            * ``codegen_opt`` – Optimization level for the code-generation backend.
        ir_module : ModuleRef
            Optimized IR module to be lowered into an executable object. Must be the
            output of prior preprocessing and optimization stages.

        Returns
        -------
        MemBufferRef
            A memory buffer containing the generated object code. This buffer can be
            loaded into a device runtime via ``DeviceModule.from_MemBuffer`` for
            execution.

        Notes
        -----
        * The code generation step is performed by the ``jit.codegen_object`` backend.
        * Code generation typically represents the final stage of the build pipeline
          before the kernel is executed on the device.
        * Returned memory buffers may include architecture-specific metadata depending
          on the JIT backend used.

        """
        return jit.codegen_object(
            ir_module, self.device_arch, config.codegen_opt
        )

    @tracing.traced("ir.build")
    def build(
        self,
        config: ExperimentConfiguration,
        root_ir: ModuleRef,
        variant: BuildVariant,
    ) -> BuildArtifact:
        """
        Build the device kernel for ``config`` from a clone of ``root_ir``.

        The build runs three timed stages on the clone:

        1. **IR preprocessing**
           Applies specialization, dimension-dependent transformations, and optional
           launch-bound insertion.

        2. **Optimization**
           Runs ``purge-autoinit`` first when ``variant.purge_autoinit`` is set,
           then the configured optimization passes (e.g., ``O3`` or user-defined
           pipelines).

        3. **Code generation**
           Lowers the optimized IR into a device-executable object.

        Parameters
        ----------
        config : ExperimentConfiguration
            Configuration controlling specialization, optimization pipeline, and
            code-generation strategy.
        root_ir : ModuleRef
            IR of the recorded kernel. It is cloned and not modified.
        variant : BuildVariant
            How the run that asked for this build wants the kernel built.

        Returns
        -------
        BuildArtifact
            The transformed IR, the device object, and the stage times and object
            size.
        """
        with tracing.span("ir.clone"):
            ir_module = root_ir.clone()

        start = time.perf_counter()
        self._preprocess_ir(config, ir_module)
        preprocessed = time.perf_counter()
        self._optimize(config, ir_module, purge_autoinit=variant.purge_autoinit)
        optimized = time.perf_counter()
        mem_buffer = self._codegen(config, ir_module)
        generated = time.perf_counter()

        stats = BuildStats(
            preprocess_ir_time=preprocessed - start,
            opt_time=optimized - preprocessed,
            codegen_time=generated - optimized,
            obj_size=mem_buffer.get_size(),
        )
        return BuildArtifact(variant, ir_module, mem_buffer, stats)

    @tracing.traced(
        "experiment",
        attributes=lambda self, config, root_ir: {
            "record_id": self.record_id,
            "config_hash": config.hash(),
            "configuration": config.to_dict(),
        },
        flush_after=True,
    )
    def execute(
        self,
        config: ExperimentConfiguration,
        root_ir: ModuleRef,
    ) -> ChainOutcome:
        """
        Execute the run chain on one configuration of the recorded kernel.

        Parameters
        ----------
        config : ExperimentConfiguration
            The experiment configuration controlling launch parameters, specialization,
            and code generation settings.
        root_ir : ModuleRef
            IR of the recorded kernel. Every build clones it; it is not modified.

        Returns
        -------
        ChainOutcome
            The results of the runs that executed, whether every run executed,
            and the build the last run used.

        Raises
        ------
        RuntimeError
            If internal prologue or epilogue state is unexpectedly ``None``.
        """
        if self._prologue._state is None or self._epilogue._state is None:
            raise RuntimeError("States should never be none when executing a kernel")

        return self.run_chain.execute(ReplayExperiment(self, config, root_ir))


class TuneWorker(BaseExecutor):
    """
    Worker-side executor used by the asynchronous tuning infrastructure.

    ``TuneWorker`` is a concrete :class:`BaseExecutor` specialization intended to run
    inside a dedicated worker process. It owns the GPU affinity, prologue/epilogue
    state, and JIT pipeline required to compile and replay a recorded
    kernel under a given :class:`ExperimentConfiguration`.

    A worker process typically:
      1) Prepares the process for its run chain and selects a GPU device.
      2) Loads the recorded execution (record DB + record ID).
      3) Links the recorded LLVM IR into a single module (``link_ir``).
      4) Enters a message-processing loop (see :meth:`run`) to evaluate configurations.

    Notes
    -----
    * The public entry point for the worker process is :meth:`run`, which is designed
      to be used as a multiprocessing target.
    * Per-request execution is handled by :meth:`process_payload`, which executes
      the run chain under the provided configuration.
    * The worker process should construct a single TuneWorker and reuse it for
      multiple requests to amortize startup overhead.
    """

    def process_payload(
        self, ir_module, config: ExperimentConfiguration
    ) -> ExperimentResult:
        """
        Execute one tuning request: run the chain on the kernel under ``config``.

        Parameters
        ----------
        ir_module : ModuleRef
            Root IR module (or clone) used as input for this experiment. Every
            build clones it.
        config : ExperimentConfiguration
            Configuration describing launch parameters, specialization options, and
            code-generation controls for this experiment.

        Returns
        -------
        ExperimentResult
            The results of the runs that executed, with start and end timestamps
            (ISO 8601, UTC) and the GPU id. An invalid pass pipeline is reported
            as a failed result instead of raising.
        """
        start_time = datetime.now(timezone.utc).isoformat()
        try:
            outcome = self.execute(config, ir_module)
            result = ExperimentResult(
                outcome.results.values(), executed=outcome.executed_every_run
            )
        except jit.InvalidPassPipelineError as error:
            result = ExperimentResult(failed=True, error=str(error))
        result.start_time = start_time
        result.end_time = datetime.now(timezone.utc).isoformat()
        result.gpu_id = self.device_id
        return result

    @staticmethod
    @tracing.traced("worker", flush_after=True)
    def run(
        request_q: Queue,
        response_q: Queue,
        record_db: str,
        record_id: str,
        device_id: int,
        results_db_dir: str,
        state: Event,
        run_chain: RunChain,
        pass_plugins: Sequence[str] = (),
    ):
        """
        Worker process entry point: initialize resources and serve requests from a queue.

        This method is designed to be used as the target function for a worker
        ``multiprocessing.Process``. It performs one-time initialization and then
        enters a blocking loop that processes messages from ``request_q``.

        Initialization performed once per worker:
          1) Redirects stdout/stderr to a per-worker log file:
             ``{results_db_dir}/Worker-{device_id}.log``. This avoids interleaved
             output across processes.
          2) Constructs a :class:`TuneWorker` with the given recording and device id.
          3) Links and caches the root IR module (``root_ir``) that will be cloned
             per experiment.
          4) Opens GPU memory/prologue/epilogue resources via the executor context
             manager (``with worker as Memory``).
          5) Signals readiness by setting ``state``.

        Message protocol:
          - ``{"payload": "terminate", ...}``:
            Stop the worker loop and exit.
          - ``{"payload": "process", "exp_id": <id>, "data": <config-dict>}``:
            Execute an experiment and respond on ``response_q`` with:
            ``{"exp_id": <id>, "payload": "result", "data": <result-dict>, "llvm_ir": ""}``.

        Parameters
        ----------
        request_q : multiprocessing.Queue
            Queue from which the worker receives control messages and experiment requests.
        response_q : multiprocessing.Queue
            Queue to which the worker publishes experiment results.
        record_db : str
            Path to the recorded execution database/file used to construct the executor.
        record_id : str
            Identifier of the recorded kernel instance inside ``record_db``.
        device_id : int
            GPU device index to which this worker process is pinned.
        run_chain : RunChain
            Runs every experiment executes.
        pass_plugins : Sequence[str]
            Pass plugin libraries the worker loads into the JIT.
        results_db_dir : str
            Directory where per-worker logs and output artifacts are written.
        state : multiprocessing.Event
            Event used to signal to the parent process that initialization is complete
            and the worker is ready to accept requests.

        Notes
        -----
        * The worker loop blocks on ``request_q.get()`` until a message arrives.
        * The worker clones ``root_ir`` per request to avoid cross-request IR mutation.
        * Exceptions raised inside the loop will currently propagate and terminate the
          worker process; higher-level infrastructure should treat this as a worker crash.

        """
        # NOTE: We open a file for every individual executor and give persmisions, then we redirect stdout/stderr
        # to that file. We do this to not conflict our messages

        fd_out = os.open(
            f"{results_db_dir}/Worker-{device_id}.log",
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        )
        os.dup2(fd_out, 1)  # 1 = stdout
        os.dup2(fd_out, 2)  # 2 = stderr
        worker = TuneWorker(
            record_db=record_db,
            record_id=record_id,
            device_id=device_id,
            run_chain=run_chain,
            pass_plugins=pass_plugins,
        )
        # Open GPU memory, setup prologue epilogue and create a single
        # LLVM IR file to start working on optimizations
        root_ir = worker.link_ir()

        with worker as Memory:
            state.set()
            logger.debug(f"Worker running on {worker.device_id} starts busy loop")
            while True:
                msg = request_q.get()
                if msg["payload"] == "terminate":
                    logger.debug(
                        f"Worker {worker.device_id} received terminate request, exiting ..."
                    )
                    break
                elif msg["payload"] == "set_ir":
                    # update the root_ir for subsequent requests
                    logger.debug(f"Worker {worker.device_id} received set_ir request")
                    new_ir_data = msg["data"]

                    root_ir = worker.set_new_ir(new_ir_data)

                    if root_ir is None:
                        logger.error(f"Worker {worker.device_id} failed to set new IR")

                elif msg["payload"] == "process":
                    logger.debug(
                        f"Worker {worker.device_id} received processing request {msg['exp_id']}"
                    )
                    with tracing.span(
                        "tuning.request",
                        lambda: {"request_id": msg["exp_id"]},
                        flush_after=True,
                    ):
                        with tracing.span("ir.clone"):
                            request_ir = root_ir.clone()

                        exp = worker.process_payload(
                            request_ir, ExperimentConfiguration.from_dict(msg["data"])
                        )
                    logger.debug(
                        f"Worker {worker.device_id} finalized processing request {msg['exp_id']}"
                    )

                    response_q.put(
                        {
                            "exp_id": msg["exp_id"],
                            "payload": "result",
                            "data": exp.to_dict(),
                            "llvm_ir": "",
                        }
                    )
                else:
                    logger.warning(f"Received unknown message {msg}")

        return
