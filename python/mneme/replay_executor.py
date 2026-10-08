"""
mneme.replay_executor

Core record–replay execution pipeline for Mneme.

This module provides the execution backbone used by both:
  * the synchronous CLI execution path (via BaseExecutor subclasses), and
  * the asynchronous tuning engine (via TuneWorker).

At a high level, an "experiment" in Mneme is:
  1) Load a recorded kernel execution (RecordedExecution + KernelInstance).
  2) Reconstruct the recorded GPU memory state (prologue/epilogue snapshots)
     into a managed virtual address space (PageManagerRef).
  3) Link recorded LLVM IR modules into a single IR module suitable for replay.
  4) Apply optional IR specializations (arguments, launch dims, launch bounds).
  5) Run an optimization pipeline and generate a device object.
  6) Load the object onto the GPU, run the kernel, and optionally profile.
  7) Verify correctness by comparing epilogue vs prologue expectations.

The pipeline is intentionally organized so that:
  * verification can be done with minimal instrumentation,
  * tracked runs collect timing/resource metrics, and
  * worker processes can amortize initialization costs by reusing a single executor.

Public API
----------
BaseExecutor:
  Base class that owns GPU affinity, recorded state, and the build/run pipeline.

TuneWorker:
  Worker-process implementation used by the async tuning infrastructure.
"""

import os
from dataclasses import replace
from datetime import datetime, timezone
from multiprocessing import Event, Queue
from pathlib import Path
from typing import Optional, Tuple

from mneme.device import (
    DeviceFunction,
    DeviceModule,
    get_device_arch,
    get_device_count,
    set_device,
)
from mneme.llvm._lib_path_config import MNEME_PASS_PLUGIN_LIB
from mneme.llvm.buffer import MemBufferRef
from mneme.llvm.module import ModuleRef, parse_assembly, parse_bitcode
from mneme.mneme_logging import logger
from mneme.mneme_types import ExperimentConfiguration, ExperimentResult
from mneme.page_manager import PageManagerRef
from mneme.profile import init_profiler
from mneme.proteus import jit
from mneme.recorded_execution import RecordedExecution, MemStateRef
from mneme.utils import cond_gpu_time, cond_time


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
      * and an iteration count for measured runs.

    Responsibilities
    ---------------
    * Load the recorded execution metadata (RecordedExecution) and select the
      target KernelInstance (kernel_descr).
    * Pin the current OS process to a specific GPU device (set_device()).
    * Manage the replay address space and recorded snapshots:
        - PageManagerRef selects/initializes the virtual address space.
        - prologue/epilogue snapshots are opened and later compared.
    * Provide a structured pipeline that takes IR -> object -> execution:
        - _preprocess_ir(): apply specialization transforms and compute a variant hash
        - _optimize(): run pass pipeline / O-level selection
        - _codegen(): lower to a device object (MemBufferRef)
        - _run(): load object, resolve kernel, execute and optionally profile
        - _execute(): orchestrate verification + cleanup + tracked run

    Lifecycle
    ---------
    BaseExecutor is designed to be used as a context manager:

        executor = MyExecutor(record_db=..., record_id=..., device_id=...)
        root_ir = executor.link_ir()
        with executor:
            res = executor.execute(...)

    The context manager ensures GPU memory state (snapshots + page manager) is
    opened exactly once and released even when execution raises.

    Notes / invariants
    ------------------
    * A BaseExecutor instance is intended to be used within a single process.
      (Workers should construct one executor per worker process.)
    * open() must be called before any execution; _execute() assumes prologue and
      epilogue states are loaded.
    * link_ir() returns a linked IR module representing the recorded kernel; callers
      should clone before mutation if reusing across experiments.
    """

    def __init__(
        self,
        record_db: str = "",
        record_id: Optional[str] = None,
        iterations: int = 3,
        device_id: int = 0,
        warmup: int = 2,
    ):
        self.record_db = record_db
        self.device_id = device_id
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
        self._page_manager = None
        self._iterations = iterations
        self._warmup = warmup
        self.num_devices = get_device_count()
        set_device(device_id)
        logger.debug(
            f"GPU Affinity of process was set to device:{self.device_id} out of {self.num_devices}"
        )
        jit.register_pass_plugin(MNEME_PASS_PLUGIN_LIB)

    def open(self):
        # Note the 'executor' allocates all resources and picks address space.
        self._page_manager = PageManagerRef(
            self.device_id, self.records.va_addr, self.records.va_size
        )
        self._prologue = self.kernel_descr.prologue.open()
        self._epilogue = self.kernel_descr.epilogue.open()
        # No kernel has run yet, so this asks whether doing nothing would verify.
        self.noop_verifies = self._prologue.matches_unlaunched(self._epilogue)
        return self

    @property
    def prologue(self):
        return self._prologue

    @property
    def epilogue(self):
        return self._epilogue

    def close(self):
        if self._epilogue is not None:
            self._epilogue.close()
            self._epilogue = None
        if self._prologue is not None:
            self._prologue.close()
            self._prologue = None
        if self._page_manager is not None:
            self._page_manager.close()
            self._page_manager = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def link_ir(self):
        return self.records.link_llvm_modules(prune=True, internalize=True)

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

    @cond_time("preprocess_ir_time")
    def _preprocess_ir(
        self,
        result: ExperimentResult,
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
        result : ExperimentResult
            The experiment result object that may be updated during preprocessing.
            (Currently unused directly, but modified by the decorator (``cond_time``)).
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

    @cond_time("opt_time")
    def _optimize(
        self,
        result: ExperimentResult,
        config: ExperimentConfiguration,
        ir_module: ModuleRef,
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
        result : ExperimentResult
            The experiment result object. Although not modified directly in this
            method, it is modified by the decorator (``cond_time``).
        config : ExperimentConfiguration
            Configuration controlling the optimization pipeline. Relevant fields
            include:
            * ``passes`` – Name or specification of the optimization pass pipeline.
            * ``codegen_opt`` – Code generation optimization level.
        ir_module : ModuleRef
            The intermediate representation to be optimized. The module is mutated
            in-place by the underlying JIT subsystem.

        Notes
        -----
        * Optimization routines are delegated to ``jit.optimize``.
        * The optimization phase typically precedes code generation and may
          significantly affect both performance and final code size.
        * This method does not return a value; the IR module is modified directly.

        """
        jit.optimize(ir_module, self.device_arch, config.passes, config.codegen_opt)

    @cond_time("codegen_time")
    def _codegen(
        self,
        result: ExperimentResult,
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
        result : ExperimentResult
            Experiment result object. Not modified directly by this method, but
            modified from the decorator (``cond_time``).
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

    @cond_gpu_time("exec_time")
    def _run_kernel(
        self,
        result: ExperimentResult,
        config: ExperimentConfiguration,
        kernel_name: str,
        device_func: DeviceFunction,
        iterations: int,
    ) -> None:
        """
        Execute the kernel on the device using the provided launch configuration.

        This method invokes the device-level profiling interface to run the kernel
        for the specified number of iterations. Launch parameters (grid, block, and
        shared memory), as well as the recorded prologue and epilogue states, are
        passed directly to the device runtime. Profiling results—such as execution
        times—are captured internally by the device function object and propagated
        into the associated :class:`ExperimentResult`.

        Parameters
        ----------
        result : ExperimentResult
            The result object that accumulates execution metrics. Although this
            method does not write to it directly, profiling performed by the device
            backend updates fields that will later be reflected in the result through
            the ``cond_gpu_time`` decorator.
        config : ExperimentConfiguration
            Experiment configuration specifying launch parameters, shared-memory
            requirements, and specialization settings.
        kernel_name: str
            The name of the kernel to be executed. The parameter is not directly used by the function,
            but it is actually used by the ``decorator``.
        device_func : DeviceFunction
            The device-side kernel entry point obtained from the compiled module.
            Must support the ``profile`` interface for execution and timing.
        iterations : int
            Number of times the kernel should be executed. Typically more than one
            iteration is used for performance characterization and variance analysis.

        Notes
        -----
        * Actual execution and profiling of the kernel is handled by
          ``device_func.profile``.
        * Both prologue and epilogue states are forwarded to the device runtime so
          that Mneme’s record–replay mechanism can validate kernel behavior and
          collect replay-specific metrics.
        * Errors raised by the device runtime will propagate upward to the caller.

        """
        device_func.profile(
            config.grid,
            config.block,
            self._prologue._state,
            self._epilogue._state,
            config.shared_mem,
            iterations,
        )

    def _build(
        self,
        result: ExperimentResult,
        config: ExperimentConfiguration,
        ir_module: ModuleRef,
        track: bool,
    ) -> MemBufferRef:
        """
        Build the executable device kernel from the given IR module.

        This method runs the full compilation pipeline on the provided IR module:
        preprocessing, optimization, and final code generation. The resulting
        device-ready binary is returned as a :class:`MemBufferRef`. When tracking
        is enabled, additional metadata such as object size is recorded into the
        provided :class:`ExperimentResult`.

        The build process consists of the following stages:

        1. **IR preprocessing**
           Applies specialization, dimension-dependent transformations, and optional
           launch-bound insertion. This step updates the internal code hash and
           prepares the IR for optimization.

        2. **Optimization**
           Runs the configured optimization passes (e.g., ``O3`` or user-defined
           pipelines). If profiling is enabled, optimization timing is recorded in
           the experiment result.

        3. **Code generation**
           Lowers the optimized IR into a device-executable artifact. The resulting
           binary is wrapped in a :class:`MemBufferRef`. When ``track=True``,
           the size of the generated object code is stored in ``result.obj_size``.

        Parameters
        ----------
        result : ExperimentResult
            Result object to be populated with build metrics such as optimization
            time and object size.
        config : ExperimentConfiguration
            Configuration controlling specialization, optimization pipeline, and
            code-generation strategy.
        ir_module : ModuleRef
            Intermediate representation on which the build pipeline operates.
            The module may be transformed during preprocessing and optimization.
        track : bool
            Whether to collect profiling information and resource-usage statistics
            during the build process.

        Returns
        -------
        MemBufferRef
            A memory buffer containing the compiled device module produced by the
            code-generation stage.

        Notes
        -----
        * This method does not execute the kernel; execution occurs in :meth:`_run`.
        * Tracking is optional but recommended when performance analysis is needed.
        """
        self._preprocess_ir(result, config, ir_module, profile=track)
        self._optimize(result, config, ir_module, profile=track)
        mem_buffer = self._codegen(result, config, ir_module, profile=track)
        if track:
            result.obj_size = mem_buffer.get_size()
        return mem_buffer

    def _run(
        self,
        result: ExperimentResult,
        config: ExperimentConfiguration,
        mem_buffer: MemBufferRef,
        prologue: MemStateRef,
        epilogue: MemStateRef,
        verify: bool,
        track: bool,
        iterations: int,
    ):
        """
        Execute a compiled kernel on the device and optionally collect resource-usage
        metadata.

        This method loads the device module from the provided memory buffer, extracts
        the kernel function, and executes it for the requested number of iterations.
        When ``track`` is enabled, the kernel launch is profiled and register usage,
        local memory usage, and constant memory usage are recorded into the provided
        :class:`ExperimentResult` object.

        Parameters
        ----------
        result : ExperimentResult
            Result object that will be populated with execution metrics and resource
            usage information.
        config : ExperimentConfiguration
            The configuration controlling launch parameters such as grid, block,
            specialization settings, and shared-memory use.
        mem_buffer : MemBufferRef
            Memory buffer containing the device-side compiled module from which the
            kernel function is loaded.
        track : bool
            If ``True``, profiling and resource-usage tracking are enabled for the
            kernel execution. This populates register usage, constant memory usage,
            and local memory usage in the experiment result.
        iterations : int
            Number of times the kernel should be executed. Typically more than one
            run is used when statistical accuracy is required.

        Notes
        -----
        * The device module is managed via a context manager to ensure allocation and
          cleanup follow the device runtime’s requirements.
        * Resource usage fields are only updated when ``track=True``.
        * Actual execution is delegated to :meth:`_run_kernel`.

        ------
        """
        with DeviceModule.from_MemBuffer(mem_buffer) as DeviceObj:
            device_func = DeviceObj.get_function(self.kernel_descr.kernel_name)
            self._run_kernel(
                result,
                config,
                self.kernel_descr.kernel_name,
                device_func,
                iterations,
                profile=track,
            )
            if verify:
                result.verified = prologue == epilogue
                result.noop_verifies = self.noop_verifies
            if track:
                result.reg_usage = device_func.reg_usage
                result.const_mem_usage = device_func.const_mem
                result.local_mem_usage = device_func.local_mem

    def _execute(
        self,
        result: ExperimentResult,
        config: ExperimentConfiguration,
        ir_module: ModuleRef,
    ) -> ModuleRef:
        """
        Execute a single Mneme experiment using the given configuration and IR module.

        This method performs the full record/replay experiment pipeline, including
        verification, IR cleanup, code generation, and timed execution. It returns
        both the populated experiment result and the transformed IR module.

        The execution consists of three stages:

        1. **Verification pass**
           A clone of the input IR module is built and executed once without
           instrumentation. This ensures the recorded prologue and epilogue states
           match, allowing the system to validate kernel determinism and correctness.

        2. **IR sanitization**
           The tracked build prepends the ``purge-autoinit`` pass to the optimization
           pipeline, removing automatically inserted Clang initialization code. Only
           IR regions explicitly marked by Clang are removed to avoid disturbing user
           code.

        3. **Instrumented execution**
           The cleaned up version of the kernel is built with tracking enabled.
           The kernel is executed ``iterations + 2`` times to allow downstream
           statistical metrics to be computed reliably. Execution time, resource
           usage, and other experiment metrics are accumulated into the resulting
           :class:`ExperimentResult`.

        Parameters
        ----------
        result : ExperimentResult
            The container to store all the collected/counted values to.
        config : ExperimentConfiguration
            The experiment configuration controlling launch parameters, specialization,
            and code generation settings.
        ir_module : ModuleRef
            The LLVM-like intermediate representation module on which the experiment
            is executed. The module is cloned and the ``ir_module`` is not modified.

        Returns
        -------
        (ExperimentResult, ModuleRef)
            A tuple containing:

            * **ExperimentResult** – Populated result object containing verification
              status, execution metrics, and profiling data.
            * **ModuleRef** – The transformed IR module after auto-initialization
              removal and other modifications performed during execution.

        Raises
        ------
        RuntimeError
            If internal prologue or epilogue state is unexpectedly ``None``.
        """
        if self._prologue._state is None or self._epilogue._state is None:
            raise RuntimeError("States should never be none when executing a kernel")

        # NOTE: 1. First we need to verify.
        ver_mod = ir_module.clone()
        mem_buffer = self._build(result, config, ver_mod, False)
        self._run(result, config, mem_buffer, self.prologue, self.epilogue, True, False, 1)

        # NOTE: 2-3. Build and run with tracking on. The tracked build prepends the
        # purge-autoinit pass (registered via Proteus' JIT pass plugin API) to strip
        # the Clang-inserted auto-init memsets; verification above ran without it.
        ir_module = ir_module.clone()
        tracked_config = replace(config, passes=f"purge-autoinit,{config.passes}")
        mem_buffer = self._build(result, tracked_config, ir_module, True)
        self._run(result, config, mem_buffer, self.prologue, self.epilogue, False, True, self._iterations + self._warmup)
        result.executed = True

        return ir_module


class TuneWorker(BaseExecutor):
    """
    Worker-side executor used by the asynchronous tuning infrastructure.

    ``TuneWorker`` is a concrete :class:`BaseExecutor` specialization intended to run
    inside a dedicated worker process. It owns the GPU affinity, prologue/epilogue
    state, page manager, and JIT pipeline required to compile and replay a recorded
    kernel under a given :class:`ExperimentConfiguration`.

    A worker process typically:
      1) Initializes profiling and selects a GPU device.
      2) Loads the recorded execution (record DB + record ID).
      3) Links the recorded LLVM IR into a single module (``link_ir``).
      4) Enters a message-processing loop (see :meth:`run`) to evaluate configurations.

    Notes
    -----
    * The public entry point for the worker process is :meth:`run`, which is designed
      to be used as a multiprocessing target.
    * Per-request execution is handled by :meth:`process_payload`, which builds,
      verifies, and runs the kernel according to the provided configuration.
    """

    def __init__(self, *args, **kwargs):
        """
        Construct a TuneWorker and initialize worker-local profiling.

        This constructor initializes the Mneme profiler (for timing breakdowns) and
        then delegates initialization to :class:`BaseExecutor`. The base class sets
        device affinity, loads the recorded execution, and prepares prologue/epilogue
        descriptors.

        Notes
        -----
        * The worker process should typically construct a single TuneWorker instance
          and reuse it for multiple requests to amortize startup overhead.
        * ``init_profiler()`` is required to be executed once by every OS process executing it
            multiple times results to undefined behavior.
        """
        init_profiler()
        super().__init__(*args, **kwargs)

    def process_payload(
        self, ir_module, config: ExperimentConfiguration
    ) -> Tuple[ExperimentResult, ModuleRef]:
        """
        Execute one tuning request: build, verify, and run the kernel under ``config``.

        This method is the unit of work performed by a worker in response to a tuning
        request. It executes the full Mneme record–replay pipeline using the provided
        IR module and configuration:

          1) Records the experiment start timestamp.
          2) Invokes the base executor pipeline (see :meth:`BaseExecutor._execute`),
             which performs verification, IR sanitization, compilation, and timed execution.
          3) Records the experiment end timestamp and annotates the result with the GPU id.
          4) Returns both the populated :class:`ExperimentResult` and the transformed IR.

        Parameters
        ----------
        ir_module : ModuleRef
            Root IR module (or clone) used as input for this experiment. The module
            is cloned internally and transformed as part of the execution pipeline.
        config : ExperimentConfiguration
            Configuration describing launch parameters, specialization options, and
            code-generation controls for this experiment.

        Returns
        -------
        (ExperimentResult, ModuleRef)
            A tuple containing:

            * **ExperimentResult** – result object populated with timing, verification
              status, and device resource usage.
            * **ModuleRef** – the transformed IR module after preprocessing and
              auto-initialization removal.

        Notes
        -----
        * This method is expected to be called repeatedly within the worker loop;
          callers should pass a cloned IR module to avoid cross-experiment mutation.
        * Timestamps are recorded in ISO 8601 format using UTC time.
        """
        result = ExperimentResult()
        result.start_time = datetime.now(timezone.utc).isoformat()
        generated_ir = super()._execute(result, config, ir_module)
        result.end_time = datetime.now(timezone.utc).isoformat()
        result.gpu_id = self.device_id
        return result, generated_ir

    @staticmethod
    def run(
        request_q: Queue,
        response_q: Queue,
        record_db: str,
        record_id: str,
        device_id: int,
        iterations: int,
        results_db_dir: str,
        state: Event,
        warmup: int = 2,
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
        iterations : int
            Number of kernel iterations to execute during the tracked run (the full
            execution may include additional runs for verification/warmup depending
            on the executor pipeline).
        warmup : int
            Number of warmup iterations to execute before measured iterations.
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
            iterations=iterations,
            warmup=warmup,
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
                    exp, ir = worker.process_payload(
                        root_ir.clone(), ExperimentConfiguration.from_dict(msg["data"])
                    )
                    # final = resdb.save_ir(ir, exp.hash())
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
