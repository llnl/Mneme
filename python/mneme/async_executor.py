import queue
import threading
import time
from enum import IntEnum
from multiprocessing import Event as ProcessEvent
from multiprocessing import Process
from multiprocessing import Queue as ProcessQueue
from pathlib import Path
from queue import Queue as ThreadQueue
from threading import Event as ThreadEvent
from typing import Callable, Dict, Optional, Union

from mneme.futures import EvalFuture
from mneme.mneme_logging import logger
from mneme.replay_executor import TuneWorker
from mneme.mneme_types import ExperimentConfiguration, ExperimentResult


def pop(q, timeout):
    """
    Pop an item from a queue with a timeout.

    This helper provides a uniform interface for both thread-based queues
    (:class:`queue.Queue`) and multiprocessing queues
    (:class:`multiprocessing.Queue`). If the queue is empty at the end of the
    timeout, ``None`` is returned.

    Parameters
    ----------
    q
        Queue-like object providing a ``get(timeout=...)`` API.
    timeout : float
        Timeout in seconds for the blocking ``get`` call.

    Returns
    -------
    Any or None
        The retrieved item, or ``None`` if the queue was empty.
    """
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


class TuneWorkerHandle:
    """
    Thread-side controller for one worker process executing tuning experiments.

    ``TuneWorkerHandle`` owns:
      - A single worker :class:`multiprocessing.Process` running :meth:`TuneWorker.run`.
      - A pair of IPC queues for requests/responses.
      - A monitoring thread that drives a small state machine for submitting jobs
        and receiving results.
      - Crash detection and automatic worker respawn.

    The handle consumes :class:`EvalFuture` objects from a shared thread queue
    (``global_q``), forwards their configurations to the worker process, and
    resolves each future when the corresponding result arrives.

    Notes
    -----
    * Each handle pins its worker process to a specific device id (GPU affinity is
      handled inside :class:`BaseExecutor` / :class:`TuneWorker`).
    * Crash recovery is best-effort: if the worker dies while running an experiment,
      the active future is marked as failed and the worker is restarted.
    """

    class StateMachine(IntEnum):
        """
        Internal action state for the monitor loop.

        SUBMIT
            Attempt to dequeue a new job from the global queue and send it to the worker.
        RECEIVE
            Poll for a worker response and resolve the currently active future.
        """

        SUBMIT = 1
        RECEIVE = 2

    def __init__(
        self,
        idx,
        global_q: ThreadQueue,
        record_db: str,
        record_id: str,
        device_id: int,
        iterations: int,
        results_db_dir: str,
        warmup: int = 2,
        max_startup_failures: int = 3,
        on_startup_failure_limit: Optional[
            Callable[["TuneWorkerHandle", str], None]
        ] = None,
        short_circuit: bool = False,
    ):
        """
        Construct a worker handle and start the worker process + monitor thread.

        Parameters
        ----------
        idx : int
            Logical worker index (primarily used for logging/debugging).
        global_q : queue.Queue
            Shared thread queue containing :class:`EvalFuture` objects to be executed
            by this handle’s worker process.
        record_db : str
            Path to the recorded execution database/file.
        record_id : str
            Identifier of the recorded kernel instance inside ``record_db``.
        device_id : int
            Device id (GPU index) assigned to the underlying worker process.
        iterations : int
            Number of kernel iterations used by the worker for the tracked execution.
        results_db_dir : str
            Directory where the worker writes logs and optional artifacts.
        warmup : int
            Number of warmup iterations executed before measured iterations.
        max_startup_failures : int
            Maximum number of consecutive worker exits before readiness before this
            handle stops respawning the worker.
        short_circuit : bool
            Whether to skip timing runs when verification fails.

        Notes
        -----
        * The worker process is spawned immediately during initialization.
        * A background thread is started to monitor the worker process and drive job
          submission/result collection.
        """
        self.idx = idx
        self.global_q = global_q

        self._ipc_write_q = None
        self._ipc_read_q = None

        self._shutdown_event = ThreadEvent()
        self._action = self.StateMachine.SUBMIT

        self.record_db = record_db
        self.record_id = record_id
        self.device_id = device_id
        self.iterations = iterations
        self.results_db_dir = results_db_dir
        self.warmup = warmup
        self.short_circuit = short_circuit
        self.max_startup_failures = max_startup_failures
        self._startup_failures = 0
        self._on_startup_failure_limit = on_startup_failure_limit

        self._state = None  # ProcessEvent
        self._process = None  # Process
        self.current = None  # EvalFuture
        logger.debug(f"[TuneWorkerHandle] Starting processes")
        self._spawn_process()

        logger.debug(
            f"[TuneWorkerHandle] Starting Thread and bind it to monitor process {self._process.pid}"
        )
        self._monitor_thread = threading.Thread(target=self._shadow_process_loop)
        self._monitor_thread.start()
        logger.debug(
            f"[TuneWorkerHandle] Done Launching TunerWorkerHandles Thread and Processing Infrastructure"
        )

    def _spawn_process(self):
        """
        Spawn (or respawn) the underlying worker process and IPC infrastructure.

        This method creates fresh IPC queues, a new readiness event, and launches the
        worker process using :meth:`TuneWorker.run`. The internal action state is
        reset to ``SUBMIT``.

        Notes
        -----
        * This is used both at startup and during crash recovery.
        * Any in-flight job must be handled by the caller before respawning.
        """
        self._state = ProcessEvent()
        self._ipc_write_q = ProcessQueue()
        self._ipc_read_q = ProcessQueue()

        self._process = Process(
            target=TuneWorker.run,
            args=(
                self._ipc_write_q,
                self._ipc_read_q,
                self.record_db,
                self.record_id,
                self.device_id,
                self.iterations,
                self.results_db_dir,
                self._state,
                self.warmup,
                self.short_circuit,
            ),
            daemon=False,
        )
        self._process.start()
        self._action = self.StateMachine.SUBMIT
        self._ir_revision = 0

    def _startup_failure_error(self):
        return (
            f"Worker failed to start after {self._startup_failures} attempts "
            f"on device {self.device_id}"
        )

    def _disable_after_startup_failures(self):
        error = self._startup_failure_error()
        logger.error(error)
        self._process.join()
        if self._on_startup_failure_limit is not None:
            self._on_startup_failure_limit(self, error)

    # ------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------
    def _process_result(self, msg):
        """
        Resolve the current in-flight future using a response message.

        Parameters
        ----------
        msg : dict
            Response message produced by :meth:`TuneWorker.run`. Expected fields:
            * ``exp_id`` – experiment id matching the active future
            * ``data`` – serialized :class:`ExperimentResult` dict

        Raises
        ------
        RuntimeError
            If a result is received for an unexpected experiment id.
        """
        if self.current is None:
            return

        if self.current.job_id != msg["exp_id"]:
            raise RuntimeError(
                f"Worker {self.idx} received result for unexpected job "
                f"{msg['exp_id']} vs {self.current.job_id}"
            )
        logger.debug(
            f"[{self.__class__.__name__}-{self.device_id}] finished experiment {self.current.job_id}"
        )
        self.current.set_result(ExperimentResult.from_dict(msg["data"]))
        self.current = None

    def _try_receive(self):
        """
        Poll the worker response queue and process one available result.

        If no message is available within the polling timeout, this method returns
        without modifying state. On a successful receive, the internal action state
        transitions back to ``SUBMIT``.
        """
        msg = pop(self._ipc_read_q, timeout=1)
        if msg is None:
            return

        self._process_result(msg)
        self._action = self.StateMachine.SUBMIT

    # ------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------
    def _submit(self):
        """
        Submit one job to the worker process if one is available.

        This method dequeues a single :class:`EvalFuture` from the shared global
        queue, sends its configuration to the worker process, and marks it as the
        current in-flight job. The internal action state transitions to ``RECEIVE``.

        Notes
        -----
        * At most one job is in-flight per worker handle at any time.
        """
        future: EvalFuture = pop(self.global_q, timeout=1)
        if future is None:
            return

        self.current = future
        if future.ir_revision != self._ir_revision:
            self._ipc_write_q.put({"payload": "set_ir", "data": future.ir_data})
            self._ir_revision = future.ir_revision

        msg = {
            "payload": "process",
            "data": future.config.to_dict(),
            "exp_id": future.job_id,
        }

        self._ipc_write_q.put(msg)
        self._action = self.StateMachine.RECEIVE

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    def _shadow_process_loop(self):
        """
        Background monitor loop that drives the worker state machine.

        This loop runs in a dedicated thread and performs:

          1) **Crash detection and recovery**
             If the worker process dies, the current in-flight job (if any) is
             marked as failed and the worker is respawned.

          2) **Worker readiness waiting**
             Before sending work, the loop waits for the process-side readiness
             event to be set.

          3) **Submit/Receive alternation**
             A simple state machine ensures that only one job is in-flight:
             ``SUBMIT`` sends a job, then ``RECEIVE`` polls for the result.

          4) **Graceful shutdown**
             When the shutdown event is set, the loop sends a ``terminate`` message
             to the worker, drains results while the process is alive, and joins
             the process.

        Notes
        -----
        * The loop uses polling timeouts to remain responsive to shutdown.
        * The readiness event is a simple synchronization primitive; future versions
          may use condition variables or a richer handshake protocol.
        """
        while not self._shutdown_event.is_set():
            if not self._process.is_alive():
                # Crash recovery
                if not self._state.is_set():
                    self._startup_failures += 1
                    if self._startup_failures >= self.max_startup_failures:
                        self._disable_after_startup_failures()
                        return
                elif self.current is not None:
                    # TODO: At some point we need to have more descrptive messages on the crash, it is not always a seg fault, somethimes it is LLVM related and we should know.
                    self.current.set_error(
                        f"Worker crashed (exit code: {self._process.exitcode}) running on device {self.device_id}"
                    )
                    self.current = None
                    _ = pop(self._ipc_read_q, timeout=0)

                self._spawn_process()
                continue

            # Wait for worker to initialize
            if not self._state.is_set():
                # TODO: We should use mp.conditional variables instead of the state.
                # That should simplify the logic.
                time.sleep(0.5)
                continue

            self._startup_failures = 0

            if self._action == self.StateMachine.SUBMIT:
                self._submit()
            else:
                self._try_receive()

        # Shutdown
        if self._process.is_alive():
            self._ipc_write_q.put({"payload": "terminate"})
            while self._process.is_alive():
                self._try_receive()
            self._process.join()

    def join(self):
        """
        Request shutdown of the monitor thread and wait for completion.

        This method signals the monitor loop to terminate, which triggers graceful
        worker shutdown and process join. It then joins the monitor thread.
        """
        self._shutdown_event.set()
        self._monitor_thread.join()


# ============================================================================
# AsyncReplayExecutor
# ============================================================================
class AsyncReplayExecutor:
    """
    Asynchronous record/replay executor backed by a pool of worker processes.

    ``AsyncReplayExecutor`` provides a lightweight interface to evaluate
    :class:`ExperimentConfiguration` objects using one or more worker processes.
    Internally it manages:

      - A global thread queue of pending :class:`EvalFuture` jobs.
      - A set of :class:`TuneWorkerHandle` instances (one per worker process).
      - A monotonic job id generator for mapping submissions to results.

    Users may submit jobs asynchronously via :meth:`submit`, or synchronously
    evaluate a configuration via :meth:`evaluate` (submit + wait).

    Notes
    -----
    * Each worker handle can execute at most one in-flight job at a time.
    * The executor is intended for repeated evaluations; startup/teardown overhead
      may dominate for microbenchmarks.
    """

    def __init__(
        self,
        record_db: str,
        record_id: str,
        iterations: int,
        results_db_dir: str,
        num_workers: int,
        warmup: int = 2,
        max_startup_failures: int = 3,
        short_circuit: bool = False,
    ):
        """
        Construct an asynchronous executor with a fixed-size worker pool.

        Parameters
        ----------
        record_db : str
            Path to the recorded execution database/file.
        record_id : str
            Identifier of the recorded kernel instance inside ``record_db``.
        iterations : int
            Number of kernel iterations performed by each worker per tracked run.
        results_db_dir : str
            Directory where workers write logs and optional output artifacts.
        num_workers : int
            Number of worker processes to launch.
        warmup : int
            Number of warmup iterations each worker executes before measured iterations.
        max_startup_failures : int
            Maximum number of consecutive pre-readiness startup failures allowed per
            worker before that worker stops respawning.
        short_circuit : bool
            Whether to skip timing runs when verification fails.
        """
        self.global_q = ThreadQueue()
        self._futures: Dict[int, EvalFuture] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._ir_revision = 0
        self._ir_data = None
        self.iterations = iterations
        self.warmup = warmup
        self.short_circuit = short_circuit
        self.max_startup_failures = max_startup_failures
        self._num_workers = num_workers
        self._failed_workers = set()
        self._broken_error = None

        self.workers = []
        for i in range(num_workers):
            self.workers.append(
                TuneWorkerHandle(
                    i,
                    self.global_q,
                    record_db,
                    record_id,
                    i,
                    iterations,
                    results_db_dir,
                    warmup=warmup,
                    max_startup_failures=max_startup_failures,
                    on_startup_failure_limit=self._handle_startup_failure_limit,
                    short_circuit=short_circuit,
                )
            )

    def _handle_startup_failure_limit(self, worker: TuneWorkerHandle, error: str):
        with self._lock:
            self._failed_workers.add(worker.idx)
            if len(self._failed_workers) < self._num_workers:
                return
            self._broken_error = error

        self._fail_pending_futures(error)

    def _fail_pending_futures(self, error: str):
        while True:
            future = pop(self.global_q, timeout=0)
            if future is None:
                return
            future.set_error(error)

    def set_ir(self, ir: Union[str, Path]):
        """Use this LLVM IR for evaluations submitted after this call.

        Parameters
        ----------
        ir : str | Path
            Path to the LLVM IR file (.bc or .ll) or the IR as a string.
        """
        if isinstance(ir, Path):
            ir_data = str(ir.absolute())
        else:
            ir_data = ir

        with self._lock:
            self._ir_revision += 1
            self._ir_data = ir_data

    # ------------------------------------------------------------------
    # Submit new job (non-blocking)
    # ------------------------------------------------------------------
    def submit(self, config: ExperimentConfiguration) -> EvalFuture:
        """
        Submit a new experiment configuration for asynchronous evaluation.

        The configuration is wrapped in an :class:`EvalFuture` and enqueued for
        execution by the first available worker handle.

        Parameters
        ----------
        config : ExperimentConfiguration
            Experiment configuration to evaluate.

        Returns
        -------
        EvalFuture
            A future that will be resolved with an :class:`ExperimentResult` once
            the worker completes the experiment (or marked as failed on crash).
        """
        with self._lock:
            job_id = self._next_id
            self._next_id += 1
            logger.debug(f"[{self.__class__.__name__}] Submitting job {job_id}")
            future = EvalFuture(job_id, config, self._ir_revision, self._ir_data)
            self._futures[job_id] = future
            if self._broken_error is not None:
                future.set_error(self._broken_error)
                return future
            self.global_q.put(future)
        return future

    def shutdown(self):
        """
        Gracefully shutdown all workers and their monitoring threads.

        This method requests each :class:`TuneWorkerHandle` to stop, which causes:
          - the worker loop to receive a terminate message,
          - the worker process to exit,
          - the monitor thread to join.

        Notes
        -----
        * After shutdown, submitting additional jobs is undefined behavior.
        """
        logger.debug(f"[{self.__class__.__name__}] Starting shutdown process")
        for w in self.workers:
            w.join()
        logger.debug(f"[{self.__class__.__name__}] Done Shutdown")

    def evaluate(self, config: ExperimentConfiguration) -> ExperimentResult:
        """
        Synchronously evaluate one configuration through the worker pool.

        This convenience method submits a configuration and blocks until the
        corresponding :class:`EvalFuture` completes.

        Parameters
        ----------
        config : ExperimentConfiguration
            Experiment configuration to evaluate.

        Returns
        -------
        ExperimentResult
            Result object containing verification status, execution time samples,
            and optional compilation/resource metrics, depending on worker settings.
        """
        future = self.submit(config)

        return future.result()
