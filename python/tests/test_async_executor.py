import inspect
import queue
import threading

import pytest

from mneme.mneme_types import (
    ExperimentConfiguration,
    ExperimentResult,
    VerificationResult,
)
from mneme.futures import EvalFuture
from mneme.replay_runs import RunChain, TimingRun, VerificationRun, default_run_chain
from mneme.async_executor import (
    pop,
    TuneWorkerHandle,
    AsyncReplayExecutor,
    _WorkQueue,
)


# -----------------------------------------------------------------------------
# pop()
# -----------------------------------------------------------------------------


def test_pop_returns_item():
    q = queue.Queue()
    q.put(42)
    assert pop(q, timeout=0.01) == 42


def test_pop_returns_none_on_timeout():
    q = queue.Queue()
    assert pop(q, timeout=0.01) is None


def test_work_queue_shutdown_wakes_all_waiters(monkeypatch):
    work = _WorkQueue()
    waiting = queue.Queue()
    results = queue.Queue()
    original_wait = work._available.wait

    def observe_wait(timeout=None):
        waiting.put(threading.get_ident())
        return original_wait(timeout)

    def consume():
        results.put(work.get(timeout=30))

    monkeypatch.setattr(work._available, "wait", observe_wait)
    threads = [threading.Thread(target=consume, daemon=True) for _ in range(3)]
    for thread in threads:
        thread.start()
    assert len({waiting.get(timeout=5) for _ in threads}) == len(threads)

    work.shutdown()

    for thread in threads:
        assert results.get(timeout=5) is None
        thread.join(timeout=5)
        assert not thread.is_alive()


# -----------------------------------------------------------------------------
# TuneWorkerHandle internals (NO real processes)
# -----------------------------------------------------------------------------


class DummyProcess:
    def __init__(self):
        self.exitcode = 1
        self._alive = True
        self.pid = 123

    def is_alive(self):
        return self._alive

    def start(self):
        pass

    def join(self):
        self._alive = False


class DeadProcess(DummyProcess):
    def __init__(self):
        super().__init__()
        self._alive = False


class DummyEvent:
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self):
        return self._set


class DummyQueue(queue.Queue):
    pass


class NoopThread:
    def __init__(self, target=None, args=(), kwargs=None):
        self._target = target

    def start(self):
        # Do not run anything
        return

    def join(self, timeout=None):
        return


@pytest.fixture
def handle(monkeypatch):
    import mneme.async_executor as mod

    # Prevent background thread from running
    monkeypatch.setattr(mod.threading, "Thread", NoopThread, raising=True)

    # Patch multiprocessing bits as before
    monkeypatch.setattr(mod, "Process", lambda *a, **k: DummyProcess())
    monkeypatch.setattr(mod, "ProcessEvent", DummyEvent)
    monkeypatch.setattr(mod, "ProcessQueue", DummyQueue)

    global_q = _WorkQueue()

    h = mod.TuneWorkerHandle(
        idx=0,
        global_q=global_q,
        record_db="db",
        record_id="rid",
        device_id=0,
        results_db_dir="/tmp",
        run_chain=default_run_chain(iterations=3),
    )
    return h


def test_process_result_sets_future(handle):
    future = EvalFuture(7, ExperimentConfiguration())
    handle.current = future

    # A failed verification without build stats still arrives as executed.
    sent = ExperimentResult([VerificationResult(verified=False)], executed=False)
    msg = {
        "exp_id": 7,
        "data": sent.to_dict(),
    }

    handle._process_result(msg)

    assert future.done()
    assert future.result() == sent
    assert future.result().verification is not None
    assert future.result().timing is None
    assert handle.current is None


def test_process_result_wrong_id_sets_error(handle):
    future = EvalFuture(1, ExperimentConfiguration())
    handle.current = future

    msg = {"exp_id": 999, "data": {}}

    out = future.result(timeout=0.01)
    with pytest.raises(RuntimeError):
        handle._process_result(msg)
    assert handle.current is future
    assert future.done() is False


def test_submit_moves_state_and_sets_current(handle):
    cfg = ExperimentConfiguration()
    future = EvalFuture(3, cfg)
    handle.global_q.put(future)

    handle._submit()

    assert handle.current is future
    assert handle._action == handle.StateMachine.RECEIVE
    assert handle._ipc_write_q.get_nowait()["payload"] == "process"


def test_submit_installs_future_ir_revision(handle):
    future = EvalFuture(
        3, ExperimentConfiguration(), ir_revision=1, ir_data="replacement ir"
    )
    handle.global_q.put(future)

    handle._submit()

    assert handle._ipc_write_q.get_nowait() == {
        "payload": "set_ir",
        "data": "replacement ir",
    }
    assert handle._ipc_write_q.get_nowait()["payload"] == "process"
    assert handle._ir_revision == 1


def test_try_receive_no_message(handle):
    handle._ipc_read_q = queue.Queue()
    handle._action = handle.StateMachine.RECEIVE

    handle._try_receive()

    # state unchanged
    assert handle._action == handle.StateMachine.RECEIVE


def test_try_receive_processes_message(handle):
    future = EvalFuture(5, ExperimentConfiguration())
    handle.current = future

    handle._ipc_read_q = queue.Queue()
    handle._ipc_read_q.put(
        {
            "exp_id": 5,
            "data": ExperimentResult(executed=True).to_dict(),
        }
    )

    handle._action = handle.StateMachine.RECEIVE
    handle._try_receive()

    assert future.done()
    assert handle._action == handle.StateMachine.SUBMIT
    assert handle.current is None


# -----------------------------------------------------------------------------
# Crash recovery path
# -----------------------------------------------------------------------------


def test_worker_crash_marks_future_failed(handle):
    future = EvalFuture(1, ExperimentConfiguration())
    handle.current = future

    handle._process._alive = False
    handle._process.exitcode = 42

    # Run one iteration of the loop body manually
    if not handle._process.is_alive():
        handle.current.set_error(
            f"Worker crashed (exit code: {handle._process.exitcode}) running on device {handle.device_id}"
        )
        handle.current = None

    out = future.result(timeout=0.01)
    assert isinstance(out, ExperimentResult)
    assert out.failed is True
    assert out.executed is False
    assert "Worker crashed" in out.error


def test_worker_crash_branch_marks_future_failed_and_respawns(handle):
    future = EvalFuture(1, ExperimentConfiguration())
    handle.current = future
    handle._state.set()
    handle._process._alive = False
    handle._process.exitcode = 42

    def stop_after_respawn():
        handle.global_q.shutdown()

    handle._spawn_process = stop_after_respawn

    handle._shadow_process_loop()

    out = future.result(timeout=0.01)
    assert out.failed is True
    assert out.executed is False
    assert "Worker crashed" in out.error
    assert handle.current is None


def test_ready_worker_resets_startup_failures_and_submits(handle):
    future = EvalFuture(3, ExperimentConfiguration())
    handle.global_q.put(future)
    handle._startup_failures = 2
    handle._state.set()

    original_submit = handle._submit
    original_receive = handle._try_receive

    def submit_once():
        original_submit()
        assert handle.current is future
        assert handle._action == handle.StateMachine.RECEIVE
        handle._ipc_read_q.put(
            {"exp_id": future.job_id, "data": ExperimentResult(executed=True).to_dict()}
        )
        handle.global_q.shutdown()

    def receive_once():
        assert handle.current is future
        original_receive()

    handle._submit = submit_once
    handle._try_receive = receive_once

    handle._shadow_process_loop()

    assert handle._startup_failures == 0
    assert future.done()


def test_idle_shutdown_sends_terminate_without_waiting_for_a_result(handle):
    handle.global_q.shutdown()

    def unexpected_receive():
        pytest.fail("Idle shutdown tried to read from the response queue")

    handle._try_receive = unexpected_receive

    handle._shadow_process_loop()

    assert handle._ipc_write_q.get_nowait()["payload"] == "terminate"
    assert handle._process.is_alive() is False


@pytest.mark.parametrize("result_waiting", [True, False])
def test_shutdown_resolves_current_future_after_worker_exit(handle, result_waiting):
    future = EvalFuture(7, ExperimentConfiguration())
    handle.current = future
    handle._process._alive = False
    if result_waiting:
        handle._ipc_read_q.put(
            {"exp_id": future.job_id, "data": ExperimentResult(executed=True).to_dict()}
        )
    handle.global_q.shutdown()

    handle._shadow_process_loop()

    assert future.done()
    assert future.result().executed is result_waiting
    assert future.result().failed is not result_waiting


def test_startup_failures_stop_respawning_after_limit(monkeypatch):
    import mneme.async_executor as mod

    monkeypatch.setattr(mod.threading, "Thread", NoopThread, raising=True)
    monkeypatch.setattr(mod, "Process", lambda *a, **k: DeadProcess())
    monkeypatch.setattr(mod, "ProcessEvent", DummyEvent)
    monkeypatch.setattr(mod, "ProcessQueue", DummyQueue)

    failures = []
    h = mod.TuneWorkerHandle(
        idx=0,
        global_q=_WorkQueue(),
        record_db="db",
        record_id="rid",
        device_id=0,
        results_db_dir="/tmp",
        run_chain=default_run_chain(iterations=3),
        max_startup_failures=3,
        on_startup_failure_limit=lambda worker, error: failures.append(
            (worker.idx, error)
        ),
    )

    h._shadow_process_loop()

    assert h._startup_failures == 3
    assert failures
    assert failures[0][0] == 0
    assert "failed to start after 3 attempts" in failures[0][1]


def test_executor_does_not_fail_pending_work_until_all_workers_fail(monkeypatch):
    import mneme.async_executor as mod

    monkeypatch.setattr(mod.threading, "Thread", NoopThread, raising=True)
    monkeypatch.setattr(mod, "Process", lambda *a, **k: DeadProcess())
    monkeypatch.setattr(mod, "ProcessEvent", DummyEvent)
    monkeypatch.setattr(mod, "ProcessQueue", DummyQueue)

    exe = mod.AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=2,
        max_startup_failures=3,
    )
    future = exe.submit(ExperimentConfiguration())

    exe.workers[0]._shadow_process_loop()
    assert future.done() is False

    exe.workers[1]._shadow_process_loop()
    out = future.result(timeout=0.01)

    assert out.failed is True
    assert out.executed is False
    assert "failed to start after 3 attempts" in out.error


def test_submit_after_executor_broken_returns_failed_future(monkeypatch):
    monkeypatch.setattr("mneme.async_executor.TuneWorkerHandle", lambda *a, **k: None)

    exe = AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=0,
    )
    exe._broken_error = "all workers failed to start"

    future = exe.submit(ExperimentConfiguration())
    out = future.result(timeout=0.01)

    assert out.failed is True
    assert out.executed is False
    assert out.error == "all workers failed to start"


# -----------------------------------------------------------------------------
# AsyncReplayExecutor
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pass_run_chain", [False, True], ids=["keywords", "run_chain"])
@pytest.mark.parametrize("short_circuit", [False, True])
def test_async_executor_preserves_run_chain_on_worker_respawn(
    monkeypatch, short_circuit, pass_run_chain
):
    import mneme.async_executor as mod

    worker_options = []

    def make_process(target, args, daemon):
        options = inspect.signature(target).bind(*args).arguments
        worker_options.append(options["run_chain"])
        return DummyProcess()

    monkeypatch.setattr(mod.threading, "Thread", NoopThread, raising=True)
    monkeypatch.setattr(mod, "Process", make_process)
    monkeypatch.setattr(mod, "ProcessEvent", DummyEvent)
    monkeypatch.setattr(mod, "ProcessQueue", DummyQueue)

    run_chain = default_run_chain(iterations=3, short_circuit=short_circuit)
    run_options = (
        {"run_chain": run_chain}
        if pass_run_chain
        else {"iterations": 3, "short_circuit": short_circuit}
    )
    exe = mod.AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        results_db_dir="/tmp",
        num_workers=1,
        **run_options,
    )
    assert (exe.iterations, exe.warmup, exe.short_circuit) == (3, 2, short_circuit)
    worker = exe.workers[0]
    worker._state.set()
    worker._process._alive = False
    spawn_process = worker._spawn_process

    def stop_after_respawn():
        spawn_process()
        worker.global_q.shutdown()
        worker._process._alive = False

    monkeypatch.setattr(worker, "_spawn_process", stop_after_respawn)
    worker._shadow_process_loop()

    assert worker_options == [run_chain, run_chain]


@pytest.mark.parametrize(
    "run_options, message",
    [
        ({"run_chain": default_run_chain(), "iterations": 3}, "not both"),
        ({"run_chain": default_run_chain(), "warmup": 0}, "not both"),
        ({"run_chain": default_run_chain(), "short_circuit": False}, "not both"),
        ({}, "needs run_chain or iterations"),
    ],
)
def test_async_executor_rejects_ambiguous_run_options(run_options, message):
    with pytest.raises(ValueError, match=message):
        AsyncReplayExecutor(
            record_db="db", record_id="rid", results_db_dir="/tmp", num_workers=0,
            **run_options,
        )


@pytest.mark.parametrize(
    "run, missing_kind, attribute",
    [
        (VerificationRun(), "timing", "iterations"),
        (VerificationRun(), "timing", "warmup"),
        (TimingRun(), "verification", "short_circuit"),
    ],
)
def test_async_executor_run_attributes_need_their_run(run, missing_kind, attribute):
    exe = AsyncReplayExecutor(
        record_db="db", record_id="rid", results_db_dir="/tmp", num_workers=0,
        run_chain=RunChain((run,)),
    )

    with pytest.raises(AttributeError, match=f"no {missing_kind} run"):
        getattr(exe, attribute)


def test_async_executor_submit(monkeypatch):
    monkeypatch.setattr("mneme.async_executor.TuneWorkerHandle", lambda *a, **k: None)

    exe = AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=0,
    )

    cfg = ExperimentConfiguration()
    fut = exe.submit(cfg)

    assert isinstance(fut, EvalFuture)
    assert fut.job_id == 0


def test_async_executor_evaluate(monkeypatch):
    monkeypatch.setattr("mneme.async_executor.TuneWorkerHandle", lambda *a, **k: None)

    exe = AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=0,
    )

    # fake future
    res = ExperimentResult(executed=True)
    fut = EvalFuture(0, ExperimentConfiguration())
    fut.set_result(res)

    monkeypatch.setattr(exe, "submit", lambda cfg: fut)

    out = exe.evaluate(ExperimentConfiguration())
    assert out.executed is True


def test_async_executor_set_ir_only_affects_subsequent_work(monkeypatch):
    monkeypatch.setattr("mneme.async_executor.TuneWorkerHandle", lambda *a, **k: None)

    exe = AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=0,
    )

    old_future = exe.submit(ExperimentConfiguration())
    exe.set_ir("define void @kernel() { ret void }")
    new_future = exe.submit(ExperimentConfiguration())

    assert old_future.ir_revision == 0
    assert old_future.ir_data is None
    assert new_future.ir_revision == 1
    assert new_future.ir_data == "define void @kernel() { ret void }"


def test_async_executor_set_ir_normalizes_path(monkeypatch, tmp_path):
    monkeypatch.setattr("mneme.async_executor.TuneWorkerHandle", lambda *a, **k: None)

    exe = AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=0,
    )
    ir_path = tmp_path / "kernel.ll"

    exe.set_ir(ir_path)
    future = exe.submit(ExperimentConfiguration())

    assert future.ir_data == str(ir_path.absolute())


def test_async_executor_shutdown(monkeypatch):
    joined = []

    class FakeHandle:
        def __init__(self, idx, global_q, *args, **kwargs):
            self.global_q = global_q

        def join(self):
            assert self.global_q.shutdown_requested.is_set()
            assert queued.done()
            joined.append(True)

    monkeypatch.setattr("mneme.async_executor.TuneWorkerHandle", FakeHandle)

    exe = AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=2,
    )
    queued = exe.submit(ExperimentConfiguration())
    waiting = threading.Event()
    original_wait = queued._cond.wait
    results = queue.Queue()

    def observe_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(queued._cond, "wait", observe_wait)
    waiter = threading.Thread(target=lambda: results.put(queued.result()), daemon=True)
    waiter.start()
    assert waiting.wait(timeout=5)

    exe.shutdown()

    assert len(joined) == 2
    result = results.get(timeout=5)
    waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert result.failed
    assert result.error == "Executor shut down before the job started"

    late = exe.submit(ExperimentConfiguration())
    assert late.done()
    assert late.result().error == result.error
    assert exe.global_q.drain() == []


def test_worker_process_receives_pass_plugins(monkeypatch):
    import mneme.async_executor as mod

    spawned = []

    def make_process(target=None, args=(), daemon=None):
        spawned.append(args)
        return DummyProcess()

    monkeypatch.setattr(mod.threading, "Thread", NoopThread, raising=True)
    monkeypatch.setattr(mod, "Process", make_process)
    monkeypatch.setattr(mod, "ProcessEvent", DummyEvent)
    monkeypatch.setattr(mod, "ProcessQueue", DummyQueue)

    pass_plugins = ["/path/libMyPass.so"]
    mod.AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=2,
        pass_plugins=pass_plugins,
    )

    assert [args[-1] for args in spawned] == [pass_plugins, pass_plugins]
