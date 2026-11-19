# tests/test_eval_future.py

import threading
import time
import pytest

from mneme.mneme_types import ExperimentConfiguration, ExperimentResult
from mneme.futures import EvalFuture


def test_done_initially_false():
    fut = EvalFuture(job_id=1, config=ExperimentConfiguration())
    assert fut.done() is False


@pytest.mark.parametrize("failed", [False, True])
def test_notify_when_done_signals_early_and_late_registrations(failed):
    fut = EvalFuture(job_id=1, config=ExperimentConfiguration())
    early = threading.Event()
    fut.notify_when_done(early)
    assert not early.is_set()

    if failed:
        fut.set_error("worker failed")
    else:
        fut.set_result(ExperimentResult(executed=True))

    late = threading.Event()
    fut.notify_when_done(late)
    assert early.is_set()
    assert late.is_set()


def test_result_timeout_returns_none_when_not_done():
    fut = EvalFuture(job_id=1, config=ExperimentConfiguration())
    t0 = time.time()
    out = fut.result(timeout=0.05)
    dt = time.time() - t0

    assert out is None
    # Loose guard to ensure we actually waited a bit (avoid flakiness)
    assert dt >= 0.04
    assert fut.done() is False


def test_set_result_unblocks_and_returns_result():
    fut = EvalFuture(job_id=1, config=ExperimentConfiguration())
    res = ExperimentResult(exec_time=[1, 2, 3], executed=True, verified=True)

    def worker():
        time.sleep(0.05)
        fut.set_result(res)

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    out = fut.result(timeout=1.0)
    assert out is res
    assert fut.done() is True


def test_set_error_without_result_returns_error_result_and_does_not_raise():
    """
    Behavior per current implementation:
      - if _error is set and _result is None, result() returns an ExperimentResult(...)
        and does NOT raise.
    """
    fut = EvalFuture(job_id=1, config=ExperimentConfiguration())

    def worker():
        time.sleep(0.05)
        fut.set_error("boom")

    threading.Thread(target=worker, daemon=True).start()

    out = fut.result(timeout=1.0)
    assert isinstance(out, ExperimentResult)
    assert out.error == "boom"
    assert out.failed is True
    assert out.executed is False
    assert fut.done() is True


def test_set_error_after_result_raises_and_attaches_error_to_result():
    fut = EvalFuture(job_id=1, config=ExperimentConfiguration())
    res = ExperimentResult(exec_time=[42], executed=True, verified=False)

    def worker():
        time.sleep(0.05)
        fut.set_result(res)
        time.sleep(0.01)
        fut.set_error("kaboom")

    threading.Thread(target=worker, daemon=True).start()

    # Wait until the result is set first, then call result() after error arrives.
    time.sleep(0.10)

    with pytest.raises(RuntimeError, match="kaboom"):
        fut.result(timeout=1.0)

    # Error should be attached to the stored result.
    assert res.error == "kaboom"
    assert fut.done() is True


def test_result_returns_immediately_when_already_done():
    fut = EvalFuture(job_id=1, config=ExperimentConfiguration())
    res = ExperimentResult(exec_time=[5], executed=True)
    fut.set_result(res)

    t0 = time.time()
    out = fut.result(timeout=1.0)
    dt = time.time() - t0

    assert out is res
    assert dt < 0.05
