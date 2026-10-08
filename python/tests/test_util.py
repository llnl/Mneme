import time
from unittest.mock import patch, MagicMock

from mneme.utils import cond_time, cond_gpu_time


# ---------------------------------------------------------------------------
# Helper objects
# ---------------------------------------------------------------------------


class DummyConfig:
    pass


class DummyRes:
    pass


class DummySelf:
    pass


# ---------------------------------------------------------------------------
# cond_time tests
# ---------------------------------------------------------------------------


def test_cond_time_no_profile_calls_original_fn():
    """If profile=False, the decorated function behaves normally."""

    @cond_time("duration")
    def fn(self, res, config):
        return 42

    config = DummyConfig()
    res = DummyRes()
    result = fn(DummySelf(), res, config, profile=False)

    assert result == 42
    assert not hasattr(res, "duration")


def test_cond_time_profile_sets_attribute():
    """If profile=True, execution time is measured and stored."""

    @cond_time("duration")
    def fn(self, res, config):
        return "OK"

    config = DummyConfig()
    res = DummyRes()

    # Patch time.perf_counter for deterministic testing
    with patch("mneme.utils.time.perf_counter", side_effect=[1.0, 3.0]):
        result = fn(DummySelf(), res, config, profile=True)

    assert result == "OK"
    assert res.duration == 2.0  # end-start


# ---------------------------------------------------------------------------
# cond_gpu_time tests
# ---------------------------------------------------------------------------


def test_cond_gpu_time_no_profile_calls_original():
    """GPU timing is skipped when profile=False."""

    @cond_gpu_time("gpu_time")
    def fn(self, res, config, kernel_name, device_func, iterations):
        return 7

    config = DummyConfig()
    res = DummyRes()

    with patch("mneme.utils.gpu_profile_start") as start, patch(
        "mneme.utils.gpu_profile_stop"
    ) as stop:

        out = fn(DummySelf(), res, config, "kernel", None, 3, profile=False)

    assert out == 7
    start.assert_not_called()
    stop.assert_not_called()
    assert not hasattr(res, "gpu_time")


def test_cond_gpu_time_profile_records_gpu_measurements():
    """GPU start/stop functions are invoked and result stored on exp."""

    @cond_gpu_time("gpu_time")
    def fn(self, res, config, kernel_name, device_func, iterations):
        return "DONE"

    config = DummyConfig()
    res = DummyRes()

    with patch("mneme.utils.gpu_profile_start", return_value="CID") as start, patch(
        "mneme.utils.gpu_profile_stop", return_value={"cycles": 123}
    ) as stop:

        out = fn(DummySelf(), res, config, "mykernel", None, 3, profile=True)

    assert out == "DONE"
    start.assert_called_once_with("mykernel.kd")
    stop.assert_called_once_with("CID", 3)
    assert res.gpu_time == {"cycles": 123}
