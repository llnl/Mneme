from unittest.mock import patch

from mneme.profile import GpuKernelTimer


def test_gpu_kernel_timer_records_gpu_measurements():
    """GPU start/stop functions bracket the block and the measurements are kept."""

    with patch("mneme.profile.gpu_profile_start", return_value="CID") as start, patch(
        "mneme.profile.gpu_profile_stop", return_value=[5, 6, 7]
    ) as stop:
        with GpuKernelTimer("mykernel", 3) as timer:
            start.assert_called_once_with("mykernel")
            stop.assert_not_called()

    stop.assert_called_once_with("CID", 3)
    assert timer.measurements == [5, 6, 7]
