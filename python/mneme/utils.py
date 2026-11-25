import json
import time
from dataclasses import fields
from functools import wraps

from mneme.profile import (
    counter_metric_name,
    gpu_counter_start,
    gpu_counter_stop,
    gpu_profile_start,
    gpu_profile_stop,
)


def cond_time(attr_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, result, config, *args, profile: bool = False, **kwargs):
            if not profile:
                return func(self, result, config, *args, **kwargs)
            start = time.perf_counter()
            fresult = func(self, result, config, *args, **kwargs)
            end = time.perf_counter()
            setattr(result, attr_name, end - start)
            return fresult

        return wrapper

    return decorator


def cond_gpu_time(attr_name):
    def decorator(func):
        @wraps(func)
        def wrapper(
            self, result, config, kernel_name, device_func, iterations,
            profile: bool = False,
        ):
            args = (self, result, config, kernel_name, device_func, iterations)
            if not profile:
                return func(*args)
            correlation_id = gpu_profile_start(kernel_name)
            fresult = func(*args)
            measurements = gpu_profile_stop(correlation_id, iterations)
            setattr(result, attr_name, measurements)
            return fresult

        return wrapper

    return decorator


def cond_gpu_counter(attr_name):
    """
    Wrap a replay pass in a hardware-counter session (initial version).

    With ``counters=True`` the wrapped method runs between ``gpu_counter_start``
    and ``gpu_counter_stop``. The per-launch counter values are stored on
    ``result`` under ``attr_name`` and printed, together with how long the
    session setup, the pass and the teardown took. Otherwise the method runs
    unchanged. This must be the last pass of an experiment: while a counter
    session is open the Activity-based timing receives no kernel records.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(
            self,
            result,
            config,
            kernel_name,
            device_func,
            iterations,
            *args,
            counters: bool = False,
            **kwargs,
        ):
            if not counters:
                return func(
                    self, result, config, kernel_name, device_func, iterations,
                    *args, **kwargs,
                )
            t0 = time.perf_counter()
            gpu_counter_start(self.device_id, iterations + 4)
            t1 = time.perf_counter()
            try:
                fresult = func(
                    self, result, config, kernel_name, device_func, iterations,
                    *args, **kwargs,
                )
            except BaseException:
                try:
                    gpu_counter_stop(iterations)
                except RuntimeError:
                    pass
                raise
            t2 = time.perf_counter()
            values = gpu_counter_stop(iterations)
            t3 = time.perf_counter()
            setattr(result, attr_name, values)
            print(
                f"[mneme] counters for {kernel_name}: "
                f"{counter_metric_name()} = {values}"
            )
            print(
                f"[mneme] counter pass: start {1e3 * (t1 - t0):.2f} ms "
                f"(includes one-time setup on the first call), "
                f"run {1e3 * (t2 - t1):.2f} ms, stop {1e3 * (t3 - t2):.2f} ms"
            )
            return fresult

        return wrapper

    return decorator


class MnemeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
