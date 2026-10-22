"""test the trace interface"""

import json
import os
import subprocess
import sys
import textwrap

import pytest


def run_script(tmp_path, script, *, enabled=True):

    env = os.environ.copy()
    env.pop("MNEME_TRACE_DIR", None)
    
    if enabled:
        env["MNEME_TRACE_DIR"] = str(tmp_path / "traces")
    
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def read_traces(tmp_path):
    # traces are in dir and all json
    return [json.loads(path.read_text()) for path in (tmp_path / "traces").glob("*.json")]


def test_disabled_tracing_is_identity_and_does_not_evaluate_metadata(tmp_path):
    run_script(tmp_path, """
        from mneme import tracing
        assert not tracing.enabled
        assert tracing._native().MnemePy_TraceTimestamp() == 0

        def unexpected(*args):
            raise AssertionError("disabled tracing performed work")

        def function(value):
            return value + 1

        tracing._native = unexpected
        assert tracing.traced("test", attributes=unexpected)(function) is function
        with tracing.span("test", unexpected, flush_after=True) as event:
            event.annotate(answer=42)
        tracing.flush()
        assert function(2) == 3
    """, enabled=False)
    assert not (tmp_path / "traces").exists()


def test_nested_events_incremental_flush_clock_and_thread_ids(tmp_path):
    result = run_script(tmp_path, r'''
        import json
        import os
        from pathlib import Path
        import threading
        import time
        from mneme import tracing

        before = time.monotonic_ns()
        with tracing.span('outer "escaped"\\name', lambda: {"text": "line\n雪"}):
            with tracing.span("inner") as event:
                time.sleep(0.002)
                event.annotate(answer=42)
        tracing.flush()
        path, = Path(os.environ["MNEME_TRACE_DIR"]).glob("*.json")
        first = path.read_bytes()
        assert json.loads(first)
        tracing.flush()
        assert path.read_bytes() == first

        def worker():
            with tracing.span("thread", lambda: {"native_id": threading.get_native_id()}):
                pass

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        # Exercise native emission on the same clock as Python scopes.
        lib = tracing._native()
        start = lib.MnemePy_TraceTimestamp()
        time.sleep(0.001)
        end = lib.MnemePy_TraceTimestamp()
        lib.MnemePy_TraceComplete(b"native", start, end, b'{"native":true}')
        tracing.flush()
        assert path.read_bytes().startswith(first[:-2])
        print(json.dumps({"before": before, "after": time.monotonic_ns(),
                          "pid": os.getpid(), "tid": threading.get_native_id()}))
    ''')
    
    bounds = json.loads(result.stdout)
    events, = read_traces(tmp_path)
    slices = {event["name"]: event for event in events if event["ph"] == "X"}
    
    outer = slices['outer "escaped"\\name']
    inner = slices["inner"]
    
    assert outer["args"]["text"] == "line\n雪"
    assert inner["args"] == {"answer": 42}
    assert inner["dur"] >= 1000  # Durations and timestamps are microseconds.
    assert outer["ts"] <= inner["ts"]
    assert inner["ts"] + inner["dur"] <= outer["ts"] + outer["dur"] + 0.01
    assert slices["native"]["args"]["native"] is True
    
    for name in (outer["name"], "inner", "native"):
        event = slices[name]
        assert event["pid"] == bounds["pid"]
        assert event["tid"] == bounds["tid"]
        assert bounds["before"] / 1000 <= event["ts"] <= bounds["after"] / 1000
    
    assert slices["thread"]["tid"] == slices["thread"]["args"]["native_id"]
    assert slices["thread"]["tid"] != bounds["tid"]
    assert sum(e["name"] == "process_name" for e in events) == 1
    assert sum(e["name"] == "thread_name" for e in events) == 2


def test_exception_flush_preserves_application_error(tmp_path):
    run_script(tmp_path, """
        from mneme import tracing

        @tracing.traced("failure", flush_after=True)
        def fail():
            with tracing.span("inner"):
                raise ValueError("application error")

        try:
            fail()
        except ValueError as error:
            assert str(error) == "application error"
        else:
            raise AssertionError("application error was swallowed")
    """)
    
    events, = read_traces(tmp_path)
    errors = [event for event in events if event["name"] in ("failure", "inner")]
    
    assert len(errors) == 2
    assert all(event["args"]["error"] == "ValueError" for event in errors)


def test_output_failure_warns_once_and_keeps_application_running(tmp_path):
    (tmp_path / "traces").write_text("a file cannot be an output directory")
    
    result = run_script(tmp_path, """
        from mneme import tracing
        for _ in range(3):
            with tracing.span("event", flush_after=True):
                pass
        print("application completed")
    """)
    
    assert result.stdout.strip() == "application completed"
    assert result.stderr.count("mneme: tracing disabled:") == 1


def test_metadata_failure_disables_both_layers_without_masking_errors(tmp_path):
    result = run_script(tmp_path, """
        from mneme import tracing
        def bad_metadata():
            raise ValueError("bad metadata")
        with tracing.span("event", bad_metadata, flush_after=True):
            pass
        assert not tracing.enabled
        assert tracing._native().MnemePy_TraceTimestamp() == 0
        print("application completed")
    """)
    
    assert result.stdout.strip() == "application completed"
    assert result.stderr.count("mneme: tracing disabled:") == 1
    assert not read_traces(tmp_path)


def test_serialization_failure_preserves_completed_batches(tmp_path):
    result = run_script(tmp_path, """
        from mneme import tracing
        with tracing.span("saved", flush_after=True):
            pass
        with tracing.span("invalid", lambda: {"value": object()}, flush_after=True):
            pass
        assert not tracing.enabled
        assert tracing._native().MnemePy_TraceTimestamp() == 0
        print("application completed")
    """)
    
    assert result.stdout.strip() == "application completed"
    assert result.stderr.count("mneme: tracing disabled:") == 1
    
    events, = read_traces(tmp_path)
    assert "saved" in {event["name"] for event in events}
    assert "invalid" not in {event["name"] for event in events}


@pytest.mark.parametrize("extra_bytes", [0, 100])
def test_partial_write_failure_preserves_completed_batches(tmp_path, extra_bytes):
    result = run_script(tmp_path, """
        import os
        from pathlib import Path
        import resource
        import signal
        from mneme import tracing

        with tracing.span("saved", flush_after=True):
            pass
        path, = Path(os.environ["MNEME_TRACE_DIR"]).glob("*.json")
        completed = path.read_bytes()

        # Allow the append to overwrite the terminator before hitting EFBIG.
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        resource.setrlimit(
            resource.RLIMIT_FSIZE, (len(completed) + EXTRA_BYTES, hard)
        )
        for _ in range(2):
            with tracing.span(
                "discarded", lambda: {"payload": "x" * 4096}, flush_after=True
            ):
                pass

        assert tracing._native().MnemePy_TraceTimestamp() == 0
        assert path.read_bytes() == completed
        print("application completed")
    """.replace("EXTRA_BYTES", str(extra_bytes)))

    assert result.stdout.strip() == "application completed"
    assert result.stderr.count("mneme: tracing disabled:") == 1
    events, = read_traces(tmp_path)
    assert "saved" in {event["name"] for event in events}
    assert "discarded" not in {event["name"] for event in events}


def test_fork_resets_native_state_and_avoids_inherited_python_lock(tmp_path):
    run_script(tmp_path, """
        import os
        import threading
        from mneme import tracing
        from mneme.llvm import ffi

        with tracing.span("parent.saved", flush_after=True):
            pass
        ready = threading.Event()
        release = threading.Event()
        def holder():
            with ffi.lib._lock:
                ready.set()
                release.wait()
        thread = threading.Thread(target=holder)
        thread.start()
        ready.wait()
        with tracing.span("parent.pending"):
            pid = os.fork()
            if pid == 0:
                with tracing.span("child", flush_after=True):
                    pass
                os._exit(0)
            release.set()
            thread.join()
            _, status = os.waitpid(pid, 0)
            assert status == 0
        tracing.flush()
    """)
    
    traces = read_traces(tmp_path)
    assert len(traces) == 2
    
    names = [{e["name"] for e in events} for events in traces]
    parent = next(group for group in names if "parent.saved" in group)
    child = next(group for group in names if "child" in group)
    
    assert "parent.pending" in parent
    assert "parent.saved" not in child and "parent.pending" not in child


def test_restart_preserves_completed_batches_and_uses_a_new_file(tmp_path):
    run_script(tmp_path, """
        import os
        from mneme import tracing
        with tracing.span("completed", flush_after=True):
            pass
        with tracing.span("unflushed"):
            pass
        os._exit(0)
    """)
    
    run_script(tmp_path, """
        from mneme import tracing
        with tracing.span("restarted", flush_after=True):
            pass
    """)
    
    traces = read_traces(tmp_path)
    assert len(traces) == 2
    assert len({events[0]["pid"] for events in traces}) == 2
    
    names = {event["name"] for events in traces for event in events}
    assert {"completed", "restarted"} <= names
    assert "unflushed" not in names
