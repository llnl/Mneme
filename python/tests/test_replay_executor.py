# tests/test_replay_executor.py

import importlib
import queue
import types
from dataclasses import dataclass
import pytest


MODULE_PATH = "mneme.replay_executor"


# --------------------------
# Fakes / Test doubles
# --------------------------


class FakeSnapshot:
    """
    Snapshot returned by kernel_descr.prologue.open() / epilogue.open().
    Must have: .close(), .args, .num_args, ._state
    """

    def __init__(self, state="STATE", args=None, num_args=0, noop_verifies=False):
        self._state = state
        self.args = args if args is not None else []
        self.num_args = num_args
        self.noop_verifies = noop_verifies
        self.closed = False

    def close(self):
        self.closed = True

    def matches_unlaunched(self, other):
        return self.noop_verifies


class FakePrologueDescr:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def open(self):
        return self._snapshot


class FakeEpilogueDescr:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def open(self):
        return self._snapshot


class FakeKernelDescr:
    def __init__(self, kernel_name="K", static_hash="H", prologue=None, epilogue=None):
        self.kernel_name = kernel_name
        self.static_hash = static_hash
        self.available_specializations = {"dummy": True}
        self.prologue = FakePrologueDescr(prologue or FakeSnapshot())
        self.epilogue = FakeEpilogueDescr(epilogue or FakeSnapshot())


class FakeModule:
    def __init__(self, name="m", clone_counter=None):
        self.name = name
        self._clone_counter = clone_counter if clone_counter is not None else {"n": 0}

    def clone(self):
        self._clone_counter["n"] += 1
        # clones share counter for observability
        return FakeModule(self.name + "_clone", self._clone_counter)


class FakeMemBuffer:
    def __init__(self, size=123):
        self._size = size

    def get_size(self):
        return self._size

class FakeMemStateRef:
    def __init__(self):
        pass

    def open(self):
        return self

    def __enter__(self):
        return self.open()

    def __eq__(self, other):
        return True

class FakeDeviceFunction:
    def __init__(self):
        self.profile_calls = []
        self.reg_usage = 17
        self.const_mem = 33
        self.local_mem = 44

    def profile(self, grid, block, pro_state, epi_state, shared_mem, iterations):
        self.profile_calls.append(
            (grid, block, pro_state, epi_state, shared_mem, iterations)
        )


class FakeDeviceModule:
    def __init__(self, device_func=None, kernel_name=None):
        self._device_func = device_func or FakeDeviceFunction()
        self._kernel_name = kernel_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_function(self, kernel_name):
        # could validate name here if you want
        return self._device_func


class FakeRecordedExecution:
    def __init__(self, kernel_descr=None, kernel_instances=None):
        self.kernel_instances = (
            {"rid": kernel_descr} if kernel_instances is None else kernel_instances
        )
        self.link_calls = []

    @classmethod
    def from_json(cls, record_db):
        raise RuntimeError("You must monkeypatch from_json in tests")

    def __getitem__(self, record_id):
        return self.kernel_instances[record_id]

    def __iter__(self):
        return iter(self.kernel_instances)

    def __len__(self):
        return len(self.kernel_instances)

    def link_llvm_modules(self, prune=True, internalize=True):
        self.link_calls.append((prune, internalize))
        return FakeModule("linked")


# --------------------------
# Helper: import module with patched decorators
# --------------------------


def _reload_with_identity_decorators(monkeypatch):
    """
    cond_time/cond_gpu_time are decorators imported into the module namespace at import time.
    For testability we reload the module after patching mneme.utils.cond_time/cond_gpu_time
    to identity decorators.
    """
    # Patch mneme.utils BEFORE importing replay_executor
    utils_mod = importlib.import_module("mneme.utils")

    def identity_decorator(_field):
        def deco(fn):
            # Accept and ignore 'profile' kwarg; BaseExecutor._build passes it.
            def wrapper(*args, **kwargs):
                kwargs.pop("profile", None)
                return fn(*args, **kwargs)

            return wrapper

        return deco

    monkeypatch.setattr(utils_mod, "cond_time", identity_decorator, raising=True)
    monkeypatch.setattr(utils_mod, "cond_gpu_time", identity_decorator, raising=True)

    # Now reload target module so the decorators wrap with the identity versions.
    mod = importlib.import_module(MODULE_PATH)
    mod = importlib.reload(mod)
    monkeypatch.setattr(mod.jit, "register_pass_plugin", lambda path: None, raising=True)
    return mod


def _patch_executor_init_dependencies(monkeypatch, mod, records):
    monkeypatch.setattr(
        mod.RecordedExecution,
        "from_json",
        staticmethod(lambda _: records),
        raising=True,
    )
    monkeypatch.setattr(mod, "set_device", lambda _: None, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", lambda: "sm", raising=True)
    monkeypatch.setattr(mod, "get_device_count", lambda: 1, raising=True)


# --------------------------
# Tests
# --------------------------


def test_baseexecutor_init_sets_gpu_affinity_and_loads_records(monkeypatch):
    mod = _reload_with_identity_decorators(monkeypatch)

    calls = {"set_device": [], "get_arch": 0, "get_count": 0, "from_json": []}

    def fake_set_device(d):
        calls["set_device"].append(d)

    def fake_get_arch():
        calls["get_arch"] += 1
        return "sm_80"

    def fake_get_count():
        calls["get_count"] += 1
        return 8

    kernel = FakeKernelDescr()
    rec = FakeRecordedExecution(kernel)

    def fake_from_json(path):
        calls["from_json"].append(path)
        return rec

    monkeypatch.setattr(mod, "set_device", fake_set_device, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", fake_get_arch, raising=True)
    monkeypatch.setattr(mod, "get_device_count", fake_get_count, raising=True)
    monkeypatch.setattr(
        mod.RecordedExecution, "from_json", staticmethod(fake_from_json), raising=True
    )

    ex = mod.BaseExecutor(
        record_db="db.json", record_id="rid", device_id=3, iterations=5
    )

    assert ex.records is rec
    assert ex.record_id == "rid"
    assert ex.kernel_descr is kernel
    assert ex.device_arch == "sm_80"
    assert ex.num_devices == 8
    assert calls["set_device"] == [3]
    assert calls["from_json"] == ["db.json"]
    assert ex._iterations == 5


def test_baseexecutor_infers_single_record_id(monkeypatch):
    mod = _reload_with_identity_decorators(monkeypatch)

    kernel = FakeKernelDescr()
    rec = FakeRecordedExecution(kernel_instances={"only-rid": kernel})
    _patch_executor_init_dependencies(monkeypatch, mod, rec)

    ex = mod.BaseExecutor(record_db="db.json")

    assert ex.record_id == "only-rid"
    assert ex.kernel_descr is kernel


@pytest.mark.parametrize(
    ("record_ids", "expected_count"),
    [([], 0), (["first", "second"], 2)],
)
def test_baseexecutor_rejects_ambiguous_record_id(
    monkeypatch, record_ids, expected_count
):
    mod = _reload_with_identity_decorators(monkeypatch)

    kernel_instances = {record_id: FakeKernelDescr() for record_id in record_ids}
    rec = FakeRecordedExecution(kernel_instances=kernel_instances)
    monkeypatch.setattr(
        mod.RecordedExecution,
        "from_json",
        staticmethod(lambda _: rec),
        raising=True,
    )
    monkeypatch.setattr(
        mod,
        "get_device_arch",
        lambda: pytest.fail("GPU initialization should not be reached"),
        raising=True,
    )

    with pytest.raises(ValueError, match=f"found {expected_count}"):
        mod.BaseExecutor(record_db="db.json")


def test_baseexecutor_uses_explicit_record_id_with_multiple_instances(monkeypatch):
    mod = _reload_with_identity_decorators(monkeypatch)

    first = FakeKernelDescr(kernel_name="first")
    selected = FakeKernelDescr(kernel_name="selected")
    rec = FakeRecordedExecution(
        kernel_instances={"first-rid": first, "selected-rid": selected}
    )
    _patch_executor_init_dependencies(monkeypatch, mod, rec)

    ex = mod.BaseExecutor(record_db="db.json", record_id="selected-rid")

    assert ex.record_id == "selected-rid"
    assert ex.kernel_descr is selected


@pytest.mark.parametrize("noop_verifies", [False, True])
def test_open_close_context_manager_opens_and_closes_resources(
    monkeypatch, noop_verifies
):
    mod = _reload_with_identity_decorators(monkeypatch)

    kernel = FakeKernelDescr(
        prologue=FakeSnapshot(state="P", noop_verifies=noop_verifies),
        epilogue=FakeSnapshot(state="E"),
    )
    rec = FakeRecordedExecution(kernel)

    monkeypatch.setattr(
        mod.RecordedExecution, "from_json", staticmethod(lambda _: rec), raising=True
    )
    monkeypatch.setattr(mod, "set_device", lambda _: None, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", lambda: "sm", raising=True)
    monkeypatch.setattr(mod, "get_device_count", lambda: 1, raising=True)

    ex = mod.BaseExecutor(record_db="x", record_id="rid", device_id=0)

    with ex as opened:
        assert opened.prologue._state == "P"
        assert opened.epilogue._state == "E"
        assert opened.noop_verifies is noop_verifies

    # After context exit
    assert ex._prologue is None
    assert ex._epilogue is None


def test_link_ir_forwards_prune_and_internalize(monkeypatch):
    mod = _reload_with_identity_decorators(monkeypatch)

    kernel = FakeKernelDescr()
    rec = FakeRecordedExecution(kernel)

    monkeypatch.setattr(
        mod.RecordedExecution, "from_json", staticmethod(lambda _: rec), raising=True
    )
    monkeypatch.setattr(mod, "set_device", lambda _: None, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", lambda: "sm", raising=True)
    monkeypatch.setattr(mod, "get_device_count", lambda: 1, raising=True)

    ex = mod.BaseExecutor(record_db="x", record_id="rid")

    m = ex.link_ir()
    assert isinstance(m, FakeModule)
    assert rec.link_calls == [(True, True)]


@pytest.mark.parametrize(
    ("ir_input", "file_mode", "file_contents", "expected_parser", "expected_data"),
    [
        (
            "define void @K() { ret void }",
            None,
            None,
            "asm",
            "define void @K() { ret void }",
        ),
        (
            "kernel.ll",
            "w",
            "define void @K() { ret void }",
            "asm",
            "define void @K() { ret void }",
        ),
        ("kernel.bc", "wb", b"bitcode", "bitcode", b"bitcode"),
    ],
)
def test_set_new_ir_parses_and_prunes_replacement_ir(
    monkeypatch,
    tmp_path,
    ir_input,
    file_mode,
    file_contents,
    expected_parser,
    expected_data,
):
    mod = _reload_with_identity_decorators(monkeypatch)

    kernel = FakeKernelDescr(kernel_name="K")
    rec = FakeRecordedExecution(kernel)

    monkeypatch.setattr(
        mod.RecordedExecution, "from_json", staticmethod(lambda _: rec), raising=True
    )
    monkeypatch.setattr(mod, "set_device", lambda _: None, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", lambda: "sm", raising=True)
    monkeypatch.setattr(mod, "get_device_count", lambda: 1, raising=True)

    parsed_module = FakeModule("parsed")
    calls = []

    def fake_parse_assembly(data):
        calls.append(("asm", data))
        return parsed_module

    def fake_parse_bitcode(data):
        calls.append(("bitcode", data))
        return parsed_module

    def fake_internalize(ir, kernel_name):
        calls.append(("internalize", ir, kernel_name))

    def fake_prune(ir):
        calls.append(("prune", ir))

    monkeypatch.setattr(mod, "parse_assembly", fake_parse_assembly, raising=True)
    monkeypatch.setattr(mod, "parse_bitcode", fake_parse_bitcode, raising=True)
    monkeypatch.setattr(mod.jit, "internalize", fake_internalize, raising=True)
    monkeypatch.setattr(mod.jit, "pruneIR", fake_prune, raising=True)

    if file_mode is not None:
        ir_path = tmp_path / ir_input
        with open(ir_path, file_mode) as f:
            f.write(file_contents)
        ir_input = str(ir_path)

    ex = mod.BaseExecutor(record_db="x", record_id="rid")

    assert ex.set_new_ir(ir_input) is parsed_module
    assert calls == [
        (expected_parser, expected_data),
        ("internalize", parsed_module, "K"),
        ("prune", parsed_module),
    ]


def test_preprocess_ir_calls_jit_hooks_based_on_config(monkeypatch):
    mod = _reload_with_identity_decorators(monkeypatch)

    # Setup executor with prologue/epilogue loaded
    pro = FakeSnapshot(state="P", args=[1, 2], num_args=2)
    epi = FakeSnapshot(state="E")
    kernel = FakeKernelDescr(static_hash="H0", prologue=pro, epilogue=epi)
    rec = FakeRecordedExecution(kernel)

    monkeypatch.setattr(
        mod.RecordedExecution, "from_json", staticmethod(lambda _: rec), raising=True
    )
    monkeypatch.setattr(mod, "set_device", lambda _: None, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", lambda: "sm", raising=True)
    monkeypatch.setattr(mod, "get_device_count", lambda: 1, raising=True)

    calls = []

    def spec_args(ir, code_hash, kname, args, num_args, avail):
        calls.append(("args", code_hash, kname, tuple(args), num_args))
        return code_hash + "|A"

    def spec_dims(ir, code_hash, kname, grid, block):
        calls.append(("dims", code_hash, kname))
        return code_hash + "|D"

    def set_lb(ir, code_hash, kname, max_thr, min_b):
        calls.append(("lb", code_hash, kname, max_thr, min_b))
        return code_hash + "|L"

    monkeypatch.setattr(mod.jit, "specialize_args", spec_args, raising=True)
    monkeypatch.setattr(mod.jit, "specialize_dims", spec_dims, raising=True)
    monkeypatch.setattr(mod.jit, "set_launch_bounds", set_lb, raising=True)

    ex = mod.BaseExecutor(record_db="x", record_id="rid")
    ex.open()

    cfg = mod.ExperimentConfiguration(
        specialize=True,
        specialize_dims=True,
        set_launch_bounds=True,
        max_threads=256,
        min_blocks_per_sm=2,
    )
    ir = FakeModule("root")
    res = mod.ExperimentResult()

    code_hash, out_ir = ex._preprocess_ir(res, cfg, ir)
    assert out_ir is ir
    assert code_hash == "H0|A|D|L"
    assert [c[0] for c in calls] == ["args", "dims", "lb"]

    ex.close()


def test_build_sets_obj_size_when_track_true(monkeypatch):
    mod = _reload_with_identity_decorators(monkeypatch)

    kernel = FakeKernelDescr()
    rec = FakeRecordedExecution(kernel)

    monkeypatch.setattr(
        mod.RecordedExecution, "from_json", staticmethod(lambda _: rec), raising=True
    )
    monkeypatch.setattr(mod, "set_device", lambda _: None, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", lambda: "sm", raising=True)
    monkeypatch.setattr(mod, "get_device_count", lambda: 1, raising=True)

    ex = mod.BaseExecutor(record_db="x", record_id="rid")

    spy = {"pre": 0, "opt": 0, "cg": 0}

    def fake_pre(result, cfg, ir):
        spy["pre"] += 1
        return "H", ir

    def fake_opt(result, cfg, ir):
        spy["opt"] += 1

    def fake_cg(result, cfg, ir):
        spy["cg"] += 1
        return FakeMemBuffer(size=999)

    monkeypatch.setattr(
        ex, "_preprocess_ir", lambda *a, **k: fake_pre(*a), raising=True
    )
    monkeypatch.setattr(ex, "_optimize", lambda *a, **k: fake_opt(*a), raising=True)
    monkeypatch.setattr(ex, "_codegen", lambda *a, **k: fake_cg(*a), raising=True)

    res = mod.ExperimentResult()
    cfg = mod.ExperimentConfiguration()
    ir = FakeModule("x")

    mb = ex._build(res, cfg, ir, track=True)
    assert isinstance(mb, FakeMemBuffer)
    assert res.obj_size == 999
    assert spy == {"pre": 1, "opt": 1, "cg": 1}

    # track=False should not set obj_size
    res2 = mod.ExperimentResult()
    mb2 = ex._build(res2, cfg, ir, track=False)
    assert res2.obj_size == 0


def test_run_records_resource_usage_when_track_true(monkeypatch):
    mod = _reload_with_identity_decorators(monkeypatch)

    kernel = FakeKernelDescr(kernel_name="K")
    rec = FakeRecordedExecution(kernel)

    monkeypatch.setattr(
        mod.RecordedExecution, "from_json", staticmethod(lambda _: rec), raising=True
    )
    monkeypatch.setattr(mod, "set_device", lambda _: None, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", lambda: "sm", raising=True)
    monkeypatch.setattr(mod, "get_device_count", lambda: 1, raising=True)

    # Provide prologue/epilogue states expected by _run_kernel
    ex = mod.BaseExecutor(record_db="x", record_id="rid")
    ex._prologue = FakeSnapshot(state="P")
    ex._epilogue = FakeSnapshot(state="E")

    device_func = FakeDeviceFunction()
    fake_dev_mod = FakeDeviceModule(device_func=device_func, kernel_name="K")

    monkeypatch.setattr(
        mod.DeviceModule,
        "from_MemBuffer",
        staticmethod(lambda mb: fake_dev_mod),
        raising=True,
    )

    run_kernel_calls = []

    def fake_run_kernel(
        result, config, kernel_name, device_func_in, iterations, **kwargs
    ):
        run_kernel_calls.append((kernel_name, iterations, kwargs.get("profile")))
        # Exercise the DeviceFunction.profile is called by BaseExecutor._run_kernel,
        # but here we shortcut; BaseExecutor._run uses this method.

    monkeypatch.setattr(ex, "_run_kernel", fake_run_kernel, raising=True)

    res = mod.ExperimentResult()
    cfg = mod.ExperimentConfiguration()
    mb = FakeMemBuffer()
    prologue = FakeMemStateRef()
    epilogue = FakeMemStateRef()

    ex.noop_verifies = True
    ex._run(res, cfg, mb, prologue, epilogue, verify=True, track=True, iterations=7)

    assert run_kernel_calls == [("K", 7, True)]
    assert res.noop_verifies is True
    assert res.reg_usage == device_func.reg_usage
    assert res.const_mem_usage == device_func.const_mem
    assert res.local_mem_usage == device_func.local_mem


def test_execute_orchestrates_verification_and_tracked_run(monkeypatch):
    mod = _reload_with_identity_decorators(monkeypatch)

    # prologue == epilogue => verified True
    shared_snap = FakeSnapshot(state="S")
    kernel = FakeKernelDescr(prologue=shared_snap, epilogue=shared_snap)
    rec = FakeRecordedExecution(kernel)

    monkeypatch.setattr(
        mod.RecordedExecution, "from_json", staticmethod(lambda _: rec), raising=True
    )
    monkeypatch.setattr(mod, "set_device", lambda _: None, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", lambda: "sm", raising=True)
    monkeypatch.setattr(mod, "get_device_count", lambda: 1, raising=True)

    # Spy on _build and _run
    build_calls = []
    run_calls = []

    def fake_build(result, cfg, ir_mod, track):
        build_calls.append((track, cfg.passes))
        return FakeMemBuffer()

    def fake_run(result, cfg, mem_buf, prologue, epilogue, verify, track, iters):
        if verify:
            result.verified = True
        run_calls.append((track, iters))

    ex = mod.BaseExecutor(record_db="x", record_id="rid", iterations=3)
    ex.open()  # load prologue/epilogue

    monkeypatch.setattr(ex, "_build", fake_build, raising=True)
    monkeypatch.setattr(ex, "_run", fake_run, raising=True)

    res = mod.ExperimentResult()
    cfg = mod.ExperimentConfiguration()
    ir = FakeModule("root")

    out_ir = ex._execute(res, cfg, ir)

    # Verification stage: track=False, iterations=1
    # Tracked stage: track=True, iterations=self._iterations + 2 = 5
    assert run_calls == [(False, 1), (True, 5)]
    assert res.executed is True
    assert res.verified is True

    # The verification build uses the caller's pipeline; the tracked build
    # prepends the purge pass.
    assert build_calls == [(False, "default<O3>"), (True, "purge-autoinit,default<O3>")]

    # The returned module is a clone; the caller's module is left untouched.
    assert out_ir is not ir

    ex.close()


def test_tuneworker_process_payload_sets_timestamps_and_gpu_id(monkeypatch):
    mod = _reload_with_identity_decorators(monkeypatch)

    kernel = FakeKernelDescr()
    rec = FakeRecordedExecution(kernel)

    monkeypatch.setattr(
        mod.RecordedExecution, "from_json", staticmethod(lambda _: rec), raising=True
    )
    monkeypatch.setattr(mod, "set_device", lambda _: None, raising=True)
    monkeypatch.setattr(mod, "get_device_arch", lambda: "sm", raising=True)
    monkeypatch.setattr(mod, "get_device_count", lambda: 1, raising=True)

    # Avoid actual profiler init
    monkeypatch.setattr(mod, "init_profiler", lambda: None, raising=True)

    worker = mod.TuneWorker(record_db="x", record_id="rid", device_id=2, iterations=3)

    # Patch _execute to avoid deeper pipeline; return a module
    def fake_execute(self, result, cfg, ir_mod):
        result.executed = True
        return FakeModule("gen")

    monkeypatch.setattr(mod.BaseExecutor, "_execute", fake_execute, raising=True)

    res, out_ir = worker.process_payload(
        FakeModule("root"), mod.ExperimentConfiguration()
    )

    assert res.gpu_id == 2
    assert res.start_time != ""
    assert res.end_time != ""
    assert res.executed is True
    assert isinstance(out_ir, FakeModule)


def test_tuneworker_run_process_and_terminate(monkeypatch, tmp_path):
    mod = _reload_with_identity_decorators(monkeypatch)

    # Save the real staticmethod BEFORE monkeypatching the class name.
    real_run = mod.TuneWorker.run

    # Patch os redirections to avoid touching real fd 1/2
    monkeypatch.setattr(mod.os, "open", lambda *a, **k: 999, raising=True)
    monkeypatch.setattr(mod.os, "dup2", lambda *a, **k: None, raising=True)

    set_ir_calls = []
    process_ir_names = []

    class FakeWorker:
        def __init__(self, record_db, record_id, device_id, iterations, warmup):
            self.record_db = record_db
            self.record_id = record_id
            self.device_id = device_id
            self.iterations = iterations
            self.warmup = warmup

        def link_ir(self):
            return FakeModule("root_ir")

        def set_new_ir(self, ir_data):
            set_ir_calls.append(ir_data)
            return FakeModule("replacement_ir")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def process_payload(self, ir_module, config):
            process_ir_names.append(ir_module.name)
            res = mod.ExperimentResult(executed=True, verified=True)
            return res, FakeModule("ir_out")

    # Replace TuneWorker constructor used inside run()
    monkeypatch.setattr(
        mod, "TuneWorker", lambda *a, **k: FakeWorker(*a, **k), raising=True
    )

    req_q = queue.Queue()
    resp_q = queue.Queue()

    class FakeEvent:
        def __init__(self):
            self.set_called = 0

        def set(self):
            self.set_called += 1

    state = FakeEvent()

    req_q.put({"payload": "set_ir", "data": "replacement asm"})
    req_q.put(
        {
            "payload": "process",
            "exp_id": 7,
            "data": mod.ExperimentConfiguration().to_dict(),
        }
    )
    req_q.put({"payload": "terminate"})

    # Call the saved real staticmethod
    real_run(
        request_q=req_q,
        response_q=resp_q,
        record_db="db.json",
        record_id="rid",
        device_id=0,
        iterations=3,
        results_db_dir=str(tmp_path),
        state=state,
    )

    assert state.set_called == 1
    assert set_ir_calls == ["replacement asm"]
    assert process_ir_names == ["replacement_ir_clone"]
    msg = resp_q.get_nowait()
    assert msg["payload"] == "result"
    assert msg["exp_id"] == 7
    assert isinstance(msg["data"], dict)
    assert msg["llvm_ir"] == ""


def test_tuneworker_run_logs_failed_set_ir(monkeypatch, tmp_path):
    mod = _reload_with_identity_decorators(monkeypatch)

    real_run = mod.TuneWorker.run

    monkeypatch.setattr(mod.os, "open", lambda *a, **k: 999, raising=True)
    monkeypatch.setattr(mod.os, "dup2", lambda *a, **k: None, raising=True)

    errors = []
    monkeypatch.setattr(
        mod.logger, "error", lambda msg: errors.append(msg), raising=True
    )

    class FakeWorker:
        def __init__(self, record_db, record_id, device_id, iterations, warmup):
            self.device_id = device_id

        def link_ir(self):
            return FakeModule("root_ir")

        def set_new_ir(self, ir_data):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        mod, "TuneWorker", lambda *a, **k: FakeWorker(*a, **k), raising=True
    )

    req_q = queue.Queue()
    resp_q = queue.Queue()

    class FakeEvent:
        def __init__(self):
            self.set_called = 0

        def set(self):
            self.set_called += 1

    state = FakeEvent()

    req_q.put({"payload": "set_ir", "data": "bad asm"})
    req_q.put({"payload": "terminate"})

    real_run(
        request_q=req_q,
        response_q=resp_q,
        record_db="db.json",
        record_id="rid",
        device_id=3,
        iterations=3,
        results_db_dir=str(tmp_path),
        state=state,
    )

    assert state.set_called == 1
    assert errors == ["Worker 3 failed to set new IR"]
    assert resp_q.empty()
