""" Tests for the main TuningSession class that handles tuning for the tune CLI.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest

from mneme.mneme_types import ExperimentConfiguration, ExperimentResult, dim3
from mneme.recorded_execution import KernelSource
from mneme.tuning import session as tune_session
from mneme.tuning.search_space import (
    BaseParam,
    BoolParam,
    CategoricalParam,
    FixedParam,
    IntRangeParam,
    RealRangeParam,
)
from mneme.tuning.session import (
    BaselineVerificationError,
    Candidate,
    CompletedCandidate,
    EXIT_BASELINE_FAILED,
    EXIT_INVALID_CONFIGURATION,
    EXIT_RECORD_LOAD_FAILED,
    TuneOptions,
    TuningSession,
    _constraints_ok,
    _objective_value,
    _sample_random_param,
    classify_result,
    metric_value,
)


def make_options(tmp_path, **overrides):
    values = {
        "record_database": "records.json",
        "record_id": "rid",
        "sampler": "exhaustive",
        "trials": None,
        "results_dir": str(tmp_path),
        "quiet": True,
    }
    values.update(overrides)
    return TuneOptions(**values)


def make_config(**overrides):
    values = {
        "grid": dim3(8, 1, 1),
        "block": dim3(128, 1, 1),
        "shared_mem": 0,
        "passes": "default<O3>",
        "codegen_opt": 3,
        "max_threads": 0,
        "min_blocks_per_sm": 0,
    }
    values.update(overrides)
    return ExperimentConfiguration(**values)


class FakeKernel:
    kernel_name = "KERNEL"
    shared_mem = 0
    block_dim = dim3(128, 1, 1)
    grid_dim = dim3(8, 1, 1)


class FakeRecorded:
    def __init__(self, source=None):
        self.source = source

    def __getitem__(self, record_id):
        if record_id != "rid":
            raise KeyError(record_id)
        return FakeKernel()

    def kernel_source(self):
        return self.source


class DoneFuture:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class RecordingExecutor:
    def __init__(self, baseline_result=None, submitted_result=None):
        self.baseline_result = baseline_result or ExperimentResult(
            verified=True,
            executed=True,
            exec_time=[10, 12, 14],
        )
        self.submitted_result = submitted_result or ExperimentResult(
            verified=True,
            executed=True,
            exec_time=[8, 8, 8],
        )
        self.evaluated = []
        self.submitted = []
        self.events = []
        self.shutdown_calls = 0

    def set_ir(self, ir):
        self.events.append(("set_ir", ir))

    def evaluate(self, config):
        self.events.append(("evaluate", config))
        self.evaluated.append(config)
        return self.baseline_result

    def submit(self, config):
        self.events.append(("submit", config))
        self.submitted.append(config)
        return DoneFuture(self.submitted_result)

    def shutdown(self):
        self.shutdown_calls += 1


class GridSpace:
    def dimensions(self):
        return {
            "block_shape": CategoricalParam("block_shape", ["bad", "ok"]),
            "codegen_opt": IntRangeParam("codegen_opt", 1, 2),
        }

    def derived(self, params):
        return make_config(codegen_opt=params["codegen_opt"])

    def constraints(self, params):
        return params["codegen_opt"] == 2

    def baseline(self):
        return make_config()


def test_metric_value_uses_requested_statistic_and_trailing_iterations():
    result = ExperimentResult(exec_time=[100, 40, 20, 60])

    assert metric_value(result, "mean", iterations=3) == 40
    assert metric_value(result, "median", iterations=3) == 40
    assert metric_value(result, "min", iterations=3) == 20
    assert metric_value(result, "max", iterations=3) == 60
    assert metric_value(ExperimentResult(exec_time=[]), "mean", iterations=3) is None

    with pytest.raises(ValueError, match="Unknown metric"):
        metric_value(result, "p95", iterations=3)


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (ExperimentResult(verified=True), "verified"),
        (ExperimentResult(error="invalid launch dimensions"), "invalid_launch"),
        (ExperimentResult(error="invalid configuration: max_threads"), "invalid_config"),
        (ExperimentResult(failed=True, error="LLVM compile failed"), "compile_error"),
        (ExperimentResult(failed=True, error="device assert"), "runtime_error"),
        (ExperimentResult(executed=True), "failed_verification"),
        (ExperimentResult(error="worker crashed"), "runtime_error"),
        (ExperimentResult(), "failed_verification"),
    ],
)
def test_classify_result_maps_replay_outcomes(result, status):
    assert classify_result(result) == status


def test_tune_options_validate_sampler_trial_combinations(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        make_options(tmp_path, sampler="random", trials=0)

    with pytest.raises(ValueError, match="required"):
        make_options(tmp_path, sampler="tpe", trials=None)

    with pytest.raises(ValueError, match="cannot be used"):
        make_options(tmp_path, sampler="grid", trials=3)


def test_tune_options_config_fills_results_and_proteus_output(tmp_path):
    opts = make_options(tmp_path, results_dir=None, proteus_output=None)

    config = opts.to_config_dict()

    assert config["results_dir"].startswith("mneme-tune-results/rid-")
    assert config["proteus_output"].endswith("proteus_tuned_kernels.json")


def test_objective_value_penalizes_failed_candidates():
    assert _objective_value("runtime_error", 10.0, 2.0, "time") == float("inf")
    assert _objective_value("runtime_error", 10.0, 2.0, "speedup") == 0.0
    assert _objective_value("verified", 10.0, 2.0, "time") == 10.0
    assert _objective_value("verified", 10.0, 2.0, "speedup") == 2.0


def test_sample_random_param_handles_supported_parameter_types():
    rng = tune_session.random.Random(1)

    assert _sample_random_param(FixedParam("fixed", "x"), rng) == "x"
    assert _sample_random_param(BoolParam("enabled"), rng) in {True, False}
    assert _sample_random_param(CategoricalParam("choice", ["a", "b"]), rng) in {"a", "b"}
    assert _sample_random_param(IntRangeParam("level", 2, 6, step=2), rng) in {2, 4, 6}

    value = _sample_random_param(RealRangeParam("ratio", 0.25, 0.5), rng)
    assert 0.25 <= value <= 0.5


def test_sample_random_param_rejects_unknown_parameter_type():
    class UnknownParam(BaseParam):
        pass

    with pytest.raises(TypeError, match="Unsupported random parameter type"):
        _sample_random_param(UnknownParam("unknown"), tune_session.random.Random(1))


def test_constraints_ok_falls_back_to_config_constraints_and_checks_validity():
    class FallbackSpace:
        def constraints(self, value):
            if isinstance(value, dict):
                raise TypeError("expects config")
            return True

    assert _constraints_ok(FallbackSpace(), {}, make_config()) is True
    invalid = make_config(set_launch_bounds=True, max_threads=1)
    assert _constraints_ok(FallbackSpace(), {}, invalid) is False


def test_load_record_reports_missing_record_id(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: FakeRecorded()),
    )

    session = TuningSession(make_options(tmp_path, record_id="missing"))

    with pytest.raises(RuntimeError, match="was not found"):
        session._load_record()


def test_validate_resume_requires_existing_config(tmp_path):
    session = TuningSession(make_options(tmp_path, resume=True))

    with pytest.raises(RuntimeError, match="requires an existing results directory"):
        session._validate_resume(session.options.to_config_dict())


def test_validate_resume_rejects_immutable_option_mismatch(tmp_path):
    session = TuningSession(make_options(tmp_path, resume=True))
    session.store.write_config(dict(session.options.to_config_dict(), record_id="other"))

    with pytest.raises(RuntimeError, match="record_id"):
        session._validate_resume(session.options.to_config_dict())


def test_validate_resume_loads_completed_hashes(tmp_path):
    session = TuningSession(make_options(tmp_path, resume=True))
    config = session.options.to_config_dict()
    session.store.write_config(config)
    session.store.append_trial({"config_hash": "abc", "status": "verified"})

    session._validate_resume(config)

    assert session._completed_hashes == {"abc"}


def test_resume_rejects_changed_ir_hash(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(TuningSession, "_load_record", lambda self: (FakeRecorded(), FakeKernel()))
    monkeypatch.setattr(TuningSession, "_build_space", lambda self, kernel: (GridSpace(), {}))

    first = TuningSession(make_options(tmp_path, print_space=True), ir="first IR")
    assert first.run() == 0
    assert first.store.load_config()["ir_hash"] == hashlib.sha256(b"first IR").hexdigest()

    same = TuningSession(make_options(tmp_path, resume=True, print_space=True), ir="first IR")
    assert same.run() == 0

    changed = TuningSession(make_options(tmp_path, resume=True, print_space=True), ir="second IR")
    assert changed.run() == EXIT_INVALID_CONFIGURATION
    assert "ir_hash" in capsys.readouterr().out


def test_resume_rejects_changed_pass_plugin(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(TuningSession, "_load_record", lambda self: (FakeRecorded(), FakeKernel()))
    monkeypatch.setattr(TuningSession, "_build_space", lambda self, kernel: (GridSpace(), {}))
    monkeypatch.setattr("mneme.proteus.jit.check_pass_plugin", lambda path: None)
    plugin = tmp_path / "libMyPass.so"
    plugin.write_bytes(b"first build")
    options = dict(pass_plugins=[str(plugin)], print_space=True)

    first = TuningSession(make_options(tmp_path, **options))
    assert first.run() == 0

    same = TuningSession(make_options(tmp_path, resume=True, **options))
    assert same.run() == 0

    plugin.write_bytes(b"second build")
    rebuilt = TuningSession(make_options(tmp_path, resume=True, **options))
    assert rebuilt.run() == EXIT_INVALID_CONFIGURATION
    assert "pass_plugin_hashes" in capsys.readouterr().out


def test_run_rejects_legacy_checkpoint_with_new_ir(tmp_path, capsys):
    # check that we reject when checkpoint is missing ir
    session = TuningSession(make_options(tmp_path, resume=True), ir="new IR")
    config = session.options.to_config_dict()
    session.store.write_config(config)

    assert session.run() == EXIT_INVALID_CONFIGURATION
    assert "ir_hash" in capsys.readouterr().out
    assert session.store.load_config() == config


def test_run_hashes_ir_file_contents_and_rejects_changed_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(TuningSession, "_load_record", lambda self: (FakeRecorded(), FakeKernel()))
    monkeypatch.setattr(TuningSession, "_build_space", lambda self, kernel: (GridSpace(), {}))
    ir_path = tmp_path / "kernel.ll"
    ir_path.write_text("first IR")

    first = TuningSession(make_options(tmp_path, print_space=True), ir=ir_path)
    assert first.run() == 0
    assert first.store.load_config()["ir_hash"] == hashlib.sha256(b"first IR").hexdigest()

    ir_path.write_text("second IR")
    resumed = TuningSession(make_options(tmp_path, resume=True, print_space=True), ir=ir_path)
    assert resumed.run() == EXIT_INVALID_CONFIGURATION
    assert "ir_hash" in capsys.readouterr().out
    assert resumed.store.load_config()["ir_hash"] == hashlib.sha256(b"first IR").hexdigest()


def test_grid_candidates_skip_banned_and_constraint_failures(tmp_path):
    class LayoutSpace(GridSpace):
        def derived(self, params):
            return make_config(grid=dim3(1 if params["block_shape"] == "bad" else 2, 1, 1),
                               codegen_opt=params["codegen_opt"])

    session = TuningSession(make_options(tmp_path, sampler="exhaustive"))
    session._banned_layouts.add("128x1x1/1x1x1")

    candidates = list(session._iter_grid_candidates(LayoutSpace()))

    assert [candidate.params for candidate in candidates] == [
        {"block_shape": "ok", "codegen_opt": 2}
    ]
    assert candidates[0].config.codegen_opt == 2


def test_random_candidates_retry_until_constraints_pass(tmp_path):
    class RetrySpace:
        def __init__(self):
            self.calls = 0

        def dimensions(self):
            return {"mode": FixedParam("mode", "candidate")}

        def derived(self, params):
            return make_config()

        def constraints(self, params):
            self.calls += 1
            return self.calls == 2

    session = TuningSession(make_options(tmp_path, sampler="random", trials=1))
    space = RetrySpace()

    candidates = list(session._iter_random_candidates(space))

    assert len(candidates) == 1
    assert candidates[0].trial == 0
    assert candidates[0].params == {"mode": "candidate"}
    assert space.calls == 2


def test_record_invalid_candidate_writes_trial_and_tells_optuna(tmp_path):
    class FakeStudy:
        def __init__(self):
            self.told = []

        def tell(self, trial, value):
            self.told.append((trial, value))

    session = TuningSession(make_options(tmp_path))
    study = FakeStudy()
    candidate = Candidate(
        trial=7,
        params={"x": 1},
        config=make_config(),
        optuna_trial=object(),
        optuna_study=study,
    )

    completed = session._record_invalid_candidate(candidate, 100.0, "not legal")
    trial = json.loads(session.store.trials_file.read_text())

    assert completed.status == "invalid_config"
    assert completed.metric is None
    assert trial["trial"] == 7
    assert trial["status"] == "invalid_config"
    assert trial["result"]["error"] == "not legal"
    assert study.told == [(candidate.optuna_trial, float("inf"))]


def test_load_resume_best_counts_trials_and_picks_lowest_verified_metric(tmp_path):
    session = TuningSession(make_options(tmp_path))
    slow_config = make_config(codegen_opt=1)
    fast_config = make_config(codegen_opt=2)
    session.store.append_trial(
        {
            "trial": 0,
            "params": {"codegen_opt": 1},
            "status": "verified",
            "config": slow_config.to_dict(),
            "result": dict(ExperimentResult(verified=True, exec_time=[90]).to_dict(), metric=90),
            "speedup": 1.1,
        }
    )
    session.store.append_trial(
        {
            "trial": 1,
            "params": {"codegen_opt": 2},
            "status": "verified",
            "config": fast_config.to_dict(),
            "result": dict(ExperimentResult(verified=True, exec_time=[50]).to_dict(), metric=50),
            "speedup": 2.0,
        }
    )
    session.store.append_trial(
        {
            "trial": 2,
            "params": {},
            "status": "compile_error",
            "config": make_config().to_dict(),
            "result": ExperimentResult(failed=True, error="compile").to_dict(),
        }
    )

    best, counts, completed = session._load_resume_best(
        make_config(),
        ExperimentResult(verified=True, exec_time=[100]),
        100.0,
    )

    assert best.trial == 1
    assert best.metric == 50
    assert best.config.codegen_opt == 2
    assert counts == {"verified": 2, "compile_error": 1}
    assert completed == 3


def test_evaluate_baseline_uses_resumed_metric_without_executor_call(tmp_path):
    session = TuningSession(make_options(tmp_path, resume=True, metric="min", iterations=2))
    session.store.write_baseline(
        {
            "config": make_config().to_dict(),
            "result": dict(
                ExperimentResult(verified=True, executed=True, exec_time=[5, 3, 4]).to_dict(),
                metric=2.5,
            ),
        }
    )
    executor = RecordingExecutor()

    result, metric = session._evaluate_baseline(executor, make_config())

    assert result.verified is True
    assert metric == 2.5
    assert executor.evaluated == []


def test_evaluate_baseline_writes_fresh_result_and_rejects_failure(tmp_path):
    session = TuningSession(make_options(tmp_path))
    executor = RecordingExecutor(
        baseline_result=ExperimentResult(
            verified=False,
            executed=True,
            exec_time=[10, 10],
            error="mismatch",
        )
    )

    with pytest.raises(BaselineVerificationError):
        session._evaluate_baseline(executor, make_config())

    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["result"]["verified"] is False
    assert baseline["result"]["metric"] == 10


def test_write_final_outputs_skips_proteus_when_disabled(tmp_path, monkeypatch):
    called = False

    def fake_export(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(tune_session, "export_proteus_tuned_kernel", fake_export)
    session = TuningSession(make_options(tmp_path, proteus_enabled=False))
    best = CompletedCandidate(
        trial=3,
        params={"x": 1},
        config=make_config(),
        result=ExperimentResult(verified=True, executed=True, exec_time=[25]),
        status="verified",
        metric=25.0,
        speedup=4.0,
    )

    summary = session._write_final_outputs(
        FakeRecorded(),
        FakeKernel(),
        baseline_metric=100.0,
        best=best,
        counts={"verified": 3, "runtime_error": 1},
        completed_trials=4,
        num_requested=5,
        noop_verifies=False,
    )

    assert called is False
    assert summary["best_speedup"] == 4.0
    assert summary["num_runtime_error"] == 1
    assert json.loads((tmp_path / "best.json").read_text())["trial"] == 3


def test_run_print_space_writes_config_and_search_space(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: FakeRecorded()),
    )
    monkeypatch.setattr(
        TuningSession,
        "_build_space",
        lambda self, kernel: (GridSpace(), {"kind": "fake-space"}),
    )

    session = TuningSession(make_options(tmp_path, print_space=True))

    assert session.run() == 0
    assert json.loads((tmp_path / "search_space.json").read_text()) == {"kind": "fake-space"}
    assert '"kind": "fake-space"' in capsys.readouterr().out


def test_run_dry_run_prints_baseline_without_executor(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: FakeRecorded()),
    )
    monkeypatch.setattr(
        TuningSession,
        "_build_space",
        lambda self, kernel: (GridSpace(), {"kind": "fake-space"}),
    )

    session = TuningSession(make_options(tmp_path, dry_run=True))

    assert session.run() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["baseline"]["passes"] == "default<O3>"
    assert output["search_space"] == {"kind": "fake-space"}


def test_run_reports_resume_configuration_error(tmp_path, capsys):
    session = TuningSession(make_options(tmp_path, resume=True))

    assert session.run() == EXIT_INVALID_CONFIGURATION
    assert "Invalid tuning configuration" in capsys.readouterr().out


def test_run_reports_record_load_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: (_ for _ in ()).throw(RuntimeError("bad db"))),
    )

    session = TuningSession(make_options(tmp_path))

    assert session.run() == EXIT_RECORD_LOAD_FAILED
    assert "Failed to load record database" in capsys.readouterr().out


def test_run_reports_search_space_configuration_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: FakeRecorded()),
    )
    monkeypatch.setattr(
        TuningSession,
        "_build_space",
        lambda self, kernel: (_ for _ in ()).throw(ValueError("bad space")),
    )

    session = TuningSession(make_options(tmp_path))

    assert session.run() == EXIT_INVALID_CONFIGURATION
    assert "Invalid tuning configuration" in capsys.readouterr().out


def test_run_reports_baseline_failure(monkeypatch, tmp_path, capsys):
    executor = RecordingExecutor(
        baseline_result=ExperimentResult(
            verified=False,
            executed=True,
            exec_time=[10],
            error="mismatch",
        )
    )
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: FakeRecorded()),
    )
    monkeypatch.setattr(
        TuningSession,
        "_build_space",
        lambda self, kernel: (GridSpace(), {"kind": "fake-space"}),
    )
    monkeypatch.setattr(tune_session, "AsyncReplayExecutor", lambda **kwargs: executor)

    session = TuningSession(make_options(tmp_path))

    assert session.run() == EXIT_BASELINE_FAILED
    assert "Baseline replay did not verify" in capsys.readouterr().out
    assert executor.submitted == []
    assert executor.shutdown_calls == 1


def test_run_reuses_borrowed_executor_and_sets_ir_before_baseline(monkeypatch, tmp_path):
    executor = RecordingExecutor()
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: FakeRecorded()),
    )
    monkeypatch.setattr(
        TuningSession,
        "_build_space",
        lambda self, kernel: (GridSpace(), {"kind": "fake-space"}),
    )
    monkeypatch.setattr(
        tune_session,
        "AsyncReplayExecutor",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("created executor")),
    )

    first = TuningSession(
        make_options(tmp_path / "first", proteus_enabled=False),
        executor=executor,
        ir="first candidate IR",
    )
    second = TuningSession(
        make_options(tmp_path / "second", proteus_enabled=False),
        executor=executor,
        ir="second candidate IR",
    )

    assert first.run() == 0
    assert second.run() == 0
    assert [name for name, _ in executor.events] == [
        "set_ir", "evaluate", "submit", "submit",
        "set_ir", "evaluate", "submit", "submit",
    ]
    assert executor.events[0][1] == "first candidate IR"
    assert executor.events[4][1] == "second candidate IR"
    assert executor.shutdown_calls == 0
    assert (tmp_path / "first" / "best.json").is_file()
    assert (tmp_path / "second" / "best.json").is_file()


def test_baseline_failure_does_not_shutdown_borrowed_executor(monkeypatch, tmp_path):
    executor = RecordingExecutor(
        baseline_result=ExperimentResult(verified=False, executed=True, exec_time=[10]),
    )
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: FakeRecorded()),
    )
    monkeypatch.setattr(
        TuningSession,
        "_build_space",
        lambda self, kernel: (GridSpace(), {"kind": "fake-space"}),
    )

    session = TuningSession(
        make_options(tmp_path, proteus_enabled=False), executor=executor, ir="bad candidate IR"
    )

    assert session.run() == EXIT_BASELINE_FAILED
    assert executor.events[0] == ("set_ir", "bad candidate IR")
    assert executor.shutdown_calls == 0


@pytest.mark.parametrize("noop_verifies", [False, True])
def test_run_baseline_only_writes_baseline_as_best(
    monkeypatch, tmp_path, capsys, noop_verifies
):
    executor = RecordingExecutor(
        baseline_result=ExperimentResult(
            verified=True,
            noop_verifies=noop_verifies,
            executed=True,
            exec_time=[100],
        )
    )
    source = KernelSource("/src/k.cu", 11, 14, "")
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: FakeRecorded(source)),
    )
    monkeypatch.setattr(
        TuningSession,
        "_build_space",
        lambda self, kernel: (GridSpace(), {"kind": "fake-space"}),
    )
    monkeypatch.setattr(tune_session, "AsyncReplayExecutor", lambda **kwargs: executor)

    session = TuningSession(
        make_options(tmp_path, baseline_only=True, proteus_enabled=False, quiet=False)
    )

    assert session.run() == 0
    captured = capsys.readouterr()
    assert "  source:          /src/k.cu:11-14" in captured.out
    assert ("a kernel that does nothing would pass verification" in captured.err) is noop_verifies
    assert executor.submitted == []
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["params"] == {"baseline": True}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["noop_verifies"] is noop_verifies
    assert best["best_metric"] == 100


def test_merge_config_and_args_flattens_search_and_prefers_cli(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "record_database": "from-config.json",
                "search": {"sampler": "random", "trials": 9},
                "iterations": 4,
            }
        )
    )
    args = SimpleNamespace(
        config=str(config_path),
        record_database="from-cli.json",
        record_id=None,
        trials=None,
        command="tune",
        func=object(),
    )

    merged = tune_session.merge_config_and_args(args)

    assert merged["record_database"] == "from-cli.json"
    assert merged["sampler"] == "random"
    assert merged["trials"] == 9
    assert merged["iterations"] == 4
    assert "command" not in merged


@pytest.mark.parametrize("overrides, message", [
    ({"sampler": "exhaustive", "trials": None}, "random or tpe"),
    ({"sampler": "grid", "trials": None}, "random or tpe"),
    ({"space_module": "custom:Space"}, "custom search space"),
    ({"require_verified_baseline": False, "objective": "speedup"}, "objective='time'"),
])
def test_new_options_reject_unsupported_combinations(tmp_path, overrides, message):
    values = dict(sampler="random", trials=2, launch_candidates=[{"block": [64, 1, 1]}])
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        make_options(tmp_path, **values)


@pytest.mark.parametrize("sampler", ["random", "tpe"])
def test_augmented_session_runs_initial_layouts_then_full_budget_and_exports_winner(monkeypatch, tmp_path, sampler):
    monkeypatch.setattr(TuningSession, "_load_record", lambda self: (FakeRecorded(), FakeKernel()))
    measured = []
    exported = []

    class Executor(RecordingExecutor):
        def submit(self, config):
            self.submitted.append(config)
            # Every measured configuration has a distinct timing; the export must
            # retain the exact layout and compiler flags from the winning sample.
            result = ExperimentResult(verified=True, executed=True, exec_time=[100 - len(self.submitted)])
            measured.append((config.to_dict(), result.to_dict()))
            return DoneFuture(result)

    executor = Executor(baseline_result=ExperimentResult(verified=True, exec_time=[1000]))
    monkeypatch.setattr(tune_session, "export_proteus_tuned_kernel",
                        lambda filename, recorded, kernel, config, **kwargs: exported.append(config.to_dict()))
    options = make_options(
        tmp_path, sampler=sampler, trials=30, workers=3, seed=7,
        space_preset="compiler", specialize_space="on-off", launch_bounds_space="on-off",
        launch_candidates=[
            {"block": [128, 1, 1]},
            {"block": [128, 1, 1], "grid": [8, 1, 1]},
            {"block": [128, 1, 1], "grid": [9, 1, 1]},
            {"block": [513, 1, 1]},
        ],
    )
    assert TuningSession(options, executor=executor).run() == 0
    assert len(executor.evaluated) == 1
    assert len(executor.submitted) == 33
    baseline = executor.evaluated[0].to_dict()
    for config, expected in zip(executor.submitted[:3], [(128, 8), (128, 9), (513, 2)]):
        assert (config.block.x, config.grid.x) == expected
        assert {k: v for k, v in config.to_dict().items() if k not in {"block", "grid"}} == {
            k: v for k, v in baseline.items() if k not in {"block", "grid"}
        }
    automatic = executor.submitted[3:]
    supplied_samples = [c for c in automatic if c.grid.x != 8]
    assert supplied_samples
    assert any(c.passes != baseline["passes"] for c in supplied_samples)
    assert any(c.codegen_opt != baseline["codegen_opt"] for c in supplied_samples)
    assert any(c.specialize or c.set_launch_bounds for c in supplied_samples)
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["config"] == measured[-1][0] == exported[0]
    assert best["result"] == measured[-1][1]
    assert best["best_metric"] == 67
    records = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert [r["trial"] for r in records] == list(range(33))
    assert records[best["trial"]]["config"] == best["config"]
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["launch_candidates"] == options.launch_candidates
    assert config["require_verified_baseline"] is True
    domain = json.loads((tmp_path / "search_space.json").read_text())["metadata"]["launch_layouts"]
    assert len(domain) == 3
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["num_trials_requested"] == 30
    assert summary["num_trials_completed"] == 33


@pytest.mark.parametrize("sampler", ["random", "tpe"])
def test_invalid_launch_bans_only_full_layout(monkeypatch, tmp_path, sampler):
    monkeypatch.setattr(TuningSession, "_load_record", lambda self: (FakeRecorded(), FakeKernel()))

    class Executor(RecordingExecutor):
        def submit(self, config):
            self.submitted.append(config)
            if config.grid.x == 8:
                return DoneFuture(ExperimentResult(failed=True, error="invalid launch"))
            return DoneFuture(ExperimentResult(verified=True, exec_time=[5]))

    executor = Executor()
    session = TuningSession(make_options(
        tmp_path, sampler=sampler, trials=10, space_preset="compiler", seed=1,
        proteus_enabled=False,
        launch_candidates=[{"block": [128, 1, 1]}, {"block": [128, 1, 1], "grid": [9, 1, 1]}],
    ), executor=executor)
    assert session.run() == 0
    assert session._banned_layouts == {"128x1x1/8x1x1"}
    assert len(executor.submitted) == 12
    assert all(c.block.x == 128 and c.grid.x == 9 for c in executor.submitted[1:])


@pytest.mark.parametrize("sampler", ["random", "tpe"])
@pytest.mark.parametrize("successful", [True, False])
def test_baseline_failure_continuation_and_complete_failure(monkeypatch, tmp_path, sampler, successful):
    monkeypatch.setattr(TuningSession, "_load_record", lambda self: (FakeRecorded(), FakeKernel()))
    failed = ExperimentResult(executed=True, exec_time=[1], error="mismatch")
    executor = RecordingExecutor(baseline_result=failed, submitted_result=(
        ExperimentResult(verified=True, executed=True, exec_time=[20]) if successful else failed
    ))
    session = TuningSession(make_options(
        tmp_path, sampler=sampler, trials=3, require_verified_baseline=False, proteus_enabled=False,
        launch_candidates=[{"block": [256, 1, 1]}],
    ), executor=executor)
    assert session.run() == (0 if successful else EXIT_BASELINE_FAILED)
    assert len(executor.submitted) == 4
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["result"]["verified"] is False
    assert baseline["config"]["block"]["x"] == 128
    assert baseline["result"]["metric"] == 1
    records = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert all(r["baseline_metric"] is None and r["speedup"] is None for r in records)
    if successful:
        best = json.loads((tmp_path / "best.json").read_text())
        assert best["config"]["block"]["x"] == 256
        assert best["baseline_metric"] is None and best["speedup"] is None
        assert best["params"].get("baseline") is not True
    else:
        assert not (tmp_path / "best.json").exists()
        assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("verified", [True, False])
def test_timeout_drains_initial_evaluation_and_keeps_best_verified(monkeypatch, tmp_path, verified):
    monkeypatch.setattr(TuningSession, "_load_record", lambda self: (FakeRecorded(), FakeKernel()))
    now = [0.0]
    monkeypatch.setattr(tune_session.time, "monotonic", lambda: now[0])

    class Executor(RecordingExecutor):
        def submit(self, config):
            now[0] = 2.0
            return super().submit(config)

    executor = Executor(
        baseline_result=ExperimentResult(executed=True),
        submitted_result=ExperimentResult(verified=verified, executed=True, exec_time=[5]),
    )
    options = make_options(
        tmp_path, sampler="random", trials=30, timeout=1, require_verified_baseline=False,
        proteus_enabled=False, launch_candidates=[{"block": [256, 1, 1]}, {"block": [512, 1, 1]}],
    )
    assert TuningSession(options, executor=executor).run() == (0 if verified else EXIT_BASELINE_FAILED)
    assert len(executor.submitted) == 1
    if verified:
        assert json.loads((tmp_path / "best.json").read_text())["config"]["block"]["x"] == 256
    else:
        assert not (tmp_path / "best.json").exists()


def test_failed_resumed_baseline_obeys_policy(tmp_path):
    session = TuningSession(make_options(tmp_path, resume=True))
    session.store.write_baseline({"result": ExperimentResult(failed=True, error="compile error").to_dict()})
    executor = RecordingExecutor()
    with pytest.raises(BaselineVerificationError):
        session._evaluate_baseline(executor, make_config())
    session.options.require_verified_baseline = False
    result, metric = session._evaluate_baseline(executor, make_config())
    assert result.failed and metric is None
    assert not executor.evaluated


@pytest.mark.parametrize("sampler", ["random", "tpe"])
def test_configuration_and_execution_exceptions_reject_samples_and_continue(monkeypatch, tmp_path, sampler):
    monkeypatch.setattr(TuningSession, "_load_record", lambda self: (FakeRecorded(), FakeKernel()))

    class Space(GridSpace):
        calls = 0

        def derived(self, params):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("bad configuration")
            return make_config()

        def constraints(self, params):
            return True

    monkeypatch.setattr(TuningSession, "_build_space", lambda self, kernel: (Space(), {}))

    class Executor(RecordingExecutor):
        def submit(self, config):
            self.submitted.append(config)
            if len(self.submitted) == 1:
                raise RuntimeError("compile failed")
            if len(self.submitted) == 2:
                return DoneFuture(RuntimeError("device error"))
            return DoneFuture(ExperimentResult(verified=True, exec_time=[5]))

    executor = Executor()
    assert TuningSession(make_options(
        tmp_path, sampler=sampler, trials=4, proteus_enabled=False,
    ), executor=executor).run() == 0
    records = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert [r["status"] for r in records] == ["invalid_config", "compile_error", "runtime_error", "verified"]
