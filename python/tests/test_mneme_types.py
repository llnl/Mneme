import pytest
import json
import copy
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional
from mneme.mneme_types import (
    dim3,
    ExperimentConfiguration,
    BuildStats,
    ExperimentResult,
    RunResult,
    TimingResult,
    VerificationResult,
    _to_serializable,
    register_run_result,
)


def test_dim3_to_dict_and_from_dict_roundtrip():
    d = dim3(7, 8, 9)
    as_dict = d.to_dict()
    assert as_dict == {"x": 7, "y": 8, "z": 9}

    d2 = dim3.from_dict(as_dict)
    assert isinstance(d2, dim3)
    assert (d2.x, d2.y, d2.z) == (7, 8, 9)


def test_to_serializable_dim3():
    d = dim3(1, 2, 3)
    assert _to_serializable(d) == {"x": 1, "y": 2, "z": 3}


def test_to_serializable_sorts_dict_keys_deterministically():
    # Key property: dict keys are sorted in the output mapping.
    obj = {"b": 2, "a": 1, "c": 3}
    ser = _to_serializable(obj)
    assert list(ser.keys()) == ["a", "b", "c"]
    assert ser == {"a": 1, "b": 2, "c": 3}


def test_to_serializable_handles_nested_containers_and_dataclasses():
    cfg = ExperimentConfiguration(
        grid=dim3(2, 1, 1),
        block=dim3(16, 2, 1),
        shared_mem=123,
        specialize=True,
        set_launch_bounds=True,
        max_threads=256,
        min_blocks_per_sm=2,
        specialize_dims=True,
        passes="default<O2>",
        codegen_opt=2,
        prune=True,
        internalize=True,
    )

    ser = _to_serializable(
        {
            "cfg": cfg,
            "list": [dim3(1, 1, 1), {"z": 3, "y": 2, "x": 1}],
            "tuple": (1, 2, dim3(9, 9, 9)),
        }
    )

    assert ser["cfg"]["grid"] == {"x": 2, "y": 1, "z": 1}
    assert ser["cfg"]["block"] == {"x": 16, "y": 2, "z": 1}
    # Ensure inner dict got sorted
    assert list(ser["list"][1].keys()) == ["x", "y", "z"]
    assert ser["tuple"] == [1, 2, {"x": 9, "y": 9, "z": 9}]


def test_experiment_configuration_to_dict_shape_and_types():
    cfg = ExperimentConfiguration(grid=dim3(4, 5, 6), block=dim3(7, 8, 9))
    d = cfg.to_dict()

    # Must be JSON-serializable (no dim3 objects left)
    json.dumps(d)

    assert d["grid"] == {"x": 4, "y": 5, "z": 6}
    assert d["block"] == {"x": 7, "y": 8, "z": 9}
    assert isinstance(d["shared_mem"], int)
    assert isinstance(d["specialize"], bool)


def test_experiment_configuration_from_dict_accepts_dim3_or_dict():
    base = ExperimentConfiguration(grid=dim3(10, 1, 1), block=dim3(32, 1, 1))
    d = base.to_dict()

    # Case 1: grid/block encoded as dicts
    cfg1 = ExperimentConfiguration.from_dict(d)
    assert (cfg1.grid.x, cfg1.grid.y, cfg1.grid.z) == (10, 1, 1)
    assert (cfg1.block.x, cfg1.block.y, cfg1.block.z) == (32, 1, 1)

    # Case 2: grid/block as actual dim3 objects in the input dict
    d2 = dict(d)
    d2["grid"] = dim3(11, 2, 3)
    d2["block"] = dim3(64, 1, 1)
    cfg2 = ExperimentConfiguration.from_dict(d2)
    assert (cfg2.grid.x, cfg2.grid.y, cfg2.grid.z) == (11, 2, 3)
    assert (cfg2.block.x, cfg2.block.y, cfg2.block.z) == (64, 1, 1)


def test_experiment_configuration_is_valid_launch_bounds_constraint():
    # Invalid: set_launch_bounds=True but max_threads smaller than block volume
    cfg = ExperimentConfiguration(
        block=dim3(16, 16, 1),  # 256 threads
        set_launch_bounds=True,
        max_threads=128,
    )
    assert cfg.is_valid() is False

    # Valid: max_threads >= block volume
    cfg2 = ExperimentConfiguration(
        block=dim3(16, 16, 1),
        set_launch_bounds=True,
        max_threads=256,
    )
    assert cfg2.is_valid() is True

    # Valid: when set_launch_bounds is False, constraint is ignored
    cfg3 = ExperimentConfiguration(
        block=dim3(32, 32, 1),
        set_launch_bounds=False,
        max_threads=1,  # meaningless in this mode
    )
    assert cfg3.is_valid() is True


def test_experiment_configuration_ground_zeroes_unused_fields_only():
    cfg = ExperimentConfiguration(
        set_launch_bounds=False,
        max_threads=999,
        min_blocks_per_sm=7,
    )
    cfg.ground()
    assert cfg.max_threads == 0
    assert cfg.min_blocks_per_sm == 0

    cfg2 = ExperimentConfiguration(
        set_launch_bounds=True,
        max_threads=888,
        min_blocks_per_sm=6,
    )
    cfg2.ground()
    assert cfg2.max_threads == 888
    assert cfg2.min_blocks_per_sm == 6


def test_experiment_configuration_hash_stable_for_equivalent_configs():
    cfg = ExperimentConfiguration(
        grid=dim3(4096, 1, 1),
        block=dim3(256, 1, 1),
        set_launch_bounds=False,
        max_threads=123,  # should be ignored after grounding
        min_blocks_per_sm=456,  # should be ignored after grounding
    )

    cfg_a = copy.deepcopy(cfg)
    cfg_b = copy.deepcopy(cfg)

    # If you *intend* hashes to be stable across "unused" fields, caller must ground().
    cfg_a.ground()
    cfg_b.ground()

    assert cfg_a.hash() == cfg_b.hash()

    # Mutating a used field should change the hash
    cfg_b.shared_mem = cfg_b.shared_mem + 1
    assert cfg_a.hash() != cfg_b.hash()


def test_experiment_configuration_hash_deterministic_across_dict_key_order():
    cfg = ExperimentConfiguration(grid=dim3(1, 2, 3), block=dim3(4, 5, 6))
    h1 = cfg.hash()

    # Build a dict with the same content but shuffled insertion order,
    # reconstruct config, and ensure hash matches.
    d = cfg.to_dict()
    shuffled = {}
    for k in ["block", "grid"] + [kk for kk in d.keys() if kk not in ("grid", "block")]:
        shuffled[k] = d[k]

    cfg2 = ExperimentConfiguration.from_dict(shuffled)
    h2 = cfg2.hash()
    assert h1 == h2


PRE_RUN_CHAIN_RESULT = {
    "preprocess_ir_time": 0.5,
    "opt_time": 1.5,
    "codegen_time": 2.5,
    "obj_size": 4096,
    "exec_time": [7, 8, 9],
    "verified": True,
    "noop_verifies": True,
    "executed": True,
    "failed": False,
    "start_time": "2025-01-01T00:00:00",
    "end_time": "2025-01-01T00:00:01",
    "gpu_id": 1,
    "const_mem_usage": 1,
    "local_mem_usage": 2,
    "reg_usage": 3,
    "error": "",
}

VERIFICATION_BUILD_FIELDS = {
    "verification_preprocess_ir_time": 1.0,
    "verification_opt_time": 2.0,
    "verification_codegen_time": 3.0,
    "verification_obj_size": 4,
}


@register_run_result
@dataclass
class CounterResult(RunResult):
    kind: ClassVar[str] = "counters"

    values: Dict[str, int] = field(default_factory=dict)
    build: Optional[BuildStats] = None

    @classmethod
    def output_key(cls, name):
        return f"counters_{name}"


def _flat_result():
    return ExperimentResult(
        preprocess_ir_time=1.0,
        opt_time=2.0,
        codegen_time=3.0,
        obj_size=1234,
        exec_time=[10, 11, 12],
        verified=True,
        executed=True,
        failed=False,
        start_time="2025-01-01T00:00:00",
        end_time="2025-01-01T00:00:01",
        gpu_id=0,
        const_mem_usage=1,
        local_mem_usage=2,
        reg_usage=3,
        error="",
    )


def _short_circuited_result():
    verification = VerificationResult(build=BuildStats(opt_time=1.0, obj_size=8))
    return ExperimentResult([verification], executed=False)


@pytest.mark.parametrize(
    "make_result, executed_runs",
    [
        (_flat_result, ["verification", "timing"]),
        (_short_circuited_result, ["verification"]),
        (
            lambda: ExperimentResult([VerificationResult(verified=False)]),
            ["verification"],
        ),
        (
            lambda: ExperimentResult(
                [VerificationResult(verified=True), CounterResult(values={})],
                executed=True,
            ),
            ["verification", "counters"],
        ),
        (
            lambda: ExperimentResult.from_dict(PRE_RUN_CHAIN_RESULT),
            ["verification", "timing"],
        ),
        (
            lambda: ExperimentResult.from_dict(
                dict(PRE_RUN_CHAIN_RESULT, **VERIFICATION_BUILD_FIELDS)
            ),
            ["verification", "timing"],
        ),
    ],
    ids=[
        "both-runs",
        "timing-short-circuited",
        "failed-verification-without-build",
        "other-kind-with-default-values",
        "legacy-dict",
        "legacy-dict-with-verification-build",
    ],
)
def test_experiment_result_to_dict_from_dict_roundtrip_and_defaults(
    make_result, executed_runs
):
    r = make_result()
    d = r.to_dict()
    json.dumps(d)  # serializable
    assert d["runs"] == executed_runs

    r2 = ExperimentResult.from_dict(d)
    assert r2 == r
    assert list(r2.runs) == executed_runs
    if "timing" in executed_runs:
        assert r2.exec_time is r2.timing.exec_time

    # to_dict and from_dict copy, so neither side aliases the other.
    d["exec_time"].append(-1)
    r2.to_dict()["exec_time"].append(-2)
    assert -1 not in r2.exec_time and -2 not in r2.exec_time
    assert -1 not in r.exec_time


@pytest.mark.parametrize(
    "verification_build",
    [None, BuildStats(1.0, 2.0, 3.0, 4)],
    ids=["default", "recorded"],
)
def test_experiment_result_serializes_each_run_build_under_its_own_fields(
    verification_build,
):
    verification = VerificationResult(
        build=verification_build, verified=True, noop_verifies=False
    )
    timing = TimingResult(
        build=BuildStats(
            preprocess_ir_time=5.0, opt_time=6.0, codegen_time=7.0, obj_size=8
        ),
        exec_time=[10, 11],
        reg_usage=12,
    )
    result = ExperimentResult([verification, timing], executed=True)
    d = result.to_dict()

    recorded = VERIFICATION_BUILD_FIELDS if verification_build else {}
    assert set(d) == {*PRE_RUN_CHAIN_RESULT, *recorded, "runs"}
    assert {name: d[name] for name in recorded} == recorded
    assert (d["preprocess_ir_time"], d["opt_time"], d["codegen_time"]) == (5.0, 6.0, 7.0)
    assert d["obj_size"] == 8
    restored = ExperimentResult.from_dict(d)
    assert restored.verification == verification
    assert restored.timing == timing
    if verification_build:
        assert result.verification_opt_time == 2.0
    else:
        with pytest.raises(AttributeError, match="record_build_stats=True"):
            result.verification_opt_time

    skipped = ExperimentResult([verification]).to_dict()
    assert skipped["exec_time"] == []
    assert (skipped["obj_size"], skipped["opt_time"], skipped["reg_usage"]) == (0, 0.0, 0)
    assert skipped["runs"] == ["verification"]


def test_experiment_result_from_dict_loads_pre_run_chain_results():
    r = ExperimentResult.from_dict(PRE_RUN_CHAIN_RESULT)

    assert r.verification == VerificationResult(verified=True, noop_verifies=True)
    assert r.timing == TimingResult(
        build=BuildStats(
            preprocess_ir_time=0.5, opt_time=1.5, codegen_time=2.5, obj_size=4096
        ),
        exec_time=[7, 8, 9],
        const_mem_usage=1,
        local_mem_usage=2,
        reg_usage=3,
    )
    assert (r.verified, r.exec_time, r.opt_time, r.gpu_id) == (True, [7, 8, 9], 1.5, 1)
    assert r.to_dict() == dict(PRE_RUN_CHAIN_RESULT, runs=["verification", "timing"])


def test_experiment_result_writes_other_run_kinds_only_when_executed():
    without = ExperimentResult([VerificationResult(verified=True)]).to_dict()
    assert set(without) == set(PRE_RUN_CHAIN_RESULT) | {"runs"}

    counters = CounterResult(build=BuildStats(obj_size=8), values={"a": 1})
    d = ExperimentResult([counters]).to_dict()
    assert {k: v for k, v in d.items() if k.startswith("counters_")} == {
        "counters_values": {"a": 1},
        "counters_preprocess_ir_time": 0.0,
        "counters_opt_time": 0.0,
        "counters_codegen_time": 0.0,
        "counters_obj_size": 8,
    }

    d["runs"] = ["sampling"]
    with pytest.raises(ValueError, match="'sampling'.*import the module"):
        ExperimentResult.from_dict(d)


@pytest.mark.parametrize(
    "field_name, taken_by",
    [
        ("exec_time", "run kind 'timing'"),
        ("noop_verifies", "run kind 'verification'"),
        ("gpu_id", "an ExperimentResult field"),
        ("runs", "the ExperimentResult list of executed runs"),
    ],
)
def test_register_run_result_rejects_colliding_keys(field_name, taken_by):
    namespace = {"__annotations__": {"kind": ClassVar[str], field_name: int}}
    namespace.update(kind="clashing", **{field_name: 0})
    clashing = dataclass(type("ClashingResult", (RunResult,), namespace))
    with pytest.raises(
        TypeError, match=rf"'clashing'.*'{field_name}' \({taken_by}\)"
    ):
        register_run_result(clashing)

    with pytest.raises(TypeError, match="more than one field to: opt_time"):

        @register_run_result
        @dataclass
        class TwiceResult(RunResult):
            kind: ClassVar[str] = "twice"

            opt_time: float = 0.0
            build: Optional[BuildStats] = None
