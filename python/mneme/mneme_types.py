import copy
import hashlib
import json
from ctypes import Structure, c_uint
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional


def _to_serializable(obj: Any) -> Any:
    """
    Turn arbitrary config objects into something JSON-serializable and
    deterministically ordered, so hashing is stable across runs.
    """
    # Special case: your dim3
    if isinstance(obj, dim3):
        return obj.to_dict()

    # Nested dataclasses
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _to_serializable(value)
        return result

    # Containers
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    if isinstance(obj, dict):
        # Sort keys to make it deterministic
        return {k: _to_serializable(obj[k]) for k in sorted(obj.keys())}

    # Everything else: ints, floats, bools, strings, None...
    return obj


class dim3(Structure):
    _fields_ = [("x", c_uint), ("y", c_uint), ("z", c_uint)]

    def __init__(self, x=1, y=1, z=1):
        super().__init__(x, y, z)

    def __repr__(self):
        return f"dim3({self.x}, {self.y}, {self.z})"

    def to_dict(self):
        return {"x": int(self.x), "y": int(self.y), "z": int(self.z)}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class ExperimentConfiguration:
    """
    Configuration for a single Mneme record/replay experiment.

    This object captures all knobs that control kernel launch configuration,
    specialization strategy, and code generation behavior. It is intended to be
    hashable (via :meth:`hash`) so the same configuration can be given a stable,
    persistent identifier across runs.

    Attributes
    ----------
    grid : dim3
        Grid dimensions (x, y, z) of the kernel launch.
    block : dim3
        Block dimensions (x, y, z) of the kernel launch.
    shared_mem : int
        Amount of dynamic shared memory to allocate for the launch.
    specialize : bool
        Whether to enable specialization based on the recorded execution
        (e.g., specializing on input sizes or recorded parameters).
    set_launch_bounds : bool
        Whether to explicitly set CUDA launch bounds for the generated kernel.
    max_threads : int
        Maximum number of threads per block to assume when setting launch bounds
        or during specialization.
    min_blocks_per_sm : int
        Minimum number of resident blocks per SM when computing launch bounds.
    specialize_dims : bool
        Whether to specialize based on the recorded grid/block dimensions.
    passes : str
        Optimization pass pipeline specification, e.g. ``"default<O3>"``.
    codegen_opt : int
        Code generation optimization level (e.g., 0–3).
    prune : bool
        Whether to enable IR pruning / dead-code elimination in the generated
        kernel. This is mandatory always true. We will explore later the impact of it.
    internalize : bool
        Whether to internalize symbols (e.g., limit symbol visibility) during
        code generation. This is mandatory always true. We will explore later the impact of it.
    """

    grid: dim3 = field(default_factory=dim3)
    block: dim3 = field(default_factory=dim3)
    shared_mem: int = 0
    specialize: bool = False
    set_launch_bounds: bool = False
    max_threads: int = 1024
    min_blocks_per_sm: int = 1
    specialize_dims: bool = False
    passes: str = "default<O3>"
    codegen_opt: int = 3
    prune: bool = True
    internalize: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfiguration":
        """
        Construct a configuration from a plain dictionary.

        The dictionary is expected to contain JSON-/YAML-serializable
        representations, with ``"grid"`` and ``"block"`` encoded as dictionaries
        compatible with :meth:`dim3.from_dict`.

        Parameters
        ----------
        data : dict
            Dictionary containing configuration fields.

        Returns
        -------
        ExperimentConfiguration
            A new configuration instance initialized from ``data``.
        """
        data = dict(data)

        grid = data.pop("grid")
        block = data.pop("block")

        if isinstance(grid, dim3):
            grid_obj = grid
        else:
            grid_obj = dim3.from_dict(grid)

        if isinstance(block, dim3):
            block_obj = block
        else:
            block_obj = dim3.from_dict(block)

        return cls(grid=grid_obj, block=block_obj, **data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a plain dictionary.

        Returns
        -------
        dict
            A JSON-/YAML-serializable dictionary representation of the
            configuration, suitable for persistence or hashing.
        """

        return _to_serializable(self)

    def is_valid(self):
        """
        Checks if the configuration follows device constraints

        Returns
        -------
        bool
            A boolean value indicating whether this is a valid configuration
        """
        if self.set_launch_bounds and (
            self.max_threads < (self.block.x * self.block.y * self.block.z)
        ):
            return False
        return True

    def ground(self):
        """
        Values that are 'unused' are set on default values. This can help when hashing configs

        Returns
        -------
            None
                This function does not return anyhing, but modifies the contents of the class in place
        """

        if not self.set_launch_bounds:
            self.max_threads = 0
            self.min_blocks_per_sm = 0

    def hash(self) -> str:
        """
        Compute a stable SHA-256 hash of the full configuration.

        The hash is computed from a normalized, JSON-serializable view of the
        configuration so that identical configurations produce the same digest
        across processes and runs.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest of the configuration.
        """
        serializable = _to_serializable(self)
        # sort_keys + compact separators => deterministic string
        payload = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BuildStats:
    """
    Measurements of one build of the replayed kernel.

    Attributes
    ----------
    preprocess_ir_time : float
        Time spent in applying ``proteus`` specific optimizations.
    opt_time : float
        Time spent in the optimization phase.
    codegen_time : float
        Time spent in the code generation / compilation phase.
    obj_size : int
        Size of the generated object.
    """

    preprocess_ir_time: float = 0.0
    opt_time: float = 0.0
    codegen_time: float = 0.0
    obj_size: int = 0


_BUILD_STAT_NAMES = tuple(f.name for f in fields(BuildStats))
_EXPERIMENT_FIELDS = ("executed", "failed", "start_time", "end_time", "gpu_id", "error")
_RUNS_KEY = "runs"
_RUN_RESULT_TYPES: Dict[str, type] = {}
_RUN_RESULT_KEY_OWNERS: Dict[str, type] = {}


@dataclass
class RunResult:
    """
    Result of one run of a replay run chain.

    Every subclass names the run ``kind`` it belongs to and is registered under
    it with :func:`register_run_result`. In the flat :class:`ExperimentResult`
    dictionary, a result writes each of its fields under the key
    :meth:`output_key` gives the field's name. A subclass that reports a build
    declares a field ``build`` of type ``Optional[BuildStats]``; it is written
    as one key per measurement, named by :meth:`output_key` of the measurement
    name, and not at all while it is ``None``. Runs that ask for the same build
    share it, so a run reports the build only when its result is meant to
    account for it.

    Attributes
    ----------
    always_written : bool
        Class attribute. Whether an experiment result writes this kind's keys
        with their default values when the run did not execute. Off by default:
        a kind's keys are then written only when its run executed.
    """

    kind: ClassVar[str]
    always_written: ClassVar[bool] = False
    unrecorded_build_hint: ClassVar[str] = "the run did not report its build"

    @classmethod
    def output_key(cls, name: str) -> str:
        """
        The flat result key of field or build measurement ``name``.

        The default is ``name`` itself. Keys must be unique among all
        registered kinds and the experiment's own fields, so a kind overrides
        this to name keys that would otherwise collide.
        """
        return name

    @classmethod
    def _reports_build(cls) -> bool:
        return any(f.name == "build" for f in fields(cls))

    @classmethod
    def _field_keys(cls) -> Dict[str, str]:
        return {
            f.name: cls.output_key(f.name) for f in fields(cls) if f.name != "build"
        }

    @classmethod
    def _build_keys(cls) -> Dict[str, str]:
        if not cls._reports_build():
            return {}
        return {name: cls.output_key(name) for name in _BUILD_STAT_NAMES}

    @classmethod
    def _flat_keys(cls) -> List[str]:
        return list(cls._field_keys().values()) + list(cls._build_keys().values())

    def to_flat(self) -> Dict[str, Any]:
        """This result's flat fields, with build measurements if ``build`` is set."""
        flat = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name != "build":
                flat[self.output_key(f.name)] = value
            elif value is not None:
                for name in _BUILD_STAT_NAMES:
                    flat[self.output_key(name)] = getattr(value, name)
        return flat

    @classmethod
    def from_flat(cls, data: Mapping[str, Any]) -> "RunResult":
        """
        Build a result from flat fields.

        Absent fields keep their defaults. ``build`` is set when any build
        measurement of this kind is present.
        """
        own = {
            name: data[key] for name, key in cls._field_keys().items() if key in data
        }
        build = {
            name: data[key] for name, key in cls._build_keys().items() if key in data
        }
        if build:
            own["build"] = BuildStats(**build)
        return cls(**own)

    @classmethod
    def _flat_values_are_defaults(cls, data: Mapping[str, Any]) -> bool:
        defaults = cls(build=BuildStats()) if cls._reports_build() else cls()
        return all(
            data[key] == value
            for key, value in defaults.to_flat().items()
            if key in data
        )


def register_run_result(cls):
    """
    Class decorator registering a :class:`RunResult` subclass under its ``kind``.

    Apply it above ``@dataclass``. Only registered result kinds can be stored in
    an :class:`ExperimentResult` or loaded by :meth:`ExperimentResult.from_dict`.
    Defining the class raises ``TypeError`` if the kind is already registered,
    if two of its fields map to the same flat key, or if one of its keys is
    written by another registered kind or by the experiment itself.
    """
    if cls.kind in _RUN_RESULT_TYPES:
        raise TypeError(f"Run result kind {cls.kind!r} is already registered")
    keys = cls._flat_keys()
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise TypeError(
            f"Run kind {cls.kind!r} maps more than one field to: {', '.join(repeated)}"
        )
    collisions = []
    for key in keys:
        if key in _EXPERIMENT_FIELDS:
            collisions.append(f"{key!r} (an ExperimentResult field)")
        elif key == _RUNS_KEY:
            collisions.append(f"{key!r} (the ExperimentResult list of executed runs)")
        elif key in _RUN_RESULT_KEY_OWNERS:
            owner = _RUN_RESULT_KEY_OWNERS[key]
            collisions.append(f"{key!r} (run kind {owner.kind!r})")
    if collisions:
        raise TypeError(
            f"Run kind {cls.kind!r} writes result keys that are already taken: "
            f"{', '.join(collisions)}; override {cls.__name__}.output_key to "
            "name them"
        )
    _RUN_RESULT_TYPES[cls.kind] = cls
    _RUN_RESULT_KEY_OWNERS.update(dict.fromkeys(keys, cls))
    return cls


@register_run_result
@dataclass
class VerificationResult(RunResult):
    """
    Result of a verification run.

    Its fields are written under their own names and its build measurements as
    ``verification_<measurement>``.

    Attributes
    ----------
    verified : bool
        Whether the replay matched the results of the recorded execution.
    noop_verifies : bool
        Whether the recorded input alone matches the recorded output, meaning a
        kernel that does nothing would also verify.
    build : BuildStats or None
        Measurements of the verification build, set only when the run was
        constructed with ``VerificationRun(record_build_stats=True)``.
    """

    kind: ClassVar[str] = "verification"
    always_written: ClassVar[bool] = True
    unrecorded_build_hint: ClassVar[str] = (
        "only a verification run constructed with "
        "VerificationRun(record_build_stats=True) records it"
    )

    verified: bool = False
    noop_verifies: bool = False
    build: Optional[BuildStats] = None

    @classmethod
    def output_key(cls, name: str) -> str:
        return f"verification_{name}" if name in _BUILD_STAT_NAMES else name


@register_run_result
@dataclass
class TimingResult(RunResult):
    """
    Result of a timing run.

    Its fields and build measurements are written under their own names.

    Attributes
    ----------
    build : BuildStats
        Measurements of the timing build, always reported.
    exec_time : list of int
        Execution time measurements for the replayed kernel, one entry per
        launch, warmup launches included.
    const_mem_usage : int
        Amount of constant memory used by the generated kernel.
    local_mem_usage : int
        Amount of local memory used by the generated kernel.
    reg_usage : int
        Number of registers used per thread by the generated kernel.
    """

    kind: ClassVar[str] = "timing"
    always_written: ClassVar[bool] = True

    build: BuildStats = field(default_factory=BuildStats)
    exec_time: List[int] = field(default_factory=list)
    const_mem_usage: int = 0
    local_mem_usage: int = 0
    reg_usage: int = 0


class ExperimentResult:
    """
    Result record for a single Mneme record/replay experiment.

    An experiment executes a chain of runs. ``runs`` holds the typed result of
    each run that executed, by kind, in chain order.

    :meth:`to_dict` writes one flat dictionary; this flat form is what crosses
    process boundaries and what tuning persists. It holds, in this order:

    - The experiment fields (``executed``, ``failed``, ``start_time``,
      ``end_time``, ``gpu_id``, ``error``).
    - The keys of each run that executed, in chain order, named by its result
      type (see :class:`RunResult`).
    - The keys of each kind whose result type sets ``always_written`` but whose
      run did not execute, with their default values. The verification and
      timing results set it, so their fields are always present.
    - ``runs``: the kinds of the runs that executed, in chain order.

    With the default chain, verification writes ``verified`` and
    ``noop_verifies``, plus ``verification_preprocess_ir_time``,
    ``verification_opt_time``, ``verification_codegen_time`` and
    ``verification_obj_size`` when it recorded its build; timing writes
    ``preprocess_ir_time``, ``opt_time``, ``codegen_time``, ``obj_size``,
    ``exec_time``, ``const_mem_usage``, ``local_mem_usage`` and ``reg_usage``.

    :meth:`from_dict` reads ``runs`` to know which runs executed. A dictionary
    without it, written before run chains existed, is read as having a run of
    a kind when any of that kind's keys holds a value other than its default.
    Such a dictionary whose verification failed with ``noop_verifies`` false
    therefore reads as having no verification run; its flat reads return the
    same values either way.

    Every run result key can be read directly on the experiment result
    (``result.exec_time``, ``result.verified``). Reading one of an executed run
    returns the stored value; reading one of a run that did not execute
    returns a fresh default if its kind is always written and raises
    ``AttributeError`` otherwise, as does reading a build measurement that was
    not recorded. These keys cannot be assigned on the experiment result; set
    them on the run result.

    Attributes
    ----------
    runs : dict of str to RunResult
        Typed run results by run kind.
    executed : bool
        Whether every run of the chain executed (without a crash).
    failed : bool
        Whether the experiment ultimately failed (e.g., compilation or runtime
        error).
    start_time : str
        ISO 8601 timestamp for when the experiment started.
    end_time : str
        ISO 8601 timestamp for when the experiment finished.
    gpu_id : int
        Identifier of the GPU device on which the experiment ran.
    error : str
        Error description, usually set by the TunerHandler on crash
    """

    def __init__(
        self,
        runs: Iterable[RunResult] = (),
        *,
        executed: bool = False,
        failed: bool = False,
        start_time: str = "",
        end_time: str = "",
        gpu_id: int = 0,
        error: str = "",
        **run_fields: Any,
    ):
        """
        Parameters
        ----------
        runs : iterable of RunResult
            Typed results of the runs that executed, at most one per kind, in
            chain order.
        run_fields
            Run result fields by their flat keys, read the way
            :meth:`from_dict` reads a dictionary without ``runs``. A kind
            cannot be given both ways.
        """
        self.runs: Dict[str, RunResult] = {}
        for run_result in runs:
            if _RUN_RESULT_TYPES.get(run_result.kind) is not type(run_result):
                raise TypeError(
                    f"{type(run_result).__name__} is not the registered result "
                    f"type of run kind {run_result.kind!r}"
                )
            if run_result.kind in self.runs:
                raise ValueError(f"Duplicate result for run kind {run_result.kind!r}")
            self.runs[run_result.kind] = run_result
        self.executed = executed
        self.failed = failed
        self.start_time = start_time
        self.end_time = end_time
        self.gpu_id = gpu_id
        self.error = error

        unknown = sorted(set(run_fields) - set(_RUN_RESULT_KEY_OWNERS))
        if unknown:
            raise TypeError(f"Unknown experiment result fields: {', '.join(unknown)}")
        for kind, result_type in _RUN_RESULT_TYPES.items():
            given = {
                key: value
                for key, value in run_fields.items()
                if _RUN_RESULT_KEY_OWNERS[key] is result_type
            }
            if not given:
                continue
            if kind in self.runs:
                raise ValueError(
                    f"Run kind {kind!r} was given both as a result and as fields"
                )
            if not result_type._flat_values_are_defaults(given):
                self.runs[kind] = result_type.from_flat(given)

    def __getattr__(self, name: str) -> Any:
        owner = _RUN_RESULT_KEY_OWNERS.get(name)
        if owner is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        run_result = self.__dict__.get("runs", {}).get(owner.kind)
        if run_result is None:
            if not owner.always_written:
                raise AttributeError(
                    f"{name!r} belongs to the {owner.kind!r} run, which did not execute"
                )
            run_result = owner()
        flat = run_result.to_flat()
        if name not in flat:
            raise AttributeError(
                f"{name!r} was not recorded; {owner.unrecorded_build_hint}"
            )
        return flat[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RUN_RESULT_KEY_OWNERS:
            raise AttributeError(
                f"{name!r} belongs to the {_RUN_RESULT_KEY_OWNERS[name].kind!r} run "
                "result; set it there"
            )
        super().__setattr__(name, value)

    @property
    def verification(self) -> Optional[VerificationResult]:
        """The verification run's result, or ``None`` if it did not execute."""
        return self.runs.get(VerificationResult.kind)

    @property
    def timing(self) -> Optional[TimingResult]:
        """The timing run's result, or ``None`` if it did not execute."""
        return self.runs.get(TimingResult.kind)

    @property
    def verified(self) -> bool:
        """Whether a verification run executed and the replay matched the recording."""
        return self.verification is not None and self.verification.verified

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExperimentResult):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        fields_repr = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ExperimentResult({fields_repr})"

    def _unexecuted_defaults(self) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for kind, result_type in _RUN_RESULT_TYPES.items():
            if result_type.always_written and kind not in self.runs:
                flat.update(result_type().to_flat())
        return flat

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """
        Construct an experiment result from a flat dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing result fields, as written by :meth:`to_dict`
            (see the class docstring). Absent fields keep their default
            values, so dictionaries written before a field existed still load.
            The values are copied.

        Returns
        -------
        ExperimentResult
            A new result instance initialized from ``data``.

        Raises
        ------
        ValueError
            If ``runs`` names a kind whose result type is not registered.
        TypeError
            If ``data`` holds a key that no experiment field or listed run owns.
        """
        data = copy.deepcopy(dict(data))
        kinds = data.pop(_RUNS_KEY, None)
        if kinds is None:
            return cls(**data)
        experiment_fields = {
            name: data.pop(name) for name in _EXPERIMENT_FIELDS if name in data
        }
        runs = []
        owned_keys = set()
        for kind in kinds:
            result_type = _RUN_RESULT_TYPES.get(kind)
            if result_type is None:
                raise ValueError(
                    f"Unknown run kind {kind!r} in experiment result; import the "
                    "module that defines its result type (registered with "
                    "@register_run_result) before loading the result"
                )
            runs.append(result_type.from_flat(data))
            owned_keys.update(result_type._flat_keys())
        result = cls(runs, **experiment_fields)
        unknown = sorted(set(data) - owned_keys - set(result._unexecuted_defaults()))
        if unknown:
            raise TypeError(f"Unknown experiment result fields: {', '.join(unknown)}")
        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result record to a flat dictionary.

        Returns
        -------
        dict
            A JSON-/YAML-serializable dictionary representation of the result
            (see the class docstring for its keys).
        """
        flat = {name: getattr(self, name) for name in _EXPERIMENT_FIELDS}
        for run_result in self.runs.values():
            flat.update(run_result.to_flat())
        flat.update(self._unexecuted_defaults())
        flat[_RUNS_KEY] = list(self.runs)
        return copy.deepcopy(flat)
