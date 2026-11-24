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


_EXPERIMENT_FIELDS = ("executed", "failed", "start_time", "end_time", "gpu_id", "error")
_RUN_RESULT_TYPES: Dict[str, type] = {}
_FLAT_FIELD_OWNERS: Dict[str, type] = {}


@dataclass
class RunResult:
    """
    Result of one run of a replay run chain.

    Every subclass names the run ``kind`` it belongs to and is registered under
    it with :func:`register_run_result`. A run result serializes into the flat
    :class:`ExperimentResult` dictionary: its own fields under their names, and its build measurements
    under ``build_field_prefix`` followed by the :class:`BuildStats` field name.

    Attributes
    ----------
    build : BuildStats
        Measurements of the build this run launched.
    """

    kind: ClassVar[str]
    build_field_prefix: ClassVar[str]

    build: BuildStats = field(default_factory=BuildStats)

    @classmethod
    def _own_field_names(cls) -> List[str]:
        return [f.name for f in fields(cls) if f.name != "build"]

    @classmethod
    def _build_field_names(cls) -> Dict[str, str]:
        return {cls.build_field_prefix + f.name: f.name for f in fields(BuildStats)}

    @classmethod
    def flat_field_names(cls) -> List[str]:
        """Names this result kind owns in the flat experiment dictionary."""
        return cls._own_field_names() + list(cls._build_field_names())

    def to_flat(self) -> Dict[str, Any]:
        flat = {name: getattr(self, name) for name in self._own_field_names()}
        for flat_name, name in self._build_field_names().items():
            flat[flat_name] = getattr(self.build, name)
        return flat

    @classmethod
    def from_flat(cls, data: Mapping[str, Any]) -> "RunResult":
        """Build a result from flat fields; absent fields keep their defaults."""
        own = {name: data[name] for name in cls._own_field_names() if name in data}
        build = {
            name: data[flat_name]
            for flat_name, name in cls._build_field_names().items()
            if flat_name in data
        }
        return cls(build=BuildStats(**build), **own)


def register_run_result(cls):
    """
    Class decorator registering a :class:`RunResult` subclass under its ``kind``.

    Apply it above ``@dataclass``. Only registered result kinds can be stored in
    an :class:`ExperimentResult`. A kind that is already registered, or a flat
    field name that another run kind or the experiment already uses, raises
    ``TypeError`` when the class is defined.
    """
    if cls.kind in _RUN_RESULT_TYPES:
        raise TypeError(f"Run result kind {cls.kind!r} is already registered")
    for name in cls.flat_field_names():
        owner = _FLAT_FIELD_OWNERS.get(name)
        if owner is not None or name in _EXPERIMENT_FIELDS:
            used_by = f"run kind {owner.kind!r}" if owner else "the experiment"
            raise TypeError(
                f"Result field {name!r} of run kind {cls.kind!r} is already "
                f"used by {used_by}"
            )
    _RUN_RESULT_TYPES[cls.kind] = cls
    for name in cls.flat_field_names():
        _FLAT_FIELD_OWNERS[name] = cls
    return cls


@register_run_result
@dataclass
class VerificationResult(RunResult):
    """
    Result of a verification run.

    Attributes
    ----------
    verified : bool
        Whether the replay matched the results of the recorded execution.
    noop_verifies : bool
        Whether the recorded input alone matches the recorded output, meaning a
        kernel that does nothing would also verify.
    """

    kind: ClassVar[str] = "verification"
    build_field_prefix: ClassVar[str] = "verification_"

    verified: bool = False
    noop_verifies: bool = False


@register_run_result
@dataclass
class TimingResult(RunResult):
    """
    Result of a timing run.

    Attributes
    ----------
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
    build_field_prefix: ClassVar[str] = ""

    exec_time: List[int] = field(default_factory=list)
    const_mem_usage: int = 0
    local_mem_usage: int = 0
    reg_usage: int = 0


class ExperimentResult:
    """
    Result record for a single Mneme record/replay experiment.

    An experiment executes a chain of runs. ``runs`` holds the typed result of
    each run by kind. The fields of every run result are also readable directly
    on the experiment result (``result.exec_time``, ``result.verified``), and
    :meth:`to_dict` writes them into one flat dictionary; this flat form is what
    crosses process boundaries and what tuning persists. A run that did not
    execute reads and serializes as its default values. Built from flat fields
    (as :meth:`from_dict` does), a run kind whose fields all equal those
    defaults is treated as not executed and left out of ``runs``. This is exact
    for the timing run, whose ``exec_time`` is never empty once it executed. A
    legacy dictionary without the ``verification_`` fields whose verification
    failed with ``noop_verifies`` false therefore reads as having no
    verification run; its flat reads return the same values either way.

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

    Flat run fields: ``verified``, ``noop_verifies`` and the
    ``verification_``-prefixed build measurements come from the verification
    run; ``exec_time``, ``reg_usage``, ``const_mem_usage``, ``local_mem_usage``
    and the un-prefixed build measurements (``preprocess_ir_time``,
    ``opt_time``, ``codegen_time``, ``obj_size``) come from the timing run.
    They cannot be assigned on the experiment result; set them on the run
    result. Reading one of an executed run returns the stored value; reading
    one of a run that did not execute returns a fresh default.
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
            Typed results of the runs that executed, at most one per kind.
        run_fields
            Flat run fields (see the class docstring). The fields of one run
            kind build that kind's result unless they all equal the defaults;
            a kind cannot be given both ways.
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

        unknown = sorted(set(run_fields) - set(_FLAT_FIELD_OWNERS))
        if unknown:
            raise TypeError(f"Unknown experiment result fields: {', '.join(unknown)}")
        for result_type in {_FLAT_FIELD_OWNERS[name] for name in run_fields}:
            if result_type.kind in self.runs:
                raise ValueError(
                    f"Run kind {result_type.kind!r} was given both as a result "
                    "and as fields"
                )
            run_result = result_type.from_flat(run_fields)
            if run_result.to_flat() != result_type().to_flat():
                self.runs[result_type.kind] = run_result

    def __getattr__(self, name: str) -> Any:
        owner = _FLAT_FIELD_OWNERS.get(name)
        if owner is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        run_result = self.__dict__.get("runs", {}).get(owner.kind) or owner()
        return run_result.to_flat()[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _FLAT_FIELD_OWNERS:
            raise AttributeError(
                f"{name!r} belongs to the {_FLAT_FIELD_OWNERS[name].kind!r} run "
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """
        Construct an experiment result from a flat dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing result fields, as written by :meth:`to_dict`.
            Absent fields keep their default values, so dictionaries written
            before a field existed still load. The values are copied.

        Returns
        -------
        ExperimentResult
            A new result instance initialized from ``data``.
        """
        return cls(**copy.deepcopy(data))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result record to a flat dictionary.

        Returns
        -------
        dict
            A JSON-/YAML-serializable dictionary representation of the result.
        """
        flat: Dict[str, Any] = {}
        for kind, result_type in _RUN_RESULT_TYPES.items():
            run_result = self.runs.get(kind) or result_type()
            flat.update(copy.deepcopy(run_result.to_flat()))
        for name in _EXPERIMENT_FIELDS:
            flat[name] = getattr(self, name)
        return flat
