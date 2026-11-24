import hashlib
import json
from ctypes import Structure, c_uint
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List


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
class ExperimentResult:
    """
    Result record for a single Mneme record/replay experiment.

    This captures timing information, code size, resource usage, and basic
    execution outcome. It is designed to be easily serializable so that
    experiment runs can be logged and analyzed offline.

    Attributes
    ----------
    preprocess_ir_time : float
        Time spent in applying ``proteus`` specific optimizations.
    opt_time : float
        Time spent purging auto-init and optimizing the tracked build.
    codegen_time : float
        Time spent in the code generation / compilation phase.
    obj_size : int
        Size of the generated object or binary artifact.
    exec_time : list of int
        Execution time measurements for the replayed kernel, one entry per run.
    verified: bool
        Whether the experiment matched the results of the recorded execution.
    noop_verifies : bool
        Whether the recorded input alone matches the recorded output, meaning a
        kernel that does nothing would also verify.
    executed : bool
        Whether the experiment was executed at least once (without a crash).
    failed : bool
        Whether the experiment ultimately failed (e.g., compilation or runtime
        error).
    start_time : str
        ISO 8601 timestamp for when the experiment started.
    end_time : str
        ISO 8601 timestamp for when the experiment finished.
    gpu_id : int
        Identifier of the GPU device on which the experiment ran.
    const_mem_usage : int
        Amount of constant memory used by the generated kernel.
    local_mem_usage : int
        Amount of local memory used by the generated kernel.
    reg_usage : int
        Number of registers used per thread by the generated kernel.
    error : str
        Error description, usually set by the TunerHandler on crash
    """

    preprocess_ir_time: float = 0.0
    opt_time: float = 0.0
    codegen_time: float = 0.0
    obj_size: int = 0
    exec_time: List[int] = field(default_factory=list)
    verified: bool = False
    noop_verifies: bool = False
    executed: bool = False
    failed: bool = False
    start_time: str = ""
    end_time: str = ""
    gpu_id: int = 0
    const_mem_usage: int = 0
    local_mem_usage: int = 0
    reg_usage: int = 0
    error: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """
        Construct an experiment result from a plain dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing result fields.

        Returns
        -------
        ExperimentResult
            A new result instance initialized from ``data``.
        """
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result record to a plain dictionary.

        Returns
        -------
        dict
            A JSON-/YAML-serializable dictionary representation of the result.
        """
        return asdict(self)
