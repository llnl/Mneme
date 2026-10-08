"""
Recorded execution database and memory snapshot bindings.

This module defines the Python-side representation of Mneme’s record/replay
artifacts:

  * **MemStateRef**: a lightweight wrapper over the native memory-snapshot object
    (prologue/epilogue) used for replay verification.
  * **RecordedExecution**: a JSON-serializable description of a recorded kernel,
    including all observed dynamic instances and the LLVM IR modules required
    for replay.

The native snapshot API is accessed via ctypes/FFI (``ffi.lib.MnemePy_*``).
Instances of :class:`MemStateRef` behave as context managers: they load the
snapshot on entry and dispose the native handle on exit.

Notes
-----
- This file is *core* to replay correctness: prologue/epilogue snapshots are used
  to verify that the replayed kernel produced the expected state.
- The JSON schema here is treated as a stable interchange format between record
  and replay tools.
"""

import hashlib
import json
from ctypes import POINTER, c_bool, c_char_p, c_int, c_void_p
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from .llvm import ffi
from .mneme_types import dim3
from .proteus import jit

MnemeRecordStateRef = ffi._make_opaque_ref("MnemeRecordState")
ffi.lib.MnemePy_initializeMemState.argtypes = [c_char_p, c_char_p, c_char_p, c_bool]
ffi.lib.MnemePy_initializeMemState.restype = MnemeRecordStateRef

ffi.lib.MnemePy_DisposeMemState.argtypes = [MnemeRecordStateRef]

ffi.lib.MnemePy_LoadMemState.argtypes = [MnemeRecordStateRef]

ffi.lib.MnemePy_CompareMemState.argtypes = [MnemeRecordStateRef, MnemeRecordStateRef]
ffi.lib.MnemePy_CompareMemState.restype = c_bool

ffi.lib.MnemePy_MatchesUnlaunched.argtypes = [MnemeRecordStateRef, MnemeRecordStateRef]
ffi.lib.MnemePy_MatchesUnlaunched.restype = c_bool

ffi.lib.MnemePy_ResetMemState.argtypes = [MnemeRecordStateRef]

ffi.lib.MnemePy_getNumArgs.argtypes = [MnemeRecordStateRef]
ffi.lib.MnemePy_getNumArgs.restype = c_int

ffi.lib.MnemePy_getArgs.argtypes = [MnemeRecordStateRef]
ffi.lib.MnemePy_getArgs.restype = POINTER(c_void_p)


class SnapshotType(Enum):
    """
    Enumeration of memory snapshot roles within a recorded execution.

    PROLOGUE
        Snapshot captured immediately before kernel execution.
    EPILOGUE
        Snapshot captured immediately after kernel execution.

    The snapshot type is used by the native snapshot loader to interpret the
    record format and to decide which parts of state are treated as inputs vs
    outputs during verification.
    """

    PROLOGUE = 1
    EPILOGUE = 2


def find_non_jsonables(obj, where="$"):
    """
    Debug helper: print paths to fields that are not JSON-serializable.

    This is used as a sanity check before writing the record database to JSON.
    It is intentionally permissive and prints to stdout rather than raising.

    Parameters
    ----------
    obj : Any
        Object graph to inspect.
    where : str
        JSONPath-like location used when printing offending fields.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return
    if isinstance(obj, Path):
        print("Non-JSON type:", where, "->", obj, type(obj))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_non_jsonables(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            find_non_jsonables(v, f"{where}[{i}]")
    else:
        # add other allowed conversions here if you plan to support them
        print("Non-JSON type:", where, "->", type(obj))


def _make_path_relative(p, base_dir):
    """
    Return ``p`` as a basename if it lives directly under ``base_dir``.

    If ``p`` is already a basename, return it unchanged so this transform is
    idempotent. Relative paths with directory components are ambiguous here
    because JSON readers resolve them against ``base_dir``.
    """
    if base_dir is None or not p:
        return p
    pp = Path(p)
    if not pp.is_absolute():
        if pp.parent == Path("."):
            return p
        raise ValueError(f"Expected absolute path or basename, got relative path: {p}")
    if pp.parent.resolve() == base_dir:
        return pp.name
    return p


class MemStateRef:
    """
    Handle to a recorded memory snapshot (prologue/epilogue).

    A :class:`MemStateRef` wraps Mneme’s native memory snapshot representation,
    which encodes the recorded kernel argument pointers and the captured device
    memory state. During replay, these snapshots serve two purposes:

      1. **Inputs**: The prologue snapshot provides the argument pointer list and
         initial memory contents required to execute the kernel deterministically.
      2. **Verification**: The epilogue snapshot represents the expected post-kernel
         state. Replay compares a reproduced epilogue against this snapshot to
         validate correctness.

    Instances are context managers. Entering the context loads the snapshot
    into the native handle; leaving the context disposes it.

    Parameters
    ----------
    fn : str
        Path to the snapshot file on disk.
    kernel_name : str
        Kernel name associated with this snapshot (used by native layer).
    snap_type : SnapshotType
        Whether this snapshot is a prologue or epilogue capture.

    Raises
    ------
    RuntimeError
        If the snapshot file does not exist.
    """

    def __init__(
        self,
        fn: str,
        kernel_name: str,
        snap_type: SnapshotType,
        base_prologue_fn: Optional[str] = None,
    ):
        if not Path(fn).exists():
            raise RuntimeError(f"Expected snapshot file: {fn} to exist")
        if base_prologue_fn is not None and not Path(base_prologue_fn).exists():
            raise RuntimeError(
                f"Expected base prologue file: {base_prologue_fn} to exist"
            )
        self.fn = fn
        self.kernel_name = kernel_name
        self.s_type = snap_type
        self.base_prologue_fn = base_prologue_fn
        self._state = None
        self._load = False
        self._num_args = None
        self._args = None

    def _dispose(self):
        if self._state is not None:
            try:
                ffi.lib.MnemePy_DisposeMemState(self._state)
            finally:
                self._state = None
                self._load = False
                self._args = None
                self._num_args = None

    def open(self):
        """
        Initialize and load the snapshot into the native handle.

        Returns
        -------
        MemStateRef
            Returns self for convenient chaining / context-manager usage.
        """
        if self._state is None:
            base_fn = self.base_prologue_fn or ""
            self._state = ffi.lib.MnemePy_initializeMemState(
                c_char_p(self.kernel_name.encode("utf-8")),
                c_char_p(self.fn.encode("utf-8")),
                c_char_p(base_fn.encode("utf-8")),
                c_bool(self.s_type == SnapshotType.PROLOGUE),
            )

        ffi.lib.MnemePy_LoadMemState(self._state)
        self._load = True
        return self

    @property
    def args(self):
        """
        Return the kernel argument pointer array stored in the snapshot.

        Returns
        -------
        ctypes.POINTER(ctypes.c_void_p)
            Pointer to an array of argument pointers as returned by the native API.

        Raises
        ------
        RuntimeError
            If the snapshot has not been loaded via :meth:`open`.
        """
        if not self._load:
            raise RuntimeError("Cannot access arguments without loading memory state")

        if self._args == None:
            self._args = ffi.lib.MnemePy_getArgs(self._state)

        return self._args

    @property
    def num_args(self):
        """
        Return the number of kernel arguments recorded in the snapshot.

        Returns
        -------
        int
            Number of arguments.

        Raises
        ------
        RuntimeError
            If the snapshot has not been loaded via :meth:`open`.
        """
        if not self._load:
            raise RuntimeError("Cannot access num_args without loading memory state")

        if self._num_args == None:
            self._num_args = ffi.lib.MnemePy_getNumArgs(self._state)

        return self._num_args

    def close(self):
        self._dispose()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def reset(self):
        """
        Reset the snapshot state in the native layer.

        This is typically used to restore the device memory state to the recorded
        baseline (e.g., before re-running a replay) without reinitializing the
        snapshot handle.

        Raises
        ------
        RuntimeError
            If the snapshot has not been loaded via :meth:`open`.
        """
        if self._load is False:
            raise RuntimeError("Cannot reset memory, if state is not read first.")

        ffi.lib.MnemePy_ResetMemState(self._state)

    def __eq__(self, other):
        """
        Compare two snapshots using the native comparison routine.

        Returns
        -------
        bool
            True if the native layer considers the states equivalent.
        """
        return bool(ffi.lib.MnemePy_CompareMemState(self._state, other._state))

    def __ne__(self, other):
        """
        Compare two snapshots using the native comparison routine.

        Returns
        -------
        bool
            True if the native layer considers the states different.
        """
        return not bool(ffi.lib.MnemePy_CompareMemState(self._state, other._state))

    def matches_unlaunched(self, other):
        """
        Check whether the recorded input already satisfies the recorded output.

        One snapshot must be a prologue and the other an epilogue. The check uses
        the same comparison as verification, so it answers whether a kernel that
        writes nothing would verify. Call it before any kernel launch writes to
        the prologue's device buffers.

        Returns
        -------
        bool
            True if the unmodified prologue matches the epilogue.
        """
        return bool(ffi.lib.MnemePy_MatchesUnlaunched(self._state, other._state))

    def __del__(self):
        try:
            self._dispose()
        except Exception:
            pass


@dataclass(frozen=True)
class KernelSource:
    """
    Source text of a recorded kernel and where it was read from.

    ``line`` and ``end_line`` are inclusive and 1-based. ``file`` is the path
    the text was read from, which is the recorded copy when one is available.
    """

    file: str
    line: int
    end_line: int
    text: str

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}-{self.end_line}"


class RecordedExecution:
    """
    Description of a recorded kernel execution and its dynamic instances.

    A :class:`RecordedExecution` captures everything needed to replay and tune
    a kernel that was observed during application execution:

      - Kernel identity (static hash, name, demangled name)
      - Argument names and specialization availability
      - Virtual address space reservation information (VA base + size)
      - LLVM IR module file paths required for linking
      - A mapping of **dynamic hash → KernelInstance**, representing each observed
        launch instance (grid/block/shared-mem and snapshot paths)

    The class behaves like a mapping over kernel instances and supports JSON
    serialization via :meth:`to_json` / :meth:`from_json`.

    Parameters
    ----------
    static_hash : str
        Stable identifier for the kernel’s static code shape.
    kernel_name : str
        Mangled or runtime kernel symbol name.
    demangled_name : str
        Human-readable kernel name (if available).
    llvm_files : list[str]
        Paths to LLVM IR modules captured during recording.
    arg_names : list[str]
        Recorded kernel argument names (for display/debugging).
    specializations : list[bool]
        Per-argument specialization availability flags.
    va_addr : str
        Base virtual address (hex string) used by Mneme’s memory manager.
    va_size : int
        Virtual address space size in bytes (or recording-specific unit).
    kernel_instances : dict[str, KernelInstance]
        Mapping from dynamic hash to recorded launch instance descriptor.
    total_launches : int, optional
        Number of times the kernel was launched during recording, or ``None``
        for records written before launches were counted.
    unrecorded_instances : dict[str, UnrecordedInstance], optional
        Mapping from dynamic hash to launch configurations that were launched
        but never snapshotted, so they cannot be replayed.
    """

    class KernelInstance:
        """
        Description of one dynamic kernel launch instance.

        A kernel may be launched multiple times with different dynamic properties
        (e.g., different grid/block sizes, argument values, or observed runtime hashes).
        Each :class:`KernelInstance` stores:

          - Launch parameters (grid, block, shared memory)
          - Dynamic hash (identifies the runtime instance)
          - Available specialization indices (derived from specialization flags)
          - Snapshot file paths for prologue and epilogue

        The prologue/epilogue snapshots are exposed via :class:`MemStateRef` objects,
        which are opened on demand by the replay executor.
        """

        def __init__(
            self,
            static_hash: str,
            dynamic_hash: str,
            kernel_name: str,
            args: List,
            shared_mem: int,
            block_dim: dim3,
            grid_dim: dim3,
            specializations: List[bool],
            occ: int,
            prologue_fn: str,
            epilogue_fn: str,
        ):
            self.static_hash = static_hash
            self.dynamic_hash = dynamic_hash
            self.kernel_name = kernel_name
            self.args = args
            self.shared_mem = shared_mem
            self.block_dim = block_dim
            self.grid_dim = grid_dim
            self.available_specializations = []
            self.specializations = specializations
            for i, v in enumerate(specializations):
                if v:
                    self.available_specializations.append(i)
            self.occ = occ
            # Set by the owning RecordedExecution so consumers handed only an
            # instance, such as custom search spaces, can reach kernel_source().
            self.execution: Optional["RecordedExecution"] = None
            self.prologue = MemStateRef(prologue_fn, kernel_name, SnapshotType.PROLOGUE)
            self.epilogue = MemStateRef(
                epilogue_fn,
                kernel_name,
                SnapshotType.EPILOGUE,
                base_prologue_fn=prologue_fn,
            )

        def __hash__(self):
            return hash(self.dynamic_hash + self.static_hash)

        def __str__(self):
            return f"Grid:{self.grid_dim}, BlockDim: {self.block_dim}, Shared Memory {self.shared_mem}"

        def to_dict(self, base_dir: "Path | None" = None):
            """
            Convert this instance into a JSON-friendly dictionary.

            If ``base_dir`` is provided, snapshot paths whose parent directory
            equals ``base_dir`` are written as basenames (relative form).
            Otherwise paths are written as-is.

            Returns
            -------
            dict
                Serializable representation containing dims, shared memory,
                occurrence count, and snapshot file paths.
            """
            res = {}
            res["Args"] = self.specializations
            res["BlockDims"] = self.block_dim.to_dict()
            res["GridDims"] = self.grid_dim.to_dict()
            res["Occurrences"] = self.occ
            res["SharedMem"] = self.shared_mem
            res["Epilogue"] = _make_path_relative(self.epilogue.fn, base_dir)
            res["Prologue"] = _make_path_relative(self.prologue.fn, base_dir)

            return res

    @dataclass
    class UnrecordedInstance:
        """
        Launch configuration that was observed but never snapshotted.

        Recording leaves a configuration unrecorded when the kernel already
        reached its maximum number of recordings or every launch with that
        configuration was skipped.
        """

        block_dim: dim3
        grid_dim: dim3
        shared_mem: int
        occ: int

        def to_dict(self):
            return {
                "BlockDims": self.block_dim.to_dict(),
                "GridDims": self.grid_dim.to_dict(),
                "Occurrences": self.occ,
                "SharedMem": self.shared_mem,
            }

    def __init__(
        self,
        static_hash: str,
        kernel_name: str,
        demangled_name: str,
        llvm_files: List[str],
        arg_names: List[str],
        specializations: List[bool],
        va_addr: str,
        va_size: int,
        kernel_instances: Dict[str, KernelInstance],
        source_file: Optional[str] = None,
        source_copy: Optional[str] = None,
        source_md5: Optional[str] = None,
        source_line: Optional[int] = None,
        source_end_line: Optional[int] = None,
        total_launches: Optional[int] = None,
        unrecorded_instances: Optional[Dict[str, UnrecordedInstance]] = None,
    ):
        self.static_hash = static_hash
        self.kernel_name = kernel_name
        self.demangled_name = demangled_name
        self.llvm_files = llvm_files
        self.arg_names = arg_names
        self.specializations = specializations
        self.va_addr = va_addr
        self.va_size = va_size
        self.kernel_instances = kernel_instances
        for instance in kernel_instances.values():
            instance.execution = self
        self.source_file = source_file
        self.source_copy = source_copy
        self.source_md5 = source_md5
        self.source_line = source_line
        self.source_end_line = source_end_line
        self.total_launches = total_launches
        self.unrecorded_instances = (
            {} if unrecorded_instances is None else unrecorded_instances
        )
        self._link_mod = None

    def __str__(self):
        return f"KernelName: {self.kernel_name} NumArgs: {len(self.arg_names)}, VASize: {self.va_size}, VAddr: {self.va_addr}"

    def __getitem__(self, key):
        return self.kernel_instances[key]

    def __setitem__(self, key, value):
        value.execution = self
        self.kernel_instances[key] = value

    def __delitem__(self, key):
        del self.kernel_instances[key]

    def __iter__(self):
        return iter(self.kernel_instances)

    def __len__(self):
        return len(self.kernel_instances)

    def __contains__(self, key):
        return key in self.kernel_instances

    def items(self):
        return self.kernel_instances.items()

    def keys(self):
        return self.kernel_instances.keys()

    def values(self):
        return self.kernel_instances.values()

    def link_llvm_modules(self, prune=True, internalize=True):
        """
        Link recorded LLVM IR modules into a single module suitable for replay.

        This is a convenience wrapper over the Proteus JIT linking layer.
        Results are cached on the first call and returned on subsequent calls.

        Parameters
        ----------
        prune : bool
            Whether to prune unused symbols/IR during linking.
        internalize : bool
            Whether to internalize symbols during linking.

        Returns
        -------
        ModuleRef
            Linked IR module produced by the JIT layer.
        """
        if self._link_mod is not None:
            return self._link_mod

        self._link_mod = jit.link_llvm_modules(
            self.llvm_files, self.kernel_name, prune, internalize
        )

        return self._link_mod

    def kernel_source(self) -> Optional[KernelSource]:
        """
        Return the recorded source text of the kernel definition.

        The range spans ``source_line`` through ``source_end_line`` inclusive
        and 1-based, where the end line is the last line of the kernel that
        generated code. The recorded copy is preferred over the original file
        because it is guaranteed to match what was compiled.

        A candidate whose contents no longer hash to ``source_md5`` is skipped:
        the recorded line numbers describe the revision that was compiled, so
        applying them to an edited file would return unrelated text.

        Returns
        -------
        Optional[KernelSource]
            The kernel's source text and location, or ``None`` when the record
            has no line information, or when no candidate is both readable and
            unchanged since it was compiled.
        """
        if self.source_line is None or self.source_end_line is None:
            return None

        for candidate in (self.source_copy, self.source_file):
            if candidate is None or not Path(candidate).exists():
                continue
            try:
                if self.source_md5 is not None:
                    digest = hashlib.md5(Path(candidate).read_bytes()).hexdigest()
                    if digest != self.source_md5:
                        continue
                with open(candidate, "r") as fd:
                    lines = fd.readlines()
            except OSError:
                continue
            return KernelSource(
                candidate,
                self.source_line,
                self.source_end_line,
                "".join(lines[self.source_line - 1 : self.source_end_line]),
            )

        return None

    def to_dict(self, base_dir: "Path | None" = None):
        res = {}
        res["ArgNames"] = self.arg_names
        res["BinaryBlobs"] = []
        res["DemangledName"] = self.demangled_name
        res["KernelName"] = self.kernel_name
        res["Modules"] = [_make_path_relative(m, base_dir) for m in self.llvm_files]
        res["Specializations"] = self.specializations
        res["StaticHash"] = self.static_hash
        if self.source_file is not None:
            res["SourceFile"] = self.source_file
        if self.source_copy is not None:
            res["SourceCopy"] = _make_path_relative(self.source_copy, base_dir)
        if self.source_md5 is not None:
            res["SourceMD5"] = self.source_md5
        if self.source_line is not None:
            res["SourceLine"] = self.source_line
        if self.source_end_line is not None:
            res["SourceEndLine"] = self.source_end_line
        if self.total_launches is not None:
            res["TotalLaunches"] = self.total_launches
            res["UnrecordedInstances"] = {
                k: v.to_dict() for k, v in self.unrecorded_instances.items()
            }
        res["VASize"] = self.va_size
        res["VAddr"] = self.va_addr
        res["instances"] = {}
        for k, v in self.items():
            res["instances"][k] = v.to_dict(base_dir)
        return res

    def to_json(self, fn: str):
        """
        Serialize this record database to a JSON file.

        Paths to artifacts that live in the same directory as the JSON file
        are written as basenames so that the resulting database is relocatable
        with plain ``cp`` / ``mv``. Paths outside that directory are kept as
        absolute paths.

        Parameters
        ----------
        fn : str
            Output JSON path.
        """
        base_dir = Path(fn).resolve().parent
        d = self.to_dict(base_dir=base_dir)
        find_non_jsonables(d)
        with open(fn, "w") as fd:
            json.dump(d, fd, indent=2)

    @classmethod
    def from_json(cls, fn: str):
        """
        Load a :class:`RecordedExecution` database from JSON.

        This reconstructs all :class:`KernelInstance` entries and validates that
        referenced LLVM module paths exist.

        Parameters
        ----------
        fn : str
            Path to the recorded execution JSON file.

        Returns
        -------
        RecordedExecution
            Loaded record database.

        Raises
        ------
        RuntimeError
            If the JSON file does not exist or referenced IR modules are missing.
        """
        if not Path(fn).exists():
            raise RuntimeError("JSON file does not exist")

        with open(fn, "r") as fd:
            record_db = json.load(fd)

        # Paths in the JSON may be either absolute (legacy) or relative to the
        # JSON file's parent directory. Resolve to absolute so all in-memory
        # consumers see a stable, cwd-independent path.
        base_dir = Path(fn).resolve().parent

        def _resolve(p: str) -> str:
            return p if Path(p).is_absolute() else str(base_dir / p)

        instances = {}
        for dhash, inst in record_db["instances"].items():
            block_dim = dim3(
                inst["BlockDims"]["x"], inst["BlockDims"]["y"], inst["BlockDims"]["z"]
            )
            grid_dim = dim3(
                inst["GridDims"]["x"], inst["GridDims"]["y"], inst["GridDims"]["z"]
            )
            instances[dhash] = cls.KernelInstance(
                record_db["StaticHash"],
                dhash,
                record_db["KernelName"],
                inst["Args"],
                inst["SharedMem"],
                block_dim,
                grid_dim,
                record_db["Specializations"],
                inst["Occurrences"],
                _resolve(inst["Prologue"]),
                _resolve(inst["Epilogue"]),
            )

        # Records written before launches were counted have neither field.
        unrecorded_instances = {
            dhash: cls.UnrecordedInstance(
                dim3.from_dict(inst["BlockDims"]),
                dim3.from_dict(inst["GridDims"]),
                inst["SharedMem"],
                inst["Occurrences"],
            )
            for dhash, inst in record_db.get("UnrecordedInstances", {}).items()
        }

        resolved_modules = [_resolve(m) for m in record_db["Modules"]]
        for llvm_fn in resolved_modules:
            if not Path(llvm_fn).exists():
                raise RuntimeError(f"File {llvm_fn} does not exist")

        # The source copy is auxiliary to replay, so its absence is not an error.
        source_copy = record_db.get("SourceCopy")
        if source_copy is not None:
            source_copy = _resolve(source_copy)

        return cls(
            record_db["StaticHash"],
            record_db["KernelName"],
            record_db["DemangledName"],
            resolved_modules,
            record_db["ArgNames"],
            record_db["Specializations"],
            record_db["VAddr"],
            record_db["VASize"],
            instances,
            source_file=record_db.get("SourceFile"),
            source_copy=source_copy,
            source_md5=record_db.get("SourceMD5"),
            source_line=record_db.get("SourceLine"),
            source_end_line=record_db.get("SourceEndLine"),
            total_launches=record_db.get("TotalLaunches"),
            unrecorded_instances=unrecorded_instances,
        )
