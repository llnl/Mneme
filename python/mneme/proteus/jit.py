"""
Python FFI bindings for the Proteus JIT transformation and code-generation pipeline.

This module provides a thin, Pythonic wrapper around Proteus’ C++ JIT
infrastructure, exposing functionality for:

  * Linking multiple LLVM IR modules into a single executable module
  * Pruning dead IR and internalizing symbols
  * Applying architecture-aware optimization pipelines
  * Specializing kernels based on runtime arguments and launch dimensions
  * Emitting device-specific executable objects (e.g., ELF / HSACO)

All operations are performed through a C FFI layer and operate directly on
LLVM modules represented by :class:`~mneme.llvm.module.ModuleRef`. Most
functions mutate the provided module in place and return either updated
metadata (such as a specialization hash) or compiled device artifacts.

This module forms the core of Mneme’s record–replay and autotuning workflow,
bridging recorded execution metadata with dynamic compilation and execution
on accelerator devices.
"""

from ctypes import POINTER, c_bool, c_char, c_char_p, c_int, c_uint, c_uint64, c_void_p
from typing import List

from ..llvm import ffi as ffi
from ..llvm.buffer import MemBufferRef
from ..llvm.common import _decode_string, _encode_string
from ..llvm.context import get_global_context
from ..llvm.module import ModuleRef
from ..mneme_types import dim3

ffi.lib.ProteusPY_pruneIR.argtypes = [ffi.LLVMModuleRef]
ffi.lib.ProteusPY_optimize.argtypes = [ffi.LLVMModuleRef, c_char_p, c_char_p, c_uint]
ffi.lib.ProteusPY_internalize.argtypes = [ffi.LLVMModuleRef, c_char_p]
ffi.lib.ProteusPY_codeGenObject.argtypes = [
    ffi.LLVMModuleRef,
    c_char_p,
    c_uint,
]
ffi.lib.ProteusPY_codeGenObject.restype = ffi.LLVMMemBufferRef
ffi.lib.ProteusPY_linkModules.argtypes = [
    POINTER(c_char_p),
    c_int,
    ffi.LLVMContextRef,
    c_char_p,
    c_bool,
    c_bool,
]
ffi.lib.ProteusPY_linkModules.restype = ffi.LLVMModuleRef
ffi.lib.ProteusPY_specializeArguments.argtypes = [
    ffi.LLVMModuleRef,  # Module
    c_uint64,  # Hash
    c_char_p,  # KernelName
    POINTER(c_void_p),  # KernelArguments
    c_int,  # Number of Arguments
    POINTER(c_int),  # Indexes to specialize
    c_int,  # num Indexes
]
ffi.lib.ProteusPY_specializeArguments.restype = c_uint64

ffi.lib.ProteusPY_specializeDims.argtypes = [
    ffi.LLVMModuleRef,
    c_uint64,
    c_char_p,
    dim3,
    dim3,
]
ffi.lib.ProteusPY_specializeDims.restype = c_uint64

ffi.lib.ProteusPY_specializeDimsAssume.argtypes = [
    ffi.LLVMModuleRef,
    c_uint64,
    c_char_p,
    dim3,
    dim3,
]
ffi.lib.ProteusPY_specializeDimsAssume.restype = c_uint64

ffi.lib.ProteusPY_setLaunchBounds.argtypes = [
    ffi.LLVMModuleRef,
    c_uint64,
    c_char_p,
    c_int,
    c_int,
]

ffi.lib.ProteusPY_setLaunchBounds.restype = c_uint64

ffi.lib.ProteusPY_getCodegenMethod.argtypes = []
ffi.lib.ProteusPY_getCodegenMethod.restype = c_char_p

ffi.lib.ProteusPY_registerJITPassPlugin.argtypes = [c_char_p]


def register_pass_plugin(path: str):
    """
    Register an LLVM pass plugin shared library with Proteus.

    Uses the load-only registration mode: the plugin's pass names become
    parseable in pipeline strings passed to :func:`optimize`, but nothing is
    added to the pipeline automatically. Registration is idempotent; Proteus
    deduplicates by resolved path.

    Parameters
    ----------
    path : str
        Filesystem path to the plugin shared library.
    """
    ffi.lib.ProteusPY_registerJITPassPlugin(_encode_string(path))


def pruneIR(mod: ModuleRef):
    """
    Remove unused functions, globals, and dead IR from an LLVM module.

    This calls Proteus' C++ pruning pass through the FFI to eliminate dead IR and
    reduce module size before further specialization or optimization. Device
    globals keep their ``externally_initialized`` flag, since replay writes
    their recorded values from the host.

    Parameters
    ----------
    mod : ModuleRef
        LLVM module to prune.

    Raises
    ------
    TypeError
        If ``mod`` is not a :class:`~mneme.llvm.module.ModuleRef`.
    """
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")
    ffi.lib.ProteusPY_pruneIR(mod)


def optimize(mod: ModuleRef, device_arch: str, opt_level: str, codegen_opt_level: int):
    """
    Run Proteus optimization passes on an LLVM module.

    Applies middle-end optimization passes customized for a target device
    architecture and a chosen LLVM optimization level. Also configures the
    code-generation optimization intensity used later by the backend.

    Parameters
    ----------
    mod : ModuleRef
        LLVM module to optimize (mutated in-place).
    device_arch : str
        Target device architecture string (e.g., ``"gfx942"``).
    opt_level : str
        LLVM optimization pipeline selector (e.g., ``"O1"``, ``"O2"``, ``"O3"``,
        ``"Os"``, ``"Oz"``). If empty, optimization is skipped.
    codegen_opt_level : int
        Backend optimization level in ``[0, 3]``.

    Raises
    ------
    TypeError
        If ``mod`` is not a :class:`~mneme.llvm.module.ModuleRef`.
    ValueError
        If ``codegen_opt_level`` is outside ``[0, 3]``.
    """
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")

    if not (codegen_opt_level >= 0 and codegen_opt_level <= 3):
        raise ValueError(
            f"Expected the codegen_opt_level to be between 0-3 instead got {codegen_opt_level}"
        )
    if len(opt_level) == 0:
        return

    ffi.lib.ProteusPY_optimize(
        mod,
        _encode_string(device_arch),
        _encode_string(opt_level),
        int(codegen_opt_level),
    )


def internalize(mod: ModuleRef, kernel_name: str):
    """
    Mark all symbols except the given kernel and the device globals as internal.

    This restricts symbol visibility to reduce linking overhead and enable more
    aggressive optimization. Device globals (``externally_initialized``
    ``__device__``/``__constant__`` variables) stay exported so replay can look
    them up by name and restore their recorded values.

    Parameters
    ----------
    mod : ModuleRef
        LLVM module to update (mutated in-place).
    kernel_name : str
        Name of the kernel whose symbol must remain externally visible.

    Raises
    ------
    TypeError
        If ``mod`` is not a :class:`~mneme.llvm.module.ModuleRef`.
    """
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")

    ffi.lib.ProteusPY_internalize(mod, _encode_string(kernel_name))


def codegen_object(
    mod: ModuleRef, device_arch, codegen_opt_level: int = 3
):
    """
    Generate a compiled device code object from an LLVM module.

    Invokes the Proteus backend code generator for the given architecture and
    returns the produced binary wrapped in a :class:`~mneme.llvm.buffer.MemBufferRef`.

    Parameters
    ----------
    mod : ModuleRef
        LLVM module to compile.
    device_arch : str
        Target architecture string.
    codegen_opt_level : int, optional
        Backend optimization level in ``[1, 3]``. Defaults to ``3``.

    Returns
    -------
    MemBufferRef
        Memory buffer containing the produced code object.

    Raises
    ------
    TypeError
        If ``mod`` is not a :class:`~mneme.llvm.module.ModuleRef`.
    RuntimeError
        If ``codegen_opt_level`` is not in ``[1, 3]``.
    """
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")

    if codegen_opt_level < 1 or codegen_opt_level > 3:
        raise RuntimeError(
            f"codegen optimization level must be in range (0,3], instead it was {codegen_opt_level}"
        )
    result = MemBufferRef(
        ffi.lib.ProteusPY_codeGenObject(
            mod,
            _encode_string(device_arch),
            codegen_opt_level,
        )
    )
    return result


def link_llvm_modules(
    modules: List[str], kernel_name: str, prune: bool, internalize: bool
) -> ModuleRef:
    """
    Link multiple LLVM IR modules into a single unified module.

    This constructs a new module by invoking Proteus' linker. Optionally performs
    pruning and internalization during the link stage.

    Parameters
    ----------
    modules : list[str]
        Filesystem paths to LLVM IR modules to link.
    kernel_name : str
        Name of the kernel entry function to preserve.
    prune : bool
        Whether to prune dead IR after linking.
    internalize : bool
        Whether to internalize symbols except the kernel.

    Returns
    -------
    ModuleRef
        Newly linked module.
    """
    c_strings = [c_char_p(s.encode("utf-8")) for s in modules]
    ArrayType = c_char_p * len(c_strings)
    c_array = ArrayType(*c_strings)
    Mod = ModuleRef(
        ffi.lib.ProteusPY_linkModules(
            c_array,
            len(modules),
            get_global_context(),
            kernel_name.encode("utf-8"),
            prune,
            internalize,
        ),
        get_global_context(),
    )
    return Mod


def specialize_args(
    mod: ModuleRef,
    mod_hash: int,
    kernel_name: str,
    kernel_args,
    num_args: int,
    specialize_indexes,
) -> int:
    """
    Specialize a subset of kernel arguments inside an LLVM module.

    Performs IR rewriting / constant propagation based on provided runtime
    arguments, and returns an updated hash reflecting the specialization.

    Parameters
    ----------
    mod : ModuleRef
        LLVM module to modify.
    mod_hash : int
        Current module hash before specialization.
    kernel_name : str
        Kernel whose arguments are specialized.
    kernel_args
        Raw pointers to argument values (FFI-compatible pointer array).
    num_args : int
        Total number of kernel arguments.
    specialize_indexes
        Indices of arguments to specialize.

    Returns
    -------
    int
        Updated module hash after specialization.

    Raises
    ------
    RuntimeError
        If more indices are requested than available arguments.
    """
    if num_args < len(specialize_indexes):
        raise RuntimeError("Trying to specialize more indexes than available")

    indexes = (c_int * len(specialize_indexes))()
    for i, v in enumerate(specialize_indexes):
        indexes[i] = v

    return int(
        ffi.lib.ProteusPY_specializeArguments(
            mod,
            c_uint64(mod_hash),
            _encode_string(kernel_name),
            kernel_args,
            num_args,
            indexes,
            len(specialize_indexes),
        )
    )


def specialize_dims(
    mod: ModuleRef, mod_hash: int, kernel_name: str, grid_dim: dim3, block_dim: dim3
):
    """
    Specialize launch dimensions (grid/block) inside the LLVM module.

    Embeds compile-time constants for launch configuration, enabling IR
    simplification and more aggressive optimization.

    Parameters
    ----------
    mod : ModuleRef
        LLVM module to update.
    mod_hash : int
        Previous module hash.
    kernel_name : str
        Kernel to specialize.
    grid_dim : dim3
        Grid dimensions.
    block_dim : dim3
        Block dimensions.

    Returns
    -------
    int
        Updated module hash.
    """
    return int(
        ffi.lib.ProteusPY_specializeDims(
            mod, c_uint64(mod_hash), _encode_string(kernel_name), grid_dim, block_dim
        )
    )


def specialize_dims_assume(
    mod: ModuleRef, mod_hash: int, kernel_name: str, grid_dim: dim3, block_dim: dim3
):
    """
    Add launch-dimension assumptions (grid/block) inside the LLVM module.

    Similar to :func:`specialize_dims`, but emits assumptions rather than (or in
    addition to) direct constant replacement, enabling downstream passes to
    simplify based on assumed launch invariants.

    Parameters
    ----------
    mod : ModuleRef
        LLVM module to update.
    mod_hash : int
        Previous module hash.
    kernel_name : str
        Kernel to specialize.
    grid_dim : dim3
        Grid dimensions.
    block_dim : dim3
        Block dimensions.

    Returns
    -------
    int
        Updated module hash.
    """
    return int(
        ffi.lib.ProteusPY_specializeDimsAssume(
            mod, c_uint64(mod_hash), _encode_string(kernel_name), grid_dim, block_dim
        )
    )


def set_launch_bounds(
    mod: ModuleRef,
    mod_hash: int,
    kernel_name: str,
    max_threads_per_block: int,
    min_blocks_per_sm: int,
):
    """
    Apply CUDA/HIP-style launch-bounds metadata to the kernel.

    Sets launch-bounds on the kernel to restrict maximum threads per block and
    communicate occupancy constraints, influencing register allocation and
    codegen decisions.

    Parameters
    ----------
    mod : ModuleRef
        LLVM module to annotate.
    mod_hash : int
        Current module hash.
    kernel_name : str
        Name of the kernel function.
    max_threads_per_block : int
        Maximum threads-per-block bound (must be ``<= 1024``).
    min_blocks_per_sm : int
        Minimum required blocks per SM.

    Returns
    -------
    int
        Updated module hash.

    Raises
    ------
    RuntimeError
        If ``max_threads_per_block`` exceeds 1024.
    """
    if max_threads_per_block > 1024:
        raise RuntimeError("Max threads cannot be larger than 1024")

    return int(
        ffi.lib.ProteusPY_setLaunchBounds(
            mod,
            c_uint64(mod_hash),
            _encode_string(kernel_name),
            max_threads_per_block,
            min_blocks_per_sm,
        )
    )

def get_proteus_codegen_method():
    """
    Returns the codegen method used by proteus to generate objects.
    """
    s = ffi.lib.ProteusPY_getCodegenMethod()
    if not s:
        raise RuntimeError("ProteusPY_getCodegenMethod returned NULL")
    # ctypes gives us bytes for c_char_p
    return s.decode("utf-8")
