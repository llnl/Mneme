import pytest
import os
import re
import subprocess

from mneme.proteus import jit
from mneme.llvm.module import ModuleRef, parse_assembly
from mneme.llvm.buffer import MemBufferRef
from ctypes import POINTER, c_bool, c_char, c_char_p, c_int, c_uint, c_uint64, c_void_p
import ctypes
from mneme.llvm import ffi
from mneme.mneme_types import dim3


# ----------------------------
# GPU detection / backend selection
# ----------------------------


def _get_complex_ir(gpu_backend):
    if gpu_backend == "amd":
        return r"""
; ModuleID = 'test'
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @kernel_add(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  ; threadIdx.x
  %tid_x = call i32 @llvm.amdgcn.workitem.id.x()

  ; blockDim.x / y / z
  %bdx = call i32 @llvm.amdgcn.workgroup.size.x()
  %bdy = call i32 @llvm.amdgcn.workgroup.size.y()
  %bdz = call i32 @llvm.amdgcn.workgroup.size.z()

  ; ------ compute global linear thread id ------
  ; global_tid = tid_x + blockIdx.x * blockDim.x
  ; but since we don't have blockIdx here, we just fold dims into tid

  ; compute  threads_per_block = blockDim.x * blockDim.y * blockDim.z
  %tmp1 = mul i32 %bdx, %bdy
  %threads_per_block = mul i32 %tmp1, %bdz

  ; global_tid = tid_x + threads_per_block  (fake linearization to use all dims)
  %global_tid = add i32 %tid_x, %threads_per_block

  ; branch: if (global_tid > 512) early return
  %cmp = icmp ugt i32 %global_tid, 512
  br i1 %cmp, label %early_return, label %continue

early_return:
  ret void

continue:
  ; main kernel body: ptr[global_tid] += val
  %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %global_tid
  %old = load i32, i32 addrspace(1)* %elem
  %new = add i32 %old, %val
  store i32 %new, i32 addrspace(1)* %elem
  ret void
}

; intrinsics
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.size.x()
declare i32 @llvm.amdgcn.workgroup.size.y()
declare i32 @llvm.amdgcn.workgroup.size.z()
"""
    elif gpu_backend == "cuda":
        return r"""
        ; ModuleID = 'block_dim.cpp'
source_filename = "block_dim.cpp"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local void @kernel_add(ptr nocapture noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %8 = mul i32 %6, %7
  %9 = add nsw i32 %8, %3
  %10 = icmp sgt i32 %9, 512
  br i1 %10, label %16, label %11

11:                                               ; preds = %2
  %12 = sext i32 %9 to i64
  %13 = getelementptr inbounds i32, ptr %0, i64 %12
  %14 = load i32, ptr %13, align 4, !tbaa !8
  %15 = add nsw i32 %14, %1
  store i32 %15, ptr %13, align 4, !tbaa !8
  br label %16

16:                                               ; preds = %2, %11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_90" "target-features"="+ptx82,+sm_90" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4}
!llvm.ident = !{!5, !6}
!nvvmir.version = !{!7}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @kernel_add, !"kernel", i32 1}
!5 = !{!"clang version 18.1.8 (https://github.com/conda-forge/clangdev-feedstock 12303b10a39c08f9cf71b2fb4eadbd70e3f8dd6b)"}
!6 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!7 = !{i32 2, i32 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
"""


def _get_add_ir(gpu_backend):
    if gpu_backend == "amd":
        return r"""
; ModuleID = 'test'
target triple = "amdgcn-amd-amdhsa"


define amdgpu_kernel void @kernel_add(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  ; get thread id
  %tid = call i32 @llvm.amdgcn.workitem.id.x()

  ; compare tid > 512
  %cmp = icmp ugt i32 %tid, 512
  br i1 %cmp, label %early_return, label %continue

early_return:
  ret void

continue:
  ; original kernel body
  %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %tid
  %old = load i32, i32 addrspace(1)* %elem
  %new = add i32 %old, %val
  store i32 %new, i32 addrspace(1)* %elem
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
"""
    elif gpu_backend == "cuda":
        return r"""
        ; ModuleID = 'gpu.cpp'
source_filename = "gpu.cpp"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local void @kernel_add(ptr nocapture noundef %0, i32 noundef %val) local_unnamed_addr #0 {
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = icmp sgt i32 %3, 512
  br i1 %4, label %10, label %5

5:                                                ; preds = %2
  %6 = sext i32 %3 to i64
  %elem = getelementptr inbounds i32, ptr %0, i64 %6
  %old = load i32, ptr %elem, align 4, !tbaa !8
  %new = add i32 %old, %val
  store i32 %new, ptr %elem, align 4, !tbaa !8
  br label %10

10:                                               ; preds = %2, %5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_90" "target-features"="+ptx82,+sm_90" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4}
!llvm.ident = !{!5, !6}
!nvvmir.version = !{!7}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @kernel_add, !"kernel", i32 1}
!5 = !{!"clang version 18.1.8 (https://github.com/conda-forge/clangdev-feedstock 12303b10a39c08f9cf71b2fb4eadbd70e3f8dd6b)"}
!6 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!7 = !{i32 2, i32 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
        """


def _run_cmd(cmd, shell=False) -> str:
    out = subprocess.check_output(cmd, shell=shell, text=True)
    return out.strip()


def _has_nvidia() -> bool:
    try:
        _run_cmd(["nvidia-smi"])
        return True
    except Exception:
        return False


def _has_amd() -> bool:
    try:
        out = _run_cmd(["rocminfo"])
        # rocminfo typically prints "gfx###" or "AMD"
        return ("gfx" in out) or ("AMD" in out)
    except Exception:
        return False


def _detect_cuda_sm_int() -> int:
    """
    Returns SM as integer, e.g. 90 for compute capability 9.0
    """
    try:
        out = _run_cmd(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
        )
        cc = out.splitlines()[0].strip()  # e.g. "9.0"
        major, minor = cc.split(".")
        return int(major) * 10 + int(minor)
    except Exception:
        # fallback if query fails
        return 80


def _detect_amd_gfx() -> str:
    """
    Attempts to detect a gfx target (e.g. gfx942) from rocminfo output.
    Falls back to MNEME_TEST_AMD_GFX or gfx942.
    """
    override = os.getenv("MNEME_TEST_AMD_GFX")
    if override:
        return override

    try:
        out = _run_cmd(["rocminfo"])
        # Common patterns include "Name: gfx942" or "gfx942"
        m = re.search(r"\bgfx\d+\b", out)
        if m:
            return m.group(0)
    except Exception:
        pass

    return "gfx942"


@pytest.fixture(scope="session")
def gpu_backend():
    forced = os.getenv("MNEME_TEST_BACKEND")
    if forced in ("amd", "cuda"):
        return forced

    if _has_nvidia():
        return "cuda"
    if _has_amd():
        return "amd"

    pytest.skip(
        "No supported GPU detected: need NVIDIA (nvidia-smi) or AMD (rocminfo)."
    )


@pytest.fixture(scope="session")
def native_arch(gpu_backend):
    """
    Returns the native arch string to pass into jit.optimize.
    You may need to adjust the returned format to match what your jit.optimize expects.
    """
    if gpu_backend == "amd":
        return _detect_amd_gfx()

    # CUDA
    override = os.getenv("MNEME_TEST_CUDA_SM")
    if override:
        sm = int(override)
    else:
        sm = _detect_cuda_sm_int()

    # Common conventions are "sm_90" or "90". Pick one; adjust if your optimize() expects another.
    return f"sm_{sm}"


def test_specialize_args(gpu_backend, native_arch):
    # 1. Construct minimal IR
    IR = _get_add_ir(gpu_backend)
    mod = parse_assembly(IR)

    # 2. Initial prune
    jit.pruneIR(mod)

    # 3. Specialize argument %val = 7
    val = c_int(7)
    val_ptr = ctypes.byref(val)
    KernelArgs = (c_void_p * 2)()
    KernelArgs[0] = c_void_p()  # first arg not specialized
    KernelArgs[1] = c_void_p(ctypes.addressof(val))

    mod_hash = 0
    new_hash = jit.specialize_args(
        mod,
        mod_hash,
        "kernel_add",
        KernelArgs,
        num_args=2,
        specialize_indexes=[1],  # specialize %val
    )

    assert new_hash != mod_hash

    assert "%new = add i32 %old, 7" in str(mod)


def test_specialize_dims_assume(gpu_backend):
    # 1. Construct minimal IR

    IR = _get_add_ir(gpu_backend)

    mod = parse_assembly(IR)
    old_mod = str(mod)

    print("before", mod)

    block = dim3(4, 4, 4)
    grid = dim3(16, 16, 16)

    mod_hash = 0
    new_hash = jit.specialize_dims_assume(mod, mod_hash, "kernel_add", grid, block)

    assert new_hash != mod_hash
    print("After", mod)
    if gpu_backend == "amd":
        assert "%tid = call i32 @llvm.amdgcn.workitem.id.x(), !range !0" in str(
            mod
        ) or "%tid = call range(i32 0, 4) i32 @llvm.amdgcn.workitem.id.x()" in str(mod)
    else:
        assert (
            ("%2 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !8"
            in str(mod) and "!8 = !{i32 0, i32 4}" in str(mod) ) 
            or ("%2 = tail call noundef range(i32 0, 4) i32 @llvm.nvvm.read.ptx.sreg.tid.x()" in str(mod)
        ))


@pytest.mark.parametrize("pipeline", ["O3", "default<O3>", "function(sroa),default<O3>"])
def test_check_pass_pipeline_accepts_valid_pipeline(gpu_backend, native_arch, pipeline):
    mod = parse_assembly(_get_add_ir(gpu_backend))

    jit.check_pass_pipeline(mod, native_arch, pipeline, 3)


@pytest.mark.parametrize(
    "pipeline, llvm_error",
    [
        ("defualt<O3>", "unknown pass name 'defualt<O3>'"),
        ("sroa,globalopt", "unknown function pass 'globalopt'"),
    ],
)
def test_optimize_raises_on_invalid_pipeline(gpu_backend, native_arch, pipeline, llvm_error):
    mod = parse_assembly(_get_add_ir(gpu_backend))

    with pytest.raises(jit.InvalidPassPipelineError, match=llvm_error):
        jit.optimize(mod, native_arch, pipeline, 3)


def test_check_pass_plugin(tmp_path):
    from mneme.llvm._lib_path_config import MNEME_PASS_PLUGIN_LIB

    jit.check_pass_plugin(MNEME_PASS_PLUGIN_LIB)

    not_a_plugin = tmp_path / "libNotAPlugin.so"
    not_a_plugin.write_text("not a shared library")
    with pytest.raises(ValueError, match=f"Cannot load pass plugin '{not_a_plugin}'"):
        jit.check_pass_plugin(str(not_a_plugin))


def test_specialize_dims(gpu_backend, native_arch):
    # 1. Construct minimal IR
    IR = _get_complex_ir(gpu_backend)
    mod = parse_assembly(IR)
    old_mod = str(mod)

    block = dim3(4, 4, 4)
    grid = dim3(16, 16, 16)

    mod_hash = 0
    new_hash = jit.specialize_dims(mod, mod_hash, "kernel_add", grid, block)

    assert new_hash != mod_hash
    opt_hash = jit.optimize(mod, native_arch, "default<O3>", 3)
    print(mod)
    assert "%bdx = call i32 @llvm.amdgcn.workgroup.size.x()" not in str(mod)
    assert "%bdy = call i32 @llvm.amdgcn.workgroup.size.y()" not in str(mod)
    assert "%bdz = call i32 @llvm.amdgcn.workgroup.size.z()" not in str(mod)

    assert "%4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()" not in str(
        mod
    )
    assert "%5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()" not in str(
        mod
    )
    assert "%7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.z()" not in str(
        mod
    )

    assert "%bdx = call i32 @llvm.amdgcn.workgroup.size.x()" not in str(mod)
    assert "%bdy = call i32 @llvm.amdgcn.workgroup.size.y()" not in str(mod)
    assert "%bdz = call i32 @llvm.amdgcn.workgroup.size.z()" not in str(mod)


def test_launch_bounds(gpu_backend, native_arch):
    # 1. Construct minimal IR
    IR = _get_complex_ir(gpu_backend)
    mod = parse_assembly(IR)
    old_mod = str(mod)

    mod_hash = 0
    new_hash = jit.set_launch_bounds(mod, mod_hash, "kernel_add", 128, 2)
    print(mod)

    assert new_hash != mod_hash
    opt_hash = jit.optimize(mod, native_arch, "default<O3>", 3)
    print("Back end is", gpu_backend)
    print(mod)
    if gpu_backend == "amd":
        assert '"amdgpu-flat-work-group-size"="1,128"' in str(mod)
        assert '"amdgpu-waves-per-eu"="2,2"' in str(mod)
    elif gpu_backend == "cuda":
        mod_str = str(mod)
        has_max_threads = '"nvvm.maxntid"="128"' in mod_str or re.search(
            r'!\d+ = !{ptr @kernel_add, !"maxntid", i32 128}', mod_str
        )
        has_min_blocks = '"nvvm.minctasm"="2"' in mod_str or re.search(
            r'!\d+ = !{ptr @kernel_add, !"minctasm", i32 2}', mod_str
        )
        assert has_max_threads
        assert has_min_blocks
    else:
        raise RuntimeError("Unknown backend")


def test_link_llvm_modules(tmp_path):
    MODULE_A = r"""
    ; ModuleID = 'module_a'
    target triple = "amdgcn-amd-amdhsa"

    define amdgpu_kernel void @kernel_a(i32 addrspace(1)* %ptr) {
    entry:
      %tid = call i32 @llvm.amdgcn.workitem.id.x()
      %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %tid
      %old = load i32, i32 addrspace(1)* %elem
      %new = add i32 %old, 1
      store i32 %new, i32 addrspace(1)* %elem
      ret void
    }

    declare i32 @llvm.amdgcn.workitem.id.x()
    """
    MODULE_B = r"""
    ; ModuleID = 'module_b'
    target triple = "amdgcn-amd-amdhsa"

    define amdgpu_kernel void @kernel_b(i32 addrspace(1)* %ptr, i32 %val) {
    entry:
      %tid = call i32 @llvm.amdgcn.workitem.id.x()
      %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %tid
      %old = load i32, i32 addrspace(1)* %elem
      %new = add i32 %old, %val
      store i32 %new, i32 addrspace(1)* %elem
      ret void
    }

    declare i32 @llvm.amdgcn.workitem.id.x()
    """
    # 1. Create paths inside pytest's temporary directory
    module_a_path = tmp_path / "module_a.ll"
    module_b_path = tmp_path / "module_b.ll"
    mod_A = parse_assembly(MODULE_A)
    mod_A.to_bitcode(str(module_a_path))

    mod_B = parse_assembly(MODULE_B)
    mod_B.to_bitcode(str(module_b_path))

    modules = [str(module_a_path), str(module_b_path)]

    mod = jit.link_llvm_modules(modules, "kernel_a", False, False)

    assert (
        "define amdgpu_kernel void @kernel_b(ptr addrspace(1) %ptr, i32 %val)"
        in str(mod)
    )
    assert "define amdgpu_kernel void @kernel_a(ptr addrspace(1) %ptr)" in str(mod)
    mod = jit.link_llvm_modules(modules, "kernel_a", False, True)
    assert (
        "define internal amdgpu_kernel void @kernel_b(ptr addrspace(1) %ptr, i32 %val)"
        in str(mod)
    )
