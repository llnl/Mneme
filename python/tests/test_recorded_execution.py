import hashlib
import json
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from mneme.recorded_execution import (
    MemStateRef,
    RecordedExecution,
    SnapshotType,
    _make_path_relative,
)
from mneme.mneme_types import dim3


# ======================================================================
#                           MemStateRef Tests
# ======================================================================

def test_memstate_constructor_checks_file_exists():
    with patch("mneme.recorded_execution.Path.exists", return_value=False):
        with pytest.raises(RuntimeError):
            MemStateRef("missing.pro", "kernel", SnapshotType.PROLOGUE)


def test_memstate_open_initializes_and_loads():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "STATE"

            m = MemStateRef("snap.pro", "kernelA", SnapshotType.PROLOGUE)
            m.open()

            fake.MnemePy_initializeMemState.assert_called_once()
            fake.MnemePy_LoadMemState.assert_called_once_with("STATE")


def test_epilogue_memstate_without_base_passes_empty_path_to_ffi():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "STATE"

            m = MemStateRef("snap.epi", "kernelA", SnapshotType.EPILOGUE)
            m.open()

            args, _ = fake.MnemePy_initializeMemState.call_args
            assert args[2].value == b""
            assert args[3].value is False


def test_epilogue_memstate_passes_base_prologue_to_ffi():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "STATE"

            m = MemStateRef(
                "snap.epi",
                "kernelA",
                SnapshotType.EPILOGUE,
                base_prologue_fn="snap.pro",
            )
            m.open()

            args, _ = fake.MnemePy_initializeMemState.call_args
            assert args[1].value == b"snap.epi"
            assert args[2].value == b"snap.pro"
            assert args[3].value is False


@pytest.mark.parametrize("epilogue_fn", ["snap.bytes.epi", "snap.diff.epi"])
def test_epilogue_formats_share_base_prologue_interface(epilogue_fn):
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "STATE"

            m = MemStateRef(
                epilogue_fn,
                "kernelA",
                SnapshotType.EPILOGUE,
                base_prologue_fn="snap.pro",
            )
            m.open()

            args, _ = fake.MnemePy_initializeMemState.call_args
            assert args[1].value == epilogue_fn.encode("utf-8")
            assert args[2].value == b"snap.pro"
            assert args[3].value is False


def test_memstate_args_lazy_loaded_once():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "S"
            fake.MnemePy_getArgs.return_value = ["A", "B"]

            m = MemStateRef("snap.pro", "kernel", SnapshotType.PROLOGUE)
            m.open()

            a1 = m.args
            a2 = m.args

            assert a1 == a2
            fake.MnemePy_getArgs.assert_called_once()


def test_memstate_num_args_lazy_loaded_once():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "S"
            fake.MnemePy_getNumArgs.return_value = 7

            m = MemStateRef("snap.pro", "kernel", SnapshotType.PROLOGUE)
            m.open()

            n1 = m.num_args
            n2 = m.num_args

            assert n1 == n2 == 7
            fake.MnemePy_getNumArgs.assert_called_once()


def test_memstate_reset_requires_load():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        m = MemStateRef("snap.pro", "kernel", SnapshotType.PROLOGUE)
        with pytest.raises(RuntimeError):
            m.reset()


def test_memstate_context_manager_calls_open_and_close():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "ST"

            m = MemStateRef("snap.pro", "kernel", SnapshotType.PROLOGUE)

            with m:
                fake.MnemePy_LoadMemState.assert_called_once_with("ST")

            fake.MnemePy_DisposeMemState.assert_called_once_with("ST")


def test_memstate_equality_uses_ffi_compare():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.side_effect = ["A", "B"]
            fake.MnemePy_CompareMemState.return_value = True

            m1 = MemStateRef("p1.pro", "k", SnapshotType.PROLOGUE).open()
            m2 = MemStateRef("p2.pro", "k", SnapshotType.PROLOGUE).open()

            assert m1 == m2
            fake.MnemePy_CompareMemState.assert_called_once_with("A", "B")


def test_memstate_matches_unlaunched_uses_ffi():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.side_effect = ["P", "E"]
            fake.MnemePy_MatchesUnlaunched.return_value = True

            pro = MemStateRef("p.pro", "k", SnapshotType.PROLOGUE).open()
            epi = MemStateRef("e.epi", "k", SnapshotType.EPILOGUE).open()

            assert pro.matches_unlaunched(epi)
            fake.MnemePy_MatchesUnlaunched.assert_called_once_with("P", "E")


# ======================================================================
#                      RecordedExecution Tests
# ======================================================================

@pytest.mark.parametrize("with_source", [False, True])
def test_recorded_execution_to_dict(with_source):
    fake_instance = MagicMock()
    fake_instance.to_dict.return_value = {"X": 1}

    source_kwargs = {}
    if with_source:
        source_kwargs = dict(
            source_file="/src/k.cu",
            source_copy="RecordedSource_ab_k.cu",
            source_md5="ab",
            source_line=11,
            source_end_line=25,
        )

    r = RecordedExecution(
        static_hash="S",
        kernel_name="K",
        demangled_name="DK",
        llvm_files=["a.ll"],
        arg_names=["a0"],
        specializations=[True],
        va_addr="0x100",
        va_size=32,
        kernel_instances={"hashX": fake_instance},
        **source_kwargs,
    )

    d = r.to_dict()

    assert d["KernelName"] == "K"
    assert d["Modules"] == ["a.ll"]
    assert d["instances"]["hashX"] == {"X": 1}
    if with_source:
        assert d["SourceFile"] == "/src/k.cu"
        assert d["SourceCopy"] == "RecordedSource_ab_k.cu"
        assert d["SourceMD5"] == "ab"
        assert d["SourceLine"] == 11
        assert d["SourceEndLine"] == 25
    else:
        assert not {
            "SourceFile",
            "SourceCopy",
            "SourceMD5",
            "SourceLine",
            "SourceEndLine",
        } & d.keys()


def test_recorded_execution_link_llvm_modules_calls_jit():
    with patch("mneme.recorded_execution.jit.link_llvm_modules") as fake_link:
        fake_link.return_value = "MOD"

        r = RecordedExecution(
            "S", "K", "DK",
            ["a.ll", "b.ll"],
            ["x"],
            [True, False],
            "0x100",
            128,
            {},
        )

        out = r.link_llvm_modules(prune=True, internalize=False)

        fake_link.assert_called_once_with(["a.ll", "b.ll"], "K", True, False)
        assert out == "MOD"


def test_make_path_relative_accepts_basename_but_rejects_nested_relative(tmp_path):
    """
    The path transform is idempotent for already-relative basenames, but
    rejects relative paths that would be reinterpreted relative to the JSON file.
    """
    assert _make_path_relative("file.epi", tmp_path) == "file.epi"

    with pytest.raises(ValueError, match="Expected absolute path or basename"):
        _make_path_relative("record-db/file.epi", tmp_path)


@pytest.mark.parametrize("with_source", [False, True])
@pytest.mark.parametrize("path_style", ["basename", "absolute"])
def test_recorded_execution_from_json_reconstructs(tmp_path, path_style, with_source):
    """
    from_json must resolve relative path entries against the JSON file's
    parent directory and pass absolute path entries through unchanged. In
    both cases, the in-memory paths end up absolute under tmp_path.
    """
    mod_path = tmp_path / "modA.ll"
    pro_path = tmp_path / "file.pro"
    epi_path = tmp_path / "file.epi"
    copy_path = tmp_path / "RecordedSource_ab_K.cu"
    for p in (mod_path, pro_path, epi_path):
        p.touch()

    if path_style == "basename":
        modules = [mod_path.name]
        prologue = pro_path.name
        epilogue = epi_path.name
        source_copy = copy_path.name
    else:
        modules = [str(mod_path)]
        prologue = str(pro_path)
        epilogue = str(epi_path)
        source_copy = str(copy_path)

    data = {
        "StaticHash": "S",
        "KernelName": "K",
        "DemangledName": "DK",
        "Modules": modules,
        "ArgNames": ["x"],
        "Specializations": [True, True, False],
        "VAddr": "ADDR",
        "VASize": 64,
        "instances": {
            "H": {
                "Args": [True, False],
                "SharedMem": 0,
                "BlockDims": {"x": 1, "y": 2, "z": 3},
                "GridDims": {"x": 4, "y": 5, "z": 6},
                "Occurrences": 3,
                "Prologue": prologue,
                "Epilogue": epilogue,
            }
        },
    }

    if with_source:
        data["SourceFile"] = "/src/K.cu"
        data["SourceCopy"] = source_copy
        data["SourceMD5"] = "ab"
        data["SourceLine"] = 11
        data["SourceEndLine"] = 25

    json_path = tmp_path / "db.json"
    json_path.write_text(json.dumps(data))

    r = RecordedExecution.from_json(str(json_path))

    assert r.kernel_name == "K"
    if with_source:
        assert r.source_file == "/src/K.cu"
        assert r.source_copy == str(copy_path)
        assert r.source_md5 == "ab"
        assert r.source_line == 11
        assert r.source_end_line == 25
    else:
        assert r.source_file is None
        assert r.source_copy is None
        assert r.source_md5 is None
        assert r.source_line is None
        assert r.source_end_line is None
    assert "H" in r.kernel_instances
    inst = r.kernel_instances["H"]
    assert inst.execution is r
    assert inst.block_dim.x == 1
    assert inst.grid_dim.z == 6
    assert inst.occ == 3

    assert r.llvm_files == [str(mod_path)]
    assert inst.prologue.fn == str(pro_path)
    assert inst.epilogue.fn == str(epi_path)
    assert inst.epilogue.base_prologue_fn == str(pro_path)
    assert inst.epilogue.s_type == SnapshotType.EPILOGUE


def test_recorded_execution_kernel_source_slices_recorded_copy(tmp_path):
    """
    kernel_source returns the inclusive, 1-based line range of the kernel, None
    when the record carries no line information, and skips a candidate whose
    contents no longer match the recorded checksum.
    """
    text = "line1\nline2\nline3\nline4\n"
    digest = hashlib.md5(text.encode()).hexdigest()
    copy_path = tmp_path / "RecordedSource_ab_K.cu"
    copy_path.write_text(text)

    def make(**source_kwargs):
        source_kwargs.setdefault("source_copy", str(copy_path))
        return RecordedExecution(
            "S", "K", "DK", ["a.ll"], ["x"], [True], "0x100", 32, {},
            **source_kwargs,
        )

    source = make(source_line=2, source_end_line=3).kernel_source()
    assert source.text == "line2\nline3\n"
    assert source.file == str(copy_path)
    assert (source.line, source.end_line) == (2, 3)
    assert source.location == f"{copy_path}:2-3"
    assert make().kernel_source() is None
    assert (
        make(source_line=2, source_end_line=3, source_md5=digest).kernel_source().text
        == "line2\nline3\n"
    )
    assert (
        make(source_line=2, source_end_line=3, source_md5="0" * 32).kernel_source()
        is None
    )

    edited = tmp_path / "edited.cu"
    edited.write_text("inserted\n" + text)
    unchanged = tmp_path / "K.cu"
    unchanged.write_text(text)
    source = make(
        source_copy=str(edited),
        source_file=str(unchanged),
        source_md5=digest,
        source_line=2,
        source_end_line=3,
    ).kernel_source()
    assert source.text == "line2\nline3\n"
    assert source.file == str(unchanged)


@pytest.mark.parametrize("layout", ["in_dir", "out_of_dir"])
def test_to_json_relativizes_in_dir_files_only(tmp_path, layout):
    """
    to_json writes basenames for artifacts in the JSON's parent directory and
    keeps absolute paths for artifacts outside it.
    """
    json_dir = tmp_path / "db"
    json_dir.mkdir()
    artifact_dir = json_dir if layout == "in_dir" else (tmp_path / "elsewhere")
    if layout != "in_dir":
        artifact_dir.mkdir()

    mod_path = artifact_dir / "mod.bc"
    pro_path = artifact_dir / "kernel.pro"
    epi_path = artifact_dir / "kernel.epi"
    for p in (mod_path, pro_path, epi_path):
        p.touch()

    instance = RecordedExecution.KernelInstance(
        static_hash="S",
        dynamic_hash="H",
        kernel_name="K",
        args=[True],
        shared_mem=0,
        block_dim=dim3(1, 1, 1),
        grid_dim=dim3(1, 1, 1),
        specializations=[True],
        occ=1,
        prologue_fn=str(pro_path),
        epilogue_fn=str(epi_path),
    )

    r = RecordedExecution(
        static_hash="S",
        kernel_name="K",
        demangled_name="DK",
        llvm_files=[str(mod_path)],
        arg_names=["x"],
        specializations=[True],
        va_addr="0x100",
        va_size=64,
        kernel_instances={"H": instance},
    )

    json_path = json_dir / "db.json"
    r.to_json(str(json_path))

    written = json.loads(json_path.read_text())

    if layout == "in_dir":
        assert written["Modules"] == ["mod.bc"]
        assert written["instances"]["H"]["Prologue"] == "kernel.pro"
        assert written["instances"]["H"]["Epilogue"] == "kernel.epi"
    else:
        assert written["Modules"] == [str(mod_path)]
        assert written["instances"]["H"]["Prologue"] == str(pro_path)
        assert written["instances"]["H"]["Epilogue"] == str(epi_path)


def _write_counted_record(tmp_path, **fields):
    mod_path = tmp_path / "modA.ll"
    pro_path = tmp_path / "file.pro"
    epi_path = tmp_path / "file.epi"
    for p in (mod_path, pro_path, epi_path):
        p.touch()

    data = {
        "StaticHash": "S",
        "KernelName": "K",
        "DemangledName": "DK",
        "Modules": [mod_path.name],
        "ArgNames": ["x"],
        "Specializations": [True],
        "VAddr": "ADDR",
        "VASize": 64,
        "instances": {
            "H": {
                "Args": [],
                "SharedMem": 0,
                "BlockDims": {"x": 32, "y": 1, "z": 1},
                "GridDims": {"x": 1, "y": 1, "z": 1},
                "Occurrences": 3,
                "Prologue": pro_path.name,
                "Epilogue": epi_path.name,
            }
        },
        **fields,
    }
    json_path = tmp_path / "db.json"
    json_path.write_text(json.dumps(data))
    return json_path


def test_recorded_execution_loads_launch_counts(tmp_path):
    unrecorded = {
        "U": {
            "BlockDims": {"x": 64, "y": 1, "z": 1},
            "GridDims": {"x": 2, "y": 1, "z": 1},
            "Occurrences": 4,
            "SharedMem": 16,
        }
    }
    json_path = _write_counted_record(
        tmp_path, TotalLaunches=7, UnrecordedInstances=unrecorded
    )

    r = RecordedExecution.from_json(str(json_path))

    assert r.total_launches == 7
    assert list(r) == ["H"]
    assert r["H"].occ == 3
    u = r.unrecorded_instances["U"]
    assert (u.block_dim.x, u.grid_dim.x, u.shared_mem, u.occ) == (64, 2, 16, 4)

    out_path = tmp_path / "out.json"
    r.to_json(str(out_path))
    written = json.loads(out_path.read_text())
    assert written["TotalLaunches"] == 7
    assert written["UnrecordedInstances"] == unrecorded
    assert written["instances"]["H"]["Occurrences"] == 3


def test_recorded_execution_loads_record_without_launch_counts(tmp_path):
    r = RecordedExecution.from_json(str(_write_counted_record(tmp_path)))

    assert r.total_launches is None
    assert r.unrecorded_instances == {}
    assert r["H"].occ == 3
    d = r.to_dict()
    assert "TotalLaunches" not in d
    assert d["UnrecordedInstances"] == {}


def test_recorded_execution_loads_record_without_instances(tmp_path):
    unrecorded = {
        "U": {
            "BlockDims": {"x": 64, "y": 1, "z": 1},
            "GridDims": {"x": 2, "y": 1, "z": 1},
            "Occurrences": 5,
            "SharedMem": 0,
        }
    }
    json_path = _write_counted_record(
        tmp_path, instances={}, TotalLaunches=5, UnrecordedInstances=unrecorded
    )

    r = RecordedExecution.from_json(str(json_path))

    assert len(r) == 0
    assert r.total_launches == 5
    assert r.unrecorded_instances["U"].occ == 5
