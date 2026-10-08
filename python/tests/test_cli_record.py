# python/tests/test_record_cli.py

import argparse
import os
from pathlib import Path

import mneme.commands as record_mod
import pytest
from mneme.commands import Record
from mneme.commands import utils as utils_mod

# -------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------


@pytest.fixture
def record_parser():
    """Create a parser with the Record CLI wired in."""
    parser = argparse.ArgumentParser(prog="mneme record")
    Record.set_cli_args(parser)
    return parser


# -------------------------------------------------------------
# Tests
# -------------------------------------------------------------


def test_record_happy_path(tmp_path, record_parser, monkeypatch):
    """
    Test the happy path: record with valid args, capturing env and cmd
    without running HIP code.
    """

    record_dir = tmp_path / "records"
    record_dir.mkdir()

    fake_binary = "/usr/bin/true"
    fake_lib = "/tmp/libmneme_record.so"

    # Monkeypatch env-wiring helpers
    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: fake_lib)

    captured = {}

    class FakeCode:
        returncode = 0

    # Monkeypatch subprocess.run so we don't run anything real
    def fake_run(cmd, env=None, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env or {}
        return FakeCode

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        [
            "--record-db-dir",
            str(record_dir),
            "--per-kernel-max-recordings",
            "3",
            "--per-kernel-skip-recordings",
            "2",
            "--",
            fake_binary,
            "42",
        ]
    )

    Record.run(args, verbosity="DEBUG")

    # Validate command passthrough
    assert captured["cmd"] == [fake_binary, "42"]

    # Validate environment
    env = captured["env"]
    assert env["LD_PRELOAD"] == fake_lib
    assert env["MNEME_MAX_RECORDINGS"] == "3"
    assert env["MNEME_SKIP_RECORDINGS"] == "2"
    assert env["MNEME_DATA_DIR"] == str(record_dir)
    assert env["MNEME_EPILOGUE_TYPE"] == "diff"
    assert env["MNEME_LOG_LEVEL"] == "DEBUG"


@pytest.mark.parametrize("epilogue_format", ["bytes", "diff", "best"])
def test_record_epilogue_format_sets_env(
    tmp_path, record_parser, monkeypatch, epilogue_format
):
    """--epilogue-format overrides the MNEME_EPILOGUE_TYPE runtime config."""
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    captured = {}

    class FakeCode:
        returncode = 0

    def fake_run(cmd, env=None, **kwargs):
        captured["env"] = env or {}
        return FakeCode

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        [
            "--record-db-dir",
            str(record_dir),
            "--epilogue-format",
            epilogue_format,
            "--",
            "/usr/bin/true",
        ]
    )

    Record.run(args, verbosity=None)

    assert captured["env"]["MNEME_EPILOGUE_TYPE"] == epilogue_format


def test_record_rejects_invalid_epilogue_format(record_parser):
    """argparse rejects epilogue formats not supported by the native config."""
    with pytest.raises(SystemExit):
        record_parser.parse_args(
            ["--epilogue-format", "delta", "--", "/usr/bin/true"]
        )


def test_record_requires_dash_dash(record_parser, monkeypatch):
    """Record requires '--' before the command."""
    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    args = record_parser.parse_args(
        ["--record-db-dir", "/tmp", "vec_add", "123"]  # <-- missing "--"
    )

    with pytest.raises(SystemExit):
        Record.run(args, verbosity=None)


def test_record_creates_dir_if_missing(record_parser, tmp_path, monkeypatch):
    """A missing record-db-dir is created automatically."""
    missing = tmp_path / "nested" / "nope"

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    class FakeCode:
        returncode = 0

    monkeypatch.setattr(record_mod.subprocess, "run", lambda cmd, env=None, **kw: FakeCode)

    args = record_parser.parse_args(
        ["--record-db-dir", str(missing), "--", "/usr/bin/true"]
    )

    Record.run(args, verbosity=None)

    assert missing.is_dir()


def test_record_fails_if_not_a_directory(record_parser, tmp_path, monkeypatch):
    """record-db-dir must be a directory (not a regular file)."""
    file_path = tmp_path / "file"
    file_path.write_text("hello")

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    args = record_parser.parse_args(
        ["--record-db-dir", str(file_path), "--", "/usr/bin/true"]
    )

    with pytest.raises(NotADirectoryError):
        Record.run(args, verbosity=None)


def test_record_handles_missing_executable(record_parser, tmp_path, monkeypatch):
    """Missing executable → parser.error → SystemExit."""
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    def fake_run(cmd, env=None, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        [
            "--record-db-dir",
            str(record_dir),
            "--",
            "/this/does/not/exist",
        ]
    )

    with pytest.raises(SystemExit):
        Record.run(args, verbosity=None)


def test_record_handles_not_executable(record_parser, tmp_path, monkeypatch):
    """Non-executable binary → parser.error → SystemExit."""
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    fake_path = tmp_path / "fake"
    fake_path.write_text("hi")  # not executable

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    def fake_run(cmd, env=None, **kwargs):
        raise PermissionError

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        [
            "--record-db-dir",
            str(record_dir),
            "--",
            str(fake_path),
        ]
    )

    with pytest.raises(SystemExit):
        Record.run(args, verbosity=None)


def test_record_defaults(record_parser, tmp_path, monkeypatch):
    """Defaults are exported when not provided explicitly."""
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    captured = {}

    class FakeCode:
        returncode = 0

    def fake_run(cmd, env=None, **kwargs):
        captured["env"] = env
        return FakeCode

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        [
            "--record-db-dir",
            str(record_dir),
            "--",
            "/usr/bin/true",
        ]
    )

    Record.run(args, verbosity=None)

    assert captured["env"]["MNEME_SKIP_RECORDINGS"] == "0"


def _capture_env_with_args(record_parser, tmp_path, monkeypatch, extra_args):
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    captured = {}

    class FakeCode:
        returncode = 0

    def fake_run(cmd, env=None, **kwargs):
        captured["env"] = env
        return FakeCode

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        ["--record-db-dir", str(record_dir), *extra_args, "--", "/usr/bin/true"]
    )
    Record.run(args, verbosity=None)
    return captured["env"]


@pytest.mark.parametrize("flag", ["-vass", "--virtual-address-space-size"])
def test_record_vass_is_ignored(record_parser, tmp_path, monkeypatch, capsys, flag):
    monkeypatch.delenv("MNEME_PAGE_SIZE", raising=False)
    env = _capture_env_with_args(record_parser, tmp_path, monkeypatch, [flag, "8"])
    assert "MNEME_PAGE_SIZE" not in env
    assert "-vass is deprecated and ignored" in capsys.readouterr().err
    assert "-vass" not in record_parser.format_help()


@pytest.mark.parametrize("record_ranks", ["0,2", "all"])
def test_record_record_ranks_explicit(
    record_parser, tmp_path, monkeypatch, record_ranks
):
    env = _capture_env_with_args(
        record_parser, tmp_path, monkeypatch, ["--record-ranks", record_ranks]
    )
    assert env["MNEME_RECORD_RANKS"] == record_ranks


@pytest.mark.parametrize("copy_source", [False, True])
def test_record_copy_source_env(record_parser, tmp_path, monkeypatch, copy_source):
    """--copy-source sets MNEME_COPY_SOURCE; omitting it leaves the env unset."""
    monkeypatch.delenv("MNEME_COPY_SOURCE", raising=False)
    env = _capture_env_with_args(
        record_parser, tmp_path, monkeypatch, ["--copy-source"] if copy_source else []
    )
    if copy_source:
        assert env["MNEME_COPY_SOURCE"] == "1"
    else:
        assert "MNEME_COPY_SOURCE" not in env


def test_record_record_ranks_default_unset(record_parser, tmp_path, monkeypatch):
    """Omitting --record-ranks must not set MNEME_RECORD_RANKS in env, so the
    C++ default-policy logic sees env-absence and applies its rank-0-only rule."""
    monkeypatch.delenv("MNEME_RECORD_RANKS", raising=False)
    env = _capture_env_with_args(record_parser, tmp_path, monkeypatch, [])
    assert "MNEME_RECORD_RANKS" not in env
