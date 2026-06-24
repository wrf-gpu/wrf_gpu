from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import b200_io_lib as b200_io  # noqa: E402
from b200_io_lib import DrainConfig, drain_once, stop_pull  # noqa: E402


def _write_block(path: Path, *, finite: bool = True, include_u10: bool = True, time_count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", time_count)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 4)
        t2 = ds.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
        values = np.arange(12 * time_count, dtype=np.float32).reshape(time_count, 3, 4)
        if not finite and time_count:
            values[0, 1, 1] = np.nan
        t2[:] = values
        if include_u10:
            u10 = ds.createVariable("U10", "f4", ("Time", "south_north", "west_east"))
            u10[:] = values + 1


def _config(
    tmp_path: Path,
    *,
    delete_after_copy: bool = False,
    target: str | None = None,
    target_cap_bytes: int | None = None,
    local_cap_bytes: int | None = None,
    expected_time_steps: int | None = None,
) -> DrainConfig:
    return DrainConfig(
        output_dir=tmp_path / "out",
        target=target or str(tmp_path / "target"),
        state_dir=tmp_path / "state",
        expected_vars=("T2", "U10"),
        expected_dims={"south_north": 3, "west_east": 4},
        min_age_seconds=0,
        local_cap_bytes=local_cap_bytes,
        target_cap_bytes=target_cap_bytes,
        delete_after_copy=delete_after_copy,
        expected_time_steps=expected_time_steps,
    )


def _fake_s3_run(remote: dict[str, dict], calls: list[list[str]], *, store_on_cp: bool = True):
    def fake_run(cmd, check=False, text=False, stdout=None, stderr=None):  # noqa: ANN001
        calls.append(list(cmd))
        if cmd[:3] == ["aws", "s3", "cp"]:
            if store_on_cp:
                src = Path(cmd[3])
                uri = cmd[4]
                metadata_arg = cmd[cmd.index("--metadata") + 1]
                metadata = dict(item.split("=", 1) for item in metadata_arg.split(","))
                remote[uri] = {"ContentLength": src.stat().st_size, "Metadata": metadata}
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["aws", "s3api", "head-object"]:
            bucket = cmd[cmd.index("--bucket") + 1]
            key = cmd[cmd.index("--key") + 1]
            uri = f"s3://{bucket}/{key}"
            if uri not in remote:
                if check:
                    raise subprocess.CalledProcessError(1, cmd, output="", stderr="NoSuchKey")
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="NoSuchKey")
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(remote[uri]), stderr="")
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    return fake_run


def test_drain_copies_verified_block_and_resume_skips_done(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    src = cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc"
    _write_block(src)
    first = drain_once(cfg)
    assert first["ok"], first["issues"]
    copied = Path(cfg.target) / src.relative_to(cfg.output_dir)
    assert copied.exists()
    assert first["blocks"][0]["state"] == "done"

    second = drain_once(cfg)
    assert second["ok"], second["issues"]
    assert second["blocks"][0]["state"] == "skipped_done"


def test_drain_delete_after_copy_removes_only_after_done_marker(tmp_path: Path) -> None:
    cfg = _config(tmp_path, delete_after_copy=True)
    src = cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc"
    _write_block(src)
    report = drain_once(cfg)
    assert report["ok"], report["issues"]
    assert not src.exists()
    done_markers = list((cfg.state_dir / "blocks").glob("*.done.json"))
    assert len(done_markers) == 1
    assert json.loads(done_markers[0].read_text(encoding="utf-8"))["source_deleted"] is True


def test_drain_rejects_nonfinite_block_without_copy(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    src = cfg.output_dir / "wrfout_d02_bad.nc"
    _write_block(src, finite=False)
    report = drain_once(cfg)
    assert not report["ok"]
    assert any(issue["code"] == "nonfinite_values" for issue in report["issues"])
    assert not (Path(cfg.target) / src.relative_to(cfg.output_dir)).exists()


def test_drain_rejects_missing_expected_variable(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _write_block(cfg.output_dir / "wrfout_d02_missing.nc", include_u10=False)
    report = drain_once(cfg)
    assert not report["ok"]
    assert any(issue["code"] == "expected_var_missing" for issue in report["issues"])


def test_drain_target_backpressure_blocks_copy(tmp_path: Path) -> None:
    cfg = _config(tmp_path, target_cap_bytes=1)
    src = cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc"
    _write_block(src)
    report = drain_once(cfg)
    assert not report["ok"]
    assert any(issue["code"] == "target_backpressure" for issue in report["issues"])
    assert not (Path(cfg.target) / src.relative_to(cfg.output_dir)).exists()


def test_s3_delete_after_copy_verifies_remote_before_deleting(tmp_path: Path, monkeypatch) -> None:
    remote: dict[str, dict] = {}
    calls: list[list[str]] = []
    monkeypatch.setattr(b200_io.subprocess, "run", _fake_s3_run(remote, calls))
    cfg = _config(tmp_path, target="s3://bucket/prefix", delete_after_copy=True)
    src = cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc"
    _write_block(src)

    report = drain_once(cfg)

    assert report["ok"], report["issues"]
    assert not src.exists()
    assert any(call[:3] == ["aws", "s3api", "head-object"] for call in calls)
    assert "s3://bucket/prefix/wrfout_d02_2026-06-23_01:00:00.nc" in remote


def test_s3_delete_after_copy_is_blocked_without_remote_readback(tmp_path: Path, monkeypatch) -> None:
    remote: dict[str, dict] = {}
    calls: list[list[str]] = []
    monkeypatch.setattr(b200_io.subprocess, "run", _fake_s3_run(remote, calls, store_on_cp=False))
    cfg = _config(tmp_path, target="s3://bucket/prefix", delete_after_copy=True)
    src = cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc"
    _write_block(src)

    report = drain_once(cfg)

    assert not report["ok"]
    assert src.exists(), "source must not be deleted when S3 head-object validation fails"
    assert any(issue["code"] == "delete_blocked" for issue in report["issues"])
    assert any(call[:3] == ["aws", "s3api", "head-object"] for call in calls)


def test_drain_reports_lost_target_after_delete(tmp_path: Path) -> None:
    cfg = _config(tmp_path, delete_after_copy=True)
    src = cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc"
    _write_block(src)
    first = drain_once(cfg)
    assert first["ok"], first["issues"]
    copied = Path(cfg.target) / src.relative_to(cfg.output_dir)
    copied.unlink()

    second = drain_once(cfg)

    assert not second["ok"]
    assert any(issue["code"] == "drained_block_lost" for issue in second["issues"])
    assert second["training_ready_blocks"] == []


def test_s3_target_byte_cap_blocks_upload(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fail_if_called(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        raise AssertionError("S3 copy/head should not run after byte-cap backpressure")

    monkeypatch.setattr(b200_io.subprocess, "run", fail_if_called)
    cfg = _config(tmp_path, target="s3://bucket/prefix", target_cap_bytes=1)
    _write_block(cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc")

    report = drain_once(cfg)

    assert not report["ok"]
    assert any(issue["code"] == "target_backpressure" for issue in report["issues"])
    assert calls == []


def test_drain_rejects_short_and_empty_time_blocks(tmp_path: Path) -> None:
    cfg = _config(tmp_path, expected_time_steps=2)
    _write_block(cfg.output_dir / "wrfout_d02_short.nc", time_count=1)
    _write_block(cfg.output_dir / "wrfout_d02_empty.nc", time_count=0)

    report = drain_once(cfg)

    assert not report["ok"]
    codes = {issue["code"] for issue in report["issues"]}
    assert "time_dimension_mismatch" in codes
    assert "time_dimension_short" in codes
    assert "empty_values" in codes


def test_atomic_local_copy_does_not_leave_final_target_on_failure(tmp_path: Path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    src = cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc"
    _write_block(src)

    def crash_copy(_src, dst):  # noqa: ANN001
        Path(dst).write_bytes(b"partial")
        raise OSError("copy crashed")

    monkeypatch.setattr(b200_io.shutil, "copy2", crash_copy)
    report = drain_once(cfg)

    final_target = Path(cfg.target) / src.relative_to(cfg.output_dir)
    assert not report["ok"]
    assert any(issue["code"] == "copy_failed" for issue in report["issues"])
    assert not final_target.exists()
    assert final_target.with_name(f"{final_target.name}.partial").exists()
    assert src.exists()


def test_over_local_cap_still_drains_before_reporting_backpressure(tmp_path: Path) -> None:
    cfg = _config(tmp_path, local_cap_bytes=1)
    src = cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc"
    _write_block(src)

    report = drain_once(cfg)

    assert not report["ok"]
    assert report["status"] == "BACKPRESSURE"
    assert (Path(cfg.target) / src.relative_to(cfg.output_dir)).exists()
    assert any(issue["code"] == "local_backpressure_remaining" for issue in report["issues"])


def test_ambiguous_relative_target_is_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path, target="relative-target")
    _write_block(cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc")

    report = drain_once(cfg)

    assert not report["ok"]
    assert any(issue["code"] == "ambiguous_target" for issue in report["issues"])


def test_stop_pull_writes_marker_and_drains(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    src = cfg.output_dir / "wrfout_d02_2026-06-23_01:00:00.nc"
    _write_block(src)
    report = stop_pull(cfg, grace_seconds=0)
    assert report["ok"], report["issues"]
    assert (cfg.output_dir / "B200_STOP_REQUESTED.json").exists()
    assert (Path(cfg.target) / src.relative_to(cfg.output_dir)).exists()
    assert report["stop_events"][0]["event"] == "stop_marker_written"


def test_drain_cli_synthetic_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "dryrun.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/b200_drain.py",
            "synthetic-dry-run",
            "--work-dir",
            str(tmp_path / "work"),
            "--json-out",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "B200 synthetic dry-run: PASS" in proc.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["manifest_validation"]["status"] == "PASS"
    assert report["stop_pull"]["status"] == "PASS"
    assert report["resume"]["status"] == "PASS"
