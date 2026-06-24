from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpuwrf.runtime.gpu_preflight import (
    GpuPreflightError,
    VramSnapshot,
    gpu_lock_status,
    min_free_vram_gib,
    resolve_min_free_vram_threshold,
    run_nested_gpu_preflight,
    select_nvidia_smi_snapshot,
)


def _locked_env(tmp_path: Path, token: str = "test-token"):
    lock_file = tmp_path / "gpu.lock"
    holder_file = tmp_path / "gpu.lock.holder"
    lock_file.touch()
    holder_file.write_text(f"holder=test pid=1 token={token} cmd=pytest\n", encoding="utf-8")
    fd = os.open(lock_file, os.O_RDWR)
    env = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_FD": str(fd),
        "GPUWRF_GPU_LOCK_FILE": str(lock_file),
        "GPUWRF_GPU_LOCK_TOKEN": token,
    }
    return env, fd, lock_file, holder_file


def test_min_free_vram_default_and_override():
    assert min_free_vram_gib({}) == 24.0
    assert min_free_vram_gib({}, total_gib=32.0) == 24.0
    assert min_free_vram_gib({}, total_gib=180.0) == 90.0
    assert min_free_vram_gib({"GPUWRF_MIN_FREE_VRAM_FRACTION": "0.25"}, total_gib=180.0) == 45.0
    assert min_free_vram_gib({"GPUWRF_MIN_FREE_VRAM_GIB": "18.5"}) == 18.5
    assert min_free_vram_gib({"GPUWRF_MIN_FREE_VRAM_GIB": "18.5"}, total_gib=180.0) == 18.5
    assert min_free_vram_gib({"GPUWRF_MIN_FREE_VRAM_GIB": "bad"}) == 24.0


def test_resolved_threshold_payload_values():
    threshold = resolve_min_free_vram_threshold({}, total_gib=180.0)
    assert threshold.min_free_gib == 90.0
    assert threshold.absolute_floor_gib == 24.0
    assert threshold.fraction == 0.5
    assert threshold.fractional_gib == 90.0
    assert "FRACTION" in threshold.source

    override = resolve_min_free_vram_threshold({"GPUWRF_MIN_FREE_VRAM_GIB": "12"}, total_gib=180.0)
    assert override.min_free_gib == 12.0
    assert override.fractional_gib is None
    assert "explicit override" in override.source


def test_nvidia_smi_selection_honors_cuda_visible_devices():
    rows = [
        "0, GPU-zero, 4096, 32768, 28672, RTX 5090",
        "1, GPU-one, 120000, 184320, 64320, NVIDIA B200",
    ]

    default = select_nvidia_smi_snapshot(rows, {})
    assert default.index == "0"
    assert default.uuid == "GPU-zero"

    selected_by_index = select_nvidia_smi_snapshot(rows, {"CUDA_VISIBLE_DEVICES": "1,0"})
    assert selected_by_index.index == "1"
    assert selected_by_index.name == "NVIDIA B200"
    assert selected_by_index.free_gib == pytest.approx(117.1875)

    selected_by_uuid = select_nvidia_smi_snapshot(rows, {"CUDA_VISIBLE_DEVICES": "GPU-one"})
    assert selected_by_uuid.index == "1"

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES selects GPU '2'"):
        select_nvidia_smi_snapshot(rows, {"CUDA_VISIBLE_DEVICES": "2"})


def test_gpu_lock_status_requires_wrapper_env(tmp_path):
    env, fd, lock_file, holder_file = _locked_env(tmp_path)
    try:
        status = gpu_lock_status(env, holder_file=holder_file, lock_file=lock_file)
        assert status["ok"] is True

        missing = dict(env)
        missing.pop("GPUWRF_GPU_LOCK_HELD")
        status = gpu_lock_status(missing, holder_file=holder_file, lock_file=lock_file)
        assert status["ok"] is False
        assert "GPUWRF_GPU_LOCK_HELD" in status["reason"]
    finally:
        os.close(fd)


def test_preflight_passes_with_lock_and_headroom(tmp_path, capsys):
    env, fd, _lock_file, _holder_file = _locked_env(tmp_path)
    env["GPUWRF_MIN_FREE_VRAM_GIB"] = "20"
    try:
        payload = run_nested_gpu_preflight(
            environ=env,
            holder_file=_holder_file,
            lock_file=_lock_file,
            query_memory=lambda: VramSnapshot("0", "RTX Test", 26 * 1024, 32 * 1024, 6 * 1024),
        )
    finally:
        os.close(fd)
    assert payload["status"] == "PASS"
    assert payload["vram"]["free_gib"] == 26.0
    assert payload["vram_threshold"]["source"] == "GPUWRF_MIN_FREE_VRAM_GIB explicit override"
    assert "nested GPU preflight PASS" in capsys.readouterr().err


def test_preflight_fails_below_headroom(tmp_path):
    env, fd, _lock_file, _holder_file = _locked_env(tmp_path)
    env["GPUWRF_MIN_FREE_VRAM_GIB"] = "24"
    try:
        with pytest.raises(GpuPreflightError) as raised:
            run_nested_gpu_preflight(
                environ=env,
                holder_file=_holder_file,
                lock_file=_lock_file,
                query_memory=lambda: VramSnapshot("0", "RTX Test", 20 * 1024, 32 * 1024, 12 * 1024),
            )
    finally:
        os.close(fd)
    assert "below resolved threshold" in str(raised.value)
    assert raised.value.payload["status"] == "FAIL"


def test_preflight_force_returns_forced_payload_without_lock():
    payload = run_nested_gpu_preflight(
        environ={"GPUWRF_FORCE_GPU_RUN": "1"},
        query_memory=lambda: VramSnapshot("0", "RTX Test", 1 * 1024, 32 * 1024, 31 * 1024),
    )
    assert payload["status"] == "FORCED"
    assert payload["forced"] is True
    assert payload["failures"]
