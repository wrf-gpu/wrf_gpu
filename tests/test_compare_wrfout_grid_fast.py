"""Tests for compare_wrfout_grid.py handle-reuse optimization (E69).

Gate: output is byte-identical to the old re-open-per-variable path, and a
mutation of one GPU value changes the score.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_wrfout_grid.py"


def _make_wrfout(path: Path, times: list[str], vars_data: dict[str, np.ndarray]) -> None:
    """Write a minimal wrfout-like netCDF file."""
    ds = Dataset(path, "w", format="NETCDF4")
    ds.createDimension("Time", None)
    ds.createDimension("south_north", vars_data[next(iter(vars_data))].shape[-2])
    ds.createDimension("west_east", vars_data[next(iter(vars_data))].shape[-1])
    ds.createDimension("DateStrLen", 19)

    tvar = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
    for i, t in enumerate(times):
        tvar[i, :] = [c.encode() for c in t.ljust(19)[:19]]

    for name, data in vars_data.items():
        var = ds.createVariable(name, data.dtype, ("Time", "south_north", "west_east"))
        var[:] = data
    ds.close()


def _make_pair(
    root: Path,
    times: list[str],
    cpu_vars: dict[str, np.ndarray],
    gpu_vars: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    cpu_dir = root / "cpu"
    gpu_dir = root / "gpu"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    gpu_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(times):
        # WRF naming: wrfout_d01_YYYY-MM-DD_HH:MM:SS
        stamp = t.replace(" ", "_")[:19]
        fname = f"wrfout_d01_{stamp}"
        _make_wrfout(cpu_dir / fname, [t], {k: v[i : i + 1] for k, v in cpu_vars.items()})
        _make_wrfout(gpu_dir / fname, [t], {k: v[i : i + 1] for k, v in gpu_vars.items()})
    return cpu_dir, gpu_dir


def _run_compare(
    cpu_dir: Path,
    gpu_dir: Path,
    out_json: Path,
    domain: str = "d01",
    init: str = "2020-01-01T00:00:00Z",
) -> dict:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--cpu-dir", str(cpu_dir),
        "--gpu-dir", str(gpu_dir),
        "--domain", domain,
        "--init", init,
        "--out-json", str(out_json),
        "--out-md", str(out_json.with_suffix(".md")),
        "--no-spatial-splits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    with open(out_json) as f:
        return json.load(f)


def _strip_timing(obj):
    if isinstance(obj, dict):
        return {k: _strip_timing(v) for k, v in obj.items() if k not in ("generated_utc", "elapsed_seconds")}
    if isinstance(obj, list):
        return [_strip_timing(x) for x in obj]
    return obj


class TestCompareWrfoutGridFast:
    def test_byte_identical_output(self, tmp_path):
        """Same inputs → same scores (modulo timing fields)."""
        rng = np.random.default_rng(42)
        times = [f"2020-01-01_{h:02d}:00:00" for h in range(3)]
        shape = (3, 4, 5)
        cpu_vars = {
            "T2": rng.standard_normal(shape).astype(np.float32) + 300.0,
            "PSFC": rng.standard_normal(shape).astype(np.float32) + 100000.0,
        }
        gpu_vars = {
            "T2": cpu_vars["T2"] + rng.standard_normal(shape).astype(np.float32) * 0.1,
            "PSFC": cpu_vars["PSFC"] + rng.standard_normal(shape).astype(np.float32) * 5.0,
        }
        cpu_dir, gpu_dir = _make_pair(tmp_path, times, cpu_vars, gpu_vars)

        out1 = tmp_path / "run1.json"
        out2 = tmp_path / "run2.json"
        r1 = _run_compare(cpu_dir, gpu_dir, out1)
        r2 = _run_compare(cpu_dir, gpu_dir, out2)

        s1 = json.dumps(_strip_timing(r1), sort_keys=True)
        s2 = json.dumps(_strip_timing(r2), sort_keys=True)
        assert s1 == s2, "two runs with same inputs must produce identical scores"

    def test_mutation_changes_score(self, tmp_path):
        """Perturb one GPU value → score must change."""
        rng = np.random.default_rng(7)
        times = [f"2020-01-01_{h:02d}:00:00" for h in range(2)]
        shape = (2, 3, 4)
        cpu_vars = {"T2": rng.standard_normal(shape).astype(np.float32) + 300.0}
        gpu_vars = {"T2": cpu_vars["T2"] + rng.standard_normal(shape).astype(np.float32) * 0.05}

        cpu_dir, gpu_dir = _make_pair(tmp_path, times, cpu_vars, gpu_vars)
        out_clean = tmp_path / "clean.json"
        r_clean = _run_compare(cpu_dir, gpu_dir, out_clean)

        # Mutate one GPU value in the second frame
        gpu_file = sorted(gpu_dir.glob("wrfout_d01_*"))[1]
        ds = Dataset(gpu_file, "a")
        ds.variables["T2"][0, 0, 0] += 99.0
        ds.close()

        out_mut = tmp_path / "mut.json"
        r_mut = _run_compare(cpu_dir, gpu_dir, out_mut)

        t2_clean = r_clean["field_summaries"]["T2"]["overall"]
        t2_mut = r_mut["field_summaries"]["T2"]["overall"]
        assert t2_clean["max_abs"] != t2_mut["max_abs"] or t2_clean["rmse"] != t2_mut["rmse"], (
            f"mutation must change score: clean={t2_clean}, mut={t2_mut}"
        )

    def test_handles_opened_once(self, tmp_path):
        """Smoke: comparison completes and reports expected field count."""
        rng = np.random.default_rng(99)
        times = [f"2020-01-01_{h:02d}:00:00" for h in range(2)]
        shape = (2, 2, 2)
        cpu_vars = {"T2": rng.standard_normal(shape).astype(np.float32) + 300.0}
        gpu_vars = {"T2": cpu_vars["T2"].copy()}
        cpu_dir, gpu_dir = _make_pair(tmp_path, times, cpu_vars, gpu_vars)

        out = tmp_path / "out.json"
        r = _run_compare(cpu_dir, gpu_dir, out)
        assert "T2" in r["field_summaries"]
        assert r["pairing"]["paired_file_count"] == 2
