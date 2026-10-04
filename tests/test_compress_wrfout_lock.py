"""scripts/compress_wrfout.py: CPU-only before any gpuwrf import, and every netCDF access under the product netCDF lock."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "compress_wrfout.py"


def _load():
    spec = importlib.util.spec_from_file_location("compress_wrfout_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(path: Path, shift: float = 0.0) -> Path:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 64)
        ds.createVariable("T2", "f4", ("x",))[:] = np.arange(64, dtype="f4") + shift
        ds.TITLE = "test"
    return path


def test_fresh_process_import_is_cpu_and_backend_dark():
    # GPU-looking caller env, no PYTHONPATH: the script must force CPU and find its own tree's gpuwrf.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "JAX_PLATFORMS")}
    env["JAX_PLATFORMS"] = "cuda"
    probe = (
        "import importlib.util, json, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('cw', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "from gpuwrf.io import netcdf_lock\n"
        "import gpuwrf\n"
        "xb = sys.modules.get('jax._src.xla_bridge')\n"
        "print(json.dumps({'jax_platforms': os.environ.get('JAX_PLATFORMS'), 'locked': m.Dataset is netcdf_lock.Dataset,\n"
        "  'gpuwrf': gpuwrf.__file__, 'backends': sorted(getattr(xb, '_backends', {}) or {}) if xb else []}))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], env=env, cwd="/", capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["jax_platforms"] == "cpu" and got["locked"] is True
    assert Path(got["gpuwrf"]).resolve().is_relative_to(REPO / "src")
    assert got["backends"] == [], "importing the compressor initialised a JAX backend"


class _CountingLock:
    def __init__(self):
        self.inner, self.entries = threading.RLock(), 0

    def __enter__(self):
        self.inner.acquire()
        self.entries += 1
        return self

    def __exit__(self, *exc):
        self.inner.release()
        return False


@pytest.mark.skipif(shutil.which("nccopy") is None, reason="nccopy not installed")
def test_compress_one_reads_and_compares_under_the_netcdf_lock(tmp_path, monkeypatch):
    module = _load()
    from gpuwrf.io import netcdf_lock

    counting = _CountingLock()
    monkeypatch.setattr(netcdf_lock, "NETCDF_LOCK", counting)
    path = _frame(tmp_path / "wrfout_d01_2026-02-28_00:00:00")
    rec = module.compress_one(str(path))
    assert rec["value_identical"] is True and rec["mismatches"] == []
    assert counting.entries >= 2, "compressor opened netCDF files outside the product lock"
    with netCDF4.Dataset(path) as ds:
        assert np.array_equal(ds["T2"][:], np.arange(64, dtype="f4"))


def test_mismatch_detection_still_reports_value_differences(tmp_path):
    module = _load()
    a, b = _frame(tmp_path / "a.nc"), _frame(tmp_path / "b.nc", shift=1.0)
    assert module.mismatches(a, b) == ["T2"]
    assert module.mismatches(a, a) == []
