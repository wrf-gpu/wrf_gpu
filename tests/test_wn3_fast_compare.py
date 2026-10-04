"""scripts/wn3_fast_compare.py (held read-only handles) on top of compare_wrfout_grid.py (E69 handle reuse).

Regression for review-writer 04:16Z: the comparator must open every file through the context-manager protocol, so a
`Dataset` substitute that is only a context manager (wn3_fast_compare's _Held) keeps working; the wrapper's report must
equal the direct comparator's report (modulo timing) with spatial splits ON; each file is opened exactly once.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
REPO = Path(__file__).resolve().parents[1]
COMPARE = REPO / "scripts" / "compare_wrfout_grid.py"
FAST = REPO / "scripts" / "wn3_fast_compare.py"
TIMES = ["2026-02-28_00:00:00", "2026-02-28_01:00:00", "2026-02-28_02:00:00"]


def _frame(path: Path, time: str, t2: np.ndarray) -> None:
    ny, nx = t2.shape
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0, :] = [c.encode() for c in time]
        yy, xx = np.meshgrid(np.arange(ny, dtype="f4"), np.arange(nx, dtype="f4"), indexing="ij")
        statics = {"HGT": yy * 100.0 + xx * 50.0, "LANDMASK": (xx > nx / 2).astype("f4"),
                   "XLAT": 28.0 + yy * 0.1, "XLONG": -17.0 + xx * 0.1, "T2": t2}
        for name, values in statics.items():
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = values


def _pairs(root: Path, perturb: float = 0.1, seed: int = 3) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    cpu, gpu = root / "cpu", root / "gpu"
    cpu.mkdir(parents=True)
    gpu.mkdir(parents=True)
    for i, t in enumerate(TIMES):
        base = rng.standard_normal((8, 10)).astype("f4") + 290.0 + i
        _frame(cpu / f"wrfout_d01_{t}", t, base)
        _frame(gpu / f"wrfout_d01_{t}", t, base + perturb * rng.standard_normal((8, 10)).astype("f4"))
    return cpu, gpu


def _args(cpu: Path, gpu: Path, out: Path) -> list[str]:
    return ["--cpu-dir", str(cpu), "--gpu-dir", str(gpu), "--domain", "d01", "--init", "2026-02-28T00:00:00Z",
            "--min-lead", "0", "--max-lead", "2", "--out-json", str(out), "--out-md", str(out.with_suffix(".md"))]


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in ("generated_utc", "elapsed_seconds")}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _run(script: Path, args: list[str]) -> dict:
    proc = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(Path(args[args.index("--out-json") + 1]).read_text())


def test_wn3_fast_compare_report_equals_direct_comparator(tmp_path):
    cpu, gpu = _pairs(tmp_path)
    direct = _run(COMPARE, _args(cpu, gpu, tmp_path / "direct.json"))
    fast = _run(FAST, _args(cpu, gpu, tmp_path / "fast.json"))
    assert json.dumps(_strip(direct), sort_keys=True) == json.dumps(_strip(fast), sort_keys=True)
    assert fast["pairing"]["paired_file_count"] == 3 and fast["spatial_splits"]["mask_counts"], "spatial splits must run"
    assert fast["field_summaries"]["T2"]["compared_lead_count"] == 3


def test_wn3_fast_compare_sees_a_mutation(tmp_path):
    cpu, gpu = _pairs(tmp_path, perturb=0.0)
    clean = _run(FAST, _args(cpu, gpu, tmp_path / "clean.json"))
    with netCDF4.Dataset(sorted(gpu.glob("wrfout_d01_*"))[2], "a") as ds:
        ds["T2"][0, 3, 4] += 7.0
    mutated = _run(FAST, _args(cpu, gpu, tmp_path / "mut.json"))
    assert clean["field_summaries"]["T2"]["overall"]["max_abs"] == 0.0
    assert mutated["field_summaries"]["T2"]["overall"]["max_abs"] == pytest.approx(7.0, abs=1e-3)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_held_handles_open_each_file_once(tmp_path, monkeypatch):
    cpu, gpu = _pairs(tmp_path)
    opened: list[str] = []
    real = netCDF4.Dataset

    def counting(path, *args, **kwargs):
        opened.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(netCDF4, "Dataset", counting)
    fast = _load(FAST, "wn3_fast_compare_under_test")
    try:
        assert fast.compare.main(_args(cpu, gpu, tmp_path / "inproc.json")) in (0, None)
    finally:
        fast._close_all()
        sys.modules.pop("wn3_fast_compare_under_test", None)
        sys.modules.pop("compare_wrfout_grid", None)
    files = sorted(str(p) for p in [*cpu.glob("wrfout_d01_*"), *gpu.glob("wrfout_d01_*")])
    assert sorted(opened) == files, "every wrfout must be opened exactly once through the held cache"


def test_comparator_uses_the_context_manager_protocol(tmp_path, monkeypatch):
    """Any context-manager Dataset substitute works: __enter__ supplies the dataset, __exit__ runs once per open."""
    cpu, gpu = _pairs(tmp_path)
    compare = _load(COMPARE, "compare_wrfout_grid_cm_probe")
    entered, exited = [], []

    class Holder:
        def __init__(self, path, mode="r"):
            self.path, self.ds = str(path), netCDF4.Dataset(path, mode)

        def __enter__(self):
            entered.append(self.path)
            return self.ds

        def __exit__(self, *exc):
            exited.append(self.path)
            self.ds.close()
            return False

    monkeypatch.setattr(compare, "Dataset", Holder)
    try:
        compare.main(_args(cpu, gpu, tmp_path / "cm.json"))
    finally:
        sys.modules.pop("compare_wrfout_grid_cm_probe", None)
    assert entered and sorted(entered) == sorted(exited), "every entered holder must be exited exactly once"
    report = json.loads((tmp_path / "cm.json").read_text())
    assert report["field_summaries"]["T2"]["compared_lead_count"] == 3
