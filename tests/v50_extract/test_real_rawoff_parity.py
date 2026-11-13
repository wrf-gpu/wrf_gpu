"""Real PROD 2-domain CPU run, M2: RAW-off extract == full-history extract; in-run == offline/WT1; sea profiles.

Produce the input with ``real_nested_extract_probe.py --mode rawoff --points POINTS_V2.csv SEA_TRANSECTS_V1.csv``
(≈20 min on CPU) into <runs>/branch_rawoff, then
``V50_REAL_RUNS=<runs> pytest tests/v50_extract/test_real_rawoff_parity.py``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from . import oracle

RUNS = os.environ.get("V50_REAL_RUNS")
pytestmark = pytest.mark.skipif(not RUNS, reason="set V50_REAL_RUNS to the probe output root")
STEM = "prod20260725probe_m000"


def _run():
    out = Path(RUNS) / "branch_rawoff"
    path = out / "receipt.json"
    if not path.is_file():
        pytest.skip("probe branch_rawoff missing")
    receipt = json.loads(path.read_text())
    assert receipt["phase"] == "PASS", receipt.get("exception")
    return out, receipt


def _variables(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        out = {}
        for n, v in ds.variables.items():
            data = v[:]
            raw = "\x1f".join(data.tolist()) if v.dtype is str else np.asarray(data).tobytes()
            out[n] = (str(v.dtype), v.dimensions, raw,
                      {k: str(v.getncattr(k)) for k in v.ncattrs()})
        return out


def test_raw_off_writes_nothing_and_fetches_less():
    out, rec = _run()
    assert rec["history_rawoff_entries"] == []
    assert {r["writer"] for r in rec["frames"]} == {"full", "rawoff"}
    by = {}
    for r in rec["d2h"]:
        by.setdefault((r["domain"], r["step"]), {})[r["writer"]] = r
    assert set(by) == {("d01", 0), ("d02", 0), ("d01", 1), ("d02", 3)}
    for key, w in by.items():
        assert w["rawoff"]["bytes"] < w["full"]["bytes"], key
        assert w["rawoff"]["n_fields"] < w["full"]["n_fields"], key


def test_raw_off_extract_equals_full_history_extract():
    out, rec = _run()
    full = json.loads((out / "extract_full/manifest.json").read_text())
    off = json.loads((out / "extract_rawoff/manifest.json").read_text())
    assert full["status"] == off["status"] and full["status"] in ("READY", "PARTIAL"), full["domains"]
    for d in ("d01", "d02"):
        assert full["domains"][d]["problems"] == [] == off["domains"][d]["problems"]
    names = sorted(p.name for p in (out / "extract_full").iterdir() if p.suffix in (".nc", ".csv"))
    assert names == sorted(p.name for p in (out / "extract_rawoff").iterdir() if p.suffix in (".nc", ".csv"))
    assert {f"extract_{STEM}_d01.nc", f"profiles_{STEM}_d02.nc", "points.csv"} <= set(names)
    for name in names:
        a, b = out / "extract_full" / name, out / "extract_rawoff" / name
        if name.endswith(".csv"):
            assert a.read_bytes() == b.read_bytes()
        else:
            assert _variables(a) == _variables(b), name


def test_in_run_extract_equals_offline_reextract_of_the_written_history(tmp_path):
    """gpuwrf.extract.offline over the FULL writer's wrfouts (ZNT from the carry) == the in-run extract."""
    from gpuwrf.extract.offline import extract_wrfouts
    from gpuwrf.extract.request import load_request
    out, rec = _run()
    znt = np.load(out / "ZNT_carry.npz")
    files = {}
    for r in rec["frames"]:
        if r["writer"] == "full":
            files.setdefault(r["domain"], []).append(Path(r["path"]))
    # Same request, point CSVs from SHA-identical copies in the run dir (pytest guards <DATA_ROOT>/alisios reads).
    doc = json.loads((out / "extract_request_full.json").read_text())
    copies = [str(out / "points_input" / Path(p).name) for p in doc["points_csv"]]
    bound = json.loads((out / "extract_full/manifest.json").read_text())["points"]["sources"]
    assert [hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in copies] == [s["sha256"] for s in bound]
    cfg = tmp_path / "req.json"
    cfg.write_text(json.dumps({**doc, "points_csv": copies}))
    req = load_request("v1", cfg, tmp_path / "ex")
    dt = {k: float(v) for k, v in rec["dt_s"].items()}
    cadence = {"d01": 1, "d02": 3}
    import netCDF4
    steps = {}
    for d in files:  # explicit schedule = the in-run extract's own_step axis (the probe writes every root step)
        with netCDF4.Dataset(out / f"extract_full/extract_{STEM}_{d}.nc") as ds:
            steps[d] = tuple(int(x) for x in ds["own_step"][:])
    sst = "file" if oracle.sst_update("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725") else "none"
    rep = extract_wrfouts(req, files, interval_s={d: cadence[d] * dt[d] for d in files}, expected_steps=steps,
                          sst_source=sst,
                          input_dir=Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725"),
                          znt_source=lambda d, step, path: znt[f"{d}_s{step}"])
    assert rep["status"] == json.loads((out / "extract_full/manifest.json").read_text())["status"]
    for name in sorted(p.name for p in (tmp_path / "ex").iterdir() if p.suffix in (".nc", ".csv")):
        a, b = out / "extract_full" / name, tmp_path / "ex" / name
        if name.endswith(".csv"):
            assert a.read_bytes() == b.read_bytes()
        else:
            assert _variables(a) == _variables(b), name


def test_in_run_profiles_equal_wt1_teacher_and_grid_equals_oracle():
    import netCDF4
    from . import wt1_pins
    from gpuwrf.extract import legacy
    out, rec = _run()
    issue_s = "20260725_18z"
    for d in ("d01", "d02"):
        frames = sorted(Path(r["path"]) for r in rec["frames"] if r["writer"] == "full" and r["domain"] == d)
        with netCDF4.Dataset(out / f"extract_full/extract_{STEM}_{d}.nc") as ds:
            ds.set_auto_maskandscale(False)
            sst_available = oracle.sst_update("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")  # PROD: no sst_update
            for i, path in enumerate(frames):
                ref = oracle.recompute(path, sst_available=sst_available)
                for name, value in ref.items():
                    if name == "actual_time_unix_s":
                        continue
                    got = np.asarray(ds[name][i] if ds[name].dimensions[0] == "time" else ds[name][:])
                    assert got.tobytes() == np.asarray(value).tobytes(), (d, i, name)
        prof = out / f"extract_full/profiles_{STEM}_{d}.nc"
        with netCDF4.Dataset(prof) as ds:
            ids = list(ds["point_id"][:])
            kinds = dict(zip(ids, ds["point_kind"][:]))
            gy, gx = np.asarray(ds["grid_y"][:]), np.asarray(ds["grid_x"][:])
        # WT1's time rule needs [nominal, nominal+60 s): this probe's frames are 0 and 54 s.
        # Sub-hourly probe frames (0 and 54 s): per-frame field parity only (the axis check refuses them).
        tables = legacy._wt1_tables(prof, attempt=1, roles=None, validate_axis=False)
        for t, path in enumerate(frames):
            with netCDF4.Dataset(path) as src:
                prof_ref = wt1_pins.frame_profiles(src)
                act = datetime.fromtimestamp(oracle.frame_time_unix(path), tz=timezone.utc)
                nominal, _ = wt1_pins.time_coordinates(act, issue_s)
                pts = [{"phys_id": i, "role": kinds[i], "grid_y": int(y), "grid_x": int(x),
                        "profile_status": "AVAILABLE"} for i, y, x in zip(ids, gy, gx)]
                ref = wt1_pins.profile_table(src, prof_ref, pts, issue_s, 1, act, nominal)
            assert tables[t].equals(ref), (d, t)


def test_frozen_point_lists_mapped_and_upstream_sea_profiles_present():
    out, rec = _run()
    man = json.loads((out / "extract_full/manifest.json").read_text())
    pts = man["points"]
    assert [s["sha256"] for s in pts["sources"]] == [
        "6838b999a00713a72e83a026ba59d63b953500d273442f7960a38ecbddead487",   # POINTS_V2.csv (cross 0803Z)
        "ef932ba17e619c44c7ff1dea63eb49f0ff0489a77020baf99d0e72b1ce03f41b"]   # SEA_TRANSECTS_V1.csv
    assert len(pts["NO_COORDINATE"]) == 68
    rows = list(csv.DictReader((out / "extract_full/points.csv").open()))
    assert hashlib.sha256((out / "extract_full/points.csv").read_bytes()).hexdigest() == pts["points_csv"]["sha256"]
    ne = [r for r in rows if r["domain"] == "d02" and r["src_zone"] == "NE_Meer"]
    ok = [r for r in ne if r["status"] == "OK"]
    assert len(ne) == 133
    assert len(ok) >= 3, {r["status"] for r in ne}
    for r in ok:
        assert float(r["distance_m"]) < 1.0 and r["grid_y"] == r["bound_y"] and r["grid_x"] == r["bound_x"]
        assert float(r["LANDMASK"]) == 0.0
    guimar = [r for r in rows if r["id"] == "wmo:60018"]
    assert {r["domain"]: r["status"] for r in guimar} == {"d01": "OK", "d02": "OK"}
