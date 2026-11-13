"""Real PROD 2-domain CPU run (probe outputs): in-run extract == offline recomputation; wrfout OFF identity.

Produce the inputs with ``real_nested_extract_probe.py`` (≈12 min each on CPU):
  main snapshot  --mode off -> <runs>/main_off
  this branch    --mode on  -> <runs>/branch_on
then ``V50_REAL_RUNS=<runs> pytest tests/v50_extract/test_real_nested_parity.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from . import oracle

RUNS = os.environ.get("V50_REAL_RUNS")
pytestmark = pytest.mark.skipif(not RUNS, reason="set V50_REAL_RUNS to the probe output root")


def _receipt(name):
    path = Path(RUNS) / name / "receipt.json"
    if not path.is_file():
        pytest.skip(f"probe {name} missing")
    receipt = json.loads(path.read_text())
    assert receipt["phase"] == "PASS", receipt.get("exception")
    return receipt


def _inventory(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        return {n: (v.dtype.str, v.dimensions, np.asarray(v[:]).tobytes(),
                    {k: np.asarray(v.getncattr(k)).tobytes() for k in v.ncattrs()})
                for n, v in ds.variables.items()}, {k: str(ds.getncattr(k)) for k in ds.ncattrs()}


def test_extract_on_writer_bytes_equal_off_writer_and_main():
    on = _receipt("branch_on")
    base = _receipt("main_off")
    frames = {}
    for rec in on["frames"] + [dict(f, writer="main") for f in base["frames"]]:
        frames.setdefault((rec["domain"], rec["step"]), {})[rec["writer"]] = rec
    assert set(frames) == {("d01", 0), ("d02", 0), ("d01", 1), ("d02", 3)}
    for key, by in frames.items():
        assert set(by) == {"off", "on", "main"}, key
        # Same process, same carry: ON writer bytes == OFF writer bytes (file-level sha).
        assert by["on"]["sha256"] == by["off"]["sha256"], key
        # Main snapshot vs this branch (separate process): every variable, attribute and global attribute.
        assert _inventory(by["main"]["path"]) == _inventory(by["off"]["path"]), key
        assert by["main"]["sha256"] == by["off"]["sha256"], key


def test_in_run_extract_equals_offline_recomputation():
    import netCDF4
    on = _receipt("branch_on")
    assert on["extract"]["status"] == "READY", on["extract"]
    out = Path(RUNS) / "branch_on"
    manifest = json.loads((out / "extract/manifest.json").read_text())
    znt = np.load(out / "ZNT_carry.npz")
    nontrivial = {"cloud_base_ok": 0, "inv_ok": 0}
    for domain in ("d01", "d02"):
        nc = out / "extract" / f"extract_prod20260725probe_m000_{domain}.nc"
        assert hashlib.sha256(nc.read_bytes()).hexdigest() == manifest["artifacts"][nc.name]["sha256"]
        wrfouts = sorted(Path(f["path"]) for f in on["frames"] if f["writer"] == "on" and f["domain"] == domain)
        steps = sorted(f["step"] for f in on["frames"] if f["writer"] == "on" and f["domain"] == domain)
        with netCDF4.Dataset(nc) as ds:
            ds.set_auto_maskandscale(False)
            assert list(ds["own_step"][:]) == steps
            assert ds["time_offset_seconds"][:].tolist() == [0, 0]  # interval = 54 s in this probe
            sst_available = oracle.sst_update("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")  # PROD: no sst_update
            for i, path in enumerate(wrfouts):
                ref = oracle.recompute(path, sst_available=sst_available)
                assert int(ds["actual_time_unix_s"][i]) == ref["actual_time_unix_s"]
                for name, value in ref.items():
                    if name == "actual_time_unix_s":
                        continue
                    got = np.asarray(ds[name][i] if ds[name].dimensions[0] == "time" else ds[name][:])
                    want = np.asarray(value)
                    assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), (domain, i, name)
                assert np.asarray(ds["ZNT"][i]).tobytes() == znt[f"{domain}_s{steps[i]}"].tobytes()
                nontrivial["cloud_base_ok"] += int((np.asarray(ds["cloud_base_status"][i]) == 0).sum())
                nontrivial["inv_ok"] += int((np.asarray(ds["inv_status"][i]) == 0).sum())
    # Cold start: the first PROD frames carry no condensate (QCLOUD+QICE == 0, CLDFRA == 0), so here the
    # cloud fields are trivially equal; non-trivial cloud parity is test_v1_extract (Swiss CPU-WRF frames).
    assert nontrivial["inv_ok"] > 0, nontrivial
