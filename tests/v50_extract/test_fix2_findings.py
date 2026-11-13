"""Regression tests for the v50-diag critic Delta 1 (CRITIC_VERDICT.md '## Delta 1', findings 2, 3, 5, 7, 8, 9).

Each test reproduces the critic's probe and fails on b6e954c90d (FIX1 head). Real inputs: Swiss CPU-WRF d01 frames.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.extract import points as PT
from gpuwrf.extract import v1_fields as V
from gpuwrf.extract.collector import DomainPlan, ExtractCollector
from gpuwrf.extract.request import load_request, require_resume_policy

from .test_fix1_findings import _Prepared, _State, _collector, _one_frame, _problems, _request, _start
from .test_v1_extract import _dims, raw_of, swiss_frames
from .test_v1_points import _static, _write_points


def _full_raw(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        names = tuple(dict.fromkeys(V.RAW_NAMES + PT.PROFILE_RAW))
        return {n: np.asarray(ds[n][0]).copy() for n in names if n in ds.variables}


# ----------------------------------------------------------------------------------------------- finding 2
def test_d2_sst_is_never_tsk_and_a_real_producer_is_named(tmp_path):
    raw = raw_of(swiss_frames(1)[0])
    diag = {n: raw[n] for n in V.SURFACE if n != "SST"}
    report, manifest = _one_frame(tmp_path, name="noprod", sources=(diag, None, None))
    st = manifest["domains"]["d01"]["field_status"]["SST"]
    assert st["status"] == "NOT_AVAILABLE" and report["status"] == "READY", _problems(manifest)
    assert manifest["domains"]["d01"]["field_stats"]["SST"]["nonfinite"] == raw["HGT"].size  # no TSK values
    report, manifest = _one_frame(tmp_path, name="prod", sources=(dict(diag, SST=raw["SST"]), None, None))
    st = manifest["domains"]["d01"]["field_status"]["SST"]
    assert st["status"] == "OK" and st["source"] == V.SST_SOURCE  # what the value is, same text as offline


def test_d2_offline_sst_provenance_is_declared(tmp_path):
    from gpuwrf.extract.offline import extract_wrfouts
    frames = swiss_frames(2)
    req = _request(tmp_path, "none")
    rep = extract_wrfouts(req, {"d01": frames}, interval_s={"d01": 3600.0}, forecast_s=3600.0, sst_source="none",
                          znt_source=lambda d, s, p: np.full((42, 42), 0.05, np.float32))
    man = json.loads((tmp_path / "none" / "manifest.json").read_text())
    assert rep["status"] == "READY" and man["domains"]["d01"]["field_status"]["SST"]["status"] == "NOT_AVAILABLE"
    with pytest.raises(ValueError, match="sst_source"):
        extract_wrfouts(_request(tmp_path, "undeclared"), {"d01": frames}, interval_s={"d01": 3600.0},
                        forecast_s=3600.0)


# ----------------------------------------------------------------------------------------------- finding 3
@pytest.mark.parametrize("probe", ["qvapor_minus_001", "fill_value_hfx", "nan_pressure"])
def test_d3_profile_and_input_qc_fail(tmp_path, probe):
    frame = swiss_frames(1)[0]
    raw = _full_raw(frame)
    if probe == "qvapor_minus_001":
        raw["QVAPOR"][3, 20, 20] = -0.01  # at the mapped station cell st:center (20, 20)
    elif probe == "fill_value_hfx":
        raw["HFX"][7, 7] = np.float32(9.96921e36)
    else:
        raw["P"][2, 20, 20] = np.nan
    points = _write_points(tmp_path, _static(frame))
    cfg = tmp_path / "req.json"
    cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "swiss", "member": "m000",
                               "issue_utc": "2023-01-14T18:00:00Z", "points_csv": [str(p) for p in points]}))
    req = load_request("v1", cfg, tmp_path / "ex")
    plan = DomainPlan("d01", 3600.0, 3600.0, (0,), mp_physics=8, dx_m=3000.0)
    col = ExtractCollector(req, run_start=_start(frame), plans={"d01": plan}, input_dir=tmp_path)
    col.submit_host("d01", 0, raw, _dims(raw), np.full(raw["HGT"].shape, 0.1, np.float32))
    report = col.finalize()
    man = json.loads((tmp_path / "ex/manifest.json").read_text())
    assert report["status"] == "FAIL", (probe, man["domains"]["d01"].get("problems"))


# ----------------------------------------------------------------------------------------------- finding 5
def test_d5_offline_requires_scheduled_nests_and_grid_id(tmp_path):
    from gpuwrf.extract.offline import extract_wrfouts
    frames = swiss_frames(2)
    znt = lambda d, s, p: np.full((42, 42), 0.05, np.float32)  # noqa: E731
    with pytest.raises(ValueError, match="d02"):
        extract_wrfouts(_request(tmp_path, "nod02"), {"d01": frames}, interval_s={"d01": 3600.0, "d02": 3600.0},
                        expected_steps={"d01": (0, 200), "d02": (0, 600)}, sst_source="file", znt_source=znt)
    with pytest.raises(ValueError, match="d02"):
        extract_wrfouts(_request(tmp_path, "reqd02", domains=["d01", "d02"]), {"d01": frames},
                        interval_s={"d01": 3600.0}, forecast_s=3600.0, sst_source="file", znt_source=znt)
    links = tmp_path / "links"; links.mkdir()
    tagged = []
    for f in frames:  # d01 content (GRID_ID=1) under d02 names
        q = links / f.name.replace("wrfout_d01_", "wrfout_d02_"); q.symlink_to(f); tagged.append(q)
    with pytest.raises(ValueError, match="GRID_ID"):
        extract_wrfouts(_request(tmp_path, "gridid"), {"d02": tagged}, interval_s={"d02": 3600.0},
                        forecast_s=3600.0, sst_source="file", znt_source=znt)


# ----------------------------------------------------------------------------------------------- finding 7
def test_d7_extension_of_a_non_ready_published_run_is_refused_before_any_write(tmp_path):
    # FIX3 (Delta 2 F7) gave the READY case a route (chained segment, test_fix3_findings); a published part that is
    # not READY is still refused at admission and at bind, before any write.
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    request, start = _request(tmp_path), _start(frame)
    col = _collector(tmp_path, request, start, expected=(0, 1))
    col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))
    assert col.finalize()["status"] == "PARTIAL"
    gen = tmp_path / "gen"; gen.mkdir()
    (gen / "manifest.json").write_text(json.dumps({"identity": {"output_policy": {"extract": request.policy()}}}))
    before = sorted((p.name, p.stat().st_mtime_ns) for p in (tmp_path / "ex").iterdir())
    with pytest.raises(ValueError, match="needs every published part READY"):
        require_resume_policy(gen, request)  # admission, before model initialization
    with pytest.raises(ValueError, match="needs every published part READY"):
        _collector(tmp_path, request, start, expected=(0, 1, 2), resumed=True)
    assert sorted((p.name, p.stat().st_mtime_ns) for p in (tmp_path / "ex").iterdir()) == before


# ----------------------------------------------------------------------------------------------- finding 8
def test_d8_no_ready_when_the_final_directory_fsync_fails(tmp_path, monkeypatch):
    import gpuwrf.extract.collector as C
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    col = _collector(tmp_path, _request(tmp_path), _start(frame))
    col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))
    col.join()
    real = C._fsync_dir

    def failing(path):
        if (path / "manifest.json").exists():
            raise OSError(5, "injected directory fsync failure")
        real(path)
    monkeypatch.setattr(C, "_fsync_dir", failing)
    report = col.finalize()
    path = tmp_path / "ex/manifest.json"
    assert report["status"] == "FAIL"
    assert not path.exists() or json.loads(path.read_text())["status"] == "FAIL"


# ----------------------------------------------------------------------------------------------- finding 9
def test_d9_raw_on_subset_does_not_expand_unrequested_nests(tmp_path):
    frame = swiss_frames(1)[0]
    req = _request(tmp_path, domains=["d01"])
    plans = {n: DomainPlan(n, 3600.0, 3600.0, (0,)) for n in ("d01", "d02")}
    col = ExtractCollector(req, run_start=_start(frame), plans=plans, input_dir=tmp_path)
    try:
        assert col.prepare_subset(("T2",), "d02") == ("T2",)
        assert col.prepare_subset(None, "d02") is None
        assert "CLDFRA" in col.prepare_subset(("T2",), "d01")
    finally:
        col.close()
    import inspect
    from gpuwrf.integration import nested_pipeline as p
    assert "extract.prepare_subset(self._variable_subset, name)" in inspect.getsource(
        p._PerDomainWrfoutWriter._materialize_and_submit)
