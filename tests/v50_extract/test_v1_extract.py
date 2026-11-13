"""V1 extract: field parity (cloud9/WT1/TM-S6 oracles), status edge cases, collector/NetCDF/manifest.

Real frames: Swiss CPU-WRF d01 history (pristine WRF output, Thompson+MYNN), read-only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.extract import v1_fields as V
from gpuwrf.extract.collector import ExtractCollector, DomainPlan, frame_times
from gpuwrf.extract.request import load_request, require_resume_policy

from . import oracle

SWISS = Path(os.environ.get("V50_SWISS_CASE", "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))


def swiss_frames(n=3):
    frames = sorted(SWISS.glob("wrfout_d01_*"))[:n]
    if len(frames) < n or not oracle.available():
        pytest.skip("Swiss CPU-WRF frames or pinned ALISIOS oracles unavailable")
    return frames


def raw_of(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        return {n: np.asarray(ds[n][0]).copy() for n in V.RAW_NAMES if n in ds.variables}


def assert_same(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype, a.shape, b.shape)
    assert a.tobytes() == b.tobytes(), (name, np.nanmax(np.abs(a.astype("f8") - b.astype("f8"))))


def test_frame_fields_bitwise_against_independent_oracles():
    counts = {"cloud_base_ok": 0, "inversion_ok": 0, "cloud_low_ok": 0}
    for path in swiss_frames(3):
        raw = raw_of(path)
        znt = np.full(raw["HGT"].shape, 0.05, np.float32)
        fields, status, qc = V.frame_fields(raw, znt)
        ref = oracle.recompute(path)
        for name in ("cloud_max", "cloud_low_max", "LWP", "IWP", "u10_east", "v10_north", "wind_speed",
                     "wind_direction_from_deg", "cloud_base_msl", "cloud_base_agl", "cloud_base_status",
                     "inv_base_msl", "inv_top_msl", "inv_strength_K", "inv_gradient_K_m", "inv_status",
                     *V.SURFACE):
            assert_same(fields[name], ref[name], name)
        assert status["WSPD10MAX"]["status"] == "NOT_AVAILABLE"
        assert np.isnan(fields["WSPD10MAX"]).all()
        assert all(status[n]["status"] == "OK" for n in V.SURFACE)
        assert qc["z_w_monotone"] and qc["mass_positive"]
        counts["cloud_base_ok"] += qc["cloud_base_status_counts"]["OK"]
        counts["inversion_ok"] += qc["inv_status_counts"]["OK"]
        counts["cloud_low_ok"] += int(np.count_nonzero(fields["cloud_low_max"] > 0))
    # The frames have clouds and inversions: the comparison is not vacuous.
    assert all(v > 0 for v in counts.values()), counts


def test_cloud9_used_directly_is_bitwise_equal():
    import netCDF4
    c9 = oracle.cloud9()
    for path in swiss_frames(2):
        with netCDF4.Dataset(path) as ds:
            ref = c9.frame_cloud(ds)
        fields, _, _ = V.frame_fields(raw_of(path), None)
        for k in c9.FIELDS:
            assert_same(fields[k], ref[k], k)


def _column(theta, z):
    """Exact single-column inputs: T perturbation (nz,1,1) and z_msl (nz,1,1), f32-representable."""
    return ((np.asarray(theta, np.float32) - np.float32(300))[:, None, None],
            np.asarray(z, np.float32)[:, None, None])


def test_inversion_status_edges_and_tie_breaks():
    z = [100.0, 400.0, 800.0, 1200.0, 1600.0, 2000.0, 4000.0]
    cases = {
        "OK": [290, 291, 291.5, 299.5, 300, 300.5, 310],     # max gk between 800-1200 (8 K / 400 m)
        "NO_INVERSION": [290, 290.25, 290.5, 290.75, 291, 291.25, 299],
        "TIE": [290, 291, 295, 299, 299.25, 299.5, 330],      # 400-800 and 800-1200 tie exactly -> lowest k
    }
    for label, theta in cases.items():
        t, zmsl = _column(theta, z)
        out = V.inversion_v1(t, zmsl)
        expect = oracle.alisios_pins.inversion_base(np.asarray(theta, np.float32), zmsl[:, 0, 0])
        if label == "NO_INVERSION":
            assert out["inv_status"][0, 0] == 1 and np.isnan(out["inv_base_msl"][0, 0]) and np.isnan(expect)
        else:
            assert out["inv_status"][0, 0] == 0
            assert float(out["inv_base_msl"][0, 0]) == expect
        if label == "OK":
            assert out["inv_base_msl"][0, 0] == 800 and out["inv_top_msl"][0, 0] == 1200
            assert out["inv_strength_K"][0, 0] == 8 and out["inv_gradient_K_m"][0, 0] == np.float32(0.02)
        if label == "TIE":
            assert out["inv_base_msl"][0, 0] == 400 and out["inv_top_msl"][0, 0] == 800
    # The 2000-4000 m layer ends above 3500 m and is never selected (310 K jump ignored).
    # Terrain above the window: no layer with both ends in [300, 3500] m MSL.
    t, zmsl = _column([300, 301, 302, 303], [3600.0, 3700.0, 3800.0, 3900.0])
    out = V.inversion_v1(t, zmsl)
    assert out["inv_status"][0, 0] == 3 and np.isnan(out["inv_gradient_K_m"][0, 0])
    # Incomplete profile is MISSING_INPUT, not NO_INVERSION.
    t = t.copy(); t[1, 0, 0] = np.nan
    assert V.inversion_v1(t, zmsl)["inv_status"][0, 0] == 2


def test_cloud_base_threshold_and_clear():
    z = np.array([10.0, 200.0, 600.0])[:, None, None].astype(np.float32)
    qc = np.array([0.0, 6e-6, 2e-5], np.float32)[:, None, None]
    qi = np.array([0.0, 4e-6, 0.0], np.float32)[:, None, None]
    msl, agl, st = V.cloud_base(qc, qi, z, z - 5)
    assert st[0, 0] == 0 and msl[0, 0] == np.float32(200.0) and agl[0, 0] == np.float32(195.0)
    msl, agl, st = V.cloud_base(qc * 0, qi * 0, z, z)
    assert st[0, 0] == 1 and np.isnan(msl[0, 0]) and np.isnan(agl[0, 0])


def test_calm_direction_missing_and_rotation():
    u = np.array([[0.0, 3.0]], np.float32); v = np.array([[0.0, -4.0]], np.float32)
    c = np.array([[1.0, 0.0]], np.float32); s = np.array([[0.0, 1.0]], np.float32)
    ue, vn, spd, wdir = V.rotate_wind(u, v, c, s)
    assert np.isnan(wdir[0, 0]) and spd[0, 0] == 0
    # grid -> earth with cos=0, sin=1: ue = -v, vn = u
    assert ue[0, 1] == np.float32(4.0) and vn[0, 1] == np.float32(3.0) and spd[0, 1] == np.float32(5.0)


def test_missing_mandatory_and_species_are_marked_not_substituted():
    path = swiss_frames(1)[0]
    raw = raw_of(path)
    for n in ("PBLH", "QGRAUP", "QSNOW", "QICE"):
        raw.pop(n)
    fields, status, qc = V.frame_fields(raw, None)
    assert status["PBLH"]["status"] == "MISSING" and np.isnan(fields["PBLH"]).all()
    assert status["ZNT"]["status"] == "MISSING"
    assert status["IWP"]["status"] == "NOT_AVAILABLE" and np.isnan(fields["IWP"]).all()
    assert status["cloud_base_msl"]["status"] == "MISSING_INPUT" and (fields["cloud_base_status"] == 2).all()
    assert status["WSPD10MAX"]["status"] == "NOT_AVAILABLE"  # not in this history
    raw = raw_of(path); raw.pop("QGRAUP")
    _, status, _ = V.frame_fields(raw, None)
    assert status["IWP"]["status"] == "PARTIAL_SPECIES" and status["IWP"]["species_used"] == ["QICE", "QSNOW"]


# --------------------------------------------------------------------------- collector
@dataclass
class _Prepared:
    fields: dict
    dimensions: dict


class _State:
    def __init__(self, ny, nx):
        self.roughness_m = np.full((ny, nx), 0.1, np.float32)


def _request(tmp_path, name="ex", **extra):
    cfg = tmp_path / f"{name}.json"
    doc = {"schema": "gpuwrf.extract.request.v1", "run_id": "swiss", "member": "m000",
           "issue_utc": "2023-01-14T18:00:00Z", **extra}
    cfg.write_text(json.dumps(doc))
    return load_request("v1", cfg, tmp_path / name)


def _collector(tmp_path, request, frames, *, expected=None, resumed=False):
    plan = DomainPlan("d01", dt_s=3600.0, interval_s=3600.0,
                      expected_steps=tuple(range(len(frames))) if expected is None else expected)
    start = datetime(2023, 1, 15, tzinfo=timezone.utc)
    return ExtractCollector(request, run_start=start, plans={"d01": plan}, input_dir=SWISS, resumed=resumed)


def _dims(raw):
    nz, ny, nx = raw["CLDFRA"].shape
    return {"west_east": nx, "south_north": ny, "bottom_top": nz, "bottom_top_stag": nz + 1,
            "west_east_stag": nx + 1, "south_north_stag": ny + 1}


def test_collector_netcdf_manifest_and_parity(tmp_path):
    import netCDF4
    frames = swiss_frames(3)
    col = _collector(tmp_path, _request(tmp_path), frames)
    raws = [raw_of(p) for p in frames]
    for step, raw in enumerate(raws):
        col.submit("d01", step, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))
    report = col.finalize(engine={"test": True})
    out = tmp_path / "ex"
    manifest = json.loads((out / "manifest.json").read_text())
    assert report["status"] == manifest["status"] == "READY", manifest["domains"]["d01"].get("problems")
    assert not (out / ".staging").exists()
    nc = out / "extract_swiss_m000_d01.nc"
    art = manifest["artifacts"][nc.name]
    assert art["bytes"] == nc.stat().st_size and art["readback"] == "PASS"
    with netCDF4.Dataset(nc) as ds:
        ds.set_auto_maskandscale(False)
        assert ds["actual_time_unix_s"][:].dtype == np.int64
        assert list(ds["tau_h"][:]) == [0.0, 1.0, 2.0] and list(ds["lead_h"][:]) == [6.0, 7.0, 8.0]
        assert ds["cloud_max"].chunking() == [1, *ds["cloud_max"].shape[1:]]
        assert ds["cloud_max"].filters()["complevel"] == 4
        for i, path in enumerate(frames):
            ref = oracle.recompute(path)
            assert int(ds["actual_time_unix_s"][i]) == ref["actual_time_unix_s"]
            for name in ("cloud_max", "cloud_low_max", "LWP", "IWP", "wind_speed", "wind_direction_from_deg",
                         "inv_base_msl", "inv_status", "cloud_base_agl", "T2", "PBLH"):
                assert_same(ds[name][i], ref[name], name)
            assert (np.asarray(ds["ZNT"][i]) == np.float32(0.1)).all()
        for name in V.STATIC:
            assert_same(ds[name][:], raws[0][name], name)
    d = manifest["domains"]["d01"]
    assert d["field_status"]["WSPD10MAX"]["status"] == "NOT_AVAILABLE"
    assert d["time"]["time_offset_seconds_max_abs"] == 0
    assert manifest["inputs_sha256"]["wrfinput_d01"]


def test_collector_partial_and_refusals(tmp_path):
    frames = swiss_frames(2)
    request = _request(tmp_path)
    col = _collector(tmp_path, request, frames, expected=(0, 1, 2))
    for step, path in enumerate(frames):
        raw = raw_of(path)
        col.submit("d01", step, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))
    report = col.finalize()
    assert report["status"] == "PARTIAL"
    manifest = json.loads((tmp_path / "ex/manifest.json").read_text())
    assert manifest["domains"]["d01"]["missing_steps"] == [2]
    assert (tmp_path / "ex/.staging").exists()  # kept for diagnosis when not READY
    with pytest.raises(FileExistsError):  # fresh dir per attempt
        _collector(tmp_path, request, frames)
    # Missing lead-0 frame is refused at bind.
    with pytest.raises(ValueError, match="lead-0"):
        _collector(tmp_path, _request(tmp_path, "nolead0"), frames, expected=(1, 2))


def test_mandatory_missing_fails(tmp_path):
    frames = swiss_frames(1)
    col = _collector(tmp_path, _request(tmp_path), frames)
    raw = raw_of(frames[0]); raw.pop("PBLH")
    col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))
    report = col.finalize()
    assert report["status"] == "FAIL"
    problems = json.loads((tmp_path / "ex/manifest.json").read_text())["domains"]["d01"]["problems"]
    assert any("PBLH" in p for p in problems)


def test_staging_resume_binding(tmp_path):
    frames = swiss_frames(2)
    request = _request(tmp_path)
    col = _collector(tmp_path, request, frames)
    raw = raw_of(frames[0])
    col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))
    col.join(); col.close()
    with pytest.raises(FileExistsError):  # a non-resume attempt cannot reuse staging
        _collector(tmp_path, request, frames)
    col = _collector(tmp_path, request, frames, resumed=True)
    raw1 = raw_of(frames[1])
    col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))  # identical replay
    col.submit("d01", 1, _Prepared(raw1, _dims(raw1)), _State(*raw1["HGT"].shape))
    assert col.finalize()["status"] == "READY"
    # A resumed run with a different request is refused at the binding.
    other = _request(tmp_path, "ex", arm="B")
    with pytest.raises((ValueError, FileExistsError)):
        _collector(tmp_path, other, frames, resumed=True)


def test_request_validation_and_resume_policy(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "a/b", "member": "m",
                               "issue_utc": "20230114_18z"}))
    with pytest.raises(ValueError):
        load_request("v1", bad, tmp_path / "x")
    bad.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "a", "member": "m",
                               "issue_utc": "20230114_18z", "points_csv": str(tmp_path / "absent.csv")}))
    with pytest.raises(FileNotFoundError):  # M2: point CSVs are read and SHA-bound at load time
        load_request("v1", bad, tmp_path / "x")
    with pytest.raises(ValueError):
        load_request("v2", bad, tmp_path / "x")
    request = _request(tmp_path)
    assert request.issue_utc == datetime(2023, 1, 14, 18, tzinfo=timezone.utc)
    gen = tmp_path / "gen"; gen.mkdir()
    (gen / "manifest.json").write_text(json.dumps({"identity": {"output_policy": {"history": {}}}}))
    require_resume_policy(gen, None)  # legacy OFF checkpoint, OFF resume
    with pytest.raises(ValueError, match="differs"):
        require_resume_policy(gen, request)
    (gen / "manifest.json").write_text(json.dumps({"identity": {"output_policy": {"extract": request.policy()}}}))
    require_resume_policy(gen, request)
    with pytest.raises(ValueError, match="differs"):
        require_resume_policy(gen, None)


def test_time_axis_d01_offset():
    plan = DomainPlan("d01", dt_s=54.0, interval_s=3600.0, expected_steps=(0, 67))
    start = datetime(2026, 7, 25, 18, tzinfo=timezone.utc)
    t = frame_times(plan, 67, start, start - timedelta(hours=6))
    assert t["time_offset_seconds"] == 18 and t["tau_h"] == 1.0 and t["lead_h"] == 7.0
    assert t["actual_time_unix_s"] == int(start.timestamp()) + 3618


def test_pins_match_alisios_sources():
    """The copied oracle functions equal the ALISIOS sources (when readable outside the data guard)."""
    import inspect
    import subprocess
    from . import alisios_pins as pins
    repo = Path("<USER_HOME>/src/alisios")
    if not (repo / ".git").exists():
        pytest.skip("ALISIOS repo unavailable")
    blob = subprocess.run(["git", "-C", str(repo), "show", f"{pins.RERETRO_COMMIT}:{pins.RERETRO_PATH}"],
                          capture_output=True, text=True)
    if blob.returncode:
        pytest.skip("pinned ALISIOS commit unavailable")
    lo, hi = pins.RERETRO_LINES
    assert "\n".join(blob.stdout.splitlines()[lo - 1:hi]).rstrip() == inspect.getsource(pins.inversion_base).rstrip()
