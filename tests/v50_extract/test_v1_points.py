"""V1 extract M2: points (frozen CSV formats, nearest cell, statuses), native profiles, legacy adapters, RAW off.

Real frames: Swiss CPU-WRF d01 history (pristine WRF output). Oracles: WT1 teacher code (verbatim pins in
wt1_pins.py, verified by verify_pins.py) and the unmodified ALISIOS extract9.py/cloud9.py (read-only, SHA-pinned).
Point rows here are synthetic rows in the frozen ALISIOS CSV formats; the real lists are never invented.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.extract import points as PT
from gpuwrf.extract import legacy
from gpuwrf.extract.request import load_request

from . import oracle
from .test_v1_extract import swiss_frames, assert_same

EXTRACT9 = oracle.ALISIOS / "ops/nightly_9nest/extract9.py"
EXTRACT9_SHA = "3c9443b8b797749d1436ed348db0ec06f18110110dd73b15e310e1360bc071e5"
STATIONS_HEADER = ",".join(PT.FORMATS["stations"])
GRID_HEADER = ",".join(PT.FORMATS["grid_bound"])


def _static(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        return {k: np.asarray(ds[k][0]).copy() for k in ("XLAT", "XLONG", "HGT", "LANDMASK")}


def _issue_and_start(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        start = datetime.strptime(ds.getncattr("SIMULATION_START_DATE"), "%Y-%m-%d_%H:%M:%S").replace(
            tzinfo=timezone.utc)
    return start - timedelta(hours=6), start


def _write_points(tmp_path, st):
    la, lo, hgt = st["XLAT"], st["XLONG"], st["HGT"]
    rows = [f"st:center,{float(la[20, 20])!r},{float(lo[20, 20])!r},1500.0,net,,station,OBS1,OBS1",
            f"st:offcell,{float(la[10, 30]) + 0.004},{float(lo[10, 30]) - 0.006},,net,,spot,unavailable,spots",
            "st:nocoord,,,,net,,station,OBS1,OBS1",
            "wmo:far,28.32,-16.38,,sonde,Guimar,sounding_support,unavailable,pin",
            f"st:beyondedge,{float(la[0, 21]) - 0.05},{float(lo[0, 21])},12.0,net,,station,OBS1,OBS1",
            f"st:onedge,{float(la[0, 21])!r},{float(lo[0, 21])!r},12.0,net,,station,OBS1,OBS1"]
    # Swiss d01 is all land: the frozen-cell fixture uses a land kind (sea kinds need LANDMASK 0).
    grid = [f"d01_NE_Meer_005_007,land_grid,d01,TF,NE_Meer,5,7,{float(la[5, 7])!r},{float(lo[5, 7])!r},"
            f"{float(hgt[5, 7])!r},0,",
            f"d01_Nordhang_001,transect,d01,TF,Nordhang,30,12,{float(la[30, 12])!r},{float(lo[30, 12])!r},"
            f"{float(hgt[30, 12])!r},812.5,1",
            f"d01_moved,sea_grid,d01,TF,NE_Meer,6,7,{float(la[5, 7])!r},{float(lo[5, 7])!r},0.0,,",
            f"d03_TF_only,transect,d03,TF,Nordhang,3,4,{float(la[12, 12])!r},{float(lo[12, 12])!r},400.0,401.0,2"]
    a = tmp_path / "POINTS_TEST.csv"
    a.write_text(STATIONS_HEADER + "\n" + "\n".join(rows) + "\n")
    b = tmp_path / "SEA_TRANSECTS_TEST.csv"
    b.write_text(GRID_HEADER + "\n" + "\n".join(grid) + "\n")
    return [a, b]


def _request(tmp_path, issue, points, name="req.json", **extra):
    cfg = tmp_path / name
    cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "swiss", "member": "m000",
                               "issue_utc": issue.strftime("%Y%m%d_%Hz"),
                               "points_csv": [str(p) for p in points], **extra}))
    return cfg


def test_point_csv_formats_and_refusals(tmp_path):
    good = tmp_path / "a.csv"
    good.write_text(STATIONS_HEADER + "\nx:1,28.1,-16.2,10,n,,station,s,c\nx:2,,,,n,,spot,s,c\n")
    pts, src = PT.load_point_csvs([good])
    assert [p.height_status for p in pts] == ["OK", "HEIGHT_UNKNOWN"] and pts[1].lat is None
    assert src[0].format == "stations" and src[0].sha256 == hashlib.sha256(good.read_bytes()).hexdigest()
    cases = {
        "header": "id,lat,lon\nx,1,2\n",
        "dup": STATIONS_HEADER + "\nx,1,2,,n,,s,s,c\nx,1,2,,n,,s,s,c\n",
        "half": STATIONS_HEADER + "\nx,1,,,n,,s,s,c\n",
        "nan": STATIONS_HEADER + "\nx,nan,2,,n,,s,s,c\n",
        "cols": STATIONS_HEADER + "\nx,1,2\n",
    }
    for key, text in cases.items():
        bad = tmp_path / f"{key}.csv"
        bad.write_text(text)
        with pytest.raises(ValueError):
            PT.load_point_csvs([bad])
    other = tmp_path / "b.csv"
    other.write_text(STATIONS_HEADER + "\nx:1,1,2,,n,,s,s,c\n")
    with pytest.raises(ValueError, match="duplicate point id"):
        PT.load_point_csvs([good, other])


def test_nearest_cell_and_distance_equal_wt1_mapped_points():
    from . import wt1_pins
    static = _static(swiss_frames(1)[0])
    la, lo = static["XLAT"], static["XLONG"]
    rng = np.random.default_rng(20261110)
    lats = rng.uniform(float(la.min()) - 0.3, float(la.max()) + 0.3, 400)
    lons = rng.uniform(float(lo.min()) - 0.3, float(lo.max()) + 0.3, 400)
    pts = [PT.Point(f"r{i}", "t", i, "station", float(a), float(b), None, "HEIGHT_UNKNOWN")
           for i, (a, b) in enumerate(zip(lats, lons))]
    limit = 4500.0
    ours = PT.map_points(pts, "d01", static, limit)
    doc = {"points": [{"phys_id": p.id, "lat": p.lat, "lon": p.lon, "role": "station"} for p in pts]}
    wt1_pins.base.bound_json = lambda path, sha: (doc, None)
    cfg = {"point_set_path": "x", "point_set_sha256": "x", "max_points": 10_000, "profile_max_distance_m": limit}
    ref = wt1_pins.mapped_points(cfg, static)
    statuses = {}
    for m, r in zip(ours, ref):
        assert (m.y, m.x) == (r["grid_y"], r["grid_x"]), m.point.id
        assert m.distance_m == r["distance_m"], (m.point.id, m.distance_m, r["distance_m"])
        statuses[m.status] = statuses.get(m.status, 0) + 1
        if m.status == "OK":
            assert r["profile_status"] == "AVAILABLE"
        elif m.status == "BEYOND_MAX_DISTANCE":
            assert r["profile_status"] != "AVAILABLE"
        else:  # outside the hull: never extrapolated, whatever the WT1 distance rule says
            assert m.status == "OUTSIDE_DOMAIN"
            assert not (0 <= m.frac_y <= la.shape[0] - 1 and 0 <= m.frac_x <= la.shape[1] - 1)
    assert statuses.get("OK", 0) > 100 and statuses.get("OUTSIDE_DOMAIN", 0) > 50, statuses


def test_point_statuses_hull_and_bound_cells(tmp_path):
    frame = swiss_frames(1)[0]
    static = _static(frame)
    pts, _ = PT.load_point_csvs(_write_points(tmp_path, static))
    by = {m.point.id: m for m in PT.map_points(pts, "d01", static, 4500.0)}
    assert by["st:center"].status == "OK" and (by["st:center"].y, by["st:center"].x) == (20, 20)
    assert by["st:center"].row()["height_minus_model_HGT_m"] == pytest.approx(1500.0 - float(static["HGT"][20, 20]))
    assert by["st:offcell"].status == "OK" and by["st:offcell"].point.height_status == "HEIGHT_UNKNOWN"
    assert by["st:nocoord"].status == "NO_COORDINATE" and by["st:nocoord"].y is None
    assert by["wmo:far"].status == "OUTSIDE_DOMAIN"
    assert by["st:beyondedge"].status == "OUTSIDE_DOMAIN" and by["st:beyondedge"].y == 0
    assert by["st:onedge"].status == "OK" and by["st:onedge"].y == 0
    assert by["d01_NE_Meer_005_007"].status == "OK" and (by["d01_NE_Meer_005_007"].y, by["d01_NE_Meer_005_007"].x) == (5, 7)
    assert by["d01_Nordhang_001"].status == "OK"
    assert by["d01_moved"].status == "GEOMETRY_MISMATCH"  # frozen (6,7) but its lat/lon is cell (5,7)
    assert by["d03_TF_only"].status == "OK"  # bound to d03; in d01 it is an ordinary nearest-cell point
    assert by["d03_TF_only"].point.height_status == "NOT_A_STATION"
    tight = {m.point.id: m.status for m in PT.map_points(pts, "d01", static, 100.0)}
    assert tight["st:offcell"] == "BEYOND_MAX_DISTANCE" and tight["st:center"] == "OK"


def _offline(tmp_path, frames, points, **extra):
    from gpuwrf.extract.offline import extract_wrfouts
    issue, _ = _issue_and_start(frames[0])
    cfg = _request(tmp_path, issue, points, **extra)
    request = load_request("v1", cfg, tmp_path / "ex")
    znt = lambda name, step, path: np.full(_static(path)["HGT"].shape, 0.05, np.float32)  # noqa: E731
    report = extract_wrfouts(request, {"d01": frames}, interval_s={"d01": 3600.0}, znt_source=znt,
                             forecast_s=3600.0 * (len(frames) - 1), sst_source="file")
    return report, tmp_path / "ex"


def test_offline_profiles_and_wt1_adapter_parity(tmp_path):
    import netCDF4
    from . import wt1_pins
    frames = swiss_frames(3)
    static = _static(frames[0])
    points = _write_points(tmp_path, static)
    report, ex = _offline(tmp_path, frames, points)
    manifest = json.loads((ex / "manifest.json").read_text())
    assert report["status"] == "PARTIAL", manifest["domains"]["d01"].get("problems")  # d01_moved mismatch
    assert manifest["domains"]["d01"]["partial_reasons"] == ["1 grid-bound points do not match this run's geometry"]
    assert manifest["points"]["NO_COORDINATE"] == ["st:nocoord"]
    assert manifest["points"]["HEIGHT_UNKNOWN"] == ["st:offcell", "st:nocoord", "wmo:far"]
    assert manifest["points"]["GEOMETRY_MISMATCH"] == ["d01_moved"]
    assert [s["sha256"] for s in manifest["points"]["sources"]] == [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in points]
    assert manifest["artifacts"]["points.csv"]["sha256"] == hashlib.sha256((ex / "points.csv").read_bytes()).hexdigest()
    prof = ex / "profiles_swiss_m000_d01.nc"
    assert manifest["artifacts"][prof.name]["sha256"] == hashlib.sha256(prof.read_bytes()).hexdigest()
    issue, _ = _issue_and_start(frames[0])
    issue_s = issue.strftime("%Y%m%d_%Hz")
    with netCDF4.Dataset(prof) as ds:
        ds.set_auto_maskandscale(False)
        ids = list(ds["point_id"][:])
        assert ids == ["st:center", "st:offcell", "st:onedge", "d01_NE_Meer_005_007", "d01_Nordhang_001",
                       "d03_TF_only"]
        kinds = dict(zip(ids, ds["point_kind"][:]))
        limit = float(ds.getncattr("max_distance_m"))
        assert limit == 1.5 * 3000.0
        qn = json.loads(ds["QNCLOUD_per_kg"].getncattr("field_status"))
        assert qn["status"] == "NOT_AVAILABLE" and np.isnan(ds["QNCLOUD_per_kg"][:]).all()
        gy, gx = np.asarray(ds["grid_y"][:]), np.asarray(ds["grid_x"][:])
        with netCDF4.Dataset(ex / "extract_swiss_m000_d01.nc") as grid:
            grid.set_auto_maskandscale(False)
            for f in ("T2", "wind_speed", "cloud_max", "LWP", "inv_status", "cloud_base_status", "ZNT"):
                assert np.asarray(ds[f][:]).tobytes() == np.asarray(grid[f][:])[:, gy, gx].tobytes(), f
        # Fields beyond WT1: independent recomputation from the raw frame.
        for t, path in enumerate(frames):
            with netCDF4.Dataset(path) as src:
                src.set_auto_maskandscale(False)
                u, v = np.asarray(src["U"][0]), np.asarray(src["V"][0])
                qi = np.asarray(src["QICE"][0])
                for p, (y, x) in enumerate(zip(gy, gx)):
                    ug = ((u[:, y, x].astype("f8") + u[:, y, x + 1].astype("f8")) / 2).astype("f4")
                    vg = ((v[:, y, x].astype("f8") + v[:, y + 1, x].astype("f8")) / 2).astype("f4")
                    assert_same(ds["U_grid_ms"][t, p], ug, "U_grid")
                    assert_same(ds["V_grid_ms"][t, p], vg, "V_grid")
                    assert_same(ds["QICE_kg_kg"][t, p], qi[:, y, x], "QICE")
    # WT1 adapter == pinned teacher profile_table on the same frames and cells.
    tables = legacy.wt1_profiles(prof, attempt=3)
    assert legacy.wt1_schema().equals(wt1_pins.PROFILE_SCHEMA, check_metadata=False)
    doc = {"points": [{"phys_id": i, "lat": None, "lon": None, "role": kinds[i]} for i in ids]}
    for t, path in enumerate(frames):
        with netCDF4.Dataset(path) as src:
            prof_ref = wt1_pins.frame_profiles(src)
            act = oracle.frame_time_unix(path)
            act_dt = datetime.fromtimestamp(act, tz=timezone.utc)
            nominal, _ = wt1_pins.time_coordinates(act_dt, issue_s)
            pts = [{**p, "grid_y": int(y), "grid_x": int(x), "profile_status": "AVAILABLE"}
                   for p, y, x in zip(doc["points"], gy, gx)]
            ref = wt1_pins.profile_table(src, prof_ref, pts, issue_s, 3, act_dt, nominal)
        assert tables[t].equals(ref), t
        assert tables[t].num_rows == len(ids) * 44


def _load_extract9(nframes):
    assert hashlib.sha256(EXTRACT9.read_bytes()).hexdigest() == EXTRACT9_SHA
    oracle.cloud9()  # pinned cloud9 into sys.modules for extract9's `import cloud9`
    sys.modules.pop("extract9", None)
    previous = os.environ.get("NFRAMES")
    os.environ["NFRAMES"] = str(nframes)
    spec = importlib.util.spec_from_file_location("extract9", EXTRACT9)
    module = importlib.util.module_from_spec(spec)
    dont = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont
        if previous is None:
            os.environ.pop("NFRAMES")
        else:
            os.environ["NFRAMES"] = previous
    return module


def test_legacy_npz_adapters_equal_unmodified_extract9_and_cloud9(tmp_path):
    frames = swiss_frames(4)
    from gpuwrf.extract.offline import extract_wrfouts
    issue, _ = _issue_and_start(frames[0])
    cfg = tmp_path / "req.json"
    cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "swiss", "member": "m000",
                               "issue_utc": issue.strftime("%Y%m%d_%Hz")}))
    znt = lambda name, step, path: np.full(_static(path)["HGT"].shape, 0.05, np.float32)  # noqa: E731
    rep = extract_wrfouts(load_request("v1", cfg, tmp_path / "ex"), {"d01": frames},
                          interval_s={"d01": 3600.0}, znt_source=znt, forecast_s=3600.0 * (len(frames) - 1),
                          sst_source="file")
    assert rep["status"] == "READY", rep
    nc = tmp_path / "ex" / "extract_swiss_m000_d01.nc"
    # Unmodified extract9.main over a 9-domain directory whose every domain is these Swiss frames.
    wdir = tmp_path / "wrfout9"
    wdir.mkdir()
    for d in range(1, 10):
        for f in frames:
            (wdir / f.name.replace("_d01_", f"_d{d:02d}_")).symlink_to(f)
    out = tmp_path / "legacy"
    _load_extract9(len(frames)).main(issue.strftime("%Y%m%d_%Hz"), wdir, out)
    ours, info = legacy.sfc_npz(nc)
    assert info["omitted_extras_not_OK"] == []
    with np.load(out / "d01_sfc.npz") as ref:
        assert list(ref.files) == list(ours)
        for k in ref.files:
            assert_same(ours[k], ref[k], k)
    ours_c, _ = legacy.cloud_npz(nc)
    with np.load(out / "d03_cloud.npz") as ref:
        assert list(ref.files) == list(ours_c)
        for k in ref.files:
            assert_same(ours_c[k], ref[k], k)
    legacy.write_npz(ours, tmp_path / "d01_sfc.npz")
    with pytest.raises(FileExistsError):
        legacy.write_npz(ours, tmp_path / "d01_sfc.npz")


def test_raw_off_request_policy_fetch_set_and_writer_skip(tmp_path):
    from gpuwrf.extract.collector import ExtractCollector, DomainPlan
    from gpuwrf.extract import v1_fields as V
    from gpuwrf.integration import nested_pipeline as p
    frame = swiss_frames(1)[0]
    issue, start = _issue_and_start(frame)
    points = _write_points(tmp_path, _static(frame))
    m1 = load_request("v1", _request(tmp_path, issue, [], name="m1.json"), tmp_path / "a")
    assert set(m1.policy()) == {"spec", "config_sha256", "extract_dir"}  # M1 policy bytes unchanged
    req = load_request("v1", _request(tmp_path, issue, points, raw_history="off", max_distance_m={"d01": 9000}),
                       tmp_path / "b")
    assert req.policy()["raw_history"] == "off" and len(req.policy()["points_sha256"]) == 2
    plan = DomainPlan("d01", 18.0, 3600.0, (0, 200), dx_m=3000.0)
    col = ExtractCollector(req, run_start=start, input_dir=tmp_path, plans={"d01": plan})
    try:
        assert col.max_distance_m == {"d01": 9000.0}
        fetch = col.prepare_subset(None)
        assert fetch == col.prepare_subset(("T2",)) == col.reduced_fetch_names
        assert (set(V.RAW_NAMES) | set(PT.PROFILE_RAW)) - set(V.COLUMN) == set(fetch) and "T2" in fetch
        calls = []
        writer = SimpleNamespace(dt_by_domain={"d01": 18.0}, run_start=start, output_dir=tmp_path / "out",
                                 written={"d01": []}, _output_pipeline=None, _extract=col,
                                 _full_variable_set=True,
                                 _materialize_and_submit=lambda **kw: calls.append(kw))
        result = p._PerDomainWrfoutWriter.__call__(writer, "d01", 0, object())
        assert result["wrfout"] is None and writer.written == {"d01": []} and len(calls) == 1
    finally:
        col.close()
    on = load_request("v1", _request(tmp_path, issue, [], name="on.json"), tmp_path / "c")
    col = ExtractCollector(on, run_start=start, input_dir=tmp_path, plans={"d01": plan})
    try:
        assert col.prepare_subset(None) is None and col.prepare_subset(("T2",))[0] == "T2"
        writer = SimpleNamespace(dt_by_domain={"d01": 18.0}, run_start=start, output_dir=tmp_path / "out",
                                 written={"d01": []}, _output_pipeline=None, _extract=col,
                                 _full_variable_set=True, _materialize_and_submit=lambda **kw: None)
        result = p._PerDomainWrfoutWriter.__call__(writer, "d01", 0, object())
        assert result["wrfout"] is not None and writer.written["d01"] == [result["wrfout"]]
    finally:
        col.close()


def test_request_points_validation(tmp_path):
    frame = swiss_frames(1)[0]
    issue, _ = _issue_and_start(frame)
    points = _write_points(tmp_path, _static(frame))
    bad = [({"points_csv": ["relative.csv"]}, "absolute"),
           ({"max_distance_m": {"d01": 5}}, "needs points_csv"),
           ({"points_csv": [str(points[0])], "max_distance_m": {"d1": 5}}, "max_distance_m"),
           ({"points_csv": [str(points[0])], "max_distance_m": {"d01": -1}}, "max_distance_m"),
           ({"raw_history": "no"}, "raw_history"),
           ({"points_csv": [str(points[0]), str(points[0])]}, "twice")]
    for i, (extra, msg) in enumerate(bad):
        cfg = tmp_path / f"bad{i}.json"
        cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "r", "member": "m",
                                   "issue_utc": "20230114_18z", **extra}))
        with pytest.raises(ValueError, match=msg):
            load_request("v1", cfg, tmp_path / "x")
    # The bound CSV changing between request and collector is refused (content SHA).
    from gpuwrf.extract.collector import ExtractCollector, DomainPlan
    req = load_request("v1", _request(tmp_path, issue, points), tmp_path / "y")
    points[0].write_text(points[0].read_text() + "x:late,46.5,8.5,,n,,station,s,c\n")
    with pytest.raises(ValueError, match="changed since"):
        ExtractCollector(req, run_start=issue + timedelta(hours=6), input_dir=tmp_path,
                         plans={"d01": DomainPlan("d01", 18.0, 3600.0, (0,), dx_m=3000.0)})


def test_column_metrics_come_from_grid_metrics_like_the_full_writer():
    """In-run DNW/C1H/C2H = GridSpec DycoreMetrics coerced like wrfout_writer._add_grid_coordinate_fields
    (a reduced payload carries zero vectors for them: PROD lead 0 measurement, payload_compare.json)."""
    from gpuwrf.extract.collector import _grid_column_metrics
    nz = 4
    metrics = SimpleNamespace(dnw=np.array([-0.1, -0.2, -0.3, -0.4]), c1h=np.array([1.0, 0.9, 0.5, 0.0]),
                              c2h=np.array([0.0, 100.0, 500.0, 95000.0]))
    got = _grid_column_metrics(SimpleNamespace(metrics=metrics), {"bottom_top": nz})
    assert set(got) == {"DNW", "C1H", "C2H"}
    for name, attr in (("DNW", "dnw"), ("C1H", "c1h"), ("C2H", "c2h")):
        assert got[name].dtype == np.float32 and got[name].tobytes() == getattr(metrics, attr).astype(np.float32).tobytes()
    assert _grid_column_metrics(None, {"bottom_top": nz}) == {}
    assert _grid_column_metrics(SimpleNamespace(metrics=metrics), {"bottom_top": 5}) == {}  # shape guard
