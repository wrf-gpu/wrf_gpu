"""Regression tests for the v50-diag critic FIX round 1 (CRITIC_VERDICT.md findings 1-9).

Each test reproduces the critic's fault probe and fails on the reviewed head 1336a239 (run there by
HANDOFF_FIX1's pre-fix check). Real inputs: Swiss CPU-WRF d01 frames (read-only), mutated in memory.
"""
from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.extract import points as PT
from gpuwrf.extract import v1_fields as V
from gpuwrf.extract.collector import DomainPlan, ExtractCollector
from gpuwrf.extract.request import load_request

from .test_v1_extract import _dims, raw_of, swiss_frames

GRID_HEADER = ",".join(PT.FORMATS["grid_bound"])


@dataclass
class _Prepared:
    fields: dict
    dimensions: dict


@dataclass
class _PreparedWithGrid:
    fields: dict
    dimensions: dict
    grid: object


class _State:
    def __init__(self, ny, nx):
        self.roughness_m = np.full((ny, nx), 0.1, np.float32)


def _start(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        return datetime.strptime(ds.getncattr("SIMULATION_START_DATE"), "%Y-%m-%d_%H:%M:%S").replace(
            tzinfo=timezone.utc)


def _request(tmp_path, name="ex", **extra):
    cfg = tmp_path / f"{name}.json"
    cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "swiss", "member": "m000",
                               "issue_utc": "2023-01-14T18:00:00Z", **extra}))
    return load_request("v1", cfg, tmp_path / name)


def _collector(tmp_path, request, start, *, expected=(0,), resumed=False, **plan):
    p = DomainPlan("d01", dt_s=3600.0, interval_s=3600.0, expected_steps=tuple(expected), **plan)
    return ExtractCollector(request, run_start=start, plans={"d01": p}, input_dir=tmp_path, resumed=resumed)


def _one_frame(tmp_path, mutate=None, *, name="ex", prepared_cls=_Prepared, grid=None, sources=None, **plan):
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    if mutate is not None:
        mutate(raw)
    col = _collector(tmp_path, _request(tmp_path, name), _start(frame), **plan)
    prepared = (_PreparedWithGrid(raw, _dims(raw), grid) if prepared_cls is _PreparedWithGrid
                else _Prepared(raw, _dims(raw)))
    col.submit("d01", 0, prepared, _State(*raw["HGT"].shape), **({"sources": sources} if sources else {}))
    report = col.finalize()
    manifest = json.loads((tmp_path / name / "manifest.json").read_text())
    return report, manifest


def _problems(manifest):
    return " | ".join(manifest["domains"]["d01"].get("problems", []))


# ----------------------------------------------------------------------------------------------- finding 1
def test_f1_resume_keeps_coverage_first_refusal_with_extract_off(tmp_path, monkeypatch):
    import gpuwrf.integration.nested_pipeline as pipeline
    import gpuwrf.io.wrfbdy_coverage as coverage

    def uncovered(*args, **kwargs):
        raise coverage.BoundaryCoverageError("existing boundary coverage refusal")
    monkeypatch.setattr(coverage, "require_wrfbdy_coverage", uncovered)
    config = pipeline.NestedPipelineConfig(
        tmp_path / "inputs", tmp_path / "out", tmp_path / "proof", 25, 3,
        checkpoint_dir=tmp_path / "ck", checkpoint_interval_steps=1,
        resume_checkpoint=tmp_path / "ck" / "absent-generation", extend_end_time=True)
    with pytest.raises(coverage.BoundaryCoverageError, match="existing boundary coverage refusal"):
        pipeline.execute_nested_pipeline(config)


# ----------------------------------------------------------------------------------------------- finding 2
def test_f2_writer_fallbacks_are_not_observed(tmp_path):
    """PBLH/SST without a producer (sources lack them): PBLH -> MISSING (FAIL); SST -> NOT_AVAILABLE, never TSK
    (Delta 1 finding 2 tightened the FIX1 water-TSK rule)."""
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    present = {n: raw[n] for n in V.SURFACE if n not in ("PBLH", "SST")}
    sources = (present, SimpleNamespace(roughness_m=None), None)
    report, manifest = _one_frame(tmp_path, sources=sources)
    status = manifest["domains"]["d01"]["field_status"]
    assert report["status"] == "FAIL" and status["PBLH"]["status"] == "MISSING", _problems(manifest)
    assert status["SST"]["status"] == "NOT_AVAILABLE"
    raw["LANDMASK"] = raw["LANDMASK"].copy()
    raw["LANDMASK"][:5] = 0
    fields, st, qc = V.frame_fields(raw, None, unobserved=("SST",))
    assert st["SST"]["status"] == "NOT_AVAILABLE" and np.isnan(fields["SST"]).all()


def test_f2_producer_lookup_uses_writer_aliases():
    from gpuwrf.extract.collector import unobserved_inputs
    state = SimpleNamespace(pblh=np.zeros(2), ustar=np.zeros(2), t_skin=np.zeros(2), rainc_acc=np.zeros(2))
    diag = {"U10": 1, "V10": 1, "T2": 1, "Q2": 1, "PSFC": 1, "HFX": 1, "LH": 1, "RAINNC": 1, "SWDOWN": 1, "GLW": 1}
    assert unobserved_inputs((diag, state, None)) == ("SST", "WSPD10MAX")
    assert set(unobserved_inputs(({}, SimpleNamespace(), None))) == set(
        ("U10", "V10", "T2", "Q2", "TSK", "PSFC", "PBLH", "UST", "HFX", "LH", "RAINC", "RAINNC", "SWDOWN",
         "GLW", "SST", "WSPD10MAX"))


# ----------------------------------------------------------------------------------------------- finding 3
@pytest.mark.parametrize("probe", ["nan_landmask", "nan_T", "qcloud_minus_001", "scalar_pblh"])
def test_f3_fault_probes_never_ready(tmp_path, probe):
    def mutate(raw):
        if probe == "nan_landmask":
            raw["LANDMASK"] = raw["LANDMASK"].copy(); raw["LANDMASK"][3, 3] = np.nan
        elif probe == "nan_T":
            raw["T"] = raw["T"].copy(); raw["T"][5, 4, 4] = np.nan
        elif probe == "qcloud_minus_001":
            raw["QCLOUD"] = raw["QCLOUD"].copy(); raw["QCLOUD"][2, 6, 6] = -0.01
            raw["QRAIN"] = raw["QRAIN"].copy(); raw["QRAIN"][2, 6, 6] += 0.02  # LWP stays positive
        elif probe == "scalar_pblh":
            raw["PBLH"] = np.float32(500.0)
    report, manifest = _one_frame(tmp_path, mutate)
    assert report["status"] == "FAIL", (probe, _problems(manifest))


def test_f3_infinite_optional_wspd10max_fails(tmp_path):
    def mutate(raw):
        w = np.full(raw["HGT"].shape, 5.0, np.float32); w[1, 1] = np.inf
        raw["WSPD10MAX"] = w
    report, manifest = _one_frame(tmp_path, mutate, nwp_diagnostics=1)
    assert report["status"] == "FAIL" and "WSPD10MAX" in _problems(manifest)


def test_f3_scheme_species_and_qncloud_are_required():
    raw = raw_of(swiss_frames(1)[0])
    raw.pop("QICE")
    _, status, qc = V.frame_fields(raw, None, mp_physics=8)
    assert status["IWP"]["status"] == "MISSING_INPUT" and qc["missing_expected_species"] == ["QICE"]
    raw = raw_of(swiss_frames(1)[0])
    raw.update(P=raw["T"] * 0, PB=raw["T"] * 0 + 1e5, QVAPOR=raw["QCLOUD"],
               U=np.zeros((raw["T"].shape[0], raw["T"].shape[1], raw["T"].shape[2] + 1), np.float32),
               V=np.zeros((raw["T"].shape[0], raw["T"].shape[1] + 1, raw["T"].shape[2]), np.float32))
    jj, ii = np.array([3]), np.array([4])
    _, pst = PT.profile_columns(raw, jj, ii, qncloud_available=True)
    assert pst["QNCLOUD_per_kg"]["status"] == "MISSING"  # mp28 without QNCLOUD is not NOT_AVAILABLE
    raw.pop("QICE")
    _, pst = PT.profile_columns(raw, jj, ii, qncloud_available=False, qice_expected=True)
    assert pst["QICE_kg_kg"]["status"] == "MISSING"


def test_f3_absent_grid_metric_never_uses_the_zero_filled_payload(tmp_path):
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    metrics = SimpleNamespace(dnw=raw["DNW"], c1h=raw["C1H"])  # c2h absent from the grid metrics

    def mutate(r):
        r["C2H"] = np.zeros_like(r["C2H"])  # what the reduced payload carries
    report, manifest = _one_frame(tmp_path, mutate, prepared_cls=_PreparedWithGrid,
                                  grid=SimpleNamespace(metrics=metrics))
    assert report["status"] == "FAIL", _problems(manifest)
    assert manifest["domains"]["d01"]["field_status"]["LWP"]["status"] == "MISSING_INPUT"


# ----------------------------------------------------------------------------------------------- finding 4
def _bound_point(tmp_path, kind, y, x, lat, lon, hgt):
    p = tmp_path / f"B_{kind}_{y}_{x}.csv"
    p.write_text(GRID_HEADER + f"\nd01_X_{y}_{x},{kind},d01,TF,Z,{y},{x},{lat!r},{lon!r},{hgt!r},0,\n")
    return PT.load_point_csvs([p])[0][0]


def test_f4_frozen_point_checked_before_hull_and_on_hgt_landmask(tmp_path):
    raw = raw_of(swiss_frames(1)[0])
    st = {k: raw[k].astype(np.float32).copy() for k in ("XLAT", "XLONG", "HGT", "LANDMASK")}
    st["LANDMASK"][5, 7] = 0
    la, lo, h = float(st["XLAT"][5, 7]), float(st["XLONG"][5, 7]), float(st["HGT"][5, 7])
    sea = _bound_point(tmp_path, "sea_grid", 5, 7, la, lo, h)
    assert PT.map_points([sea], "d01", st, 1e9)[0].status == "OK"
    shifted = dict(st, XLAT=st["XLAT"] + np.float32(2.0))  # footprint moved clear of the frozen cell
    m = PT.map_points([sea], "d01", shifted, 1e9)[0]
    assert m.status == "GEOMETRY_MISMATCH", m.status  # not OUTSIDE_DOMAIN
    for change in ({"HGT": st["HGT"] + np.float32(1.0)}, {"LANDMASK": np.ones_like(st["LANDMASK"])}):
        m = PT.map_points([sea], "d01", dict(st, **change), 1e9)[0]
        assert m.status == "GEOMETRY_MISMATCH", (list(change), m.status)


# ----------------------------------------------------------------------------------------------- finding 5
def _offline(tmp_path, files, name, **kw):
    from gpuwrf.extract.offline import extract_wrfouts
    req = _request(tmp_path, name)
    znt = lambda d, step, path: np.full((42, 42), 0.05, np.float32)  # noqa: E731
    return extract_wrfouts(req, {"d01": files}, interval_s={"d01": 3600.0}, znt_source=znt, sst_source="file", **kw)


def test_f5_offline_schedule_times_and_duplicates(tmp_path):
    frames = swiss_frames(3)
    assert _offline(tmp_path, frames[:1], "prefix", forecast_s=7200.0)["status"] == "PARTIAL"
    assert _offline(tmp_path, [frames[0], frames[2]], "gap", forecast_s=7200.0)["status"] == "PARTIAL"
    assert _offline(tmp_path, frames, "full", forecast_s=7200.0)["status"] == "READY"
    links = tmp_path / "links"; links.mkdir()
    wrong = links / frames[2].name  # file name of hour 2, content of hour 1
    wrong.symlink_to(frames[1])
    with pytest.raises(ValueError, match="Times"):
        _offline(tmp_path, [frames[0], frames[1], wrong], "renamed", forecast_s=7200.0)
    dup = links / frames[1].name.replace(":", "_")  # same instant, colon-free spelling
    dup.symlink_to(frames[1])
    with pytest.raises(ValueError, match="same history step"):
        _offline(tmp_path, [frames[0], frames[1], dup, frames[2]], "dup", forecast_s=7200.0)


def test_f5_duplicate_submission_in_one_process_fails(tmp_path):
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    col = _collector(tmp_path, _request(tmp_path), _start(frame))
    for _ in range(2):
        col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))
    assert col.finalize()["status"] == "FAIL"


# ----------------------------------------------------------------------------------------------- finding 6
def _profiles_nc(path, actual, nominal):
    import netCDF4
    from gpuwrf.extract import legacy
    with netCDF4.Dataset(path, "w") as ds:
        ds.setncattr("issue_utc", "2023-01-14T18:00:00+00:00")
        ds.setncattr("wrf_start_utc", "2023-01-15T00:00:00+00:00")
        for d, n in (("time", len(actual)), ("point", 1), ("level", 2)):
            ds.createDimension(d, n)
        ds.createVariable("actual_time_unix_s", "i8", ("time",))[:] = actual
        ds.createVariable("nominal_time_unix_s", "i8", ("time",))[:] = nominal
        for k in ("point_id", "point_kind"):
            ds.createVariable(k, str, ("point",))[0] = "p1"
        for k in ("grid_y", "grid_x"):
            ds.createVariable(k, "i4", ("point",))[:] = 1
        for f in legacy.WT1_FIELDS:
            v = ds.createVariable(f, "f4", ("time", "point", "level"))
            v[:] = 1.0
            v.field_status = json.dumps({"status": "OK"})


def test_f6_wt1_refuses_duplicate_or_off_axis_leads(tmp_path):
    from gpuwrf.extract import legacy
    start = int(datetime(2023, 1, 15, tzinfo=timezone.utc).timestamp())
    ok = tmp_path / "ok.nc"
    _profiles_nc(ok, [start, start + 3600 + 18], [start, start + 3600])
    assert [t.column("lead_h")[0].as_py() for t in legacy.wt1_profiles(ok)] == [6, 7]
    dup = tmp_path / "dup.nc"
    _profiles_nc(dup, [start, start + 54], [start, start])  # the saved PROD probe axis
    with pytest.raises(ValueError, match="not unique"):
        legacy.wt1_profiles(dup)
    off = tmp_path / "off.nc"
    _profiles_nc(off, [start, start + 3600], [start, start + 7200])
    with pytest.raises(ValueError, match="nominal"):
        legacy.wt1_profiles(off)


# ----------------------------------------------------------------------------------------------- finding 7
def test_f7_checkpoint_drain_waits_for_extract_staging():
    from gpuwrf.integration import nested_pipeline as p
    order = []
    worker = SimpleNamespace(_queue=SimpleNamespace(join=lambda: order.append("writer")),
                             _raise_if_failed=lambda: None)
    writer = SimpleNamespace(_extract=SimpleNamespace(join=lambda: order.append("extract")))
    p._drain_outputs_for_checkpoint(worker, None, writer)
    assert order == ["writer", "extract"]
    # ... and the checkpoint block uses it before saving the restart generation.
    src = inspect.getsource(p._execute_nested_pipeline)
    assert src.index("_drain_outputs_for_checkpoint(output_pipeline, async_writer, writer)") < src.index(
        "restart_store.save(")


def test_f7_same_policy_extension_resumes_but_shortening_is_refused(tmp_path):
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    request, start = _request(tmp_path), _start(frame)
    col = _collector(tmp_path, request, start, expected=(0, 1))
    col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))
    col.join(); col.close()
    _collector(tmp_path, request, start, expected=(0, 1, 2), resumed=True).close()  # extension
    with pytest.raises(ValueError, match="extend"):
        _collector(tmp_path, request, start, expected=(0,), resumed=True)


# ----------------------------------------------------------------------------------------------- finding 8
def test_f8_failed_cleanup_never_publishes_ready(tmp_path):
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    col = _collector(tmp_path, _request(tmp_path), _start(frame))
    col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))
    col.join()
    (tmp_path / "ex/.staging/d01/foreign.txt").write_text("x")  # rmdir of the staging dir fails
    report = col.finalize()
    manifest = json.loads((tmp_path / "ex/manifest.json").read_text())
    assert report["status"] == manifest["status"] == "FAIL", manifest.get("finalize_problems")


def test_f8_publication_after_stream_close_and_no_submit_after_close(tmp_path):
    from gpuwrf.integration import nested_pipeline as p
    src = inspect.getsource(p._execute_nested_pipeline)
    assert src.index("regional_streams.close()") < src.index("writer._extract.finalize(")
    assert src.index('context="finally"') < src.index("writer._extract.close()")  # after the writer joins
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    col = _collector(tmp_path, _request(tmp_path), _start(frame))
    col.close()
    with pytest.raises(RuntimeError, match="closed"):
        col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape))


# ----------------------------------------------------------------------------------------------- finding 9
def test_f9_raw_off_skips_unrequested_nests_before_any_fetch(tmp_path, monkeypatch):
    from gpuwrf.integration import nested_pipeline as p
    frame = swiss_frames(1)[0]
    req = _request(tmp_path, raw_history="off", domains=["d01"])
    plans = {n: DomainPlan(n, 3600.0, 3600.0, (0,)) for n in ("d01", "d02")}
    col = ExtractCollector(req, run_start=_start(frame), plans=plans, input_dir=tmp_path)
    try:
        monkeypatch.setattr(p, "assert_state_finite_at_boundary", lambda *a, **k: None)
        monkeypatch.setattr(p, "prepare_wrfout_payload",
                            lambda *a, **k: pytest.fail("RAW-off fetch for an unrequested nest"))
        writer = SimpleNamespace(_extract=col, _variable_subset=None, bundles={})
        assert p._PerDomainWrfoutWriter._materialize_and_submit(
            writer, name="d02", own_step=0, carry=SimpleNamespace(state=None), path=tmp_path / "x",
            lead_seconds=0.0, lead_hours=0.0, valid_time=None) is None
    finally:
        col.close()


def test_f9_queue_holds_inputs_only_and_readback_opens_each_frame_once(tmp_path, monkeypatch):
    import gpuwrf.extract.collector as C
    frames = swiss_frames(3)
    col = _collector(tmp_path, _request(tmp_path), _start(frames[0]), expected=(0, 1, 2))
    seen = []
    orig = col._process
    monkeypatch.setattr(col, "_process", lambda *job: (seen.append(set(job[2])), orig(*job)))
    for step, path in enumerate(frames):
        raw = raw_of(path)
        payload = dict(raw, EXTRA_HISTORY_FIELD=np.zeros((50, 42, 42), np.float32))
        col.submit("d01", step, _Prepared(payload, _dims(raw)), _State(*raw["HGT"].shape))
    col.join()
    assert seen and all("EXTRA_HISTORY_FIELD" not in s for s in seen)
    opened = []
    real_load = np.load
    monkeypatch.setattr(C.np, "load", lambda *a, **k: (opened.append(a[0]), real_load(*a, **k))[1])
    report = col.finalize()
    assert report["status"] == "READY"
    assert len(opened) <= 3 * len(frames) + 1, len(opened)  # metas + write + readback, static once
    cost = json.loads((tmp_path / "ex/manifest.json").read_text())["host_cost"]
    assert cost["queued_input_bytes_max"] > 0 and cost["rss_delta_kib_per_frame"]["n"] == len(frames)
