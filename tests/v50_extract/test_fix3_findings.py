"""Critic Delta 2 (FIX3) regression tests: each fails on 52d62e578a and passes after the fix.

F5   forecast-mode offline schedule requires every nest that has a history interval (and the CLI takes the nest
     set from the case namelist, never from the files found).
F7   a completed run's extension has one successful route: the same request resumes into a chained segment
     <extract_dir>/segment_NN; published parts stay untouched; non-READY/other-policy parts are refused.
WSPD the WSPD10MAX window is documented per producer and frame (AFWA: since the previous output; nwp_diagnostics:
     WRF's trunc(interval/dt) reset), never claimed as 'since the previous output' when it is not.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import gpuwrf.extract.collector as C
import gpuwrf.extract.offline as O
from gpuwrf.extract import v1_fields as V
from gpuwrf.extract.request import load_request, require_resume_policy

from .test_fix1_findings import _Prepared, _State, _collector, _request, _start
from .test_v1_extract import _dims, raw_of, swiss_frames

ZNT = lambda d, s, p: np.full((42, 42), 0.05, np.float32)  # noqa: E731


# ----------------------------------------------------------------------------------------------- F5
def test_f5_forecast_mode_requires_every_nest_with_an_interval(tmp_path):
    frames = swiss_frames(3)
    with pytest.raises(ValueError, match="d02"):  # critic case: d01 files, d01+d02 intervals, forecast 7200 s
        O.extract_wrfouts(_request(tmp_path, "f5"), {"d01": frames}, interval_s={"d01": 3600.0, "d02": 3600.0},
                          forecast_s=7200.0, sst_source="file", znt_source=ZNT)
    assert not (tmp_path / "f5" / "manifest.json").exists()


def test_f5_cli_takes_the_nest_set_from_the_namelist(tmp_path):
    frames = swiss_frames(3)
    case = tmp_path / "case"; case.mkdir()
    (case / "namelist.input").write_text("&time_control\n history_interval = 60, 60,\n/\n&domains\n max_dom = 2,\n/\n")
    wrf = tmp_path / "wrf"; wrf.mkdir()
    for f in frames:
        (wrf / f.name).symlink_to(f)
    cfg = tmp_path / "req.json"
    cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "swiss", "member": "m000",
                               "issue_utc": "2023-01-14T18:00:00Z"}))
    base = ["--config", str(cfg), "--wrfout-dir", str(wrf), "--hours", "2", "--sst-source", "file"]
    with pytest.raises(ValueError, match="d02"):  # namelist schedules d02; only d01 files exist
        O.main([*base, "--extract-dir", str(tmp_path / "ex1"), "--input-dir", str(case)])
    assert not (tmp_path / "ex1" / "manifest.json").exists()
    with pytest.raises(SystemExit):  # no namelist and no explicit nest set: the files may not define it
        O.main([*base, "--extract-dir", str(tmp_path / "ex2")])


# ----------------------------------------------------------------------------------------------- F7
def _snapshot(d):
    return sorted((p.relative_to(d).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in d.rglob("*") if p.is_file() and "segment_" not in p.as_posix())


def test_f7_completed_run_extension_publishes_a_chained_segment(tmp_path):
    frames = swiss_frames(3)
    raws = [raw_of(f) for f in frames]
    request, start = _request(tmp_path), _start(frames[0])
    base = tmp_path / "ex"

    def run(expected, steps, resumed):
        col = _collector(tmp_path, request, start, expected=expected, resumed=resumed)
        for s in steps:
            col.submit("d01", s, _Prepared(raws[s], _dims(raws[s])), _State(*raws[s]["HGT"].shape))
        return col.finalize()

    assert run((0,), (0,), False)["status"] == "READY"
    published = _snapshot(base)
    base_sha = hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest()
    gen = tmp_path / "gen"; gen.mkdir()
    (gen / "manifest.json").write_text(json.dumps({"identity": {"output_policy": {"extract": request.policy()}}}))
    require_resume_policy(gen, request)  # the same request is admitted (checkpoint policy unchanged)

    rep = run((0, 1), (0, 1), True)  # step 0 = history replay at the resume step: not restaged
    assert rep["status"] == "READY", rep
    seg = json.loads((base / "segment_01" / "manifest.json").read_text())
    assert seg["status"] == "READY" and seg["policy"] == request.policy()
    assert seg["segment"]["index"] == 1 and seg["segment"]["parents"] == [{"dir": ".", "manifest_sha256": base_sha}]
    assert seg["segment"]["replayed_published_frames_dropped"] == ["d01:0"]
    assert seg["domains"]["d01"]["output_cadence"]["frame_own_steps"] == [1]
    assert _snapshot(base) == published  # published files untouched

    require_resume_policy(gen, request)
    assert run((0, 1, 2), (2,), True)["status"] == "READY"
    seg2 = json.loads((base / "segment_02" / "manifest.json").read_text())
    assert [p["dir"] for p in seg2["segment"]["parents"]] == [".", "segment_01"]
    assert seg2["domains"]["d01"]["output_cadence"]["frame_own_steps"] == [2]
    assert _snapshot(base) == published

    with pytest.raises(ValueError, match="adds no history frame"):  # nothing new: refused before any write
        _collector(tmp_path, request, start, expected=(0, 1, 2), resumed=True)
    assert not (base / "segment_03").exists()
    cfg = tmp_path / "other.json"  # same extract dir, different request bytes -> other policy
    cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "other", "member": "m000",
                               "issue_utc": "2023-01-14T18:00:00Z"}))
    with pytest.raises(ValueError, match="policy"):
        C.published_chain(load_request("v1", cfg, base))


# ----------------------------------------------------------------------------------------------- WSPD10MAX window
def test_wspd10max_window_per_producer_and_frame():
    plan = lambda **kw: C.DomainPlan("d01", dt_s=54.0, interval_s=3600.0, expected_steps=(0, 67, 134), **kw)  # noqa
    nwp, afwa, none = plan(nwp_diagnostics=1), plan(afwa_severe_opt=1), plan()
    # nwp_diagnostics: idump = trunc(3600/54) = 66, resets at steps 1, 67, 133 (critic's rule calculation)
    assert C.wspd10max_window(nwp, 67) == (67, 67) and C.wspd10max_window(nwp, 134) == (133, 134)
    # AFWA: reset on the step after each history write -> since the previous output
    assert C.wspd10max_window(afwa, 67) == (1, 67) and C.wspd10max_window(afwa, 134) == (68, 134)
    assert C.wspd10max_window(none, 67) is None and C.wspd10max_window(nwp, 0) is None
    cad = C.output_cadence(nwp)["wspd10max"]
    assert cad["producer"] == "nwp_diagnostics" and cad["frame_window_own_steps"] == {
        "0": None, "67": [67, 67], "134": [133, 134]} and "NOT since the previous output" in cad["window_rule"]
    assert C.output_cadence(afwa)["wspd10max"]["frame_window_own_steps"]["134"] == [68, 134]
    assert "NOT always since the previous" in V.SAMPLING["WSPD10MAX"]
    assert "since the previous history output" not in V.DEFINITIONS["WSPD10MAX"]
