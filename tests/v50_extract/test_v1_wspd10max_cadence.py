"""ALISIOS follow-up (manager 01:06Z): UST/U10/V10 every frame, WSPD10MAX when produced (nwp_diagnostics or AFWA
severe, lane v50-gust: 'max since last output'), never a fallback; per-nest output cadence in the manifest."""
from __future__ import annotations

import json

import numpy as np

from gpuwrf.extract import v1_fields as V
from gpuwrf.extract.collector import DomainPlan, output_cadence

from .test_fix1_findings import _one_frame
from .test_v1_extract import raw_of, swiss_frames


def _with_wspd(raw):
    raw["WSPD10MAX"] = (np.asarray(raw["U10"], np.float32) ** 2 + 7.25).astype(np.float32)


def test_wspd10max_present_is_extracted_without_nwp_diagnostics(tmp_path):
    """AFWA-style producer (nwp_diagnostics=0): present -> OK, bytes as written, sampling disclosed."""
    raw = raw_of(swiss_frames(1)[0]); _with_wspd(raw)
    report, manifest = _one_frame(tmp_path, _with_wspd, nwp_diagnostics=0)
    d = manifest["domains"]["d01"]
    assert report["status"] == "READY", d.get("problems")
    st = d["field_status"]["WSPD10MAX"]
    assert st["status"] == "OK" and "reset window" in st["sampling"] and "NOT always since the previous" in st["sampling"]
    import netCDF4
    with netCDF4.Dataset(tmp_path / "ex" / "extract_swiss_m000_d01.nc") as ds:
        ds.set_auto_maskandscale(False)
        assert np.asarray(ds["WSPD10MAX"][0]).tobytes() == raw["WSPD10MAX"].tobytes()
        assert "not an observed gust" in ds["WSPD10MAX"].getncattr("definition")
    for f in ("UST", "U10", "V10"):  # always, every frame
        assert d["field_status"][f]["status"] == "OK" and d["field_stats"][f]["nonfinite"] == 0


def test_wspd10max_absent_is_not_available_and_ready(tmp_path):
    report, manifest = _one_frame(tmp_path, nwp_diagnostics=1)  # requested but this history lacks it
    d = manifest["domains"]["d01"]
    assert report["status"] == "READY" and d["field_status"]["WSPD10MAX"]["status"] == "NOT_AVAILABLE"
    assert d["field_stats"]["WSPD10MAX"]["nonfinite"] == d["shape_yx"][0] * d["shape_yx"][1]


def test_wspd10max_payload_without_producer_is_never_reported(tmp_path):
    frame_raw = raw_of(swiss_frames(1)[0])
    diag = {n: frame_raw[n] for n in V.SURFACE}
    _, manifest = _one_frame(tmp_path, _with_wspd, name="noprod", sources=(diag, None, None))
    assert manifest["domains"]["d01"]["field_status"]["WSPD10MAX"]["status"] == "NOT_AVAILABLE"
    _, manifest = _one_frame(tmp_path, _with_wspd, name="prod",
                             sources=(dict(diag, WSPD10MAX=np.zeros(1)), None, None))
    assert manifest["domains"]["d01"]["field_status"]["WSPD10MAX"]["status"] == "OK"


def test_manifest_documents_output_cadence_per_nest(tmp_path):
    _, manifest = _one_frame(tmp_path)
    oc = manifest["domains"]["d01"]["output_cadence"]
    assert oc["history_interval_s"] == 3600.0 and oc["dt_s"] == 3600.0 and oc["frame_own_steps"] == [0]
    assert set(oc["sampling"]) >= {"WSPD10MAX", "RAINC", "RAINNC", "other_fields"}
    d02 = output_cadence(DomainPlan("d02", 18.0, 3600.0, (0, 200, 400), nwp_diagnostics=0, afwa_severe_opt=1))
    assert d02["own_steps_per_interval"] == "200" and d02["wspd10max_producers"]["afwa_severe_opt"] == 1
    d01 = output_cadence(DomainPlan("d01", 54.0, 3600.0, (0, 67, 134)))
    assert d01["own_steps_per_interval"] == "200/3"  # alarms at ceil(i*200/3): actual times drift 18 s per hour
    json.dumps(d01)
