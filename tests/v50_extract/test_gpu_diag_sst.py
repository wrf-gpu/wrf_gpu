"""GPU arm 11.11 gate5 (in-run extract == offline re-extract): only SST differed, in all 9 nests, and only in its
``field_status`` attribute; the values were byte-identical (oracle_grid: in-run SST == wrfout SST bytes; both ==
wrflowinp record 0). The in-run path recorded HOW it reached the value ("history diagnostics: SST ...") and the
offline path a different HOW ("wrfout SST variable ..."). Both now record WHAT the value is (V.SST_SOURCE).

Fails on 195aa5db82 (field_status differs), passes after the fix. Also covers sst_update=1 at all: the local
real-run parity suite uses PROD (sst_update=0 -> SST NOT_AVAILABLE on both sides) and never reached this path.
"""
from __future__ import annotations

import json

import numpy as np

import gpuwrf.extract.collector as C
from gpuwrf.extract import v1_fields as V
from gpuwrf.extract.offline import extract_wrfouts

from .test_fix1_findings import _Prepared, _State, _request, _start
from .test_v1_extract import _dims, raw_of, swiss_frames


def _sst(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        v = ds["SST"]
        return np.asarray(v[:]).copy(), {k: str(v.getncattr(k)) for k in v.ncattrs()}


def test_in_run_and_offline_sst_are_identical_with_an_sst_producer(tmp_path):
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    assert "SST" in raw
    # In-run: the payload's SST comes from the history diagnostics (the model's held lower-boundary record).
    req_in = _request(tmp_path, "inrun")
    plan = C.DomainPlan("d01", dt_s=3600.0, interval_s=3600.0, expected_steps=(0,))
    col = C.ExtractCollector(req_in, run_start=_start(frame), plans={"d01": plan}, input_dir=tmp_path)
    col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape),
               sources=(raw, None, None))
    assert col.finalize()["domains"]["d01"] == "READY"
    # Offline: the same frame's wrfout SST, declared sst_source=file (sst_update=1 run).
    req_off = _request(tmp_path, "offline")
    rep = extract_wrfouts(req_off, {"d01": [frame]}, interval_s={"d01": 3600.0}, expected_steps={"d01": (0,)},
                          sst_source="file", znt_source=lambda d, s, p: np.full(raw["HGT"].shape[-2:], 0.1, np.float32))
    assert rep["domains"]["d01"] == "READY"
    a_val, a_att = _sst(tmp_path / "inrun" / req_in.file_name("d01"))
    b_val, b_att = _sst(tmp_path / "offline" / req_off.file_name("d01"))
    assert a_val.dtype == b_val.dtype and a_val.tobytes() == b_val.tobytes()
    assert a_att == b_att, (a_att, b_att)
    assert json.loads(a_att["field_status"]) == {"status": "OK", "source": V.SST_SOURCE}
    np.testing.assert_array_equal(a_val[0], np.asarray(raw["SST"], np.float32).reshape(a_val[0].shape))


def test_without_a_producer_sst_stays_not_available_on_both_paths(tmp_path):
    frame = swiss_frames(1)[0]
    raw = raw_of(frame)
    req_in = _request(tmp_path, "inrun")
    plan = C.DomainPlan("d01", dt_s=3600.0, interval_s=3600.0, expected_steps=(0,))
    col = C.ExtractCollector(req_in, run_start=_start(frame), plans={"d01": plan}, input_dir=tmp_path)
    col.submit("d01", 0, _Prepared(raw, _dims(raw)), _State(*raw["HGT"].shape), sources=({k: v for k, v in raw.items() if k != "SST"}, None, None))
    col.finalize()
    req_off = _request(tmp_path, "offline")
    extract_wrfouts(req_off, {"d01": [frame]}, interval_s={"d01": 3600.0}, expected_steps={"d01": (0,)},
                    sst_source="none", znt_source=lambda d, s, p: np.full(raw["HGT"].shape[-2:], 0.1, np.float32))
    a_val, a_att = _sst(tmp_path / "inrun" / req_in.file_name("d01"))
    b_val, b_att = _sst(tmp_path / "offline" / req_off.file_name("d01"))
    assert np.isnan(a_val).all() and np.isnan(b_val).all()
    assert a_att == b_att and json.loads(a_att["field_status"])["status"] == "NOT_AVAILABLE"
