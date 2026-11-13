"""Critic Delta 3 (FIX4) regression tests: each fails on d26d609f1b and passes after the fix.

1  a d01-only extract in a d01+d02 run extends (NOT_REQUESTED nests do not break the resume).
2  published_chain verifies every parent artifact (bytes + SHA256) and every segment's recorded parent manifest
   hashes: a removed base NetCDF or a changed ancestor manifest refuses the extension before any write.
"""
from __future__ import annotations

import json

import pytest

import gpuwrf.extract.collector as C
from gpuwrf.extract.request import require_resume_policy

from .test_fix1_findings import _Prepared, _State, _request, _start
from .test_v1_extract import _dims, raw_of, swiss_frames


def _run(tmp_path, request, start, raws, expected, steps, resumed, *, nests=("d01",)):
    plans = {n: C.DomainPlan(n, dt_s=3600.0, interval_s=3600.0, expected_steps=tuple(expected)) for n in nests}
    col = C.ExtractCollector(request, run_start=start, plans=plans, input_dir=tmp_path, resumed=resumed)
    for s in steps:
        col.submit("d01", s, _Prepared(raws[s], _dims(raws[s])), _State(*raws[s]["HGT"].shape))
    return col.finalize()


def _gen(tmp_path, request):
    gen = tmp_path / "gen"
    gen.mkdir(exist_ok=True)
    (gen / "manifest.json").write_text(json.dumps({"identity": {"output_policy": {"extract": request.policy()}}}))
    return gen


def test_subset_extract_extends_with_not_requested_nests(tmp_path):
    frames = swiss_frames(2)
    raws = [raw_of(f) for f in frames]
    request, start = _request(tmp_path, domains=["d01"]), _start(frames[0])
    rep = _run(tmp_path, request, start, raws, (0,), (0,), False, nests=("d01", "d02"))
    assert rep["status"] == "READY"
    base = json.loads((tmp_path / "ex" / "manifest.json").read_text())
    assert base["domains"]["d02"] == {"status": "NOT_REQUESTED"}
    require_resume_policy(_gen(tmp_path, request), request)
    rep = _run(tmp_path, request, start, raws, (0, 1), (0, 1), True, nests=("d01", "d02"))
    assert rep["status"] == "READY", rep
    seg = json.loads((tmp_path / "ex" / "segment_01" / "manifest.json").read_text())
    assert seg["domains"]["d01"]["output_cadence"]["frame_own_steps"] == [1]
    assert seg["domains"]["d02"] == {"status": "NOT_REQUESTED"}


def test_chain_refuses_a_removed_base_artifact(tmp_path):
    frames = swiss_frames(2)
    raws = [raw_of(f) for f in frames]
    request, start = _request(tmp_path), _start(frames[0])
    assert _run(tmp_path, request, start, raws, (0,), (0,), False)["status"] == "READY"
    (tmp_path / "ex" / "extract_swiss_m000_d01.nc").unlink()  # critic probe
    with pytest.raises(ValueError, match="missing or changed"):
        require_resume_policy(_gen(tmp_path, request), request)
    with pytest.raises(ValueError, match="missing or changed"):
        _run(tmp_path, request, start, raws, (0, 1), (1,), True)
    assert not (tmp_path / "ex" / "segment_01").exists()


def test_chain_refuses_a_changed_ancestor_manifest(tmp_path):
    frames = swiss_frames(3)
    raws = [raw_of(f) for f in frames]
    request, start = _request(tmp_path), _start(frames[0])
    assert _run(tmp_path, request, start, raws, (0,), (0,), False)["status"] == "READY"
    assert _run(tmp_path, request, start, raws, (0, 1), (1,), True)["status"] == "READY"
    path = tmp_path / "ex" / "manifest.json"  # critic probe: the ancestor manifest changes after segment_01
    doc = json.loads(path.read_text())
    doc["note"] = "rewritten after segment_01 was published"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="ancestor manifest changed"):
        require_resume_policy(_gen(tmp_path, request), request)
    with pytest.raises(ValueError, match="ancestor manifest changed"):
        _run(tmp_path, request, start, raws, (0, 1, 2), (2,), True)
    assert not (tmp_path / "ex" / "segment_02").exists()
