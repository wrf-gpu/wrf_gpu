"""scripts/compare_branch_streams.py verdicts on synthetic wrfout streams (CPU, structural)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("compare_streams", ROOT / "scripts" / "compare_branch_streams.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)
TIMES = ("2026-02-28_00:09:00", "2026-02-28_00:18:00", "2026-02-28_00:27:00")


def _write(path, t2, hgt=1.0, title="gpuwrf"):
    with netCDF4.Dataset(path, "w") as ds:
        ds.TITLE = title
        ds.createDimension("y", 2)
        ds.createDimension("x", 3)
        ds.createVariable("T2", "f4", ("y", "x"))[:] = t2
        ds.createVariable("HGT", "f4", ("y", "x"))[:] = hgt


def _stream(root, *, delta=None, hgt=None, times=TIMES, title="gpuwrf"):
    root.mkdir(parents=True)
    for t in times:
        for d in ("d01", "d02"):
            t2 = np.full((2, 3), 280.0, np.float32)
            if delta and d in delta and t > TIMES[0]:
                t2 += delta[d]
            _write(root / f"wrfout_{d}_{t}", t2, hgt=(hgt or {}).get(d, 1.0), title=title)
    return root


def _run(tmp_path, ref, test, mode, *extra):
    out = tmp_path / f"v{len(list(tmp_path.glob('v*.json')))}.json"
    rc = cmp.main(["--ref", str(ref), "--test", str(test), "--out", str(out), "--mode", mode,
                   "--after", "2026-02-28T00:09:00", *extra])
    return rc, json.loads(out.read_text())


def test_exact_pass_and_completeness(tmp_path):
    ref = _stream(tmp_path / "ref")
    test = _stream(tmp_path / "test", times=TIMES[1:])
    rc, v = _run(tmp_path, ref, test, "exact")
    assert rc == 0 and v["verdict"] == "PASS" and len(v["frames"]) == 4
    partial = _stream(tmp_path / "partial", times=TIMES[1:2])
    rc, v = _run(tmp_path, ref, partial, "exact")
    assert rc == 1 and any("missing in test" in f for f in v["failures"])
    early = _stream(tmp_path / "early")  # contains the branch-time frame
    rc, v = _run(tmp_path, ref, early, "exact")
    assert rc == 1 and any("at/before" in f for f in v["failures"])


def test_exact_fails_on_values_and_on_attributes(tmp_path):
    ref = _stream(tmp_path / "ref")
    rc, v = _run(tmp_path, ref, _stream(tmp_path / "val", delta={"d02": 1e-3}, times=TIMES[1:]), "exact")
    assert rc == 1 and "T2" in v["frames"]["wrfout_d02_2026-02-28_00:18:00"]["differing"]
    rc, v = _run(tmp_path, ref, _stream(tmp_path / "attr", title="other", times=TIMES[1:]), "exact")
    assert rc == 1 and v["frames"]["wrfout_d01_2026-02-28_00:18:00"]["attrs_differing"] == ["TITLE"]


def test_edited_mode(tmp_path):
    ref = _stream(tmp_path / "ref")
    good = _stream(tmp_path / "good", delta={"d02": 0.5}, times=TIMES[1:])
    assert _run(tmp_path, ref, good, "edited", "--expect-differ", "d02")[0] == 0
    rc, v = _run(tmp_path, ref, good, "edited", "--expect-differ", "d01")  # d02 moved, d01 did not
    assert rc == 1 and any("must be identical" in f for f in v["failures"])
    assert any("expected a response in d01" in f for f in v["failures"])
    geo = _stream(tmp_path / "geo", delta={"d02": 0.5}, hgt={"d02": 2.0}, times=TIMES[1:])
    rc, v = _run(tmp_path, ref, geo, "edited", "--expect-differ", "d02")
    assert rc == 1 and any("geography" in f for f in v["failures"])
    nan = _stream(tmp_path / "nan", delta={"d02": np.nan}, times=TIMES[1:])
    rc, v = _run(tmp_path, ref, nan, "edited", "--expect-differ", "d02")
    assert rc == 1 and any("non-finite" in f for f in v["failures"])


def test_digest_references_give_the_same_verdicts(tmp_path):
    ref = _stream(tmp_path / "ref")
    assert cmp.main(["digest", str(ref), str(tmp_path / "ref.json")]) == 0
    same = _stream(tmp_path / "same", times=TIMES[1:])
    assert _run(tmp_path, tmp_path / "ref.json", same, "exact")[0] == 0
    assert cmp.main(["digest", str(same), str(tmp_path / "same.json")]) == 0
    assert _run(tmp_path, tmp_path / "ref.json", tmp_path / "same.json", "exact")[0] == 0
    rc, v = _run(tmp_path, tmp_path / "ref.json", _stream(tmp_path / "val", delta={"d02": 1e-3}, times=TIMES[1:]), "exact")
    assert rc == 1 and "T2" in v["frames"]["wrfout_d02_2026-02-28_00:18:00"]["differing"]
    edited = _stream(tmp_path / "ed", delta={"d02": 0.5}, hgt={"d02": 2.0}, times=TIMES[1:])
    rc, v = _run(tmp_path, tmp_path / "ref.json", edited, "edited", "--expect-differ", "d02")
    assert rc == 1 and any("geography" in f for f in v["failures"])
    assert cmp.main(["digest", str(tmp_path / "empty_missing"), str(tmp_path / "x.json")]) == 1


def test_whole_stream_compares_pre_branch_frames_of_a_resumed_stream(tmp_path):
    ref = _stream(tmp_path / "ref")
    resumed = _stream(tmp_path / "resumed")  # parent part + continuation in one stream
    assert _run(tmp_path, ref, resumed, "exact", "--whole-stream")[0] == 0
    bad = _stream(tmp_path / "bad", delta={"d01": 1.0})
    (bad / "wrfout_d01_2026-02-28_00:09:00").unlink()
    _write(bad / "wrfout_d01_2026-02-28_00:09:00", np.full((2, 3), 281.0, np.float32))  # pre-branch frame differs
    rc, v = _run(tmp_path, ref, bad, "exact", "--whole-stream")
    assert rc == 1 and "T2" in v["frames"]["wrfout_d01_2026-02-28_00:09:00"]["differing"]
    assert _run(tmp_path, ref, resumed, "edited", "--whole-stream", "--expect-differ", "d01")[0] == 1


# ---- v50-branch FIX1 (critic 2026-11-11): completeness, independent inventory, seam from the generation -------
def test_edited_mode_requires_every_ref_frame_after_the_seam(tmp_path):
    ref = _stream(tmp_path / "ref")
    test = _stream(tmp_path / "test", delta={"d02": 0.5}, times=TIMES[1:])
    for name in sorted(p.name for p in test.iterdir())[1:]:  # keep 1 of 4 post-seam frames
        (test / name).unlink()
    rc, v = _run(tmp_path, ref, test, "edited", "--expect-differ", "d02")
    assert rc == 1 and any("missing in test" in f for f in v["failures"])


def test_whole_stream_requires_the_pre_seam_frames(tmp_path):
    ref = _stream(tmp_path / "ref")
    resumed = _stream(tmp_path / "resumed")
    for d in ("d01", "d02"):
        (resumed / f"wrfout_{d}_{TIMES[0]}").unlink()  # the seam frames of the first segment are gone
    rc, v = _run(tmp_path, ref, resumed, "exact", "--whole-stream")
    assert rc == 1 and any("whole stream missing in test" in f for f in v["failures"])


def test_expected_inventory_is_exact(tmp_path):
    ref = _stream(tmp_path / "ref")
    test = _stream(tmp_path / "test", times=TIMES[1:])
    inv = tmp_path / "inv.json"
    names = [f"wrfout_{d}_{t}" for d in ("d01", "d02") for t in TIMES[1:]]
    inv.write_text(json.dumps({"frames": names}))
    assert _run(tmp_path, ref, test, "exact", "--expect-frames", str(inv))[0] == 0
    inv.write_text(json.dumps({"frames": names + ["wrfout_d03_2026-02-28_00:18:00"]}))  # a nest the run never wrote
    rc, v = _run(tmp_path, ref, test, "exact", "--expect-frames", str(inv))
    assert rc == 1 and any("expected frames missing in test" in f for f in v["failures"])
    inv.write_text(json.dumps({"frames": names[1:]}))
    rc, v = _run(tmp_path, ref, test, "exact", "--expect-frames", str(inv))
    assert rc == 1 and any("not in the expected inventory" in f for f in v["failures"])


def _generation(tmp_path, steps=200, dt=54.0, start="2026-02-28T00:00:00+00:00"):
    gen = tmp_path / "gen"
    gen.mkdir(parents=True)
    (gen / "manifest.json").write_text(json.dumps({"domains": ["d01", "d02"], "own_steps": {"d01": steps, "d02": 3 * steps},
                                                   "dt_s": {"d01": dt, "d02": dt / 3}, "identity": {"run_start": start}}))
    return gen


def test_seam_comes_from_the_generation_not_the_start(tmp_path):
    assert cmp.seam_time(_generation(tmp_path)) == "2026-02-28_03:00:00"
    ref = _stream(tmp_path / "ref")
    test = _stream(tmp_path / "test", times=TIMES[1:])
    out = tmp_path / "v.json"
    gen = _generation(tmp_path / "g", steps=10, dt=54.0)  # seam 00:09:00
    assert cmp.main(["--ref", str(ref), "--test", str(test), "--out", str(out), "--mode", "exact",
                     "--branch-generation", str(gen)]) == 0
    assert json.loads(out.read_text())["after"] == "2026-02-28_00:09:00"


def test_inventory_follows_the_first_own_step_after_each_history_time(tmp_path):
    nl = tmp_path / "namelist.input"
    nl.write_text(" &time_control\n start_year = 2026, 2026,\n start_month = 2, 2,\n start_day = 28, 28,\n"
                  " start_hour = 0, 0,\n history_interval = 60, 60,\n /\n &domains\n time_step = 54,\n max_dom = 2,\n"
                  " parent_id = 1, 1,\n parent_time_step_ratio = 1, 3,\n /\n")
    names = cmp.inventory(nl, 3, "2026-02-28_01:00:18", False, None)
    assert names == ["wrfout_d01_2026-02-28_02:00:36", "wrfout_d01_2026-02-28_03:00:00",
                     "wrfout_d02_2026-02-28_02:00:00", "wrfout_d02_2026-02-28_03:00:00"]
    assert "wrfout_d01_2026-02-28_00:00:00" in cmp.inventory(nl, 1, None, True, None)


def test_stray_wrfout_files_fail_and_are_reported(tmp_path):
    ref = _stream(tmp_path / "ref")
    test = _stream(tmp_path / "test", times=TIMES[1:])
    names = [f"wrfout_{d}_{t}" for d in ("d01", "d02") for t in TIMES[1:]]
    inv = tmp_path / "inv.json"
    inv.write_text(json.dumps({"frames": names}))
    (test / "wrfout_d01_2026-02-28_00:18:00.bak").write_bytes(b"x")
    rc, v = _run(tmp_path, ref, test, "exact", "--expect-frames", str(inv))
    assert rc == 1 and v["stray"]["test"] == ["wrfout_d01_2026-02-28_00:18:00.bak"]
    assert cmp.main(["digest", str(test), str(tmp_path / "t.json")]) == 0
    assert json.loads((tmp_path / "t.json").read_text())["stray"] == ["wrfout_d01_2026-02-28_00:18:00.bak"]
    rc, v = _run(tmp_path, ref, tmp_path / "t.json", "exact", "--expect-frames", str(inv))
    assert rc == 1 and any("not well-formed" in f for f in v["failures"])
