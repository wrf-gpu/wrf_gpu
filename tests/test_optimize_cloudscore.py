"""Hand-calculated cloud metrics and a native-WRF-shaped NetCDF caller."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.optimize import cloud

ROOT = Path(__file__).resolve().parents[1]


def helper(relative):
    spec = importlib.util.spec_from_file_location("cloud_test_helper", ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def facade(tmp_path, **overrides):
    values = {"times": np.array([1767226500, 1767227400, 1767228300], dtype="<i8"),
              "cloud": np.array([[0, 1], [1, 0], [255, 0]], dtype="u1"),
              "lat": np.array([0, 0], dtype="<f4"), "lon": np.array([0, 1], dtype="<f4")}
    values.update(overrides)
    path = tmp_path/"truth.npz"
    np.savez(path, **values)
    return cloud.load_config({"format": cloud.FORMAT, "truth": str(path),
        "daylight": [{"time": f"2026-01-01T00:{m}:00Z", "pixel_ids": "all"} for m in (15, 30)],
        "regions": {"east": [1]}, "superobs": {"correlation_length_km": 10., "contract": "SYNTHETIC_TEST_ONLY"}}, lambda p: p)


def history(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset

    path = tmp_path/"wrfout_d01_test"
    with Dataset(path, "w") as ds:
        for name, length in (("Time", 3), ("DateStrLen", 19), ("bottom_top", 2),
                             ("bottom_top_stag", 3), ("south_north", 1), ("west_east", 2)):
            ds.createDimension(name, length)
        ds.GRID_ID, ds.SIMULATION_START_DATE, ds.DX, ds.DY = 1, "2026-01-01_00:00:00", 1000., 1000.
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array(
            [list(f"2026-01-01_00:{minute:02d}:00") for minute in (15, 30, 45)], dtype="S1")
        for name, value in (("XLAT", [[0, 0]]), ("XLONG", [[0, 1]]), ("MAPFAC_M", [[1, 1]])):
            ds.createVariable(name, "f8", ("Time", "south_north", "west_east"))[:] = value
        dims = ("Time", "bottom_top", "south_north", "west_east")
        for name, value, unit in (("P", 0, "Pa"), ("PB", 100000, "Pa"), ("T", 0, "K"),
                                  ("QVAPOR", 0, "kg kg-1"), ("QCLOUD", 0, "kg kg-1"), ("QICE", 0, "kg kg-1")):
            var = ds.createVariable(name, "f8", dims)
            var.units = unit
            var[:] = value
        # density=100000/(287*300); 2 layers of 100m -> precisely chosen 20g/m².
        q20 = 20/(1000*200*(100000/(287*300)))
        ds.variables["QCLOUD"][0, :, 0, 1] = q20
        ds.variables["QICE"][1, :, 0, 0] = q20
        for name in ("PH", "PHB"):
            var = ds.createVariable(name, "f8", ("Time", "bottom_top_stag", "south_north", "west_east"))
            var.units = "m2 s-2"
            var[:] = np.array([0, 981, 1962])[None, :, None, None] if name == "PHB" else 0
    return path


def test_column_integral_signed_and_species():
    rho, dz = np.ones((2, 1, 2))*2, np.ones((2, 1, 2))*100
    qc = np.array([[[1e-5, -1e-5]], [[2e-5, -2e-5]]])
    qi = np.ones_like(qc)*1e-5
    np.testing.assert_allclose(cloud.cwp({"QCLOUD": qc, "QICE": qi}, rho, dz), [[10, -2]], atol=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["QCLOUD", "QICE", "rho", "dz"])
def test_nonfinite_column_rejected_even_one_cell(bad, field):
    values = {s: np.ones((2, 1, 2)) for s in ("QCLOUD", "QICE", "rho", "dz")}
    values[field][0, 0, 0] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        cloud.cwp({s: values[s] for s in ("QCLOUD", "QICE")}, values["rho"], values["dz"])


def test_footprint_threshold_before_area_aggregation_and_distance_cap():
    op = cloud.Footprints.build([[0, 0, 0, 0]], [[0, .001, 1, 9]], [[1, 3, 1, 1]], [0, 0, 0], [0, 1, 2])
    mean = op.apply(np.array([[20, 0, 12, 9999]]) >= 10)
    np.testing.assert_array_equal(mean, [.25, 1, 0])
    np.testing.assert_array_equal(op.cell_count, [2, 1, 0])
    np.testing.assert_array_equal(op.weight, [4, 1, 0])
    assert op.total_area == 6 and op.max_distance_m < 1000
    with pytest.raises(ValueError, match="occurrence"):
        op.apply([[20, 0, 12, 9999]])


def test_contingency_known_answers():
    result = cloud.contingency(2, 1, 1, 2)
    assert result == {"hits": 2, "false_alarms": 1, "misses": 1, "correct_negatives": 2,
                      "valid": 6, "frequency_bias": 1., "csi": .5, "ets": .2}
    empty = cloud.contingency(0, 0, 0, 10)
    assert empty["frequency_bias"] is None and empty["csi"] is None and empty["ets"] is None


def test_shift_fss_centroid_and_continuous_loss():
    report, pred, obs, indices = cloud.verify([[1, 0]], [[0, 1]], np.array([[True, True]]), [1, 1], [0, 0], [0, 1],
        radii_km=[0, 100, 120])
    region = report["regions"]["all"]
    assert region["fss"] == {"0.0": 0., "100.0": 0., "120.0": 1.}
    assert region["centroid_displacement_km"] == pytest.approx(111.19508023)
    assert report["loss"] == 1
    assert len(indices) == 2 and obs.tolist() == [0, 1]
    assert pred[0] > .99 and pred[1] < .01
    # One cloudy pair with no spatial shift has FSS=1; clear/clear is also perfect.
    same, *_ = cloud.verify([[1, 0]], [[1, 0]], np.array([[True, True]]), [1, 1], [0, 0], [0, 1])
    assert set(same["regions"]["all"]["fss"].values()) == {1.}


def test_invalid_night_uncovered_and_pixel_region_counts():
    values = np.array([[.5, 1, 0, 1], [0, 0, 1, 1]])
    clm = np.array([[1, 255, 0, 0], [1, 1, 0, 0]], dtype="u1")
    report, pred, obs, _ = cloud.verify(values, clm, np.array([[True]*4, [False]*4]), [1, 1, 1, 0],
        [0]*4, [0, 1, 2, 3], regions={"west": np.array([True, False, False, False])})
    assert report["counts"] == {"total": 8, "invalid_clm": 1, "night": 4, "uncovered": 2, "valid": 2}
    assert report["regions"]["all"]["hits"] == 1
    assert report["regions"]["all"]["correct_negatives"] == 1
    assert report["regions"]["west"]["valid"] == 1
    assert report["pixels"]["valid"] == [1, 0, 1, 0]
    assert pred[0] == .5 and obs.tolist() == [1, 0]  # inclusive pixel threshold exactly .5
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_cannot_hide_under_invalid_or_night(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        cloud.verify([[0, bad]], [[1, 255]], np.array([[False, False]]), [1, 1], [0, 0], [0, 1])


def test_zero_support_and_bad_codes_refused():
    for obs, day in (([[255]], True), ([[1]], False), ([[2]], True)):
        with pytest.raises(ValueError):
            cloud.verify([[0]], obs, np.array([[day]]), [1], [0], [0])


def test_facade_schema_times_and_daylight_are_explicit(tmp_path):
    config = facade(tmp_path)
    assert cloud.read_truth(config)["clm"].shape == (3, 2)
    for delta in ({"schema": np.array("raw.MSG.guess")}, {"times": np.array([1, 2, 3])},
                  {"cloud": np.ones((3, 2), dtype=float)}, {"lat": np.array([0, np.nan], dtype="f4")}):
        with pytest.raises(ValueError):
            facade(tmp_path, **delta)


def test_native_wrf_caller_temporal_match_determinism_and_collect(tmp_path):
    config, path = facade(tmp_path), history(tmp_path)
    args = ([path], config, "2026-01-01T00:00:00Z", 1)
    a, pred, obs, keys = cloud.score_case(*args)
    b, *_ = cloud.score_case(*args)
    assert json.dumps(a, allow_nan=False) == json.dumps(b, allow_nan=False)
    assert a["regions"]["all"]["hits"] == 2 and a["regions"]["all"]["misses"] == 0
    assert a["counts"]["valid"] == 4 and len(keys) == 4
    assert a["matching"]["matched_slots"] == 3
    assert a["footprints"]["parallax"] == "not applied"
    matrix, truth, sigma, k = cloud.collect([[path], [path]], config, args[2], args[3], sigma=.2)
    np.testing.assert_array_equal(matrix, np.array([pred, pred]))
    np.testing.assert_array_equal(truth, obs)
    np.testing.assert_array_equal(sigma, [.2]*4)
    assert len(k) == 4
    report, *_ = cloud.score_case(*args, before="2026-01-01T00:30:00Z")
    assert report["counts"]["valid"] == 2  # held-out boundary excludes even exactly-on-boundary slots
    changed = copy.deepcopy(config)
    changed["truth_identity"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="truth changed"):
        cloud.score_case([path], changed, args[2], args[3])


def test_thermodynamics_finite_units_and_init_checks(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    config, path = facade(tmp_path), history(tmp_path)
    with pytest.raises(ValueError, match="initialization"):
        cloud.score_case([path], config, "2026-01-02T00:00:00Z", 1)
    with Dataset(path, "a") as ds:
        ds.variables["QCLOUD"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        cloud.score_case([path], config, "2026-01-01T00:00:00Z", 1)


def test_protected_truth_refused_before_payload_open(monkeypatch):
    from gpuwrf.assimilation.letkf import alisios_guard
    from gpuwrf.optimize import data
    monkeypatch.setattr(alisios_guard, "is_alisios_data", lambda p: True)
    monkeypatch.setattr(data, "identity", lambda *a: pytest.fail("protected input opened during hashing"))
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("protected input opened"))
    with pytest.raises(ValueError, match="ALISIOS"):
        cloud.load_config({"format": cloud.FORMAT, "truth": "/protected/msg.npz", "daylight": []}, lambda p: p)


def test_frozen_day_and_area_contract_refusals(tmp_path):
    config = facade(tmp_path)
    naive = copy.deepcopy(config)
    naive["daylight"][0]["time"] = "2026-01-01T00:15:00"
    with pytest.raises(ValueError, match="daylight slot"):
        cloud.read_truth(naive)
    with pytest.raises(ValueError, match="min_common_area"):
        cloud.load_config({"format": cloud.FORMAT, "truth": config["truth"], "daylight": config["daylight"],
                           "min_common_area": .5}, lambda p: p)


def test_exact_match_no_nearest_and_common_area_floor(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset

    config, path = facade(tmp_path), history(tmp_path)
    with Dataset(path, "a") as ds:
        ds.variables["Times"][0] = np.array(list("2026-01-01_00:15:01"), dtype="S1")
    with pytest.raises(ValueError, match="missing required exact daytime"):
        cloud.score_case([path], config, "2026-01-01T00:00:00Z", 1)
    with Dataset(path, "a") as ds:
        ds.variables["Times"][0] = np.array(list("2026-01-01_00:15:00"), dtype="S1")
        ds.variables["Times"][2] = np.array(list("2026-01-01_00:45:01"), dtype="S1")
    report, *_ = cloud.score_case([path], config, "2026-01-01T00:00:00Z", 1)
    assert report["matching"]["unmatched_slots"] == 1
    assert report["counts"]["valid"] == 4  # only the unrequested night slot was unmatched
    # Restore first exact slot, invalidate half of the second slot's model area.
    with Dataset(path, "a") as ds:
        ds.variables["Times"][0] = np.array(list("2026-01-01_00:15:00"), dtype="S1")
        ds.variables["Times"][2] = np.array(list("2026-01-01_00:45:00"), dtype="S1")
    config = facade(tmp_path, cloud=np.array([[0, 1], [255, 0], [255, 0]], dtype="u1"))
    report, *_ = cloud.score_case([path], config, "2026-01-01T00:00:00Z", 1)
    assert report["matching"]["below_common_area_slots"] == 1
    assert report["matching"]["common_area_fraction"] == [1, .5, 0]
    assert report["counts"]["valid"] == 2


def test_dry_density_not_moist_density(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset

    path = history(tmp_path)
    with Dataset(path, "a") as ds:
        ds.variables["QVAPOR"][:] = .1
    with Dataset(path) as ds:
        value = cloud.wrf_columns(ds, 0)
    assert value[0, 1] == pytest.approx(20/(1+.1/(287/461.6)))


def test_cwp_density_uses_original_wrf_heat_capacity(tmp_path):
    """Original WRF cp=7*Rd/2; ALISIOS4V control from the W1a operator note.

    The reference uses the independent exact ratio2/7, not the imported product
    constant. Reverting only the production exponent to287/1004 must fail.
    """
    from gpuwrf.io.netcdf_lock import Dataset

    path = history(tmp_path)
    pressure, theta, qv = 80000., 290., .01
    with Dataset(path, "a") as ds:
        ds.variables["P"][:] = 0.
        ds.variables["PB"][:] = pressure
        ds.variables["T"][:] = theta-300.
        ds.variables["QVAPOR"][:] = qv
        ds.variables["QCLOUD"][:] = 1.e-5
        ds.variables["QICE"][:] = 2.e-5
    # WRF share/module_model_constants.F:19-20,31; ALISIOS4V frozen reader31e13cc207.
    original_temperature = theta*(pressure/100000.)**(2./7.)
    original_density = pressure/(287.*original_temperature*(1.+qv/(287./461.6)))
    legacy_temperature = theta*(pressure/100000.)**(287./1004.)
    legacy_density = pressure/(287.*legacy_temperature*(1.+qv/(287./461.6)))
    assert legacy_density/original_density == pytest.approx(1.000031751, abs=5.e-10, rel=0)
    with Dataset(path) as ds:
        column, density, dz = cloud.wrf_columns(ds, 0, return_geometry=True)
    np.testing.assert_allclose(density/original_density, 1., rtol=0, atol=2.e-15)
    original_cwp = 1000.*(1.e-5+2.e-5)*original_density*np.sum(dz, axis=0)
    np.testing.assert_allclose(column/original_cwp, 1., rtol=0, atol=2.e-15)


def test_fss_excludes_invalid_neighbours_and_undefined_centroid():
    # The invalid middle pixel cannot dilute either fraction at a 120-km radius.
    r, *_ = cloud.verify([[1, 1, 0]], [[1, 255, 0]], np.ones((1, 3), bool), [1, 1000, 1],
                         [0, 0, 0], [0, .5, 1], radii_km=[0, 120])
    assert set(r["regions"]["all"]["fss"].values()) == {1}
    r, *_ = cloud.verify([[0, 0]], [[1, 0]], np.ones((1, 2), bool), [1, 1], [0, 0], [0, 1])
    assert r["regions"]["all"]["centroid_displacement_km"] is None
    assert r["regions"]["all"]["misses"] == 1


def test_discrete_cloud_study_load_and_real_fetch_wiring(tmp_path, monkeypatch):
    from gpuwrf.optimize import study

    module = helper("tests/test_optimize.py")
    path, raw = module.study_files.__wrapped__(tmp_path)
    config = facade(tmp_path)
    raw.update(objective="cloud", cloud={k: config[k] for k in
        ("format", "truth", "daylight", "regions", "threshold_g_m2", "pixel_threshold")})
    raw.pop("stations")
    raw.pop("observations")
    path.write_text(json.dumps(raw))
    loaded = study.load_study(path)
    assert loaded["stations"] is None and loaded["observations"] is None
    assert all(v in loaded["inputs"] for v in loaded["cloud"]["truth_identity"])
    calls = []
    different_footprint = [False]

    def score_case(paths, config, issue, hours, *, before=None):
        calls.append((paths, before))
        p = .1 if Path(paths[0]).name == "good" else .9
        operator = Path(paths[0]).name if different_footprint[0] else "fixed"
        return {"loss": p*p, "footprints": {"sha256": operator}}, np.array([p]), np.array([0.]), [(issue, 0)]

    monkeypatch.setattr(cloud, "score_case", score_case)
    loaded["candidates"] = [{"id": "good"}, {"id": "bad"}]
    loaded["rounds"] = [{"train_dates": 2, "keep": 1}]
    report = study.evaluate(loaded, lambda c, cases: {case["id"]: ([c["id"]], {}, []) for case in cases})
    assert report["recommended"]["id"] == "good"
    assert report["held_out"]["loss"] == pytest.approx(.01)
    assert any(before is None for _, before in calls) and any(before is not None for _, before in calls)
    inventory = {"schema": "gpuwrf.optimize.forecasts.v1", "forecasts": {}}
    for candidate in loaded["candidates"]:
        (tmp_path/candidate["id"]).write_text("synthetic offline inventory; collector stubbed")
        inventory["forecasts"][candidate["id"]] = {c["id"]: [candidate["id"]] for c in loaded["cases"]}
    inv = tmp_path/"forecasts.json"
    inv.write_text(json.dumps(inventory))
    assert study.score(loaded, inv)["recommended"]["id"] == "good"
    different_footprint[0] = True
    with pytest.raises(ValueError, match="footprint differs"):
        study.evaluate(loaded, lambda c, cases: {case["id"]: ([c["id"]], {}, []) for case in cases})


@pytest.mark.parametrize("observations", [1, 1100])
def test_continuous_cloud_load_and_driver_wiring(tmp_path, monkeypatch, observations):
    from gpuwrf.optimize import continuous
    from gpuwrf.io.netcdf_lock import Dataset

    module = helper("tests/w41_optcont/test_continuous.py")
    path, raw = module.study_files.__wrapped__(tmp_path)
    config = facade(tmp_path)
    raw["scorer"] = {"kind": "cloud", "sigma": .2,
                     "cloud": {k: config[k] for k in ("format", "truth", "daylight", "regions", "superobs")}}
    raw.pop("stations")
    raw["eki"] = {"members": 4, "iterations": 1, "seed": 11}
    path.write_text(json.dumps(raw))
    loaded = continuous.load_continuous(path)
    assert loaded["stations"] is None and loaded["scorer"]["kind"] == "cloud"
    assert all(v in loaded["inputs"] for v in loaded["scorer"]["cloud"]["truth_identity"])
    calls = []

    def collect_cloud(paths, config, issue, hours, *, before=None, sigma=1, reports=None, operator="superobservations"):
        assert operator == "superobservations"
        calls.append((issue, before, sigma))
        # Exact hand-shaped EKI collector; real NetCDF cloud caller is tested separately.
        values = []
        for member in paths:
            with Dataset(member[0]) as ds:
                values.append([float(ds.variables["T2"][0, 0, 0]-273.15)/100]*observations)
            reports.append({"loss": values[-1][0]**2, "regions": {"all": {"hits": 0}}})
        return np.array(values), np.zeros(observations), np.full(observations, sigma), [
            (n, issue[:10]+"T01:00:00+00:00", "cloud") for n in range(observations)]

    monkeypatch.setattr(cloud, "collect", collect_cloud)
    if observations > 1:
        from gpuwrf.surface_params import eki
        monkeypatch.setattr(eki, "ENSEMBLE_SPACE_UPDATE", False, raising=False)
        monkeypatch.setattr(eki, "update", lambda *a, **kw: pytest.fail("update entered before total observation guard"))
        with pytest.raises(ValueError, match="legacy safe cap"):
            module.run_study(loaded, tmp_path/"run", monkeypatch, wave_mode=False, wall_seconds=None, max_retries=None)
        assert len(calls) == 2  # each case is safe alone; their concatenation is not
        return
    # Explicit legacy-mode reference; production defaults must still pause as waves.
    report, batches = module.run_study(loaded, tmp_path/"run", monkeypatch,
                                       wave_mode=False, wall_seconds=None, max_retries=None)
    assert report["scorer"]["kind"] == "cloud" and report["program"]["verdict"] == "ONE_PROGRAM"
    assert len(report["cloud_diagnostics"]) == 6  # prior/posterior x three cases
    assert report["evaluation"]["test"]["posterior"]["observations"] == 1
    # Sensitivity adds one extraction per training case before the EKI iteration.
    assert len(calls) == 7 and all(sigma == .2 for _, _, sigma in calls)
    assert any(before is None for _, before, _ in calls)
    assert any(before is not None for _, before, _ in calls)
    wave_output = tmp_path/"cloud-waves"
    paused, _ = module.run_study(loaded, wave_output, monkeypatch)
    # Prior/control plus four probes on two training dates precede eight EKI members.
    assert paused["status"] == "PAUSED" and paused["completed_members"] == 18
    state = json.loads((wave_output/"wave_state.json").read_text())
    wave_report = json.loads((wave_output/state["pending"]["report"]).read_text())
    assert state["checkpoint"]["kind"] == "eki" and state["checkpoint"]["iteration"] == 1
    assert wave_report["resources"]["accounting"]["extract_qc_s"] > 0


def test_native_cwp_threshold_inclusive(tmp_path, monkeypatch):
    config, path = facade(tmp_path), history(tmp_path)
    original = cloud.wrf_columns
    def columns(*args, **kwargs):
        _, rho, dz = original(*args, return_geometry=True)
        return np.array([[10., 0.]]), rho, dz
    monkeypatch.setattr(cloud, "wrf_columns", columns)
    _, pred, _, _ = cloud.score_case([path], config, "2026-01-01T00:00:00Z", 1)
    assert pred.tolist() == [1, 0, 1, 0]


def test_nearest_centre_uses_equal_area_projection():
    # At60N, EPSG6933 compresses north-south enough that the second centre is
    # nearer; raw geographic-degree Euclidean matching would choose the first.
    op = cloud.Footprints.build([[60]], [[0]], [[1]], [60, 61.2], [1, 0], max_distance_m=100000)
    assert op.index.tolist() == [1]
    assert 70000 < op.max_distance_m < 90000


def test_multiple_native_dates_without_merging_or_dropping_sources(tmp_path):
    a, b = tmp_path/"a", tmp_path/"b"
    a.mkdir(); b.mkdir()
    first = facade(a)
    # The helper's day contract is Jan1; construct a second native payload explicitly.
    with np.load(a/"truth.npz", allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    values["times"] = values["times"]+86400
    np.savez(b/"truth.npz", **values)
    raw = {"format": cloud.FORMAT, "truth": [str(a/"truth.npz"), str(b/"truth.npz")],
           "daylight": first["daylight"]+[{"time": f"2026-01-02T00:{m}:00Z", "pixel_ids": "all"} for m in (15, 30)]}
    config = cloud.load_config(raw, lambda p: p)
    truth = cloud.read_truth(config)
    assert truth["clm"].shape == (6, 2) and truth["daylight"].sum() == 8
    assert len(config["truth_identity"]) == 2
    path = history(a)
    assert cloud.score_case([path], config, "2026-01-01T00:00:00Z", 1)[0]["counts"]["valid"] == 4


@pytest.mark.parametrize("conflict", ["cloud", "grid", "none"])
def test_native_d18_duplicate_endpoint_and_grid_contract(tmp_path, conflict):
    a, b = tmp_path/"a", tmp_path/"b"
    a.mkdir(); b.mkdir()
    first = facade(a)
    with np.load(a/"truth.npz", allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    if conflict == "cloud":
        values["cloud"][0, 0] = 1
    elif conflict == "grid":
        values["lon"][0] = .1
    np.savez(b/"truth.npz", **values)
    raw = {"format": cloud.FORMAT, "truth": [str(a/"truth.npz"), str(b/"truth.npz")], "daylight": first["daylight"]}
    if conflict == "none":
        assert cloud.read_truth(cloud.load_config(raw, lambda p: p))["clm"].shape == (3, 2)
    else:
        with pytest.raises(ValueError, match="conflicting duplicate|grid/order differs"):
            cloud.load_config(raw, lambda p: p)


def test_shared_footprint_identity_required_even_with_same_pixel_support(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    config, path = facade(tmp_path), history(tmp_path)
    other = tmp_path/"other"
    other.mkdir()
    changed = history(other)
    with Dataset(changed, "a") as ds:
        ds.DX = 2000.  # Same pixels, different physical area operator.
    with pytest.raises(ValueError, match="footprint differs"):
        cloud.collect([[path], [changed]], config, "2026-01-01T00:00:00Z", 1)


def test_superobservations_area_weight_both_sides_and_keep_sigma(tmp_path):
    config, path = facade(tmp_path), history(tmp_path)
    config["regions"] = {}
    config["superobs"]["correlation_length_km"] = 200.
    reports = []
    pred, obs, sigma, keys = cloud.collect([[path], [path]], config, "2026-01-01T00:00:00Z", 1,
                                           sigma=.2, reports=reports)
    np.testing.assert_array_equal(pred, np.full((2, 2), .5))
    np.testing.assert_array_equal(obs, [.5, .5])
    np.testing.assert_array_equal(sigma, [.2, .2])  # correlated pixels never earn sqrt(2) precision
    assert len(keys) == 2
    assert reports[0]["superobservations"]["native_samples"] == 4
    assert reports[0]["superobservations"]["pixel_counts"] == [2, 2]
    assert reports[0]["counts"]["valid"] == 4  # native controls remain visible


def test_superobservation_known_area_weights_and_region_partition():
    truth = {"lat": np.array([0, 0, .5]), "lon": np.array([0, .001, 1]), "regions": {}}
    report = {"footprints": {"assigned_area_m2": [1, 3, 2]}}
    config = {"superobs": {"correlation_length_km": 10, "contract": "SYNTHETIC_TEST_ONLY"}}
    keys = [("2026-01-01T00:15:00+00:00", n) for n in range(3)]
    pred, obs, _ = cloud.superobserve(report, np.array([1, 0, .5]), np.array([0, 1, 1]), keys, config, truth)
    np.testing.assert_array_equal(pred, [.25, .5])
    np.testing.assert_array_equal(obs, [.75, 1])
    assert report["superobservations"]["pixel_counts"] == [2, 1]
    truth["regions"] = {"a": np.array([True, True, False]), "b": np.array([False, True, True])}
    with pytest.raises(ValueError, match="disjoint registered regions"):
        cloud.superobserve(report, np.ones(3), np.ones(3), keys, config, truth)


def test_native_and_large_eki_are_refused_before_dense_allocation(monkeypatch):
    from gpuwrf.surface_params import eki
    from gpuwrf.optimize import continuous
    monkeypatch.setattr(eki, "ENSEMBLE_SPACE_UPDATE", False, raising=False)
    monkeypatch.setattr(np, "diag", lambda *a: pytest.fail("dense observation matrix allocated"))
    with pytest.raises(ValueError, match="full-native cloud EKI"):
        continuous._scorer({"kind": "cloud", "observation_operator": "native", "sigma": .2, "cloud": {}}, [],
                           lambda p: pytest.fail("native study payload read before refusal"))
    with pytest.raises(ValueError, match="legacy safe cap"):
        cloud.guard_observations(25200)
    cloud.guard_observations(2048)
    monkeypatch.setattr(eki, "ENSEMBLE_SPACE_UPDATE", True)
    cloud.guard_observations(25200, "native")  # trusted source capability, no real solve in this control


@pytest.mark.parametrize("spec", [None, {"contract": "LABEL_WITHOUT_LENGTH"}, {"correlation_length_km": 10.}])
def test_eki_requires_a_frozen_superobservation_contract(tmp_path, spec):
    from gpuwrf.optimize import continuous
    config = facade(tmp_path)
    raw = {k: config[k] for k in ("format", "truth", "daylight", "regions")}
    if spec is not None:
        raw["superobs"] = spec
    with pytest.raises(ValueError, match="admitted superobs"):
        continuous._scorer({"kind": "cloud", "sigma": .2, "cloud": raw}, [], lambda p: p)


def test_memory_capability_does_not_admit_native_correlation(tmp_path, monkeypatch):
    from gpuwrf.optimize import continuous
    from gpuwrf.surface_params import eki

    assert eki.ENSEMBLE_SPACE_UPDATE is True  # actual reviewed memory capability
    config, path = facade(tmp_path), history(tmp_path)
    raw = {k: config[k] for k in ("format", "truth", "daylight", "regions")}
    cloud.guard_observations(25200, "native")  # memory only, NOT statistical admission
    for fields in (raw, {**raw, "superobs": config["superobs"]}):
        with pytest.raises(ValueError, match="ALISIOS correlation contract"):
            continuous._scorer({"kind": "cloud", "sigma": .2, "cloud": fields,
                                "observation_operator": "native"}, [],
                               lambda p: pytest.fail("truth read before native correlation refusal"))
    direct_study = {"parameters": [], "configuration": {}, "cases": [], "test_start": "2026-01-05T00:00:00Z",
                    "scorer": {"kind": "cloud", "variables": ["cloud"], "sigma": {"cloud": .2},
                               "cloud": config, "observation_operator": "native"}}
    with pytest.raises(ValueError, match="ALISIOS correlation contract"):
        continuous.driver(direct_study)  # normalized/mutated study cannot bypass pre-launch admission
    with monkeypatch.context() as patch:
        patch.setattr(cloud, "score_case", lambda *a, **kw: pytest.fail("forecast read before native correlation refusal"))
        for contract in (None, config["superobs"]):
            with pytest.raises(ValueError, match="ALISIOS correlation contract"):
                cloud.collect([[path]], {**config, "superobs": contract}, "2026-01-01T00:00:00Z", 1,
                              operator="native", sigma=.2)
    with pytest.raises(ValueError, match="admitted superobs"):
        continuous._scorer({"kind": "cloud", "sigma": .2, "cloud": raw}, [], lambda p: p)
    default = continuous._scorer({"kind": "cloud", "sigma": .2,
                                  "cloud": {**raw, "superobs": config["superobs"]}}, [], lambda p: p)
    assert default["observation_operator"] == "superobservations"




def add_diagnostics(path):
    from gpuwrf.io.netcdf_lock import Dataset
    with Dataset(path, "a") as ds:
        for name, value in (("QC_BL", 2e-4), ("QI_BL", 1e-4), ("CLDFRA_BL", .8),
                            ("QC_RAD", 1e-4), ("QI_RAD", 1e-5), ("QS_RAD", 0), ("QG_RAD", 0), ("CLDFRA", .3)):
            var = ds.createVariable(name, "f8", ("Time", "bottom_top", "south_north", "west_east"))
            var.units = "" if name.startswith("CLDFRA") else "kg kg-1"
            var[:] = value


def test_separate_prognostic_subgrid_and_active_clouds(tmp_path):
    config, path = facade(tmp_path), history(tmp_path)
    report, *_ = cloud.score_case([path], config, "2026-01-01T00:00:00Z", 1)
    channels = report["channel_diagnostics"][0]["channels"]
    assert channels["pbl_subgrid"]["status"] == "UNAVAILABLE"
    assert channels["radiative_active"]["status"] == "UNAVAILABLE"
    add_diagnostics(path)
    report, *_ = cloud.score_case([path], config, "2026-01-01T00:00:00Z", 1)
    channels = report["channel_diagnostics"][0]["channels"]
    assert all(c["status"] == "AVAILABLE" for c in channels.values())
    values = [channels[k]["regions"]["all"]["column_condensate_g_m2"] for k in channels]
    assert len(set(values)) == 3  # prognostic, PBL and actual solver inputs are distinct
    assert channels["pbl_subgrid"]["regions"]["all"]["layer_area_means"]["CLDFRA_BL"] == [.8, .8]
    assert channels["radiative_active"]["regions"]["all"]["layer_area_means"]["CLDFRA"] == [.3, .3]
    assert "not recorded" in channels["radiative_active"]["call_time"]
    assert report["channel_diagnostics"][2]["common_pixels"] == 0  # night stays excluded for all channels
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("name", ["QC_BL", "QI_BL", "CLDFRA_BL", "QC_RAD", "QI_RAD", "QS_RAD", "QG_RAD", "CLDFRA"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_diagnostics_refused_before_night_mask(tmp_path, name, bad):
    from gpuwrf.io.netcdf_lock import Dataset
    config, path = facade(tmp_path), history(tmp_path)
    add_diagnostics(path)
    with Dataset(path, "a") as ds:
        ds.variables[name][2, 0, 0, 0] = bad  # even when every pixel is night/invalid
    with pytest.raises(ValueError, match="nonfinite"):
        cloud.score_case([path], config, "2026-01-01T00:00:00Z", 1)


@pytest.mark.skipif(not os.environ.get("GPUWRF_CLOUD_ARCHIVED_WRF"), reason="archived WRF structure fixture path not supplied")
def test_archived_wrf_with_synthetic_msg(tmp_path):
    """Real archived fields, invented CLM: structure/operator test, never fidelity evidence."""
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.verify.wrfout import _times, _field

    source = Path(os.environ["GPUWRF_CLOUD_ARCHIVED_WRF"])
    with Dataset(source) as ds:
        lat, lon = _field(ds, "XLAT", 0), _field(ds, "XLONG", 0)
        valid = _times(ds)[0]
        issue = str(ds.SIMULATION_START_DATE).replace("_", "T")+"Z"
        expected = cloud.wrf_columns(ds, 0)
    truth = tmp_path/"synthetic-msg-archive.npz"
    # Match the original mass-grid centres, explicitly synthetic clear CLM.
    assert valid.value % (900*10**9) == 0, "select an archived exact-slot frame"
    np.savez(truth, times=np.array([valid.value//10**9], dtype="<i8"),
        cloud=np.zeros((1, lat.size), dtype="u1"), lat=lat.ravel().astype("<f4"), lon=lon.ravel().astype("<f4"))
    config = cloud.load_config({"format": cloud.FORMAT, "truth": str(truth),
        "daylight": [{"time": valid.isoformat(), "pixel_ids": "all"}]}, lambda p: p)
    report, pred, _, _ = cloud.score_case([source], config, issue, 72)
    assert report["counts"]["valid"] == lat.size
    np.testing.assert_array_equal(pred, (expected >= 10).ravel())
    assert report["matching"]["unmatched_slots"] == 0
    assert report["regions"]["all"]["false_alarms"] == int((expected >= 10).sum())
