"""Grid-lock admission at the real model consumers (critic z0nightlycr FIX 1). Host-only, no model steps.

The changed-grid case is the critic's counterexample: same shape, one barren cell becomes class 7 and HGT
changes. Every consumer (public pipeline entry B=1/B>1 incl. GPUWRF_BATCH_INPUT_DIRS lane overrides,
_load_domains, single and batch binders) must refuse it; the unchanged grid and unlocked files pass.
"""
import shutil
import sys
from dataclasses import make_dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("netCDF4")

from gpuwrf.integration import nested_pipeline as p
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.surface_params import controls as sc
from gpuwrf.surface_params import grid_lock as gl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "w41_trackp"))
from w41_trackp_fixtures import SHAPE, _fixture, _namelist  # noqa: E402

RULE = {"control": "z0bare", "class": 16}


def _case(path: Path, domains=("d03",)) -> Path:
    path.mkdir()
    for dom in domains:
        with Dataset(path / f"wrfinput_{dom}", "w") as ds:
            ds.createDimension("Time", 1)
            ds.createDimension("south_north", SHAPE[0])
            ds.createDimension("west_east", SHAPE[1])
            for key in gl.ATTRS:
                ds.setncattr(key, "MODIFIED_IGBP_MODIS_NOAH" if key == "MMINLU" else np.int32(3))
            for key in gl.ARRAYS:
                var = ds.createVariable(key, "i4" if key == "IVGTYP" else "f4", ("Time", "south_north", "west_east"))
                var[:] = 16 if key in ("LU_INDEX", "IVGTYP") else 1
    (path / "namelist.input").write_text("&physics\n mp_physics=8,\n/\n&dfi_control\n dfi_opt=0,\n/\n")
    return path


def _change_grid(path: Path) -> None:
    with Dataset(path / "wrfinput_d03", "a") as ds:
        ds["HGT"][0, 0, 0] += np.float32(1)
        ds["IVGTYP"][0, 0, 0] = 7
        ds["LU_INDEX"][0, 0, 0] = 7


@pytest.fixture
def world(tmp_path, monkeypatch):
    gl._GRID_PASSED.clear()
    monkeypatch.setenv("WRFOUT_ZNT", "0")
    monkeypatch.delenv("GPUWRF_BATCH_ENSEMBLE", raising=False)
    monkeypatch.delenv("GPUWRF_BATCH_INPUT_DIRS", raising=False)
    good = _case(tmp_path / "good")
    twin = tmp_path / "twin"
    shutil.copytree(good, twin)
    bad = tmp_path / "bad"
    shutil.copytree(good, bad)
    _change_grid(bad)
    field = np.full(SHAPE, .02, np.float32)
    locked = tmp_path / "locked.npz"
    sc.save_controls(locked, {"d03": {"z0bare": field}},
                     provenance={"grid_lock": gl.make_lock(good, ("d03",), RULE)})
    unlocked = tmp_path / "unlocked.npz"
    sc.save_controls(unlocked, {"d03": {"z0bare": field}})
    calls = []
    monkeypatch.setattr(p, "_execute_nested_pipeline", lambda config, **kw: calls.append(config) or {"inner": True})
    return {"good": good, "twin": twin, "bad": bad, "locked": locked, "unlocked": unlocked, "field": field,
            "calls": calls, "tmp": tmp_path}


def _config(w, case, **kw):
    return p.NestedPipelineConfig(case, w["tmp"] / "out", w["tmp"] / "proof", 1., 3, aot_prefetch=False, **kw)


def _nl(case):
    import jax.numpy as jnp
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    _, _, static = _fixture()
    with Dataset(case / "wrfinput_d03") as ds:
        classes = np.asarray(ds["IVGTYP"][0])
    static = static.replace(ivgtyp=jnp.asarray(classes), lu_index=jnp.asarray(classes))
    f = _namelist(static)
    return OperationalNamelist(grid=None, tendencies=None, metrics=None, use_noahmp=True, noahmp_static=static,
                               noahmp_energy_params=f.noahmp_energy_params, noahmp_rad_params=f.noahmp_rad_params)


# ---------------------------------------------------------------- public entry, B=1
def test_b1_public_entry_admits_same_grid(world):
    assert p.execute_nested_pipeline(_config(world, world["good"], surface_controls=world["locked"])) == {"inner": True}


def test_b1_public_entry_refuses_changed_grid_before_inner_pipeline(world):
    with pytest.raises(gl.GridLockError, match=r"d03 grid differs.*(HGT|IVGTYP|LU_INDEX)"):
        p.execute_nested_pipeline(_config(world, world["bad"], surface_controls=world["locked"]))
    assert world["calls"] == []


def test_b1_public_entry_unlocked_file_keeps_legacy_behaviour(world):
    assert p.execute_nested_pipeline(_config(world, world["bad"], surface_controls=world["unlocked"])) == {"inner": True}


# ---------------------------------------------------------------- public entry, B>1 and lane-dir overrides
def test_b2_lane_dir_override_refuses_changed_lane(world, monkeypatch):
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "2")
    monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", f"{world['good']},{world['bad']}")
    with pytest.raises(gl.GridLockError, match="d03 grid differs"):
        p.execute_nested_pipeline(_config(world, world["good"], surface_controls=world["locked"]))
    assert world["calls"] == []


def test_b2_homogeneous_changed_lanes_are_refused(world, monkeypatch):
    bad2 = world["tmp"] / "bad2"
    shutil.copytree(world["bad"], bad2)
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "2")
    monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", f"{world['bad']},{bad2}")
    with pytest.raises(gl.GridLockError):  # the CLI's args.input_dir would still point at the good case
        p.execute_nested_pipeline(_config(world, world["good"], surface_controls=world["locked"]))
    assert world["calls"] == []


def test_b2_same_grid_lanes_pass(world, monkeypatch):
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "2")
    monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", f"{world['good']},{world['twin']}")
    assert p.execute_nested_pipeline(_config(world, world["good"], surface_controls=world["locked"])) == {"inner": True}


def test_b2_per_lane_files_checked_against_their_own_lane(world, monkeypatch):
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "2")
    monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", f"{world['good']},{world['bad']}")
    cfg = _config(world, world["good"], batch_surface_controls=(world["unlocked"], world["locked"]))
    with pytest.raises(gl.GridLockError, match="d03 grid differs"):
        p.execute_nested_pipeline(cfg)
    cfg = _config(world, world["good"], batch_surface_controls=(world["locked"], world["unlocked"]))
    assert p.execute_nested_pipeline(cfg) == {"inner": True}  # locked file sits on the good lane


def test_b2_config_batch_input_dirs_are_checked(world, monkeypatch):
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "2")
    cfg = _config(world, world["good"], surface_controls=world["locked"], batch_input_dirs=(world["good"], world["bad"]))
    with pytest.raises(gl.GridLockError):
        p.execute_nested_pipeline(cfg)


# ---------------------------------------------------------------- inner consumers
def _binder_files(w, nl):
    """Locked/unlocked files in the consumer dtype of this process (x64 may be on under pytest)."""
    field = w["field"].astype(sc.target_dtype("z0bare", nl))
    locked, unlocked = w["tmp"] / "locked_c.npz", w["tmp"] / "unlocked_c.npz"
    if not locked.exists():
        sc.save_controls(locked, {"d03": {"z0bare": field}},
                         provenance={"grid_lock": gl.make_lock(w["good"], ("d03",), RULE)})
        sc.save_controls(unlocked, {"d03": {"z0bare": field}})
    return locked, unlocked, field


def test_load_domains_refuses_before_any_init(world, monkeypatch):
    monkeypatch.setattr("gpuwrf.io.cammg_config.validate_input_dir", lambda d: None)
    with pytest.raises(gl.GridLockError, match="d03 grid differs"):
        p._load_domains(_config(world, world["bad"], surface_controls=world["locked"]), ("d01", "d02", "d03"))


def test_single_binder_refuses_locked_file_without_admission_receipt(world):
    nl = _nl(world["bad"])
    locked, unlocked, field = _binder_files(world, nl)
    run = ("d01", "d02", "d03")
    with pytest.raises(ValueError, match="needs grid admission"):
        sc.bind_domain_controls(nl, locked, "d03")  # the critic's direct call
    with pytest.raises(gl.GridLockError, match="d03 grid differs"):
        gl.admit_control_source(locked, world["bad"], run)  # no receipt exists for the changed grid
    receipt = gl.admit_control_source(locked, world["good"], run)
    forged = {**receipt, "digest": "0" * 64}
    with pytest.raises(ValueError, match="needs grid admission"):
        sc.bind_domain_controls(nl, locked, "d03", grid_admission=forged)
    good = sc.bind_domain_controls(_nl(world["good"]), locked, "d03", grid_admission=receipt)
    assert np.array_equal(good.surface_controls["z0bare"], field)
    legacy = sc.bind_domain_controls(nl, unlocked, "d03")  # unlocked: unchanged behaviour, no receipt needed
    assert np.array_equal(legacy.surface_controls["z0bare"], field)


def test_batch_binder_refuses_changed_lane_and_missing_dirs(world):
    Bundle = make_dataclass("Bundle", ["namelist"], frozen=True)
    nl = _nl(world["good"])
    locked, _, field = _binder_files(world, nl)
    loaded = [(None, {"d03": Bundle(nl)}, {}, None, {}, {})] * 2
    src = (locked, locked)
    with pytest.raises(gl.GridLockError, match="case input dir"):
        p._bind_batch_surface_controls(loaded, ("d03",), src, 2)  # the critic's direct call
    with pytest.raises(gl.GridLockError, match="d03 grid differs"):
        p._bind_batch_surface_controls(loaded, ("d03",), src, 2, input_dirs=(world["good"], world["bad"]))
    ok = p._bind_batch_surface_controls(loaded, ("d03",), src, 2, input_dirs=(world["good"], world["twin"]))
    assert all(np.array_equal(r[1]["d03"].namelist.surface_controls["z0bare"], field) for r in ok)


# ---------------------------------------------------------------- once per case/control pair, replacement noticed
def test_admission_is_memoized_and_notices_replaced_wrfinput(world, monkeypatch):
    seen = []
    real = gl.fingerprint
    monkeypatch.setattr(gl, "fingerprint", lambda d, doms: seen.append(str(d)) or real(d, doms))
    for _ in range(3):
        gl.admit_control_source(world["locked"], world["good"], ("d01", "d02", "d03"))
    assert len(seen) == 1
    victim = world["good"] / "wrfinput_d03"
    shutil.copyfile(world["bad"] / "wrfinput_d03", victim.with_suffix(".new"))
    victim.with_suffix(".new").replace(victim)  # same path, new content
    with pytest.raises(gl.GridLockError, match="d03 grid differs"):
        gl.admit_control_source(world["locked"], world["good"], ("d01", "d02", "d03"))


# ---------------------------------------------------------------- DELTA 1: distinct manifests sharing one lock
# A controls every locked domain (d03+d04); B shares A's lock provenance but controls only d03. The memo is
# deliberately NOT cleared between steps: a remembered grid PASS must never skip B's structural checks.
RUN4 = ("d01", "d02", "d03", "d04")


@pytest.fixture
def shared_lock(tmp_path, monkeypatch):
    gl._GRID_PASSED.clear()  # once per fixture, never between the steps of a test
    monkeypatch.setenv("WRFOUT_ZNT", "0")
    monkeypatch.delenv("GPUWRF_BATCH_ENSEMBLE", raising=False)
    monkeypatch.delenv("GPUWRF_BATCH_INPUT_DIRS", raising=False)
    good = _case(tmp_path / "good4", ("d03", "d04"))
    twin = tmp_path / "twin4"
    shutil.copytree(good, twin)
    lock = gl.make_lock(good, ("d03", "d04"), RULE)
    field = np.full(SHAPE, .02, np.float32)
    a, b = tmp_path / "A.npz", tmp_path / "B.npz"
    sc.save_controls(a, {"d03": {"z0bare": field}, "d04": {"z0bare": field}}, provenance={"grid_lock": lock})
    sc.save_controls(b, {"d03": {"z0bare": field}}, provenance={"grid_lock": lock})
    assert sc.load_controls(b)[1]["provenance"]["grid_lock"]["digest"] == lock["digest"]  # B passes the loader
    calls = []
    monkeypatch.setattr(p, "_execute_nested_pipeline", lambda config, **kw: calls.append(config) or {"inner": True})
    return {"good": good, "twin": twin, "A": a, "B": b, "calls": calls, "tmp": tmp_path}


def test_cold_b_refused_then_a_admitted_then_warm_b_refused(shared_lock, monkeypatch):
    w = shared_lock
    seen = []
    real = gl.fingerprint
    monkeypatch.setattr(gl, "fingerprint", lambda d, doms: seen.append(str(d)) or real(d, doms))
    with pytest.raises(gl.GridLockError, match=r"without controls in the file: \['d04'\]"):
        gl.admit_control_source(w["B"], w["good"], RUN4)  # cold
    assert gl.admit_control_source(w["A"], w["good"], RUN4)["status"] == "GRID_LOCK_PASS"
    with pytest.raises(gl.GridLockError, match=r"without controls in the file: \['d04'\]"):
        gl.admit_control_source(w["B"], w["good"], RUN4)  # warm: the grid PASS of A is remembered
    assert gl.admit_control_source(w["A"], w["good"], RUN4)["status"] == "GRID_LOCK_PASS"
    with pytest.raises(gl.GridLockError, match="not in this run"):
        gl.admit_control_source(w["A"], w["good"], RUN4[:3])  # run-domain check also runs on a warm memo
    assert len(seen) == 1  # the expensive fingerprint is still reused


def test_public_batch_shared_a_and_per_lane_b_refused_in_one_call(shared_lock, monkeypatch):
    w = shared_lock
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "2")
    monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", f"{w['good']},{w['twin']}")
    cfg = p.NestedPipelineConfig(w["good"], w["tmp"] / "out", w["tmp"] / "proof", 1., 4, aot_prefetch=False,
                                 surface_controls=w["A"], batch_surface_controls=(w["B"], w["B"]))
    with pytest.raises(gl.GridLockError, match="without controls in the file"):
        p.execute_nested_pipeline(cfg)
    assert w["calls"] == []
    ok = p.NestedPipelineConfig(w["good"], w["tmp"] / "out", w["tmp"] / "proof", 1., 4, aot_prefetch=False,
                                surface_controls=w["A"], batch_surface_controls=(w["A"], w["A"]))
    assert p.execute_nested_pipeline(ok) == {"inner": True}


def test_binders_refuse_b_even_with_a_receipt(shared_lock):
    w = shared_lock
    receipt_a = gl.admit_control_source(w["A"], w["good"], RUN4)
    with pytest.raises(ValueError, match="without controls in the file"):
        sc.bind_domain_controls(None, w["B"], "d04", grid_admission=receipt_a)  # would silently return OFF for d04
    Bundle = make_dataclass("Bundle", ["namelist"], frozen=True)
    loaded = [(None, {"d03": Bundle(None), "d04": Bundle(None)}, {}, None, {}, {})] * 2
    with pytest.raises(gl.GridLockError, match="without controls in the file"):
        p._bind_batch_surface_controls(loaded, ("d03", "d04"), (w["B"], w["B"]), 2, input_dirs=(w["good"], w["twin"]))
