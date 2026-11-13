"""AFWA severe WSPD10MAX (item 0.g): pristine-WRF oracle, reset clock, refusals, carry/history/restart."""
import base64
import json
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from gpuwrf.diagnostics import afwa

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "proofs/v50_gust/afwa_oracle.json"


def _fixture():
    if not FIXTURE.exists():
        pytest.skip("proofs/v50_gust oracle fixture not on disk (skip-worktree checkout)")
    return json.loads(FIXTURE.read_text())


def _dec(text, shape):
    return np.frombuffer(base64.b64decode(text), np.float32).reshape(shape)


def _run_regime(regime, mutate=None):
    shapes = regime["shapes"]
    states = [{k: _dec(v, shapes[k]) for k, v in st.items()} for st in regime["states"]]
    p = regime["patch"][2]
    carry = afwa.AfwaDiagnostics(jnp.asarray(_dec(regime["initial"], (p, p))))
    outputs = []
    for step in range(1, regime["steps"] + 1):
        st = dict(states[(step - 1) % len(states)])
        if mutate is not None:
            st = mutate(st)
        carry = afwa.update_afwa(carry, u10=st["u10"], v10=st["v10"], raincv=st["raincv"],
                                 rainncv=st["rainncv"], u=st["u"], v=st["v"], ph=st["ph"], phb=st["phb"],
                                 terrain=st["ht"], itimestep=step, dt=regime["dt"],
                                 history_interval=regime["history_interval_min"], interpret=True)
        outputs.append(np.asarray(carry.wspd10max))
    return np.stack(outputs)


def test_oracle_provenance_and_regime_coverage():
    fx = _fixture()
    names = {r["name"] for r in fx["regimes"]}
    assert {"convective_148_d06", "calm_148_d03_night", "stable_prod_d02_night"} <= names
    rates = [max(r["max_prate_mm_h"]) for r in fx["regimes"]]
    # Heavy branch: inside the 50-150 blend ramp and beyond 150 (factor 0).
    assert any(50 < x < 150 for x in rates) and any(x > 150 for x in rates)
    assert set(fx["blocks_sha256"]) == {"zagl", "precip", "severe", "uv_wind"}


@pytest.mark.parametrize("index", range(5))
def test_port_matches_pristine_afwa_oracle(index):
    """The compiled CPU program (opaque REAL rounding) is bit-exact against gfortran."""
    fx = _fixture()
    regime = fx["regimes"][index]
    p = regime["patch"][2]
    expected = _dec(regime["expected"], (regime["steps"], p, p))
    got = _run_regime(regime)
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got.view(np.int32), expected.view(np.int32))


def test_reset_clock_matches_independent_esmf_emulation():
    fx = _fixture()
    for regime in fx["regimes"]:
        resets = [int(afwa.reset_due(s, regime["dt"], regime["history_interval_min"]))
                  for s in range(1, regime["steps"] + 1)]
        assert resets == regime["reset"]


@pytest.mark.parametrize("dt,minutes,first", [(54, 60, 68), (18, 60, 201), (6, 60, 601), (54, 3, 5)])
def test_reset_step_unaligned_and_aligned(dt, minutes, first):
    # First reset after step 1 is the step that STARTS at the first time >= one interval.
    resets = [s for s in range(1, first + 2) if bool(afwa.reset_due(s, dt, minutes))]
    assert resets[:2] == [1, first]


def test_blend_branch_is_exercised_and_deletion_sensitive():
    """Killing the 1 km blend (no precipitation) must change the heavy regimes only."""
    fx = _fixture()
    for regime in fx["regimes"]:
        p = regime["patch"][2]
        expected = _dec(regime["expected"], (regime["steps"], p, p))
        dry = _run_regime(regime, mutate=lambda st: {**st, "rainncv": st["rainncv"] * 0, "raincv": st["raincv"] * 0})
        heavy = max(regime["max_prate_mm_h"]) > 50
        assert (not np.array_equal(dry, expected)) == heavy, regime["name"]
        # The blend only ever raises the plain 10 m maximum.
        assert (expected >= dry).all()


def test_off_and_validation():
    assert afwa.configuration({}) == {"afwa_severe_opt": 0}
    assert afwa.configuration({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 1}}) == {"afwa_severe_opt": 1}
    assert afwa.configuration({"afwa": {"afwa_diag_opt": 0, "afwa_severe_opt": 1}}) == {"afwa_severe_opt": 0}
    afwa.validate_restart_afwa(None, False, None)
    plane = afwa.AfwaDiagnostics(jnp.zeros((3, 4), jnp.float32))
    afwa.validate_restart_afwa(plane, True, 1, (3, 4))
    for args in ((None, True, 1), (plane, False, 1), (plane, True, None), (plane, True, 1, (4, 3)),
                 (afwa.AfwaDiagnostics(jnp.zeros((3, 4), jnp.float64)) if jax.config.jax_enable_x64 else plane, True, 2)):
        with pytest.raises(ValueError):
            afwa.validate_restart_afwa(*args)
    assert afwa.history_fields(None) == {}
    assert set(afwa.history_fields(plane)) == {"WSPD10MAX"}


# ---------------------------------------------------------------- integration seams
from dataclasses import replace
import pickle
from netCDF4 import Dataset

from gpuwrf.io.namelist_binding import NamelistNotHonouredError, bind_namelist_options, namelist_honour_report
from gpuwrf.io import restart as rp, wrfrst_netcdf as nc
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state
from test_m7_netcdf_writer import writer_authority

ON = {"afwa": {"afwa_diag_opt": [1, 1], "afwa_severe_opt": [0, 1]},
      "physics": {"mp_physics": 8, "cu_physics": [1, 0]}, "time_control": {"history_interval": [60, 60]}}


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_SFC_WSPD", "0")
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re, "_MP_RE", False)
    grid = make_ideal_grid(4, 3, 5)
    state = _state(grid)
    nl = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, afwa_severe_opt=1)
    shape = state.t_skin.shape
    plane = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 8 + .125
    from gpuwrf.runtime.operational_state import seed_history_diagnostics
    history = seed_history_diagnostics(state)._replace(u10=plane + 1, v10=plane - 2)
    return grid, nl, initial_operational_carry(state).replace(afwa_diagnostics=afwa.AfwaDiagnostics(plane),
                                                              history_diagnostics=history)


def _save(path, fmt, grid, nl, carry):
    if fmt == "pickle":
        rp.write_restart(carry, nl, grid, 2, path)
    else:
        nc.write_wrfrst_carry(carry, grid, nl, path, valid_time="2026-06-03_00:30:00",
                              run_start="2026-06-03_00:00:00", step_index=2)


def _read(path, fmt, nl=None):
    return (rp.read_restart(path) if fmt == "pickle" else nc.read_wrfrst_carry(path, namelist=nl))[0]


def test_binding_per_domain_and_static_tree(case):
    _, nl, _ = case
    off = replace(nl, afwa_severe_opt=0)
    with pytest.raises(NamelistNotHonouredError, match="afwa_diag_opt=1 .d01. without"):
        bind_namelist_options(off, ON, "d01")  # native entry refuses like the CLI (critic FIX 5)
    errors, _ = namelist_honour_report(ON, ("d01", "d02"))
    assert any("afwa_diag_opt=1 (d01) without afwa_severe_opt=1" in e for e in errors)
    good = {**ON, "afwa": {"afwa_diag_opt": [0, 1], "afwa_severe_opt": [0, 1]}}
    errors, _ = namelist_honour_report(good, ("d01", "d02"))
    assert not errors, errors
    bound = bind_namelist_options(off, good, "d02")
    assert bound.afwa_severe_opt == 1 and bind_namelist_options(off, good, "d01").afwa_severe_opt == 0
    # OFF keeps the released treedef; ON adds one static aux entry carrying the cadence.
    leaves_off, tree_off = jax.tree.flatten(off)
    leaves_on, tree_on = jax.tree.flatten(bound)
    assert tree_off != tree_on and len(leaves_off) == len(leaves_on)
    assert jax.tree.unflatten(tree_on, leaves_on).afwa_severe_opt == 1
    assert jax.tree.unflatten(tree_on, leaves_on).diagnostic_history_interval == bound.diagnostic_history_interval
    assert jax.tree.flatten(OperationalNamelist(grid=nl.grid, tendencies=None, metrics=nl.metrics))[1] == tree_off


@pytest.mark.parametrize("patch,needle", [
    ({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 1, "afwa_cloud_opt": 1}}, "afwa_cloud_opt"),
    ({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 1, "afwa_bad_data_check": 1}}, "afwa_bad_data_check"),
    ({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 1, "afwa_gust_opt": 1}}, "afwa_gust_opt"),
    ({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 2}}, "only 0/1"),
    ({"afwa": {"afwa_severe_opt": 1}}, "requires afwa_diag_opt=1"),
    ({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 1}, "physics": {"mp_physics": 6}}, "mp_physics=8"),
    ({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 1}, "physics": {"mp_physics": 8, "cu_physics": 3}}, "cu_physics=0/1"),
    ({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 1}, "physics": {"mp_physics": 8},
      "time_control": {"nwp_diagnostics": 1}}, "both own WSPD10MAX"),
    ({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 1}, "physics": {"mp_physics": 8},
      "domains": {"use_adaptive_time_step": True}}, "fixed time_step"),
])
def test_unsupported_afwa_selections_refused(patch, needle):
    errors, _ = namelist_honour_report(patch, ("d01",))
    assert any(needle in e for e in errors), errors


def test_replay_refused():
    errors, _ = namelist_honour_report({"afwa": {"afwa_diag_opt": 1, "afwa_severe_opt": 1},
                                        "physics": {"mp_physics": 8}}, ("d01",), replay=True)
    assert any("native mode" in e for e in errors)


def test_carry_treedef_unchanged_when_off(case):
    _, _, carry = case
    carry = carry.replace(history_diagnostics=None)
    off = carry.replace(afwa_diagnostics=None)
    assert len(jax.tree.leaves(carry)) == len(jax.tree.leaves(off)) + 1
    assert jax.tree.structure(off) == jax.tree.structure(initial_operational_carry(off.state))


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_restart_roundtrip_exact(case, tmp_path, fmt):
    grid, nl, carry = case
    path = tmp_path / fmt
    _save(path, fmt, grid, nl, carry)
    restored = _read(path, fmt, nl if fmt == "netcdf" else None)
    assert np.asarray(restored.afwa_diagnostics.wspd10max).dtype == np.float32
    assert np.asarray(restored.afwa_diagnostics.wspd10max).tobytes() == np.asarray(carry.afwa_diagnostics.wspd10max).tobytes()


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
@pytest.mark.parametrize("damage", ["marker", "group"])
def test_damaged_active_restart_refused(case, tmp_path, fmt, damage):
    grid, nl, carry = case
    path = tmp_path / fmt
    _save(path, fmt, grid, nl, carry)
    if fmt == "pickle":
        payload = pickle.loads(path.read_bytes())
        c = payload["carry"]
        if damage == "marker":
            c.pop("afwa_diagnostics_schema_version")
        else:
            c["extra_fields"]["afwa_diagnostics"] = None
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path, "a") as ds:
            if damage == "marker":
                ds.delncattr("GPUWRF_AFWA_DIAGNOSTICS_SCHEMA_VERSION")
            else:
                ds.delncattr("GPUWRF_AFWA_SEVERE_OPT")
    with pytest.raises(ValueError, match="AFWA.*E78"):
        _read(path, fmt, nl if fmt == "netcdf" else None)


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_off_restart_refuses_on_request(case, tmp_path, fmt):
    grid, nl, carry = case
    path = tmp_path / fmt
    off_nl = replace(nl, afwa_severe_opt=0)
    _save(path, fmt, grid, off_nl, carry.replace(afwa_diagnostics=None))
    assert _read(path, fmt).afwa_diagnostics is None
    if fmt == "netcdf":
        with pytest.raises(ValueError, match="AFWA.*E78"):
            _read(path, fmt, nl)
    else:
        with pytest.raises(ValueError, match="AFWA.*E78"):
            rp.read_restart(path, namelist=nl)


def test_selection_without_carry_refused_before_write(case, tmp_path):
    grid, nl, carry = case
    with pytest.raises(ValueError, match="AFWA.*E78"):
        _save(tmp_path / "x", "pickle", grid, nl, carry.replace(afwa_diagnostics=None))


def test_history_metadata_equals_registry(case, tmp_path):
    """Registry.EM_COMMON:2081 WSPD10MAX ij misc rh02 "WIND SPD MAX 10 M" "m s-1"."""
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
    registry = (ROOT / "data/wrf_pristine/WRF/Registry/Registry.EM_COMMON")
    if not registry.exists():
        pytest.skip("pristine WRF symlink missing")
    line = next(l for l in registry.read_text().splitlines() if l.split()[:3] == ["state", "real", "WSPD10MAX"])
    fields = line.split()
    assert fields[3] == "ij" and fields[7] == "rh02"
    description, units = line.split('"')[3], line.split('"')[5]
    grid, nl, carry = case
    path = tmp_path / "wrfout"
    write_wrfout_netcdf(carry.state, grid, nl, path, domain="d02", domain_authority=writer_authority(grid),
                        lead_hours=.5, diagnostics=afwa.history_fields(carry.afwa_diagnostics),
                        valid_time="2026-06-03_00:30:00", run_start="2026-06-03_00:00:00")
    with Dataset(path) as ds:
        var = ds.variables["WSPD10MAX"]
        assert var.description == description and var.units == units
        assert var.dimensions == ("Time", "south_north", "west_east") and var.MemoryOrder == "XY "
        assert var.stagger == "" and var.dtype == np.float32
        np.testing.assert_array_equal(var[0], np.asarray(carry.afwa_diagnostics.wspd10max))


def test_nested_history_callback_transcribes_afwa(tmp_path):
    from types import SimpleNamespace
    from test_initial_history_wrf_t0 import _writer
    writer, state = _writer(tmp_path, lambda *a, **kw: {})
    value = np.full(np.shape(state.t_skin), 7.25, np.float32)
    output = writer("d01", 200, SimpleNamespace(state=state, afwa_diagnostics=afwa.AfwaDiagnostics(value)))
    with Dataset(output["wrfout"]) as ds:
        np.testing.assert_array_equal(ds["WSPD10MAX"][0], value)



# ---------------------------------------------------------------- critic FIX 1-8 (Sol, ff3d1a5f4f)
AFWA_ON = {"afwa_diag_opt": 1, "afwa_severe_opt": 1}
SUITE = {"mp_physics": 8, "sf_sfclay_physics": 5, "sf_surface_physics": 4}


def _errors(**groups):
    errors, _ = namelist_honour_report({"afwa": AFWA_ON, "physics": dict(SUITE),
                                        "time_control": {"history_interval": 60}, **groups}, ("d01",))
    return errors


def test_fix1_surface_wind_producer_and_bldt_refused(case):
    """U10/V10 are produced every step only by MYNN-SL + Noah-MP; other suites/bldt refused."""
    _, nl, _ = case
    for over in ({"use_noahmp": False, "sf_surface_physics": 0},          # critic: sfclay5, no LSM
                 {"use_noahmp": True, "sf_surface_physics": 4, "sf_sfclay_physics": 1}):
        with pytest.raises(ValueError, match="sf_sfclay_physics=5 with Noah-MP"):
            afwa.validate_options(replace(nl, mp_physics=8, cu_physics=0, run_physics=True, **over),
                                  wrf_order=True, kf_rk=True)
    afwa.validate_options(replace(nl, mp_physics=8, cu_physics=0, run_physics=True, use_noahmp=True,
                                  sf_surface_physics=4, sf_sfclay_physics=5), wrf_order=True, kf_rk=True)
    assert not _errors()
    assert any("bldt=0" in e for e in _errors(physics={**SUITE, "bldt": 10}))
    assert any("Noah-MP" in e for e in _errors(physics={**SUITE, "sf_surface_physics": 0}))
    assert any("Noah-MP" in e for e in _errors(physics={**SUITE, "sf_sfclay_physics": 1}))


def test_fix2_heights_merge_entry_ring_and_pre_microphysics_interior():
    from types import SimpleNamespace
    nz, ny, nx = 3, 5, 6
    entry = SimpleNamespace(ph_perturbation=jnp.full((nz + 1, ny, nx), 1., jnp.float32), t_skin=jnp.zeros((ny, nx)))
    pre = SimpleNamespace(ph_perturbation=jnp.full((nz + 1, ny, nx), 2., jnp.float32), t_skin=jnp.zeros((ny, nx)))
    base = SimpleNamespace(phb=jnp.full((nz + 1, ny, nx), 7., jnp.float32))
    ph, phb = afwa.heights(entry, pre, base, 1)
    ph = np.asarray(ph)
    assert (ph[:, 1:-1, 1:-1] == 2).all() and (ph[:, 0] == 1).all() and (ph[:, :, -1] == 1).all()
    assert (np.asarray(afwa.heights(entry, pre, base, 0)[0]) == 2).all() and (np.asarray(phb) == 7).all()


@pytest.mark.parametrize("dt,minutes", [(54, 60), (18, 60), (6, 60), (13.5, 7), (2.5, 1), (0.5, 0.75), (54, 3),
                                        (40, 1)])
def test_fix3_reset_follows_the_writer_alarm_schedule(dt, minutes):
    from gpuwrf.integration.nested_pipeline import _history_alarm_steps
    total = 3 * int(np.ceil(minutes * 60 / dt)) + 5
    schedule, _ = _history_alarm_steps(total_steps=total, dt_s=dt, interval_minutes=minutes)
    steps = np.arange(1, total + 1, dtype=np.int32)
    hits = np.asarray(jax.jit(jax.vmap(lambda step: afwa.reset_due(step, dt, minutes)))(steps))
    assert steps[hits].tolist() == [1] + [a + 1 for a in schedule if a + 1 <= total]


def test_fix3_wide_rationals_and_alarm_controls_refused():
    with pytest.raises(ValueError, match="exact rational"):
        afwa.reset_due(5401, 54 / 3**4, 60)  # writer: 0.6666666666666666 s, not 2/3: refused, not rounded
    for key in ("history_interval_s", "history_interval_m", "history_begin_h", "history_end_h"):
        assert any(key in e for e in _errors(time_control={"history_interval": 60, key: 30})), key
    assert not _errors(time_control={"history_interval": 60, "history_interval_s": 0})


def test_fix3_explicit_history_slot_per_active_domain():
    two = {"afwa": {"afwa_diag_opt": [0, 1], "afwa_severe_opt": [0, 1]}, "physics": dict(SUITE)}
    errors, _ = namelist_honour_report({**two, "time_control": {"history_interval": [60]}}, ("d01", "d02"))
    assert any("explicit history_interval" in e and "d02" in e for e in errors)
    errors, _ = namelist_honour_report({**two, "time_control": {"history_interval": [60, 30]}}, ("d01", "d02"))
    assert not errors, errors


def test_fix4_omitted_nest_slots_are_wrf_default_off(case):
    from gpuwrf.io.afwa_options import severe_selected
    nml = {"afwa": {"afwa_diag_opt": [0, 1], "afwa_severe_opt": [0, 1]}, "physics": dict(SUITE),
           "time_control": {"history_interval": [60, 60, 60]}}
    assert [severe_selected(nml, d) for d in ("d01", "d02", "d03")] == [False, True, False]
    assert not severe_selected({"afwa": AFWA_ON}, "d02")  # scalar = d01 slot only
    _, nl, _ = case
    off = replace(nl, afwa_severe_opt=0)
    assert bind_namelist_options(off, nml, "d03").afwa_severe_opt == 0
    assert bind_namelist_options(off, nml, "d02").afwa_severe_opt == 1
    errors, _ = namelist_honour_report(nml, ("d01", "d02", "d03"))
    assert not errors, errors


def test_fix5_native_entry_and_batched_refusals(case):
    from types import SimpleNamespace
    from gpuwrf.integration.nested_pipeline import run_batched_operational_domain_tree
    _, nl, _ = case
    with pytest.raises(NamelistNotHonouredError, match="afwa_cloud_opt"):
        bind_namelist_options(nl, {"afwa": {**AFWA_ON, "afwa_cloud_opt": 1}, "physics": dict(SUITE),
                                   "time_control": {"history_interval": 60}}, "d01")
    tree = SimpleNamespace(domains={"d01": SimpleNamespace(namelist=nl)}, feedback_enabled=False)
    with pytest.raises(ValueError, match="AFWA WSPD10MAX .* B=1 only"):
        run_batched_operational_domain_tree(tree, batch_namelists={}, batch_size=2, root_steps=1,
                                            carries={"d01": None})


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_fix6_restart_cadence_change_refused(case, tmp_path, fmt):
    grid, nl, carry = case
    path = tmp_path / fmt
    _save(path, fmt, grid, nl, carry)
    with pytest.raises(ValueError, match="cadence differs"):
        (rp.read_restart if fmt == "pickle" else nc.read_wrfrst_carry)(
            path, namelist=replace(nl, diagnostic_history_interval=30))
    if fmt == "netcdf":
        with Dataset(path, "a") as ds:
            ds.delncattr("GPUWRF_AFWA_HISTORY_INTERVAL")
        with pytest.raises(ValueError, match="AFWA.*cadence.*E78"):
            nc.read_wrfrst_carry(path, namelist=nl)


def test_fix6_active_restart_needs_surface_history(case, tmp_path):
    from gpuwrf.runtime.restart_store import RestartStore
    from test_online_diagnostics_restart_store import identity
    grid, nl, carry = case
    bare = carry.replace(history_diagnostics=None)
    with pytest.raises(ValueError, match="U10/V10 history"):
        _save(tmp_path / "x", "pickle", grid, nl, bare)
    from test_online_diagnostics_restart_store import rewrite
    store = RestartStore(tmp_path / "store", 10**7, 2, 0)
    with pytest.raises(ValueError, match="U10/V10 history"):
        store.save({"d01": bare}, {"d01": 2}, {"d01": nl.dt_s}, identity(nl))
    gen, _ = store.save({"d01": carry}, {"d01": 4}, {"d01": nl.dt_s}, identity(nl))
    restored = store.read(gen, expected_identity=identity(nl))[0]["carries"]["d01"]
    np.testing.assert_array_equal(restored.afwa_diagnostics.wspd10max, carry.afwa_diagnostics.wspd10max)
    # An authenticated snapshot without the surface history (critic scenario) is refused on read.
    rewrite(gen, lambda payload, manifest: payload["extra_fields"].__setitem__("history_diagnostics", None))
    with pytest.raises(ValueError, match="U10/V10 history"):
        store.read(gen, expected_identity=identity(nl))


def test_fix7_jitted_wrapper_with_clock_matches_oracle(monkeypatch):
    """update_afwa jitted together with its clock construction stays bit-exact (critic: 29/2160)."""
    fx = _fixture()
    monkeypatch.setattr(afwa, "update_afwa",
                        jax.jit(afwa.update_afwa, static_argnames=("dt", "history_interval", "interpret")))
    for regime in fx["regimes"]:
        p = regime["patch"][2]
        expected = _dec(regime["expected"], (regime["steps"], p, p))
        np.testing.assert_array_equal(_run_regime(regime).view(np.int32), expected.view(np.int32))


def _reference_step(prev, reset, u10, v10, rain, u, v, ph, phb, ht, dt):
    """numpy REAL (IEEE f32 per op, no contraction) of module_diag_afwa.F:338-393 + uv_wind."""
    f = np.float32
    prev = np.zeros_like(prev) if reset else prev
    wind = np.sqrt(u10 * u10 + v10 * v10)
    prate = (rain / f(dt)) * f(3600)
    out = wind.copy()
    zw = (ph + phb) / f(9.81)
    zagl = f(.5) * (zw[:-1] + zw[1:]) - ht[None]
    ua = f(.5) * (u[:, :, :-1] + u[:, :, 1:])
    va = f(.5) * (v[:, :-1, :] + v[:, 1:, :])
    for y, x in zip(*np.nonzero(prate > f(50))):
        k = next(k for k in range(1, zagl.shape[0]) if zagl[k, y, x] >= f(1000))
        dz = f(1000) - zagl[k - 1, y, x]
        u1 = ua[k - 1, y, x] + dz * ((ua[k, y, x] - ua[k - 1, y, x]) / zagl[k, y, x])
        v1 = va[k - 1, y, x] + dz * ((va[k, y, x] - va[k - 1, y, x]) / zagl[k, y, x])
        factor = max((f(150) - prate[y, x]) / f(100), f(0))
        ub = u10[y, x] * factor + u1 * (f(1) - factor)
        vb = v10[y, x] * factor + v1 * (f(1) - factor)
        gust = np.sqrt(ub * ub + vb * vb)
        out[y, x] = gust if gust > wind[y, x] else wind[y, x]
    return np.where(out > prev, out, prev).astype(np.float32)


def test_fix7_compiled_caller_operand_time_levels(monkeypatch):
    """The real _physics_boundary_step, jitted: step-entry U/V, merged grid%z (pre-MP interior, entry
    ring), this step's U10/V10 and rain, and the history-window reset. A no-op hook fails."""
    import importlib
    from gpuwrf.diagnostics import precipitation
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import seed_history_diagnostics
    cfix = importlib.import_module("test_t_cfix_cu_cu_rk")
    grid, state, carry, nml = cfix._fixture(0)
    nml = replace(nml, afwa_severe_opt=1, diagnostic_history_interval=0.5)  # dt 10 s: alarms at steps 3, 6
    carry = carry.replace(base_state=om._base_state_from_totals(state))  # production carries base_state
    nz, ny, nx = state.theta.shape
    rng = np.random.default_rng(5)
    f32 = lambda a: np.asarray(a, np.float32)
    phb = f32(carry.base_state.phb)
    rk_dph = jnp.asarray(rng.uniform(-400, 400, (nz + 1, ny, nx)) * 9.81)
    winds = [(jnp.asarray(rng.uniform(-12, 12, state.u.shape)), jnp.asarray(rng.uniform(-12, 12, state.v.shape)))
             for _ in range(4)]

    def fake_rk(c, nl, **kw):
        s = c.state
        return c.replace(state=s.replace(u=s.u + 7., v=s.v - 5., ph_perturbation=s.ph_perturbation + rk_dph,
                                         ph_total=s.ph_total + rk_dph))

    def fake_mp(s, nl, return_precipitation=False, **kw):  # stands in for a later ring pin of ph_2
        s = s.replace(ph_perturbation=s.ph_perturbation + 333., ph_total=s.ph_total + 333.)
        return (s, None) if return_precipitation else s

    rain = jnp.asarray(rng.choice([0., .05, .2, .3, .6], (ny, nx)).astype(np.float32))  # mm/step at dt 10 s
    monkeypatch.setattr(om, "_rk_scan_step", fake_rk)
    monkeypatch.setattr(om, "_apply_post_rk_microphysics", fake_mp)
    monkeypatch.setattr(om, "_microphysics_spec_zone", lambda nl: 1)
    monkeypatch.setattr(precipitation, "thompson_amounts", lambda precip, spec_zone=0: (rain, rain * 0))
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")
    step_fn = jax.jit(lambda c, i: om._physics_boundary_step(c, nml, i, run_radiation=False))
    seed = jnp.full((ny, nx), 2.5, jnp.float32)
    carry = carry.replace(afwa_diagnostics=afwa.AfwaDiagnostics(seed))
    expected = np.asarray(seed)
    sensitive = {"final_ph": False, "final_uv": False}
    for step in range(1, 5):
        u, v = winds[step - 1]
        entry = carry.state.replace(u=u, v=v)
        u10 = jnp.asarray(rng.uniform(-9, 9, (ny, nx)).astype(np.float32))
        v10 = jnp.asarray(rng.uniform(-9, 9, (ny, nx)).astype(np.float32))
        history = seed_history_diagnostics(entry)._replace(u10=u10, v10=v10)
        out = step_fn(carry.replace(state=entry, history_diagnostics=history), jnp.asarray(step, jnp.int32))
        np.testing.assert_array_equal(np.asarray(out.history_diagnostics.u10), np.asarray(u10))
        ph_entry = f32(entry.ph_perturbation)
        ph_pre = f32(np.asarray(entry.ph_perturbation) + np.asarray(rk_dph))
        ph = ph_pre.copy()
        ph[:, 0], ph[:, -1], ph[:, :, 0], ph[:, :, -1] = ph_entry[:, 0], ph_entry[:, -1], ph_entry[:, :, 0], ph_entry[:, :, -1]
        args = (expected, step in (1, 4), f32(u10), f32(v10), np.asarray(rain))
        ref = _reference_step(*args, f32(u), f32(v), ph, phb, f32(grid.terrain_height), nml.dt_s)
        sensitive["final_ph"] |= not np.array_equal(
            ref, _reference_step(*args, f32(u), f32(v), f32(out.state.ph_perturbation), phb,
                                 f32(grid.terrain_height), nml.dt_s))
        sensitive["final_uv"] |= not np.array_equal(
            ref, _reference_step(*args, f32(out.state.u), f32(out.state.v), ph, phb, f32(grid.terrain_height),
                                 nml.dt_s))
        got = np.asarray(out.afwa_diagnostics.wspd10max)
        np.testing.assert_array_equal(got.view(np.int32), ref.view(np.int32), err_msg=f"step {step}")
        expected, carry = ref, out
    assert all(sensitive.values()), sensitive  # the fixture distinguishes the operand time levels


def test_fix8_cuda_clock_is_one_int(monkeypatch):
    captured = {}

    def fake_call(kernel, **kw):
        def run(*args):
            captured["clock"] = args[-1]
            return jnp.zeros(kw["out_shape"].shape, jnp.float32)
        return run

    monkeypatch.setattr(afwa.pl, "pallas_call", fake_call)
    z = lambda *s: jnp.zeros(s, jnp.float32)
    kw = dict(u10=z(2, 3), v10=z(2, 3), raincv=z(2, 3), rainncv=z(2, 3), u=z(4, 2, 4), v=z(4, 3, 3),
              ph=z(5, 2, 3), phb=z(5, 2, 3), terrain=z(2, 3), itimestep=1, dt=54, history_interval=60)
    afwa.update_afwa(afwa.AfwaDiagnostics(z(2, 3)), interpret=False, **kw)
    assert captured["clock"].shape == (1,)
    afwa.update_afwa(afwa.AfwaDiagnostics(z(2, 3)), interpret=True, **kw)
    assert captured["clock"].shape == (2,)


@pytest.mark.parametrize("fmt", ["pickle", "netcdf", "native"])
def test_delta1_restart_requires_history_carrier_schema(case, tmp_path, fmt):
    """A dict with correctly shaped U10/V10 is not the HistoryDiagnostics carrier (critic Delta 1 FIX 1)."""
    grid, nl, carry = case
    h = carry.history_diagnostics
    bad = carry.replace(history_diagnostics={"u10": h.u10, "v10": h.v10})
    if fmt == "native":
        from gpuwrf.runtime.restart_store import RestartStore
        from test_online_diagnostics_restart_store import identity, rewrite
        store = RestartStore(tmp_path / "store", 10**7, 2, 0)
        with pytest.raises(ValueError, match="HistoryDiagnostics carrier"):
            store.save({"d01": bad}, {"d01": 2}, {"d01": nl.dt_s}, identity(nl))
        gen, _ = store.save({"d01": carry}, {"d01": 2}, {"d01": nl.dt_s}, identity(nl))
        rewrite(gen, lambda payload, manifest: payload["extra_fields"].__setitem__(
            "history_diagnostics", {"u10": np.asarray(h.u10), "v10": np.asarray(h.v10)}))
        with pytest.raises(ValueError, match="HistoryDiagnostics carrier"):
            store.read(gen, expected_identity=identity(nl))
        return
    with pytest.raises(ValueError, match="HistoryDiagnostics carrier"):
        _save(tmp_path / "w", fmt, grid, nl, bad)  # refused before publication
    path = tmp_path / fmt
    _save(path, fmt, grid, nl, carry)
    if fmt == "pickle":
        payload = pickle.loads(path.read_bytes())
        payload["carry"]["extra_fields"]["history_diagnostics"] = {"u10": np.asarray(h.u10), "v10": np.asarray(h.v10)}
        path.write_bytes(pickle.dumps(payload))
        with pytest.raises(ValueError, match="HistoryDiagnostics carrier"):
            rp.read_restart(path, namelist=nl)
    else:
        restored = nc.read_wrfrst_carry(path, namelist=nl)[0]
        from gpuwrf.runtime.operational_state import HistoryDiagnostics
        assert isinstance(restored.history_diagnostics, HistoryDiagnostics)
        import base64
        as_dict = jax.tree.structure({k: v for k, v in h._asdict().items() if v is not None})
        assert as_dict.num_leaves == jax.tree.structure(h).num_leaves
        with Dataset(path, "a") as ds:  # same leaves, plain-dict tree: refused on restoration
            ds.setncattr("GPUWRF_TREE_HISTORY_DIAGNOSTICS", base64.b64encode(pickle.dumps(as_dict, protocol=5)).decode())
        with pytest.raises(ValueError, match="HistoryDiagnostics carrier"):
            nc.read_wrfrst_carry(path, namelist=nl)
