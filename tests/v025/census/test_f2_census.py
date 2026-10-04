"""F2 Phase A: resident phase classification and guard water budget (regression, not WRF fidelity)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.diagnostics import census as cs
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry

_spec = importlib.util.spec_from_file_location("census_test_helpers", Path(__file__).with_name("test_census.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
assert_fields_exact, fixture_state = _helpers.assert_fields_exact, _helpers.fixture_state

# Rain01 d02 own step 219 (WN3_B35_01): RK qc -0.0069547 restored to the +0.0069547 origin.
RAIN01_QC = 0.0069547


def _namelist(grid, disable_guards=False):
    return SimpleNamespace(dt_s=54, disable_guards=disable_guards, run_boundary=True, run_physics=False,
        boundary_config=SimpleNamespace(nested_frozen_wrf_boundary_bundle=False),
        metrics=grid.metrics, grid=grid, lower_boundary=None, force_fp64=True,
        acoustic_precision_mode="fp64_default")


def _step(monkeypatch, grid, origin, rk_state, post_rk=None, boundary=None, disable_guards=False):
    """Real guard path; RK returns ``rk_state``, ``post_rk``/``boundary`` edit only their phase."""
    monkeypatch.setattr(om, "_physics_step_forcing", lambda carry, *a, **k:
                        om._PhysicsStepForcing(carry.state, carry, om.DryPhysicsTendencies(), False))
    monkeypatch.setattr(om, "_rk_scan_step", lambda carry, *a, **k: carry.replace(state=rk_state))
    monkeypatch.setattr(om, "apply_lateral_boundaries",
                        lambda state, *a, **k: state if boundary is None else boundary(state))
    monkeypatch.setattr(om, "_enforce_operational_precision", lambda state, **k: state)
    monkeypatch.setattr(om, "_microphysics_wrf_order_enabled", lambda: post_rk is not None)
    monkeypatch.setattr(om, "_apply_post_rk_microphysics",
                        lambda state, namelist: state if post_rk is None else post_rk(state))
    namelist = _namelist(grid, disable_guards)
    return jax.jit(lambda c, step: om._physics_boundary_step_with_limiter_diagnostics(
        c, namelist, step, run_radiation=False)[0])


def _origin():
    grid, state = fixture_state()
    origin = state.replace(mu_total=jnp.full_like(state.mu_total, 90000.0),
                           mu_perturbation=jnp.zeros_like(state.mu_perturbation),
                           qc=jnp.full_like(state.qc, RAIN01_QC), Nr=jnp.full_like(state.Nr, 1000.0))
    return grid, origin


def _event(census, phase, field, event):
    index = (cs.F2_PHASES.index(phase), cs.F2_FIELDS.index(field), cs.F2_EVENTS.index(event))
    return (int(census.f2_counts[index]), int(census.f2_first_step[index]),
            int(census.f2_first_index[index]), float(census.f2_first_value[index]))


def test_rk_birth_and_post_rk_birth_are_separate_phases_and_state_is_unchanged(monkeypatch):
    grid, origin = _origin()
    origin = origin.replace(Ng=origin.Ng.at[3, 3, 3].set(-1.0))
    rk = origin.replace(Nr=origin.Nr.at[1, 2, 3].set(-3.36e-21))
    run = _step(monkeypatch, grid, origin, rk, post_rk=lambda s: s.replace(Nr=s.Nr.at[0, 0, 1].set(-5.0)))
    carry = initial_operational_carry(origin)
    off = run(carry, jnp.asarray(83))
    on = run(carry.replace(census=cs.initial_census()), jnp.asarray(83))
    assert_fields_exact(off, on.replace(census=None))
    rk_flat = int(np.ravel_multi_index((1, 2, 3), origin.Nr.shape))
    post_flat = int(np.ravel_multi_index((0, 0, 1), origin.Nr.shape))
    assert _event(on.census, "step_entry", "Nr", "negative")[0] == 0
    assert _event(on.census, "step_entry", "Ng", "negative")[:3] == (1, 83, origin.Ng.size - 1)
    count, step, flat, value = _event(on.census, "post_rk", "Nr", "negative")
    assert (count, step, flat) == (1, 83, rk_flat) and value == np.float32(-3.36e-21)
    count, step, flat, value = _event(on.census, "pre_dynamics_guard", "Nr", "negative")
    assert (count, step, flat, value) == (2, 83, post_flat, -5.0)
    assert _event(on.census, "pre_boundary_guard", "Nr", "negative") == (2, 83, post_flat, -5.0)
    # Numbers carry no water; only the six moisture fields enter the budget.
    assert not np.asarray(on.census.water_changes).any()


def test_first_event_persists_and_counts_accumulate_inactive_numbers_ignored(monkeypatch):
    grid, origin = _origin()
    inactive = ("Nh", "nwfa", "nifa")
    assert all(getattr(origin, name) is None for name in inactive)
    rk = origin.replace(qr=origin.qr.at[0, 1, 1].set(jnp.nan).at[2, 0, 0].set(0.06))
    run = _step(monkeypatch, grid, origin, rk)
    first = run(initial_operational_carry(origin).replace(census=cs.initial_census()), jnp.asarray(5))
    second = run(first.replace(state=origin), jnp.asarray(6))
    nan_flat = int(np.ravel_multi_index((0, 1, 1), origin.qr.shape))
    cap_flat = int(np.ravel_multi_index((2, 0, 0), origin.qr.shape))
    count, step, flat, value = _event(second.census, "post_rk", "qr", "nonfinite")
    assert (count, step, flat) == (2, 5, nan_flat) and np.isnan(value)
    count, step, flat, value = _event(second.census, "post_rk", "qr", "above_moisture_cap")
    assert (count, step, flat, value) == (2, 5, cap_flat, np.float32(0.06))
    for field in inactive:
        f = cs.F2_FIELDS.index(field)
        assert not np.asarray(second.census.f2_counts[:, f]).any()
        assert (np.asarray(second.census.f2_first_step[:, f]) == -1).all()
    # Legacy guard counters keep their meaning next to F2.
    guards = np.asarray(second.census.guards[cs.GUARDS.index("dynamics.qr")])
    np.testing.assert_array_equal(guards, [2, 2, 4])


def test_rain01_qc_restore_records_created_water_with_wrf_dry_mass(monkeypatch):
    grid, origin = _origin()
    rk = origin.replace(qc=origin.qc.at[1, 2, 3].set(-RAIN01_QC).at[0, 0, 0].set(0.06))
    run = _step(monkeypatch, grid, origin, rk)
    on = run(initial_operational_carry(origin).replace(census=cs.initial_census()), jnp.asarray(219))
    assert on.state.qc[1, 2, 3] == origin.qc[1, 2, 3]  # the guard restores the origin (water creation)
    mass = np.asarray(cs.guard_dry_mass_kg(on.state, _namelist(grid)))
    metrics = grid.metrics
    expected_mass = ((float(metrics.c1h[1]) * 90000.0 + float(metrics.c2h[1])) * -float(metrics.dnw[1])
                     / 9.81 * 3000.0 * 3000.0 / float(metrics.msftx[2, 3] * metrics.msfty[2, 3]))
    np.testing.assert_allclose(mass[1, 2, 3], expected_mass, rtol=1e-6)
    added, removed = np.asarray(on.census.water_changes[cs.F2_REPAIRS.index("dynamics"),
                                                        cs.MOISTURE_FIELDS.index("qc")])
    np.testing.assert_allclose(added, 2 * RAIN01_QC * mass[1, 2, 3], rtol=1e-6)
    np.testing.assert_allclose(removed, (0.06 - RAIN01_QC) * mass[0, 0, 0], rtol=1e-6)
    # Unit-mass form of the registered Rain01 number: +0.0139094 kg/kg.
    unit = cs.count_water_repair(cs.initial_census(), "dynamics", rk, on.state, jnp.ones_like(origin.qc))
    np.testing.assert_allclose(unit.water_changes[0, cs.MOISTURE_FIELDS.index("qc"), 0],
                               2 * RAIN01_QC, rtol=1e-6)
    others = np.delete(np.asarray(on.census.water_changes), cs.MOISTURE_FIELDS.index("qc"), axis=1)
    assert not others.any() and not np.asarray(on.census.unknown_water_changes).any()


def test_nonfinite_repair_has_undefined_budget_counted_separately(monkeypatch):
    grid, origin = _origin()
    rk = origin.replace(qv=origin.qv.at[0, 0, 0].set(jnp.inf))
    run = _step(monkeypatch, grid, origin, rk)
    on = run(initial_operational_carry(origin).replace(census=cs.initial_census()), jnp.asarray(1))
    q = cs.MOISTURE_FIELDS.index("qv")
    assert int(on.census.unknown_water_changes[cs.F2_REPAIRS.index("dynamics"), q]) == 1
    assert not np.asarray(on.census.water_changes[:, q]).any()


def test_boundary_qv_restore_is_budgeted_in_its_own_phase(monkeypatch):
    grid, origin = _origin()
    run = _step(monkeypatch, grid, origin, origin,
                boundary=lambda s: s.replace(qv=s.qv.at[0, 1, 2].set(-0.001)))
    on = run(initial_operational_carry(origin).replace(census=cs.initial_census()), jnp.asarray(3))
    assert _event(on.census, "pre_boundary_guard", "qv", "negative")[:3] == (
        1, 3, int(np.ravel_multi_index((0, 1, 2), origin.qv.shape)))
    assert _event(on.census, "pre_dynamics_guard", "qv", "negative")[0] == 0
    mass = np.asarray(cs.guard_dry_mass_kg(on.state, _namelist(grid)))
    q = cs.MOISTURE_FIELDS.index("qv")
    added, removed = np.asarray(on.census.water_changes[cs.F2_REPAIRS.index("boundary"), q])
    np.testing.assert_allclose(added, (float(origin.qv[0, 1, 2]) + 0.001) * mass[0, 1, 2], rtol=1e-5)
    assert removed == 0 and not np.asarray(on.census.water_changes[cs.F2_REPAIRS.index("dynamics")]).any()


def test_disabled_census_adds_no_operations(monkeypatch):
    grid, origin = _origin()
    rk = origin.replace(qc=origin.qc.at[0, 0, 0].set(-1.0))
    carry = initial_operational_carry(origin)
    real = _step(monkeypatch, grid, origin, rk).lower(carry, jnp.asarray(1)).as_text()
    monkeypatch.setattr(cs, "observe_f2_state", lambda census, *a, **k: census)
    monkeypatch.setattr(cs, "count_water_repair", lambda census, *a, **k: census)
    monkeypatch.setattr(cs, "guard_dry_mass_kg", lambda *a, **k: pytest.fail("mass weights traced"))
    stubbed = _step(monkeypatch, grid, origin, rk).lower(carry, jnp.asarray(1)).as_text()
    assert real == stubbed
    assert not jax.make_jaxpr(lambda s: cs.observe_f2_state(None, "post_rk", s, 1))(origin).jaxpr.eqns


def test_segment_manifest_reports_f2_events_with_index_and_schema(tmp_path, monkeypatch):
    import jsonschema
    grid, origin = _origin()
    rk = origin.replace(qc=origin.qc.at[1, 2, 3].set(-RAIN01_QC), Nr=origin.Nr.at[2, 1, 0].set(jnp.nan))
    run = _step(monkeypatch, grid, origin, rk)
    on = run(initial_operational_carry(origin).replace(census=cs.initial_census()), jnp.asarray(7))
    namelist = SimpleNamespace(dt_s=54, acoustic_substeps=10, rk_order=3,
        radiation_cadence_steps=33, cumulus_cadence_steps=6)
    writer = SimpleNamespace(written={"d02": ["out1"]})
    cs.write_segment(tmp_path, {"d02": on}, {"d02": SimpleNamespace(namelist=namelist)},
                     {"d02": 1}, writer, {"d02": 67})
    payload = json.loads((tmp_path / "census.json").read_text())
    jsonschema.validate(payload, json.loads(Path(cs.__file__).with_name("census_schema.json").read_text()))
    f2 = payload["domains"]["d02"]["f2"]
    events = {(e["phase"], e["field"], e["event"]): e for e in f2["events"]}
    qc = events["post_rk", "qc", "negative"]
    assert (qc["count"], qc["first_step"], qc["first_index_kji"]) == (1, 7, [1, 2, 3])
    nr = events["pre_dynamics_guard", "Nr", "nonfinite"]
    assert nr["first_index_kji"] == [2, 1, 0] and nr["first_value"] == "nan"
    assert f2["guard_water"]["dynamics"]["qc"]["added_kg"] > 0
    assert all(e["count"] > 0 for e in f2["events"])


def test_strict_guards_flag_resolves_into_namelist(monkeypatch):
    from gpuwrf.contracts import state as state_contract
    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices()[0])
    grid, _ = _origin()
    monkeypatch.delenv("GPUWRF_STRICT_GUARDS", raising=False)
    assert om.OperationalNamelist.from_grid(grid).disable_guards is True  # Phase C default
    monkeypatch.setenv("GPUWRF_STRICT_GUARDS", "0")
    assert om.OperationalNamelist.from_grid(grid).disable_guards is False
    monkeypatch.setenv("GPUWRF_STRICT_GUARDS", "1")
    assert om.OperationalNamelist.from_grid(grid).disable_guards is True
    assert om.OperationalNamelist.from_grid(grid, disable_guards=False).disable_guards is False


def test_strict_mode_keeps_candidates_unrepaired_and_still_counts_them(monkeypatch):
    grid, origin = _origin()
    rk = origin.replace(qc=origin.qc.at[1, 2, 3].set(-RAIN01_QC))
    bdy = lambda s: s.replace(qv=s.qv.at[0, 1, 2].set(-0.001))
    run = _step(monkeypatch, grid, origin, rk, boundary=bdy, disable_guards=True)
    on = run(initial_operational_carry(origin).replace(census=cs.initial_census()), jnp.asarray(219))
    assert on.state.qc[1, 2, 3] == rk.qc[1, 2, 3] and float(on.state.qv[0, 1, 2]) == -0.001
    assert _event(on.census, "pre_dynamics_guard", "qc", "negative")[:2] == (1, 219)
    assert _event(on.census, "pre_boundary_guard", "qv", "negative")[:2] == (1, 219)
    guards = np.asarray(on.census.guards)
    assert guards[cs.GUARDS.index("dynamics.qc")].tolist() == [0, 1, 1]
    assert guards[cs.GUARDS.index("boundary.qv")].tolist() == [0, 1, 1]
    assert not np.asarray(on.census.water_changes).any()
