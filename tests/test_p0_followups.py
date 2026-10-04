"""P0 critic follow-ups (critic e7105fee4; manager 0937cd993 + ab7422481).

F1  KF receives WRF phy_prep TOTAL density rho = (1+qv)/alt (grid-backed), EOS total density without metrics.
F2  controls the gate-1 adapter tests could not see: (a) the operational dispatcher ``_physics_step_forcing`` routes
    mp=8 and cu=1 through the P0 coupling; (b) an fp32 production-dtype State keeps fp32 through the adapters and
    matches WRF REAL*4 order bitwise (eager), so an adapter-level fp64 slip is caught.
USE_THETA_M=0 (kept supported, not fail-closed): every case-builder ingest carries WRF's dry option-0 prognostic
    as State theta_m by the exact change of variables, so the P0 physics view recovers WRF's dry theta; the residual
    option-0 vs option-1 finish-form difference is bounded and recorded.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import pathlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.coupling import scan_adapters as sa
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Gate-1 helpers loaded by FILE PATH from this test's own directory (the snapshot), never through a bare ``tests``
# namespace package (not importable from the isolated regression entry; P0REG r1 incident cf8ca43df).
_LP_SPEC = importlib.util.spec_from_file_location(
    "_p0_live_path_helpers", pathlib.Path(__file__).resolve().parent / "test_p0_moist_theta_live_path.py")
_lp = importlib.util.module_from_spec(_LP_SPEC)
_LP_SPEC.loader.exec_module(_lp)
RVRD, _dry, _exner_port, _grid, _state, _wrf_phy_prep_rho = (
    _lp.RVRD, _lp._dry, _lp._exner_port, _lp._grid, _lp._state, _lp._wrf_phy_prep_rho)


def _kf_spy(record):
    """KF kernel stub.  kf_adapter vmaps the kernel, so inputs are tracers: like gate 1, T0 and RHOE are carried
    out through the qc/qr tendency channels (read back with _kf_channels); only a trace-time flag is recorded."""
    def spy(T0, QV0, P0_, DZQ, RHOE, w0a, U0, V0, dt, dx, *, w=None, nca=None, stepcu=5, cudt=0.0):
        record["called"] = True
        z = jnp.zeros_like(T0)
        tend = {"theta": z + 2.0e-3, "qv": z - 3.0e-6, "qc": T0, "qr": RHOE, "qi": z, "qs": z}
        return SimpleNamespace(
            tendency=SimpleNamespace(state_tendencies=tend, accumulator_increments={"rainc_acc": jnp.zeros(())}),
            carry=SimpleNamespace(cumulus={"w0avg": w0a, "nca": nca}),
            diagnostics=SimpleNamespace(cumulus={"pratec": jnp.zeros(())}),
        )
    return spy


def _kf_channels(before, after, dt):
    """(T0, RHOE) seen by the KF kernel, in (nz, ny, nx) layout, from the qc/qr channels of the direct update."""
    t = (np.asarray(after.qc, np.float64) - np.asarray(before.qc, np.float64)) / dt
    r = (np.asarray(after.qr, np.float64) - np.asarray(before.qr, np.float64)) / dt
    return t, r


# ------------------------------------------------------------------ F1: KF density
def test_kf_rho_is_wrf_phy_prep_total_density(monkeypatch):
    grid = _grid()
    state = _state(grid, 0.012)
    rec = {}
    monkeypatch.setattr(sa, "step_kf_column", _kf_spy(rec))
    out, _, _ = sa.kf_adapter(state, 54.0, jnp.zeros((grid.nz, grid.ny, grid.nx)), jnp.zeros((grid.ny, grid.nx)),
                              grid=grid)
    _, rho = _kf_channels(state, out, 54.0)
    want = _wrf_phy_prep_rho(state, grid.metrics)
    np.testing.assert_allclose(rho, want, rtol=1e-6, atol=0)
    # named negative control: the dry phy_prep density 1/alt differs by exactly the (1+qv) factor (~1.2 % here)
    qv = np.asarray(state.qv, np.float64)
    ratio = rho / (want / (1.0 + qv))
    np.testing.assert_allclose(ratio, 1.0 + qv, rtol=1e-6)
    assert ratio.max() > 1.01


def test_kf_rho_without_metrics_is_eos_total_density(monkeypatch):
    grid = _grid()
    state = _state(grid, 0.012)
    rec = {}
    monkeypatch.setattr(sa, "step_kf_column", _kf_spy(rec))
    out, _, _ = sa.kf_adapter(state, 54.0, jnp.zeros((grid.nz, grid.ny, grid.nx)), jnp.zeros((grid.ny, grid.nx)),
                              grid=None)
    _, rho = _kf_channels(state, out, 54.0)
    want = np.asarray(pc._rho_from_state(state), np.float64) * (1.0 + np.asarray(state.qv, np.float64))
    np.testing.assert_allclose(rho, want, rtol=1e-12, atol=0)


# ------------------------------------------------------------------ F2a: operational dispatcher path
def test_operational_dispatcher_routes_mp8_and_cu1_through_p0_coupling(monkeypatch):
    from gpuwrf.runtime.operational_mode import (
        OperationalNamelist, _apply_post_rk_microphysics, _microphysics_wrf_order_enabled, _physics_step_forcing)
    from gpuwrf.runtime.operational_state import initial_operational_carry

    grid = _grid()
    state = _state(grid, 0.012)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    tend = Tendencies(z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)), z((nz, ny, nx)),
                      z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)))
    nml = dataclasses.replace(
        OperationalNamelist.from_grid(grid, dt_s=18.0, tendencies=tend), time_utc="2026-07-26T00:00:00Z",
        run_physics=True, mp_physics=8, cu_physics=1, bl_pbl_physics=0, sf_sfclay_physics=0, ra_sw_physics=0,
        ra_lw_physics=0, use_noahmp=False)
    seen = {}

    def thompson_spy(column, dt, debug=False):   # identity microphysics: the exit must round-trip theta_m
        seen["mp_T"] = np.asarray(column.T, np.float64)
        zeros = jnp.zeros((ny, nx), jnp.float64)
        return column, {"rain": zeros, "snow": zeros, "graupel": zeros, "ice": zeros}

    rec = {}
    monkeypatch.setattr(pc, "step_thompson_column_with_precip", thompson_spy)
    monkeypatch.setattr(sa, "step_kf_column", _kf_spy(rec))
    forcing = _physics_step_forcing(initial_operational_carry(state), nml, 0.0, run_radiation=False)
    if _microphysics_wrf_order_enabled():   # WRF order: MP is the post-RK call (solve_em.F:3809)
        assert "mp_T" not in seen
        _apply_post_rk_microphysics(state, nml)
    T_dry = _dry(state) * _exner_port(state.p)
    np.testing.assert_allclose(seen["mp_T"], np.moveaxis(T_dry, 0, -1), rtol=1e-13, atol=0)   # _to_columns: (ny,nx,nz)
    assert rec.get("called"), "cu=1 was not dispatched to kf_adapter"
    t_kf, rho_kf = _kf_channels(state, forcing.state, 18.0)   # identity Thompson leaves qc/qr untouched
    np.testing.assert_allclose(t_kf, T_dry, rtol=1e-10, atol=0)
    np.testing.assert_allclose(rho_kf, _wrf_phy_prep_rho(state, grid.metrics), rtol=1e-6, atol=0)
    assert np.all(np.isfinite(np.asarray(forcing.state.theta)))


# ------------------------------------------------------------------ F2b: fp32 production-dtype adapters
def _state_fp32(qv_scale=0.012):
    """Production-gated fp32 theta/qv leaves (State.replace would re-cast to the existing leaf dtype, so the State is
    built directly, as the gate-2 compare builds its DEFAULT_DTYPES state)."""
    s = _state(_grid(), qv_scale)
    fields = {f: getattr(s, f) for f in _state_field_shapes(_grid())}
    fields.update(theta=jnp.asarray(s.theta, jnp.float32), qv=jnp.asarray(s.qv, jnp.float32))
    return State(**fields)


def test_thompson_adapter_fp32_state_matches_wrf_real4_bitwise(monkeypatch):
    s = _state_fp32()
    assert jnp.asarray(s.theta).dtype == jnp.float32 and jnp.asarray(s.qv).dtype == jnp.float32
    seen = {}

    def spy(column, dt, debug=False):
        seen["T"] = np.asarray(column.T)
        out = column.replace(T=column.T + 0.75, qv=column.qv - 1.0e-3)
        zeros = jnp.zeros(np.asarray(s.rain_acc).shape, jnp.float64)
        return out, {"rain": zeros, "snow": zeros, "graupel": zeros, "ice": zeros}

    monkeypatch.setattr(pc, "step_thompson_column_with_precip", spy)
    out = pc.thompson_adapter(s, 18.0)
    assert jnp.asarray(out.theta).dtype == jnp.float32 and jnp.asarray(out.qv).dtype == jnp.float32
    r = np.float32(461.6) / np.float32(287.0)
    th, q = np.asarray(s.theta, np.float32), np.asarray(s.qv, np.float32)
    th_o = th / (np.float32(1) + r * q)                                                      # phy_prep order
    q_n = np.asarray(out.qv, np.float32)
    th_n = np.asarray(pc._theta_from_temperature(pc._from_columns(jnp.asarray(seen["T"]) + 0.75), s.p, jnp.float32))
    wrf = th_o * (np.float32(1) + r * q) + (th_n - th_o) * (np.float32(1) + r * q_n) + r * (q_n - q) * th_n
    assert np.array_equal(np.asarray(out.theta, np.float32), wrf)


def test_kf_adapter_fp32_state_theta_update_bitwise(monkeypatch):
    s = _state_fp32()
    assert jnp.asarray(s.theta).dtype == jnp.float32
    rec = {}
    monkeypatch.setattr(sa, "step_kf_column", _kf_spy(rec))
    out, _, _ = sa.kf_adapter(s, 54.0, jnp.zeros(s.theta.shape), jnp.zeros(s.theta.shape[1:]), grid=_grid())
    assert jnp.asarray(out.theta).dtype == jnp.float32
    r = np.float32(461.6) / np.float32(287.0)
    th, q = np.asarray(s.theta, np.float32), np.asarray(s.qv, np.float32)
    fac = np.float32(1) + r * q
    tend = fac * np.float32(2.0e-3) + r * th / fac * np.float32(-3.0e-6)                     # conv_t_tendf_to_moist
    assert np.array_equal(np.asarray(out.theta, np.float32), th + np.float32(54.0) * tend)


# ------------------------------------------------------------------ USE_THETA_M=0 support
def test_use_theta_m0_every_case_builder_ingest_carries_theta_m():
    """Source contract: in d02_replay no theta recoupling ``*(1 + _RVOVRD*qv)`` is guarded by use_theta_m == 1.
    wrfinput/wrfout T is dry under both options; the wrfbdy T strip (moist under option 1) is the only
    option-dependent site, and there option 0 is the branch that recouples."""
    src = (ROOT / "src/gpuwrf/integration/d02_replay.py").read_text()
    tree = ast.parse(src)
    guarded = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if "use_theta_m" in test and ("== 1" in test):
                body = "\n".join(ast.unparse(b) for b in node.body)
                if "_RVOVRD" in body:
                    guarded.append(test)
    assert guarded == [], f"option-1-only theta recoupling remains: {guarded}"
    assert src.count("theta = theta * (1.0 + _RVOVRD * qv_initial)") == 2          # IC + live-nest option-0
    assert "if use_theta_m != 1:" in src


def test_use_theta_m0_dry_input_reaches_physics_dry_with_bounded_finish_gap(monkeypatch):
    """Option-0 IC theta_d is ingested as theta_m = theta_d (1+R qv): Thompson then sees WRF's option-0 dry T exactly.
    WRF option 0 updates dry theta directly (moist_physics_finish_em, use_theta_m=0: t = th_n - T0), while the
    port's single theta_m convention applies the option-1 finish; the decoupled result differs from th_n by the
    second-order term, bounded here (<= 2e-3 K for dT=0.75 K, dq=-1e-3), i.e. far inside the gate-2 0.03 K band."""
    grid = _grid()
    base = _state(grid, 0.012)
    theta_d = _dry(base)                       # WRF option-0 prognostic (dry)
    qv = np.asarray(base.qv, np.float64)
    state = base.replace(theta=jnp.asarray(theta_d * (1.0 + RVRD * qv)))   # the repaired ingest change of variables
    seen = {}

    def spy(column, dt, debug=False):
        seen["T"] = np.asarray(column.T, np.float64)
        zeros = jnp.zeros(np.asarray(state.rain_acc).shape, jnp.float64)
        return column.replace(T=column.T + 0.75, qv=column.qv - 1.0e-3), {"rain": zeros, "snow": zeros,
                                                                           "graupel": zeros, "ice": zeros}

    monkeypatch.setattr(pc, "step_thompson_column_with_precip", spy)
    out = pc.thompson_adapter(state, 18.0)
    exner = _exner_port(state.p)
    np.testing.assert_allclose(seen["T"], np.moveaxis(theta_d * exner, 0, -1), rtol=1e-13)       # _to_columns: (ny,nx,nz)
    q_n = np.asarray(out.qv, np.float64)
    th_n_wrf_opt0 = (theta_d * exner + 0.75) / exner
    gap = np.abs(np.asarray(out.theta, np.float64) / (1.0 + RVRD * q_n) - th_n_wrf_opt0)
    assert gap.max() <= 2.0e-3, gap.max()
