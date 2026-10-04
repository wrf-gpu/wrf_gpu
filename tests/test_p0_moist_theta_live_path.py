"""P0 gate 1: moist theta_m must reach Thompson/KF/GWD/rho as WRF dry thermodynamics.

Contract: ``.agent/sprints/2026-09-23-v0250-endgame-frontrunner/P0_MOIST_THETA_CONTRACT.md``.

The production step binds ``physics_couplers.thompson_adapter``, ``physics_couplers.gwdo_adapter`` and
``scan_adapters.kf_adapter`` (``runtime/operational_mode.py`` microphysics/GWD/cumulus slots). ``State.theta`` is
WRF ``use_theta_m=1`` moist theta ``theta_m = theta_dry * (1 + Rv/Rd * qv)``. WRF ``phy_prep``
(``module_big_step_utilities_em.F:4830-4855``) hands physics ``th_phy = theta_m / (1 + Rv/Rd qv)`` and
``t_phy = th_phy * pi``; ``moist_physics_finish_em`` and ``conv_t_tendf_to_moist`` recouple theta_m with the exact
forms below. These tests intercept each kernel at its import site in the coupler module, so they exercise the
exact production coupling code, and derive every expectation independently from those WRF formulas.

Exact WRF forms (manager review): Thompson exit follows ``moist_physics_finish_em`` (:5735-5741) with explicit old/new
terms, ``th_o(1+R q_o) + (th_n-th_o)(1+R q_n) + R (q_n-q_o) th_n``, which is NOT the full product ``th_n(1+R q_n)``
(they differ by ``R (th_n-th_o)(q_n-q_o)``); KF follows ``conv_t_tendf_to_moist`` (:6699-6703), the theta_m tendency
``(1+R q_o) rth + R th_o rqv`` at the OLD state (no ``dt^2 R rth rqv`` term). Named negative controls prove each
full-product mutant is separable from the WRF form on these fixtures.

Controls: qv = 0 (the conversion must be an exact identity, i.e. bitwise-unchanged coupling), a low-qv case, the
unaffected species leaves, and a jaxpr scan proving the coupling adds no host callback (no step-loop transfer).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.coupling import scan_adapters as sa

RVRD = 461.6 / 287.0            # WRF rvovrd = r_v / r_d (module_model_constants)
P0 = 100000.0


def _grid(ny=2, nx=3, nz=6) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain = TerrainProvenance(
        source_path="unit-test", sha256="unit-test", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0, coastline_sanity_check_passed=True,
    )
    metrics = DycoreMetrics.flat(ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0,
                                 provenance="unit-test-flat")
    return GridSpec(projection, terrain, VerticalCoord("hybrid_eta", nz, 5000.0, eta), BCMetadata("ideal", (), 1,
                    "linear", True), eta, jnp.zeros((ny, nx), dtype=jnp.float64), metrics=metrics)


def _state(grid: GridSpec, qv_scale: float) -> State:
    shapes = _state_field_shapes(grid)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    k = np.arange(nz, dtype=np.float64)[:, None, None]
    jj = np.arange(ny, dtype=np.float64)[None, :, None]
    ii = np.arange(nx, dtype=np.float64)[None, None, :]
    theta_dry = 288.0 + 3.0 * k + 0.1 * jj + 0.05 * ii                    # warm moist marine BL -> stable aloft
    qv = qv_scale * np.exp(-k / 3.0) * (1.0 + 0.02 * jj + 0.01 * ii)       # 0.012 kg/kg at the surface
    theta_m = theta_dry * (1.0 + RVRD * qv)
    p = 101000.0 - 9000.0 * k + 0.0 * jj + 0.0 * ii
    ph = (np.arange(nz + 1, dtype=np.float64)[:, None, None] * 800.0 * 9.80665) * np.ones((1, ny, nx))
    fields.update(
        theta=jnp.asarray(theta_m), qv=jnp.asarray(qv), p=jnp.asarray(p), p_total=jnp.asarray(p),
        ph=jnp.asarray(ph), ph_total=jnp.asarray(ph), qc=jnp.full((nz, ny, nx), 1.0e-5),
        qr=jnp.full((nz, ny, nx), 2.0e-6), u=jnp.full((nz, ny, nx + 1), 12.0), v=jnp.full((nz, ny + 1, nx), -3.0),
        mu=jnp.full((ny, nx), 90000.0), mu_total=jnp.full((ny, nx), 90000.0), t_skin=jnp.full((ny, nx), 291.0),
    )
    return State(**fields)


def _exner(p):
    return (np.maximum(np.asarray(p, np.float64), 1.0) / P0) ** (287.0 / 1004.0)


def _dry(state):
    theta_m = np.asarray(state.theta, np.float64)
    qv = np.asarray(state.qv, np.float64)
    return theta_m / (1.0 + RVRD * qv)


def _cols(a):
    return np.moveaxis(np.asarray(a, np.float64), 0, -1)


R_D_OVER_CP = float(pc.R_D_OVER_CP)


def _wrf_phy_prep_rho(state, metrics):
    """WRF phy_prep density rho = (1+qv)/alt (module_big_step_utilities_em.F:4856), alt rebuilt in REAL*4 from the
    hydrostatic faces as calc_p_rho_phi does: alt = dph / p_mid / log(p_down/p_up).  Transcribed independently."""
    f = np.float32
    mut = np.asarray(state.mu_total, f)
    ph = np.asarray(state.ph_total, f)
    qv = np.asarray(state.qv, f)
    c3h, c4h, c3f, c4f = (np.asarray(getattr(metrics, n), f) for n in ("c3h", "c4h", "c3f", "c4f"))
    p_top = f(np.asarray(metrics.p_top).reshape(()))
    p_up = c3f[1:, None, None] * mut[None] + c4f[1:, None, None] + p_top
    p_down = c3f[:-1, None, None] * mut[None] + c4f[:-1, None, None] + p_top
    p_mid = c3h[:, None, None] * mut[None] + c4h[:, None, None] + p_top
    alt = (ph[1:] - ph[:-1]) / p_mid / np.log(p_down / p_up)
    return ((f(1.0) + qv) / alt).astype(np.float64)


def _exner_port(p):
    return (np.maximum(np.asarray(p, np.float64), 1.0) / float(pc.P0_PA)) ** R_D_OVER_CP


# ------------------------------------------------------------------ Thompson (mp=8)
def _run_thompson(monkeypatch, state, dq=-1.0e-3, dT=0.75):
    seen = {}

    def spy(column, dt, debug=False):
        seen["T"] = np.asarray(column.T, np.float64)
        seen["rho"] = np.asarray(column.rho, np.float64)
        seen["qv"] = np.asarray(column.qv, np.float64)
        out = column.replace(T=column.T + dT, qv=column.qv + dq, qc=column.qc - dq)
        zeros = jnp.zeros(np.asarray(state.rain_acc).shape, jnp.float64)
        return out, {"rain": zeros, "snow": zeros, "graupel": zeros, "ice": zeros}

    monkeypatch.setattr(pc, "step_thompson_column_with_precip", spy)
    return seen, pc.thompson_adapter(state, 18.0)


@pytest.mark.parametrize("qv_scale", [0.012, 1.0e-6])
def test_thompson_gets_dry_temperature_and_recouples_theta_m(monkeypatch, qv_scale):
    grid = _grid()
    state = _state(grid, qv_scale)
    seen, out = _run_thompson(monkeypatch, state)
    exner = _exner_port(state.p)
    T_dry = _dry(state) * exner
    np.testing.assert_allclose(seen["T"], _cols(T_dry), rtol=1e-13, atol=0)
    rho_expected = 0.622 * _cols(state.p) / (287.04 * _cols(T_dry) * (_cols(state.qv) + 0.622))
    np.testing.assert_allclose(seen["rho"], rho_expected, rtol=1e-12, atol=0)
    # exit: WRF moist_physics_finish_em explicit old/new terms
    qv_old = np.asarray(state.qv, np.float64)
    qv_new = qv_old - 1.0e-3
    th_o = _dry(state)
    th_n = (T_dry + 0.75) / exner
    theta_m_new = _wrf_mp_finish(th_o, qv_old, th_n, qv_new)
    np.testing.assert_allclose(np.asarray(out.theta, np.float64), theta_m_new, rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.asarray(out.qv, np.float64), qv_new, rtol=0, atol=1e-18)


def _wrf_mp_finish(th_o, q_o, th_n, q_n):
    """WRF moist_physics_finish_em :5735-5741 (use_theta_m=1), transcribed independently."""
    return th_o * (1.0 + RVRD * q_o) + (th_n - th_o) * (1.0 + RVRD * q_n) + RVRD * (q_n - q_o) * th_n


def test_thompson_full_product_mutant_is_separable(monkeypatch):
    """Negative control: the full-product exit th_n(1+R q_n) must differ from WRF's form beyond the gate tolerance."""
    state = _state(_grid(), 0.012)
    seen, out = _run_thompson(monkeypatch, state)
    exner = _exner_port(state.p)
    th_o, q_o = _dry(state), np.asarray(state.qv, np.float64)
    th_n, q_n = (th_o * exner + 0.75) / exner, q_o - 1.0e-3
    product = th_n * (1.0 + RVRD * q_n)
    got = np.asarray(out.theta, np.float64)
    assert np.max(np.abs(product - _wrf_mp_finish(th_o, q_o, th_n, q_n)) / np.abs(got)) > 1e-7   # >> rtol 1e-13
    assert not np.allclose(got, product, rtol=1e-9, atol=0)


def test_thompson_moist_convention_matters_for_this_fixture(monkeypatch):
    """Negative control for the fixture itself: theta_m*pi differs from the dry T by more than 1 K."""
    state = _state(_grid(), 0.012)
    exner = _exner_port(state.p)
    gap = np.asarray(state.theta, np.float64) * exner - _dry(state) * exner
    assert gap.max() > 1.0


def test_thompson_qv_zero_is_bitwise_identity_of_the_old_coupling(monkeypatch):
    state = _state(_grid(), 0.0)
    seen, out = _run_thompson(monkeypatch, state, dq=0.0, dT=0.0)
    T_old = np.asarray(pc._temperature_from_theta(state.theta, state.p), np.float64)
    assert np.array_equal(seen["T"], _cols(T_old))
    theta_old = np.asarray(pc._theta_from_temperature(jnp.asarray(T_old), state.p, jnp.float64), np.float64)
    assert np.array_equal(np.asarray(out.theta, np.float64), theta_old)
    for name in ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng"):
        assert np.array_equal(np.asarray(getattr(out, name)), np.asarray(getattr(state, name))), name


def test_thompson_aero_mp28_gets_dry_temperature_and_wrf_finish(monkeypatch):
    """mp=28 aerosol-aware Thompson is dispatched by operational_mode (mp_opt == 28): same P0 contract as mp=8."""
    state = _state(_grid(), 0.012)
    seen = {}

    def spy(column, dt, debug=False):
        seen["T"] = np.asarray(column.T, np.float64)
        seen["rho"] = np.asarray(column.rho, np.float64)
        out = column.replace(T=column.T + 0.75, qv=column.qv - 1.0e-3, qc=column.qc + 1.0e-3)
        zeros = jnp.zeros(np.asarray(state.rain_acc).shape, jnp.float64)
        return out, {"rain": zeros, "snow": zeros, "graupel": zeros, "ice": zeros}

    monkeypatch.setattr(pc, "step_thompson_aero_column_with_precip", spy)
    monkeypatch.setattr(pc, "apply_surface_aerosol_emission", lambda out, nwfa2d, nifa2d, dt: out)
    out = pc.thompson_aero_adapter(state, 18.0)
    exner = _exner_port(state.p)
    T_dry = _dry(state) * exner
    np.testing.assert_allclose(seen["T"], _cols(T_dry), rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        seen["rho"], 0.622 * _cols(state.p) / (287.04 * _cols(T_dry) * (_cols(state.qv) + 0.622)), rtol=1e-12, atol=0)
    q_o = np.asarray(state.qv, np.float64)
    expected = _wrf_mp_finish(_dry(state), q_o, (T_dry + 0.75) / exner, q_o - 1.0e-3)
    np.testing.assert_allclose(np.asarray(out.theta, np.float64), expected, rtol=1e-13, atol=0)


def test_rho_from_state_uses_dry_temperature():
    state = _state(_grid(), 0.012)
    T_dry = _dry(state) * _exner_port(state.p)
    expected = 0.622 * np.asarray(state.p) / (287.04 * T_dry * (np.asarray(state.qv) + 0.622))
    np.testing.assert_allclose(np.asarray(pc._rho_from_state(state), np.float64), expected, rtol=1e-12, atol=0)


# ------------------------------------------------------------------ KF (cu=1)
def test_kf_gets_dry_temperature_and_recouples_theta_m(monkeypatch):
    grid = _grid()
    state = _state(grid, 0.012)
    rth, rqv = 2.0e-3, -3.0e-6

    def spy(T0, QV0, P0_, DZQ, RHOE, w0a, U0, V0, dt, dx, *, w=None, nca=None, stepcu=5, cudt=0.0):
        z = jnp.zeros_like(T0)
        tend = {"theta": z + rth, "qv": z + rqv, "qc": T0, "qr": RHOE, "qi": z, "qs": z}
        return SimpleNamespace(
            tendency=SimpleNamespace(state_tendencies=tend, accumulator_increments={"rainc_acc": jnp.zeros(())}),
            carry=SimpleNamespace(cumulus={"w0avg": w0a, "nca": nca}),
            diagnostics=SimpleNamespace(cumulus={"pratec": jnp.zeros(())}),
        )

    monkeypatch.setattr(sa, "step_kf_column", spy)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    dt = 54.0                                   # d01 step: the full-product dt^2 cross term is visible
    out, _, _ = sa.kf_adapter(state, dt, jnp.zeros((nz, ny, nx)), jnp.zeros((ny, nx)), grid=grid)
    exner = _exner_port(state.p)
    theta_dry = _dry(state)
    T_seen = (np.asarray(out.qc, np.float64) - np.asarray(state.qc, np.float64)) / dt  # qc channel carried T0
    np.testing.assert_allclose(T_seen, theta_dry * exner, rtol=1e-12, atol=0)
    rho_seen = (np.asarray(out.qr, np.float64) - np.asarray(state.qr, np.float64)) / dt  # qr channel carried RHOE
    # critic F1: KF gets WRF phy_prep TOTAL density (1+qv)/alt (module_big_step_utilities_em.F:4856), not the dry
    # EOS density.
    np.testing.assert_allclose(rho_seen, _wrf_phy_prep_rho(state, grid.metrics), rtol=1e-6, atol=0)
    # WRF conv_t_tendf_to_moist: theta_m tendency at the OLD state, applied over dt by the direct update
    qv_old = np.asarray(state.qv, np.float64)
    wrf = np.asarray(state.theta, np.float64) + dt * ((1.0 + RVRD * qv_old) * rth + RVRD * theta_dry * rqv)
    got = np.asarray(out.theta, np.float64)
    np.testing.assert_allclose(got, wrf, rtol=1e-13, atol=0)
    # named negative control: the full-product mutant (theta_dry + dt rth)(1 + R (qv + dt rqv)) is separable
    product = (theta_dry + dt * rth) * (1.0 + RVRD * (qv_old + dt * rqv))
    assert np.max(np.abs(product - wrf) / np.abs(wrf)) > 1e-8                        # >> rtol 1e-13
    assert not np.allclose(got, product, rtol=1e-10, atol=0)


def test_kf_qv_zero_is_bitwise_identity_of_the_old_update(monkeypatch):
    grid = _grid()
    state = _state(grid, 0.0)
    rth = 2.0e-3

    def spy(T0, QV0, P0_, DZQ, RHOE, w0a, U0, V0, dt, dx, *, w=None, nca=None, stepcu=5, cudt=0.0):
        z = jnp.zeros_like(T0)
        tend = {"theta": z + rth, "qv": z, "qc": z, "qr": z, "qi": z, "qs": z}
        return SimpleNamespace(
            tendency=SimpleNamespace(state_tendencies=tend, accumulator_increments={"rainc_acc": jnp.zeros(())}),
            carry=SimpleNamespace(cumulus={"w0avg": w0a, "nca": nca}),
            diagnostics=SimpleNamespace(cumulus={"pratec": jnp.zeros(())}),
        )

    monkeypatch.setattr(sa, "step_kf_column", spy)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    out, _, _ = sa.kf_adapter(state, 54.0, jnp.zeros((nz, ny, nx)), jnp.zeros((ny, nx)), grid=grid)
    old = np.asarray((state.theta + 54.0 * jnp.full(state.theta.shape, rth)).astype(state.theta.dtype))
    assert np.array_equal(np.asarray(out.theta), old)


# ------------------------------------------------------------------ GWD (gwd_opt=1)
def test_gwdo_gets_dry_temperature(monkeypatch):
    grid = _grid()
    state = _state(grid, 0.012)
    seen = {}

    def spy(column, statics, dt):
        seen["t1"] = np.asarray(column.t1, np.float64)
        z = jnp.zeros_like(column.t1)
        surface = jnp.zeros((column.t1.shape[0],), dtype=column.t1.dtype)
        return SimpleNamespace(rublten=z, rvblten=z, dtaux3d=z, dtauy3d=z,
                               dusfcg=surface, dvsfcg=surface)

    monkeypatch.setattr(pc, "gwdo_columns", spy)
    pc.gwdo_adapter(state, 54.0, None, grid)
    ny, nx = grid.ny, grid.nx
    expected = _cols(_dry(state) * _exner_port(state.p)).reshape(ny * nx, -1)
    np.testing.assert_allclose(seen["t1"], expected, rtol=1e-13, atol=0)


# ------------------------------------------------------------------ no step-loop transfer
def test_thompson_coupling_adds_no_host_callback(monkeypatch):
    state = _state(_grid(), 0.012)

    def spy(column, dt, debug=False):
        zeros = jnp.zeros(np.asarray(state.rain_acc).shape, jnp.float64)
        return column, {"rain": zeros, "snow": zeros, "graupel": zeros, "ice": zeros}

    monkeypatch.setattr(pc, "step_thompson_column_with_precip", spy)
    jaxpr = str(jax.make_jaxpr(lambda s: pc.thompson_adapter(s, 18.0))(state))
    for primitive in ("callback", "io_callback", "pure_callback", "debug_callback", "host_callback"):
        assert primitive not in jaxpr, primitive


# ------------------------------------------------------------------ fp32 carry: WRF REAL*4 order, no fp64 island
def _state32(qv_scale=0.012):
    """Minimal fp32 carry view: State.replace re-casts to the frozen dtype contract, so the helpers (which read only
    ``theta``/``qv``) get plain fp32 arrays here -- the prospective full-fp32 carry."""
    s = _state(_grid(), qv_scale)
    return SimpleNamespace(theta=jnp.asarray(s.theta, jnp.float32), qv=jnp.asarray(s.qv, jnp.float32),
                           p=jnp.asarray(s.p, jnp.float32))


def test_dry_view_fp32_carry_matches_wrf_real4_bitwise():
    s = _state32()
    r = np.float32(461.6) / np.float32(287.0)                      # WRF REAL*4 R_v/R_d
    th = np.asarray(s.theta, np.float32)
    q = np.asarray(s.qv, np.float32)
    wrf = th / (np.float32(1.0) + r * q)                            # phy_prep :4834 order, fp32
    got = np.asarray(pc._dry_theta_view(s))
    assert got.dtype == np.float32 and np.array_equal(got, wrf)


def test_finish_and_tendency_fp32_match_wrf_real4_bitwise():
    s = _state32()
    r = np.float32(461.6) / np.float32(287.0)
    th_o = np.asarray(pc._dry_theta_view(s), np.float32)
    q_o = np.asarray(s.qv, np.float32)
    th_n = th_o + np.float32(0.75)
    q_n = q_o - np.float32(1.0e-3)
    mpten, qvten = th_n - th_o, q_n - q_o
    wrf = th_o * (np.float32(1) + r * q_o) + mpten * (np.float32(1) + r * q_n) + r * qvten * th_n   # :5735-5737
    got = np.asarray(pc._theta_m_from_microphysics(th_o, q_o, th_n, q_n, jnp.float32))
    assert got.dtype == np.float32 and np.array_equal(got, wrf)
    rth = np.full_like(th_o, np.float32(2.0e-3))
    rqv = np.full_like(th_o, np.float32(-3.0e-6))
    thm = np.asarray(s.theta, np.float32)
    fac = np.float32(1) + r * q_o
    wrf_t = fac * rth + r * thm / fac * rqv                          # conv_t_tendf_to_moist :6701-6702 order
    got_t = np.asarray(pc._theta_m_tendency_from_dry(rth, rqv, thm, q_o, jnp.float32))
    assert got_t.dtype == np.float32 and np.array_equal(got_t, wrf_t)


def test_helpers_have_no_fp64_island_on_fp32_carry():
    s = _state32()
    th = pc._dry_theta_view(s)
    jaxprs = [
        str(jax.make_jaxpr(lambda t, q: pc._dry_theta_view(SimpleNamespace(theta=t, qv=q)))(s.theta, s.qv)),
        str(jax.make_jaxpr(lambda a, b, c, d: pc._theta_m_from_microphysics(a, b, c, d, jnp.float32))(th, s.qv, th, s.qv)),
        str(jax.make_jaxpr(lambda a, b, c, d: pc._theta_m_tendency_from_dry(a, b, c, d, jnp.float32))(th, th, s.theta, s.qv)),
    ]
    for text in jaxprs:
        assert "f64" not in text and "float64" not in text, text[:400]


def _ulp(a, b):
    return np.abs(np.asarray(a, np.float32).view(np.int32).astype(np.int64)
                  - np.asarray(b, np.float32).view(np.int32).astype(np.int64))


def test_jitted_fp32_helpers_rounding_floor_on_cpu():
    """Production context is jit. MEASURED (jax/jaxlib 0.10.0, XLA:CPU): the dry view stays bitwise equal to WRF's
    REAL*4 separate-rounding order, while the moist_physics_finish form differs by <=1 ulp in ~10/36 cells --
    XLA:CPU contracts mul+add into FMA, which the production WRF build (-march=nocona) cannot. Eager evaluation is
    bitwise (tests above). This is a recorded compiler floor of the port's fp32 path, bounded here at 1 ulp; the GPU
    contraction behaviour is recorded in gate 2 (optimized HLO), not asserted here."""
    s = _state32()
    r = np.float32(461.6) / np.float32(287.0)
    th = np.asarray(s.theta, np.float32)
    q = np.asarray(s.qv, np.float32)
    jit_view = jax.jit(lambda t, qq: pc._dry_theta_view(SimpleNamespace(theta=t, qv=qq)))
    assert np.array_equal(np.asarray(jit_view(s.theta, s.qv)), th / (np.float32(1.0) + r * q))
    th_n, q_n = th + np.float32(0.75), q - np.float32(1.0e-3)
    jit_fin = jax.jit(lambda a, b, c, d: pc._theta_m_from_microphysics(a, b, c, d, jnp.float32))
    wrf = th * (np.float32(1) + r * q) + (th_n - th) * (np.float32(1) + r * q_n) + r * (q_n - q) * th_n
    assert int(_ulp(jit_fin(th, q, th_n, q_n), wrf).max()) <= 1
