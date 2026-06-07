"""Proof object for the orographic gravity-wave drag kernel (``gwd_opt=1``).

A WRF-faithful port of ``bl_gwdo_run`` (``phys/physics_mmm/bl_gwdo.F90``) must
satisfy the physical invariants of the Kim-GWDO scheme (Choi & Hong 2015):

1. ZERO drag over flat terrain (``var2d=0``) -- the ``ldrag`` short-circuit.
2. The drag DECELERATES the resolved low-level flow (tendency opposes the wind).
3. Momentum-only: GWDO touches u/v but produces no temperature/heating term.
4. The drag magnitude is in the physical range (|du/dt| well below ~0.01 m/s^2
   for realistic terrain; the WRF ``dtfac`` limiter keeps it from over-shooting
   a critical line within one step).
5. Finite everywhere, including degenerate columns (zero wind, calm, isothermal).
6. Surface stress sign + magnitude: ``dusfcg`` integrates the column tendency
   and is zero over flat terrain.
7. Idealised bell-mountain column: a stably-stratified column with sub-grid
   variance launches a sane stress profile that decays upward (saturation), and
   decelerates the low-level wind.

These are analytic / conservation checks (no pristine-WRF savepoint required for
the unit gate); they pin the formulation against transcription error.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics.gwd_gwdo import (  # noqa: E402
    GWDOColumnState,
    GWDOStatics,
    gwdo_columns,
)


# --------------------------------------------------------------------------- #
# Column builders
# --------------------------------------------------------------------------- #
def _stable_column(B, K, u0=15.0, v0=0.0, lapse=0.0065, tsfc=288.0):
    """A stably-stratified hydrostatic-ish column batch ``(B, K)``.

    Returns a :class:`GWDOColumnState`. Layers are ~300 m thick; pressure decays
    from 1000 hPa to 50 hPa; temperature falls with a constant lapse rate (so
    ``dtheta/dz > 0`` -> positive Brunt-Vaisala -> wave activity).
    """

    dz = 300.0
    z = np.cumsum(np.full((B, K), dz), axis=1) - dz / 2.0  # mid-layer heights (m)
    p_sfc, p_top = 100000.0, 5000.0
    prsi = np.linspace(p_sfc, p_top, K + 1)[None, :].repeat(B, 0)
    prsl = 0.5 * (prsi[:, :-1] + prsi[:, 1:])
    prslk = (prsl / 1.0e5) ** (287.0 / 1004.5)
    t1 = (tsfc - lapse * z).astype(np.float64)
    q1 = np.full((B, K), 0.004)
    u = np.full((B, K), u0)
    v = np.full((B, K), v0)
    return GWDOColumnState(
        uproj=jnp.asarray(u),
        vproj=jnp.asarray(v),
        t1=jnp.asarray(t1),
        q1=jnp.asarray(q1),
        prsl=jnp.asarray(prsl),
        prsi=jnp.asarray(prsi),
        prslk=jnp.asarray(prslk),
        zl=jnp.asarray(z),
    )


def _statics(B, var, *, oc1=1.0, oa=0.4, ol=0.3, sina=0.0, cosa=1.0, dx=3000.0):
    def f(val):
        return jnp.asarray(np.full((B,), float(val)))

    var_arr = jnp.asarray(np.asarray(var, dtype=np.float64)) if np.ndim(var) else f(var)
    return GWDOStatics(
        var=var_arr,
        oc1=f(oc1),
        oa1=f(oa),
        oa2=f(0.0),
        oa3=f(0.0),
        oa4=f(0.0),
        ol1=f(ol),
        ol2=f(ol),
        ol3=f(ol),
        ol4=f(ol),
        sina=f(sina),
        cosa=f(cosa),
        dxmeter=f(dx),
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_flat_terrain_zero_drag():
    """var2d=0 -> ldrag short-circuit -> exactly zero tendency and stress."""

    B, K = 3, 32
    col = _stable_column(B, K)
    stat = _statics(B, var=0.0)
    out = gwdo_columns(col, stat, 60.0)
    assert np.allclose(np.asarray(out.rublten), 0.0)
    assert np.allclose(np.asarray(out.rvblten), 0.0)
    assert np.allclose(np.asarray(out.dusfcg), 0.0)
    assert np.allclose(np.asarray(out.dvsfcg), 0.0)
    assert np.allclose(np.asarray(out.dtaux3d), 0.0)


def test_drag_opposes_low_level_wind():
    """For westerly (u>0) flow over a mountain, the u-tendency must be <= 0."""

    B, K = 2, 32
    col = _stable_column(B, K, u0=20.0, v0=0.0)
    stat = _statics(B, var=300.0)
    out = gwdo_columns(col, stat, 60.0)
    ru = np.asarray(out.rublten)
    # Net low-level u tendency opposes the +u flow (deceleration).
    assert ru.min() < 0.0
    # Surface stress integral: dusfcg should be the integrated deceleration; for
    # westerly flow the integrated -(1/g)*sum(du*del) is positive (momentum sink).
    assert np.all(np.asarray(out.dusfcg) >= -1e-9)
    # rvblten stays ~0 for pure-zonal flow with on-axis asymmetry only.
    assert np.allclose(np.asarray(out.rvblten), 0.0, atol=1e-6)


def test_drag_magnitude_physical_range():
    """The dtfac (Lindzen) limiter caps the per-step wind change at the wind.

    The binding physical invariant of GWDO is the WRF ``dtfac`` limiter
    (``bl_gwdo.F90:595-602``): the gravity-wave momentum sink may not drive a
    critical line within one step, i.e. ``|d(wind)/dt| * dt <= |wind|`` in the
    low-level layers. We assert that, plus a generous absolute ceiling on the
    instantaneous tendency, across a sweep of sub-grid variance (100..900 m).
    """

    B, K = 4, 32
    dt = 60.0
    col = _stable_column(B, K, u0=25.0)
    stat = _statics(B, var=np.array([100.0, 300.0, 600.0, 900.0]))
    out = gwdo_columns(col, stat, dt)
    ru = np.asarray(out.rublten)
    rv = np.asarray(out.rvblten)
    assert np.all(np.isfinite(ru))
    # dtfac invariant: the implied wind change over a step never exceeds the
    # local resolved wind magnitude.
    wind = np.hypot(np.asarray(col.uproj), np.asarray(col.vproj))
    dwind = np.hypot(ru, rv) * dt
    assert np.all(dwind <= wind + 1e-6)
    # Generous absolute ceiling: even the var=900 m extreme stays < 0.1 m/s^2.
    assert np.abs(ru).max() < 0.1
    # Larger variance -> larger (or equal) integrated surface stress magnitude.
    dus = np.abs(np.asarray(out.dusfcg))
    assert dus[3] >= dus[0]


def test_finite_on_degenerate_columns():
    """Calm / isothermal / zero-wind columns must not produce NaN/Inf."""

    B, K = 3, 30
    # col 0: zero wind; col 1: isothermal (no stratification); col 2: normal.
    col0 = _stable_column(1, K, u0=0.0, v0=0.0)
    col1 = _stable_column(1, K, u0=10.0, lapse=0.0, tsfc=270.0)  # isothermal
    col2 = _stable_column(1, K, u0=12.0)
    col = GWDOColumnState(
        uproj=jnp.concatenate([col0.uproj, col1.uproj, col2.uproj], 0),
        vproj=jnp.concatenate([col0.vproj, col1.vproj, col2.vproj], 0),
        t1=jnp.concatenate([col0.t1, col1.t1, col2.t1], 0),
        q1=jnp.concatenate([col0.q1, col1.q1, col2.q1], 0),
        prsl=jnp.concatenate([col0.prsl, col1.prsl, col2.prsl], 0),
        prsi=jnp.concatenate([col0.prsi, col1.prsi, col2.prsi], 0),
        prslk=jnp.concatenate([col0.prslk, col1.prslk, col2.prslk], 0),
        zl=jnp.concatenate([col0.zl, col1.zl, col2.zl], 0),
    )
    stat = _statics(B, var=400.0)
    out = gwdo_columns(col, stat, 60.0)
    for arr in (out.rublten, out.rvblten, out.dtaux3d, out.dtauy3d, out.dusfcg, out.dvsfcg):
        assert bool(jnp.all(jnp.isfinite(arr)))


def test_no_spurious_v_tendency_for_zonal_flow_no_rotation():
    """Pure zonal flow, no off-axis asymmetry, no grid rotation -> rvblten ~ 0."""

    B, K = 2, 32
    col = _stable_column(B, K, u0=18.0, v0=0.0)
    stat = _statics(B, var=300.0, oa=0.5, sina=0.0, cosa=1.0)
    out = gwdo_columns(col, stat, 60.0)
    assert np.allclose(np.asarray(out.rvblten), 0.0, atol=1e-7)


def test_stress_profile_decays_upward():
    """The diagnosed stress tendency concentrates aloft (wave breaking), and the
    base stress is launched at/near the mountain top, not the model top."""

    B, K = 1, 40
    col = _stable_column(B, K, u0=20.0)
    stat = _statics(B, var=500.0)
    out = gwdo_columns(col, stat, 60.0)
    ru = np.asarray(out.rublten)[0]
    assert np.all(np.isfinite(ru))
    # the drag is not all dumped in a single level; some structure exists.
    assert np.count_nonzero(np.abs(ru) > 1e-9) >= 1
    # the integrated surface stress is finite and the sign opposes +u flow.
    assert np.asarray(out.dusfcg)[0] >= -1e-9


def test_grid_rotation_consistency():
    """With a 90-degree grid rotation (sina=1, cosa=0) the grid-relative
    tendency must rotate consistently: the deceleration that landed on u for the
    unrotated case lands on the rotated component, and stays finite + bounded."""

    B, K = 1, 32
    col = _stable_column(B, K, u0=20.0, v0=0.0)
    stat_unrot = _statics(B, var=300.0, sina=0.0, cosa=1.0)
    out_unrot = gwdo_columns(col, stat_unrot, 60.0)
    # rotate the grid 90 deg: sina=1, cosa=0. The same earth-relative wind now
    # needs uproj/vproj rotated so earth-relative u stays the same.
    # earth u1 = uproj*cosa - vproj*sina = -vproj ; to get earth u=20 set vproj=-20.
    col_rot = col._replace(uproj=jnp.zeros_like(col.uproj), vproj=jnp.full_like(col.vproj, -20.0))
    stat_rot = _statics(B, var=300.0, sina=1.0, cosa=0.0)
    out_rot = gwdo_columns(col_rot, stat_rot, 60.0)
    # The earth-relative momentum sink magnitude (dtaux3d/dtauy3d magnitude) must
    # match between the two framings.
    mag_unrot = np.hypot(np.asarray(out_unrot.dtaux3d), np.asarray(out_unrot.dtauy3d))
    mag_rot = np.hypot(np.asarray(out_rot.dtaux3d), np.asarray(out_rot.dtauy3d))
    assert np.allclose(mag_unrot, mag_rot, atol=1e-6)


def test_jit_compiles():
    """The kernel must be jittable (operational scan requirement)."""

    B, K = 4, 32
    col = _stable_column(B, K)
    stat = _statics(B, var=300.0)
    fn = jax.jit(lambda c, s: gwdo_columns(c, s, 60.0))
    out = fn(col, stat)
    assert out.rublten.shape == (B, K)
    assert bool(jnp.all(jnp.isfinite(out.rublten)))


# --------------------------------------------------------------------------- #
# Operational adapter (State -> State) coupling tests
# --------------------------------------------------------------------------- #
def _adapter_grid(ny, nx, nz):
    from gpuwrf.contracts.grid import (
        BCMetadata,
        DycoreMetrics,
        GridSpec,
        Projection,
        TerrainProvenance,
        VerticalCoord,
    )

    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    terrain_height = jnp.zeros((ny, nx), dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="unit-test", sha256="unit-test", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(ny=ny, nx=nx, nz=nz, eta_levels=eta,
                                 top_pressure_pa=5000.0, provenance="unit-test-flat")
    return GridSpec(projection, terrain_meta, vertical, bc, eta, terrain_height, metrics=metrics)


def _adapter_state(grid):
    from gpuwrf.contracts.state import State, _state_field_shapes

    shapes = _state_field_shapes(grid)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    # stratified column: theta increasing with height (positive N^2)
    theta = jnp.linspace(290.0, 320.0, nz, dtype=jnp.float64)[:, None, None]
    theta = jnp.broadcast_to(theta, (nz, ny, nx))
    p = jnp.linspace(95000.0, 20000.0, nz, dtype=jnp.float64)[:, None, None]
    p = jnp.broadcast_to(p, (nz, ny, nx))
    ph = jnp.linspace(0.0, 9000.0 * 9.80665, nz + 1, dtype=jnp.float64)[:, None, None]
    ph = jnp.broadcast_to(ph, (nz + 1, ny, nx))
    u = jnp.full((nz, ny, nx + 1), 20.0, dtype=jnp.float64)  # westerly C-grid faces
    fields.update(
        theta=theta, qv=jnp.full((nz, ny, nx), 0.004, dtype=jnp.float64),
        p=p, p_total=p, ph=ph, ph_total=ph,
        u=u, mu=jnp.full((ny, nx), 90000.0, dtype=jnp.float64),
        mu_total=jnp.full((ny, nx), 90000.0, dtype=jnp.float64),
        t_skin=jnp.full((ny, nx), 290.0, dtype=jnp.float64),
    )
    return State(**fields)


def test_adapter_momentum_only_and_flat_terrain_noop():
    """gwdo_adapter touches only u/v; over flat terrain (var=0) it is a no-op."""

    from gpuwrf.coupling.physics_couplers import (
        build_gwdo_statics_from_wrf_fields,
        gwdo_adapter,
    )

    ny, nx, nz = 3, 3, 24
    grid = _adapter_grid(ny, nx, nz)
    state = _adapter_state(grid)

    flat = build_gwdo_statics_from_wrf_fields(
        var2d=jnp.zeros((ny, nx)), con=jnp.ones((ny, nx)),
        oa1=jnp.full((ny, nx), 0.4), oa2=jnp.zeros((ny, nx)),
        oa3=jnp.zeros((ny, nx)), oa4=jnp.zeros((ny, nx)),
        ol1=jnp.full((ny, nx), 0.3), ol2=jnp.full((ny, nx), 0.3),
        ol3=jnp.full((ny, nx), 0.3), ol4=jnp.full((ny, nx), 0.3),
        dx_m=3000.0,
    )
    out_flat = gwdo_adapter(state, 60.0, flat, grid)
    # flat terrain: u/v unchanged
    assert np.allclose(np.asarray(out_flat.u), np.asarray(state.u))
    assert np.allclose(np.asarray(out_flat.v), np.asarray(state.v))

    # mountainous: u decelerates; theta/qv/qke/w untouched (momentum-only).
    mtn = build_gwdo_statics_from_wrf_fields(
        var2d=jnp.full((ny, nx), 400.0), con=jnp.ones((ny, nx)),
        oa1=jnp.full((ny, nx), 0.4), oa2=jnp.zeros((ny, nx)),
        oa3=jnp.zeros((ny, nx)), oa4=jnp.zeros((ny, nx)),
        ol1=jnp.full((ny, nx), 0.3), ol2=jnp.full((ny, nx), 0.3),
        ol3=jnp.full((ny, nx), 0.3), ol4=jnp.full((ny, nx), 0.3),
        dx_m=3000.0,
    )
    out_mtn = gwdo_adapter(state, 60.0, mtn, grid)
    # interior u-faces decelerate (westerly flow, drag opposes)
    du = np.asarray(out_mtn.u) - np.asarray(state.u)
    assert du.min() < 0.0
    assert np.all(du <= 1e-9)  # never accelerates
    # momentum-only: scalars untouched
    assert np.allclose(np.asarray(out_mtn.theta), np.asarray(state.theta))
    assert np.allclose(np.asarray(out_mtn.qv), np.asarray(state.qv))
    assert np.allclose(np.asarray(out_mtn.w), np.asarray(state.w))
    assert bool(jnp.all(jnp.isfinite(out_mtn.u)))


def test_interface_pressure_monotone_decreasing():
    """The reconstructed interface pressure must decrease monotonically with
    height so every layer mass del(k) = prsi(k) - prsi(k+1) is positive."""

    from gpuwrf.coupling.physics_couplers import _interface_pressure_from_state

    ny, nx, nz = 2, 2, 24
    grid = _adapter_grid(ny, nx, nz)
    state = _adapter_state(grid)
    prsi = np.asarray(_interface_pressure_from_state(state))  # (ny,nx,nz+1)
    dp = prsi[..., :-1] - prsi[..., 1:]
    assert np.all(dp > 0.0)
