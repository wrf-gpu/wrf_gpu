#!/usr/bin/env python3
"""ADR-023 warm-bubble vs slice diagnostic probe.

Diagnostic-only instrumentation for the M6.x warm-bubble failure sprint.
Compares the warm-bubble harness (`m6_warm_bubble_test.py`) path against the
MPAS column-slice oracle and against directly-driven invocations of
``_mpas_recurrence_vertical_update`` / ``vertical_acoustic_update`` /
``acoustic_substep_carry``. Reads source files only; writes a single JSON
artifact for the diagnostic report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import (
    BCMetadata,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics import acoustic_wrf as aw
from gpuwrf.dynamics.acoustic_wrf import (
    AcousticConfig,
    MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    MPAS_OMEGA_TO_W_METRIC,
    _mpas_recurrence_vertical_update,
    acoustic_substep_carry,
    diagnose_pressure_al_alt,
    initialize_acoustic_carry,
    run_acoustic_scan_carry,
    vertical_acoustic_update,
)
from gpuwrf.dynamics.damping import RayleighConfig, SmdivConfig
from gpuwrf.dynamics.metrics import flat_metrics_for_grid
from gpuwrf.validation.mpas_oracles import mpas_column_slice


config.update("jax_enable_x64", True)


GRAVITY_M_S2 = 9.80665
R_DRY_AIR = 287.05
CP_DRY_AIR = 1004.0
KAPPA = R_DRY_AIR / CP_DRY_AIR
P0_PA = 100_000.0
T0_K = 300.0


def _pressure_at_height(z_m):
    return P0_PA * np.exp(-GRAVITY_M_S2 * np.asarray(z_m) / (R_DRY_AIR * T0_K))


def _theta_from_pressure(p):
    return T0_K * (P0_PA / p) ** KAPPA


# ----- A. Warm-bubble harness setup (mirrors scripts/m6_warm_bubble_test.py) -----

def build_warm_bubble_3d(
    nx: int = 64,
    ny: int = 64,
    nz: int = 40,
    dx_m: float = 400.0,
    dz_m: float = 100.0,
    bubble_center_z_m: float = 2000.0,
    bubble_radius_m: float = 2000.0,
    bubble_amplitude_k: float = 2.0,
):
    projection = Projection("lambert", 0.0, 0.0, float(dx_m), float(dx_m), int(nx), int(ny))
    z_top = float(nz) * float(dz_m)
    p_top = float(_pressure_at_height(z_top))
    terrain = TerrainProvenance(
        source_path="diagnostic://warm-bubble-3d",
        sha256="diagnostic-wb-3d",
        shape=(int(ny), int(nx)),
        units="m",
        projection_transform="cartesian",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta = jnp.linspace(1.0, 0.0, int(nz) + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", int(nz), p_top, eta)
    bc = BCMetadata("ideal", ("u", "v", "w", "theta", "p", "pb", "ph", "mu"), 0, "linear", False)
    terrain_h = jnp.zeros((int(ny), int(nx)), dtype=jnp.float64)
    grid = GridSpec(projection, terrain, vertical, bc, eta, terrain_h)

    z_face_1d = np.arange(nz + 1, dtype=np.float64) * float(dz_m)
    z_mass_1d = 0.5 * (z_face_1d[:-1] + z_face_1d[1:])
    z_mass = np.broadcast_to(z_mass_1d[:, None, None], (nz, ny, nx))
    z_face = np.broadcast_to(z_face_1d[:, None, None], (nz + 1, ny, nx))

    pb_1d = _pressure_at_height(z_mass_1d)
    pb = np.broadcast_to(pb_1d[:, None, None], (nz, ny, nx)).copy()
    theta_base_1d = _theta_from_pressure(pb_1d)
    theta_base = np.broadcast_to(theta_base_1d[:, None, None], (nz, ny, nx)).copy()
    phb = GRAVITY_M_S2 * z_face
    mub = np.full((ny, nx), P0_PA - p_top, dtype=np.float64)

    x_mass = (np.arange(nx, dtype=np.float64) + 0.5) * dx_m
    domain_x = nx * dx_m
    bubble_center_x_m = 0.5 * domain_x
    periodic_dx = np.minimum(np.abs(x_mass - bubble_center_x_m), domain_x - np.abs(x_mass - bubble_center_x_m))
    r2 = periodic_dx[None, None, :] ** 2 + (z_mass - bubble_center_z_m) ** 2
    theta_perturbation = bubble_amplitude_k * np.exp(-r2 / (bubble_radius_m * bubble_radius_m))
    theta = theta_base + theta_perturbation

    state = State.zeros(grid).replace(
        theta=jnp.asarray(theta),
        p_total=jnp.asarray(pb),
        p_perturbation=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        ph_total=jnp.asarray(phb),
        ph_perturbation=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        mu_total=jnp.asarray(mub),
        mu_perturbation=jnp.zeros((ny, nx), dtype=jnp.float64),
        t_skin=jnp.full((ny, nx), T0_K, dtype=jnp.float64),
        xland=jnp.ones((ny, nx), dtype=jnp.float32),
        mavail=jnp.ones((ny, nx), dtype=jnp.float32),
        roughness_m=jnp.full((ny, nx), 0.1, dtype=jnp.float64),
        rhosfc=jnp.full((ny, nx), P0_PA / (R_DRY_AIR * T0_K), dtype=jnp.float64),
    )
    base = BaseState(
        pb=jnp.asarray(pb),
        phb=jnp.asarray(phb),
        mub=jnp.asarray(mub),
        t0=jnp.asarray(theta_base),
        theta_base=jnp.asarray(theta_base),
    )
    return grid, state, base, theta_base, z_mass


# ----- B. Single-column "shrunk slice" pulled from harness center -----

def shrink_to_center_column(grid, state, base, theta_base, z_mass):
    """Returns a (nz, 1, 1) version of the state at the bubble center column."""
    nx = grid.nx
    ny = grid.ny
    icx = nx // 2
    icy = ny // 2

    proj1 = Projection("lambert", 0.0, 0.0, float(grid.projection.dx_m), float(grid.projection.dy_m), 1, 1)
    terrain1 = TerrainProvenance(
        source_path="diagnostic://wb-center-column",
        sha256="diagnostic-wb-cc",
        shape=(1, 1),
        units="m",
        projection_transform="flat-column",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = grid.vertical
    bc = grid.bc
    grid1 = GridSpec(
        proj1,
        terrain1,
        vertical,
        bc,
        grid.eta_levels,
        jnp.zeros((1, 1), dtype=jnp.float64),
    )

    def slice_field(arr):
        a = np.asarray(arr)
        if a.ndim == 3:
            return jnp.asarray(a[:, icy : icy + 1, icx : icx + 1])
        if a.ndim == 2:
            return jnp.asarray(a[icy : icy + 1, icx : icx + 1])
        raise ValueError(f"unexpected shape {a.shape}")

    arrays = {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid1).items()}
    arrays["theta"] = slice_field(state.theta)
    arrays["p_total"] = slice_field(state.p_total)
    arrays["p"] = slice_field(state.p_total)
    arrays["p_perturbation"] = jnp.zeros_like(arrays["p_total"])
    arrays["ph_total"] = slice_field(state.ph_total)
    arrays["ph"] = slice_field(state.ph_total)
    arrays["ph_perturbation"] = jnp.zeros_like(arrays["ph_total"])
    arrays["mu_total"] = slice_field(state.mu_total)
    arrays["mu"] = slice_field(state.mu_total)
    arrays["mu_perturbation"] = jnp.zeros_like(arrays["mu_total"])
    arrays["t_skin"] = jnp.full((1, 1), T0_K, dtype=jnp.float64)
    arrays["xland"] = jnp.ones((1, 1), dtype=jnp.float32)
    arrays["mavail"] = jnp.ones((1, 1), dtype=jnp.float32)
    arrays["roughness_m"] = jnp.full((1, 1), 0.1, dtype=jnp.float64)
    arrays["rhosfc"] = jnp.full((1, 1), P0_PA / (R_DRY_AIR * T0_K), dtype=jnp.float64)
    s1 = State(**arrays)
    b1 = BaseState(
        pb=slice_field(base.pb),
        phb=slice_field(base.phb),
        mub=slice_field(base.mub),
        t0=slice_field(base.t0),
        theta_base=slice_field(base.theta_base),
    )
    return grid1, s1, b1


# ----- Probes -----

def _snap(state, theta_base):
    theta_pert = np.asarray(state.theta) - np.asarray(theta_base)
    p_pert = np.asarray(state.p_perturbation)
    w = np.asarray(state.w)
    mu_pert = np.asarray(state.mu_perturbation)
    ph_pert = np.asarray(state.ph_perturbation)
    return {
        "w_max": float(np.max(np.abs(w))),
        "w_signed_max": float(np.max(w)),
        "w_min": float(np.min(w)),
        "theta_pert_max": float(np.max(theta_pert)),
        "theta_pert_min": float(np.min(theta_pert)),
        "p_pert_max": float(np.max(np.abs(p_pert))),
        "ph_pert_max": float(np.max(np.abs(ph_pert))),
        "mu_pert_max": float(np.max(np.abs(mu_pert))),
        "finite": bool(np.all(np.isfinite(w)) and np.all(np.isfinite(state.theta))),
    }


def probe_3d_via_acoustic_scan(
    *,
    duration_s: float = 600.0,
    dt_macro_s: float = 2.0,
    n_acoustic: int = 8,
    epssm: float = 0.1,
    mu_continuity: bool = True,
    snapshot_substeps=(1, 8, 80, 800, 2400),
):
    grid, state0, base, theta_base, z_mass = build_warm_bubble_3d()
    metrics = grid.metrics
    cfg = AcousticConfig(
        n_substeps=int(n_acoustic),
        dx_m=float(grid.projection.dx_m),
        dy_m=float(grid.projection.dy_m),
        non_hydrostatic=True,
        top_lid=True,
        mu_continuity=bool(mu_continuity),
        epssm=float(epssm),
        smdiv=SmdivConfig(enabled=False, coefficient=0.0),
        rayleigh=RayleighConfig(enabled=False, coefficient=0.0),
    )
    sub_dt = float(dt_macro_s) / float(n_acoustic)

    init_carry = initialize_acoustic_carry(state0, state0.p_perturbation, metrics, base, cfg)

    @jax.jit
    def one_substep(carry):
        return acoustic_substep_carry(carry, metrics, cfg, sub_dt, base)

    carry = init_carry
    snapshots = {0: _snap(state0, theta_base)}
    total = int(round(duration_s / sub_dt))
    snap_set = set(int(s) for s in snapshot_substeps)
    for step in range(1, total + 1):
        carry = one_substep(carry)
        if step in snap_set or step == total:
            jax.block_until_ready(carry.state.w)
            snapshots[step] = _snap(carry.state, theta_base)
    return {
        "total_substeps": total,
        "sub_dt_s": sub_dt,
        "epssm": float(epssm),
        "mu_continuity": bool(mu_continuity),
        "snapshots": {str(k): v for k, v in snapshots.items()},
    }


def probe_3d_direct_recurrence(
    *,
    duration_s: float = 600.0,
    dt_macro_s: float = 2.0,
    n_acoustic: int = 8,
    epssm: float = 0.1,
    buoyancy_scale: float = MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    snapshot_substeps=(1, 8, 80, 800, 2400),
):
    """Bypass acoustic_substep_carry; drive _mpas_recurrence_vertical_update only."""
    grid, state0, base, theta_base, z_mass = build_warm_bubble_3d()
    metrics = grid.metrics
    sub_dt = float(dt_macro_s) / float(n_acoustic)

    @jax.jit
    def one_substep(state):
        return _mpas_recurrence_vertical_update(
            state,
            base,
            metrics,
            dt=sub_dt,
            epssm=float(epssm),
            top_lid=True,
            buoyancy_scale=float(buoyancy_scale),
        )

    state = state0
    snapshots = {0: _snap(state, theta_base)}
    total = int(round(duration_s / sub_dt))
    snap_set = set(int(s) for s in snapshot_substeps)
    for step in range(1, total + 1):
        state = one_substep(state)
        if step in snap_set or step == total:
            jax.block_until_ready(state.w)
            snapshots[step] = _snap(state, theta_base)
    return {
        "total_substeps": total,
        "sub_dt_s": sub_dt,
        "epssm": float(epssm),
        "buoyancy_scale": float(buoyancy_scale),
        "snapshots": {str(k): v for k, v in snapshots.items()},
    }


def probe_1d_shrunk_via_acoustic_scan(
    *,
    duration_s: float = 600.0,
    dt_macro_s: float = 2.0,
    n_acoustic: int = 8,
    epssm: float = 0.1,
    mu_continuity: bool = True,
    snapshot_substeps=(1, 8, 80, 800, 2400),
):
    grid3, state3, base3, theta_base3, z_mass3 = build_warm_bubble_3d()
    grid1, state0, base = shrink_to_center_column(grid3, state3, base3, theta_base3, z_mass3)
    theta_base1 = np.asarray(base.theta_base)
    metrics = grid1.metrics
    cfg = AcousticConfig(
        n_substeps=int(n_acoustic),
        dx_m=float(grid1.projection.dx_m),
        dy_m=float(grid1.projection.dy_m),
        non_hydrostatic=True,
        top_lid=True,
        mu_continuity=bool(mu_continuity),
        epssm=float(epssm),
    )
    sub_dt = float(dt_macro_s) / float(n_acoustic)
    init_carry = initialize_acoustic_carry(state0, state0.p_perturbation, metrics, base, cfg)

    @jax.jit
    def one_substep(carry):
        return acoustic_substep_carry(carry, metrics, cfg, sub_dt, base)

    carry = init_carry
    snapshots = {0: _snap(state0, theta_base1)}
    total = int(round(duration_s / sub_dt))
    snap_set = set(int(s) for s in snapshot_substeps)
    for step in range(1, total + 1):
        carry = one_substep(carry)
        if step in snap_set or step == total:
            jax.block_until_ready(carry.state.w)
            snapshots[step] = _snap(carry.state, theta_base1)
    return {
        "total_substeps": total,
        "sub_dt_s": sub_dt,
        "epssm": float(epssm),
        "mu_continuity": bool(mu_continuity),
        "snapshots": {str(k): v for k, v in snapshots.items()},
    }


def probe_1d_shrunk_direct_recurrence(
    *,
    duration_s: float = 600.0,
    dt_macro_s: float = 2.0,
    n_acoustic: int = 8,
    epssm: float = 0.1,
    buoyancy_scale: float = MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    snapshot_substeps=(1, 8, 80, 800, 2400),
):
    grid3, state3, base3, theta_base3, z_mass3 = build_warm_bubble_3d()
    grid1, state0, base = shrink_to_center_column(grid3, state3, base3, theta_base3, z_mass3)
    theta_base1 = np.asarray(base.theta_base)
    metrics = grid1.metrics
    sub_dt = float(dt_macro_s) / float(n_acoustic)

    @jax.jit
    def one_substep(state):
        return _mpas_recurrence_vertical_update(
            state,
            base,
            metrics,
            dt=sub_dt,
            epssm=float(epssm),
            top_lid=True,
            buoyancy_scale=float(buoyancy_scale),
        )

    state = state0
    snapshots = {0: _snap(state, theta_base1)}
    total = int(round(duration_s / sub_dt))
    snap_set = set(int(s) for s in snapshot_substeps)
    for step in range(1, total + 1):
        state = one_substep(state)
        if step in snap_set or step == total:
            jax.block_until_ready(state.w)
            snapshots[step] = _snap(state, theta_base1)
    return {
        "total_substeps": total,
        "sub_dt_s": sub_dt,
        "epssm": float(epssm),
        "buoyancy_scale": float(buoyancy_scale),
        "snapshots": {str(k): v for k, v in snapshots.items()},
    }


def probe_one_substep_decomposition():
    """Capture the first-substep buoyancy, restoring, and density-coupling magnitudes."""
    grid, state, base, theta_base, _z_mass = build_warm_bubble_3d()
    metrics = grid.metrics
    sub_dt = 0.25
    epssm = 0.1

    # Internals analogous to _mpas_recurrence_vertical_update first iter.
    from gpuwrf.dynamics.vertical_implicit_solver import build_epssm_column_coefficients

    theta_arr = np.asarray(state.theta)
    theta_base_arr = np.asarray(theta_base)
    theta_pert = theta_arr - theta_base_arr
    rho_pp = np.asarray(state.p_perturbation) / (
        aw.GAMMA_DRY_AIR * aw.R_D * theta_base_arr
    )
    rw_p = np.asarray(state.w) * MPAS_OMEGA_TO_W_METRIC
    dz = np.asarray(aw._layer_thickness_m(state, base, metrics))
    cofrz, cofwr, cofwz, coftz, cofwt, rdzw, a, b, c = (
        np.asarray(x)
        for x in build_epssm_column_coefficients(state.theta, dz, dt=sub_dt, epssm=epssm)
    )
    resm = (1.0 - epssm) / (1.0 + epssm)

    rs = rho_pp - cofrz * resm * (rw_p[1:, :, :] - rw_p[:-1, :, :])
    ts = theta_pert - resm * rdzw * (
        coftz[1:, :, :] * rw_p[1:, :, :] - coftz[:-1, :, :] * rw_p[:-1, :, :]
    )
    buoyancy_face = np.asarray(aw._vertical_buoyancy_acceleration(state, base))

    buoy_term = sub_dt * MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE * buoyancy_face[1:-1, :, :]
    cofwz_term = -cofwz[1:-1, :, :] * (
        (ts[1:, :, :] - ts[:-1, :, :]) + resm * (theta_pert[1:, :, :] - theta_pert[:-1, :, :])
    )
    cofwr_term = -cofwr[1:-1, :, :] * (
        (rs[1:, :, :] + rs[:-1, :, :]) + resm * (rho_pp[1:, :, :] + rho_pp[:-1, :, :])
    )
    cofwt_term = (
        cofwt[1:, :, :] * (ts[1:, :, :] + resm * theta_pert[1:, :, :])
        + cofwt[:-1, :, :] * (ts[:-1, :, :] + resm * theta_pert[:-1, :, :])
    )

    icy = grid.ny // 2
    icx = grid.nx // 2

    def col_stats(name, arr3d):
        col = arr3d[:, icy, icx]
        return {
            "name": name,
            "max_abs": float(np.max(np.abs(arr3d))),
            "col_center_min": float(np.min(col)),
            "col_center_max": float(np.max(col)),
            "col_center_argmax_k": int(np.argmax(np.abs(col))),
        }

    decomposition = {
        "buoyancy_face_full_3d": col_stats("buoyancy_face[1:-1]", buoyancy_face[1:-1, :, :]),
        "buoyancy_rhs_contribution_dt0p25": col_stats("dt*0.38*buoy", buoy_term),
        "cofwz_theta_restoring": col_stats("-cofwz*(ts_diff + resm*theta_diff)", cofwz_term),
        "cofwr_density_term": col_stats("-cofwr*(rs_sum + resm*rho_pp_sum)", cofwr_term),
        "cofwt_pressure_term": col_stats("cofwt*(ts + resm*theta_pert)*pair", cofwt_term),
        "ts_at_center_column": [float(v) for v in ts[:, icy, icx]],
        "theta_pert_at_center_column": [float(v) for v in theta_pert[:, icy, icx]],
        "cofwz_at_center_column": [float(v) for v in cofwz[:, icy, icx]],
        "buoyancy_face_at_center_column": [float(v) for v in buoyancy_face[:, icy, icx]],
        "tridiagonal_b_at_center_column": [float(v) for v in b[:, icy, icx]],
        "MPAS_OMEGA_TO_W_METRIC": float(MPAS_OMEGA_TO_W_METRIC),
        "MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE": float(MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE),
    }
    return decomposition


def probe_buoyancy_scale_sweep(
    *,
    duration_s: float = 60.0,
    dt_macro_s: float = 2.0,
    n_acoustic: int = 8,
    scales=(0.38, 1.0, 2.6, 1.0 / 0.38),
):
    """Run the harness for 60s under different buoyancy_scale values.

    Uses direct recurrence to isolate the buoyancy parameter from coupling
    overhead.
    """
    grid, state0, base, theta_base, _z_mass = build_warm_bubble_3d()
    metrics = grid.metrics
    sub_dt = float(dt_macro_s) / float(n_acoustic)
    total = int(round(duration_s / sub_dt))
    out = {}
    for scale in scales:
        @jax.jit
        def one_substep(state, sc=float(scale)):
            return _mpas_recurrence_vertical_update(
                state,
                base,
                metrics,
                dt=sub_dt,
                epssm=0.1,
                top_lid=True,
                buoyancy_scale=sc,
            )

        state = state0
        for _ in range(total):
            state = one_substep(state)
        jax.block_until_ready(state.w)
        out[str(scale)] = _snap(state, theta_base)
        out[str(scale)]["total_substeps"] = total
    return out


def probe_acoustic_substep_carry_overwrite_check():
    """Does diagnose_pressure_al_alt overwrite the recurrence's p_perturbation?"""
    grid, state, base, theta_base, _z_mass = build_warm_bubble_3d()
    metrics = grid.metrics
    sub_dt = 0.25

    # Path 1: only the recurrence kernel
    @jax.jit
    def recurrence_only(s):
        return _mpas_recurrence_vertical_update(
            s, base, metrics, dt=sub_dt, epssm=0.1, top_lid=True,
            buoyancy_scale=MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
        )
    s_after_recur = recurrence_only(state)
    p_pert_after_recurrence = float(np.max(np.abs(np.asarray(s_after_recur.p_perturbation))))

    # Path 2: full acoustic_substep_carry
    cfg = AcousticConfig(
        n_substeps=8,
        dx_m=float(grid.projection.dx_m),
        dy_m=float(grid.projection.dy_m),
        non_hydrostatic=True,
        top_lid=True,
        mu_continuity=True,
        epssm=0.1,
    )
    init_carry = initialize_acoustic_carry(state, state.p_perturbation, metrics, base, cfg)

    @jax.jit
    def one_carry(c):
        return acoustic_substep_carry(c, metrics, cfg, sub_dt, base)
    after_carry = one_carry(init_carry)
    p_pert_after_carry = float(np.max(np.abs(np.asarray(after_carry.state.p_perturbation))))

    # Verify diagnose_pressure_al_alt independently
    pp, al, alt = diagnose_pressure_al_alt(s_after_recur, base, metrics)
    p_pert_diagnose_from_post_recur = float(np.max(np.abs(np.asarray(pp))))

    pp0, _al0, _alt0 = diagnose_pressure_al_alt(state, base, metrics)
    p_pert_diagnose_initial = float(np.max(np.abs(np.asarray(pp0))))

    return {
        "p_perturbation_after_recurrence_only": p_pert_after_recurrence,
        "p_perturbation_after_acoustic_substep_carry": p_pert_after_carry,
        "p_perturbation_from_diagnose_applied_to_post_recurrence_state": p_pert_diagnose_from_post_recur,
        "p_perturbation_from_diagnose_on_initial_state": p_pert_diagnose_initial,
        "interpretation": (
            "If p_perturbation_after_acoustic_substep_carry is far smaller than "
            "p_perturbation_after_recurrence_only, the diagnose_pressure_al_alt "
            "overwrite is erasing the recurrence's density-derived pressure."
        ),
    }


def probe_slice_oracle_vs_unified_first_steps():
    """First-3-substep apples-to-apples comparison."""
    # Slice oracle: warm_bubble_2km scenario at N_LEVELS=16, dz≈625m, 3K cos² bubble
    sliced = mpas_column_slice(
        scenario="warm_bubble_2km",
        n_levels=16,
        column_height_m=10000.0,
        dt_acoustic_s=1.0,
        n_substeps=40,
        epssm=0.1,
    )
    slice_w_peaks = [float(np.max(np.abs(sliced["w"][i]))) for i in (0, 1, 2, 3, 10, 40)]

    # Unified path applied to same setup (mirrors test_mpas_slice_trajectory)
    proj = Projection("lambert", 0.0, 0.0, 1000.0, 1000.0, 1, 1)
    terrain = TerrainProvenance(
        source_path="diagnostic://slice-mirror",
        sha256="diagnostic-slice-mirror",
        shape=(1, 1),
        units="m",
        projection_transform="flat-column",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    n = 16
    z_faces = jnp.linspace(0.0, 10000.0, n + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", n, 5000.0, jnp.linspace(1.0, 0.0, n + 1, dtype=jnp.float64))
    bc = BCMetadata("ideal", ("w", "ph", "theta"), 1, "linear", True)
    grid_s = GridSpec(proj, terrain, vertical, jnp.linspace(1.0, 0.0, n + 1, dtype=jnp.float64), bc, jnp.zeros((1, 1), dtype=jnp.float64)) if False else None  # noqa
    # Use the same grid construction as the test:
    grid_s = GridSpec(proj, terrain, vertical, bc, jnp.linspace(1.0, 0.0, n + 1, dtype=jnp.float64), jnp.zeros((1, 1), dtype=jnp.float64))
    arrays = {f: jnp.zeros(sh, dtype=jnp.float64) for f, sh in _state_field_shapes(grid_s).items()}
    theta_base_s = jnp.ones((n, 1, 1), dtype=jnp.float64) * T0_K
    pb = jnp.ones_like(theta_base_s) * 90_000.0
    phb = jnp.broadcast_to((GRAVITY_M_S2 * z_faces)[:, None, None], (n + 1, 1, 1))
    mub = jnp.ones((1, 1), dtype=jnp.float64) * 90_000.0
    arrays["w"] = jnp.asarray(sliced["w"][0], dtype=jnp.float64)[:, None, None]
    arrays["theta"] = theta_base_s + jnp.asarray(sliced["theta_perturbation"][0], dtype=jnp.float64)[:, None, None]
    arrays["p"] = pb
    arrays["p_total"] = pb
    arrays["p_perturbation"] = jnp.zeros_like(pb)
    arrays["ph"] = phb + jnp.asarray(sliced["ph_perturbation"][0], dtype=jnp.float64)[:, None, None]
    arrays["ph_total"] = arrays["ph"]
    arrays["ph_perturbation"] = jnp.asarray(sliced["ph_perturbation"][0], dtype=jnp.float64)[:, None, None]
    arrays["mu"] = mub
    arrays["mu_total"] = mub
    arrays["mu_perturbation"] = jnp.zeros_like(mub)
    state = State(**arrays)
    base = BaseState(pb=pb, phb=phb, mub=mub, t0=theta_base_s, theta_base=theta_base_s)
    cfg = AcousticConfig(n_substeps=1, non_hydrostatic=True, mu_continuity=True, epssm=0.1)
    metrics = grid_s.metrics
    init_carry = initialize_acoustic_carry(state, state.p_perturbation, metrics, base, cfg)

    @jax.jit
    def one(c):
        return acoustic_substep_carry(c, metrics, cfg, 1.0, base)
    carry = init_carry
    unified_w_peaks = [float(np.max(np.abs(np.asarray(carry.state.w))))]
    targets = {1, 2, 3, 10, 40}
    for s in range(1, 41):
        carry = one(carry)
        if s in targets:
            jax.block_until_ready(carry.state.w)
            unified_w_peaks.append(float(np.max(np.abs(np.asarray(carry.state.w)))))

    return {
        "slice_oracle_w_peaks_steps_0_1_2_3_10_40": slice_w_peaks,
        "unified_acoustic_substep_carry_w_peaks_steps_0_1_2_3_10_40": unified_w_peaks,
        "ratio_at_40": float(unified_w_peaks[-1] / max(slice_w_peaks[-1], 1e-12)),
        "note": (
            "Single-column slice scenario where mu_perturbation stays 0 (horizontal "
            "fluxes are zero). If ratio_at_40 ≈ 1, then the column-only mu=0 case "
            "matches; the 3D bubble failure must come from the 3D coupling."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".agent" / "sprints" / "2026-05-23-m6x-warm-bubble-failure-diagnostic" / "probe_warm_bubble_vs_slice.json",
    )
    parser.add_argument("--full-duration-s", type=float, default=600.0)
    args = parser.parse_args(argv)

    payload = {}

    print("[probe] 1. 3D warm-bubble via acoustic_substep_carry (production path)...", flush=True)
    payload["a_3d_acoustic_substep_carry_baseline"] = probe_3d_via_acoustic_scan(
        duration_s=float(args.full_duration_s),
    )

    print("[probe] 2. 3D warm-bubble via _mpas_recurrence_vertical_update direct...", flush=True)
    payload["b_3d_direct_recurrence"] = probe_3d_direct_recurrence(
        duration_s=float(args.full_duration_s),
    )

    print("[probe] 3. 1D shrunk-center via acoustic_substep_carry...", flush=True)
    payload["c_1d_shrunk_acoustic_substep_carry"] = probe_1d_shrunk_via_acoustic_scan(
        duration_s=float(args.full_duration_s),
    )

    print("[probe] 4. 1D shrunk-center via direct recurrence...", flush=True)
    payload["d_1d_shrunk_direct_recurrence"] = probe_1d_shrunk_direct_recurrence(
        duration_s=float(args.full_duration_s),
    )

    print("[probe] 5. First-substep rhs decomposition...", flush=True)
    payload["e_first_substep_rhs_decomposition"] = probe_one_substep_decomposition()

    print("[probe] 6. Buoyancy-scale sweep (60s direct recurrence)...", flush=True)
    payload["f_buoyancy_scale_sweep"] = probe_buoyancy_scale_sweep(
        duration_s=60.0,
    )

    print("[probe] 7. acoustic_substep_carry overwrite check...", flush=True)
    payload["g_overwrite_check"] = probe_acoustic_substep_carry_overwrite_check()

    print("[probe] 8. Slice oracle apples-to-apples...", flush=True)
    payload["h_slice_oracle_vs_unified"] = probe_slice_oracle_vs_unified_first_steps()

    print("[probe] 9. epssm sweep on 3D harness...", flush=True)
    sweep = {}
    for eps in (0.0, 0.1, 0.3):
        result = probe_3d_via_acoustic_scan(
            duration_s=300.0,
            epssm=eps,
            snapshot_substeps=(1, 8, 80, 800, 1200),
        )
        sweep[str(eps)] = result["snapshots"][str(result["total_substeps"])]
    payload["i_epssm_sweep_300s"] = sweep

    print("[probe] 10. mu_continuity ON/OFF sweep on 3D harness (60s)...", flush=True)
    payload["j_mu_continuity_ablation_60s"] = {
        "on": probe_3d_via_acoustic_scan(duration_s=60.0, mu_continuity=True, snapshot_substeps=(1, 8, 80, 240))["snapshots"],
        "off": probe_3d_via_acoustic_scan(duration_s=60.0, mu_continuity=False, snapshot_substeps=(1, 8, 80, 240))["snapshots"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    print(f"[probe] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
