"""v0.6.0 State<->scheme SCAN ADAPTERS for the operational forecast loop.

This module supplies the per-scheme ``State -> State`` adapters that the
operational scan (``runtime.operational_mode._physics_boundary_step``) routes
through the dispatcher (``coupling.physics_dispatch``). Each adapter mirrors the
established v0.2.0 coupler contract (``coupling.physics_couplers.thompson_adapter``
/ ``surface_adapter``): slice the resident :class:`State` into the scheme's
column-kernel inputs, call the WRF-savepoint-parity-passed kernel, then reassemble
``State`` from the kernel output (applying the scheme's
``PhysicsTendency.state_replacements`` / ``accumulator_increments`` per the frozen
``contracts.physics_interfaces`` contract, in WRF call order).

Scope (HONEST tracability, audited 2026-06-03 -- see the scan-wire handoff):

* **Microphysics** -- Kessler (1), WSM6 (6), Morrison (10), WDM6 (16) are pure
  ``jnp`` column kernels batched over ``(ncol, nlev)`` returning a
  ``PhysicsTendency`` of ``state_replacements``. They are jit/vmap-traceable and
  wire into the GPU scan as drop-in microphysics-slot adapters (the same slot
  ``thompson_adapter`` fills for mp=8).
* **Surface layer** -- revised-MM5 (1) and Pleim-Xiu (7) ``*_run`` paths are
  vectorized ``jnp`` and write the SAME B2 surface-flux handles
  (``ustar``/``theta_flux``/``qv_flux``/``tau_u``/``tau_v``/``rhosfc``/``fltv``)
  that ``surface_adapter`` writes for sf_sfclay=5, so they drop into the
  surface-layer slot.
* **Cumulus** -- KF (1) is a jit-able (``jax.lax.cond``) per-column kernel; its
  adapter vmaps the column step over the grid and threads its ``w0avg``/``nca``
  persistent carry through the operational carry's additive ``cumulus_carry``
  leaf. Tendencies (``RTHCUTEN``/``RQ*CUTEN``) are applied as ``state += dt*tend``
  and ``RAINCV`` accumulates into ``rainc_acc`` (mm).
* **PBL** -- YSU (1) / ACM2 (7) are the v0.6.0 ``jax.lax.scan``-traceable / vmap-
  batched rewrites of the host-NumPy single-column kernels
  (``physics.pbl_{ysu,acm2}.{ysu,acm2}_columns``). Their adapters re-derive the
  per-cell surface forcing the kernels consume via the revised-MM5 surface layer
  (``surface_layer_with_diagnostics``) and apply the PBL momentum increment
  A2C-averaged onto the C-grid faces (WRF ``add_a2c_u``/``add_a2c_v``), exactly
  like the MYNN adapter. Per-case savepoint parity re-passes on the traceable path
  (``pbl_gpuop_report.json``).

NOT wired here (kept fail-closed in the scan with scheme-specific reasons; the
dispatcher selects them, ``runtime.operational_mode._resolve_operational_suite``
rejects them loudly):

* **Noah-classic (2) land** -- wired in ``coupling.noahclassic_surface_hook`` as
  the land-surface slot analogue of ``coupling.noahmp_surface_hook``. It is not a
  table entry here because it threads a land carry rather than a plain
  ``State -> State`` adapter.
* **Grell-Freitas (3) / Tiedtke (6,16) cumulus** -- faithful CPU-NumPy reference
  ports (``gpu_runnable=False``); excluded from the GPU scan by design
  (GPU-batching TODO).

Every adapter is a pure ``State -> State`` (or ``(State, carry) -> (State, carry)``
for KF) function, allocates nothing at import, and writes at the live State dtype
(so ``force_fp64`` stays truly fp64 through physics; fp32-defeat fix).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.state import State
from gpuwrf.coupling.physics_couplers import (
    GRAVITY_M_S2,
    P0_PA,
    R_D_OVER_CP,
    _add_a2c_u_increment,
    _add_a2c_v_increment,
    _column_dz_from_state,
    _output_dtype,
    _rho_from_state,
    _surface_column_view,
    _temperature_from_theta,
    _u_mass,
    _v_mass,
    _w_mass,
)
from gpuwrf.physics.microphysics_kessler import kessler_physics_tendency
from gpuwrf.physics.microphysics_morrison import morrison_tendency
from gpuwrf.physics.microphysics_wdm6 import wdm6_physics_tendency
from gpuwrf.physics.microphysics_wsm6 import wsm6_physics_tendency
from gpuwrf.physics.cumulus_kf import step_kf_column
from gpuwrf.physics.pbl_acm2 import acm2_columns
from gpuwrf.physics.pbl_boulac import TEMIN as BOULAC_TKE_MIN, boulac_columns
from gpuwrf.physics.pbl_ysu import ysu_columns
from gpuwrf.physics.sfclay_pleim_xiu import step_pxsfclay_column
from gpuwrf.physics.sfclay_revised_mm5 import step_sfclay_revised_mm5_column
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

CP_DRY = 1004.0  # J kg^-1 K^-1 (matches R_D_OVER_CP = 287/1004)


# --- shared helpers -----------------------------------------------------------

def _exner_columns(p_columns: jax.Array) -> jax.Array:
    """WRF Exner function ``pi = (p/p0)**(Rd/cp)`` on column arrays."""

    return (jnp.maximum(p_columns, 1.0) / P0_PA) ** R_D_OVER_CP


def _mp_in(field3d: jax.Array, ny: int, nx: int, nz: int) -> jax.Array:
    """State leaf ``(nz, ny, nx)`` -> MP-kernel batch ``(ncol=ny*nx, nlev=nz)``.

    The WRF MP column kernels (``kessler_run`` / ``wsm6_run`` / ``morrison_run`` /
    ``wdm6_run``) are ``jax.vmap`` over ONE leading column axis: profiles are
    ``(ncol, nlev)`` (vertical trailing). The operational State is ``(nz, ny, nx)``;
    flatten the two spatial axes to a single batch axis (cf. MYNN's
    ``_flatten_columns_to_batch``).
    """

    return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)


def _apply_mp_replacements(state: State, tendency, *, ny: int, nx: int, nz: int) -> State:
    """Reassemble State from an MP ``PhysicsTendency`` (batch-major kernel output).

    The MP kernels return ``state_replacements`` keyed by State leaf as
    ``(ncol=ny*nx, nlev=nz)`` arrays and per-call mm ``accumulator_increments`` as
    ``(ncol,)``. This applies them WRF-faithfully: replacements overwrite the leaf
    (in-place style), accumulators ``+=`` (per-step). Writes at the live State
    dtype (fp32-defeat fix).
    """

    def _back3d(value):  # (ncol, nlev) -> (nz, ny, nx)
        return jnp.moveaxis(jnp.asarray(value).reshape(ny, nx, nz), -1, 0)

    updates: dict[str, jax.Array] = {}
    for leaf, value in tendency.state_replacements.items():
        updates[leaf] = _back3d(value).astype(_output_dtype(state, leaf))
    for acc, inc in tendency.accumulator_increments.items():
        prev = jnp.asarray(getattr(state, acc), dtype=jnp.float64)
        updates[acc] = (prev + jnp.asarray(inc, dtype=jnp.float64).reshape(ny, nx)).astype(
            _output_dtype(state, acc)
        )
    return state.replace(**updates)


# --- Microphysics adapters (mp_physics) ---------------------------------------

def kessler_adapter(state: State, dt: float, grid=None) -> State:
    """mp=1 Kessler warm-rain microphysics ``State -> State`` scan adapter."""

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2  # (nz+1, ny, nx)
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = kessler_physics_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(rho), mp(pii), mp(z_mass), mp(dz), float(dt),
    )
    tend.validate_keys()
    return _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)


def wsm6_adapter(state: State, dt: float, grid=None) -> State:
    """mp=6 WSM6 single-moment 6-class microphysics scan adapter."""

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = wsm6_physics_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
    )
    tend.validate_keys()
    return _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)


def morrison_adapter(state: State, dt: float, grid=None) -> State:
    """mp=10 Morrison two-moment microphysics scan adapter."""

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = morrison_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        mp(state.Ni), mp(state.Ns), mp(state.Nr), mp(state.Ng),
        mp(pii), mp(state.p), mp(dz), mp(_w_mass(state)), float(dt),
    )
    tend.validate_keys()
    return _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)


def wdm6_adapter(state: State, dt: float, grid=None) -> State:
    """mp=16 WDM6 double-moment warm-rain microphysics scan adapter.

    WDM6 introduces the additive State leaves ``Nc`` (cloud-droplet number) and
    ``Nn`` (CCN). The kernel returns ``Nc``/``Nr`` as replacements and ``Nn`` as a
    diagnostic; this adapter materializes ``Nn`` into the State leaf so the CCN
    prognostic actually evolves (it would otherwise be lost). ``slmsk`` (land/sea
    mask: 1 land / 0 sea) is derived from ``xland`` (1 land / 2 water).
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    slmsk_2d = jnp.where(jnp.asarray(state.xland) < 1.5, 1.0, 0.0)
    slmsk = _mp_in(jnp.broadcast_to(slmsk_2d[None, :, :], state.theta.shape), ny, nx, nz)
    tend = wdm6_physics_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        mp(state.Nn), mp(state.Nc), mp(state.Nr),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt), slmsk,
    )
    tend.validate_keys()
    next_state = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    # Nn is returned in diagnostics (the kernel predates the additive leaf); thread
    # it into the State leaf so the CCN prognostic evolves.
    nn = tend.diagnostics.get("Nn")
    if nn is not None:
        nn3d = jnp.moveaxis(jnp.asarray(nn).reshape(ny, nx, nz), -1, 0)
        next_state = next_state.replace(Nn=nn3d.astype(_output_dtype(state, "Nn")))
    return next_state


# --- Surface-layer adapters (sf_sfclay_physics) -------------------------------

def _surface_layer_bottom_inputs(state: State):
    """Lowest-model-level 2-D inputs the column surface-layer kernels consume."""

    u_mass = _u_mass(state)
    v_mass = _v_mass(state)
    T = _temperature_from_theta(state.theta, state.p)
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz_full = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    # Lowest model level (k=0). WRF passes the half-layer reference height; the
    # kernel's WRF wrapper uses 0.5*dz of the lowest layer for the surface stencil,
    # but the column entry takes the lowest full-layer dz like the savepoint cases.
    return {
        "u": u_mass[0],
        "v": v_mass[0],
        "temperature": T[0],
        "qv": state.qv[0],
        "pressure": state.p[0],
        "dz": dz_full[0],
        "psfc": state.p[0],  # surface pressure proxy from lowest-level pressure
    }


def _apply_surface_flux_replacements(state: State, tendency) -> State:
    """Write the B2 surface-flux handles a surface-layer scheme produced."""

    updates: dict[str, jax.Array] = {}
    for leaf, value in tendency.state_replacements.items():
        updates[leaf] = jnp.asarray(value).astype(_output_dtype(state, leaf))
    return state.replace(**updates)


def sfclay_revised_mm5_adapter(state: State, dt: float, grid=None) -> State:
    """sf_sfclay=1 revised-MM5 surface layer scan adapter (writes B2 flux handles)."""

    del dt
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    inp = _surface_layer_bottom_inputs(state)
    result = step_sfclay_revised_mm5_column(
        inp["u"], inp["v"], inp["temperature"], inp["qv"], inp["pressure"], inp["dz"],
        psfc=inp["psfc"],
        tsk=state.t_skin,
        xland=state.xland,
        lakemask=state.lakemask,
        mavail=state.mavail,
        znt=jnp.maximum(state.roughness_m, 1.0e-4),
        ust=jnp.maximum(state.ustar, 1.0e-3),
        dx=dx,
    )
    return _apply_surface_flux_replacements(state, result.tendency)


def pleim_xiu_sfclay_adapter(state: State, dt: float, grid=None) -> State:
    """sf_sfclay=7 Pleim-Xiu surface layer scan adapter (writes B2 flux handles)."""

    del dt
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    inp = _surface_layer_bottom_inputs(state)
    theta_bottom = state.theta[0]
    result = step_pxsfclay_column(
        inp["u"], inp["v"], inp["temperature"], inp["qv"], inp["pressure"], inp["dz"],
        theta=theta_bottom,
        psfc=inp["psfc"],
        tsk=state.t_skin,
        xland=state.xland,
        mavail=state.mavail,
        znt=jnp.maximum(state.roughness_m, 1.0e-4),
        ust=jnp.maximum(state.ustar, 1.0e-3),
        dx=dx,
    )
    return _apply_surface_flux_replacements(state, result.tendency)


# --- Cumulus adapter (cu_physics) ---------------------------------------------

def kf_adapter(state: State, dt: float, w0avg, nca, *, grid=None):
    """cu=1 Kain-Fritsch cumulus scan adapter.

    Returns ``(next_state, w0avg_next, nca_next)``. KF is a per-column kernel; the
    adapter vmaps it over the ``(ny*nx)`` grid columns. Its ``w0avg`` running mean
    of vertical velocity and ``nca`` cloud-relaxation countdown are persistent
    state threaded through the operational carry's ``cumulus_carry`` leaf.
    Tendencies are applied ``state += dt*tend`` (WRF ``RTHCUTEN``/``RQ*CUTEN`` are
    rates); ``RAINCV`` (mm/step) accumulates into ``rainc_acc``.
    """

    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    nz, ny, nx = state.theta.shape
    rho = _rho_from_state(state)
    T = _temperature_from_theta(state.theta, state.p)
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz_full = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    w_mass = _w_mass(state)

    # Reshape to (ncol, nz) with ncol = ny*nx and vmap the per-column KF step.
    def _cols(field3d):
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)

    T_c = _cols(T)
    qv_c = _cols(state.qv)
    p_c = _cols(state.p)
    dz_c = _cols(dz_full)
    rho_c = _cols(rho)
    u_c = _cols(_u_mass(state))
    v_c = _cols(_v_mass(state))
    w_c = _cols(w_mass)
    w0avg_c = jnp.asarray(w0avg, jnp.float64).reshape(ny * nx, nz)
    nca_c = jnp.asarray(nca, jnp.float64).reshape(ny * nx)

    def _one(T0, QV0, P0, DZQ, RHOE, w0a, U0, V0, w_col, nca0):
        res = step_kf_column(
            T0, QV0, P0, DZQ, RHOE, w0a, U0, V0, float(dt), dx,
            w=w_col, nca=nca0,
        )
        st = res.tendency.state_tendencies
        cc = res.carry.cumulus
        return (
            st["theta"], st["qv"], st["qc"], st["qr"], st["qi"], st["qs"],
            res.tendency.accumulator_increments["rainc_acc"],
            cc["w0avg"], cc["nca"],
        )

    (rth, rqv, rqc, rqr, rqi, rqs, raincv, w0avg_next_c, nca_next_c) = jax.vmap(_one)(
        T_c, qv_c, p_c, dz_c, rho_c, w0avg_c, u_c, v_c, w_c, nca_c
    )

    def _back(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    dt_f = float(dt)
    next_state = state.replace(
        theta=(state.theta + dt_f * _back(rth)).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back(rqv)).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * _back(rqc)).astype(_output_dtype(state, "qc")),
        qr=(state.qr + dt_f * _back(rqr)).astype(_output_dtype(state, "qr")),
        qi=(state.qi + dt_f * _back(rqi)).astype(_output_dtype(state, "qi")),
        qs=(state.qs + dt_f * _back(rqs)).astype(_output_dtype(state, "qs")),
        rainc_acc=(
            jnp.asarray(state.rainc_acc, jnp.float64) + raincv.reshape(ny, nx)
        ).astype(_output_dtype(state, "rainc_acc")),
    )
    w0avg_next = w0avg_next_c.reshape(ny, nx, nz)
    w0avg_next = jnp.moveaxis(w0avg_next, -1, 0)  # (nz, ny, nx) carry layout
    nca_next = nca_next_c.reshape(ny, nx)
    return next_state, w0avg_next, nca_next


def initial_kf_carry(state: State):
    """Seed the KF ``(w0avg, nca)`` carry: zero w-mean, nca=-100 (no active cloud)."""

    nz, ny, nx = state.theta.shape
    w0avg = jnp.zeros((nz, ny, nx), dtype=jnp.float64)
    nca = jnp.full((ny, nx), -100.0, dtype=jnp.float64)
    return (w0avg, nca)


# --- PBL adapters (bl_pbl_physics) --------------------------------------------
#
# YSU (1) / ACM2 (7) are the v0.6.0 ``jax.lax.scan``-traceable / vmap-batched PBL
# kernels (``physics.pbl_ysu.ysu_columns`` / ``physics.pbl_acm2.acm2_columns``).
# They occupy the same operational PBL slot as the v0.2.0 ``mynn_adapter`` and
# consume the surface-layer forcing the surface-layer slot wrote earlier in the
# WRF call chain. Both kernels need MORE surface inputs than the State's frozen B2
# kinematic-flux contract carries (YSU: hfx/qfx/br/psim/psih/u10/v10/znt; ACM2:
# hfx/qfx/pblh/wspd) -- the FROZEN B2 handles are kinematic fluxes only. The
# WRF-faithful surface-forcing assembler is the SAME revised-MM5 surface layer the
# operational scan already runs (``surface_layer_with_diagnostics``): it returns
# HFX/LH (W m^-2), psim/psih, the bulk Richardson number BR, U10/V10, ZNT and 1/L,
# so the PBL adapter re-derives the full per-cell forcing here -- no host transfer,
# fully traceable, exactly the inputs the savepoint-parity kernel was validated on.
#
# Momentum coupling mirrors the MYNN adapter (``_state_from_mynn_output``): form the
# A-grid PBL momentum INCREMENT on mass points and add it A2C-averaged onto the
# dynamics' ORIGINAL C-grid u/v faces (WRF ``add_a2c_u``/``add_a2c_v``), so the
# large-scale C-grid winds + the surface-w BC are preserved exactly. theta/qv live
# on the mass grid (read-back is identity) so they take the increment directly.

def _pbl_surface_forcing(state: State, grid):
    """Re-derive the YSU/ACM2 surface forcing via the revised-MM5 surface layer.

    Returns the per-cell ``(ny, nx)`` diagnostics the PBL kernels consume, plus the
    column-major ``(ncol, nz)`` profile views (mass-point winds, lowest->top) and
    the interface pressure column. Pure-``jnp``, no host transfer.
    """

    nz, ny, nx = state.theta.shape
    ncol = ny * nx
    diag = surface_layer_with_diagnostics(_surface_column_view(state))

    def _flat2d(field2d):  # (ny, nx) -> (ncol,)
        return jnp.asarray(field2d, jnp.float64).reshape(ncol)

    rho = _rho_from_state(state)
    T = _temperature_from_theta(state.theta, state.p)
    pii = _exner_columns(state.p)
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2  # (nz+1, ny, nx)
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)  # (nz, ny, nx)
    # Interface pressure (nz+1): interior faces = mean of adjacent levels; edges by
    # zero-gradient extrapolation (a reasonable assembler; the GPU forecast gate vs
    # CPU-WRF refines this against the true half-level pressure).
    p = state.p.astype(jnp.float64)
    p_int_interior = 0.5 * (p[:-1] + p[1:])  # (nz-1, ny, nx)
    p_int = jnp.concatenate([p[:1], p_int_interior, p[-1:]], axis=0)  # (nz+1, ny, nx)

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ncol, nz)

    rhosfc = jnp.asarray(state.rhosfc, jnp.float64)  # (ny, nx)
    cpm = CP_DRY * (1.0 + 0.84 * jnp.maximum(state.qv[0], 0.0))  # moist cp at k0
    hfx = jnp.asarray(diag.hfx, jnp.float64)  # W m^-2 upward
    lh = jnp.asarray(diag.lh, jnp.float64)  # W m^-2 upward
    XLV = 2.5e6
    qfx = lh / XLV  # kg m^-2 s^-1 (LH = XLV*QFX)
    u_mass = _u_mass(state)
    v_mass = _v_mass(state)
    return {
        "ncol": ncol, "ny": ny, "nx": nx, "nz": nz,
        "u_mass": u_mass, "v_mass": v_mass,
        "u_cols": _cols(u_mass), "v_cols": _cols(v_mass),
        "theta_cols": _cols(state.theta), "T_cols": _cols(T),
        "qv_cols": _cols(jnp.maximum(state.qv, 0.0)),
        "p_cols": _cols(p), "pii_cols": _cols(pii), "rho_cols": _cols(rho),
        "dz_cols": _cols(dz),
        "p_int_cols": jnp.moveaxis(p_int, 0, -1).reshape(ncol, nz + 1),
        # surface forcing (ncol,)
        "hfx": _flat2d(hfx), "qfx": _flat2d(qfx),
        "psim": _flat2d(diag.psim), "psih": _flat2d(diag.psih),
        "br": _flat2d(diag.br),
        "ust": jnp.maximum(_flat2d(state.ustar), 1.0e-3),
        "znt": jnp.maximum(_flat2d(jnp.maximum(state.roughness_m, 1.0e-4)), 1.0e-7),
        "wspd": jnp.maximum(jnp.sqrt(_flat2d(u_mass[0]) ** 2 + _flat2d(v_mass[0]) ** 2), 0.1),
        "u10": _flat2d(diag.u10), "v10": _flat2d(diag.v10),
        "xland": _flat2d(state.xland),
        # ACM2 needs an INCOMING PBLH guess (pblh_initial) before it diagnoses its
        # own height. The revised-MM5 surface layer does not export a PBLH leaf and
        # State carries none, so seed with the surface-layer's own pblh assumption
        # (1000 m, sf_sfclayrev default) -- ACM2 overwrites it with its diagnosis.
        "pblh": jnp.full((ncol,), 1000.0, dtype=jnp.float64),
        "rhosfc": rhosfc, "cpm": cpm,
    }


def _apply_pbl_increment(state: State, dt: float, tend: dict, *, ny: int, nx: int, nz: int) -> State:
    """Apply batched PBL tendencies (``(ncol, nz)`` rates) WRF-faithfully.

    u/v: A2C-averaged mass-point increment onto the ORIGINAL C-grid faces (WRF
    ``add_a2c_u``/``add_a2c_v``); theta/qv: direct ``state += dt*tend`` on the mass
    grid. Writes at the live State dtype (fp32-defeat fix).
    """

    def _back3d(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    dt_f = float(dt)
    du_mass = dt_f * _back3d(tend["u"])
    dv_mass = dt_f * _back3d(tend["v"])
    u_new = _add_a2c_u_increment(state.u, du_mass).astype(_output_dtype(state, "u"))
    v_new = _add_a2c_v_increment(state.v, dv_mass).astype(_output_dtype(state, "v"))
    return state.replace(
        u=u_new,
        v=v_new,
        theta=(state.theta + dt_f * _back3d(tend["theta"])).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back3d(tend["qv"])).astype(_output_dtype(state, "qv")),
    )


def ysu_pbl_adapter(state: State, dt: float, grid=None) -> State:
    """bl_pbl=1 YSU PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel)."""

    f = _pbl_surface_forcing(state, grid)
    out = ysu_columns(
        f["u_cols"], f["v_cols"], f["T_cols"], f["qv_cols"], f["p_cols"],
        f["p_int_cols"], f["pii_cols"], f["dz_cols"],
        psfc=f["p_cols"][:, 0], znt=f["znt"], ust=f["ust"], hfx=f["hfx"], qfx=f["qfx"],
        wspd=f["wspd"], br=f["br"], psim=f["psim"], psih=f["psih"], dt=float(dt),
        xland=f["xland"], u10=f["u10"], v10=f["v10"],
    )
    return _apply_pbl_increment(state, dt, out, ny=f["ny"], nx=f["nx"], nz=f["nz"])


def acm2_pbl_adapter(state: State, dt: float, grid=None) -> State:
    """bl_pbl=7 ACM2 PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel)."""

    f = _pbl_surface_forcing(state, grid)
    mut = jnp.asarray(state.mu_total if hasattr(state, "mu_total") else state.mu, jnp.float64).reshape(f["ncol"])
    out = acm2_columns(
        f["u_cols"], f["v_cols"], f["theta_cols"], f["T_cols"], f["qv_cols"],
        f["rho_cols"], f["dz_cols"],
        pblh_initial=f["pblh"], ust=f["ust"], hfx=f["hfx"], qfx=f["qfx"],
        wspd=f["wspd"], mut=mut, dt=float(dt), xtime=60.0,
    )
    return _apply_pbl_increment(state, dt, out, ny=f["ny"], nx=f["nx"], nz=f["nz"])


def boulac_pbl_adapter(state: State, dt: float, grid=None) -> State:
    """bl_pbl=8 BouLac PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel)."""

    del grid
    f = _pbl_surface_forcing(state, None)

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(f["ncol"], f["nz"])

    out = boulac_columns(
        f["u_cols"],
        f["v_cols"],
        f["theta_cols"],
        f["qv_cols"],
        _cols(state.qc),
        f["rho_cols"],
        f["dz_cols"],
        jnp.maximum(_cols(state.qke), BOULAC_TKE_MIN),
        hfx=f["hfx"],
        qfx=f["qfx"],
        ust=f["ust"],
        dt=float(dt),
    )
    next_state = _apply_pbl_increment(state, dt, out, ny=f["ny"], nx=f["nx"], nz=f["nz"])

    def _back3d(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(f["ny"], f["nx"], f["nz"]), -1, 0)

    dt_f = float(dt)
    return next_state.replace(
        qc=(state.qc + dt_f * _back3d(out["qc"])).astype(_output_dtype(state, "qc")),
        qke=_back3d(out["tke"]).astype(_output_dtype(state, "qke")),
    )


# --- dispatch tables ----------------------------------------------------------

# Microphysics options whose State->State scan adapter is threaded into the GPU
# scan. mp=8 (Thompson) is the existing physics_couplers.thompson_adapter; mp=0 is
# passive (no microphysics). The rest map to the adapters above.
MP_SCAN_ADAPTERS = {
    1: kessler_adapter,
    6: wsm6_adapter,
    10: morrison_adapter,
    16: wdm6_adapter,
}

# Surface-layer options whose adapter is threaded (sf_sfclay=5 MYNN-sfclay is the
# existing surface_adapter; sf_sfclay=0 disables).
SFCLAY_SCAN_ADAPTERS = {
    1: sfclay_revised_mm5_adapter,
    7: pleim_xiu_sfclay_adapter,
}

# Cumulus options whose adapter is threaded (cu=0 = no cumulus; GF/Tiedtke are
# CPU-reference, not in the GPU scan).
CU_SCAN_ADAPTERS = {
    1: kf_adapter,
}

# PBL options whose scan adapter is threaded (bl=5 MYNN is the existing
# physics_couplers.mynn_adapter; bl=0 disables). YSU(1)/ACM2(7)/BouLac(8) are
# v0.6.0 jax.lax.scan-traceable rewrites -- GPU-operational.
PBL_SCAN_ADAPTERS = {
    1: ysu_pbl_adapter,
    7: acm2_pbl_adapter,
    8: boulac_pbl_adapter,
}


__all__ = [
    "kessler_adapter",
    "wsm6_adapter",
    "morrison_adapter",
    "wdm6_adapter",
    "sfclay_revised_mm5_adapter",
    "pleim_xiu_sfclay_adapter",
    "kf_adapter",
    "ysu_pbl_adapter",
    "acm2_pbl_adapter",
    "boulac_pbl_adapter",
    "initial_kf_carry",
    "MP_SCAN_ADAPTERS",
    "SFCLAY_SCAN_ADAPTERS",
    "CU_SCAN_ADAPTERS",
    "PBL_SCAN_ADAPTERS",
]
