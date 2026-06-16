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

* **Microphysics** -- Kessler (1), Purdue-Lin (2), WSM3 (3), WSM5 (4), WSM6 (6),
  Morrison (10), WDM6 (16) are pure
  ``jnp`` column kernels batched over ``(ncol, nlev)`` returning a
  ``PhysicsTendency`` of ``state_replacements``. They are jit/vmap-traceable and
  wire into the GPU scan as drop-in microphysics-slot adapters (the same slot
  ``thompson_adapter`` fills for mp=8).
* **Surface layer** -- revised-MM5 (1) and Pleim-Xiu (7) ``*_run`` paths are
  vectorized ``jnp`` and write the SAME B2 surface-flux handles
  (``ustar``/``theta_flux``/``qv_flux``/``tau_u``/``tau_v``/``rhosfc``/``fltv``)
  that ``surface_adapter`` writes for sf_sfclay=5, so they drop into the
  surface-layer slot.
* **Cumulus** -- KF (1) and BMJ (2) are jit/vmap-traceable per-column kernels;
  their adapters vmap the column step over the grid and thread scheme-specific
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
* **Grell-Freitas (3) cumulus** -- faithful CPU-NumPy reference port
  (``gpu_runnable=False``); excluded from the GPU scan by design (GPU-batching of
  the sequential 16-member closure ensemble + beta-PDF gamma is a post-0.9.0 TODO).
* **New-Tiedtke (16) cumulus** -- interface-compatible but NOT separately
  savepoint-gated by a distinct WRF source path, so it stays fail-closed.

  (NOTE: modified-Tiedtke ``cu=6`` IS wired -- it is the v0.6.0 GPU-batched jit/vmap
  ``tiedtke_adapter`` below, a stateless ``State -> State`` scan adapter in
  ``CU_STATELESS_SCAN_ADAPTERS``; only GF(3) and New-Tiedtke(16) remain unwired.)

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
from gpuwrf.physics.microphysics_lin import lin_physics_tendency
from gpuwrf.physics.microphysics_morrison import morrison_tendency
from gpuwrf.physics.microphysics_wdm5 import wdm5_physics_tendency
from gpuwrf.physics.microphysics_wdm6 import wdm6_physics_tendency
from gpuwrf.physics.microphysics_wdm7 import wdm7_physics_tendency
from gpuwrf.physics.microphysics_wsm3 import wsm3_physics_tendency
from gpuwrf.physics.microphysics_wsm5 import wsm5_physics_tendency
from gpuwrf.physics.microphysics_wsm6 import wsm6_physics_tendency
from gpuwrf.physics.microphysics_wsm7 import wsm7_physics_tendency
from gpuwrf.physics.cumulus_bmj import initial_bmj_cldefi, step_bmj_column
from gpuwrf.physics.cumulus_kf import step_kf_column
from gpuwrf.physics.cumulus_tiedtke_jax import tiedtke_column_jax
from gpuwrf.physics._gf_jax import gfdrv_batched
from gpuwrf.physics.pbl_acm2 import acm2_columns
from gpuwrf.physics.pbl_boulac import TEMIN as BOULAC_TKE_MIN, boulac_columns
from gpuwrf.physics.pbl_ysu import ysu_columns
from gpuwrf.physics.bl_gfs import gfs_columns
from gpuwrf.physics.bl_mrf import mrf_columns
from gpuwrf.physics.sfclay_pleim_xiu import step_pxsfclay_column
from gpuwrf.physics.sfclay_revised_mm5 import step_sfclay_revised_mm5_column
from gpuwrf.physics.sfclay_old_mm5 import sfclay_old_mm5_columns
from gpuwrf.physics.sfclay_gfs import sf_gfs_columns
from gpuwrf.physics.surface_constants import CP_D, EP1, R_D
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


def lin_adapter(state: State, dt: float, grid=None) -> State:
    """mp=2 Purdue-Lin single-moment 6-class microphysics scan adapter.

    Lin's terminal-velocity sedimentation is Courant-limited on the geometric
    LEVEL HEIGHT ``z`` (not just the thickness ``dz``), so this adapter passes
    the mass-level height (mean of the bounding interface heights from ``ph/g``)
    in addition to ``dz``. The scheme is theta-based (no Exner conversion of the
    moist update) and returns the 6 moist species + surface precip.
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2  # (nz+1, ny, nx)
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = lin_physics_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        mp(pii), mp(rho), mp(state.p), mp(z_mass), mp(dz), float(dt),
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


def wsm7_adapter(state: State, dt: float, grid=None) -> State:
    """mp=24 WSM7 single-moment 7-class microphysics scan adapter.

    WSM7 = WSM6 (rain/snow/graupel) plus a SEPARATE precipitating hail class. It
    consumes/produces the ADR-032 hail mixing-ratio leaf ``qh`` and accumulates
    grid-scale hail into the v0.17 ``hail_acc`` surface accumulator (WRF HAILNC).
    Identical shape/Exner/dz prep to ``wsm6_adapter`` with ``qh`` threaded; the
    generic ``_apply_mp_replacements`` writes back qh (state replacement) and
    hail_acc (accumulator increment) -- both are real State leaves in v0.17.
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = wsm7_physics_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg), mp(state.qh),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
    )
    tend.validate_keys()
    return _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)


def wsm3_adapter(state: State, dt: float, grid=None) -> State:
    """mp=3 WSM3 simple-ice microphysics ``State -> State`` scan adapter."""

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = wsm3_physics_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(_w_mass(state)), mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
    )
    tend.validate_keys()
    return _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)


def wsm5_adapter(state: State, dt: float, grid=None) -> State:
    """mp=4 WSM5 single-moment 5-class microphysics ``State -> State`` scan adapter."""

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = wsm5_physics_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
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


def wdm5_adapter(state: State, dt: float, grid=None) -> State:
    """mp=14 WDM5 double-moment warm-rain 5-class microphysics scan adapter.

    WDM5 reuses the WDM6 number leaves ``Nc`` (cloud-droplet number) and ``Nn``
    (CCN); no new State leaf is introduced. The kernel returns ``Nc``/``Nr`` as
    replacements and ``Nn`` as a diagnostic; this adapter materializes ``Nn`` into
    the State leaf so the CCN prognostic actually evolves (it would otherwise be
    lost). WDM5 has NO graupel and NO land/sea ``qcr`` switch, so (unlike WDM6) it
    takes no ``slmsk`` argument.
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = wdm5_physics_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs),
        mp(state.Nn), mp(state.Nc), mp(state.Nr),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
    )
    tend.validate_keys()
    next_state = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    # Nn is returned in diagnostics (the kernel reuses the WDM6 number layout but
    # returns Nn as a diagnostic); thread it into the State leaf so CCN evolves.
    nn = tend.diagnostics.get("Nn")
    if nn is not None:
        nn3d = jnp.moveaxis(jnp.asarray(nn).reshape(ny, nx, nz), -1, 0)
        next_state = next_state.replace(Nn=nn3d.astype(_output_dtype(state, "Nn")))
    return next_state


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
    # The WDM6 kernel vmaps over columns and accepts one scalar land/sea mask per
    # column; keep the existing 1/0 values without materializing a full column.
    slmsk = jnp.asarray(slmsk_2d).reshape(ny * nx)
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


def wdm7_adapter(state: State, dt: float, grid=None) -> State:
    """mp=26 WDM7 double-moment warm-rain + single-moment hail scan adapter.

    WDM7 = WDM6 (double-moment cloud/rain number Nc/Nr + CCN Nn) plus a SEPARATE
    single-moment precipitating hail class ``qh`` (no hail number). Identical
    prep to ``wdm6_adapter`` (slmsk land/sea mask, Exner, dz) with ``qh`` threaded
    and the ``hail_acc`` surface accumulator (WRF HAILNC) advanced by the generic
    ``_apply_mp_replacements``. As in WDM6, the kernel returns ``Nn`` as a
    diagnostic; this adapter materializes it into the State leaf so CCN evolves.
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    # WDM7 kernel uses slmsk=1 land / 2 water (it tests slmsk==2.0 for the water
    # qc0 autoconversion threshold, matching proofs/v013_wdm7 SLMSK); derive it
    # from xland (1 land / 2 water).
    slmsk = jnp.where(jnp.asarray(state.xland) < 1.5, 1.0, 2.0).reshape(ny * nx)
    tend = wdm7_physics_tendency(
        mp(state.theta), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg), mp(state.qh),
        mp(state.Nn), mp(state.Nc), mp(state.Nr),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt), slmsk,
    )
    tend.validate_keys()
    next_state = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
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


# v0.13 Tier-3 surface-layer ports. Both produce HFX/QFX/USTAR/U10/V10 directly
# (the WRF in-place style); the helper below converts those to the SAME B2
# kinematic flux handles (theta_flux/qv_flux/tau_u/tau_v/rhosfc/fltv) that the
# revised-MM5 / MYNN adapters write, so they drop into the surface-layer slot.

def _flux_handles_from_hfx_qfx(state: State, *, hfx, qfx, ust, cpm) -> State:
    """Map a scheme's HFX/QFX/USTAR onto the B2 kinematic surface handles.

    Mirrors ``sfclay_revised_mm5.py`` (theta_flux=hfx/(rho*cpm), qv_flux=qfx/rho,
    tau_u/tau_v=-ust^2*u/|wind|, fltv=(1+ep1*qv)*theta_flux + ep1*thx*qv_flux).
    """

    inp = _surface_layer_bottom_inputs(state)
    T0 = _temperature_from_theta(state.theta, state.p)[0]
    rhox = state.p[0] / (R_D * T0 * (1.0 + EP1 * state.qv[0]))
    theta_flux = hfx / jnp.maximum(rhox * cpm, 1.0e-12)
    qv_flux = qfx / jnp.maximum(rhox, 1.0e-12)
    u_b = inp["u"]
    v_b = inp["v"]
    wind = jnp.maximum(jnp.sqrt(u_b * u_b + v_b * v_b), 0.1)
    tau_u = -(ust * ust) * u_b / wind
    tau_v = -(ust * ust) * v_b / wind
    thx = T0 * (P0_PA / state.p[0]) ** R_D_OVER_CP
    fltv = (1.0 + EP1 * state.qv[0]) * theta_flux + EP1 * thx * qv_flux
    updates = {
        "ustar": ust, "theta_flux": theta_flux, "qv_flux": qv_flux,
        "tau_u": tau_u, "tau_v": tau_v, "rhosfc": rhox, "fltv": fltv,
    }
    return state.replace(**{k: jnp.asarray(v).astype(_output_dtype(state, k))
                            for k, v in updates.items()})


def sfclay_old_mm5_adapter(state: State, dt: float, grid=None) -> State:
    """sf_sfclay=91 old-MM5 surface layer scan adapter (writes B2 flux handles)."""

    del dt
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    inp = _surface_layer_bottom_inputs(state)
    z = jnp.zeros_like(state.t_skin)
    out = sfclay_old_mm5_columns(
        inp["u"], inp["v"], inp["temperature"], inp["qv"], inp["pressure"], inp["dz"],
        inp["psfc"], state.t_skin, state.xland,
        jnp.maximum(state.roughness_m, 1.0e-4), jnp.maximum(state.ustar, 1.0e-3),
        z,                                       # mol carry (0 = no prior unstable)
        jnp.full_like(state.t_skin, -1.0),       # qsfc<0 -> recomputed from tsk
        jnp.full_like(state.t_skin, 1000.0),     # pblh (default, as revised-MM5)
        state.mavail, state.lakemask,
        jnp.full_like(state.t_skin, dx),
        z, z,                                    # hfx_in/qfx_in (0 on entry)
        isfflx=1,
    )
    cpm = CP_D * (1.0 + 0.8 * state.qv[0])
    return _flux_handles_from_hfx_qfx(
        state, hfx=out["hfx"], qfx=out["qfx"], ust=out["ust"], cpm=cpm
    )


def gfs_sfclay_adapter(state: State, dt: float, grid=None) -> State:
    """sf_sfclay=3 NCEP-GFS surface layer scan adapter (writes B2 flux handles)."""

    del dt, grid
    inp = _surface_layer_bottom_inputs(state)
    # GFS derives its reference height Z1 from the hydrostatic thickness
    # -RD*Tv*log(p_mid/psfc)/g, so it needs a genuine surface pressure ABOVE the
    # lowest-level pressure (psfc==p_mid -> Z1=0 -> NaN). Derive psfc by adding the
    # hydrostatic weight of the half lowest layer below the mass point:
    # psfc = p_mid + rho * g * 0.5 * dz.
    T0 = inp["temperature"]
    rhox = inp["pressure"] / (R_D * T0 * (1.0 + EP1 * inp["qv"]))
    psfc = inp["pressure"] + rhox * GRAVITY_M_S2 * 0.5 * inp["dz"]
    out = sf_gfs_columns(
        inp["u"], inp["v"], inp["temperature"], inp["qv"], inp["pressure"],
        psfc, state.t_skin, state.xland,
        jnp.maximum(state.roughness_m, 1.0e-4), jnp.maximum(state.ustar, 1.0e-3),
        isfflx=1,
    )
    return _flux_handles_from_hfx_qfx(
        state, hfx=out["hfx"], qfx=out["qfx"], ust=out["ust"], cpm=out["cpm"]
    )


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


def tiedtke_adapter(
    state: State,
    dt: float,
    grid=None,
    *,
    stepcu: int = 1,
    qvften: jax.Array | None = None,
    qvpblten: jax.Array | None = None,
) -> State:
    """cu=6 modified-Tiedtke cumulus ``State -> State`` scan adapter.

    Tiedtke is a per-column kernel (``cumulus_tiedtke_jax.tiedtke_column_jax``); the
    adapter vmaps it over the ``(ny*nx)`` grid columns. Unlike KF, modified-Tiedtke
    carries NO persistent cumulus state, so this is a plain ``State -> State``
    adapter (like the microphysics adapters).

    Tendencies are applied ``state += dt*tend`` (WRF ``RTHCUTEN``/``RQ*CUTEN`` are
    rates over the cumulus step); ``RAINCV`` (mm/step) accumulates into
    ``rainc_acc``. The cumulus call cadence ``stepcu`` defaults to 1 in the
    operational scan (the cumulus slot is invoked every dynamics step here; the
    savepoint oracle uses WRF's STEPCU=5 -- the kernel multiplies dt by stepcu
    internally, so ``stepcu=1, dt=dt`` is the every-step coupling).

    The Tiedtke-specific inputs not in the frozen B2 contract are assembled here,
    pure-``jnp``, no host transfer:
      * ``P8W`` interface pressure (nz+1): interior faces = mean of adjacent
        levels, edges by zero-gradient (same assembler the YSU/ACM2 adapter uses).
      * ``ZNU`` eta at mass levels: WRF eta relation ``(p - p_top)/(p_sfc - p_top)``
        per column (used only by the below-cloud rain-evaporation coefficient).
      * ``QVFTEN`` / ``QVPBLTEN`` advective / PBL moisture forcing: threaded by
        the operational runtime from the WRF flux-form qv-advection diagnostic
        and the PBL-slot qv increment.  Direct adapter calls may omit them, in
        which case they remain zero for isolated kernel tests.
      * ``QFX`` surface moisture flux from the B2 kinematic ``qv_flux`` handle.
    """

    nz, ny, nx = state.theta.shape
    rho = _rho_from_state(state)
    T = _temperature_from_theta(state.theta, state.p)
    pii = _exner_columns(state.p)
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz_full = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)  # (nz, ny, nx)
    w_mass = _w_mass(state)

    p = state.p.astype(jnp.float64)
    p_int_interior = 0.5 * (p[:-1] + p[1:])  # (nz-1, ny, nx)
    p8w = jnp.concatenate([p[:1], p_int_interior, p[-1:]], axis=0)  # (nz+1, ny, nx)
    # eta-coordinate proxy (mass levels): (p - p_top)/(p_sfc - p_top), clipped [0,1].
    p_top = jnp.maximum(p8w[-1], 1.0)
    p_sfc = jnp.maximum(p8w[0], p_top + 1.0)
    znu = jnp.clip((p - p_top) / (p_sfc - p_top), 0.0, 1.0)  # (nz, ny, nx)

    # W on (nz+1) interfaces: vertical velocity at faces (state.w is C-grid w).
    w_int = jnp.asarray(state.w, jnp.float64)
    if w_int.shape[0] == nz:  # mass-level w fallback: pad a top face
        w_int = jnp.concatenate([w_int, w_int[-1:]], axis=0)

    qv_flux = jnp.asarray(state.qv_flux, jnp.float64)  # kg kg^-1 m s^-1 (kinematic)
    rhosfc = jnp.asarray(state.rhosfc, jnp.float64)
    qfx_2d = qv_flux * rhosfc  # kg m^-2 s^-1
    xland_2d = jnp.asarray(state.xland, jnp.float64)

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)

    def _cols1(field3d):  # (nz+1, ny, nx) -> (ncol, nz+1)
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz + 1)

    T_c = _cols(T)
    qv_c = _cols(jnp.maximum(state.qv, 0.0))
    qc_c = _cols(jnp.maximum(state.qc, 0.0))
    qi_c = _cols(jnp.maximum(state.qi, 0.0))
    p_c = _cols(p)
    p8w_c = _cols1(p8w)
    dz_c = _cols(dz_full)
    rho_c = _cols(rho)
    pii_c = _cols(pii)
    u_c = _cols(_u_mass(state))
    v_c = _cols(_v_mass(state))
    w_c = _cols1(w_int)
    qvften_c = _cols(
        jnp.zeros_like(state.qv) if qvften is None else jnp.asarray(qvften, jnp.float64)
    )
    qvpblten_c = _cols(
        jnp.zeros_like(state.qv) if qvpblten is None else jnp.asarray(qvpblten, jnp.float64)
    )
    znu_c = _cols(znu)
    qfx_c = qfx_2d.reshape(ny * nx)
    xland_c = xland_2d.reshape(ny * nx)
    dt_f = float(dt)

    def _one(T0, QV0, QC0, QI0, P0, P8W0, DZ0, RHO0, PI0, U0, V0, W0,
             QVF0, QVB0, QFX0, XL0, ZNU0):
        out = tiedtke_column_jax(
            T0, QV0, QC0, QI0, P0, P8W0, DZ0, RHO0, PI0, U0, V0, W0,
            QVF0, QVB0, QFX0, XL0, ZNU0, dt_f, stepcu=int(stepcu),
        )
        return (out["RTHCUTEN"], out["RQVCUTEN"], out["RQCCUTEN"], out["RQRCUTEN"],
                out["RQICUTEN"], out["RQSCUTEN"], out["RUCUTEN"], out["RVCUTEN"],
                out["RAINCV"])

    (rth, rqv, rqc, rqr, rqi, rqs, ru, rv, raincv) = jax.vmap(_one)(
        T_c, qv_c, qc_c, qi_c, p_c, p8w_c, dz_c, rho_c, pii_c, u_c, v_c, w_c,
        qvften_c, qvpblten_c, qfx_c, xland_c, znu_c,
    )

    def _back(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    # Momentum increment from RUCUTEN/RVCUTEN onto C-grid faces (A2C, like PBL).
    du_mass = dt_f * _back(ru)
    dv_mass = dt_f * _back(rv)
    u_new = _add_a2c_u_increment(state.u, du_mass).astype(_output_dtype(state, "u"))
    v_new = _add_a2c_v_increment(state.v, dv_mass).astype(_output_dtype(state, "v"))

    next_state = state.replace(
        u=u_new,
        v=v_new,
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
    return next_state


def _kpbl_bulk_richardson(thv_c: jax.Array, z_mass_c: jax.Array) -> jax.Array:
    """Diagnose the PBL-top mass level (1-based, GF convention) per column.

    GF's driver (``module_cu_gf_wrfdrv.F``) receives ``KPBL`` from the active PBL
    scheme. The operational scan does not thread a PBL-height leaf, so the cumulus
    slot reconstructs it the way every WRF PBL driver does: the lowest mass level
    whose bulk Richardson number relative to the surface level exceeds a critical
    value ``Ricr=0.25`` (YSU/ACM2 ``BRCR``). ``thv`` is virtual potential
    temperature; ``z_mass`` mass-level heights above ground. Returns a 1-based
    level index in ``[2, nz]`` (GF arrays are 1-based, surface = level 1), so the
    PDF/inversion logic sees a physically-located PBL top, not a fabricated const.

    Inputs are 0-based ``(ncol, nz)``; the +1 shift to GF's 1-based indexing is
    applied by the caller (which prepends the dummy level-0).
    """

    ricr = 0.25
    ncol, nz = thv_c.shape
    thv_sfc = thv_c[:, :1]  # (ncol, 1)
    dz = jnp.maximum(z_mass_c - z_mass_c[:, :1], 1.0)  # height above surface
    # Bulk Richardson vs surface (no shear term -> thermal-only BRN, WRF's
    # convective limit; the PBL-top estimate GF needs is robust to this).
    rib = (9.81 / jnp.maximum(thv_sfc, 1.0)) * (thv_c - thv_sfc) * dz / jnp.maximum(dz, 1.0)
    above = (rib > ricr) & (jnp.arange(nz)[None, :] >= 1)  # exclude surface level
    any_above = jnp.any(above, axis=1)
    first0 = jnp.argmax(above.astype(jnp.int32), axis=1)  # 0-based first crossing
    # 0-based -> GF 1-based (+1); default to level 2 when no crossing (shallow PBL).
    kpbl1 = jnp.where(any_above, first0 + 1, 1) + 1
    kpbl1 = jnp.clip(kpbl1, 2, nz).astype(jnp.int32)
    return kpbl1


def gf_adapter(state: State, dt: float, grid=None, *, ishallow_g3: int = 1,
               ichoice: int = 0) -> State:
    """cu=3 Grell-Freitas scale-aware cumulus ``State -> State`` scan adapter.

    Grell-Freitas is a per-column kernel (``physics._gf_jax.gfdrv_column``, the
    GPU-batched jit/vmap port of WRF ``GFDRV -> cup_gf (deep) + cup_gf_sh
    (shallow)``); this adapter ``jax.vmap``s it over the ``(ny*nx)`` grid columns
    via ``gfdrv_batched`` -- the whole deep+shallow column physics (16-member
    closure ensemble + beta-PDF gamma + scale-aware ``sig``) runs inside ONE
    vmapped jit, no host transfer inside the column loop. Like modified-Tiedtke,
    GF carries NO persistent cumulus state, so this is a plain ``State -> State``
    adapter (no carry, threaded via ``CU_STATELESS_SCAN_ADAPTERS``).

    Tendencies are applied ``state += dt*tend``: ``RTHCUTEN`` is the THETA tendency
    (K s^-1; the kernel already divides the temperature tendency by Exner), and
    ``RQVCUTEN/RQCCUTEN/RQICUTEN`` are mixing-ratio tendencies (kg kg^-1 s^-1).
    ``RAINCV`` is the per-step convective precip (mm; the kernel returns
    ``pratec*dt``) and accumulates into ``rainc_acc`` (exactly the KF/Tiedtke
    accumulation convention). GF momentum tendencies are diagnostic in the WRF
    GFDRV path (not added to U/V there), so this adapter writes none -- matching
    the kernel's ``gfdrv_column`` output set (no RUCUTEN/RVCUTEN).

    The GF kernel uses 1-based length-(nz+1) column arrays (index 0 unused,
    level 1 = surface) -- the operational State is bottom-up ``(nz, ny, nx)``
    (index 0 = surface), so each column is mapped by prepending a dummy level-0.

    GF-specific inputs not in the frozen B2 contract are assembled here, pure
    ``jnp``, no host transfer:
      * ``HFX`` surface sensible-heat flux (W m^-2) = ``rho_sfc*cp*theta_flux``;
        ``QFX`` surface moisture flux (kg m^-2 s^-1) = ``rho_sfc*qv_flux`` (the
        kinematic B2 ``theta_flux``/``qv_flux`` handles -> GF flux units).
      * ``HT`` terrain height (m) = surface geopotential ``ph[0]/g``.
      * ``DX`` grid spacing (m) for the scale-aware factor (from the grid, like KF;
        defaults to 3 km when no grid is supplied).
      * ``KPBL`` PBL-top mass level, bulk-Richardson-diagnosed per column
        (``_kpbl_bulk_richardson``) -- the value the active PBL scheme would hand
        GF in WRF (no PBL-height leaf is threaded in this scan).
      * ``RTHBLTEN``/``RQVBLTEN`` PBL theta/qv forcing tendencies: zero in the
        per-slot scan (no separate PBL forcing tracked into the cumulus slot here;
        the PBL slot already applied its tendency to State the same step -- WRF
        folds this forcing via the "forced sounding", documented carry-over,
        IDENTICAL to the Tiedtke adapter's zero QVFTEN/QVPBLTEN treatment). With
        zero forcing the forced sounding collapses to the current sounding and GF
        triggers on the actual column state.
    """

    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    nz, ny, nx = state.theta.shape
    rho = _rho_from_state(state)
    T = _temperature_from_theta(state.theta, state.p)
    pii = _exner_columns(state.p)
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2  # (nz+1, ny, nx)
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])  # (nz, ny, nx)
    dz_full = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    ht_2d = interface_z[0]  # surface geopotential height (m), (ny, nx)
    w_mass = _w_mass(state)

    # Surface fluxes in GF units (W m^-2 / kg m^-2 s^-1) from kinematic handles.
    rhosfc = jnp.asarray(state.rhosfc, jnp.float64)
    theta_flux = jnp.asarray(state.theta_flux, jnp.float64)  # K m s^-1
    qv_flux = jnp.asarray(state.qv_flux, jnp.float64)        # kg kg^-1 m s^-1
    hfx_2d = rhosfc * CP_DRY * theta_flux                    # W m^-2
    qfx_2d = rhosfc * qv_flux                                # kg m^-2 s^-1
    xland_2d = jnp.asarray(state.xland, jnp.float64)

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)

    # Virtual potential temperature for the bulk-Richardson PBL-top diagnosis.
    qv_pos = jnp.maximum(state.qv, 0.0)
    thv = state.theta.astype(jnp.float64) * (1.0 + 0.608 * qv_pos)
    kpbl_c = _kpbl_bulk_richardson(_cols(thv), _cols(z_mass))  # (ncol,), 1-based

    # 0-based (ncol, nz) -> GF 1-based (ncol, nz+1): prepend a dummy level-0.
    def _cols1(field3d):
        c = _cols(field3d)  # (ncol, nz)
        return jnp.concatenate([jnp.zeros((ny * nx, 1), jnp.float64), c], axis=1)

    T_c = _cols1(T)
    qv_c = _cols1(jnp.maximum(state.qv, 0.0))
    p_c = _cols1(state.p)
    pii_c = _cols1(pii)
    dz_c = _cols1(dz_full)
    rho_c = _cols1(rho)
    u_c = _cols1(_u_mass(state))
    v_c = _cols1(_v_mass(state))
    w_c = _cols1(w_mass)
    zero_c = jnp.zeros((ny * nx, nz + 1), jnp.float64)  # RTHBLTEN/RQVBLTEN forcing
    dt_f = float(dt)
    kx = nz

    dt_b = jnp.full((ny * nx,), dt_f, jnp.float64)
    dx_b = jnp.full((ny * nx,), dx, jnp.float64)
    hfx_b = hfx_2d.reshape(ny * nx)
    qfx_b = qfx_2d.reshape(ny * nx)
    xland_b = xland_2d.reshape(ny * nx)
    ht_b = ht_2d.reshape(ny * nx)

    out = gfdrv_batched(
        T_c, qv_c, p_c, pii_c, dz_c, rho_c, u_c, v_c, w_c, zero_c, zero_c,
        kx, dt_b, dx_b, hfx_b, qfx_b, kpbl_c, xland_b, ht_b,
        ishallow_g3=int(ishallow_g3), ichoice=int(ichoice),
    )

    def _back(field2d_1based):  # (ncol, nz+1) -> (nz, ny, nx); drop GF level-0
        c0 = field2d_1based[:, 1:]  # (ncol, nz)
        return jnp.moveaxis(c0.reshape(ny, nx, nz), -1, 0)

    rth = _back(out['RTHCUTEN'])   # K s^-1 (theta tendency)
    rqv = _back(out['RQVCUTEN'])   # kg kg^-1 s^-1
    rqc = _back(out['RQCCUTEN'])
    rqi = _back(out['RQICUTEN'])
    raincv = jnp.asarray(out['RAINCV'], jnp.float64).reshape(ny, nx)  # mm/step

    next_state = state.replace(
        theta=(state.theta + dt_f * rth).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * rqv).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * rqc).astype(_output_dtype(state, "qc")),
        qi=(state.qi + dt_f * rqi).astype(_output_dtype(state, "qi")),
        rainc_acc=(
            jnp.asarray(state.rainc_acc, jnp.float64) + raincv
        ).astype(_output_dtype(state, "rainc_acc")),
    )
    return next_state


def bmj_adapter(state: State, dt: float, cldefi, *, grid=None):
    """cu=2 Betts-Miller-Janjic cumulus scan adapter.

    BMJ writes WRF ``RTHCUTEN``/``RQVCUTEN`` tendencies and deep-convective
    ``RAINCV``.  Its persistent WRF state member is ``CLDEFI`` (cloud
    efficiency), carried as a cumulus sibling tree rather than a dycore leaf.
    """

    del grid
    nz, ny, nx = state.theta.shape
    rho = _rho_from_state(state)
    T = _temperature_from_theta(state.theta, state.p)
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz_full = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    pii = _exner_columns(_mp_in(state.p, ny, nx, nz))

    def _cols(field3d):
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)

    T_c = _cols(T)
    qv_c = _cols(state.qv)
    p_c = _cols(state.p)
    dz_c = _cols(dz_full)
    rho_c = _cols(rho)
    cldefi_c = jnp.asarray(cldefi, jnp.float64).reshape(ny * nx)
    xland_c = jnp.asarray(state.xland, jnp.float64).reshape(ny * nx)

    def _one(T0, QV0, P0, DZQ, RHOE, PII, XLAND, CLDEFI):
        res = step_bmj_column(
            T0,
            QV0,
            P0,
            DZQ,
            RHOE,
            PII,
            float(dt),
            stepcu=1,
            xland=XLAND,
            cldefi=CLDEFI,
        )
        st = res.tendency.state_tendencies
        cc = res.carry.cumulus
        return (
            st["theta"],
            st["qv"],
            res.tendency.accumulator_increments["rainc_acc"],
            cc["cldefi"],
        )

    rth, rqv, raincv, cldefi_next_c = jax.vmap(_one)(
        T_c, qv_c, p_c, dz_c, rho_c, pii, xland_c, cldefi_c
    )

    def _back(field2d):
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    dt_f = float(dt)
    next_state = state.replace(
        theta=(state.theta + dt_f * _back(rth)).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back(rqv)).astype(_output_dtype(state, "qv")),
        rainc_acc=(
            jnp.asarray(state.rainc_acc, jnp.float64) + raincv.reshape(ny, nx)
        ).astype(_output_dtype(state, "rainc_acc")),
    )
    return next_state, cldefi_next_c.reshape(ny, nx)


def initial_bmj_carry(state: State):
    """Seed BMJ ``CLDEFI`` carry with BMJINIT's default value."""

    return initial_bmj_cldefi(state.theta.shape[1:])


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
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])  # (nz, ny, nx) full-level height
    tsk = jnp.asarray(state.t_skin, jnp.float64)  # (ny, nx)
    return {
        "ncol": ncol, "ny": ny, "nx": nx, "nz": nz,
        "u_mass": u_mass, "v_mass": v_mass,
        "u_cols": _cols(u_mass), "v_cols": _cols(v_mass),
        "theta_cols": _cols(state.theta), "T_cols": _cols(T),
        "qv_cols": _cols(jnp.maximum(state.qv, 0.0)),
        "p_cols": _cols(p), "pii_cols": _cols(pii), "rho_cols": _cols(rho),
        "dz_cols": _cols(dz), "z_cols": _cols(z_mass),
        "tsk": _flat2d(tsk),
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


def mrf_pbl_adapter(state: State, dt: float, grid=None) -> State:
    """bl_pbl=99 MRF PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel).

    The MRF (Hong-Pan 1996) nonlocal-K PBL is YSU's predecessor; it consumes the
    SAME revised-MM5 surface-layer forcing the YSU adapter re-derives, plus the
    skin temperature ``tsk`` and the ``gz1oz0 = ln(za_1/znt)`` surface-layer log
    ratio that MRF reads. The momentum increment is applied A2C-faithfully like the
    other PBL adapters.
    """

    del grid
    f = _pbl_surface_forcing(state, None)
    znt = f["znt"]
    za1 = 0.5 * f["dz_cols"][:, 0]  # ~mid-height of the lowest mass level
    gz1oz0 = jnp.log(jnp.maximum(za1, znt) / znt)
    out = mrf_columns(
        f["u_cols"], f["v_cols"], f["T_cols"], f["qv_cols"],
        jnp.zeros_like(f["qv_cols"]),
        f["p_cols"], f["pii_cols"], f["dz_cols"], f["z_cols"],
        psfc=f["p_cols"][:, 0], znt=znt, ust=f["ust"], hfx=f["hfx"], qfx=f["qfx"],
        tsk=f["tsk"], gz1oz0=gz1oz0, wspd=f["wspd"], br=f["br"],
        psim=f["psim"], psih=f["psih"], xland=f["xland"], dt=float(dt),
    )
    # The MRF kernel emits WRF-named tendency RATES (rublten/.../rthblten with the
    # theta tendency already divided by exner, exactly the WRF wrapper convention);
    # map them to the shared (u/v/theta/qv) increment-rate contract.
    tend = {"u": out["rublten"], "v": out["rvblten"],
            "theta": out["rthblten"], "qv": out["rqvblten"]}
    return _apply_pbl_increment(state, dt, tend, ny=f["ny"], nx=f["nx"], nz=f["nz"])


def gfs_pbl_adapter(state: State, dt: float, grid=None) -> State:
    """bl_pbl=3 GFS PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel).

    GFS consumes the same revised-MM5 surface-layer forcing as YSU/MRF plus the
    ``gz1oz0 = ln(za_1/znt)`` log ratio. The kernel emits WRF-named tendency
    rates; map them to the shared PBL increment contract.
    """

    del grid
    f = _pbl_surface_forcing(state, None)
    znt = f["znt"]
    za1 = 0.5 * f["dz_cols"][:, 0]
    gz1oz0 = jnp.log(jnp.maximum(za1, znt) / znt)
    out = gfs_columns(
        f["u_cols"], f["v_cols"], f["T_cols"], f["qv_cols"],
        jnp.zeros_like(f["qv_cols"]),
        f["p_cols"], f["pii_cols"], f["dz_cols"], f["z_cols"],
        psfc=f["p_cols"][:, 0], ust=f["ust"], hfx=f["hfx"], qfx=f["qfx"],
        tsk=f["tsk"], gz1oz0=gz1oz0, psim=f["psim"], psih=f["psih"],
        wspd=f["wspd"], br=f["br"], dt=float(dt),
    )
    tend = {"u": out["rublten"], "v": out["rvblten"],
            "theta": out["rthblten"], "qv": out["rqvblten"]}
    return _apply_pbl_increment(state, dt, tend, ny=f["ny"], nx=f["nx"], nz=f["nz"])


# --- dispatch tables ----------------------------------------------------------

# Microphysics options whose State->State scan adapter is threaded into the GPU
# scan. mp=8 (Thompson) is the existing physics_couplers.thompson_adapter; mp=0 is
# passive (no microphysics). The rest map to the adapters above.
MP_SCAN_ADAPTERS = {
    1: kessler_adapter,
    2: lin_adapter,
    3: wsm3_adapter,
    4: wsm5_adapter,
    6: wsm6_adapter,
    10: morrison_adapter,
    14: wdm5_adapter,
    16: wdm6_adapter,
    # v0.17 WSM7 = WSM6 + separate precipitating hail (qh + hail_acc).
    24: wsm7_adapter,
    # v0.17 WDM7 = WDM6 double-moment + separate single-moment hail.
    26: wdm7_adapter,
}

# Surface-layer options whose adapter is threaded (sf_sfclay=5 MYNN-sfclay is the
# existing surface_adapter; sf_sfclay=0 disables).
SFCLAY_SCAN_ADAPTERS = {
    1: sfclay_revised_mm5_adapter,
    3: gfs_sfclay_adapter,
    7: pleim_xiu_sfclay_adapter,
    91: sfclay_old_mm5_adapter,
}

# Cumulus options whose adapter is threaded (cu=0 = no cumulus). KF (1) is the
# carry-threaded ``(State, carry) -> (State, carry)`` adapter; Tiedtke (6) and
# Grell-Freitas (3) are GPU-batched (jit/vmap) ``State -> State`` adapters (no
# persistent carry). GF (3) is the v0.9.0 GPU-batched port of the scale-aware
# closure-ensemble kernel (physics._gf_jax.gfdrv_batched), savepoint-parity gated
# (proofs/v060/gf_gpubatch_savepoint_parity.json) and scan-wired here.
CU_SCAN_ADAPTERS = {
    1: kf_adapter,
    2: bmj_adapter,
    3: gf_adapter,
    6: tiedtke_adapter,
}

# Cumulus options that carry NO persistent cumulus state (plain State->State, like
# the microphysics adapters). KF (1) and BMJ (2) are excluded -- KF threads
# (w0avg, nca) and BMJ threads CLDEFI. Tiedtke (6) and Grell-Freitas (3) are the
# GPU-batched stateless adapters.
CU_STATELESS_SCAN_ADAPTERS = {
    3: gf_adapter,
    6: tiedtke_adapter,
}

# PBL options whose scan adapter is threaded (bl=5 MYNN is the existing
# physics_couplers.mynn_adapter; bl=0 disables). YSU(1)/ACM2(7)/BouLac(8) are
# v0.6.0 jax.lax.scan-traceable rewrites; MRF(99) is the v0.13 jit/vmap-traceable
# port of phys/module_bl_mrf.F (savepoint-parity gated, proofs/v013/mrf_oracle.py).
PBL_SCAN_ADAPTERS = {
    1: ysu_pbl_adapter,
    3: gfs_pbl_adapter,
    7: acm2_pbl_adapter,
    8: boulac_pbl_adapter,
    99: mrf_pbl_adapter,
}


__all__ = [
    "kessler_adapter",
    "lin_adapter",
    "wsm3_adapter",
    "wsm5_adapter",
    "wsm6_adapter",
    "morrison_adapter",
    "wdm5_adapter",
    "wdm6_adapter",
    "sfclay_revised_mm5_adapter",
    "pleim_xiu_sfclay_adapter",
    "sfclay_old_mm5_adapter",
    "gfs_sfclay_adapter",
    "kf_adapter",
    "tiedtke_adapter",
    "gf_adapter",
    "bmj_adapter",
    "ysu_pbl_adapter",
    "gfs_pbl_adapter",
    "acm2_pbl_adapter",
    "boulac_pbl_adapter",
    "initial_kf_carry",
    "initial_bmj_carry",
    "MP_SCAN_ADAPTERS",
    "SFCLAY_SCAN_ADAPTERS",
    "CU_SCAN_ADAPTERS",
    "CU_STATELESS_SCAN_ADAPTERS",
    "PBL_SCAN_ADAPTERS",
]
