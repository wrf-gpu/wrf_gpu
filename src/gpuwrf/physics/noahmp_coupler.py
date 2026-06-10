"""Noah-MP <-> surface/PBL/dycore coupling adapter (Sprint S6a — INTEGRATION).

The handshake that plugs prognostic Noah-MP into the existing surface->PBL->dycore
chain. LAND-ONLY MASKED: ocean/lake columns keep the current prescribed-SST bulk
``surface_layer_with_diagnostics`` path VERBATIM (byte-for-byte unchanged).

This is a NEW module. ``runtime.operational_mode.surface_adapter`` will call it
once wired; operational_mode.py is NOT edited here (perf-sidecar worktree owns
it). See ADR-NOAHMP-INTERFACES.md §4.

Per physics step (frozen sequence):
  1. sfclay first (UNCHANGED): ``surface_layer_with_diagnostics(state)`` over ALL
     columns -> CH/CM/ustar/tau + water/lake HFX/LH/QFX/T2/Q2. opt_sfc=1: sfclay
     OWNS CH/CM and feeds them INTO Noah-MP (seeded into land_state.cm/ch).
  2. Noah-MP over land: ``noah_mp_step(land_state, forcing, static, dt)`` -> land
     HFX/LH/QFX/TSK/albedo/emiss/ZNT. Noah-MP RE-DERIVES the land-tile CH/CM in its
     own VEGE_FLUX/BARE_FLUX loop (AUTHORITATIVE over land; sfclay CH is NOT forced
     onto the land tile — ADR §4 / S1).
  3. Masked blend (the ONLY land/water flux switch):
       hfx = where(is_land, noahmp.hfx, sfclay.hfx); likewise lh/qfx/tsk/znt.
     Rebuild kinematic handles (theta_flux/qv_flux/fltv) from the BLENDED flux with
     the identical formulae as surface_layer.py:710-715.
  4. PBL bottom BC: the blended SurfaceFluxes is passed as ``surface=`` into the
     MYNN column (the frozen Gate-1 hand-off). mynn_pbl is NOT changed here.
  5. Write-back: return (state', land_state') with state' carrying blended
     t_skin/roughness_m via State.replace; land_state' threaded to next step.

Invariants: no in-loop host transfer; ocean path unchanged (the where(is_land,...)
selection is the sole switch); CH/CM from sfclay over water, Noah-MP-own over land;
dycore + microphysics + State.__slots__ untouched.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.physics.noahmp.types import NoahMPForcing
from gpuwrf.physics.surface_constants import CP_D, EP1, P0_PA, R_D, R_D_OVER_CP
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

# R_v / R_d for the WRF moist-potential-temperature decoupling. MUST match the
# constant the dycore couples with (operational_mode._RVRD = 461.6/287.0) so the
# theta_m -> theta_dry inverse is exact; see assemble_noahmp_forcing.
RVOVRD = 461.6 / R_D


def _get(obj, name, default=None):
    """Attribute or dict-style lookup with a default (tolerant accessor)."""
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


def _surface(field):
    """Lowest model level of a column field, as the 2-D surface (ny, nx).

    Column fields are stored trailing-z ``(ny, nx, nz)`` (the surface-layer/State
    convention), so the lowest level is the last-axis index 0. A field already 2-D
    is returned unchanged.
    """
    a = jnp.asarray(field, dtype=jnp.float64)
    if a.ndim >= 3:
        return a[..., 0]
    return a


def assemble_noahmp_forcing(
    state: Any,
    static: NoahMPStatic,
    radiation: Any,
    clock: Any,
    dt: float,
) -> NoahMPForcing:
    """Build the Noah-MP forcing pytree from device state (no host transfer).

    Pulls the atmosphere lowest level (sfctmp/sfcprs/uu/vv/qair/qc) from ``state``,
    radiation (soldn/lwdn/cosz) from ``radiation``, the microphysics precip
    partition from ``state``/``radiation``, and the clock (julian/yearlen) from
    ``clock``. Missing optional inputs default to physically inert values (no
    precip, neutral radiation) so the adapter is usable before the operational
    radiation/precip plumbing is finalised.
    """
    sfcprs = _surface(_get(state, "p", _get(state, "sfcprs")))
    psfc = _surface(_get(state, "psfc", sfcprs))
    uu = _surface(_get(state, "u"))
    vv = _surface(_get(state, "v"))
    qair = jnp.maximum(_surface(_get(state, "qv", _get(state, "qair"))), 0.0)
    # lowest-level air temperature: prescribed t_air/sfctmp if present, else from
    # theta via the Exner function at the lowest-level pressure (matches sfclay).
    t_air = _get(state, "t_air", _get(state, "sfctmp", None))
    if t_air is not None:
        sfctmp = _surface(t_air)
    else:
        # ``state.theta`` is the WRF MOIST potential temperature
        # theta_m = theta_dry * (1 + R_v/R_d * q_v) (use_theta_m=1; the dycore
        # prognostic -- see operational_mode conv_t_tendf_to_moist, which divides
        # ``before.theta`` by the SAME (1 + _RVRD*qv) to recover dry theta). WRF
        # hands noahmplsm the DRY sensible temperature T3D = t_phy =
        # theta_dry*(p/p0)^kappa (module_sf_noahmpdrv.F:755), so decouple
        # theta_m -> theta_dry BEFORE the Exner conversion. Skipping it left the
        # lowest-level air temperature ~+4 K too warm (= the (1+R_v/R_d*q_v)
        # factor), biasing the whole Noah-MP land-tile surface energy balance.
        theta_m0 = _surface(_get(state, "theta"))
        theta_dry0 = theta_m0 / (1.0 + RVOVRD * qair)
        sfctmp = theta_dry0 * (jnp.maximum(sfcprs, 1.0) / P0_PA) ** R_D_OVER_CP
    qc = _surface(_get(state, "qc", None)) if _get(state, "qc", None) is not None else jnp.zeros_like(qair)
    shape = sfctmp.shape

    def rad2d(name, default):
        v = _get(radiation, name, None)
        return _surface(v) if v is not None else jnp.full(shape, float(default))

    soldn = jnp.maximum(rad2d("soldn", 0.0), 0.0)
    lwdn = rad2d("lwdn", 0.0)
    cosz = rad2d("cosz", 0.0)

    def prc2d(name):
        v = _get(state, name, None)
        if v is None:
            v = _get(radiation, name, None)
        return _surface(v) if v is not None else jnp.zeros(shape)

    zero = jnp.zeros(shape)
    julian = jnp.asarray(float(_get(clock, "julian", 1.0)))
    yearlen = jnp.asarray(float(_get(clock, "yearlen", 365.0)))
    zlvl_field = _get(state, "zlvl", None)
    if zlvl_field is not None:
        zlvl = _surface(zlvl_field)
    else:
        dz_field = _get(state, "dz", None)
        dz = _surface(dz_field) if dz_field is not None else jnp.full(shape, 100.0)
        zlvl = 0.5 * dz

    return NoahMPForcing(
        sfctmp=sfctmp, sfcprs=sfcprs, psfc=psfc, uu=uu, vv=vv, qair=qair, qc=qc,
        soldn=soldn, lwdn=lwdn,
        prcpconv=prc2d("prcpconv"), prcpnonc=prc2d("prcpnonc"),
        prcpsnow=prc2d("prcpsnow"), prcpgrpl=prc2d("prcpgrpl"), prcphail=prc2d("prcphail"),
        cosz=cosz, zlvl=zlvl, julian=julian, yearlen=yearlen,
    )


def noahmp_surface_adapter(
    state: Any,
    land_state: NoahMPLandState,
    static: NoahMPStatic,
    radiation: Any = None,
    clock: Any = None,
    dt: float = 1.0,
    forcing: NoahMPForcing | None = None,
    energy_params: Any = None,
    rad_params: Any = None,
    first_timestep: Any = False,
) -> tuple[Any, NoahMPLandState, SurfaceFluxes]:
    """Run the land-masked Noah-MP / sfclay blend for one physics step.

    Returns ``(state', land_state', blended_surface_fluxes)``. Ocean/lake columns
    take the sfclay branch unchanged (the ``where(is_land,...)`` selection is the
    sole land/water switch). ``forcing`` may be supplied pre-assembled; otherwise
    it is built from ``state``/``radiation``/``clock`` via
    :func:`assemble_noahmp_forcing`.

    ``energy_params``/``rad_params`` (S6b ACTIVATE) may be supplied pre-built so the
    operational scan never re-runs the (concrete-``nroot``) ``build_energy_params``
    inside jit; when None the driver builds them itself (the eager S6a gate path).
    """
    # ---- 1. sfclay over ALL columns (UNCHANGED formulae). ``first_timestep``
    #         engages the WRF MYNN surface FIRST-CALL semantics (UST first guess,
    #         MOL=0, land QSFC, Li_etal_2010 z/L seed) on the Noah-MP path too;
    #         without it the blend ran the warm-call branch at step 1 while the
    #         standalone surface slot ran the fixed first-call branch. ----
    diag = surface_layer_with_diagnostics(state, first_timestep=first_timestep)
    sf = diag.fluxes                          # SurfaceFluxes (kinematic)
    rhosfc = jnp.asarray(sf.rhosfc, dtype=jnp.float64)

    # is_land mask (xland: 1 land / 2 water) — identical convention to sfclay.
    xland = _surface(_get(state, "xland", jnp.ones_like(rhosfc)))
    is_land = (xland - 1.5) < 0.0

    # ---- 2. Noah-MP over land. sfclay SEEDS CH/CM into the land carry; Noah-MP
    #         RE-DERIVES the authoritative land-tile CH/CM internally (ADR §4). ----
    if forcing is None:
        forcing = assemble_noahmp_forcing(state, static, radiation, clock, dt)
    # seed sfclay CH/CM (opt_sfc=1 supplies the drag coeffs Noah-MP consumes).
    ch_seed = _surface(_get(diag, "ch", land_state.ch))
    cm_seed = _surface(_get(diag, "cm", land_state.cm))
    land_state = land_state.replace(ch=ch_seed, cm=cm_seed)

    land_state_out, nm = noah_mp_step(
        land_state, forcing, static, dt,
        energy_params=energy_params, rad_params=rad_params,
    )

    # ---- 3. masked blend (land vs water). Water path = sfclay diagnostics. ----
    # rho*cpm with the WRF MYNN moist heat capacity (surface_layer.py): cpm =
    # CP_D*(1+0.84*qx) (module_sf_mynn.F:552). theta_flux = hfx/(rho*cpm) inverts the
    # MYNN-SL flux mapping, so the coefficient MUST match surface_layer.py (0.84).
    qx = jnp.maximum(_surface(_get(state, "qv", jnp.zeros_like(rhosfc))), 0.0)
    rho_cpm = rhosfc * (CP_D * (1.0 + 0.84 * qx))

    # blended physical fluxes (W/m2 and kg/m2/s)
    hfx_water = jnp.asarray(diag.hfx, dtype=jnp.float64)
    lh_water = jnp.asarray(diag.lh, dtype=jnp.float64)
    qfx_water = rhosfc * jnp.asarray(sf.qv_flux, dtype=jnp.float64)
    znt_water = jnp.asarray(diag.znt, dtype=jnp.float64)
    tsk_water = _surface(_get(state, "t_skin", jnp.asarray(nm.tsk)))

    hfx = jnp.where(is_land, jnp.asarray(nm.hfx), hfx_water)
    qfx = jnp.where(is_land, jnp.asarray(nm.qfx), qfx_water)
    znt = jnp.where(is_land, jnp.asarray(nm.znt), znt_water)
    tsk = jnp.where(is_land, jnp.asarray(nm.tsk), tsk_water)
    # blended LH (diagnostic; the PBL bottom BC is rebuilt from HFX/QFX below).
    lh = jnp.where(is_land, jnp.asarray(nm.lh), lh_water)  # noqa: F841

    # ---- 4. rebuild kinematic handles from the BLENDED flux (surface_layer.py
    #         :710-715), so the MYNN bottom BC sees the land flux over land. ----
    thx = jnp.asarray(_thx(state, rhosfc), dtype=jnp.float64)
    theta_flux = hfx / jnp.maximum(rho_cpm, 1.0e-12)
    qv_flux = qfx / jnp.maximum(rhosfc, 1.0e-12)
    fltv = (1.0 + EP1 * qx) * theta_flux + EP1 * thx * qv_flux

    blended = SurfaceFluxes(
        ustar=sf.ustar,            # momentum: sfclay owns ustar/tau everywhere (opt_sfc=1)
        theta_flux=theta_flux,
        qv_flux=qv_flux,
        tau_u=sf.tau_u,
        tau_v=sf.tau_v,
        rhosfc=rhosfc,
        fltv=fltv,
        xland=sf.xland,            # carry land/sea mask through to MYNN mixing length
    )

    # ---- 5. state write-back (blended t_skin / roughness_m). State.__slots__
    #         unchanged; qsfc written only if the State carries that slot. ----
    updates = {"t_skin": _broadcast_like(state, "t_skin", tsk),
               "roughness_m": _broadcast_like(state, "roughness_m", znt)}
    if hasattr(state, "qsfc"):
        updates["qsfc"] = _broadcast_like(state, "qsfc", jnp.asarray(nm.qsfc))
    state_out = state.replace(**updates) if hasattr(state, "replace") else state

    return state_out, land_state_out, blended


def _thx(state, rhosfc):
    """Lowest-level potential temperature (theta), as a 2-D field for fltv."""
    th = _get(state, "theta", None)
    if th is not None:
        return _surface(th)
    return jnp.full_like(rhosfc, 300.0)


def _broadcast_like(state, name, value):
    """Match a 2-D blended field to the State field's stored shape (t_skin/znt are
    2-D surface fields, so this is normally an identity / reshape)."""
    cur = _get(state, name, None)
    if cur is None:
        return value
    cur = jnp.asarray(cur)
    if cur.shape == value.shape:
        return value
    return value.reshape(cur.shape) if value.size == cur.size else jnp.broadcast_to(value, cur.shape)


__all__ = ["assemble_noahmp_forcing", "noahmp_surface_adapter"]
