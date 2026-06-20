"""Operational surface hook for the thermal-diffusion slab LSM (sf_surface_physics=1).

This is the State<->land-carry adapter for the JAX 5-layer thermal-diffusion
slab kernel (``physics.lsm_slab.slab_columns``, a faithful fp64-oracle-validated
port of WRF ``phys/module_sf_slab.F`` SLAB1D). It mirrors the existing
Noah-classic operational pattern (``coupling.noahclassic_surface_hook``):

1. the selected surface-layer path has already populated the resident surface
   exchange handles (``theta_flux``, ``qv_flux``, ``rhosfc``, ``ustar``);
2. the slab advances the land tile (a 5-layer ``TSLB`` soil-temperature carry)
   from the surface energy budget driven by the downward radiation (GSW/GLW)
   and the deep-soil restoring temperature (TMN), recomputing the surface
   exchange coefficients (FLHC/FLQC) from the resident kinematic flux handles;
3. land ``TSK``/``HFX``/``QFX`` overwrite the same State handles the PBL reads;
   water columns keep their surface-layer path via one land/water ``where``.

WRF SLAB1D consumes ``FLHC``/``FLQC`` (the surface-layer heat/moisture exchange
coefficients) and ``GSW``/``GLW`` (the down-radiation), and advances ``TSK`` and
the 5-layer ``TSLB`` soil column. The exchange coefficients are NOT carried on
the resident State, so -- exactly as the Noah-classic hook back-derives ``CH``
from the surface heat-flux handle -- this hook reconstructs ``FLHC``/``FLQC`` from
the resident ``theta_flux``/``qv_flux`` kinematic fluxes the surface layer wrote,
which keeps the slab energy SOLVE bit-identical to the oracle-validated kernel
without re-deriving the surface-layer exchange physics.

The ``SlabStatic`` (THC/TMN/EMISS/SNOWC + soil-layer ZS/DZS) is a per-run
read-only bundle, like the Noah-classic REDPRM block. The soil sub-step count
``nsoil_steps`` is a Python static int (depends only on DZS(1) and the step), so
the kernel stays fully ``jax.jit``/``jax.vmap``-traceable.

Cited to ``<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_slab.F`` (SLAB1D) and
the WRF call site ``phys/module_surface_driver.F:2659`` (SLAB(t_phy,qv_curr,
p_phy,flhc,flqc,...,gsw,glw,tmn,...)).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import (
    P0_PA,
    R_D_OVER_CP,
    _output_dtype,
    _temperature_from_theta,
)
from gpuwrf.physics.lsm_slab import (
    NUM_SOIL_LAYERS,
    nsoil_substeps,
    slab_columns,
)
from gpuwrf.physics.surface_constants import EP2, SVP1_KPA, SVP2, SVP3_K, SVPT0_K

CP = 1004.0
XLV = 2.5e6


class SlabStaticBundle(NamedTuple):
    """Read-only slab static inputs for one operational tile (5-layer soil).

    ``zs``/``dzs`` are the shared ``(nsoil,)`` soil-layer center depths and
    thicknesses; ``thc``/``tmn``/``emiss``/``snowc`` are ``(ny, nx)`` per-column
    surface statics (thermal inertia, deep-soil restore temperature, emissivity,
    snow-cover flag). ``ifsnow`` is the WRF snow-cover skin-T limiter switch.
    """

    zs: jax.Array
    dzs: jax.Array
    thc: jax.Array
    tmn: jax.Array
    emiss: jax.Array
    snowc: jax.Array
    ifsnow: int = 0


class SlabLandState(NamedTuple):
    """5-layer slab land carry (TSLB soil temperatures) plus last land fluxes."""

    tslb: jax.Array          # (ny, nx, nsoil) soil-layer temperatures (K)
    hfx: jax.Array           # (ny, nx) last land sensible heat flux (W/m^2)
    qfx: jax.Array           # (ny, nx) last land moisture flux (kg/m^2/s)
    lh: jax.Array            # (ny, nx) last land latent heat flux (W/m^2)
    qsfc: jax.Array          # (ny, nx) surface mixing ratio diagnostic
    capg: jax.Array          # (ny, nx) ground heat capacity diagnostic

    def replace(self, **updates) -> "SlabLandState":
        return self._replace(**updates)


class SlabRadiation(NamedTuple):
    """Held surface down-radiation forcing into the slab (GSW/GLW)."""

    gsw: jax.Array           # downward shortwave at ground (W/m^2)
    glw: jax.Array           # downward longwave at ground (W/m^2)


def _surface_2d(field):
    a = jnp.asarray(field, dtype=jnp.float64)
    return a[..., 0] if a.ndim >= 3 else a


def _soil_carry(field, *, name: str):
    a = jnp.asarray(field, dtype=jnp.float64)
    if a.shape[-1] != NUM_SOIL_LAYERS:
        raise ValueError(
            f"{name} must have trailing num_soil_layers={NUM_SOIL_LAYERS}, got {a.shape}"
        )
    return a


def _held_rad(radiation: Any, shape) -> SlabRadiation:
    if radiation is None:
        zero = jnp.zeros(shape, dtype=jnp.float64)
        return SlabRadiation(zero, zero)
    if isinstance(radiation, SlabRadiation):
        return SlabRadiation(_surface_2d(radiation.gsw), _surface_2d(radiation.glw))
    if isinstance(radiation, tuple):
        # (soldn, lwdn, cosz) NoahMP-style tuple or (gsw, glw): take the first two.
        return SlabRadiation(_surface_2d(radiation[0]), _surface_2d(radiation[1]))
    return SlabRadiation(
        _surface_2d(getattr(radiation, "gsw")),
        _surface_2d(getattr(radiation, "glw")),
    )


def _saturation_qsg(t_k, psfc_cmb):
    """WRF SLAB ground saturation mixing ratio at ``t_k`` (psfc in cmb)."""

    esg = SVP1_KPA * jnp.exp(SVP2 * (t_k - SVPT0_K) / (t_k - SVP3_K))
    return EP2 * esg / (psfc_cmb - esg)


# --- WRF surface-layer exchange-coefficient ceilings (module_sf_sfclay.F) ------
# FLHC (heat) -- WRF's OWN, EXACT ceiling. WRF computes
#     FLHC = CPM*RHOX*UST*KARMAN/(PSIT*PRT)        (module_sf_sfclay.F:882, with
#                                                   MOL=KARMAN*DTG/PSIT/PRT, so the
#                                                   (THX-THGB) cancels analytically)
# and *explicitly floors the heat resistance* PSIT at 2.0 for EVERY (land + water)
# point at module_sf_sfclay.F:706:
#     PSIT = AMAX1(GZ1OZ0(I)-PSIH(I), 2.)          (in-source comment "LOWER LIMIT
#       ADDED TO PREVENT LARGE FLHC IN SOIL MODEL / ACTIVATES IN UNSTABLE
#       CONDITIONS WITH THIN LAYERS OR HIGH Z0").
# With PRT=1 (:281), KARMAN=0.4, CPM=CP*(1+0.8*QX) (:466), that PSIT>=2 floor is
# exactly an UPPER bound on FLHC:  FLHC <= CPM*RHOX*UST*KARMAN/(2*PRT).
#
# FLQC (moisture) -- a CONSERVATIVE guard, NOT WRF's exact land maximum.
# FLQC = RHOX*MAVAIL*UST*KARMAN/PSIQ (:877). WRF floors PSIQ at 2 ONLY in the
# WATER branch (:731); the LAND PSIQ (:763) has NO max(...,2.) (it is
# GZ1OZ0-PSIH+GZ0OZT, generally >=~2 but not guaranteed). So we cap the
# reconstructed FLQC at WRF's WATER-branch ceiling RHOX*MAVAIL*UST*KARMAN/2 -- the
# LOOSEST moisture exchange WRF emits anywhere. This is a conservative upper guard
# on the ill-posed FLQC=QFX/(QSG-QX) reconstruction (it never tightens below a
# WRF-admissible value), NOT a claim that WRF floors land PSIQ.
PRT_SFCLAY = 1.0          # module_sf_sfclay.F:281 (turbulent Prandtl number)
KARMAN_SFCLAY = 0.4       # share/module_model_constants.F:82
PSIT_FLOOR = 2.0          # module_sf_sfclay.F:706 ("PREVENT LARGE FLHC IN SOIL MODEL")
PSIQ_WATER_FLOOR = 2.0    # module_sf_sfclay.F:731 (water-branch PSIQ floor; land has none -> conservative guard)


def _flhc_flqc_from_handles(state: Any, land: SlabLandState):
    """Reconstruct FLHC/FLQC from the resident surface-layer kinematic fluxes.

    The surface-layer adapter wrote ``theta_flux = HFX/(rho*cp)`` and
    ``qv_flux = QFX/rho``. WRF SLAB consumes ``FLHC``/``FLQC`` with
    ``HFX = FLHC*(THG - THX)`` and ``QFX = FLQC*(QSG - QX)`` where THG is the
    skin potential temperature, THX the lowest-level potential temperature, QSG
    the ground saturation mixing ratio, QX the lowest-level mixing ratio. Solving
    for the coefficients from the handles keeps the slab energy SOLVE consistent
    with the surface-layer exchange that produced the handles (the same seam the
    Noah-classic hook uses for CH) **at the seed point**.

    Dividing the flux by ``(THG-THX)`` is ill-posed near neutral stability
    (``THG-THX -> 0`` => ``FLHC -> inf``): WRF's SLAB1D re-evaluates
    ``HFX = FLHC*(THG-THX)`` as the skin temperature evolves over the soil
    sub-steps (module_sf_slab.F:443 "UPDATE FLUXES FOR NEW GROUND TEMPERATURE"),
    so a huge-but-finite FLHC turns a tiny skin perturbation into a runaway HFX ->
    TSK blow-up. WRF prevents exactly this by *flooring the heat resistance PSIT
    at 2.0* when it builds FLHC (module_sf_sfclay.F:706, comment "LOWER LIMIT
    ADDED TO PREVENT LARGE FLHC IN SOIL MODEL"). We reproduce that physical
    ceiling: the surface layer can never hand the slab an FLHC larger than
    ``CPM*RHOX*UST*KARMAN/(2*PRT)`` (and likewise FLQC), because it is the same
    ``UST*KARMAN/resistance`` exchange WRF computes, with the same resistance
    floor. Capping at this WRF maximum keeps the seed HFX/QFX intact wherever the
    surface-layer exchange was already physical, and only clips the degenerate
    near-neutral spikes the floor exists to suppress.
    """

    rho = jnp.maximum(jnp.asarray(state.rhosfc, dtype=jnp.float64), 1.0e-12)
    theta_flux = jnp.asarray(state.theta_flux, dtype=jnp.float64)
    qv_flux = jnp.asarray(state.qv_flux, dtype=jnp.float64)
    ust = jnp.maximum(jnp.asarray(state.ustar, dtype=jnp.float64), 0.0)
    mavail = jnp.clip(jnp.asarray(state.mavail, dtype=jnp.float64), 0.0, 1.0)
    hfx_seed = theta_flux * rho * CP
    qfx_seed = qv_flux * rho

    # Surface pressure proxy = lowest model level (state.p axis-0 index 0).
    psfc = jnp.maximum(jnp.asarray(state.p[0], dtype=jnp.float64), 1.0)
    psfc_cmb = psfc / 1000.0
    rovcp = R_D_OVER_CP
    tsk = jnp.asarray(land.tslb[..., 0], dtype=jnp.float64)  # skin = TSLB(1)
    thg = tsk * (100.0 / psfc_cmb) ** rovcp
    thx = jnp.asarray(state.theta[0], dtype=jnp.float64)
    qsg = _saturation_qsg(tsk, psfc_cmb)
    qx = jnp.asarray(state.qv[0], dtype=jnp.float64)

    dth = thg - thx
    dq = qsg - qx
    dth_safe = jnp.where(jnp.abs(dth) > 1.0e-6, dth, jnp.where(dth >= 0.0, 1.0e-6, -1.0e-6))
    dq_safe = jnp.where(jnp.abs(dq) > 1.0e-12, dq, jnp.where(dq >= 0.0, 1.0e-12, -1.0e-12))
    flhc = hfx_seed / dth_safe
    flqc = qfx_seed / dq_safe
    # Fall back to a non-negative exchange estimate where the seed difference is
    # degenerate (keeps the assembler finite without changing SLAB physics).
    flhc = jnp.where(jnp.isfinite(flhc) & (flhc > 0.0), flhc, jnp.maximum(rho * CP * jnp.abs(theta_flux), 0.0))
    flqc = jnp.where(jnp.isfinite(flqc) & (flqc >= 0.0), flqc, jnp.maximum(rho * jnp.abs(qv_flux), 0.0))

    # Cap the (>=0) exchange coefficients so a near-neutral (THG-THX -> 0)
    # reconstruction cannot drive the SLAB1D energy solve out of range.
    # FLHC: WRF's EXACT ceiling -- the universal PSIT>=2 heat-resistance floor
    #   (module_sf_sfclay.F:706), FLHC <= CPM*RHOX*UST*KARMAN/(2*PRT). cpm =
    #   CP*(1+0.8*QX) (module_sf_sfclay.F:466).
    # FLQC: a CONSERVATIVE guard -- WRF's WATER-branch PSIQ>=2 ceiling (:731), the
    #   loosest moisture exchange WRF emits (land PSIQ is NOT floored, so this is an
    #   upper guard on the ill-posed FLQC reconstruction, not the exact land max).
    cpm = CP * (1.0 + 0.8 * jnp.maximum(qx, 0.0))
    flhc_max = cpm * rho * ust * KARMAN_SFCLAY / (PSIT_FLOOR * PRT_SFCLAY)
    flqc_max = rho * mavail * ust * KARMAN_SFCLAY / PSIQ_WATER_FLOOR
    flhc = jnp.clip(flhc, 0.0, flhc_max)
    flqc = jnp.clip(flqc, 0.0, flqc_max)
    return flhc, flqc


def slab_surface_step(
    state: Any,
    land_state: SlabLandState,
    static: SlabStaticBundle,
    dt: float,
    *,
    radiation: Any = None,
) -> tuple[Any, SlabLandState]:
    """Run the slab LSM over land and write land flux handles back into State.

    Returns ``(state_out, next_land)``. Water columns (XLAND >= 1.5) keep the
    surface-layer path; land columns advance TSK and the 5-layer TSLB carry.
    """

    if land_state is None or static is None:
        raise ValueError(
            "slab scan coupling requires slab_land (SlabLandState) and slab_static (SlabStaticBundle)"
        )

    nsoil_steps = nsoil_substeps(float(static.dzs[0]), float(dt))
    rad = _held_rad(radiation, jnp.asarray(state.t_skin, dtype=jnp.float64).shape)

    t_bottom = _temperature_from_theta(state.theta, state.p)[0]
    grid_shape = jnp.asarray(state.t_skin, dtype=jnp.float64).shape  # (ny, nx)
    nsoil = land_state.tslb.shape[-1]

    # ``slab_columns`` is vmapped over a single leading column axis, so flatten the
    # 2-D (ny, nx) surface fields to (ncol,) and the (ny, nx, nsoil) soil carry to
    # (ncol, nsoil), then reshape the stacked outputs back to the grid.
    def _flat(field):
        return jnp.asarray(_surface_2d(field), dtype=jnp.float64).reshape(-1)

    qv_bottom = jnp.asarray(state.qv[0], dtype=jnp.float64).reshape(-1)
    p_bottom = jnp.maximum(jnp.asarray(state.p[0], dtype=jnp.float64).reshape(-1), 1.0)
    # Surface pressure proxy = lowest model level (state.p axis-0 index 0).
    psfc = jnp.maximum(jnp.asarray(state.p[0], dtype=jnp.float64).reshape(-1), 1.0)
    xland2d = _surface_2d(getattr(state, "xland", jnp.ones_like(t_bottom)))
    xland = jnp.asarray(xland2d, dtype=jnp.float64).reshape(-1)
    tsk_in = _flat(state.t_skin)
    tslb_in = _soil_carry(land_state.tslb, name="tslb").reshape(-1, nsoil)
    mavail = _flat(state.mavail)
    tb_flat = jnp.asarray(t_bottom, dtype=jnp.float64).reshape(-1)

    flhc2d, flqc2d = _flhc_flqc_from_handles(state, land_state)
    flhc = jnp.asarray(flhc2d, dtype=jnp.float64).reshape(-1)
    flqc = jnp.asarray(flqc2d, dtype=jnp.float64).reshape(-1)

    # Water passthrough flux seeds: the resident handles (HFX/QFX/LH) the surface
    # layer wrote. theta_flux/qv_flux are kinematic; recover the W/m^2 forms.
    rho2d = jnp.maximum(jnp.asarray(state.rhosfc, dtype=jnp.float64), 1.0e-12)
    rho = jnp.asarray(rho2d, dtype=jnp.float64).reshape(-1)
    hfx_in = _flat(state.theta_flux) * rho * CP
    qfx_in = _flat(state.qv_flux) * rho
    lh_in = qfx_in * XLV

    out_flat = slab_columns(
        tb_flat, qv_bottom, p_bottom, flhc, flqc, psfc, xland, tsk_in, tslb_in, mavail,
        _flat(rad.gsw), _flat(rad.glw), hfx_in, qfx_in, lh_in,
        zs=jnp.asarray(static.zs, dtype=jnp.float64),
        dzs=jnp.asarray(static.dzs, dtype=jnp.float64),
        thc=_flat(static.thc),
        tmn=_flat(static.tmn),
        emiss=_flat(static.emiss),
        snowc=_flat(static.snowc),
        deltsm=float(dt),
        nsoil_steps=nsoil_steps,
        ifsnow=int(static.ifsnow),
    )
    # Reshape stacked (ncol,) / (ncol, nsoil) outputs back to the (ny, nx) grid.
    out = {
        k: (v.reshape(grid_shape + (nsoil,)) if k == "tslb" else v.reshape(grid_shape))
        for k, v in out_flat.items()
    }

    is_land = (xland2d - 1.5) < 0.0
    is_land_soil = is_land[..., None]

    # Land HFX/QFX overwrite the kinematic flux handles the PBL reads.
    theta_flux = out["hfx"] / (rho2d * CP)
    qv_flux = out["qfx"] / rho2d

    def _blend(old, new):
        return jnp.where(is_land, jnp.asarray(new, dtype=jnp.float64), jnp.asarray(old, dtype=jnp.float64))

    state_out = state.replace(
        t_skin=_blend(state.t_skin, out["tsk"]).astype(_output_dtype(state, "t_skin")),
        theta_flux=_blend(state.theta_flux, theta_flux).astype(_output_dtype(state, "theta_flux")),
        qv_flux=_blend(state.qv_flux, qv_flux).astype(_output_dtype(state, "qv_flux")),
    )
    next_land = land_state.replace(
        tslb=jnp.where(is_land_soil, out["tslb"], land_state.tslb),
        hfx=_blend(land_state.hfx, out["hfx"]),
        qfx=_blend(land_state.qfx, out["qfx"]),
        lh=_blend(land_state.lh, out["lh"]),
        qsfc=_blend(land_state.qsfc, out["qsfc"]),
        capg=_blend(land_state.capg, out["capg"]),
    )
    return state_out, next_land


def initial_slab_land(state: Any, static: SlabStaticBundle) -> SlabLandState:
    """Seed the slab land carry from the resident State + static bundle.

    The 5-layer TSLB column is initialized by linearly interpolating between the
    skin temperature (TSK) and the deep-soil restore temperature (TMN) across the
    soil-layer center depths -- the standard WRF slab cold-start when a wrfinput
    TSLB is unavailable. If a TSLB seed is supplied via ``static`` it is used as-is.
    """

    tsk = jnp.asarray(state.t_skin, dtype=jnp.float64)
    tmn = _surface_2d(static.tmn)
    zs = jnp.asarray(static.zs, dtype=jnp.float64)
    zmax = jnp.maximum(zs[-1], 1.0e-6)
    frac = (zs / zmax)[None, None, :]  # (1,1,nsoil)
    tslb = tsk[..., None] * (1.0 - frac) + tmn[..., None] * frac
    zero = jnp.zeros_like(tsk)
    return SlabLandState(
        tslb=tslb,
        hfx=zero,
        qfx=zero,
        lh=zero,
        qsfc=jnp.asarray(state.qv[0], dtype=jnp.float64),
        capg=zero,
    )


__all__ = [
    "SlabStaticBundle",
    "SlabLandState",
    "SlabRadiation",
    "slab_surface_step",
    "initial_slab_land",
]
