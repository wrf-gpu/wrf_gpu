"""WRF 5-layer thermal-diffusion slab land-surface model (``sf_surface_physics=1``).

This is the v0.13 Tier-3 per-scheme lane port of WRF's
``phys/module_sf_slab.F`` (subroutine ``SLAB1D``). The slab LSM solves the
ground-temperature tendency from the residual of the surface energy budget
(Blackadar 1978b): a 5-layer soil thermal-diffusion column with an adaptive
soil sub-timestep, driven by the surface-layer exchange coefficients
(``flhc``/``flqc``), the net radiation (``gsw`` + ``emiss*(glw - upflux)``),
and the deep-soil restoring temperature (``tmn``).

The column kernel is a pure ``jnp`` function, ``jax.jit``/``jax.vmap``-traceable
(no Python branching on data, no host allocations), batched over the grid by the
operational LSM hook. It is faithful to ``SLAB1D`` for the operational WRF
configuration the generated pristine-WRF oracle exercises:

* ``num_soil_layers = 5`` (the only WRF slab configuration; ``CAPG`` uses the
  multi-layer ``5.9114E7*THC`` heat-capacity form),
* ``radiation = .TRUE.`` (longwave upflux is included: ``RADSWTCH = 1``),
* land columns advance ``TSK``/``TSLB``; ocean/lake columns (``XLAND >= 1.5``)
  keep their skin temperature unchanged, exactly as ``SLAB1D`` skips them.

The adaptive soil sub-step count ``NSOIL`` depends only on ``DZS(1)`` and the
time step (static at trace time), so it is a Python-level static argument, not a
data-dependent branch -- the kernel stays fully traceable while reproducing the
WRF substep cadence bit-for-bit.

Cited to ``/home/user/src/wrf_pristine/WRF/phys/module_sf_slab.F`` (SLAB1D,
lines 194-515) and the WRF call site
``phys/module_surface_driver.F:2659`` (dtbl, rcp, dtmin, ifsnow, radiation).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsStepResult, PhysicsTendency


config.update("jax_enable_x64", True)


# --- module_sf_slab.F parameters (lines 8-12) ---
DIFSL = 5.0e-7       # soil diffusion constant (m^2/s)
SOILFAC = 1.25       # factor to make the soil sub-step more conservative

# --- thermodynamic / saturation constants WRF passes into SLAB ---
# (module_surface_driver.F:2659-2665 -> svp1,svp2,svp3,svpt0,ep_2,xlv,p1000mb;
#  share/module_model_constants.F: EOMEG, STBOLT.)
SVP1_KPA = 0.6112
SVP2 = 17.67
SVP3_K = 29.65
SVPT0_K = 273.15
EP2 = 0.622
XLV = 2.5e6
P0_PA = 100000.0      # p1000mb
EOMEG = 7.2921e-5     # angular velocity of earth rotation (rad/s)
STBOLT = 5.67051e-8   # Stefan-Boltzmann constant (W/m^2/K^4)

NUM_SOIL_LAYERS = 5   # WRF slab is a fixed 5-layer model


def nsoil_substeps(dzs1: float, deltsm: float) -> int:
    """WRF SLAB1D soil sub-step count ``1 + IFIX(SOILFAC*4*DIFSL/DZS1 * DT/DZS1)``.

    Depends only on the (static) first soil-layer thickness and the surface time
    step, so it is resolved at trace time as a Python int -- the kernel loops it
    with ``lax.fori_loop`` without any data-dependent control flow.
    (module_sf_slab.F:423; ``IFIX`` truncates toward zero.)
    """

    return 1 + int(SOILFAC * 4.0 * DIFSL / dzs1 * deltsm / dzs1)


class SlabStatic(NamedTuple):
    """Read-only slab static inputs (per column), constant over the run."""

    zs: jax.Array          # (nsoil,) soil-layer center depths (m)
    dzs: jax.Array         # (nsoil,) soil-layer thicknesses (m)
    thc: jax.Array         # thermal inertia (Cal/cm/K/s^0.5)
    tmn: jax.Array         # deep-soil (lower boundary) temperature (K)
    emiss: jax.Array       # surface emissivity
    snowc: jax.Array       # snow-cover flag (1 = snow)
    rovcp: float = 287.0 / 1004.0


def slab_column(
    t_bottom: jax.Array,     # T at lowest model level (K)
    qv_bottom: jax.Array,    # qv at lowest model level (kg/kg)
    p_bottom: jax.Array,     # pressure at lowest model level (Pa)
    flhc: jax.Array,         # heat exchange coefficient (from surface layer)
    flqc: jax.Array,         # moisture exchange coefficient
    psfc: jax.Array,         # surface pressure (Pa)
    xland: jax.Array,        # land mask (1 land, 2 water)
    tsk: jax.Array,          # skin temperature (K) -- INOUT
    tslb: jax.Array,         # (nsoil,) soil-layer temperatures (K) -- INOUT
    mavail: jax.Array,       # surface moisture availability (0..1)
    gsw: jax.Array,          # downward shortwave at ground (W/m^2)
    glw: jax.Array,          # downward longwave at ground (W/m^2)
    hfx_in: jax.Array,       # surface-layer HFX (W/m^2) -- water passthrough
    qfx_in: jax.Array,       # surface-layer QFX (kg/m^2/s) -- water passthrough
    lh_in: jax.Array,        # surface-layer LH (W/m^2) -- water passthrough
    *,
    static: SlabStatic,
    deltsm: float,           # surface time step (s) -- DELTSM
    nsoil_steps: int,        # soil sub-step count (static; = nsoil_substeps(...))
    ifsnow: int = 0,         # ifsnow=1 enables snow-cover skin-T limiting
):
    """Faithful per-column port of WRF ``SLAB1D`` (num_soil_layers=5, radiation=T).

    Returns ``(PhysicsStepResult, outputs)`` where ``outputs`` is a dict of the
    updated land-carry fields (tslb, tsk, hfx, qfx, lh, qsfc, chklowq, capg).
    All math is in fp64 when the inputs are fp64; nothing allocates outside the
    trace.
    """

    rovcp = static.rovcp
    zs = static.zs
    dzs = static.dzs
    thc = static.thc
    tmn = static.tmn
    emiss = static.emiss
    snowc = static.snowc
    nsoil = zs.shape[-1]

    psfc_cmb = psfc / 1000.0           # PSFC(I) in cmb (line 316)
    pl_cmb = p_bottom / 1000.0         # PL cmb (line 322)
    thcon = (P0_PA * 0.001 / pl_cmb) ** rovcp
    thx = t_bottom * thcon             # THX (line 326)
    qx = qv_bottom                     # QX (line 332)

    # CAPG multi-layer heat-capacity form (line 347): num_soil_layers > 1.
    capg = 5.9114e7 * thc

    # Land/ocean switch: XLD1 = 1 for land (XLAND-1.5 < 0). (lines 363-368)
    is_land = xland < 1.5

    tg0 = tsk                          # TG0 = TSK (line 383)

    # --- initial surface energy budget (lines 401-418) ---
    upflux = STBOLT * tg0 ** 4         # RADSWTCH=1 (radiation=.TRUE.)
    xinet = emiss * (glw - upflux)
    rnet = gsw + xinet
    esg = SVP1_KPA * jnp.exp(SVP2 * (tg0 - SVPT0_K) / (tg0 - SVP3_K))
    qsg = EP2 * esg / (psfc_cmb - esg)
    thg = tsk * (100.0 / psfc_cmb) ** rovcp
    hfx = flhc * (thg - thx)
    qfx = flqc * (qsg - qx)
    qs = hfx + qfx * XLV
    # num_soil_layers > 1 -> DTHGDT initialised to 0 (line 418).
    dthgdt0 = jnp.zeros_like(tg0)

    # --- soil sub-timestep loop (lines 422-471) ---
    ps1 = (psfc / P0_PA) ** rovcp        # PS1 (line 434)
    rnsoil = 1.0 / float(nsoil_steps)

    def soil_substep(itsoil, carry):
        tslb_c, hfx_c, qfx_c, dthgdt_c, qs_c = carry

        def layer_body(L, inner):
            # L is 0-based for Fortran layers 1..nsoil-1.
            tslb_i, hfx_i, qfx_i, dthgdt_i, qs_i, flux_lo = inner
            # On L==0 & itsoil>0: recompute fluxes from the updated skin layer
            # (Fortran L.EQ.1.AND.ITSOIL.GT.1; lines 432-449).
            recompute = jnp.logical_and(L == 0, itsoil > 0)
            thg_r = tslb_i[0] / ps1
            esg_r = SVP1_KPA * jnp.exp(
                SVP2 * (tslb_i[0] - SVPT0_K) / (tslb_i[0] - SVP3_K)
            )
            qsg_r = EP2 * esg_r / (psfc_cmb - esg_r)
            hfxt = flhc * (thg_r - thx)
            qfxt = flqc * (qsg_r - qx)
            qs_new = hfxt + qfxt * XLV
            qs_i = jnp.where(recompute, qs_new, qs_i)
            hfx_i = jnp.where(recompute, hfx_i + hfxt, hfx_i)
            qfx_i = jnp.where(recompute, qfx_i + qfxt, qfx_i)

            # FLUX(L) lower-face: on L==0 it is RNET-QS, else carried. (line 450)
            flux_l = jnp.where(L == 0, rnet - qs_i, flux_lo)
            t_lp1 = tslb_i[L + 1]
            t_l = tslb_i[L]
            flux_up = -DIFSL * capg * (t_lp1 - t_l) / (zs[L + 1] - zs[L])
            dtsdt = -(flux_up - flux_l) / (dzs[L] * capg)
            tslb_new_L = t_l + dtsdt * deltsm * rnsoil
            tslb_i = tslb_i.at[L].set(tslb_new_L)

            # Snow-cover limit on layer 1 (lines 455-459).
            if ifsnow == 1:
                clip = jnp.logical_and(
                    L == 0, jnp.logical_and(snowc > 0.0, tslb_i[0] > 273.16)
                )
                tslb_i = tslb_i.at[0].set(jnp.where(clip, 273.16, tslb_i[0]))

            # DTHGDT accumulation on L==0 (line 460).
            dthgdt_i = jnp.where(L == 0, dthgdt_i + rnsoil * dtsdt, dthgdt_i)

            # Last substep & L==0: average HFX/QFX over substeps (lines 461-465).
            last_avg = jnp.logical_and(itsoil == (nsoil_steps - 1), L == 0)
            hfx_i = jnp.where(last_avg, hfx_i * rnsoil, hfx_i)
            qfx_i = jnp.where(last_avg, qfx_i * rnsoil, qfx_i)

            return (tslb_i, hfx_i, qfx_i, dthgdt_i, qs_i, flux_up)

        inner0 = (tslb_c, hfx_c, qfx_c, dthgdt_c, qs_c, jnp.zeros_like(tg0))
        tslb_c, hfx_c, qfx_c, dthgdt_c, qs_c, _ = jax.lax.fori_loop(
            0, nsoil - 1, layer_body, inner0
        )
        return (tslb_c, hfx_c, qfx_c, dthgdt_c, qs_c)

    tslb_out, hfx_out, qfx_out, dthgdt_out, _ = jax.lax.fori_loop(
        0, nsoil_steps, soil_substep, (tslb, hfx, qfx, dthgdt0, qs)
    )
    lh_out = qfx_out * XLV

    # --- skin temperature update (lines 473-480) ---
    tsk_out = tg0 + deltsm * dthgdt_out

    # --- snow-cover skin-T limiting (lines 486-499) ---
    if ifsnow == 1:
        clip_skin = jnp.logical_and(snowc > 0.0, tsk_out > 273.16)
        tsk_out = jnp.where(clip_skin, 273.16, tsk_out)

    # Ocean/lake columns (XLD1<0.5) are SKIPPED in WRF (every land-guarded DO has
    # ``IF(XLD1(I).LT.0.5)GOTO`` / ``IF(XLDCOL.GT.1.5)GOTO 90``): TSK, TSLB, and
    # the surface fluxes keep their INPUT (surface-layer) values for water.
    tsk_final = jnp.where(is_land, tsk_out, tsk)
    tslb_final = jnp.where(is_land[..., None], tslb_out, tslb)
    hfx_final = jnp.where(is_land, hfx_out, hfx_in)
    qfx_final = jnp.where(is_land, qfx_out, qfx_in)
    lh_final = jnp.where(is_land, lh_out, lh_in)

    # --- QSFC and CHKLOWQ (lines 502-511) ---
    # The final DO loop runs for ALL columns (no XLD1 guard), using the
    # column's final QFX (land=updated, water=input passthrough).
    qfx_for_qsfc = qfx_final
    flqc_safe = jnp.where(flqc != 0.0, flqc, 1.0)
    qsfc_final = jnp.where(flqc != 0.0, qx + qfx_for_qsfc / flqc_safe, qx)
    chklowq_final = mavail

    tendency = PhysicsTendency(
        state_replacements={
            "t_skin": tsk_final,
            "hfx": hfx_final,
            "qfx": qfx_final,
        }
    )
    result = PhysicsStepResult(tendency=tendency)
    return result, {
        "tslb": tslb_final,
        "tsk": tsk_final,
        "hfx": hfx_final,
        "qfx": qfx_final,
        "lh": lh_final,
        "qsfc": qsfc_final,
        "chklowq": chklowq_final,
        "capg": capg,
    }


def slab_columns(
    t_bottom,
    qv_bottom,
    p_bottom,
    flhc,
    flqc,
    psfc,
    xland,
    tsk,
    tslb,
    mavail,
    gsw,
    glw,
    hfx_in,
    qfx_in,
    lh_in,
    *,
    zs,
    dzs,
    thc,
    tmn,
    emiss,
    snowc,
    rovcp: float = 287.0 / 1004.0,
    deltsm: float,
    nsoil_steps: int,
    ifsnow: int = 0,
):
    """Batched (vmap) entry: 2-D fields are ``(ncol,)``; tslb is ``(ncol, nsoil)``.

    ``zs``/``dzs`` are shared ``(nsoil,)`` vectors. Returns a dict of stacked
    outputs (tslb, tsk, hfx, qfx, lh, qsfc, chklowq, capg).
    """

    def one(tb, qb, pb, fhc, fqc, ps, xl, tk, ts, mv, g_sw, g_lw, h_in, q_in, l_in, th_c, tm, em, sn):
        st = SlabStatic(zs=zs, dzs=dzs, thc=th_c, tmn=tm, emiss=em, snowc=sn, rovcp=rovcp)
        _, out = slab_column(
            tb, qb, pb, fhc, fqc, ps, xl, tk, ts, mv, g_sw, g_lw, h_in, q_in, l_in,
            static=st, deltsm=deltsm, nsoil_steps=nsoil_steps, ifsnow=ifsnow,
        )
        return out

    return jax.vmap(one, in_axes=(0,) * 19)(
        t_bottom, qv_bottom, p_bottom, flhc, flqc, psfc, xland, tsk, tslb,
        mavail, gsw, glw, hfx_in, qfx_in, lh_in, thc, tmn, emiss, snowc,
    )


__all__ = [
    "DIFSL",
    "SOILFAC",
    "NUM_SOIL_LAYERS",
    "SlabStatic",
    "nsoil_substeps",
    "slab_column",
    "slab_columns",
]
