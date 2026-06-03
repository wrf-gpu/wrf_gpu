"""Operational surface hook for Noah-classic (sf_surface_physics=2).

This is the State<->land-carry adapter for the JAX Noah-classic SFLX kernel.
It mirrors the existing Noah-MP operational pattern:

1. the selected surface-layer path has already populated the resident surface
   exchange handles (ustar, rhosfc, theta_flux, qv_flux, tau_u, tau_v);
2. Noah-classic advances the land tile from explicit REDPRM/static inputs and
   the 4-layer land carry;
3. land HFX/QFX/TSK/Z0 overwrite the same State handles that MYNN reads; water
   columns keep the surface-layer path via one land/water where switch.

The REDPRM block is deliberately an explicit static input. The parity-passed
Noah-classic kernel does not derive REDPRM from land-use categories, so the
operational scan must be configured with a WRF-derived NoahClassicStatic bundle.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import (
    GRAVITY_M_S2,
    P0_PA,
    R_D_OVER_CP,
    _output_dtype,
    _rho_from_state,
    _temperature_from_theta,
    _u_mass,
    _v_mass,
)
from gpuwrf.physics.lsm_noah_classic import (
    CP,
    RD,
    NSOIL,
    NoahClassicForcing,
    NoahClassicParams,
    NoahClassicState,
    sflx_step,
)
from gpuwrf.physics.surface_constants import EP1, EP2, SVP1_KPA, SVP2, SVP3_K, SVPT0_K


class NoahClassicStatic(NamedTuple):
    """Read-only Noah-classic static/REDPRM inputs for one operational tile."""

    params: NoahClassicParams
    zsoil: jax.Array
    sldpth: jax.Array
    snoalb: jax.Array
    tbot: jax.Array
    solnet_albedo: jax.Array | None = None
    lwdn_emissivity: jax.Array | None = None


class NoahClassicLandState(NamedTuple):
    """4-layer Noah-classic land carry plus last land flux diagnostics."""

    t1: jax.Array
    stc: jax.Array
    smc: jax.Array
    sh2o: jax.Array
    cmc: jax.Array
    sneqv: jax.Array
    snowh: jax.Array
    sncovr: jax.Array
    snotime1: jax.Array
    ribb: jax.Array
    flx4: jax.Array
    fvb: jax.Array
    fbur: jax.Array
    fgsn: jax.Array
    smcrel: jax.Array
    xlaidyn: jax.Array
    hfx: jax.Array
    qfx: jax.Array
    lh: jax.Array
    grdflx: jax.Array

    def replace(self, **updates) -> "NoahClassicLandState":
        return self._replace(**updates)


class NoahClassicRadiation(NamedTuple):
    """Held surface radiation tuple used by the Noah-classic forcing assembler."""

    soldn: jax.Array
    lwdn: jax.Array
    cosz: jax.Array


def _surface_2d(field):
    a = jnp.asarray(field, dtype=jnp.float64)
    return a[..., 0] if a.ndim >= 3 else a


def _soil4(field, *, name: str):
    a = jnp.asarray(field, dtype=jnp.float64)
    if a.shape[-1] != NSOIL:
        raise ValueError(f"{name} must have trailing num_soil_layers={NSOIL}, got {a.shape}")
    return a


def _as_sflx_state(land: NoahClassicLandState) -> NoahClassicState:
    return NoahClassicState(
        t1=jnp.asarray(land.t1, dtype=jnp.float64),
        stc=_soil4(land.stc, name="stc"),
        smc=_soil4(land.smc, name="smc"),
        sh2o=_soil4(land.sh2o, name="sh2o"),
        cmc=jnp.asarray(land.cmc, dtype=jnp.float64),
        sneqv=jnp.asarray(land.sneqv, dtype=jnp.float64),
        snowh=jnp.asarray(land.snowh, dtype=jnp.float64),
        sncovr=jnp.asarray(land.sncovr, dtype=jnp.float64),
        snotime1=jnp.asarray(land.snotime1, dtype=jnp.float64),
        ribb=jnp.asarray(land.ribb, dtype=jnp.float64),
    )


def _saturation_qsat_dqdt(temp_k, pressure_pa):
    """WRF SVP saturation mixing ratio and d(qsat)/dT at ``temp_k``."""

    t = jnp.asarray(temp_k, dtype=jnp.float64)
    p_kpa = jnp.maximum(jnp.asarray(pressure_pa, dtype=jnp.float64), 1.0) * 0.001
    es = SVP1_KPA * jnp.exp(SVP2 * (t - SVPT0_K) / (t - SVP3_K))
    desdt = es * SVP2 * (SVPT0_K - SVP3_K) / ((t - SVP3_K) * (t - SVP3_K))
    denom = p_kpa - es
    qsat = EP2 * es / denom
    dqsdt = EP2 * p_kpa * desdt / (denom * denom)
    return qsat, dqsdt


def _wind_speed(state) -> jax.Array:
    u0 = _u_mass(state)[0]
    v0 = _v_mass(state)[0]
    return jnp.sqrt(u0 * u0 + v0 * v0)


def _reference_height(state) -> jax.Array:
    interface_z = jnp.asarray(state.ph, dtype=jnp.float64) / GRAVITY_M_S2
    return 0.5 * jnp.maximum(interface_z[1] - interface_z[0], 1.0)


def _held_rad(radiation: Any, shape):
    if radiation is None:
        zero = jnp.zeros(shape, dtype=jnp.float64)
        return NoahClassicRadiation(zero, zero, zero)
    if isinstance(radiation, NoahClassicRadiation):
        return radiation
    if isinstance(radiation, tuple):
        return NoahClassicRadiation(*radiation)
    return NoahClassicRadiation(
        _surface_2d(getattr(radiation, "soldn")),
        _surface_2d(getattr(radiation, "lwdn")),
        _surface_2d(getattr(radiation, "cosz")),
    )


def _surface_layer_ch_from_handles(state, land: NoahClassicLandState) -> jax.Array:
    """Infer CH from the surface-layer heat flux handle entering Noah-classic."""

    t_air = _temperature_from_theta(state.theta, state.p)[0]
    sfcprs = jnp.maximum(jnp.asarray(state.p[0], dtype=jnp.float64), 1.0)
    th2 = jnp.asarray(state.theta[0], dtype=jnp.float64)
    t1 = jnp.asarray(land.t1, dtype=jnp.float64)
    rho = jnp.maximum(_rho_from_state(state)[0], 1.0e-12)
    hfx_seed = jnp.asarray(state.theta_flux, dtype=jnp.float64) * rho * CP
    t2v = t_air * (1.0 + 0.61 * jnp.asarray(state.qv[0], dtype=jnp.float64))
    denom = CP * sfcprs * (th2 - t1)
    denom_safe = jnp.where(jnp.abs(denom) > 1.0e-12, denom, jnp.where(denom >= 0.0, 1.0e-12, -1.0e-12))
    ch = -hfx_seed * RD * t2v / denom_safe
    # If the prior heat-flux difference is degenerate, fall back to a momentum
    # exchange estimate from UST and wind speed. This keeps the assembler finite
    # without altering the SFLX physics solve.
    cm_like = (jnp.asarray(state.ustar, dtype=jnp.float64) ** 2) / jnp.maximum(_wind_speed(state), 1.0e-6)
    return jnp.where(jnp.isfinite(ch) & (ch > 0.0), ch, cm_like)


def assemble_noahclassic_forcing(
    state: Any,
    land: NoahClassicLandState,
    static: NoahClassicStatic,
    radiation: Any = None,
) -> NoahClassicForcing:
    """Build Noah-classic SFLX forcing from the operational State and held radiation."""

    t_air = _temperature_from_theta(state.theta, state.p)[0]
    sfcprs = jnp.maximum(jnp.asarray(state.p[0], dtype=jnp.float64), 1.0)
    q2 = jnp.maximum(jnp.asarray(state.qv[0], dtype=jnp.float64), 0.0)
    q2sat, dqsdt2 = _saturation_qsat_dqdt(land.t1, sfcprs)
    shape = t_air.shape
    rad = _held_rad(radiation, shape)
    params = static.params
    solnet_albedo = params.alb if static.solnet_albedo is None else _surface_2d(static.solnet_albedo)
    lwdn_emiss = params.embrd if static.lwdn_emissivity is None else _surface_2d(static.lwdn_emissivity)
    soldn = jnp.maximum(_surface_2d(rad.soldn), 0.0)
    lwdn = _surface_2d(rad.lwdn) * lwdn_emiss
    ch = _surface_layer_ch_from_handles(state, land)
    cm = (jnp.asarray(state.ustar, dtype=jnp.float64) ** 2) / jnp.maximum(_wind_speed(state), 1.0e-6)
    return NoahClassicForcing(
        sfctmp=t_air,
        sfcprs=sfcprs,
        th2=jnp.asarray(state.theta[0], dtype=jnp.float64),
        q2=q2,
        q2sat=q2sat,
        dqsdt2=dqsdt2,
        soldn=soldn,
        solnet=soldn * (1.0 - solnet_albedo),
        lwdn=lwdn,
        prcp=jnp.zeros(shape, dtype=jnp.float64),
        ffrozp=jnp.zeros(shape, dtype=jnp.float64),
        sfcspd=_wind_speed(state),
        zlvl=_reference_height(state),
        snoalb=_surface_2d(static.snoalb),
        tbot=_surface_2d(static.tbot),
        ch=ch,
        cm=cm,
    )


def _blend_surface(old, new, is_land):
    return jnp.where(is_land, jnp.asarray(new, dtype=jnp.float64), jnp.asarray(old, dtype=jnp.float64))


def noahclassic_surface_step(
    state: Any,
    land_state: NoahClassicLandState,
    static: NoahClassicStatic,
    dt: float,
    *,
    radiation: Any = None,
) -> tuple[Any, NoahClassicLandState]:
    """Run Noah-classic over land and write land flux handles back into State."""

    if land_state is None or static is None:
        raise ValueError("Noah-classic scan coupling requires noahclassic_land and noahclassic_static")
    forcing = assemble_noahclassic_forcing(state, land_state, static, radiation)
    out = sflx_step(
        forcing,
        static.params,
        _as_sflx_state(land_state),
        float(dt),
        _soil4(static.zsoil, name="zsoil"),
        _soil4(static.sldpth, name="sldpth"),
    )
    xland = _surface_2d(getattr(state, "xland", jnp.ones_like(out.hfx)))
    is_land = (xland - 1.5) < 0.0
    is_land4 = is_land[..., None]
    rho = jnp.maximum(jnp.asarray(state.rhosfc, dtype=jnp.float64), 1.0e-12)
    theta_flux = out.hfx / (rho * CP)
    qv_flux = out.qfx / rho
    thx = jnp.asarray(state.theta[0], dtype=jnp.float64)
    qx = jnp.asarray(state.qv[0], dtype=jnp.float64)
    fltv = (1.0 + EP1 * qx) * theta_flux + EP1 * thx * qv_flux

    state_out = state.replace(
        t_skin=_blend_surface(state.t_skin, out.state.t1, is_land).astype(_output_dtype(state, "t_skin")),
        soil_moisture=_blend_surface(state.soil_moisture, out.state.smc[..., 0], is_land).astype(
            _output_dtype(state, "soil_moisture")
        ),
        mavail=_blend_surface(state.mavail, out.smav[..., 0], is_land).astype(_output_dtype(state, "mavail")),
        roughness_m=_blend_surface(state.roughness_m, out.z0, is_land).astype(_output_dtype(state, "roughness_m")),
        theta_flux=_blend_surface(state.theta_flux, theta_flux, is_land).astype(_output_dtype(state, "theta_flux")),
        qv_flux=_blend_surface(state.qv_flux, qv_flux, is_land).astype(_output_dtype(state, "qv_flux")),
        fltv=_blend_surface(state.fltv, fltv, is_land).astype(_output_dtype(state, "fltv")),
    )
    next_land = land_state.replace(
        t1=_blend_surface(land_state.t1, out.state.t1, is_land),
        stc=jnp.where(is_land4, out.state.stc, land_state.stc),
        smc=jnp.where(is_land4, out.state.smc, land_state.smc),
        sh2o=jnp.where(is_land4, out.state.sh2o, land_state.sh2o),
        cmc=_blend_surface(land_state.cmc, out.state.cmc, is_land),
        sneqv=_blend_surface(land_state.sneqv, out.state.sneqv, is_land),
        snowh=_blend_surface(land_state.snowh, out.state.snowh, is_land),
        sncovr=_blend_surface(land_state.sncovr, out.state.sncovr, is_land),
        snotime1=_blend_surface(land_state.snotime1, out.state.snotime1, is_land),
        ribb=_blend_surface(land_state.ribb, out.state.ribb, is_land),
        smcrel=jnp.where(is_land4, out.smav, land_state.smcrel),
        xlaidyn=_blend_surface(land_state.xlaidyn, static.params.xlai, is_land),
        hfx=_blend_surface(land_state.hfx, out.hfx, is_land),
        qfx=_blend_surface(land_state.qfx, out.qfx, is_land),
        lh=_blend_surface(land_state.lh, out.lh, is_land),
        grdflx=_blend_surface(land_state.grdflx, out.grdflx, is_land),
    )
    return state_out, next_land


def overlay_noahclassic_land_diagnostics(state: Any, land_state: NoahClassicLandState, bulk_hfx, bulk_lh, bulk_tsk):
    """Overlay last Noah-classic land HFX/LH/TSK onto bulk diagnostics."""

    xland = _surface_2d(getattr(state, "xland", jnp.ones_like(jnp.asarray(bulk_hfx, dtype=jnp.float64))))
    is_land = (xland - 1.5) < 0.0
    hfx = jnp.where(is_land, jnp.asarray(land_state.hfx, dtype=jnp.float64), jnp.asarray(bulk_hfx, dtype=jnp.float64))
    lh = jnp.where(is_land, jnp.asarray(land_state.lh, dtype=jnp.float64), jnp.asarray(bulk_lh, dtype=jnp.float64))
    tsk = jnp.where(is_land, jnp.asarray(land_state.t1, dtype=jnp.float64), jnp.asarray(bulk_tsk, dtype=jnp.float64))
    return hfx, lh, tsk


__all__ = [
    "NoahClassicLandState",
    "NoahClassicRadiation",
    "NoahClassicStatic",
    "assemble_noahclassic_forcing",
    "noahclassic_surface_step",
    "overlay_noahclassic_land_diagnostics",
]
