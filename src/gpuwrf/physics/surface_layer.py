"""MM5 sfclay-style Monin-Obukhov surface layer for M6 coupling.

The implementation follows the local WRF `module_sf_sfclay.F` algebra cited in
ADR-012. It is vectorized over horizontal columns and returns the M5-S2.x
`SurfaceFluxes` contract consumed by coupled adapters.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import config
import jax.numpy as jnp

from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.physics.surface_constants import (
    CP_D,
    DEFAULT_DX_M,
    DEFAULT_LAND_ROUGHNESS_M,
    DEFAULT_PBLH_M,
    DEFAULT_WATER_ROUGHNESS_M,
    EP2,
    G,
    KARMAN,
    MAX_ROUGHNESS_M,
    MIN_ROUGHNESS_M,
    MIN_WIND_M_S,
    P0_PA,
    P608,
    R_D,
    R_D_OVER_CP,
    SALINITY_FACTOR,
    SVP1_KPA,
    SVP2,
    SVP3_K,
    SVPT0_K,
    THERMAL_DIFFUSIVITY_M2_S,
    VCONVC,
)


config.update("jax_enable_x64", True)


class SurfaceLayerDiagnostics(NamedTuple):
    """Diagnostic fields produced by the sfclay solve."""

    fluxes: SurfaceFluxes
    u10: object
    v10: object
    th2: object
    t2: object
    q2: object
    bulk_richardson: object
    roughness_m: object
    z_over_l: object
    fm: object
    fh: object


def _field(state, name: str, default):
    return getattr(state, name, default)


def _surface(field):
    """Return the lowest model level from a column-last or 2-D field."""

    if getattr(field, "ndim", 0) >= 3:
        return field[..., 0]
    return field


def _as_surface(value, shape):
    data = jnp.asarray(value, dtype=jnp.float64)
    if data.ndim >= 3:
        data = data[..., 0]
    if data.shape == ():
        return jnp.ones(shape, dtype=jnp.float64) * data
    return data.astype(jnp.float64)


def _potential_to_temperature(theta, pressure_pa):
    exner = (jnp.maximum(pressure_pa, 1.0) / P0_PA) ** R_D_OVER_CP
    return theta * exner


def _saturation_mixing_ratio(t_k, pressure_pa, xland, lakemask):
    """WRF sfclay saturation relation, `module_sf_sfclay.F:455-466`."""

    e_kpa = SVP1_KPA * jnp.exp(SVP2 * (t_k - SVPT0_K) / jnp.maximum(t_k - SVP3_K, 1.0e-6))
    e_kpa = jnp.where((xland > 1.5) & (lakemask == 0.0), e_kpa * SALINITY_FACTOR, e_kpa)
    pressure_kpa = jnp.maximum(pressure_pa * 0.001, e_kpa + 1.0e-6)
    return EP2 * e_kpa / (pressure_kpa - e_kpa)


def _psi_m_unstable(zol):
    """Unstable momentum stability table formula, `module_sf_sfclay.F:960-966`."""

    x = (1.0 - 16.0 * zol) ** 0.25
    return 2.0 * jnp.log(0.5 * (1.0 + x)) + jnp.log(0.5 * (1.0 + x * x)) - 2.0 * jnp.arctan(x) + 2.0 * jnp.arctan(1.0)


def _psi_h_unstable(zol):
    y = (1.0 - 16.0 * zol) ** 0.5
    return 2.0 * jnp.log(0.5 * (1.0 + y))


def _roughness_from_state(state, shape, xland):
    roughness = _field(state, "roughness_m", None)
    if roughness is not None:
        return jnp.clip(_as_surface(roughness, shape), MIN_ROUGHNESS_M, MAX_ROUGHNESS_M)

    cm = _field(state, "cm", None)
    za = _field(state, "measurement_height_m", None)
    if cm is not None and za is not None:
        cm_s = jnp.maximum(_as_surface(cm, shape), 1.0e-6)
        za_s = jnp.maximum(_as_surface(za, shape), 2.0)
        z0 = za_s * jnp.exp(-KARMAN / jnp.sqrt(cm_s))
        return jnp.clip(z0, MIN_ROUGHNESS_M, MAX_ROUGHNESS_M)

    soil = _field(state, "soil_moisture", None)
    if soil is not None:
        soil_s = jnp.clip(_as_surface(soil, shape), 0.0, 1.0)
        land_z0 = DEFAULT_LAND_ROUGHNESS_M * (0.5 + soil_s)
    else:
        land_z0 = jnp.ones(shape, dtype=jnp.float64) * DEFAULT_LAND_ROUGHNESS_M
    water_z0 = jnp.ones(shape, dtype=jnp.float64) * DEFAULT_WATER_ROUGHNESS_M
    return jnp.where(xland > 1.5, water_z0, jnp.clip(land_z0, MIN_ROUGHNESS_M, MAX_ROUGHNESS_M))


def surface_layer(state) -> SurfaceFluxes:
    """Return the M5-S2.x surface-flux contract."""

    return surface_layer_with_diagnostics(state).fluxes


def surface_layer_with_diagnostics(state) -> SurfaceLayerDiagnostics:
    """Run one vectorized MM5 sfclay surface-layer solve.

    State fields are column-last for 3-D arrays. Optional fields accepted by this
    function are `t_skin`, `roughness_m`, `xland`, `lakemask`, `mavail`,
    `ustar`, `mol`, `pblh`, `dx_m`, and `dz`.
    """

    u0 = _surface(jnp.asarray(state.u, dtype=jnp.float64))
    v0 = _surface(jnp.asarray(state.v, dtype=jnp.float64))
    theta0 = _surface(jnp.asarray(state.theta, dtype=jnp.float64))
    qv0 = jnp.maximum(_surface(jnp.asarray(state.qv, dtype=jnp.float64)), 0.0)
    p0 = jnp.maximum(_surface(jnp.asarray(state.p, dtype=jnp.float64)), 1.0)
    shape = u0.shape

    t_air = _potential_to_temperature(theta0, p0)
    t_skin = _as_surface(_field(state, "t_skin", t_air), shape)
    dz = _as_surface(_field(state, "dz", 100.0), shape)
    za = jnp.maximum(0.5 * dz, 2.1)
    xland = _as_surface(_field(state, "xland", 1.0), shape)
    lakemask = _as_surface(_field(state, "lakemask", 0.0), shape)
    mavail = jnp.clip(_as_surface(_field(state, "mavail", _field(state, "soil_moisture", 1.0)), shape), 0.0, 1.0)
    old_ust = jnp.maximum(_as_surface(_field(state, "ustar", 0.0), shape), 0.0)
    old_mol = _as_surface(_field(state, "mol", 0.0), shape)
    pblh = jnp.maximum(_as_surface(_field(state, "pblh", DEFAULT_PBLH_M), shape), 1.0)
    dx_m = jnp.maximum(_as_surface(_field(state, "dx_m", DEFAULT_DX_M), shape), 1.0)
    znt = _roughness_from_state(state, shape, xland)

    theta_ground = t_skin * (P0_PA / p0) ** R_D_OVER_CP
    theta_v_air = theta0 * (1.0 + P608 * qv0)
    q_sfc = _saturation_mixing_ratio(t_skin, p0, xland, lakemask)
    theta_v_ground = theta_ground * (1.0 + P608 * q_sfc)
    rho = jnp.maximum(p0 / (R_D * t_air * (1.0 + P608 * qv0)), 1.0e-4)
    cpm = CP_D * (1.0 + 0.8 * qv0)

    gz1oz0 = jnp.log(jnp.maximum(za / znt, 1.000001))
    gz2oz0 = jnp.log(jnp.maximum(2.0 / znt, 1.000001))
    gz10oz0 = jnp.log(jnp.maximum(10.0 / znt, 1.000001))
    wind_raw = jnp.sqrt(u0 * u0 + v0 * v0)
    dthvdz = theta_v_air - theta_v_ground
    fluxc = jnp.zeros(shape, dtype=jnp.float64)
    vconv_land = VCONVC * jnp.maximum(G / jnp.maximum(t_skin, 1.0) * pblh * fluxc, 0.0) ** (1.0 / 3.0)
    vconv_water = jnp.sqrt(jnp.maximum(-dthvdz, 0.0))
    vconv = jnp.where(xland < 1.5, vconv_land, vconv_water)
    vsgd = 0.32 * jnp.maximum(dx_m / 5000.0 - 1.0, 0.0) ** (1.0 / 3.0)
    wspd = jnp.maximum(jnp.sqrt(wind_raw * wind_raw + vconv * vconv + vsgd * vsgd), MIN_WIND_M_S)
    br = G / jnp.maximum(theta0, 1.0) * za * dthvdz / jnp.maximum(wspd * wspd, 1.0e-12)
    br = jnp.where(old_mol < 0.0, jnp.minimum(br, 0.0), br)

    stable = br >= 0.2
    damped = (br > 0.0) & (br < 0.2)
    unstable = br < 0.0

    psim_stable = jnp.maximum(-10.0 * gz1oz0, -10.0)
    psim10_stable = jnp.maximum(10.0 / za * psim_stable, -10.0)
    psim2_stable = jnp.maximum(2.0 / za * psim_stable, -10.0)

    psim_damped = jnp.maximum(-5.0 * br * gz1oz0 / jnp.maximum(1.1 - 5.0 * br, 1.0e-6), -10.0)
    psim10_damped = jnp.maximum(10.0 / za * psim_damped, -10.0)
    psim2_damped = jnp.maximum(2.0 / za * psim_damped, -10.0)

    zol_old = jnp.where(
        old_ust < 0.01,
        br * gz1oz0,
        KARMAN * G / jnp.maximum(theta0, 1.0) * za * old_mol / jnp.maximum(old_ust * old_ust, 1.0e-12),
    )
    zol = jnp.clip(jnp.where(unstable, zol_old, br * gz1oz0 / jnp.maximum(1.00001 - 5.0 * br, 1.0e-6)), -9.9999, 9.999)
    zol10 = jnp.clip(10.0 / za * zol, -9.9999, 0.0)
    zol2 = jnp.clip(2.0 / za * zol, -9.9999, 0.0)
    zol_unstable = jnp.clip(zol, -9.9999, 0.0)
    psim_unstable = _psi_m_unstable(zol_unstable)
    psih_unstable = _psi_h_unstable(zol_unstable)
    psim10_unstable = _psi_m_unstable(zol10)
    psih10_unstable = _psi_h_unstable(zol10)
    psim2_unstable = _psi_m_unstable(zol2)
    psih2_unstable = _psi_h_unstable(zol2)
    psim_unstable = jnp.minimum(psim_unstable, 0.9 * gz1oz0)
    psih_unstable = jnp.minimum(psih_unstable, 0.9 * gz1oz0)
    psim10_unstable = jnp.minimum(psim10_unstable, 0.9 * gz10oz0)
    psih10_unstable = jnp.minimum(psih10_unstable, 0.9 * gz10oz0)
    psih2_unstable = jnp.minimum(psih2_unstable, 0.9 * gz2oz0)

    zeros = jnp.zeros(shape, dtype=jnp.float64)
    psim = jnp.where(stable, psim_stable, jnp.where(damped, psim_damped, jnp.where(unstable, psim_unstable, zeros)))
    psih = jnp.where(stable, psim_stable, jnp.where(damped, psim_damped, jnp.where(unstable, psih_unstable, zeros)))
    psim10 = jnp.where(
        stable, psim10_stable, jnp.where(damped, psim10_damped, jnp.where(unstable, psim10_unstable, zeros))
    )
    psih10 = jnp.where(
        stable, psim10_stable, jnp.where(damped, psim10_damped, jnp.where(unstable, psih10_unstable, zeros))
    )
    psih2 = jnp.where(
        stable, psim2_stable, jnp.where(damped, psim2_damped, jnp.where(unstable, psih2_unstable, zeros))
    )

    psix = jnp.maximum(gz1oz0 - psim, 1.0e-6)
    psix10 = gz10oz0 - psim10
    psit = jnp.maximum(gz1oz0 - psih, 2.0)
    zl = jnp.where(xland > 1.5, znt, 0.01)
    psit2 = gz2oz0 - psih2
    psim2 = jnp.where(stable, psim2_stable, jnp.where(damped, psim2_damped, jnp.where(unstable, psim2_unstable, zeros)))
    del psim2
    psim10_safe = psih10
    psiq = jnp.log(KARMAN * old_ust * za / THERMAL_DIFFUSIVITY_M2_S + za / zl) - psih
    psiq2 = jnp.log(KARMAN * old_ust * 2.0 / THERMAL_DIFFUSIVITY_M2_S + 2.0 / zl) - psih2
    psiq10 = jnp.log(KARMAN * old_ust * 10.0 / THERMAL_DIFFUSIVITY_M2_S + 10.0 / zl) - psim10_safe

    viscosity = (1.32 + 0.009 * (t_air - 273.15)) * 1.0e-5
    restar = jnp.maximum(old_ust * znt / jnp.maximum(viscosity, 1.0e-12), 1.0e-12)
    z0t = jnp.clip(5.5e-5 * restar ** (-0.60), 2.0e-9, 1.0e-4)
    water = xland > 1.5
    psiq = jnp.where(water, jnp.maximum(jnp.log((za + z0t) / z0t) - psih, 2.0), psiq)
    psit = jnp.where(water, jnp.maximum(jnp.log((za + z0t) / z0t) - psih, 2.0), psit)
    psiq2 = jnp.where(water, jnp.maximum(jnp.log((2.0 + z0t) / z0t) - psih2, 2.0), psiq2)
    psit2 = jnp.where(water, jnp.maximum(jnp.log((2.0 + z0t) / z0t) - psih2, 2.0), psit2)
    psiq10 = jnp.where(water, jnp.maximum(jnp.log((10.0 + z0t) / z0t) - psih10, 2.0), psiq10)

    ustar = 0.5 * old_ust + 0.5 * KARMAN * wspd / psix
    ustar = jnp.where(xland < 1.5, jnp.maximum(ustar, 0.1), ustar)
    u10 = u0 * psix10 / psix
    v10 = v0 * psix10 / psix
    dtg = theta0 - theta_ground
    th2 = theta_ground + dtg * psit2 / jnp.maximum(psit, 1.0e-6)
    q2 = q_sfc + (qv0 - q_sfc) * psiq2 / jnp.maximum(psiq, 1.0e-6)
    t2 = th2 * (p0 / P0_PA) ** R_D_OVER_CP
    mol = KARMAN * dtg / jnp.maximum(psit, 1.0e-6)

    flqc = rho * mavail * ustar * KARMAN / jnp.maximum(psiq, 1.0e-6)
    heat_denom = jnp.where(jnp.abs(dtg) > 1.0e-5, theta0 - theta_ground, 1.0)
    flhc = jnp.where(jnp.abs(dtg) > 1.0e-5, cpm * rho * ustar * mol / heat_denom, 0.0)
    qfx = flqc * (q_sfc - qv0)
    hfx = flhc * (theta_ground - theta0)
    theta_flux = hfx / jnp.maximum(rho * cpm, 1.0e-12)
    qv_flux = qfx / jnp.maximum(rho, 1.0e-12)
    wind_for_tau = jnp.maximum(wind_raw, MIN_WIND_M_S)
    tau_u = -(ustar * ustar) * u0 / wind_for_tau
    tau_v = -(ustar * ustar) * v0 / wind_for_tau
    fltv = (1.0 + P608 * qv0) * theta_flux + P608 * theta0 * qv_flux

    fluxes = SurfaceFluxes(
        ustar=ustar.astype(jnp.float64),
        theta_flux=theta_flux.astype(jnp.float64),
        qv_flux=qv_flux.astype(jnp.float64),
        tau_u=tau_u.astype(jnp.float64),
        tau_v=tau_v.astype(jnp.float64),
        rhosfc=rho.astype(jnp.float64),
        fltv=fltv.astype(jnp.float64),
    )
    return SurfaceLayerDiagnostics(
        fluxes=fluxes,
        u10=u10.astype(jnp.float64),
        v10=v10.astype(jnp.float64),
        th2=th2.astype(jnp.float64),
        t2=t2.astype(jnp.float64),
        q2=q2.astype(jnp.float64),
        bulk_richardson=br.astype(jnp.float64),
        roughness_m=znt.astype(jnp.float64),
        z_over_l=zol.astype(jnp.float64),
        fm=psix.astype(jnp.float64),
        fh=psit.astype(jnp.float64),
    )


__all__ = ["SurfaceFluxes", "SurfaceLayerDiagnostics", "surface_layer", "surface_layer_with_diagnostics"]
