"""JAX shortwave RRTMG-style radiation column kernel for M5-S3."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.physics.rrtmg_constants import (
    AVOGADRO,
    CH4_VMR,
    CO2_VMR,
    CP_AIR,
    DRY_AIR_MOLECULAR_WEIGHT,
    GRAVITY,
    MAX_OPTICAL_DEPTH,
    MIN_COSZEN,
    MIN_LAYER_MASS,
    MIN_OPTICAL_DEPTH,
    N2O_VMR,
    O2_VMR,
    O3_BACKGROUND_VMR,
    SOLAR_CONSTANT,
    WATER_VAPOR_MOLECULAR_WEIGHT_RATIO,
)
from gpuwrf.physics.rrtmg_tables import RRTMGTableBundle, RRTMG_TABLES


config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class RRTMGSWColumnState:
    """Pytree for independent shortwave radiation columns on mass levels."""

    __slots__ = ("T", "p", "qv", "qc", "qi", "qs", "qg", "cloud_fraction", "surface_albedo", "coszen", "dz", "rho")

    def __init__(self, T, p, qv, qc, qi, qs, qg, cloud_fraction, surface_albedo, coszen, dz, rho) -> None:
        self.T = T
        self.p = p
        self.qv = qv
        self.qc = qc
        self.qi = qi
        self.qs = qs
        self.qg = qg
        self.cloud_fraction = cloud_fraction
        self.surface_albedo = surface_albedo
        self.coszen = coszen
        self.dz = dz
        self.rho = rho

    def replace(self, **updates) -> "RRTMGSWColumnState":
        """Returns a same-layout state with named fields replaced."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        """Presents all state arrays as JAX leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds the state after JAX transforms."""

        del aux
        return cls(*children)

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality outside JIT for tests."""

        if not isinstance(other, RRTMGSWColumnState):
            return NotImplemented
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(_leaves(self), _leaves(other), strict=True)
        )

    def __hash__(self) -> int:
        """Hashes small column states outside the physics hot path."""

        parts = []
        for leaf in _leaves(self):
            host = np.asarray(leaf)
            parts.append((tuple(host.shape), str(host.dtype), host.tobytes()))
        return hash(tuple(parts))


class RRTMGSWColumnResult(NamedTuple):
    """Shortwave column outputs with bottom-to-top interface fluxes."""

    heating_rate: jnp.ndarray
    flux_down: jnp.ndarray
    flux_up: jnp.ndarray
    toa_down: jnp.ndarray
    toa_up: jnp.ndarray
    surface_down: jnp.ndarray
    surface_up: jnp.ndarray
    column_absorbed: jnp.ndarray
    surface_absorbed: jnp.ndarray


def _leaves(state: RRTMGSWColumnState):
    """Centralizes leaf iteration for equality and hashing."""

    return (getattr(state, name) for name in RRTMGSWColumnState.__slots__)


def _clip_state(state: RRTMGSWColumnState) -> RRTMGSWColumnState:
    """Applies radiation-safe physical bounds before optical calculations."""

    return state.replace(
        T=jnp.maximum(state.T, 120.0),
        p=jnp.maximum(state.p, 1.0),
        qv=jnp.maximum(state.qv, 0.0),
        qc=jnp.maximum(state.qc, 0.0),
        qi=jnp.maximum(state.qi, 0.0),
        qs=jnp.maximum(state.qs, 0.0),
        qg=jnp.maximum(state.qg, 0.0),
        cloud_fraction=jnp.clip(state.cloud_fraction, 0.0, 1.0),
        surface_albedo=jnp.clip(state.surface_albedo, 0.0, 1.0),
        coszen=jnp.maximum(state.coszen, MIN_COSZEN),
        dz=jnp.maximum(state.dz, 1.0),
        rho=jnp.maximum(state.rho, MIN_LAYER_MASS),
    )


def _pressure_interfaces(p):
    """Reconstructs WRF harness pressure interfaces from midpoint pressures."""

    nz = p.shape[-1]
    dp_bottom = jnp.maximum(10.0, p[..., 0] - p[..., 1])
    bottom = p[..., :1] + 0.5 * dp_bottom[..., None]
    middle = 0.5 * (p[..., :-1] + p[..., 1:])
    dp_top = jnp.maximum(10.0, p[..., -2] - p[..., -1])
    top = jnp.maximum(400.0, p[..., -1:] - 0.5 * dp_top[..., None])
    return jnp.concatenate((bottom, middle, top), axis=-1)


def _pressure_layer_mass(p):
    """Reconstructs WRF harness layer mass from midpoint pressure interfaces."""

    nz = p.shape[-1]
    interfaces = _pressure_interfaces(p)
    return jnp.maximum((interfaces[..., :nz] - interfaces[..., 1 : nz + 1]) / GRAVITY, MIN_LAYER_MASS)


def _extend_with_wrf_top_layer(values, top_values=None):
    """Adds WRF's isothermal extra top layer used by the RRTMG wrapper."""

    top = values[..., -1:] if top_values is None else top_values
    return jnp.concatenate((values, top), axis=-1)


def _nearest_pressure_coefficients(state_p, tables: RRTMGTableBundle):
    """Selects WRF reference-pressure absorption coefficients per layer."""

    ref_log = jnp.log(tables.sw_reference_pressure_pa)
    layer_log = jnp.log(jnp.maximum(state_p, 1.0))
    idx = jnp.argmin(jnp.abs(layer_log[..., None] - ref_log), axis=-1)
    gathered = jnp.take(tables.sw_absorption_coefficients, idx, axis=1)
    return jnp.moveaxis(gathered, 0, -2)


def _rrtmg_column_amounts(qv, pressure_interfaces):
    """Computes RRTMG-style scaled molecular columns from WRF interface pressure."""

    h2ovmr = qv * WATER_VAPOR_MOLECULAR_WEIGHT_RATIO
    amm = (1.0 - h2ovmr) * DRY_AIR_MOLECULAR_WEIGHT + h2ovmr * 18.0160
    dp_mb = jnp.maximum((pressure_interfaces[..., :-1] - pressure_interfaces[..., 1:]) * 0.01, 1.0e-8)
    coldry = dp_mb * 1.0e3 * AVOGADRO / (1.0e2 * GRAVITY * amm * (1.0 + h2ovmr))
    colh2o = 1.0e-20 * coldry * h2ovmr
    colco2 = 1.0e-20 * coldry * CO2_VMR
    colo3 = 1.0e-20 * coldry * O3_BACKGROUND_VMR
    coln2o = 1.0e-20 * coldry * N2O_VMR
    colch4 = 1.0e-20 * coldry * CH4_VMR
    colo2 = 1.0e-20 * coldry * O2_VMR
    colmol = 1.0e-20 * coldry + colh2o
    absorber = colh2o + 0.02 * colco2 + 0.02 * colch4 + 0.01 * coln2o + 0.0002 * colo2 + 0.5 * colo3
    return absorber, colmol


def _delta_scale(tau, omega, asymmetry):
    """Applies Joseph-Wiscombe-Weinman delta scaling to optical properties."""

    f = asymmetry * asymmetry
    denom_tau = jnp.maximum(1.0 - f * omega, 1.0e-12)
    denom_g = jnp.maximum(1.0 - f, 1.0e-12)
    tau_scaled = denom_tau * tau
    omega_scaled = (1.0 - f) * omega / denom_tau
    asym_scaled = (asymmetry - f) / denom_g
    return tau_scaled, jnp.clip(omega_scaled, 0.0, 0.999999), jnp.clip(asym_scaled, -0.999999, 0.999999)


def _reftra_eddington(tau, omega, asymmetry, mu0, active):
    """Computes Eddington direct/diffuse layer reflectance and transmittance."""

    tau = jnp.clip(tau, MIN_OPTICAL_DEPTH, MAX_OPTICAL_DEPTH)
    omega = jnp.clip(omega, 0.0, 0.999999)
    asymmetry = jnp.clip(asymmetry, -0.999999, 0.999999)
    mu0 = jnp.maximum(mu0[..., None, None, None], 1.0e-6)
    g3 = 3.0 * asymmetry
    gamma1 = (7.0 - omega * (4.0 + g3)) * 0.25
    gamma2 = -(1.0 - omega * (4.0 - g3)) * 0.25
    gamma3 = (2.0 - g3 * mu0) * 0.25
    gamma4 = 1.0 - gamma3

    zwo_denom = 1.0 - (1.0 - omega) * (asymmetry / jnp.maximum(1.0 - asymmetry, 1.0e-12)) ** 2
    zwo = jnp.where((omega > 0.0) & (jnp.abs(zwo_denom) > 1.0e-12), omega / zwo_denom, 0.0)
    conservative = zwo >= 0.9999995

    za = gamma1 * mu0
    za_cons = za - gamma3
    zgt = gamma1 * tau
    exp_mu = jnp.exp(-jnp.minimum(tau / mu0, 500.0))
    pref_cons = (zgt - za_cons * (1.0 - exp_mu)) / (1.0 + zgt)
    ptra_cons = 1.0 - pref_cons
    prefd_cons = zgt / (1.0 + zgt)
    ptrad_cons = 1.0 - prefd_cons

    za1 = gamma1 * gamma4 + gamma2 * gamma3
    za2 = gamma1 * gamma3 + gamma2 * gamma4
    zrk = jnp.sqrt(jnp.maximum(gamma1 * gamma1 - gamma2 * gamma2, 1.0e-14))
    zrp = zrk * mu0
    zrp1 = 1.0 + zrp
    zrm1 = 1.0 - zrp
    zrk2 = 2.0 * zrk
    zrpp = 1.0 - zrp * zrp
    zrkg = zrk + gamma1
    zr1 = zrm1 * (za2 + zrk * gamma3)
    zr2 = zrp1 * (za2 - zrk * gamma3)
    zr3 = zrk2 * (gamma3 - za2 * mu0)
    zr4 = zrpp * zrkg
    zr5 = zrpp * (zrk - gamma1)
    zt1 = zrp1 * (za1 + zrk * gamma4)
    zt2 = zrm1 * (za1 - zrk * gamma4)
    zt3 = zrk2 * (gamma4 + za1 * mu0)
    zt4 = zr4
    zt5 = zr5
    zbeta = (gamma1 - zrk) / jnp.maximum(zrkg, 1.0e-12)

    exp_h = jnp.exp(-jnp.minimum(zrk * tau, 500.0))
    inv_exp_h = 1.0 / jnp.maximum(exp_h, 1.0e-300)
    inv_exp_mu = 1.0 / jnp.maximum(exp_mu, 1.0e-300)
    zdenr = zr4 * inv_exp_h + zr5 * exp_h
    zdent = zt4 * inv_exp_h + zt5 * exp_h
    pref_non = omega * (zr1 * inv_exp_h - zr2 * exp_h - zr3 * exp_mu) / jnp.where(jnp.abs(zdenr) > 1.0e-8, zdenr, 1.0)
    ptra_non = exp_mu - exp_mu * omega * (zt1 * inv_exp_h - zt2 * exp_h - zt3 * inv_exp_mu) / jnp.where(
        jnp.abs(zdent) > 1.0e-8, zdent, 1.0
    )
    exp_2 = exp_h * exp_h
    zdend = 1.0 / jnp.maximum((1.0 - zbeta * exp_2) * zrkg, 1.0e-12)
    prefd_non = gamma2 * (1.0 - exp_2) * zdend
    ptrad_non = zrk2 * exp_h * zdend

    pref = jnp.where(conservative, pref_cons, pref_non)
    ptra = jnp.where(conservative, ptra_cons, ptra_non)
    prefd = jnp.where(conservative, prefd_cons, prefd_non)
    ptrad = jnp.where(conservative, ptrad_cons, ptrad_non)
    identity = active <= 0.0
    pref = jnp.where(identity, 0.0, pref)
    ptra = jnp.where(identity, 1.0, ptra)
    prefd = jnp.where(identity, 0.0, prefd)
    ptrad = jnp.where(identity, 1.0, ptrad)
    return (jnp.clip(pref, 0.0, 1.0), jnp.clip(prefd, 0.0, 1.0), jnp.clip(ptra, 0.0, 1.0), jnp.clip(ptrad, 0.0, 1.0))


def _vertical_quadrature(pref, prefd, ptra, ptrad, direct_trans):
    """Applies the WRF `vrtqdr_sw` adding-method vertical quadrature."""

    surface_shape = pref.shape[:-3] + (1, pref.shape[-2], pref.shape[-1])
    ptdbt = jnp.concatenate((jnp.ones(surface_shape, dtype=pref.dtype), jnp.cumprod(direct_trans, axis=-3)), axis=-3)
    pdbt = jnp.concatenate((direct_trans, jnp.zeros(surface_shape, dtype=pref.dtype)), axis=-3)
    nlev = pref.shape[-3] - 1
    prup = jnp.zeros_like(pref)
    prupd = jnp.zeros_like(pref)
    prdnd = jnp.zeros_like(pref)
    ztdn = jnp.zeros_like(pref)
    prup = prup.at[..., nlev : nlev + 1, :, :].set(pref[..., nlev : nlev + 1, :, :])
    prupd = prupd.at[..., nlev : nlev + 1, :, :].set(prefd[..., nlev : nlev + 1, :, :])

    zreflect = 1.0 / jnp.maximum(1.0 - prefd[..., nlev : nlev + 1, :, :] * prefd[..., nlev - 1 : nlev, :, :], 1.0e-12)
    bottom_prup = pref[..., nlev - 1 : nlev, :, :] + (
        ptrad[..., nlev - 1 : nlev, :, :]
        * (
            (ptra[..., nlev - 1 : nlev, :, :] - pdbt[..., nlev - 1 : nlev, :, :]) * prefd[..., nlev : nlev + 1, :, :]
            + pdbt[..., nlev - 1 : nlev, :, :] * pref[..., nlev : nlev + 1, :, :]
        )
        * zreflect
    )
    bottom_prupd = prefd[..., nlev - 1 : nlev, :, :] + (
        ptrad[..., nlev - 1 : nlev, :, :] * ptrad[..., nlev - 1 : nlev, :, :] * prefd[..., nlev : nlev + 1, :, :] * zreflect
    )
    prup = prup.at[..., nlev - 1 : nlev, :, :].set(bottom_prup)
    prupd = prupd.at[..., nlev - 1 : nlev, :, :].set(bottom_prupd)

    for idx in range(nlev - 2, -1, -1):
        zreflect = 1.0 / jnp.maximum(1.0 - prupd[..., idx + 1 : idx + 2, :, :] * prefd[..., idx : idx + 1, :, :], 1.0e-12)
        value_up = pref[..., idx : idx + 1, :, :] + (
            ptrad[..., idx : idx + 1, :, :]
            * (
                (ptra[..., idx : idx + 1, :, :] - pdbt[..., idx : idx + 1, :, :]) * prupd[..., idx + 1 : idx + 2, :, :]
                + pdbt[..., idx : idx + 1, :, :] * prup[..., idx + 1 : idx + 2, :, :]
            )
            * zreflect
        )
        value_upd = prefd[..., idx : idx + 1, :, :] + (
            ptrad[..., idx : idx + 1, :, :] * ptrad[..., idx : idx + 1, :, :] * prupd[..., idx + 1 : idx + 2, :, :] * zreflect
        )
        prup = prup.at[..., idx : idx + 1, :, :].set(value_up)
        prupd = prupd.at[..., idx : idx + 1, :, :].set(value_upd)

    ztdn = ztdn.at[..., :1, :, :].set(1.0)
    prdnd = prdnd.at[..., :1, :, :].set(0.0)
    ztdn = ztdn.at[..., 1:2, :, :].set(ptra[..., :1, :, :])
    prdnd = prdnd.at[..., 1:2, :, :].set(prefd[..., :1, :, :])

    for idx in range(1, nlev):
        zreflect = 1.0 / jnp.maximum(1.0 - prefd[..., idx : idx + 1, :, :] * prdnd[..., idx : idx + 1, :, :], 1.0e-12)
        value_tdn = ptdbt[..., idx : idx + 1, :, :] * ptra[..., idx : idx + 1, :, :] + (
            ptrad[..., idx : idx + 1, :, :]
            * (
                (ztdn[..., idx : idx + 1, :, :] - ptdbt[..., idx : idx + 1, :, :])
                + ptdbt[..., idx : idx + 1, :, :] * pref[..., idx : idx + 1, :, :] * prdnd[..., idx : idx + 1, :, :]
            )
            * zreflect
        )
        value_rdnd = prefd[..., idx : idx + 1, :, :] + (
            ptrad[..., idx : idx + 1, :, :] * ptrad[..., idx : idx + 1, :, :] * prdnd[..., idx : idx + 1, :, :] * zreflect
        )
        ztdn = ztdn.at[..., idx + 1 : idx + 2, :, :].set(value_tdn)
        prdnd = prdnd.at[..., idx + 1 : idx + 2, :, :].set(value_rdnd)

    zreflect = 1.0 / jnp.maximum(1.0 - prdnd * prupd, 1.0e-12)
    flux_up = (ptdbt * prup + (ztdn - ptdbt) * prupd) * zreflect
    flux_down = (ptdbt + (ztdn - ptdbt + ptdbt * prup * prdnd) * zreflect)
    return flux_down, flux_up


def _shortwave_impl(state: RRTMGSWColumnState, tables: RRTMGTableBundle, debug: bool) -> RRTMGSWColumnResult:
    """Unjitted SW implementation shared by production and stripped paths."""

    state = _clip_state(state)
    original_layers = state.p.shape[-1]
    original_interfaces = _pressure_interfaces(state.p)
    top_pressure = 0.5 * original_interfaces[..., -1:]
    pressure_interfaces = jnp.concatenate((original_interfaces, jnp.full_like(top_pressure, 1.0e-3)), axis=-1)
    p_ext = jnp.concatenate((state.p, top_pressure), axis=-1)
    qv_ext = _extend_with_wrf_top_layer(state.qv)
    qc_ext = jnp.concatenate((state.qc, jnp.zeros_like(state.qc[..., -1:])), axis=-1)
    qi_ext = jnp.concatenate((state.qi, jnp.zeros_like(state.qi[..., -1:])), axis=-1)
    qs_ext = jnp.concatenate((state.qs, jnp.zeros_like(state.qs[..., -1:])), axis=-1)
    qg_ext = jnp.concatenate((state.qg, jnp.zeros_like(state.qg[..., -1:])), axis=-1)
    cloud_ext = jnp.concatenate((state.cloud_fraction, jnp.zeros_like(state.cloud_fraction[..., -1:])), axis=-1)
    layer_mass = _pressure_layer_mass(state.p)
    layer_mass_ext = jnp.maximum((pressure_interfaces[..., :-1] - pressure_interfaces[..., 1:]) / GRAVITY, MIN_LAYER_MASS)
    gas_column, dry_column = _rrtmg_column_amounts(qv_ext, pressure_interfaces)
    liquid_path_g = (qc_ext + 0.25 * qg_ext) * layer_mass_ext * 1000.0
    ice_path_g = (qi_ext + qs_ext + 0.75 * qg_ext) * layer_mass_ext * 1000.0

    weights = tables.sw_gpoint_weights
    gas_coeff = _nearest_pressure_coefficients(p_ext, tables)
    rayleigh_coeff = tables.sw_rayleigh_coefficients
    liquid_coeff = tables.sw_cloud_liquid_extinction[:, None]
    ice_coeff = tables.sw_cloud_ice_extinction[:, None]
    liquid_ssa = tables.sw_cloud_liquid_ssa[:, None]
    ice_ssa = tables.sw_cloud_ice_ssa[:, None]
    liquid_asy = tables.sw_cloud_liquid_asymmetry[:, None]
    ice_asy = tables.sw_cloud_ice_asymmetry[:, None]
    mask = tables.sw_gpoint_mask

    tau_gas = gas_column[..., None, None] * jnp.maximum(gas_coeff, 0.0)
    tau_rayleigh = dry_column[..., None, None] * rayleigh_coeff
    tau_liquid = cloud_ext[..., None, None] * liquid_path_g[..., None, None] * liquid_coeff
    tau_ice = cloud_ext[..., None, None] * ice_path_g[..., None, None] * ice_coeff
    tau_total = jnp.clip(tau_gas + tau_rayleigh + tau_liquid + tau_ice, MIN_OPTICAL_DEPTH, MAX_OPTICAL_DEPTH)
    scattering = tau_rayleigh + tau_liquid * liquid_ssa + tau_ice * ice_ssa
    asymmetry_num = tau_liquid * liquid_ssa * liquid_asy + tau_ice * ice_ssa * ice_asy
    omega = jnp.clip(scattering / jnp.maximum(tau_total, MIN_OPTICAL_DEPTH), 0.0, 0.999999)
    asymmetry = jnp.where(scattering > MIN_OPTICAL_DEPTH, asymmetry_num / jnp.maximum(scattering, MIN_OPTICAL_DEPTH), 0.0)
    tau_scaled, omega_scaled, asymmetry_scaled = _delta_scale(tau_total, omega, asymmetry)
    tau_scaled = tau_scaled * mask
    omega_scaled = omega_scaled * mask
    asymmetry_scaled = asymmetry_scaled * mask

    tau_top_down = jnp.flip(tau_scaled, axis=-3)
    omega_top_down = jnp.flip(omega_scaled, axis=-3)
    asymmetry_top_down = jnp.flip(asymmetry_scaled, axis=-3)
    active_top_down = jnp.flip(mask + jnp.zeros_like(tau_scaled), axis=-3)
    pref_lay, prefd_lay, ptra_lay, ptrad_lay = _reftra_eddington(
        tau_top_down, omega_top_down, asymmetry_top_down, state.coszen, active_top_down
    )
    surface = jnp.broadcast_to(
        state.surface_albedo[..., None, None, None],
        pref_lay.shape[:-3] + (1, pref_lay.shape[-2], pref_lay.shape[-1]),
    )
    zero_surface = jnp.zeros_like(surface)
    pref = jnp.concatenate((pref_lay, surface), axis=-3)
    prefd = jnp.concatenate((prefd_lay, surface), axis=-3)
    ptra = jnp.concatenate((ptra_lay, zero_surface), axis=-3)
    ptrad = jnp.concatenate((ptrad_lay, zero_surface), axis=-3)
    direct_trans = jnp.exp(-jnp.minimum(tau_top_down / jnp.maximum(state.coszen[..., None, None, None], 1.0e-6), 500.0)) * active_top_down
    direct_trans = jnp.where(active_top_down > 0.0, direct_trans, 1.0)
    down_top_down, up_top_down = _vertical_quadrature(pref, prefd, ptra, ptrad, direct_trans)
    top_flux_band = SOLAR_CONSTANT * state.coszen[..., None, None] * weights
    down_band = jnp.flip(down_top_down * top_flux_band[..., None, :, :], axis=-3)
    up_band = jnp.flip(up_top_down * top_flux_band[..., None, :, :], axis=-3)

    flux_down_model = jnp.sum(down_band, axis=(-1, -2))
    flux_up_model = jnp.sum(up_band, axis=(-1, -2))
    net_down = flux_down_model - flux_up_model
    column_absorbed_layers = net_down[..., 1 : original_layers + 1] - net_down[..., :original_layers]
    heating_rate = column_absorbed_layers / (layer_mass * CP_AIR)
    surface_absorbed = flux_down_model[..., 0] - flux_up_model[..., 0]
    flux_down = flux_down_model
    flux_up = flux_up_model

    heating_rate = assert_finite(heating_rate, "rrtmg_sw.heating_rate", enabled=debug)
    flux_down = assert_physical_bounds(flux_down, 0.0, 2000.0, "rrtmg_sw.flux_down", enabled=debug)
    flux_up = assert_physical_bounds(flux_up, 0.0, 2000.0, "rrtmg_sw.flux_up", enabled=debug)
    return RRTMGSWColumnResult(
        heating_rate=heating_rate,
        flux_down=flux_down,
        flux_up=flux_up,
        toa_down=flux_down[..., -1],
        toa_up=flux_up[..., -1],
        surface_down=flux_down[..., 0],
        surface_up=flux_up[..., 0],
        column_absorbed=jnp.sum(column_absorbed_layers, axis=-1),
        surface_absorbed=surface_absorbed,
    )


@partial(jax.jit, static_argnames=("debug",))
def solve_rrtmg_sw_column(
    state: RRTMGSWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
    *,
    debug: bool = False,
) -> RRTMGSWColumnResult:
    """Computes one fused shortwave column radiation call."""

    return _shortwave_impl(state, tables, debug)


@jax.jit
def solve_rrtmg_sw_column_debug_stripped(
    state: RRTMGSWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
) -> RRTMGSWColumnResult:
    """Hand-stripped sibling used for the HLO debug identity proof."""

    return _shortwave_impl(state, tables, False)
