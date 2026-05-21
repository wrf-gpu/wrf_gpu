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


class _SWSetCoefState(NamedTuple):
    """WRF `setcoef_sw` interpolation state, kept as JAX arrays."""

    jp: jnp.ndarray
    jt: jnp.ndarray
    jt1: jnp.ndarray
    lower_mask: jnp.ndarray
    colh2o: jnp.ndarray
    colco2: jnp.ndarray
    colo3: jnp.ndarray
    coln2o: jnp.ndarray
    colch4: jnp.ndarray
    colo2: jnp.ndarray
    colmol: jnp.ndarray
    coldry: jnp.ndarray
    fac00: jnp.ndarray
    fac01: jnp.ndarray
    fac10: jnp.ndarray
    fac11: jnp.ndarray
    selffac: jnp.ndarray
    selffrac: jnp.ndarray
    indself: jnp.ndarray
    forfac: jnp.ndarray
    forfrac: jnp.ndarray
    indfor: jnp.ndarray


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


def _trunc_int(value):
    """Fortran-style positive-range integer truncation."""

    return value.astype(jnp.int32)


def _take_rows(table, idx):
    """Gathers flattened WRF coefficient rows for every layer."""

    clipped = jnp.clip(idx, 0, table.shape[0] - 1)
    return jnp.take(table, clipped, axis=0)


def _sw_setcoef(qv, p_pa, t_k, pressure_interfaces_pa, tables: RRTMGTableBundle) -> _SWSetCoefState:
    """Ports WRF `setcoef_sw` pressure/temperature interpolation factors."""

    h2ovmr = qv * WATER_VAPOR_MOLECULAR_WEIGHT_RATIO
    amm = (1.0 - h2ovmr) * DRY_AIR_MOLECULAR_WEIGHT + h2ovmr * 18.0160
    pavel = jnp.maximum(p_pa * 0.01, 1.0e-6)
    pz = jnp.maximum(pressure_interfaces_pa * 0.01, 1.0e-12)
    dp_mb = jnp.maximum(pz[..., :-1] - pz[..., 1:], 1.0e-12)
    coldry = dp_mb * 1.0e3 * AVOGADRO / (1.0e2 * GRAVITY * amm * (1.0 + h2ovmr))

    plog = jnp.log(pavel)
    jp = _trunc_int(36.0 - 5.0 * (plog + 0.04))
    jp = jnp.clip(jp, 1, 58)
    jp1 = jp + 1
    fp = 5.0 * (jnp.take(tables.sw_preflog, jp - 1, axis=0) - plog)
    tref0 = jnp.take(tables.sw_tref, jp - 1, axis=0)
    tref1 = jnp.take(tables.sw_tref, jp1 - 1, axis=0)
    jt = _trunc_int(3.0 + (t_k - tref0) / 15.0)
    jt = jnp.clip(jt, 1, 4)
    ft = ((t_k - tref0) / 15.0) - (jt - 3).astype(jnp.float64)
    jt1 = _trunc_int(3.0 + (t_k - tref1) / 15.0)
    jt1 = jnp.clip(jt1, 1, 4)
    ft1 = ((t_k - tref1) / 15.0) - (jt1 - 3).astype(jnp.float64)

    water = h2ovmr
    scalefac = pavel * (296.0 / 1013.0) / t_k
    lower = plog > 4.56
    forfac = scalefac / (1.0 + water)
    lower_for_factor = (332.0 - t_k) / 36.0
    upper_for_factor = (t_k - 188.0) / 36.0
    indfor_lower = jnp.minimum(2, jnp.maximum(1, _trunc_int(lower_for_factor)))
    indfor = jnp.where(lower, indfor_lower, 3)
    forfrac = jnp.where(lower, lower_for_factor - indfor.astype(jnp.float64), upper_for_factor - 1.0)

    selffac = jnp.where(lower, water * forfac, 0.0)
    self_factor = (t_k - 188.0) / 7.2
    indself = jnp.minimum(9, jnp.maximum(1, _trunc_int(self_factor) - 7))
    selffrac = jnp.where(lower, self_factor - (indself + 7).astype(jnp.float64), 0.0)
    indself = jnp.where(lower, indself, 1)

    compfp = 1.0 - fp
    fac10 = compfp * ft
    fac00 = compfp * (1.0 - ft)
    fac11 = fp * ft1
    fac01 = fp * (1.0 - ft1)

    colh2o = 1.0e-20 * coldry * h2ovmr
    colco2 = 1.0e-20 * coldry * CO2_VMR
    colo3 = 1.0e-20 * coldry * O3_BACKGROUND_VMR
    coln2o = 1.0e-20 * coldry * N2O_VMR
    colch4 = 1.0e-20 * coldry * CH4_VMR
    colo2 = 1.0e-20 * coldry * O2_VMR
    colmol = 1.0e-20 * coldry + colh2o

    return _SWSetCoefState(
        jp=jp,
        jt=jt,
        jt1=jt1,
        lower_mask=lower,
        colh2o=colh2o,
        colco2=colco2,
        colo3=colo3,
        coln2o=coln2o,
        colch4=colch4,
        colo2=colo2,
        colmol=colmol,
        coldry=coldry,
        fac00=fac00,
        fac01=fac01,
        fac10=fac10,
        fac11=fac11,
        selffac=selffac,
        selffrac=selffrac,
        indself=indself,
        forfac=forfac,
        forfrac=forfrac,
        indfor=indfor,
    )


def _interp_four_rows(table, idx0_1b, idx1_1b, stride_1b, coef: _SWSetCoefState):
    """WRF four-corner pressure/temperature interpolation for one species row."""

    idx0 = idx0_1b - 1
    idx1 = idx1_1b - 1
    stride = stride_1b
    return (
        coef.fac00[..., None] * _take_rows(table, idx0)
        + coef.fac10[..., None] * _take_rows(table, idx0 + stride)
        + coef.fac01[..., None] * _take_rows(table, idx1)
        + coef.fac11[..., None] * _take_rows(table, idx1 + stride)
    )


def _interp_binary(table, idx0_1b, idx1_1b, stride_1b, fs, coef: _SWSetCoefState):
    """WRF binary-species interpolation over pressure, temperature, and species ratio."""

    idx0 = idx0_1b - 1
    idx1 = idx1_1b - 1
    stride = stride_1b
    fac000 = (1.0 - fs) * coef.fac00
    fac010 = (1.0 - fs) * coef.fac10
    fac100 = fs * coef.fac00
    fac110 = fs * coef.fac10
    fac001 = (1.0 - fs) * coef.fac01
    fac011 = (1.0 - fs) * coef.fac11
    fac101 = fs * coef.fac01
    fac111 = fs * coef.fac11
    return (
        fac000[..., None] * _take_rows(table, idx0)
        + fac100[..., None] * _take_rows(table, idx0 + 1)
        + fac010[..., None] * _take_rows(table, idx0 + stride)
        + fac110[..., None] * _take_rows(table, idx0 + stride + 1)
        + fac001[..., None] * _take_rows(table, idx1)
        + fac101[..., None] * _take_rows(table, idx1 + 1)
        + fac011[..., None] * _take_rows(table, idx1 + stride)
        + fac111[..., None] * _take_rows(table, idx1 + stride + 1)
    )


def _continuum_terms(band, coef: _SWSetCoefState, tables: RRTMGTableBundle):
    """Water-vapor self and foreign continuum contribution used below `laytrop`."""

    self_table = tables.sw_selfref[band]
    for_table = tables.sw_forref[band]
    inds = coef.indself - 1
    indf = coef.indfor - 1
    self_interp = _take_rows(self_table, inds) + coef.selffrac[..., None] * (
        _take_rows(self_table, inds + 1) - _take_rows(self_table, inds)
    )
    for_interp = _take_rows(for_table, indf) + coef.forfrac[..., None] * (
        _take_rows(for_table, indf + 1) - _take_rows(for_table, indf)
    )
    return coef.colh2o[..., None] * (coef.selffac[..., None] * self_interp + coef.forfac[..., None] * for_interp)


def _binary_params(spec_a, spec_b, strrat, multiplier):
    """Builds WRF binary-species interpolation parameters."""

    speccomb = spec_a + strrat * spec_b
    specparm = spec_a / jnp.maximum(speccomb, 1.0e-300)
    specparm = jnp.minimum(specparm, 1.0 - 1.0e-6)
    specmult = multiplier * specparm
    js = 1 + _trunc_int(specmult)
    fs = jnp.mod(specmult, 1.0)
    return speccomb, js, fs


def _sw_taumol(coef: _SWSetCoefState, tables: RRTMGTableBundle):
    """Ports WRF `taumol_sw` gas and Rayleigh optical-depth branches."""

    taug = []
    taur = []
    for band in range(14):
        absa = tables.sw_absa[band]
        absb = tables.sw_absb[band]
        nspa = tables.sw_nspa[band]
        nspb = tables.sw_nspb[band]
        strrat = tables.sw_strrat[band]
        lower = coef.lower_mask
        lower_idx0 = ((coef.jp - 1) * 5 + (coef.jt - 1)) * nspa + 1
        lower_idx1 = (coef.jp * 5 + (coef.jt1 - 1)) * nspa + 1
        upper_idx0 = ((coef.jp - 13) * 5 + (coef.jt - 1)) * nspb + 1
        upper_idx1 = ((coef.jp - 12) * 5 + (coef.jt1 - 1)) * nspb + 1

        if band in (0, 2):
            speccomb, js, fs = _binary_params(coef.colh2o, coef.colch4, strrat, 8.0)
            low = speccomb[..., None] * _interp_binary(absa, lower_idx0 + js - 1, lower_idx1 + js - 1, nspa, fs, coef) + _continuum_terms(
                band, coef, tables
            )
            high = coef.colch4[..., None] * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        elif band in (1, 5):
            speccomb_l, js_l, fs_l = _binary_params(coef.colh2o, coef.colco2, strrat, 8.0)
            low = speccomb_l[..., None] * _interp_binary(absa, lower_idx0 + js_l - 1, lower_idx1 + js_l - 1, nspa, fs_l, coef) + _continuum_terms(
                band, coef, tables
            )
            speccomb_u, js_u, fs_u = _binary_params(coef.colh2o, coef.colco2, strrat, 4.0)
            high = speccomb_u[..., None] * _interp_binary(absb, upper_idx0 + js_u - 1, upper_idx1 + js_u - 1, nspb, fs_u, coef)
            high = high + coef.colh2o[..., None] * coef.forfac[..., None] * (
                _take_rows(tables.sw_forref[band], coef.indfor - 1)
                + coef.forfrac[..., None]
                * (_take_rows(tables.sw_forref[band], coef.indfor) - _take_rows(tables.sw_forref[band], coef.indfor - 1))
            )
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        elif band == 3:
            speccomb, js, fs = _binary_params(coef.colh2o, coef.colco2, strrat, 8.0)
            low = speccomb[..., None] * _interp_binary(absa, lower_idx0 + js - 1, lower_idx1 + js - 1, nspa, fs, coef) + _continuum_terms(
                band, coef, tables
            )
            high = coef.colco2[..., None] * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        elif band == 4:
            low = coef.colh2o[..., None] * (_interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef) + _continuum_terms(band, coef, tables) / jnp.maximum(coef.colh2o[..., None], 1.0e-300))
            low = low + coef.colch4[..., None] * tables.sw_abs_ch4[band]
            high = coef.colh2o[..., None] * (
                _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef)
                + coef.forfac[..., None]
                * (
                    _take_rows(tables.sw_forref[band], coef.indfor - 1)
                    + coef.forfrac[..., None]
                    * (_take_rows(tables.sw_forref[band], coef.indfor) - _take_rows(tables.sw_forref[band], coef.indfor - 1))
                )
            ) + coef.colch4[..., None] * tables.sw_abs_ch4[band]
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        elif band == 6:
            o2adj = 1.6
            o2cont = 4.35e-4 * coef.colo2 / (350.0 * 2.0)
            speccomb, js, fs = _binary_params(coef.colh2o, o2adj * coef.colo2, strrat, 8.0)
            low = speccomb[..., None] * _interp_binary(absa, lower_idx0 + js - 1, lower_idx1 + js - 1, nspa, fs, coef) + _continuum_terms(
                band, coef, tables
            ) + o2cont[..., None]
            high = coef.colo2[..., None] * o2adj * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef) + o2cont[..., None]
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        elif band == 7:
            low = coef.colh2o[..., None] * (
                tables.sw_givfac[band] * _interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef)
                + _continuum_terms(band, coef, tables) / jnp.maximum(coef.colh2o[..., None], 1.0e-300)
            )
            tau = jnp.where(lower[..., None], low, jnp.zeros_like(low))
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        elif band == 8:
            speccomb, js, fs = _binary_params(coef.colh2o, coef.colo2, strrat, 8.0)
            low = speccomb[..., None] * _interp_binary(absa, lower_idx0 + js - 1, lower_idx1 + js - 1, nspa, fs, coef)
            low = low + coef.colo3[..., None] * tables.sw_abs_o3a[band] + _continuum_terms(band, coef, tables)
            high = coef.colo2[..., None] * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef) + coef.colo3[..., None] * tables.sw_abs_o3b[band]
            tau = jnp.where(lower[..., None], low, high)
            ray_low = coef.colmol[..., None] * (
                _take_rows(tables.sw_rayla[band].T, js - 1)
                + fs[..., None] * (_take_rows(tables.sw_rayla[band].T, js) - _take_rows(tables.sw_rayla[band].T, js - 1))
            )
            ray_high = coef.colmol[..., None] * tables.sw_raylb[band]
            ray = jnp.where(lower[..., None], ray_low, ray_high)
        elif band == 9:
            low = coef.colh2o[..., None] * _interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef) + coef.colo3[..., None] * tables.sw_abs_o3a[band]
            high = coef.colo3[..., None] * tables.sw_abs_o3b[band]
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        elif band == 10:
            tau = jnp.zeros(coef.colh2o.shape + (tables.sw_gpoint_mask.shape[1],), dtype=jnp.float64)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        elif band == 11:
            low = coef.colo3[..., None] * _interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef)
            high = coef.colo3[..., None] * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        elif band == 12:
            speccomb_l, js_l, fs_l = _binary_params(coef.colo3, coef.colo2, strrat, 8.0)
            low = speccomb_l[..., None] * _interp_binary(absa, lower_idx0 + js_l - 1, lower_idx1 + js_l - 1, nspa, fs_l, coef)
            speccomb_u, js_u, fs_u = _binary_params(coef.colo3, coef.colo2, strrat, 4.0)
            high = speccomb_u[..., None] * _interp_binary(absb, upper_idx0 + js_u - 1, upper_idx1 + js_u - 1, nspb, fs_u, coef)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        else:
            low = coef.colh2o[..., None] * (
                _interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef)
                + _continuum_terms(band, coef, tables) / jnp.maximum(coef.colh2o[..., None], 1.0e-300)
            ) + coef.colco2[..., None] * tables.sw_abs_co2[band]
            high = coef.colco2[..., None] * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef) + coef.colh2o[..., None] * tables.sw_abs_h2o[band]
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band]
        taug.append(tau * tables.sw_gpoint_mask[band])
        taur.append(ray * tables.sw_gpoint_mask[band])
    return jnp.stack(taug, axis=-2), jnp.stack(taur, axis=-2)


def _select_layer(values, idx):
    """Selects one layer per column from a bottom-to-top layer array."""

    gather_idx = idx[..., None]
    return jnp.take_along_axis(values, gather_idx, axis=-1)[..., 0]


def _source_indices(coef: _SWSetCoefState, layreffr, mode: str):
    """Approximates WRF `laysolfr` source-layer selection for SW bands."""

    nlay = coef.jp.shape[-1]
    laytrop_count = jnp.sum(coef.lower_mask.astype(jnp.int32), axis=-1)
    lower_default = jnp.maximum(laytrop_count - 1, 0)
    upper_default = jnp.full_like(lower_default, nlay - 1)
    crosses = (coef.jp[..., :-1] < layreffr) & (coef.jp[..., 1:] >= layreffr)
    has_cross = jnp.any(crosses, axis=-1)
    cross_idx = jnp.argmax(crosses.astype(jnp.int32), axis=-1) + 1
    if mode == "lower":
        target = jnp.minimum(cross_idx, lower_default)
        return jnp.where(has_cross, target, lower_default)
    return jnp.where(has_cross, cross_idx, upper_default)


def _interp_source(source_table, js, fs):
    """Interpolates WRF reduced solar source functions over species ratio."""

    table = source_table.T
    base = _take_rows(table, js - 1)
    return base + fs[..., None] * (_take_rows(table, js) - base)


def _sw_sfluxzen(coef: _SWSetCoefState, tables: RRTMGTableBundle):
    """Builds WRF `sfluxzen` source functions from reduced `sfluxref` tables."""

    sources = []
    for band in range(14):
        src = tables.sw_sfluxref[band]
        layreffr = tables.sw_layreffr[band]
        if band in (0, 4, 7, 9, 10, 11, 13):
            source = jnp.broadcast_to(src[:, 0], coef.colh2o.shape[:-1] + src[:, 0].shape)
        elif band == 1:
            idx = _source_indices(coef, layreffr, "upper")
            h2o = _select_layer(coef.colh2o, idx)
            co2 = _select_layer(coef.colco2, idx)
            _, js, fs = _binary_params(h2o, co2, tables.sw_strrat[band], 4.0)
            source = _interp_source(src, js, fs)
        elif band in (2,):
            idx = _source_indices(coef, layreffr, "lower")
            h2o = _select_layer(coef.colh2o, idx)
            ch4 = _select_layer(coef.colch4, idx)
            _, js, fs = _binary_params(h2o, ch4, tables.sw_strrat[band], 8.0)
            source = _interp_source(src, js, fs)
        elif band in (3, 5):
            idx = _source_indices(coef, layreffr, "lower")
            h2o = _select_layer(coef.colh2o, idx)
            co2 = _select_layer(coef.colco2, idx)
            _, js, fs = _binary_params(h2o, co2, tables.sw_strrat[band], 8.0)
            source = _interp_source(src, js, fs)
        elif band in (6, 8):
            idx = _source_indices(coef, layreffr, "lower")
            h2o = _select_layer(coef.colh2o, idx)
            o2 = _select_layer(coef.colo2, idx)
            o2_factor = 1.6 if band == 6 else 1.0
            _, js, fs = _binary_params(h2o, o2_factor * o2, tables.sw_strrat[band], 8.0)
            source = _interp_source(src, js, fs)
        elif band == 12:
            idx = _source_indices(coef, layreffr, "upper")
            o3 = _select_layer(coef.colo3, idx)
            o2 = _select_layer(coef.colo2, idx)
            _, js, fs = _binary_params(o3, o2, tables.sw_strrat[band], 4.0)
            source = _interp_source(src, js, fs)
        else:
            source = jnp.broadcast_to(src[:, 0], coef.colh2o.shape[:-1] + src[:, 0].shape)
        if band == 11:
            source = source * tables.sw_scalekur[band]
        sources.append(source * tables.sw_gpoint_mask[band])
    return jnp.stack(sources, axis=-2)


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
    coef = _sw_setcoef(qv_ext, p_ext, _extend_with_wrf_top_layer(state.T), pressure_interfaces, tables)
    tau_gas, tau_rayleigh = _sw_taumol(coef, tables)
    liquid_path_g = (qc_ext + 0.25 * qg_ext) * layer_mass_ext * 1000.0
    ice_path_g = (qi_ext + qs_ext + 0.75 * qg_ext) * layer_mass_ext * 1000.0

    sfluxzen = _sw_sfluxzen(coef, tables)
    liquid_coeff = tables.sw_cloud_liquid_extinction[:, None]
    ice_coeff = tables.sw_cloud_ice_extinction[:, None]
    liquid_ssa = tables.sw_cloud_liquid_ssa[:, None]
    ice_ssa = tables.sw_cloud_ice_ssa[:, None]
    liquid_asy = tables.sw_cloud_liquid_asymmetry[:, None]
    ice_asy = tables.sw_cloud_ice_asymmetry[:, None]
    mask = tables.sw_gpoint_mask

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
    top_flux_band = state.coszen[..., None, None] * sfluxzen
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
