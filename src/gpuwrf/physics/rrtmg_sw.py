"""JAX shortwave RRTMG-style radiation column kernel for M5-S3."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import lax
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
    MIN_COSZEN,
    MIN_LAYER_MASS,
    MIN_OPTICAL_DEPTH,
    N2O_VMR,
    O2_VMR,
    O3_BACKGROUND_VMR,
    WATER_VAPOR_MOLECULAR_WEIGHT_RATIO,
)
from gpuwrf.physics.rrtmg_tables import RRTMGTableBundle, RRTMG_TABLES


config.update("jax_enable_x64", True)

_SW_GPOINT_COUNTS = (6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12)
_SW_GLOBAL_GPOINT_INDEX = jnp.asarray(
    tuple(
        tuple((sum(_SW_GPOINT_COUNTS[:band]) + gpoint) if gpoint < _SW_GPOINT_COUNTS[band] else 0 for gpoint in range(12))
        for band in range(14)
    ),
    dtype=jnp.int32,
)
_SW_NTBL = 10000
_SW_TBLINT = 10000.0
_SW_OD_LO = 0.06
_SW_BPADE = 1.0 / 0.278
_SW_EXP_EPS = 1.0e-20

# Number of spectral bands in the SW two-stream solve.  The full per-g-point
# temporary that dominates fp64 VRAM is ~(ncol, nlev+1, _SW_NBANDS, 16); since
# the TOA->surface flux is an associative sum over (band, g-point), the
# reflectance/transmittance + vertical-quadrature work is tiled over the band
# axis so the peak live buffer is one tile of `_SW_GPOINT_CHUNK_BANDS` bands
# instead of all 14 at once.  Each tile reduces its g-points to a fp64 band
# partial; the disjoint per-band partials are summed in fp64 at the call site.
# Because that reduction is over distinct band columns in fp64, the result is
# PROVABLY INDEPENDENT of the chunk width (bit-identical across tile sizes — see
# proofs/v013/gpoint_chunk_rrtmg.json).  It differs from the LEGACY single-pass
# code only in that the original reduced the band axis in the working float32
# precision; doing the band sum in fp64 is a one-time, knob-independent
# precision *improvement* of <=~1e-5 rel, not a per-tile chunking artifact.
_SW_NBANDS = 14
# Default tile width.  One band per tile gives the lowest, most robust peak: the
# scan visits the 14 bands sequentially and XLA frees each band's two-stream
# working set before the next, roughly HALVING the deep-column peak vs the
# single-pass solve (proofs/v013: nlev=48 ncol=24576 -> 16730 MiB unchunked ->
# ~8030 MiB at chunk=1).  Intermediate widths (e.g. 7) can be WORSE than either
# extreme because XLA schedules the two large tiles concurrently, so the default
# is the smallest tile.  Set to `_SW_NBANDS` to recover the single-pass solve.
# The numerical result is independent of this value (proofs/v013).
_SW_GPOINT_CHUNK_BANDS = 1


@jax.tree_util.register_pytree_node_class
class RRTMGSWColumnState:
    """Pytree for independent shortwave radiation columns on mass levels."""

    __slots__ = (
        "T",
        "p",
        "qv",
        "qc",
        "qi",
        "qs",
        "qg",
        "cloud_fraction",
        "surface_albedo",
        "coszen",
        "dz",
        "rho",
        "solar_source_scale",
    )

    def __init__(
        self,
        T,
        p,
        qv,
        qc,
        qi,
        qs,
        qg,
        cloud_fraction,
        surface_albedo,
        coszen,
        dz,
        rho,
        solar_source_scale=1.0,
    ) -> None:
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
        self.solar_source_scale = jnp.asarray(solar_source_scale)

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
    surface_direct: jnp.ndarray
    surface_diffuse: jnp.ndarray
    surface_diffuse_fraction: jnp.ndarray
    topographic_correction_factor: jnp.ndarray
    surface_down_topographic: jnp.ndarray
    surface_up_topographic: jnp.ndarray
    surface_absorbed_topographic: jnp.ndarray


class RRTMGSWTopographyState(NamedTuple):
    """WRF `TOPO_RAD_ADJ` geometry fields for terrain-adjusted SW flux."""

    latitude_deg: jnp.ndarray
    declination_rad: jnp.ndarray
    hour_angle_rad: jnp.ndarray
    slope_rad: jnp.ndarray
    slope_azimuth_rad: jnp.ndarray
    shadow_mask: jnp.ndarray


class RRTMGSWTopographicAdjustment(NamedTuple):
    """Surface SW fluxes after WRF slope/aspect/shadow correction."""

    surface_down: jnp.ndarray
    surface_up: jnp.ndarray
    surface_absorbed: jnp.ndarray
    correction_factor: jnp.ndarray


class RRTMGSWIntermediateState(NamedTuple):
    """SW solver-entry state exposed for M5-S3.z WRF intermediate-oracle checks."""

    jp: jnp.ndarray
    jt: jnp.ndarray
    jt1: jnp.ndarray
    fac00: jnp.ndarray
    fac01: jnp.ndarray
    fac10: jnp.ndarray
    fac11: jnp.ndarray
    indself: jnp.ndarray
    indfor: jnp.ndarray
    selffac: jnp.ndarray
    forfac: jnp.ndarray
    colmol: jnp.ndarray
    taug: jnp.ndarray
    taur: jnp.ndarray
    sfluxzen: jnp.ndarray
    pcldfmc: jnp.ndarray
    ptaucmc: jnp.ndarray
    pasycmc: jnp.ndarray
    pomgcmc: jnp.ndarray
    ptaormc: jnp.ndarray
    spcvmc_zref: jnp.ndarray
    spcvmc_ztra: jnp.ndarray
    spcvmc_zrefd: jnp.ndarray
    spcvmc_ztrad: jnp.ndarray
    spcvmc_zref_clear: jnp.ndarray
    spcvmc_ztra_clear: jnp.ndarray
    spcvmc_zrefd_clear: jnp.ndarray
    spcvmc_ztrad_clear: jnp.ndarray
    spcvmc_zref_cloud: jnp.ndarray
    spcvmc_ztra_cloud: jnp.ndarray
    spcvmc_zrefd_cloud: jnp.ndarray
    spcvmc_ztrad_cloud: jnp.ndarray
    spcvmc_direct_trans: jnp.ndarray
    spcvmc_zfd_flux: jnp.ndarray
    spcvmc_zfu_flux: jnp.ndarray


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


def _setcoef_state_dtype(coef: _SWSetCoefState, dtype) -> _SWSetCoefState:
    """Casts WRF real-valued setcoef fields while preserving integer indices."""

    return coef._replace(
        colh2o=coef.colh2o.astype(dtype),
        colco2=coef.colco2.astype(dtype),
        colo3=coef.colo3.astype(dtype),
        coln2o=coef.coln2o.astype(dtype),
        colch4=coef.colch4.astype(dtype),
        colo2=coef.colo2.astype(dtype),
        colmol=coef.colmol.astype(dtype),
        coldry=coef.coldry.astype(dtype),
        fac00=coef.fac00.astype(dtype),
        fac01=coef.fac01.astype(dtype),
        fac10=coef.fac10.astype(dtype),
        fac11=coef.fac11.astype(dtype),
        selffac=coef.selffac.astype(dtype),
        selffrac=coef.selffrac.astype(dtype),
        forfac=coef.forfac.astype(dtype),
        forfrac=coef.forfrac.astype(dtype),
    )


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
        solar_source_scale=jnp.maximum(state.solar_source_scale, 0.0),
    )


def _solar_source_scale(state: RRTMGSWColumnState, dtype) -> jnp.ndarray:
    """Broadcasts WRF's RRTMG `scon / rrsw_scon` source multiplier by column."""

    scale = jnp.asarray(state.solar_source_scale, dtype=dtype)
    return jnp.broadcast_to(scale, state.coszen.shape).astype(dtype)


def wrf_topographic_sw_correction_factor(
    coszen,
    diffuse_fraction,
    latitude_deg,
    declination_rad,
    hour_angle_rad,
    slope_rad,
    slope_azimuth_rad,
    shadow_mask,
):
    """Ports WRF `TOPO_RAD_ADJ` slope/aspect/shadow SW correction factor."""

    dtype = jnp.result_type(
        coszen,
        diffuse_fraction,
        latitude_deg,
        declination_rad,
        hour_angle_rad,
        slope_rad,
        slope_azimuth_rad,
        jnp.float32,
    )
    csza = jnp.asarray(coszen, dtype=dtype)
    diffuse = jnp.asarray(diffuse_fraction, dtype=dtype)
    xlat = jnp.asarray(latitude_deg, dtype=dtype) * jnp.asarray(jnp.pi / 180.0, dtype=dtype)
    declin = jnp.asarray(declination_rad, dtype=dtype)
    hrang = jnp.asarray(hour_angle_rad, dtype=dtype)
    slope = jnp.asarray(slope_rad, dtype=dtype)
    slp_azi = jnp.asarray(slope_azimuth_rad, dtype=dtype)
    shadowed = jnp.asarray(shadow_mask) == 1

    sin_slope = jnp.sin(slope)
    cos_slope = jnp.cos(slope)
    sin_slp_azi = jnp.sin(slp_azi)
    cos_slp_azi = jnp.cos(slp_azi)
    sin_lat = jnp.sin(xlat)
    cos_lat = jnp.cos(xlat)
    sin_hrang = jnp.sin(hrang)
    cos_hrang = jnp.cos(hrang)

    csza_slp = (
        (sin_lat * cos_hrang) * (-cos_slp_azi * sin_slope)
        - sin_hrang * (sin_slp_azi * sin_slope)
        + (cos_lat * cos_hrang) * cos_slope
    ) * jnp.cos(declin) + (cos_lat * (cos_slp_azi * sin_slope) + sin_lat * cos_slope) * jnp.sin(declin)
    csza_slp = jnp.where(csza_slp <= 1.0e-4, 0.0, csza_slp)
    csza_slp = jnp.where(shadowed, 0.0, csza_slp)

    no_slope_effect = (slope == 0.0) | (diffuse == 1.0)
    branch_correction = jnp.where(shadowed, diffuse, 1.0)
    slope_correction = diffuse + (1.0 - diffuse) * csza_slp / jnp.maximum(csza, 1.0e-12)
    correction = jnp.where(no_slope_effect, branch_correction, slope_correction)
    return jnp.where(csza <= 1.0e-4, 1.0, correction)


def apply_wrf_topographic_sw_adjustment(
    surface_down,
    surface_up,
    surface_absorbed,
    coszen,
    diffuse_fraction,
    topography: RRTMGSWTopographyState,
) -> RRTMGSWTopographicAdjustment:
    """Applies WRF terrain SW correction while preserving surface albedo."""

    correction = wrf_topographic_sw_correction_factor(
        coszen=coszen,
        diffuse_fraction=diffuse_fraction,
        latitude_deg=topography.latitude_deg,
        declination_rad=topography.declination_rad,
        hour_angle_rad=topography.hour_angle_rad,
        slope_rad=topography.slope_rad,
        slope_azimuth_rad=topography.slope_azimuth_rad,
        shadow_mask=topography.shadow_mask,
    )
    return RRTMGSWTopographicAdjustment(
        surface_down=surface_down * correction,
        surface_up=surface_up * correction,
        surface_absorbed=surface_absorbed * correction,
        correction_factor=correction,
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


_O3SUM = jnp.asarray(
    (
        5.297e-8,
        5.852e-8,
        6.579e-8,
        7.505e-8,
        8.577e-8,
        9.895e-8,
        1.175e-7,
        1.399e-7,
        1.677e-7,
        2.003e-7,
        2.571e-7,
        3.325e-7,
        4.438e-7,
        6.255e-7,
        8.168e-7,
        1.036e-6,
        1.366e-6,
        1.855e-6,
        2.514e-6,
        3.240e-6,
        4.033e-6,
        4.854e-6,
        5.517e-6,
        6.089e-6,
        6.689e-6,
        1.106e-5,
        1.462e-5,
        1.321e-5,
        9.856e-6,
        5.960e-6,
        5.960e-6,
    ),
    dtype=jnp.float64,
)
_PPSUM = jnp.asarray(
    (
        955.890,
        850.532,
        754.599,
        667.742,
        589.841,
        519.421,
        455.480,
        398.085,
        347.171,
        301.735,
        261.310,
        225.360,
        193.419,
        165.490,
        141.032,
        120.125,
        102.689,
        87.829,
        75.123,
        64.306,
        55.086,
        47.209,
        40.535,
        34.795,
        29.865,
        19.122,
        9.277,
        4.660,
        2.421,
        1.294,
        0.647,
    ),
    dtype=jnp.float64,
)
_O3WIN = jnp.asarray(
    (
        4.629e-8,
        4.686e-8,
        5.017e-8,
        5.613e-8,
        6.871e-8,
        8.751e-8,
        1.138e-7,
        1.516e-7,
        2.161e-7,
        3.264e-7,
        4.968e-7,
        7.338e-7,
        1.017e-6,
        1.308e-6,
        1.625e-6,
        2.011e-6,
        2.516e-6,
        3.130e-6,
        3.840e-6,
        4.703e-6,
        5.486e-6,
        6.289e-6,
        6.993e-6,
        7.494e-6,
        8.197e-6,
        9.632e-6,
        1.113e-5,
        1.146e-5,
        9.389e-6,
        6.135e-6,
        6.135e-6,
    ),
    dtype=jnp.float64,
)
_PPWIN = jnp.asarray(
    (
        955.747,
        841.783,
        740.199,
        649.538,
        568.404,
        495.815,
        431.069,
        373.464,
        322.354,
        277.190,
        237.635,
        203.433,
        174.070,
        148.949,
        127.408,
        108.915,
        93.114,
        79.551,
        67.940,
        58.072,
        49.593,
        42.318,
        36.138,
        30.907,
        26.362,
        16.423,
        7.583,
        3.620,
        1.807,
        0.938,
        0.469,
    ),
    dtype=jnp.float64,
)


def _wrf_o3_vmr(pressure_interfaces_pa):
    """Replicates WRF `inirad/O3DATA` climatological ozone for o3input=0."""

    o3sum = _O3SUM.astype(jnp.float32)
    ppsum = _PPSUM.astype(jnp.float32)
    o3win = _O3WIN.astype(jnp.float32)
    ppwin = _PPWIN.astype(jnp.float32)
    o3ann_tail = o3win[:-1] + ((o3win[1:] - o3win[:-1]) / (ppwin[1:] - ppwin[:-1])) * (ppsum[1:] - ppwin[:-1])
    o3ann = jnp.concatenate(
        (
            jnp.asarray([jnp.float32(0.5) * (jnp.float32(5.297e-8) + jnp.float32(4.629e-8))], dtype=jnp.float32),
            jnp.float32(0.5) * (o3ann_tail + o3sum[1:]),
        )
    )
    ppwrkh = jnp.concatenate(
        (
            jnp.asarray([1100.0], dtype=jnp.float32),
            jnp.float32(0.5) * (ppsum[1:] + ppsum[:-1]),
            jnp.asarray([0.0], dtype=jnp.float32),
        )
    )
    plev = (pressure_interfaces_pa * 0.01).astype(jnp.float32)
    pb = plev[..., :-1, None]
    pt = plev[..., 1:, None]
    lower = ppwrkh[:-1]
    upper = ppwrkh[1:]
    zero = jnp.float32(0.0)
    pb1 = jnp.where(pb <= lower, zero, pb - lower)
    pb2 = jnp.where(pb <= upper, zero, pb - upper)
    pt1 = jnp.where(pt <= lower, zero, pt - lower)
    pt2 = jnp.where(pt <= upper, zero, pt - upper)
    o3_mmr = jnp.sum((pb2 - pb1 - pt2 + pt1) * o3ann, axis=-1) / jnp.maximum(plev[..., :-1] - plev[..., 1:], jnp.float32(1.0e-12))
    return (o3_mmr * jnp.float32(0.603461)).astype(jnp.float64)


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

    dtype = jnp.float32
    qv = qv.astype(dtype)
    p_pa = p_pa.astype(dtype)
    t_k = t_k.astype(dtype)
    pressure_interfaces_pa = pressure_interfaces_pa.astype(dtype)
    preflog = tables.sw_preflog.astype(dtype)
    tref = tables.sw_tref.astype(dtype)
    h2ovmr = qv * WATER_VAPOR_MOLECULAR_WEIGHT_RATIO
    amm = (1.0 - h2ovmr) * jnp.asarray(DRY_AIR_MOLECULAR_WEIGHT, dtype=dtype) + h2ovmr * jnp.asarray(18.0160, dtype=dtype)
    pavel = jnp.maximum(p_pa * jnp.asarray(0.01, dtype=dtype), jnp.asarray(1.0e-6, dtype=dtype))
    pz = jnp.maximum(pressure_interfaces_pa * jnp.asarray(0.01, dtype=dtype), jnp.asarray(1.0e-12, dtype=dtype))
    dp_mb = jnp.maximum(pz[..., :-1] - pz[..., 1:], jnp.asarray(1.0e-12, dtype=dtype))
    coldry = dp_mb * jnp.asarray(1.0e3, dtype=dtype) * jnp.asarray(AVOGADRO, dtype=dtype) / (
        jnp.asarray(1.0e2, dtype=dtype) * jnp.asarray(GRAVITY, dtype=dtype) * amm * (1.0 + h2ovmr)
    )

    plog = jnp.log(pavel)
    jp = _trunc_int(36.0 - 5.0 * (plog + 0.04))
    jp = jnp.clip(jp, 1, 58)
    jp1 = jp + 1
    fp = 5.0 * (jnp.take(preflog, jp - 1, axis=0) - plog)
    tref0 = jnp.take(tref, jp - 1, axis=0)
    tref1 = jnp.take(tref, jp1 - 1, axis=0)
    jt = _trunc_int(3.0 + (t_k - tref0) / 15.0)
    jt = jnp.clip(jt, 1, 4)
    ft = ((t_k - tref0) / 15.0) - (jt - 3).astype(dtype)
    jt1 = _trunc_int(3.0 + (t_k - tref1) / 15.0)
    jt1 = jnp.clip(jt1, 1, 4)
    ft1 = ((t_k - tref1) / 15.0) - (jt1 - 3).astype(dtype)

    water = h2ovmr
    scalefac = pavel * (296.0 / 1013.0) / t_k
    lower = plog > 4.56
    forfac = scalefac / (1.0 + water)
    lower_for_factor = (332.0 - t_k) / 36.0
    upper_for_factor = (t_k - 188.0) / 36.0
    indfor_lower = jnp.minimum(2, jnp.maximum(1, _trunc_int(lower_for_factor)))
    indfor = jnp.where(lower, indfor_lower, 3)
    forfrac = jnp.where(lower, lower_for_factor - indfor.astype(dtype), upper_for_factor - 1.0)

    selffac = jnp.where(lower, water * forfac, 0.0)
    self_factor = (t_k - 188.0) / 7.2
    indself = jnp.minimum(9, jnp.maximum(1, _trunc_int(self_factor) - 7))
    selffrac = jnp.where(lower, self_factor - (indself + 7).astype(dtype), 0.0)
    indself = jnp.where(lower, indself, 0)

    compfp = 1.0 - fp
    fac10 = compfp * ft
    fac00 = compfp * (1.0 - ft)
    fac11 = fp * ft1
    fac01 = fp * (1.0 - ft1)

    o3_vmr = _wrf_o3_vmr(pressure_interfaces_pa)
    scale = jnp.asarray(1.0e-20, dtype=dtype)
    colh2o = scale * coldry * h2ovmr
    colco2 = scale * coldry * jnp.asarray(CO2_VMR, dtype=dtype)
    colo3 = scale * coldry * o3_vmr.astype(dtype)
    coln2o = scale * coldry * jnp.asarray(N2O_VMR, dtype=dtype)
    colch4 = scale * coldry * jnp.asarray(CH4_VMR, dtype=dtype)
    colo2 = scale * coldry * jnp.asarray(O2_VMR, dtype=dtype)
    colmol = scale * coldry + colh2o

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

    dtype = coef.colh2o.dtype
    self_table = tables.sw_selfref[band].astype(dtype)
    for_table = tables.sw_forref[band].astype(dtype)
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

    dtype = spec_a.dtype
    strrat = jnp.asarray(strrat, dtype=dtype)
    multiplier = jnp.asarray(multiplier, dtype=dtype)
    speccomb = spec_a + strrat * spec_b
    specparm = spec_a / jnp.maximum(speccomb, jnp.asarray(1.0e-30, dtype=dtype))
    specparm = jnp.minimum(specparm, jnp.asarray(1.0 - 1.0e-6, dtype=dtype))
    specmult = multiplier * specparm
    js = 1 + _trunc_int(specmult)
    fs = jnp.mod(specmult, jnp.asarray(1.0, dtype=dtype))
    return speccomb, js, fs


def _sw_taumol(coef: _SWSetCoefState, tables: RRTMGTableBundle):
    """Ports WRF `taumol_sw` gas and Rayleigh optical-depth branches."""

    dtype = jnp.float32
    coef = _setcoef_state_dtype(coef, dtype)
    gpoint_mask = tables.sw_gpoint_mask.astype(dtype)
    taug = []
    taur = []
    for band in range(14):
        absa = tables.sw_absa[band].astype(dtype)
        absb = tables.sw_absb[band].astype(dtype)
        nspa = tables.sw_nspa[band]
        nspb = tables.sw_nspb[band]
        strrat = jnp.asarray(tables.sw_strrat[band], dtype=dtype)
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
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        elif band in (1, 5):
            speccomb_l, js_l, fs_l = _binary_params(coef.colh2o, coef.colco2, strrat, 8.0)
            low = speccomb_l[..., None] * _interp_binary(absa, lower_idx0 + js_l - 1, lower_idx1 + js_l - 1, nspa, fs_l, coef) + _continuum_terms(
                band, coef, tables
            )
            speccomb_u, js_u, fs_u = _binary_params(coef.colh2o, coef.colco2, strrat, 4.0)
            high = speccomb_u[..., None] * _interp_binary(absb, upper_idx0 + js_u - 1, upper_idx1 + js_u - 1, nspb, fs_u, coef)
            high = high + coef.colh2o[..., None] * coef.forfac[..., None] * (
                _take_rows(tables.sw_forref[band].astype(dtype), coef.indfor - 1)
                + coef.forfrac[..., None]
                * (_take_rows(tables.sw_forref[band].astype(dtype), coef.indfor) - _take_rows(tables.sw_forref[band].astype(dtype), coef.indfor - 1))
            )
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        elif band == 3:
            speccomb, js, fs = _binary_params(coef.colh2o, coef.colco2, strrat, 8.0)
            low = speccomb[..., None] * _interp_binary(absa, lower_idx0 + js - 1, lower_idx1 + js - 1, nspa, fs, coef) + _continuum_terms(
                band, coef, tables
            )
            high = coef.colco2[..., None] * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        elif band == 4:
            low = coef.colh2o[..., None] * (_interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef) + _continuum_terms(band, coef, tables) / jnp.maximum(coef.colh2o[..., None], 1.0e-300))
            low = low + coef.colch4[..., None] * tables.sw_abs_ch4[band].astype(dtype)
            high = coef.colh2o[..., None] * (
                _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef)
                + coef.forfac[..., None]
                * (
                    _take_rows(tables.sw_forref[band].astype(dtype), coef.indfor - 1)
                    + coef.forfrac[..., None]
                    * (_take_rows(tables.sw_forref[band].astype(dtype), coef.indfor) - _take_rows(tables.sw_forref[band].astype(dtype), coef.indfor - 1))
                )
            ) + coef.colch4[..., None] * tables.sw_abs_ch4[band].astype(dtype)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        elif band == 6:
            o2adj = jnp.asarray(1.6, dtype=dtype)
            o2cont = jnp.asarray(4.35e-4 / (350.0 * 2.0), dtype=dtype) * coef.colo2
            speccomb, js, fs = _binary_params(coef.colh2o, o2adj * coef.colo2, strrat, 8.0)
            low = speccomb[..., None] * _interp_binary(absa, lower_idx0 + js - 1, lower_idx1 + js - 1, nspa, fs, coef) + _continuum_terms(
                band, coef, tables
            ) + o2cont[..., None]
            high = coef.colo2[..., None] * o2adj * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef) + o2cont[..., None]
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        elif band == 7:
            low = coef.colh2o[..., None] * (
                tables.sw_givfac[band].astype(dtype) * _interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef)
                + _continuum_terms(band, coef, tables) / jnp.maximum(coef.colh2o[..., None], jnp.asarray(1.0e-30, dtype=dtype))
            )
            tau = jnp.where(lower[..., None], low, jnp.zeros_like(low))
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        elif band == 8:
            speccomb, js, fs = _binary_params(coef.colh2o, coef.colo2, strrat, 8.0)
            low = speccomb[..., None] * _interp_binary(absa, lower_idx0 + js - 1, lower_idx1 + js - 1, nspa, fs, coef)
            low = low + coef.colo3[..., None] * tables.sw_abs_o3a[band].astype(dtype) + _continuum_terms(band, coef, tables)
            high = coef.colo2[..., None] * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef) + coef.colo3[..., None] * tables.sw_abs_o3b[band].astype(dtype)
            tau = jnp.where(lower[..., None], low, high)
            ray_low = coef.colmol[..., None] * (
                _take_rows(tables.sw_rayla[band].T.astype(dtype), js - 1)
                + fs[..., None] * (_take_rows(tables.sw_rayla[band].T.astype(dtype), js) - _take_rows(tables.sw_rayla[band].T.astype(dtype), js - 1))
            )
            ray_high = coef.colmol[..., None] * tables.sw_raylb[band].astype(dtype)
            ray = jnp.where(lower[..., None], ray_low, ray_high)
        elif band == 9:
            low = coef.colh2o[..., None] * _interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef) + coef.colo3[..., None] * tables.sw_abs_o3a[band].astype(dtype)
            high = coef.colo3[..., None] * tables.sw_abs_o3b[band].astype(dtype)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        elif band == 10:
            tau = jnp.zeros(coef.colh2o.shape + (gpoint_mask.shape[1],), dtype=dtype)
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        elif band == 11:
            low = coef.colo3[..., None] * _interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef)
            high = coef.colo3[..., None] * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        elif band == 12:
            speccomb_l, js_l, fs_l = _binary_params(coef.colo3, coef.colo2, strrat, 8.0)
            low = speccomb_l[..., None] * _interp_binary(absa, lower_idx0 + js_l - 1, lower_idx1 + js_l - 1, nspa, fs_l, coef)
            speccomb_u, js_u, fs_u = _binary_params(coef.colo3, coef.colo2, strrat, 4.0)
            high = speccomb_u[..., None] * _interp_binary(absb, upper_idx0 + js_u - 1, upper_idx1 + js_u - 1, nspb, fs_u, coef)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        else:
            low = coef.colh2o[..., None] * (
                _interp_four_rows(absa, lower_idx0, lower_idx1, nspa, coef)
                + _continuum_terms(band, coef, tables) / jnp.maximum(coef.colh2o[..., None], jnp.asarray(1.0e-30, dtype=dtype))
            ) + coef.colco2[..., None] * tables.sw_abs_co2[band].astype(dtype)
            high = coef.colco2[..., None] * _interp_four_rows(absb, upper_idx0, upper_idx1, nspb, coef) + coef.colh2o[..., None] * tables.sw_abs_h2o[band].astype(dtype)
            tau = jnp.where(lower[..., None], low, high)
            ray = coef.colmol[..., None] * tables.sw_rayl[band].astype(dtype)
        taug.append(tau * gpoint_mask[band])
        taur.append(ray * gpoint_mask[band])
    return jnp.stack(taug, axis=-2), jnp.stack(taur, axis=-2)


def _sw_taumol_fused(coef: _SWSetCoefState, tables: RRTMGTableBundle):
    """Returns validated SW optical depths through a band-axis `lax.scan` barrier."""

    taug, taur = _sw_taumol(coef, tables)

    def keep_band(_, band_index):
        return None, (taug[..., band_index, :], taur[..., band_index, :])

    _, (taug_scan, taur_scan) = lax.scan(keep_band, None, jnp.arange(14, dtype=jnp.int32))
    return jnp.moveaxis(taug_scan, 0, -2), jnp.moveaxis(taur_scan, 0, -2)


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


def _source_active(coef: _SWSetCoefState, mode: str):
    """Matches whether WRF's lower/upper `taumol_sw` source loop executes."""

    lower_count = jnp.sum(coef.lower_mask.astype(jnp.int32), axis=-1)
    if mode == "lower":
        return lower_count > 0
    return lower_count < coef.jp.shape[-1]


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
        mode = "upper" if band in (0, 1, 11, 12, 13) else "lower"
        active = _source_active(coef, mode)
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
        source = jnp.where(active[..., None], source, 0.0)
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


def _kissvec_step(seed1, seed2, seed3, seed4):
    """Advances WRF MCICA's KISS vector RNG for all active columns."""

    seed1 = seed1 * jnp.uint32(69069) + jnp.uint32(1327217885)

    def shift_xor(value, amount):
        if amount > 0:
            shifted = jnp.left_shift(value, jnp.uint32(amount))
        else:
            shifted = jnp.right_shift(value, jnp.uint32(-amount))
        return jnp.bitwise_xor(value, shifted)

    seed2 = shift_xor(shift_xor(shift_xor(seed2, 13), -17), 5)
    seed3 = jnp.uint32(18000) * jnp.bitwise_and(seed3, jnp.uint32(65535)) + jnp.right_shift(seed3, jnp.uint32(16))
    seed4 = jnp.uint32(30903) * jnp.bitwise_and(seed4, jnp.uint32(65535)) + jnp.right_shift(seed4, jnp.uint32(16))
    kiss = seed1 + seed2 + jnp.left_shift(seed3, jnp.uint32(16)) + seed4
    signed = kiss.astype(jnp.int32).astype(jnp.float64)
    random = signed * 2.328306e-10 + 0.5
    return seed1, seed2, seed3, seed4, random


def _mcica_random_overlap_mask(p_pa, cloud_fraction, gpoint_mask):
    """Builds WRF `mcica_subcol_sw` random-overlap cloud masks for reduced SW g-points."""

    # WRF passes play in mb, then mcica_subcol_sw converts it back to Pa in
    # real*4 before deriving KISS seeds from the bottom four layer pressures.
    p_seed = (p_pa.astype(jnp.float32) * jnp.float32(0.01)).astype(jnp.float32) * jnp.float32(100.0)
    p_seed = p_seed.astype(jnp.float64)
    frac = p_seed - jnp.floor(p_seed)
    seed = (frac[..., :4].astype(jnp.float32) * jnp.float32(1.0e9)).astype(jnp.uint32)
    seed1, seed2, seed3, seed4 = (seed[..., 0], seed[..., 1], seed[..., 2], seed[..., 3])
    seed1, seed2, seed3, seed4, _ = _kissvec_step(seed1, seed2, seed3, seed4)
    nlay = p_pa.shape[-1]
    nsteps = sum(_SW_GPOINT_COUNTS) * nlay

    def advance(carry, _):
        s1, s2, s3, s4 = carry
        s1, s2, s3, s4, random = _kissvec_step(s1, s2, s3, s4)
        return (s1, s2, s3, s4), random

    _, cdf_flat = lax.scan(advance, (seed1, seed2, seed3, seed4), None, length=nsteps)
    cdf = jnp.reshape(cdf_flat, (sum(_SW_GPOINT_COUNTS), nlay) + p_pa.shape[:-1])
    cdf = jnp.moveaxis(cdf, (0, 1), (-1, -2))
    cloudy_global = cdf >= (1.0 - cloud_fraction[..., :, None])
    cloudy_reduced = jnp.take(cloudy_global, _SW_GLOBAL_GPOINT_INDEX, axis=-1)
    return cloudy_reduced.astype(jnp.float64) * gpoint_mask


def _sw_transmittance_lookup(optical_depth):
    """Mirrors WRF `rrsw_tbl` exponential lookup with low-tau expansion."""

    dtype = optical_depth.dtype
    one = jnp.asarray(1.0, dtype=dtype)
    tau = jnp.minimum(optical_depth, jnp.asarray(500.0, dtype=dtype))
    tblint = jnp.asarray(_SW_TBLINT, dtype=dtype)
    bpade = jnp.asarray(_SW_BPADE, dtype=dtype)
    tblind = tau / (bpade + tau)
    idx = jnp.clip((tblint * tblind + jnp.asarray(0.5, dtype=dtype)).astype(jnp.int32), 0, _SW_NTBL)
    tfn = idx.astype(dtype) / tblint
    tau_tbl = bpade * tfn / jnp.maximum(one - tfn, jnp.asarray(1.0e-30, dtype=dtype))
    table_value = jnp.maximum(jnp.exp(-tau_tbl), jnp.asarray(_SW_EXP_EPS, dtype=dtype))
    table_value = jnp.where(idx == 0, one, table_value)
    table_value = jnp.where(idx == _SW_NTBL, jnp.asarray(_SW_EXP_EPS, dtype=dtype), table_value)
    expansion = one - tau + jnp.asarray(0.5, dtype=dtype) * tau * tau
    return jnp.where(tau <= jnp.asarray(_SW_OD_LO, dtype=dtype), expansion, table_value)


def _reftra_eddington(tau, omega, asymmetry, mu0, active):
    """Computes Eddington direct/diffuse layer reflectance and transmittance."""

    tau = jnp.maximum(tau, MIN_OPTICAL_DEPTH).astype(jnp.float32)
    omega = jnp.clip(omega, 0.0, 0.999999).astype(jnp.float32)
    asymmetry = jnp.clip(asymmetry, -0.999999, 0.999999).astype(jnp.float32)
    mu0 = jnp.maximum(mu0[..., None, None, None], 1.0e-6).astype(jnp.float32)
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
    exp_mu = _sw_transmittance_lookup(tau / mu0)
    pref_cons = (zgt - za_cons * (1.0 - exp_mu)) / (1.0 + zgt)
    ptra_cons = 1.0 - pref_cons
    prefd_cons = zgt / (1.0 + zgt)
    ptrad_cons = 1.0 - prefd_cons
    low_tau_identity = exp_mu == 1.0
    pref_cons = jnp.where(low_tau_identity, 0.0, pref_cons)
    ptra_cons = jnp.where(low_tau_identity, 1.0, ptra_cons)
    prefd_cons = jnp.where(low_tau_identity, 0.0, prefd_cons)
    ptrad_cons = jnp.where(low_tau_identity, 1.0, ptrad_cons)

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

    exp_h = _sw_transmittance_lookup(zrk * tau)
    inv_exp_h = 1.0 / jnp.maximum(exp_h, 1.0e-300)
    inv_exp_mu = 1.0 / jnp.maximum(exp_mu, 1.0e-300)
    zdenr = zr4 * inv_exp_h + zr5 * exp_h
    zdent = zt4 * inv_exp_h + zt5 * exp_h
    singular = (zdenr >= -1.0e-8) & (zdenr <= 1.0e-8)
    pref_non_raw = omega * (zr1 * inv_exp_h - zr2 * exp_h - zr3 * exp_mu) / jnp.where(singular, 1.0, zdenr)
    ptra_non_raw = exp_mu - exp_mu * omega * (zt1 * inv_exp_h - zt2 * exp_h - zt3 * inv_exp_mu) / jnp.where(
        jnp.abs(zdent) > 1.0e-300, zdent, 1.0
    )
    pref_non = jnp.where(singular, 1.0e-8, pref_non_raw)
    ptra_non = jnp.where(singular, exp_mu, ptra_non_raw)
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
    return (pref, prefd, ptra, ptrad)


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
    return flux_down, flux_up, ptdbt


def _sw_band_tile_fluxes(
    tau_clear,
    omega_clear,
    asymmetry_clear,
    tau_total_cloud,
    omega_total_cloud,
    asymmetry_total_cloud,
    cloud_amount,
    mask,
    sfluxzen,
    coszen,
    surface_albedo,
    source_scale,
    partial_dtype,
):
    """Two-stream solve + (band, g-point) reduction for one band tile.

    Operates on a band slice ``[..., nlev, tile_bands, gpoint]`` of the SW
    optics.  The Eddington reflectance/transmittance and the vertical-quadrature
    adding method are purely elementwise in the (band, g-point) trailing axes —
    bands never couple — so processing a tile in isolation yields exactly the
    same per-(band, g-point) flux that the full single-pass solve would.  The
    returned arrays are the (tile-band, g-point)-summed fluxes
    ``[..., nlev+1]`` for down / up / direct (cast to ``partial_dtype`` =
    fp64 under force_fp64 BEFORE reducing), so the heavy
    ``[..., nlev+1, tile_bands, gpoint]`` temporary is freed before the next
    tile.  Accumulating these fp64 per-tile fluxes over the (disjoint) tiles
    reproduces the original ``sum(axis=(-1, -2))`` reduction to fp64 precision,
    independent of the tile width (proofs/v013).
    """

    tau_clear_top_down = jnp.flip(tau_clear, axis=-3)
    omega_clear_top_down = jnp.flip(omega_clear, axis=-3)
    asymmetry_clear_top_down = jnp.flip(asymmetry_clear, axis=-3)
    tau_cloud_top_down = jnp.flip(tau_total_cloud, axis=-3)
    omega_cloud_top_down = jnp.flip(omega_total_cloud, axis=-3)
    asymmetry_cloud_top_down = jnp.flip(asymmetry_total_cloud, axis=-3)
    cloud_top_down = jnp.flip(jnp.clip(cloud_amount, 0.0, 1.0), axis=-3)
    active_top_down = jnp.flip(mask + jnp.zeros_like(tau_clear), axis=-3)
    cloud_active_top_down = active_top_down * (cloud_top_down > 1.0e-12)
    pref_clear, prefd_clear, ptra_clear, ptrad_clear = _reftra_eddington(
        tau_clear_top_down, omega_clear_top_down, asymmetry_clear_top_down, coszen, active_top_down
    )
    pref_cloud, prefd_cloud, ptra_cloud, ptrad_cloud = _reftra_eddington(
        tau_cloud_top_down, omega_cloud_top_down, asymmetry_cloud_top_down, coszen, cloud_active_top_down
    )
    cloud_top_down = cloud_top_down.astype(pref_clear.dtype)
    active_top_down = active_top_down.astype(pref_clear.dtype)
    pref_lay = (1.0 - cloud_top_down) * pref_clear + cloud_top_down * pref_cloud
    prefd_lay = (1.0 - cloud_top_down) * prefd_clear + cloud_top_down * prefd_cloud
    ptra_lay = (1.0 - cloud_top_down) * ptra_clear + cloud_top_down * ptra_cloud
    ptrad_lay = (1.0 - cloud_top_down) * ptrad_clear + cloud_top_down * ptrad_cloud
    surface = jnp.broadcast_to(
        surface_albedo[..., None, None, None].astype(pref_lay.dtype),
        pref_lay.shape[:-3] + (1, pref_lay.shape[-2], pref_lay.shape[-1]),
    )
    zero_surface = jnp.zeros_like(surface)
    pref = jnp.concatenate((pref_lay, surface), axis=-3)
    prefd = jnp.concatenate((prefd_lay, surface), axis=-3)
    ptra = jnp.concatenate((ptra_lay, zero_surface), axis=-3)
    ptrad = jnp.concatenate((ptrad_lay, zero_surface), axis=-3)
    mu0 = jnp.maximum(coszen[..., None, None, None], 1.0e-6)
    direct_clear = _sw_transmittance_lookup((tau_clear_top_down / mu0).astype(pref_lay.dtype))
    direct_cloud = _sw_transmittance_lookup((tau_cloud_top_down / mu0).astype(pref_lay.dtype))
    direct_trans = ((1.0 - cloud_top_down) * direct_clear + cloud_top_down * direct_cloud) * active_top_down
    direct_trans = jnp.where(active_top_down > 0.0, direct_trans, 1.0).astype(pref_lay.dtype)
    down_top_down, up_top_down, direct_top_down = _vertical_quadrature(pref, prefd, ptra, ptrad, direct_trans)
    top_flux_band = (coszen[..., None, None] * source_scale[..., None, None] * sfluxzen).astype(down_top_down.dtype)
    down_band = jnp.flip(down_top_down * top_flux_band[..., None, :, :], axis=-3)
    up_band = jnp.flip(up_top_down * top_flux_band[..., None, :, :], axis=-3)
    direct_band = jnp.flip(direct_top_down * top_flux_band[..., None, :, :], axis=-3)
    # Reduce over BOTH the tile-band and g-point axes here, casting the (float32)
    # per-g-point flux to `partial_dtype` (fp64 under force_fp64) FIRST.  Because
    # each band tile contributes a DISJOINT set of bands, accumulating these fp64
    # per-tile fluxes across tiles makes the band reduction associative to fp64
    # precision: the result is provably independent of the chunk width
    # (bit-identical across tile sizes, see proofs/v013).  It differs from the
    # legacy single-pass band-sum (which reduced in float32) by <=~1e-5 rel — a
    # knob-independent, one-time precision *improvement*, not a chunking artifact.
    down_partial = jnp.sum(down_band.astype(partial_dtype), axis=(-1, -2))
    up_partial = jnp.sum(up_band.astype(partial_dtype), axis=(-1, -2))
    direct_partial = jnp.sum(direct_band.astype(partial_dtype), axis=(-1, -2))
    return down_partial, up_partial, direct_partial


def _sw_chunk_bands(n_bands: int) -> int:
    """Snaps the requested chunk width to a divisor of ``n_bands``.

    `lax.scan` needs uniform tile shapes, so the band axis must split evenly.
    The requested ``_SW_GPOINT_CHUNK_BANDS`` is clamped to [1, n_bands] and then
    rounded DOWN to the nearest divisor of ``n_bands`` (14: divisors 1,2,7,14),
    guaranteeing ``n_bands % chunk == 0``.  The numerical result is independent
    of the chosen width (proofs/v013); only the peak-VRAM / op-count trade moves.
    """

    requested = max(1, min(int(_SW_GPOINT_CHUNK_BANDS), n_bands))
    chunk = requested
    while n_bands % chunk != 0:
        chunk -= 1
    return chunk


def _sw_band_scan_fluxes(
    tau_clear,
    omega_clear,
    asymmetry_clear,
    tau_total_cloud,
    omega_total_cloud,
    asymmetry_total_cloud,
    cloud_amount,
    mask,
    sfluxzen,
    coszen,
    surface_albedo,
    source_scale,
    out_dtype,
    chunk: int,
):
    """Accumulates the band-tiled two-stream fluxes via a sequential ``lax.scan``.

    ``lax.scan`` runs the ``n_tiles`` tiles SEQUENTIALLY with a carry
    dependency, which is what forces XLA to free each tile's
    ``[..., nlev+1, chunk, gpoint]`` two-stream working set before the next tile
    — a Python ``for`` loop leaves the tiles mutually independent and lets XLA
    schedule them concurrently, inflating peak VRAM above even the single-pass
    solve.  Each tile is carved from the full-band optics with
    ``lax.dynamic_slice`` along the band axis (NO leading-axis reshape/transpose,
    which would transiently copy the whole ~(ncol,nlev,14,16) input).  The carry
    is the fp64 running flux ``(down, up, direct)``; each step adds the tile's
    (band, g-point)-summed fp64 flux.  Result is bit-identical across chunk
    widths (proofs/v013).
    """

    n_bands = tau_clear.shape[-2]
    n_tiles = n_bands // chunk
    band_axis = tau_clear.ndim - 2  # axis of size n_bands in [..., nlev, band, gpoint]
    sflux_band_axis = sfluxzen.ndim - 2

    def _band_slice(arr, start, axis):
        starts = [0] * arr.ndim
        starts[axis] = start
        sizes = list(arr.shape)
        sizes[axis] = chunk
        return lax.dynamic_slice(arr, starts, sizes)

    flux_shape = tau_clear.shape[:-3] + (tau_clear.shape[-3] + 1,)
    init = (
        jnp.zeros(flux_shape, dtype=out_dtype),
        jnp.zeros(flux_shape, dtype=out_dtype),
        jnp.zeros(flux_shape, dtype=out_dtype),
    )

    def body(carry, tile_index):
        d_acc, u_acc, dir_acc = carry
        start = tile_index * chunk
        d_p, u_p, dir_p = _sw_band_tile_fluxes(
            _band_slice(tau_clear, start, band_axis),
            _band_slice(omega_clear, start, band_axis),
            _band_slice(asymmetry_clear, start, band_axis),
            _band_slice(tau_total_cloud, start, band_axis),
            _band_slice(omega_total_cloud, start, band_axis),
            _band_slice(asymmetry_total_cloud, start, band_axis),
            _band_slice(cloud_amount, start, band_axis),
            _band_slice(mask, start, 0),
            _band_slice(sfluxzen, start, sflux_band_axis),
            coszen,
            surface_albedo,
            source_scale,
            out_dtype,
        )
        return (d_acc + d_p, u_acc + u_p, dir_acc + dir_p), None

    (flux_down_model, flux_up_model, direct_down_model), _ = lax.scan(
        body, init, jnp.arange(n_tiles)
    )
    return flux_down_model, flux_up_model, direct_down_model


def _shortwave_impl(
    state: RRTMGSWColumnState,
    tables: RRTMGTableBundle,
    debug: bool,
    topography: RRTMGSWTopographyState | None = None,
) -> RRTMGSWColumnResult:
    """Unjitted SW implementation shared by production and stripped paths."""

    state = _clip_state(state)
    # Precision boundary (ADR-007 / Phase-B coupler_interface §5): WRF RRTMG cloud
    # optics + the reftra/vrtqdr two-stream are single precision in
    # `module_ra_rrtmg_sw.F`, and we keep that internally for WRF fidelity
    # (cloud_dtype below).  But the EXPORTED diagnostics (SWDOWN/SWUP, heating
    # rate) must not silently downcast below the state's precision regime — under
    # force_fp64 every emitted field is fp64.  Derive the export dtype from the
    # fp64-locked pressure input and cast result fields at the kernel boundary.
    out_dtype = jnp.result_type(state.p.dtype, jnp.float32)
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
    cloud_dtype = jnp.float32
    liquid_path_g = (qc_ext * layer_mass_ext * 1000.0).astype(cloud_dtype)
    ice_path_g = (qi_ext * layer_mass_ext * 1000.0).astype(cloud_dtype)
    snow_path_g = (0.99 * qs_ext * layer_mass_ext * 1000.0).astype(cloud_dtype)

    coef = _sw_setcoef(qv_ext, p_ext, _extend_with_wrf_top_layer(state.T), pressure_interfaces, tables)
    tau_gas, tau_rayleigh = _sw_taumol_fused(coef, tables)
    sfluxzen = _sw_sfluxzen(coef, tables)
    liquid_coeff = tables.sw_cloud_liquid_extinction.astype(cloud_dtype)[:, None]
    ice_coeff = tables.sw_cloud_ice_extinction.astype(cloud_dtype)[:, None]
    snow_coeff = tables.sw_cloud_snow_extinction.astype(cloud_dtype)[:, None]
    liquid_ssa = tables.sw_cloud_liquid_ssa.astype(cloud_dtype)[:, None]
    ice_ssa = tables.sw_cloud_ice_ssa.astype(cloud_dtype)[:, None]
    snow_ssa = tables.sw_cloud_snow_ssa.astype(cloud_dtype)[:, None]
    liquid_asy = tables.sw_cloud_liquid_asymmetry.astype(cloud_dtype)[:, None]
    ice_asy = tables.sw_cloud_ice_asymmetry.astype(cloud_dtype)[:, None]
    snow_asy = tables.sw_cloud_snow_asymmetry.astype(cloud_dtype)[:, None]
    liquid_forward = liquid_asy * liquid_asy
    ice_forward = tables.sw_cloud_ice_forward_fraction.astype(cloud_dtype)[:, None]
    snow_forward = tables.sw_cloud_snow_forward_fraction.astype(cloud_dtype)[:, None]
    mask = tables.sw_gpoint_mask.astype(cloud_dtype)

    cloud_box = cloud_ext.astype(cloud_dtype)[..., None, None]
    cloud_amount = _mcica_random_overlap_mask(p_ext, cloud_ext, mask).astype(cloud_dtype)
    cloud_safe = jnp.maximum(cloud_box, 0.01)
    cloud_present = cloud_box > 0.0
    liquid_incloud = jnp.where(cloud_present, liquid_path_g[..., None, None] / cloud_safe, 0.0)
    ice_incloud = jnp.where(cloud_present, ice_path_g[..., None, None] / cloud_safe, 0.0)
    snow_incloud = jnp.where(cloud_present, snow_path_g[..., None, None] / cloud_safe, 0.0)

    tau_clear_orig = tau_gas + tau_rayleigh
    omega_clear_orig = jnp.clip(tau_rayleigh / jnp.maximum(tau_clear_orig, MIN_OPTICAL_DEPTH), 0.0, 0.999999)
    asymmetry_clear_orig = jnp.zeros_like(tau_clear_orig)
    tau_clear, omega_clear, asymmetry_clear = _delta_scale(tau_clear_orig, omega_clear_orig, asymmetry_clear_orig)

    tau_liquid_orig = liquid_incloud * liquid_coeff * cloud_amount
    tau_ice_orig = ice_incloud * ice_coeff * cloud_amount
    tau_snow_orig = snow_incloud * snow_coeff * cloud_amount

    def scale_cloud_component(tau_orig, omega_orig, asym_orig, forward_fraction):
        denom = jnp.maximum(1.0 - forward_fraction * omega_orig, 1.0e-12)
        tau_scaled = denom * tau_orig
        omega_scaled = jnp.clip(omega_orig * (1.0 - forward_fraction) / denom, 0.0, 0.999999)
        asym_scaled = jnp.clip((asym_orig - forward_fraction) / jnp.maximum(1.0 - forward_fraction, 1.0e-12), -0.999999, 0.999999)
        scattering = tau_scaled * omega_scaled
        return tau_scaled, scattering, asym_scaled

    tau_liquid, scat_liquid, asym_liquid = scale_cloud_component(tau_liquid_orig, liquid_ssa, liquid_asy, liquid_forward)
    tau_ice, scat_ice, asym_ice = scale_cloud_component(tau_ice_orig, ice_ssa, ice_asy, ice_forward)
    tau_snow, scat_snow, asym_snow = scale_cloud_component(tau_snow_orig, snow_ssa, snow_asy, snow_forward)
    tau_cloud = tau_liquid + tau_ice + tau_snow
    scattering_cloud = scat_liquid + scat_ice + scat_snow
    omega_cloud = jnp.clip(scattering_cloud / jnp.maximum(tau_cloud, MIN_OPTICAL_DEPTH), 0.0, 0.999999)
    omega_cloud = jnp.where(cloud_amount > 0.0, omega_cloud, 1.0)
    asymmetry_cloud = jnp.where(
        scattering_cloud > MIN_OPTICAL_DEPTH,
        (scat_liquid * asym_liquid + scat_ice * asym_ice + scat_snow * asym_snow) / jnp.maximum(scattering_cloud, MIN_OPTICAL_DEPTH),
        0.0,
    )

    scattering_total_cloud = tau_clear * omega_clear + tau_cloud * omega_cloud
    tau_total_cloud = jnp.maximum(tau_clear + tau_cloud, MIN_OPTICAL_DEPTH)
    omega_total_cloud = jnp.clip(scattering_total_cloud / jnp.maximum(tau_total_cloud, MIN_OPTICAL_DEPTH), 0.0, 0.999999)
    asymmetry_total_cloud = jnp.where(
        scattering_total_cloud > MIN_OPTICAL_DEPTH,
        (tau_clear * omega_clear * asymmetry_clear + tau_cloud * omega_cloud * asymmetry_cloud) / jnp.maximum(scattering_total_cloud, MIN_OPTICAL_DEPTH),
        0.0,
    )

    tau_clear = jnp.maximum(tau_clear, MIN_OPTICAL_DEPTH) * mask
    omega_clear = omega_clear * mask
    asymmetry_clear = asymmetry_clear * mask
    tau_total_cloud = tau_total_cloud * mask
    omega_total_cloud = omega_total_cloud * mask
    asymmetry_total_cloud = asymmetry_total_cloud * mask

    # Two-stream + vertical quadrature + g-point reduction, TILED over the band
    # axis (axis=-2 of the optics).  Bands never couple in `_reftra_eddington`
    # or `_vertical_quadrature`, so each tile reproduces exactly the per-(band,
    # g-point) flux of the single-pass solve; the heavy
    # `[..., nlev, tile_bands, gpoint]` working set is freed before the next
    # tile.  Each tile returns g-point-summed band partials
    # `[..., nlev+1, tile_bands]`; concatenating the (small) partials and
    # summing the band axis once reproduces the original `sum(axis=(-1, -2))`.
    # The cast to `out_dtype` (fp64 under force_fp64) happens at the per-tile
    # partial so the band-summed fluxes carry export precision exactly, keeping
    # heating_rate / flux_down/up / column_absorbed mutually consistent (the
    # energy-closure invariant flux-divergence == heating*mass*cp).
    #
    # `source_scale` MUST be built in the working two-stream dtype (`cloud_dtype`
    # == float32), exactly as the original single-pass code did with
    # `_solar_source_scale(state, down_top_down.dtype)`: the TOA flux
    # `coszen * source_scale * sfluxzen` is then multiplied in float32, so using
    # the fp64 export dtype here would silently change the rounding of every band
    # flux (~7e-5 rel) — a real numerical drift, not VRAM chunking.
    source_scale = _solar_source_scale(state, cloud_dtype)
    n_bands = tau_clear.shape[-2]
    chunk = _sw_chunk_bands(n_bands)
    flux_down_model, flux_up_model, direct_down_model = _sw_band_scan_fluxes(
        tau_clear,
        omega_clear,
        asymmetry_clear,
        tau_total_cloud,
        omega_total_cloud,
        asymmetry_total_cloud,
        cloud_amount,
        mask,
        sfluxzen,
        state.coszen,
        state.surface_albedo,
        source_scale,
        out_dtype,
        chunk,
    )
    net_down = flux_down_model - flux_up_model
    column_absorbed_layers = net_down[..., 1 : original_layers + 1] - net_down[..., :original_layers]
    column_absorbed_total = net_down[..., -1] - net_down[..., 0]
    heating_rate = column_absorbed_layers / (layer_mass.astype(out_dtype) * CP_AIR)
    surface_absorbed = flux_down_model[..., 0] - flux_up_model[..., 0]
    flux_down = flux_down_model
    flux_up = flux_up_model
    surface_down = flux_down[..., 0]
    surface_up = flux_up[..., 0]
    surface_direct = direct_down_model[..., 0]
    surface_diffuse = surface_down - surface_direct
    surface_diffuse_fraction = jnp.where(surface_down > 0.001, jnp.minimum(surface_diffuse / surface_down, 1.0), 0.0).astype(out_dtype)
    if topography is None:
        topographic_correction_factor = jnp.ones_like(surface_down)
        surface_down_topographic = surface_down
        surface_up_topographic = surface_up
        surface_absorbed_topographic = surface_absorbed
    else:
        topographic = apply_wrf_topographic_sw_adjustment(
            surface_down=surface_down,
            surface_up=surface_up,
            surface_absorbed=surface_absorbed,
            coszen=state.coszen.astype(out_dtype),
            diffuse_fraction=surface_diffuse_fraction,
            topography=topography,
        )
        topographic_correction_factor = topographic.correction_factor.astype(out_dtype)
        surface_down_topographic = topographic.surface_down.astype(out_dtype)
        surface_up_topographic = topographic.surface_up.astype(out_dtype)
        surface_absorbed_topographic = topographic.surface_absorbed.astype(out_dtype)

    heating_rate = assert_finite(heating_rate, "rrtmg_sw.heating_rate", enabled=debug)
    flux_down = assert_physical_bounds(flux_down, 0.0, 2000.0, "rrtmg_sw.flux_down", enabled=debug)
    flux_up = assert_physical_bounds(flux_up, 0.0, 2000.0, "rrtmg_sw.flux_up", enabled=debug)
    return RRTMGSWColumnResult(
        heating_rate=heating_rate,
        flux_down=flux_down,
        flux_up=flux_up,
        toa_down=flux_down[..., -1],
        toa_up=flux_up[..., -1],
        surface_down=surface_down,
        surface_up=surface_up,
        column_absorbed=column_absorbed_total,
        surface_absorbed=surface_absorbed,
        surface_direct=surface_direct,
        surface_diffuse=surface_diffuse,
        surface_diffuse_fraction=surface_diffuse_fraction,
        topographic_correction_factor=topographic_correction_factor,
        surface_down_topographic=surface_down_topographic,
        surface_up_topographic=surface_up_topographic,
        surface_absorbed_topographic=surface_absorbed_topographic,
    )


def compute_rrtmg_sw_intermediates(
    state: RRTMGSWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
) -> RRTMGSWIntermediateState:
    """Returns JAX SW state compared to WRF `setcoef_sw`/`taumol_sw`/`spcvmc_sw` oracles."""

    state = _clip_state(state)
    original_interfaces = _pressure_interfaces(state.p)
    top_pressure = 0.5 * original_interfaces[..., -1:]
    pressure_interfaces = jnp.concatenate((original_interfaces, jnp.full_like(top_pressure, 1.0e-3)), axis=-1)
    p_ext = jnp.concatenate((state.p, top_pressure), axis=-1)
    qv_ext = _extend_with_wrf_top_layer(state.qv)
    t_ext = _extend_with_wrf_top_layer(state.T)
    qc_ext = jnp.concatenate((state.qc, jnp.zeros_like(state.qc[..., -1:])), axis=-1)
    qi_ext = jnp.concatenate((state.qi, jnp.zeros_like(state.qi[..., -1:])), axis=-1)
    qs_ext = jnp.concatenate((state.qs, jnp.zeros_like(state.qs[..., -1:])), axis=-1)
    cloud_ext = jnp.concatenate((state.cloud_fraction, jnp.zeros_like(state.cloud_fraction[..., -1:])), axis=-1)
    layer_mass_ext = jnp.maximum((pressure_interfaces[..., :-1] - pressure_interfaces[..., 1:]) / GRAVITY, MIN_LAYER_MASS)
    cloud_dtype = jnp.float32
    liquid_path_g = (qc_ext * layer_mass_ext * 1000.0).astype(cloud_dtype)
    ice_path_g = (qi_ext * layer_mass_ext * 1000.0).astype(cloud_dtype)
    snow_path_g = (0.99 * qs_ext * layer_mass_ext * 1000.0).astype(cloud_dtype)
    coef = _sw_setcoef(qv_ext, p_ext, t_ext, pressure_interfaces, tables)
    tau_gas, tau_rayleigh = _sw_taumol_fused(coef, tables)
    sfluxzen = _sw_sfluxzen(coef, tables)

    liquid_coeff = tables.sw_cloud_liquid_extinction.astype(cloud_dtype)[:, None]
    ice_coeff = tables.sw_cloud_ice_extinction.astype(cloud_dtype)[:, None]
    snow_coeff = tables.sw_cloud_snow_extinction.astype(cloud_dtype)[:, None]
    liquid_ssa = tables.sw_cloud_liquid_ssa.astype(cloud_dtype)[:, None]
    ice_ssa = tables.sw_cloud_ice_ssa.astype(cloud_dtype)[:, None]
    snow_ssa = tables.sw_cloud_snow_ssa.astype(cloud_dtype)[:, None]
    liquid_asy = tables.sw_cloud_liquid_asymmetry.astype(cloud_dtype)[:, None]
    ice_asy = tables.sw_cloud_ice_asymmetry.astype(cloud_dtype)[:, None]
    snow_asy = tables.sw_cloud_snow_asymmetry.astype(cloud_dtype)[:, None]
    liquid_forward = liquid_asy * liquid_asy
    ice_forward = tables.sw_cloud_ice_forward_fraction.astype(cloud_dtype)[:, None]
    snow_forward = tables.sw_cloud_snow_forward_fraction.astype(cloud_dtype)[:, None]
    mask = tables.sw_gpoint_mask.astype(cloud_dtype)

    cloud_box = cloud_ext.astype(cloud_dtype)[..., None, None]
    cloud_amount = _mcica_random_overlap_mask(p_ext, cloud_ext, mask).astype(cloud_dtype)
    cloud_safe = jnp.maximum(cloud_box, 0.01)
    cloud_present = cloud_box > 0.0
    liquid_incloud = jnp.where(cloud_present, liquid_path_g[..., None, None] / cloud_safe, 0.0)
    ice_incloud = jnp.where(cloud_present, ice_path_g[..., None, None] / cloud_safe, 0.0)
    snow_incloud = jnp.where(cloud_present, snow_path_g[..., None, None] / cloud_safe, 0.0)

    tau_clear_orig = tau_gas + tau_rayleigh
    omega_clear_orig = jnp.clip(tau_rayleigh / jnp.maximum(tau_clear_orig, MIN_OPTICAL_DEPTH), 0.0, 0.999999)
    asymmetry_clear_orig = jnp.zeros_like(tau_clear_orig)
    tau_clear, omega_clear, asymmetry_clear = _delta_scale(tau_clear_orig, omega_clear_orig, asymmetry_clear_orig)

    tau_liquid_orig = liquid_incloud * liquid_coeff * cloud_amount
    tau_ice_orig = ice_incloud * ice_coeff * cloud_amount
    tau_snow_orig = snow_incloud * snow_coeff * cloud_amount
    ptaormc = tau_liquid_orig + tau_ice_orig + tau_snow_orig

    def scale_cloud_component(tau_orig, omega_orig, asym_orig, forward_fraction):
        denom = jnp.maximum(1.0 - forward_fraction * omega_orig, 1.0e-12)
        tau_scaled = denom * tau_orig
        omega_scaled = jnp.clip(omega_orig * (1.0 - forward_fraction) / denom, 0.0, 0.999999)
        asym_scaled = jnp.clip((asym_orig - forward_fraction) / jnp.maximum(1.0 - forward_fraction, 1.0e-12), -0.999999, 0.999999)
        scattering = tau_scaled * omega_scaled
        return tau_scaled, scattering, asym_scaled

    tau_liquid, scat_liquid, asym_liquid = scale_cloud_component(tau_liquid_orig, liquid_ssa, liquid_asy, liquid_forward)
    tau_ice, scat_ice, asym_ice = scale_cloud_component(tau_ice_orig, ice_ssa, ice_asy, ice_forward)
    tau_snow, scat_snow, asym_snow = scale_cloud_component(tau_snow_orig, snow_ssa, snow_asy, snow_forward)
    tau_cloud = tau_liquid + tau_ice + tau_snow
    scattering_cloud = scat_liquid + scat_ice + scat_snow
    omega_cloud = jnp.clip(scattering_cloud / jnp.maximum(tau_cloud, MIN_OPTICAL_DEPTH), 0.0, 0.999999)
    omega_cloud = jnp.where(cloud_amount > 0.0, omega_cloud, 1.0)
    asymmetry_cloud = jnp.where(
        scattering_cloud > MIN_OPTICAL_DEPTH,
        (scat_liquid * asym_liquid + scat_ice * asym_ice + scat_snow * asym_snow) / jnp.maximum(scattering_cloud, MIN_OPTICAL_DEPTH),
        0.0,
    )

    scattering_total_cloud = tau_clear * omega_clear + tau_cloud * omega_cloud
    tau_total_cloud = jnp.maximum(tau_clear + tau_cloud, MIN_OPTICAL_DEPTH)
    omega_total_cloud = jnp.clip(scattering_total_cloud / jnp.maximum(tau_total_cloud, MIN_OPTICAL_DEPTH), 0.0, 0.999999)
    asymmetry_total_cloud = jnp.where(
        scattering_total_cloud > MIN_OPTICAL_DEPTH,
        (tau_clear * omega_clear * asymmetry_clear + tau_cloud * omega_cloud * asymmetry_cloud) / jnp.maximum(scattering_total_cloud, MIN_OPTICAL_DEPTH),
        0.0,
    )

    tau_clear = jnp.maximum(tau_clear, MIN_OPTICAL_DEPTH) * mask
    omega_clear = omega_clear * mask
    asymmetry_clear = asymmetry_clear * mask
    tau_total_cloud = tau_total_cloud * mask
    omega_total_cloud = omega_total_cloud * mask
    asymmetry_total_cloud = asymmetry_total_cloud * mask

    tau_clear_top_down = jnp.flip(tau_clear, axis=-3)
    omega_clear_top_down = jnp.flip(omega_clear, axis=-3)
    asymmetry_clear_top_down = jnp.flip(asymmetry_clear, axis=-3)
    tau_cloud_top_down = jnp.flip(tau_total_cloud, axis=-3)
    omega_cloud_top_down = jnp.flip(omega_total_cloud, axis=-3)
    asymmetry_cloud_top_down = jnp.flip(asymmetry_total_cloud, axis=-3)
    cloud_top_down = jnp.flip(jnp.clip(cloud_amount, 0.0, 1.0), axis=-3)
    active_top_down = jnp.flip(mask + jnp.zeros_like(tau_clear), axis=-3)
    cloud_active_top_down = active_top_down * (cloud_top_down > 1.0e-12)
    pref_clear, prefd_clear, ptra_clear, ptrad_clear = _reftra_eddington(
        tau_clear_top_down, omega_clear_top_down, asymmetry_clear_top_down, state.coszen, active_top_down
    )
    pref_cloud, prefd_cloud, ptra_cloud, ptrad_cloud = _reftra_eddington(
        tau_cloud_top_down, omega_cloud_top_down, asymmetry_cloud_top_down, state.coszen, cloud_active_top_down
    )
    cloud_top_down = cloud_top_down.astype(pref_clear.dtype)
    active_top_down = active_top_down.astype(pref_clear.dtype)
    pref_lay = (1.0 - cloud_top_down) * pref_clear + cloud_top_down * pref_cloud
    prefd_lay = (1.0 - cloud_top_down) * prefd_clear + cloud_top_down * prefd_cloud
    ptra_lay = (1.0 - cloud_top_down) * ptra_clear + cloud_top_down * ptra_cloud
    ptrad_lay = (1.0 - cloud_top_down) * ptrad_clear + cloud_top_down * ptrad_cloud
    surface = jnp.broadcast_to(
        state.surface_albedo[..., None, None, None].astype(pref_lay.dtype),
        pref_lay.shape[:-3] + (1, pref_lay.shape[-2], pref_lay.shape[-1]),
    )
    zero_surface = jnp.zeros_like(surface)
    pref = jnp.concatenate((pref_lay, surface), axis=-3)
    prefd = jnp.concatenate((prefd_lay, surface), axis=-3)
    ptra = jnp.concatenate((ptra_lay, zero_surface), axis=-3)
    ptrad = jnp.concatenate((ptrad_lay, zero_surface), axis=-3)
    pref_clear_full = jnp.concatenate((pref_clear, surface), axis=-3)
    prefd_clear_full = jnp.concatenate((prefd_clear, surface), axis=-3)
    ptra_clear_full = jnp.concatenate((ptra_clear, zero_surface), axis=-3)
    ptrad_clear_full = jnp.concatenate((ptrad_clear, zero_surface), axis=-3)
    pref_cloud_full = jnp.concatenate((pref_cloud, surface), axis=-3)
    prefd_cloud_full = jnp.concatenate((prefd_cloud, surface), axis=-3)
    ptra_cloud_full = jnp.concatenate((ptra_cloud, zero_surface), axis=-3)
    ptrad_cloud_full = jnp.concatenate((ptrad_cloud, zero_surface), axis=-3)
    mu0 = jnp.maximum(state.coszen[..., None, None, None], 1.0e-6)
    direct_clear = _sw_transmittance_lookup((tau_clear_top_down / mu0).astype(pref_lay.dtype))
    direct_cloud = _sw_transmittance_lookup((tau_cloud_top_down / mu0).astype(pref_lay.dtype))
    direct_trans = ((1.0 - cloud_top_down) * direct_clear + cloud_top_down * direct_cloud) * active_top_down
    direct_trans = jnp.where(active_top_down > 0.0, direct_trans, 1.0).astype(pref_lay.dtype)
    down_top_down, up_top_down, _direct_top_down = _vertical_quadrature(pref, prefd, ptra, ptrad, direct_trans)
    source_scale = _solar_source_scale(state, down_top_down.dtype)
    top_flux_band = (state.coszen[..., None, None] * source_scale[..., None, None] * sfluxzen).astype(down_top_down.dtype)
    down_band = jnp.flip(down_top_down * top_flux_band[..., None, :, :], axis=-3)
    up_band = jnp.flip(up_top_down * top_flux_band[..., None, :, :], axis=-3)

    return RRTMGSWIntermediateState(
        jp=coef.jp,
        jt=coef.jt,
        jt1=coef.jt1,
        fac00=coef.fac00,
        fac01=coef.fac01,
        fac10=coef.fac10,
        fac11=coef.fac11,
        indself=coef.indself,
        indfor=coef.indfor,
        selffac=coef.selffac,
        forfac=coef.forfac,
        colmol=jnp.stack((coef.colh2o, coef.colco2, coef.colo3, coef.colch4, coef.coln2o, coef.colo2), axis=-1),
        taug=tau_gas,
        taur=tau_rayleigh,
        sfluxzen=sfluxzen,
        pcldfmc=cloud_amount,
        ptaucmc=tau_cloud,
        pasycmc=asymmetry_cloud,
        pomgcmc=omega_cloud,
        ptaormc=ptaormc,
        spcvmc_zref=pref,
        spcvmc_ztra=ptra,
        spcvmc_zrefd=prefd,
        spcvmc_ztrad=ptrad,
        spcvmc_zref_clear=pref_clear_full,
        spcvmc_ztra_clear=ptra_clear_full,
        spcvmc_zrefd_clear=prefd_clear_full,
        spcvmc_ztrad_clear=ptrad_clear_full,
        spcvmc_zref_cloud=pref_cloud_full,
        spcvmc_ztra_cloud=ptra_cloud_full,
        spcvmc_zrefd_cloud=prefd_cloud_full,
        spcvmc_ztrad_cloud=ptrad_cloud_full,
        spcvmc_direct_trans=direct_trans,
        spcvmc_zfd_flux=down_band,
        spcvmc_zfu_flux=up_band,
    )


@partial(jax.jit, static_argnames=("debug",))
def solve_rrtmg_sw_column(
    state: RRTMGSWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
    *,
    debug: bool = False,
    topography: RRTMGSWTopographyState | None = None,
) -> RRTMGSWColumnResult:
    """Computes one fused shortwave column radiation call."""

    return _shortwave_impl(state, tables, debug, topography)


@jax.jit
def solve_rrtmg_sw_column_debug_stripped(
    state: RRTMGSWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
) -> RRTMGSWColumnResult:
    """Hand-stripped sibling used for the HLO debug identity proof."""

    return _shortwave_impl(state, tables, False)
