"""WRF mp8 effective radii and the has_reqc/i/s RRTMG mediation contract.

Radii are held mass-grid diagnostics in meters.  Radiation converts them to
microns and applies the *caller* fallbacks before cldprmc, not inside Thompson.
"""
from __future__ import annotations

from functools import lru_cache
from collections.abc import Mapping
import os
import re
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_constants import (
    AM_I, AM_R, AM_S, NT_C, NT_C_MAX, R1, R2, R_D, WRF_REAL_CONSTANTS,
)
from gpuwrf.physics.thompson_tables import THOMPSON_TABLES

RE_CLOUD_BG = 2.49e-6
RE_ICE_BG = 4.99e-6
RE_SNOW_BG = 9.99e-6


_MP_RE = os.environ.get("GPUWRF_RRTMG_MP_RE", "0").lower() in {"1", "true", "yes", "on"}


def mp_re_enabled() -> bool:
    return _MP_RE


def mp_re_config(namelist=None) -> dict[str, int]:
    """Persist the static physics selection needed to resume the radius carry."""
    defaults = dict(use_mp_re=1, mp_physics=8, ra_lw_physics=4, ra_sw_physics=4)
    if isinstance(namelist, Mapping):
        selection = namelist.get("physics", namelist)
        return {name: int(selection.get(name, default)) for name, default in defaults.items()}
    return {name: int(getattr(namelist, name, default)) for name, default in defaults.items()}


def mp_re_active(namelist=None, **selection) -> bool:
    """WRF physics_init: supported mp8 radii require the RRTMG LW/SW pair.

    The environment flag is resolved at import (E80). The four namelist
    controls are static. Use this predicate for seeding, handoff and restart.
    """
    config = mp_re_config(namelist)
    config.update(selection)
    return (_MP_RE and config["use_mp_re"] == 1 and config["mp_physics"] == 8
            and config["ra_lw_physics"] == 4 and config["ra_sw_physics"] == 4)


class EffectiveRadii(NamedTuple):
    cloud: jax.Array
    ice: jax.Array
    snow: jax.Array


def calc_thompson_effective_radii(T, p, qv, qc, qi, Ni, qs, *, Nc=None, aerosol_aware=False):
    """calc_effectRad + mp_gt_driver clamps (Thompson.F:5594–5699,1475–1477).

    Non-aerosol mp8 ignores nc1d and uses Nt_c.  The column has_q* tests only
    skip inactive levels; the calculation at each active level is independent.
    lamc/lami are WRF DOUBLE locals.  Their RHS is REAL, then the radius divide
    is DOUBLE and SNGL returns REAL, just as in the pristine expression.
    """
    T, p, qv, qc, qi, Ni, qs = (jnp.asarray(v, jnp.float32) for v in (T, p, qv, qc, qi, Ni, qs))
    f = lambda v: jnp.asarray(v, jnp.float32)
    rho = f(.622) * p / (f(R_D) * T * (qv + f(.622)))
    rc = jnp.maximum(f(R1), qc * rho)
    ri = jnp.maximum(f(R1), qi * rho)
    ni = jnp.maximum(f(R2), Ni * rho)
    rs = jnp.maximum(f(R1), qs * rho)
    if aerosol_aware:
        if Nc is None:
            raise ValueError("aerosol-aware calc_effectRad requires cloud number Nc")
        nc = jnp.maximum(f(2), jnp.minimum(jnp.asarray(Nc, jnp.float32) * rho, f(NT_C_MAX)))
    else:
        nc = jnp.full_like(rc, f(NT_C))
    inu = jnp.where(nc < f(100), 15, jnp.where(nc > f(1e10), 2, jnp.minimum(15, jnp.floor(f(1000e6) / nc + f(.5)).astype(jnp.int32) + 2)))
    g_ratio = jnp.asarray([24,60,120,210,336,504,720,990,1320,1716,2184,2730,3360,4080,4896], jnp.float32)
    # GPUWRF_THOMPSON_WRF_CONSTANTS (BD94): WRF's REAL-folded am_r / am_i PARAMETERs (module_mp_thompson.F:128/:137);
    # g_ratio is WRF's integer literal table and cig(2)*oig1 = WGAMMA(4.)/WGAMMA(1.) = 6 exactly.
    wrf_k = os.environ.get("GPUWRF_THOMPSON_WRF_CONSTANTS", "0") == "1"
    am_r, am_i = (WRF_REAL_CONSTANTS["AM_R"], WRF_REAL_CONSTANTS["AM_I"]) if wrf_k else (AM_R, AM_I)
    lamc_real = (nc * f(am_r) * g_ratio[inu - 1] / rc) ** f(1 / 3)
    lami_real = (f(am_i) * f(6) * f(1) * ni / ri) ** f(1 / 3)
    with jax.enable_x64(True):
        rec = (jnp.float64(.5) * (f(3) + inu.astype(jnp.float32)).astype(jnp.float64) / lamc_real.astype(jnp.float64)).astype(jnp.float32)
        rei = (jnp.float64(.5) * jnp.float64(3) / lami_real.astype(jnp.float64)).astype(jnp.float32)
    rec = jnp.where((rc > f(R1)) & (nc > f(R2)), jnp.maximum(f(2.51e-6), jnp.minimum(rec, f(50e-6))), f(RE_CLOUD_BG))
    rei = jnp.where((ri > f(R1)) & (ni > f(R2)), jnp.maximum(f(2.51e-6), jnp.minimum(rei, f(125e-6))), f(RE_ICE_BG))
    # mp8 bm_s=2: reference snow moment is rs*oams; cse(1)=3.
    tc = jnp.minimum(f(-.1), T - f(273.15))
    smob = rs * f(1 / AM_S)
    sa = THOMPSON_TABLES.snow_sa.astype(jnp.float32)
    sb = THOMPSON_TABLES.snow_sb.astype(jnp.float32)
    order = f(3)
    def polynomial(c):
        return (c[0] + c[1]*tc + c[2]*order + c[3]*tc*order + c[4]*tc*tc + c[5]*order*order
                + c[6]*tc*tc*order + c[7]*tc*order*order + c[8]*tc*tc*tc + c[9]*order*order*order)
    smoc = f(10) ** polynomial(sa) * smob ** polynomial(sb)
    res = jnp.where(rs > f(R1), jnp.maximum(f(5.01e-6), jnp.minimum(f(.5) * (smoc / smob), f(999e-6))), f(RE_SNOW_BG))
    # The driver repeats the background/min/max bounds on writeback.
    return EffectiveRadii(jnp.maximum(f(RE_CLOUD_BG), jnp.minimum(rec, f(50e-6))),
                          jnp.maximum(f(RE_ICE_BG), jnp.minimum(rei, f(125e-6))),
                          jnp.maximum(f(RE_SNOW_BG), jnp.minimum(res, f(999e-6))))


@lru_cache(maxsize=1)
def _extractor():
    from gpuwrf.physics.rrtmg_lw import _extract_rrtmg_tables_module
    return _extract_rrtmg_tables_module()


@lru_cache(maxsize=1)
def _retab():
    text = _extractor().LW_SOURCE.read_text()
    match = re.search(r"data\s+retab\s*/(.*?)/", text, re.I | re.S)
    if match is None:
        raise ValueError("missing pristine RRTMG retab")
    body = re.sub(r"!.*", "", match[1]).replace("_rb", "")
    values = np.asarray([float(v.replace("D","E").replace("d","e")) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?", body)], np.float32)
    if values.size != 95:
        raise ValueError(f"RRTMG retab expected95 entries, got{values.size}")
    return values


class RadiationRadii(NamedTuple):
    liquid_um: jax.Array
    ice_um: jax.Array
    snow_um: jax.Array
    snow_mass_factor: jax.Array


def prepare_radiation_radii(T, cf, xland, re_cloud, re_ice, re_snow):
    """WRF RRTMG caller: radius fallbacks + snow>130µm mass reduction.

    The temperature-dependent ice fallback is precisely the driver's integer
    index/corr expression.  No iceflag3 factor1.0315 applies to has_reqi=1.
    """
    T, cf, rc, ri, rs = (jnp.asarray(v, jnp.float32) for v in (T, cf, re_cloud, re_ice, re_snow))
    land = jnp.asarray(xland, jnp.float32)
    if land.ndim == rc.ndim - 1:
        land = land[..., None]
    liq = jnp.maximum(jnp.float32(2.5), rc * jnp.float32(1e6))
    fallback = (liq <= jnp.float32(2.5)) & (cf > 0)
    liq = jnp.where(fallback & (land > jnp.float32(1.5)), jnp.float32(10.5), liq)
    liq = jnp.where(fallback & (land < jnp.float32(1.5)), jnp.float32(7.5), liq)
    ice = jnp.maximum(jnp.float32(5), ri * jnp.float32(1e6))
    idx = jnp.clip((T - jnp.float32(179)).astype(jnp.int32), 1, 75) - 1
    corr = T - T.astype(jnp.int32).astype(jnp.float32)
    retab = jnp.asarray(_retab())
    bg_ice = jnp.maximum(jnp.float32(5), retab[idx] * (jnp.float32(1) - corr) + retab[idx + 1] * corr)
    ice = jnp.where((ice <= jnp.float32(5)) & (cf > 0), bg_ice, ice)
    snow = jnp.maximum(jnp.float32(10), rs * jnp.float32(1e6))
    factor = jnp.where(snow > jnp.float32(130), jnp.minimum(jnp.float32(.99), (jnp.float32(130)/snow)**jnp.float32(2)), jnp.float32(.99))
    return RadiationRadii(liq, ice, jnp.minimum(snow, jnp.float32(130)), factor)


@lru_cache(maxsize=2)
def raw_cloud_tables(kind):
    """Original radius-grid tables, cached at trace/init; never read per step."""
    ex = _extractor()
    source = ex.LW_SOURCE if kind == "lw" else ex.SW_SOURCE
    bands = range(1,17) if kind == "lw" else range(16,30)
    names = ("absliq1","absice3") if kind == "lw" else ("extliq1","ssaliq1","asyliq1","extice3","ssaice3","asyice3","fdlice3")
    return {n:np.stack([ex._parse_source_array(source,n,b) for b in bands],axis=0) for n in names}


def interpolate_radius(table, radius_um, *, ice=False, dtype=None):
    """cldprmc uniform-grid interpolation incl. WRF's endpoint extrapolation."""
    dtype = radius_um.dtype if dtype is None else dtype
    t = jnp.asarray(table,dtype)
    r = radius_um.astype(dtype)
    start, step = (5,3) if ice else (2.5,1)
    idx = jnp.clip(jnp.floor((r-start)/step).astype(jnp.int32),0,t.shape[-1]-2)
    frac = (r - (start + idx.astype(dtype)*step))/step
    # Output (..., layer, band), with one gather at each layer/band.
    band = jnp.arange(t.shape[0])
    return t[band,idx[...,None]] + frac[...,None]*(t[band,idx[...,None]+1]-t[band,idx[...,None]])


def lw_radius_coefficients(radii, dtype):
    tables = raw_cloud_tables("lw")
    return (interpolate_radius(tables["absliq1"],radii.liquid_um,dtype=dtype),
            interpolate_radius(tables["absice3"],radii.ice_um,ice=True,dtype=dtype),
            interpolate_radius(tables["absice3"],radii.snow_um,ice=True,dtype=dtype))


def sw_radius_coefficients(radii, dtype, *, band_start=None, band_count=None):
    t = raw_cloud_tables("sw")
    def table(n):
        a = jnp.asarray(t[n], dtype)
        return a if band_start is None else jax.lax.dynamic_slice_in_dim(a, band_start, band_count, axis=0)
    def liq(n):return interpolate_radius(table(n),radii.liquid_um,dtype=dtype)[...,None]
    def frozen(n,r):return interpolate_radius(table(n),r,ice=True,dtype=dtype)[...,None]
    le,ls,la = (liq(n) for n in ("extliq1","ssaliq1","asyliq1"))
    ie,is_,ia,ifd = (frozen(n,radii.ice_um) for n in ("extice3","ssaice3","asyice3","fdlice3"))
    se,ss,sa,sfd = (frozen(n,radii.snow_um) for n in ("extice3","ssaice3","asyice3","fdlice3"))
    return le,ie,se,ls,is_,ss,la,ia,sa,la*la,jnp.minimum(ia,ifd+0.5/is_),jnp.minimum(sa,sfd+0.5/ss)
