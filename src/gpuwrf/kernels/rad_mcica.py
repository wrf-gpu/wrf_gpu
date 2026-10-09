"""Parallel, integer-exact WRF McICA KISS jump-ahead.

Each (column, layer, g-point) computes its own stream position. The small
shape-dependent tables are compile constants, never timestep host transfers.
MWC modular jump-ahead is valid for the pressure-derived seeds [0, 1e9):
these are below both moduli, including their carry bounds. This is not an
arbitrary-uint32 seed API. Random conversion and pressure seeding use WRF
real*4. GPUWRF_MCICA_LEGACY_FP64=1 retains the old solver's conversion for
baseline compatibility; it is not the B precision path.
"""

from functools import lru_cache
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

_LEGACY_FP64 = os.environ.get("GPUWRF_MCICA_LEGACY_FP64", "0") == "1"


# Reused from cadence-out kiss-prototype-product.patch (C3 prototype).
@lru_cache(maxsize=16)
def _kiss_jump_tables(start: int, count: int):
    """Integer-exact powers from the existing McICA jump-ahead prototype.

    LCG powers act modulo 2**32; xorshift powers are linear over GF(2).
    For WRF pressure-derived MWC seeds (<1e9), state n is a**n*s modulo
    a*65536-1. All products fit uint64. Tables depend only on layer count.
    """
    a, c = 1, 0
    basis = np.asarray([1 << b for b in range(32)], dtype=np.uint32)
    p3 = p4 = 1
    rows = []
    for n in range(start + count):
        if n >= start:
            rows.append((a, c, basis.copy(), p3, p4))
        a, c = (69069 * a) & 0xFFFFFFFF, (69069 * c + 1327217885) & 0xFFFFFFFF
        basis ^= basis << np.uint32(13)
        basis ^= basis >> np.uint32(17)
        basis ^= basis << np.uint32(5)
        p3 = p3 * 18000 % (18000 * 65536 - 1)
        p4 = p4 * 30903 % (30903 * 65536 - 1)
    return (
        np.asarray([r[0] for r in rows], np.uint32),
        np.asarray([r[1] for r in rows], np.uint32),
        np.asarray([r[2] for r in rows], np.uint32),
        np.asarray([r[3] for r in rows], np.uint64),
        np.asarray([r[4] for r in rows], np.uint64),
    )


def _kiss_jump_stream(seeds, start: int, count: int):
    """WRF's serial KISS values, computed independently per column/draw.

    ``start`` counts generator advances (first LW draw=151, SW draw=2).
    The caller supplies only WRF pressure-derived seeds, all in [0, 1e9).
    No floating-point RNG arithmetic and no device recurrence is used.
    """
    s1, s2, s3, s4 = seeds
    a, c, x, p3, p4 = _kiss_jump_tables(start, count)
    def column(v):
        return jnp.asarray(v).reshape((-1,) + (1,) * s1.ndim)
    k1 = column(a) * s1[None] + column(c)
    k2 = jnp.zeros_like(k1)
    for bit in range(32):
        active = ((s2 >> jnp.uint32(bit)) & jnp.uint32(1)).astype(bool)
        k2 ^= jnp.where(active[None], column(x[:, bit]), jnp.uint32(0))
    k3 = ((column(p3) * s3.astype(jnp.uint64)[None]) % jnp.uint64(18000 * 65536 - 1)).astype(jnp.uint32)
    k4 = ((column(p4) * s4.astype(jnp.uint64)[None]) % jnp.uint64(30903 * 65536 - 1)).astype(jnp.uint32)
    return k1 + k2 + (k3 << jnp.uint32(16)) + k4

def kiss_jump_stream(seeds, *, first: int, count: int):
    """Exact KISS uint32 words; pressure-derived seeds only, [0, 1e9)."""
    # Preserve the integer-64 modular products under an enclosing REAL32
    # radiation context. This is integer width, not an fp64 physics island.
    with jax.enable_x64(True):
        return _kiss_jump_stream(seeds, first, count)


def _pressure_seeds(p):
    if p.shape[-1] < 4:
        raise ValueError("WRF McICA needs at least four pressure layers")
    p_seed = p[..., :4].astype(jnp.float32) * jnp.float32(0.01)
    p_seed = p_seed * jnp.float32(100.0)
    fraction = p_seed - jnp.floor(p_seed)
    seed = (fraction * jnp.float32(1.0e9)).astype(jnp.uint32)
    return tuple(seed[..., i] for i in range(4))


def kiss_kernel_enabled() -> bool:
    """BP61 (trace time): GPUWRF_MCICA_KISS_KERNEL=1 draws the McICA stream with
    one Pallas lane per (column, g-point) -- integer-exact jump to the lane's
    first draw, then the serial WRF KISS recurrence over the layers."""
    return os.environ.get("GPUWRF_MCICA_KISS_KERNEL", "0") == "1"


_KISS_BLOCK = 128


@lru_cache(maxsize=16)
def _kiss_lane_tables(first: int, ng: int, nlay: int):
    """Jump-ahead rows for each g-point's first draw n = first + g*nlay."""
    a, c, x, p3, p4 = _kiss_jump_tables(first, ng * nlay)
    rows = np.arange(ng) * nlay
    return (a[rows].astype(np.uint32), c[rows].astype(np.uint32),
            np.ascontiguousarray(x[rows]).astype(np.uint32),
            p3[rows].astype(np.int64), p4[rows].astype(np.int64))


def _kiss_lane_kernel(s1_ref, s2_ref, s3_ref, s4_ref, a_ref, c_ref, x_ref, p3_ref, p4_ref, out_ref,
                      *, ncol, ng, nlay):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    lane = pl.program_id(0) * _KISS_BLOCK + jnp.arange(_KISS_BLOCK, dtype=jnp.int32)
    valid = lane < ncol * ng
    col = jnp.minimum(lane // ng, ncol - 1)
    g = lane % ng
    u32 = jnp.uint32
    s1, s2, s3, s4 = s1_ref[col], s2_ref[col], s3_ref[col], s4_ref[col]
    # Jump to the lane's first draw (same integer-exact algebra as _kiss_jump_stream).
    k1 = a_ref[g] * s1 + c_ref[g]
    k2 = jnp.zeros((_KISS_BLOCK,), u32)
    for bit in range(32):
        on = ((s2 >> u32(bit)) & u32(1)) == u32(1)
        k2 = k2 ^ jnp.where(on, x_ref[g, bit], u32(0))
    k3 = ((p3_ref[g] * s3.astype(jnp.int64)) % jnp.int64(18000 * 65536 - 1)).astype(u32)
    k4 = ((p4_ref[g] * s4.astype(jnp.int64)) % jnp.int64(30903 * 65536 - 1)).astype(u32)

    def emit(k, state):
        q1, q2, q3, q4 = state
        word = q1 + q2 + (q3 << u32(16)) + q4
        signed = lax.bitcast_convert_type(word, jnp.int32)
        value = signed.astype(jnp.float32) * jnp.float32(2.328306e-10) + jnp.float32(0.5)
        pt.store(out_ref.at[col, k, g], value, mask=valid)

    def step(k, state):
        emit(k, state)
        q1, q2, q3, q4 = state
        # WRF kissvec (module_ra_rrtmg_lw.F:2688-2706): one serial advance.
        q1 = q1 * u32(69069) + u32(1327217885)
        q2 = q2 ^ (q2 << u32(13))
        q2 = q2 ^ (q2 >> u32(17))
        q2 = q2 ^ (q2 << u32(5))
        q3 = u32(18000) * (q3 & u32(65535)) + (q3 >> u32(16))
        q4 = u32(30903) * (q4 & u32(65535)) + (q4 >> u32(16))
        return q1, q2, q3, q4

    state = lax.fori_loop(0, nlay - 1, step, (k1, k2, k3, k4))
    emit(nlay - 1, state)


def kiss_random_lanes(seeds, *, first: int, ng: int, nlay: int):
    """``(..., nlay, ng)`` REAL McICA uniforms; equals the jump-ahead stream draw-for-draw."""
    from jax.experimental import pallas as pl

    s1 = seeds[0]
    leading = s1.shape
    ncol = int(np.prod(leading)) if leading else 1
    tables = _kiss_lane_tables(first, ng, nlay)
    interpret = jax.default_backend() == "cpu"
    with jax.enable_x64(True):  # int64 MWC jump products (integer width, E75)
        flat = [jnp.asarray(v, jnp.uint32).reshape(ncol) for v in seeds]
        out = pl.pallas_call(
            lambda *r: _kiss_lane_kernel(*r, ncol=ncol, ng=ng, nlay=nlay),
            grid=(-(-ncol * ng // _KISS_BLOCK),),
            out_shape=jax.ShapeDtypeStruct((ncol, nlay, ng), jnp.float32),
            interpret=interpret, name="mcica_kiss_lanes",
        )(*flat, *(jnp.asarray(t) for t in tables))
    return out.reshape(leading + (nlay, ng))


def _random_values(p, *, first, ng, legacy_fp64, sw=False):
    nlay = p.shape[-1]
    # CPU runs the bitwise-equal jump-ahead stream: the lane kernel's CPU interpret mode aliases the
    # masked tail lanes onto (last column, g < tail) and returns NaN there when ncol*ng % 128 != 0.
    if kiss_kernel_enabled() and not legacy_fp64 and jax.default_backend() != "cpu":
        return kiss_random_lanes(_pressure_seeds(p), first=first, ng=ng, nlay=nlay)
    if sw and legacy_fp64:
        # Retain the exact legacy SW seeding graph. Its explicit fp64 floor/
        # subtraction prevents CUDA reassociation that changes fp64-input seeds.
        p_seed = (p.astype(jnp.float32) * jnp.float32(0.01)).astype(jnp.float32) * jnp.float32(100.0)
        p_seed = p_seed.astype(jnp.float64)
        fraction = p_seed - jnp.floor(p_seed)
        values = (fraction[..., :4].astype(jnp.float32) * jnp.float32(1e9)).astype(jnp.uint32)
        seeds = tuple(values[..., i] for i in range(4))
    else:
        seeds = _pressure_seeds(p)
    words = kiss_jump_stream(seeds, first=first, count=ng * nlay)
    signed = lax.bitcast_convert_type(words, jnp.int32)
    dtype = jnp.float64 if legacy_fp64 else jnp.float32
    random = signed.astype(dtype) * dtype(2.328306e-10) + dtype(0.5)
    random = random.reshape((ng, nlay) + p.shape[:-1])
    return jnp.moveaxis(random, (0, 1), (-1, -2))


# Overlap: icld=1 (random) by default. GPUWRF_RRTMG_MAXRAND=1 selects WRF's cldovrlp=2
# maximum-random overlap (Registry default): the SAME KISS draws, then the case(2) chain.
# Defined below the KISS-lane kernel so its Pallas source locations stay put (E58).
_MAXRAND = os.environ.get("GPUWRF_RRTMG_MAXRAND", "0") == "1"


def max_random_cdf(random, cldf):
    """WRF McICA maximum-random overlap (icld=2, irng=0) CDF chain.

    module_ra_rrtmg_lw.F:2481-2490 / module_ra_rrtmg_sw.F:1803-1812, after the
    same KISS draws as icld=1: for ilev = 2..nlay (layer 1 = bottom),
    CDF(ilev) = CDF(ilev-1) if CDF(ilev-1) > 1 - cldf(ilev-1) else
    CDF(ilev) * (1 - cldf(ilev-1)). ``random``: ``(..., nlay, ng)`` draws;
    ``cldf``: ``(..., nlay)`` cldmin-floored fractions. Arithmetic in the draws'
    dtype (WRF real(kind=rb) = REAL4 for the REAL32 path).
    """
    dtype = random.dtype
    clear = dtype.type(1.0) - cldf.astype(dtype)

    def step(prev, xs):
        draw, clear_below = xs
        cur = jnp.where(prev > clear_below[..., None], prev, draw * clear_below[..., None])
        return cur, cur

    draws = jnp.moveaxis(random, -2, 0)
    _, upper = lax.scan(step, draws[0], (draws[1:], jnp.moveaxis(clear, -1, 0)[:-1]))
    return jnp.moveaxis(jnp.concatenate((draws[:1], upper), axis=0), 0, -2)


def lw_cloud_mask(p_layer_pa, cloud_fraction, *, output_dtype=jnp.float32,
                  legacy_fp64=_LEGACY_FP64, maxrand=None):
    """Reference-compatible LW irng=0/permuteseed=150 mask: icld=1, or icld=2 with ``maxrand``."""
    maxrand = _MAXRAND if maxrand is None else maxrand
    random = _random_values(p_layer_pa, first=151, ng=140, legacy_fp64=legacy_fp64)
    if not legacy_fp64:
        cloud_fraction = cloud_fraction.astype(jnp.float32)
    dtype = cloud_fraction.dtype
    cldf = jnp.where(cloud_fraction < dtype.type(1.0e-20), dtype.type(0.0), cloud_fraction)
    if maxrand:
        random = max_random_cdf(random, cldf)
    return (random >= (1.0 - cldf[..., :, None])).astype(output_dtype)


def sw_cloud_mask(p_pa, cloud_fraction, gpoint_mask, *, legacy_fp64=_LEGACY_FP64, maxrand=None):
    """Reference-compatible SW mask (icld=1, or icld=2 with ``maxrand``), including padded points."""
    # Imported lazily: this standalone kernel does not change solver ownership.
    from gpuwrf.physics.rrtmg_sw import _SW_GLOBAL_GPOINT_INDEX, _SW_GPOINT_COUNTS

    maxrand = _MAXRAND if maxrand is None else maxrand
    random = _random_values(p_pa, first=2, ng=sum(_SW_GPOINT_COUNTS), legacy_fp64=legacy_fp64, sw=True)
    if not legacy_fp64:
        cloud_fraction = cloud_fraction.astype(jnp.float32)
    if maxrand:
        # WRF SW floors at cldmin too (module_ra_rrtmg_sw.F:1721); the icld=1 path keeps its graph.
        dtype = cloud_fraction.dtype
        cloud_fraction = jnp.where(cloud_fraction < dtype.type(1.0e-20), dtype.type(0.0), cloud_fraction)
        random = max_random_cdf(random, cloud_fraction)
    cloudy = random >= (1.0 - cloud_fraction[..., :, None])
    reduced = jnp.take(cloudy, _SW_GLOBAL_GPOINT_INDEX, axis=-1)
    return reduced.astype(gpoint_mask.dtype) * gpoint_mask
