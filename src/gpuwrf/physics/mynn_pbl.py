"""JAX MYNN2.5 PBL column kernel for M5-S2."""

from __future__ import annotations

import os
from functools import partial
from typing import Iterable

import jax
from jax import config, lax
import jax.numpy as jnp
import numpy as np

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.physics.mynn_constants import (
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    C5,
    CKMOD,
    CPHM_UNST,
    CTAU,
    E1C,
    E2C,
    E3C,
    E4C,
    E5C,
    G1,
    G2,
    GRAV,
    GTR,
    KARMAN,
    LOCAL_ALP1,
    LOCAL_ALP2,
    LOCAL_ALP3,
    LOCAL_ALP4,
    LOCAL_CNS,
    LOCAL_ELF_SOFT_MAX,
    LOCAL_ELT_MAX,
    LOCAL_ELT_MIN,
    MAX_PBLH_TRANSITION,
    MIN_PBLH,
    NL_ALP1,
    NL_ALP2,
    NL_ALP3,
    NL_ALP4,
    NL_ALP5,
    NL_BOULAC_LMAX,
    NL_CNS,
    NL_ELT_MAX,
    NL_ELT_MAX_WATER,
    NL_ELT_MIN,
    NL_QKW_ELB_MIN,
    NL_UONSET,
    P608,
    QKEMIN,
    QMIN,
    SQFAC,
    TKE_EPS,
    ZMAX,
)
from gpuwrf.physics.mynn_sgs_cloud import (
    dmp_shallow_cu_overwrite,
    mym_condensation_cloudpdf2,
    sgs_cloud_enabled,
)
from gpuwrf.physics.mynn_surface_stub import surface_layer
from gpuwrf.physics.tridiagonal_solver import solve_tridiagonal


config.update("jax_enable_x64", True)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


# Leading-column tiling over the flattened batch axis (the RRTMG pattern,
# proofs/v013/rrtmg_column_tile.json).  The default BouLac free-atmosphere
# length (`_boulac_length_dense`) materialises dense ``(B, nz, nz)`` PE matrices:
# at 641x321x50 fp64 ONE such matrix is ~3.8 GiB and several are live at once, so
# a whole-domain batch dominates the non-radiation physics peak.  The v0.17
# `GPUWRF_MYNN_BOULAC_ONZ` opt-in avoids the full matrix with the production
# source-chunked schedule; the legacy pure scan remains diagnostic-only because
# of its XLA compile/command-buffer pathology.  Every MYNN op is per-column
# (reductions run over the level axis; the EDMF plume sum runs inside a
# per-column vmap).
# Tile-vs-untiled is
# proven bit-identical on GPU including ragged tiles at the production tile
# width; on the CPU backend the implicit tridiagonal solves can differ by
# 1-2 ulp on scattered columns from batch-width-dependent SIMD codegen (all
# turbulence/diagnostic fields stay bit-exact) -- see
# proofs/v014/mythos_memory_fixes_260609.json.  CPU test/savepoint batches are
# below the default tile width, so they take the structurally untiled path.
# Set `GPUWRF_MYNN_COLUMN_TILING=0` or `GPUWRF_MYNN_COLUMN_TILE_COLS=0` for the
# whole-batch reference path used by proofs.
_MYNN_COLUMN_TILING = _env_bool("GPUWRF_MYNN_COLUMN_TILING", True)
_MYNN_COLUMN_TILE_COLS = max(0, _env_int("GPUWRF_MYNN_COLUMN_TILE_COLS", 16384))

# v0.15 perf lever (default OFF, proven opt-in): run the dense (B, nz, nz)
# BouLac free-atmosphere length-scale parcel search (`_boulac_length_dense`) in
# fp32 and cast the (lb1, lb2) lengths back to the caller dtype.  The BouLac
# matrices are the measured non-radiation hot spot (NVTX: MYNN_PBL(EDMF) ~88
# ms/step on d01, ~87% of the step; one fp64 (16384,44,44) array = 254 MB and
# ~20 live).
# The length scale is a capped/blended diagnostic mixing length, so fp32 on it is
# an operational-impact question (NOT bitwise).  CORRECTNESS PROVEN: 1-step
# injection theta rel 1.9e-10 (proofs/perf/v015/mynn_fp32_boulac_ab.json) and the
# short-forecast frozen-tolerance field gate PASSES at 0.06% of the manifest
# limits (proofs/perf/v015/tiered_gate_combined.json).
#
# Why DEFAULT OFF despite being correctness-safe and ~8 ms faster on the MYNN
# step: turning it ON in the FULL operational pipeline triggers an XLA "Very slow
# compile" pathology -- the mixed fp32/fp64 BouLac fusion makes the operational
# forecast jit compile crawl (>4 min, then stalls), reproduced on this branch
# across BFC/cuda_async allocators and with autotuning on/off.  The SAME pipeline
# with this knob OFF compiles cleanly in ~80 s (0 slow-compile warnings).  The
# DOMINANT v0.15 wall win (GPUWRF_MYNN_COND_NITER=16, -70 ms/step) is independent
# of this and stays default-ON.  Set GPUWRF_MYNN_BOULAC_FP32=1 to opt in where the
# one-time compile cost is acceptable (it is a validated, field-gate-passing win).
_MYNN_BOULAC_FP32 = _env_bool("GPUWRF_MYNN_BOULAC_FP32", False)

# v0.17 BouLac O(nz)-working-set memory lever (default OFF, production opt-in).
# The default `_boulac_length` path is deliberately unchanged unless this flag is
# set.  With GPUWRF_MYNN_BOULAC_ONZ=1 the dispatcher selects the production
# source-chunked BouLac schedule with chunk=1: it keeps the dense path's
# cumsum/where/reduce arithmetic but processes one SOURCE level at a time, so no
# full (B, nz, nz) PE matrix is live.  That is a memory-shape rewrite, not a
# precision change, and is bit-identical to the whole-domain dense result when
# GPUWRF_MYNN_BOULAC_FP32 is off.
#
# The original v0.15 pure scan implementation (`_boulac_length_onz`) remains in
# this file because it is WRF-faithful and useful for oracle/pathology probes,
# but it is NOT the production implementation: inside the full operational jit
# its deep data-dependent chain fuses into huge ptxas-spilling kernels and at
# 147k columns OOMs while instantiating CUDA command buffers.  Set
# GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN=1 together with GPUWRF_MYNN_BOULAC_ONZ=1
# only to reproduce that rejected path.
_MYNN_BOULAC_ONZ = _env_bool("GPUWRF_MYNN_BOULAC_ONZ", False)
_MYNN_BOULAC_ONZ_LEGACY_SCAN = _env_bool("GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN", False)

# v0.16 BouLac SOURCE-CHUNKED dense lever (default OFF; the 1km-UNLOCK path).
# The dense parcel search materialises (B, nz, nz) PE matrices; at 1km
# (147456x44x44 fp64 ~ 21 GiB) XLA coalesces a ~18.8 GiB scratch buffer that OOMs
# the GPU (MEASURED: dense FAILs at 147k trying to allocate 18.80 GiB).  The pure
# O(nz) `_boulac_length_onz` removes the matrices but defeats XLA fusion (deep
# data-dependent sequential scan -> "Very slow compile").  The CHUNKED form
# (`_boulac_length_chunked`) keeps the dense form's fusion-friendly
# cumsum/where/reduce kernel but slices the SOURCE-level axis into chunks of
# `GPUWRF_MYNN_BOULAC_CHUNK`: each chunk materialises only (B, chunk, nz) and the
# chunks are concatenated, so the coalesced BouLac scratch shrinks by ~nz/chunk
# while the outer loop is a SHORT shallow structural unroll (ceil(nz/chunk)
# fusion-friendly dense blocks), NOT the deep 86-iteration data-dependent chain
# that trips the compile pathology.  Numerically BIT-IDENTICAL to the dense form
# (oracle proofs/perf/v016/boulac_chunked_oracle.json: max_abs==0.0 vs whole-
# domain dense for every chunk size; only the source axis is partitioned, cumsums
# run over the FULL target axis inside every chunk).
#
# MEASURED 1km (147456-col fp64) threshold (proofs/perf/v016/boulac_chunk*_147k.json):
#   dense / chunk=4 / chunk=2 -> OOM (18.80 / 15.98 / 15.70 GiB single alloc);
#   chunk=1 -> RUNS, peak 18.25 GiB, finite.
# So DEFAULT=1 (the proven-fitting value) when CHUNKED is engaged: each block is
# (B,1,nz) so XLA never coalesces a multi-GiB BouLac buffer.  The cost is a longer
# compile (nz blocks: ~150 s at 147k vs ~60 s dense; NO slow-compile pathology)
# and the per-step cost is unchanged.  Raise GPUWRF_MYNN_BOULAC_CHUNK for smaller
# grids where the BouLac matrix is not the binding allocation and a shorter
# compile is preferred (chunk>=nz or <=0 == the whole-domain dense form).
_MYNN_BOULAC_CHUNK = _env_int("GPUWRF_MYNN_BOULAC_CHUNK", 1)
_MYNN_BOULAC_CHUNKED = _env_bool("GPUWRF_MYNN_BOULAC_CHUNKED", False)

# v0.15 IDENTITY fix (default ON): the WRF MYNN-EDMF subgrid-cloud chain --
# mym_condensation CASE(2) (bl_mynn_cloudpdf=2, the WRF Registry default the
# CPU truth runs with), the closure-2.6 prognostic total-water variance qsq,
# the DMP shallow-cu cldfra/qc_bl overwrite, and the SGS-cloud-aware thlv
# buoyancy. Without it the cloud-topped marine boundary layer under-entrains
# (no cloud buoyancy + no cloud for radiation) -> the Canary L2 d02 QVAPOR
# inversion-height miss (rmse 1.45e-3 vs the 1.0e-3 frozen limit).
# GPUWRF_MYNN_SGS_CLOUD=0 restores the pre-v0.15 dry-buoyancy path (shared
# gate with the icloud_bl radiation merge; see mynn_sgs_cloud.sgs_cloud_enabled).
_MYNN_SGS_CLOUD = sgs_cloud_enabled()

CP_MYNN = 7.0 * 287.0 / 2.0
RCP_MYNN = 287.0 / CP_MYNN
XLVCP_MYNN = 2.5e6 / CP_MYNN
XLSCP_MYNN = 2.85e6 / CP_MYNN


@jax.tree_util.register_pytree_node_class
class MynnPBLColumnState:
    """Pytree for a batch of independent MYNN PBL columns on mass levels."""

    __slots__ = (
        "u",
        "v",
        "w",
        "theta",
        "qv",
        "tke",
        "p",
        "rho",
        "dz",
        "km",
        "kh",
        "el",
        "qc",
        "qi",
        # --- v0.15 SGS-cloud leaves (append-only so old flattens reconstruct) ---
        # qs: resolved snow (feeds the WRF thlv rebuild + rh_hack);
        # qsq: PROGNOSED total-water variance <q'w q'w> (closure 2.6);
        # qc_bl/qi_bl/cldfra_bl: mym_condensation CASE(2) SGS cloud outputs.
        "qs",
        "qsq",
        "qc_bl",
        "qi_bl",
        "cldfra_bl",
    )

    def __init__(self, u, v, w, theta, qv, tke, p, rho, dz, km, kh, el, qc=None, qi=None,
                 qs=None, qsq=None, qc_bl=None, qi_bl=None, cldfra_bl=None) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.tke = tke
        self.p = p
        self.rho = rho
        self.dz = dz
        self.km = km
        self.kh = kh
        self.el = el
        self.qc = jnp.zeros_like(qv) if qc is None else qc
        self.qi = jnp.zeros_like(qv) if qi is None else qi
        self.qs = jnp.zeros_like(qv) if qs is None else qs
        self.qsq = jnp.zeros_like(qv) if qsq is None else qsq
        self.qc_bl = jnp.zeros_like(qv) if qc_bl is None else qc_bl
        self.qi_bl = jnp.zeros_like(qv) if qi_bl is None else qi_bl
        self.cldfra_bl = jnp.zeros_like(qv) if cldfra_bl is None else cldfra_bl

    def replace(self, **updates) -> "MynnPBLColumnState":
        """Returns a same-layout pytree with named fields replaced."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        """Presents all column arrays as JAX leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds the state after a JAX transform."""

        del aux
        return cls(*children)

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality outside JIT for cache/debug tests."""

        if not isinstance(other, MynnPBLColumnState):
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


def _leaves(state: MynnPBLColumnState) -> Iterable[jax.Array]:
    """Centralizes leaf iteration for equality, hashing, and byte accounting."""

    return (getattr(state, name) for name in MynnPBLColumnState.__slots__)


def _clip_state(state: MynnPBLColumnState) -> MynnPBLColumnState:
    """Applies WRF-like lower bounds before column physics."""

    return state.replace(
        qv=jnp.maximum(state.qv, 0.0),
        qc=jnp.maximum(state.qc, 0.0),
        qi=jnp.maximum(state.qi, 0.0),
        qs=jnp.maximum(state.qs, 0.0),
        qsq=jnp.maximum(state.qsq, 0.0),
        tke=jnp.maximum(state.tke, TKE_EPS),
        rho=jnp.maximum(state.rho, 1.0e-4),
        dz=jnp.maximum(state.dz, 1.0),
    )


def _zero_like(x):
    return x * 0.0


def _zero_edge_like(x):
    """Returns one zero edge column matching the batch dimensions of `x`."""

    return x[..., :1] * 0.0


def _edge_heights(dz):
    """Builds WRF-style interface heights with `zw(kts)=0`."""

    return jnp.concatenate((_zero_edge_like(dz), jnp.cumsum(dz, axis=-1)), axis=-1)


def _wrf_zw(dz):
    """Returns the WRF `zw(kts:kte)` subset used by MYNN internals."""

    return _edge_heights(dz)[..., :-1]


def _interface_dz(dz):
    """Returns `0.5*(dz(k)+dz(k-1))` with the surface slot unused."""

    interior = 0.5 * (dz[..., 1:] + dz[..., :-1])
    return jnp.concatenate((dz[..., :1], interior), axis=-1)


def _virtual_potential(theta, qv):
    """WRF virtual potential temperature ``thv = theta*(1+p608*sqv)``.

    WRF MYNN forms the buoyancy variable from the SPECIFIC humidity
    ``sqv = qv/(1+qv)`` (``module_bl_mynnedmf.F:673`` ``thv1(k)=th1(k)*(1+p608*sqv1(k))``
    and the ``thlv1`` build at :1006-1008), NOT the mixing ratio. The kernel
    previously used the mixing ratio ``qv`` directly; at the well-mixed-layer top
    the buoyancy gradient ``dthv/dz`` is a knife-edge near zero, and the ~1%
    ``qv`` vs ``qv/(1+qv)`` difference is enough to flip its sign — which decides
    whether the level-2.5 Sh treats the entrainment cell as (un)stable. Using the
    faithful specific humidity suppresses the spurious entrainment-zone over-mixing
    (the dominant clear-sky exch_h miss). qc/qi liquid-water (thl) and the SGS
    cloud-PDF condensate that WRF also folds into ``thlv1`` are zero in the dry
    unsaturated column and remain out of scope (see report)."""

    sqv = qv / (1.0 + jnp.maximum(qv, 0.0))
    return theta * (1.0 + P608 * sqv)


def _exner_from_pressure(p):
    return (p / 100000.0) ** RCP_MYNN


def _specific_moisture_components(state: MynnPBLColumnState):
    """Return WRF MYNN specific contents from WRF-style mixing ratios.

    WRF ``mynnedmf_pre_run`` converts vapor and cloud mixing ratios to specific
    contents before MYNN (`sqv=qv/(1+qv)`, and the cloud species use the same
    denominator).  The column state stores the prognostic WRF mixing ratios, so
    the cloud-aware thermodynamic closure has to do that conversion explicitly.
    """

    qv = jnp.maximum(state.qv, 0.0)
    denom = 1.0 + qv
    sqv = qv / denom
    sqc = jnp.maximum(state.qc, 0.0) / denom
    sqi = jnp.maximum(state.qi, 0.0) / denom
    sqw = sqv + sqc + sqi
    return sqv, sqc, sqi, sqw


def _specific_snow_content(state: MynnPBLColumnState):
    """WRF ``sqs`` (snow specific content); excluded from ``sqw`` (driver :675)."""

    qv = jnp.maximum(state.qv, 0.0)
    return jnp.maximum(state.qs, 0.0) / (1.0 + qv)


def _liquid_potential_temperature(state: MynnPBLColumnState):
    """WRF ``thl`` from theta plus cloud liquid/ice specific contents."""

    _sqv, sqc, sqi, _sqw = _specific_moisture_components(state)
    exner = _exner_from_pressure(state.p)
    return state.theta - XLVCP_MYNN / exner * sqc - XLSCP_MYNN / exner * sqi


def _moist_virtual_potential(state: MynnPBLColumnState, sgs_cloud: bool = False):
    """WRF ``thv`` and cloud-aware ``thlv`` used by MYNN level-2 closure.

    ``sgs_cloud=True`` applies the WRF driver ``thlv1`` rebuild AFTER
    ``mym_condensation``/``DMP_mf`` (module_bl_mynnedmf.F:1005-1009):

        thlv1(k) = (th1 - xlvcp/ex*max(qc_bl1, sqc1)
                        - xlscp/ex*max(qi_bl1, sqi1+sqs1)) * (1+p608*sqv1)

    i.e. the SGS cloud condensate (``state.qc_bl``/``state.qi_bl``) enters the
    buoyancy variable wherever it exceeds the resolved condensate. This is the
    load-bearing SGS-cloud buoyancy coupling (``use_buoy=.false.`` makes the
    vt/vq route dead); without it a cloud-topped marine boundary layer sees a
    spuriously stable cloud layer and under-entrains (the Canary QVAPOR
    inversion-height miss). ``sgs_cloud=False`` keeps the resolved-only form
    used at the driver entry and by the cold-start ``mym_initialize`` path.
    """

    sqv, sqc, sqi, _sqw = _specific_moisture_components(state)
    exner = _exner_from_pressure(state.p)
    thv = state.theta * (1.0 + P608 * sqv)
    if sgs_cloud:
        liq = jnp.maximum(state.qc_bl, sqc)
        ice = jnp.maximum(state.qi_bl, sqi + _specific_snow_content(state))
    else:
        liq = sqc
        ice = sqi
    thlv = (
        state.theta
        - XLVCP_MYNN / exner * liq
        - XLSCP_MYNN / exner * ice
    ) * (1.0 + P608 * sqv)
    return thv, thlv


def _surface_terms(state: MynnPBLColumnState, surface=None):
    """Builds the surface fluxes used by the column kernel.

    When ``surface`` (a :class:`mynn_surface_stub.SurfaceFluxes`) is provided —
    the operational coupling path, where the real WRF revised surface layer
    (``physics.surface_layer.surface_layer``) ran first in ``surface_adapter`` —
    the kernel consumes those fluxes directly. When it is ``None`` (the analytic
    Tier-1/Tier-2 MYNN fixture path) the legacy neutral-bulk stub is used so the
    standalone MYNN fixture stays a pure PBL test. The surface→MYNN order and
    flux hand-off are the FROZEN Gate-1 coupling (coupler_interface.md §3)."""

    flux = surface_layer(state) if surface is None else surface
    wind = jnp.maximum(jnp.sqrt(state.u[..., 0] * state.u[..., 0] + state.v[..., 0] * state.v[..., 0]), 0.2)
    return flux, wind, flux.fltv, flux.rhosfc


def _flux_richardson(ri, ri1, ri2, ri3, ri4, rfc):
    """WRF-faithful flux Richardson formula with an unguarded SQRT radicand."""

    radicand = ri * ri - ri3 * ri + ri4
    return jnp.minimum(ri1 * (ri + ri2 - jnp.sqrt(radicand)), rfc)


def _mym_level2(state: MynnPBLColumnState):
    """WRF MYNN `mym_level2`: level-2 gradients and stability functions.

    The buoyancy gradient uses the SGS-cloud-aware ``thlv`` (the WRF driver
    thlv1 rebuild with ``max(qc_bl, sqc)``); for states whose ``qc_bl``/``qi_bl``
    leaves are zero (cold start, analytic fixtures, pre-condensation callers)
    this reduces EXACTLY to the previous resolved-only form.
    """

    _thv, thlv = _moist_virtual_potential(state, sgs_cloud=True)
    thl = _liquid_potential_temperature(state)
    _sqv, _sqc, _sqi, sqw = _specific_moisture_components(state)
    dzk = _interface_dz(state.dz)[..., 1:]
    duz = ((state.u[..., 1:] - state.u[..., :-1]) ** 2 + (state.v[..., 1:] - state.v[..., :-1]) ** 2) / (dzk * dzk)
    dtl_i = (thl[..., 1:] - thl[..., :-1]) / dzk
    dqw_i = (sqw[..., 1:] - sqw[..., :-1]) / dzk
    dtv_i = (thlv[..., 1:] - thlv[..., :-1]) / dzk
    gm_i = duz
    gh_i = -dtv_i * GTR

    ri = -gh_i / jnp.maximum(duz, 1.0e-10)
    a2fac = jnp.where(CKMOD == 1.0, 1.0 / (1.0 + jnp.maximum(ri, 0.0)), 1.0)
    rfc = G1 / (G1 + G2)
    f1 = B1 * (G1 - C1) + 3.0 * A2 * a2fac * (1.0 - C2) * (1.0 - C5) + 2.0 * A1 * (3.0 - 2.0 * C2)
    f2 = B1 * (G1 + G2) - 3.0 * A1 * (1.0 - C2)
    rf1 = B1 * (G1 - C1) / f1
    rf2 = B1 * G1 / f2
    smc = A1 / (A2 * a2fac) * f1 / f2
    shc = 3.0 * (A2 * a2fac) * (G1 + G2)

    ri1 = 0.5 / smc
    ri2 = rf1 * smc
    ri3 = 4.0 * rf2 * smc - 2.0 * ri2
    ri4 = ri2 * ri2
    rf = _flux_richardson(ri, ri1, ri2, ri3, ri4, rfc)
    sh_i = shc * (rfc - rf) / (1.0 - rf)
    sm_i = smc * (rf1 - rf) / (rf2 - rf) * sh_i

    z = _zero_edge_like(state.theta)
    dtv = jnp.concatenate((z, dtv_i), axis=-1)
    dtl = jnp.concatenate((z, dtl_i), axis=-1)
    dqw = jnp.concatenate((z, dqw_i), axis=-1)
    gm = jnp.concatenate((z, gm_i), axis=-1)
    gh = jnp.concatenate((z, gh_i), axis=-1)
    sm = jnp.concatenate((z, sm_i), axis=-1)
    sh = jnp.concatenate((z, sh_i), axis=-1)
    return dtl, dqw, dtv, gm, gh, sm, sh


def _first_or_fallback(candidates, fallback):
    """Returns the lowest-height finite candidate along the column."""

    value = jnp.min(candidates, axis=-1)
    return jnp.where(jnp.isfinite(value), value, fallback)


def _get_pblh(state: MynnPBLColumnState, qke):
    """Dry JAX transcription of WRF `GET_PBLH` for MYNN length-scale input."""

    thv, _thlv = _moist_virtual_potential(state)
    zw = _wrf_zw(state.dz)
    nz = state.dz.shape[-1]
    idx = jnp.arange(nz)
    lowest_200 = (idx >= 1) & (zw <= 200.0)
    minthv = jnp.min(jnp.where(lowest_200, thv, jnp.inf), axis=-1)
    minthv = jnp.where(jnp.isfinite(minthv), minthv, thv[..., 0])
    delt_thv = 1.0
    theta_hit = (idx >= 1) & (idx <= nz - 2) & (thv >= (minthv[..., None] + delt_thv))
    prev_thv = jnp.concatenate((thv[..., :1], thv[..., :-1]), axis=-1)
    prev_dz = jnp.concatenate((state.dz[..., :1], state.dz[..., :-1]), axis=-1)
    theta_candidate = zw - prev_dz * jnp.minimum((thv - (minthv[..., None] + delt_thv)) / jnp.maximum(thv - prev_thv, 1.0e-6), 1.0)
    theta_pblh = _first_or_fallback(jnp.where(theta_hit, theta_candidate, jnp.inf), zw[..., 1])

    maxqke = jnp.maximum(qke[..., 0], 0.0)
    tke_eps = jnp.maximum(maxqke / 40.0, 0.01)
    qtke = jnp.maximum(0.5 * qke, 0.0)
    qtkem1 = jnp.concatenate((qtke[..., :1], qtke[..., :-1]), axis=-1)
    tke_hit = (idx >= 1) & (idx <= nz - 2) & (qtke <= tke_eps[..., None])
    tke_candidate = zw - prev_dz * jnp.minimum((tke_eps[..., None] - qtke) / jnp.maximum(qtkem1 - qtke, 1.0e-6), 1.0)
    tke_pblh = jnp.maximum(_first_or_fallback(jnp.where(tke_hit, tke_candidate, jnp.inf), zw[..., 1]), zw[..., 1])
    tke_pblh = jnp.minimum(tke_pblh, theta_pblh + 350.0)
    tke_pblh = jnp.maximum(tke_pblh, jnp.maximum(theta_pblh - 350.0, 10.0))
    wt = 0.5 * jnp.tanh((theta_pblh - 200.0) / 400.0) + 0.5
    return jnp.where(maxqke <= tke_eps, theta_pblh, tke_pblh * (1.0 - wt) + theta_pblh * wt)


def _scale_aware_psig_bl(dx, pblh):
    """WRF ``SCALE_AWARE`` Psig_bl (``module_bl_mynnedmf.F:7487-7512``).

    Honnert-et-al taper that keeps parameterized PBL mixing on coarse grids
    (Psig_bl~1 when dx >> PBLH) and reduces it as the grid begins to resolve the
    boundary-layer eddies. ``dx`` is the grid spacing (m); ``pblh`` per column."""

    pblh_pos = jnp.maximum(pblh, MIN_PBLH)
    dxdh = jnp.maximum(2.5 * dx, 10.0) / jnp.minimum(pblh_pos, 3000.0)
    dxdh2 = dxdh * dxdh
    dxdh23 = dxdh ** 0.667
    return (dxdh2 + 0.106 * dxdh23) / (dxdh2 + 0.066 * dxdh23 + 0.071)


def _boulac_length_onz(zw, dz, qtke, theta):
    """WRF ``boulac_length`` (``module_bl_mynnedmf.F:2192-2338``), O(nz) memory.

    DEFAULT-OFF (``GPUWRF_MYNN_BOULAC_ONZ``): proven correct + faster/lighter in
    ISOLATION but BLOCKED by an XLA "Very slow compile" pathology in the full
    operational forecast jit -- see the ``_MYNN_BOULAC_ONZ`` note above and the
    dispatcher :func:`_boulac_length`.

    For each level ``iz`` it integrates the buoyant displacement a parcel with
    TKE ``qtke(iz)`` can travel up (``dlu``) and down (``dld``) before its TKE is
    consumed by potential energy, then returns ``lb1=min(dlu,dld)`` and
    ``lb2=sqrt(dlu*dld)``.  ``beta = gtr`` is the buoyancy coefficient.

    WRF runs two nested data-dependent ``DO WHILE`` loops per source level
    ``iz`` (the inner search over ``izz``).  This is a faithful structural
    transcription: the inner ``izz`` search is a Python-unrolled loop over the
    search axis, with the *source level* carried as a batched ``(..., nz)``
    vector so every column AND every source level advances its own
    ``zup``/``zup_inf``/``zzz``/``found`` accumulators in lock-step.  At search
    position ``izz`` (a scalar level), source ``iz`` is active iff ``iz <= izz``
    (upward) / ``iz >= izz`` (downward); that single conditional reproduces
    WRF's "start the search at ``izz=iz``" semantics.  Only O(B, nz) state is
    live -- the dense ``_boulac_length_dense`` materialises ``(B, nz, nz)`` PE
    matrices.  The per-crossing arithmetic is the verbatim Fortran formula; this
    is an algorithmic/memory change, NOT a numerics change.

    Arrays are last-axis (``..., nz``); ``zw`` is the WRF ``zw(kts:kte)`` subset
    (length nz).  WRF references ``zw(kte+1)`` (the top interface) and
    ``zw(iz+1)``; we reconstruct those from the cumulative layer depths so no
    extra interface array is threaded through.
    """

    # v0.15 optional fp32 island (default OFF): the parcel search is
    # bandwidth-bound; computing it in fp32 halves the transient HBM traffic.
    # Cast inputs in, cast (lb1, lb2) back to caller dtype at the return.
    # Engaged via GPUWRF_MYNN_BOULAC_FP32=1; proven by the frozen
    # field-tolerance gate, not assumed bit-identical.
    _out_dtype = jnp.result_type(zw, dz, qtke, theta)
    work_dtype = _out_dtype
    if _MYNN_BOULAC_FP32 and _out_dtype == jnp.float64:
        work_dtype = jnp.float32
        zw = zw.astype(work_dtype)
        dz = dz.astype(work_dtype)
        qtke = qtke.astype(work_dtype)
        theta = theta.astype(work_dtype)

    beta = jnp.asarray(GTR, dtype=work_dtype)
    nz = dz.shape[-1]
    batch_shape = jnp.broadcast_shapes(
        zw.shape[:-1], dz.shape[:-1], qtke.shape[:-1], theta.shape[:-1]
    )
    col_shape = batch_shape + (nz,)
    zw = jnp.broadcast_to(zw, col_shape)
    dz = jnp.broadcast_to(dz, col_shape)
    qtke = jnp.broadcast_to(qtke, col_shape)
    theta = jnp.broadcast_to(theta, col_shape)

    # Full interface heights zwf(kts:kte+1), length nz+1: zwf[k]=sum(dz[:k]).
    zwf = _edge_heights(dz)                 # (..., nz+1); zwf[...,0]=0
    zw_top = zwf[..., -1:]                   # zw(kte+1)
    zw_kp1 = zwf[..., 1:]                     # zw(k+1) for k=0..nz-1, length nz

    i_idx = jnp.arange(nz)
    src_is_top = i_idx == (nz - 1)           # iz==kte: no upward integration
    src_is_bot = i_idx == 0                  # iz==kts: no downward integration

    zero_cols = jnp.zeros(col_shape, dtype=work_dtype)
    false_cols = jnp.zeros(col_shape, dtype=bool)

    # The inner ``izz`` search is a Python-unrolled loop over the search axis,
    # NOT a lax.scan: the loop body is the same per-step accumulator update, but
    # unrolling keeps the working set at O(B, nz) (no dense (B,nz,nz)) while
    # producing a STRAIGHT-LINE elementwise graph that XLA fuses quickly.  A
    # lax.scan here lowers to an XLA ``while`` whose body, nested inside the full
    # operational forecast jit, tripped a "Very slow compile" pathology; the
    # unrolled form has identical numerics (verified by the column oracle) and
    # compiles cleanly.  The accumulators are (..., nz) over the SOURCE level.

    th_jp1 = jnp.concatenate((theta[..., 1:], theta[..., -1:]), axis=-1)  # theta(k+1)
    th_jm1 = jnp.concatenate((theta[..., :1], theta[..., :-1]), axis=-1)  # theta(k-1)
    dz_jm1 = jnp.concatenate((dz[..., :1], dz[..., :-1]), axis=-1)        # dz(k-1)

    def _bcast(arr, izz):
        # broadcast the scalar-level-izz slice (a (...,) leaf) over (..., nz).
        return arr[..., izz][..., None]

    # ---------------- UPWARD search (dlu) ----------------
    # WRF default (no crossing found): dlu(iz)=zw(kte+1)-zw(iz)-dz(iz)*0.5.
    dlu = zw_top - zw - dz * 0.5
    zup, zup_inf, zzz, found = zero_cols, zero_cols, zero_cols, false_cols
    for izz in range(nz - 1):                     # izz = 0 .. nz-2 (upward)
        th_j = _bcast(theta, izz)
        th_jp = _bcast(th_jp1, izz)               # theta(izz+1)
        dzt = _bcast(dz, izz)
        active = (i_idx <= izz) & (~src_is_top)   # source iz<=izz can integrate up
        zup_new = zup + beta * (th_jp + th_j) * dzt * 0.5 - beta * theta * dzt
        zzz_new = zzz + dzt
        cross = active & (~found) & (qtke < zup_new) & (qtke >= zup_inf)
        bbb = jnp.where(jnp.abs(th_jp - th_j) > 0.0, (th_jp - th_j) / dzt, 0.0)
        rad = jnp.maximum(
            (beta * (th_j - theta)) ** 2 + 2.0 * bbb * beta * (qtke - zup_inf), 0.0
        )
        tl_b = (-beta * (th_j - theta) + jnp.sqrt(rad)) / jnp.where(
            bbb != 0.0, bbb * beta, 1.0
        )
        tl_lin = jnp.where(
            th_j != theta,
            (qtke - zup_inf) / (beta * jnp.where(th_j != theta, th_j - theta, 1.0)),
            0.0,
        )
        tl = jnp.where(bbb != 0.0, tl_b, tl_lin)
        dlu = jnp.where(cross, zzz_new - dzt + tl, dlu)
        found = found | cross
        # accumulators only advance on active source levels (WRF's zup/zzz/
        # zup_inf move only while the search runs).
        zup = jnp.where(active, zup_new, zup)
        zzz = jnp.where(active, zzz_new, zzz)
        zup_inf = jnp.where(active, zup_new, zup_inf)

    # ---------------- DOWNWARD search (dld) ----------------
    # WRF default (no crossing found): dld(iz)=zw(iz).
    dld = zw
    zdo, zdo_sup, zzz, found = zero_cols, zero_cols, zero_cols, false_cols
    for izz in range(nz - 1, 0, -1):              # izz = nz-1 .. 1 (downward)
        th_j = _bcast(theta, izz)
        th_jm = _bcast(th_jm1, izz)               # theta(izz-1)
        dzt = _bcast(dz_jm1, izz)                  # dz(izz-1)
        active = (i_idx >= izz) & (~src_is_bot)   # source iz>=izz can integrate down
        zdo_new = zdo + beta * theta * dzt - beta * (th_jm + th_j) * dzt * 0.5
        zzz_new = zzz + dzt
        cross = active & (~found) & (qtke < zdo_new) & (qtke >= zdo_sup)
        bbb = jnp.where(jnp.abs(th_j - th_jm) > 0.0, (th_j - th_jm) / dzt, 0.0)
        rad = jnp.maximum(
            (beta * (th_j - theta)) ** 2 + 2.0 * bbb * beta * (qtke - zdo_sup), 0.0
        )
        tl_b = (beta * (th_j - theta) + jnp.sqrt(rad)) / jnp.where(
            bbb != 0.0, bbb * beta, 1.0
        )
        tl_lin = jnp.where(
            th_j != theta,
            (qtke - zdo_sup) / (beta * jnp.where(th_j != theta, th_j - theta, 1.0)),
            0.0,
        )
        tl = jnp.where(bbb != 0.0, tl_b, tl_lin)
        dld = jnp.where(cross, zzz_new - dzt + tl, dld)
        found = found | cross
        zdo = jnp.where(active, zdo_new, zdo)
        zzz = jnp.where(active, zzz_new, zzz)
        zdo_sup = jnp.where(active, zdo_new, zdo_sup)

    # dld(iz) = min(dld(iz), zw(iz+1)); soft Lmax limit on both.
    dld = jnp.minimum(dld, zw_kp1)
    dlu = jnp.maximum(0.1, dlu / (1.0 + dlu / NL_BOULAC_LMAX))
    dld = jnp.maximum(0.1, dld / (1.0 + dld / NL_BOULAC_LMAX))

    lb1 = jnp.minimum(dlu, dld)
    lb2 = jnp.sqrt(dlu * dld)
    # WRF copies the top level from kte-1.
    lb1 = jnp.concatenate((lb1[..., :-1], lb1[..., -2:-1]), axis=-1)
    lb2 = jnp.concatenate((lb2[..., :-1], lb2[..., -2:-1]), axis=-1)
    if _MYNN_BOULAC_FP32 and _out_dtype == jnp.float64:
        lb1 = lb1.astype(_out_dtype)
        lb2 = lb2.astype(_out_dtype)
    return lb1, lb2


def _boulac_length_dense(zw, dz, qtke, theta):
    """Vectorized WRF ``boulac_length`` (``module_bl_mynnedmf.F:2192-2338``).

    PRODUCTION DEFAULT.  For each level ``iz`` it integrates the buoyant
    displacement a parcel with TKE ``qtke(iz)`` can travel up (``dlu``) and down
    (``dld``) before its TKE is consumed by potential energy, then returns
    ``lb1=min(dlu,dld)`` and ``lb2=sqrt(dlu*dld)``. The Fortran does this with
    two nested data-dependent while loops per ``iz``; here the inner search is
    unrolled into the dense (nz x nz) PE-accumulation matrices and the first
    TKE-crossing is selected with a cumulative-OR mask. ``beta = gtr`` is the
    buoyancy coefficient.

    This O(nz^2)-memory form is the DEFAULT because it compiles in ~80 s in the
    full operational forecast jit; the O(nz) :func:`_boulac_length_onz` is
    correct and lighter in isolation but trips an XLA slow-compile pathology in
    that jit (see the ``_MYNN_BOULAC_ONZ`` note).

    Arrays are last-axis (``..., nz``); ``zw`` is the WRF ``zw(kts:kte)`` subset
    (length nz). WRF references ``zw(kte+1)`` (the top interface) and
    ``zw(iz+1)``; we reconstruct those from the cumulative layer depths so no
    extra interface array is threaded through.
    """

    # v0.15 optional fp32 island (default OFF): the dense (B, nz, nz) parcel
    # search below is bandwidth-bound; computing it in fp32 halves the transient
    # HBM traffic.  Cast inputs in, cast (lb1, lb2) back to caller dtype at the
    # return.  Engaged via GPUWRF_MYNN_BOULAC_FP32=1; proven by the frozen
    # field-tolerance gate, not assumed bit-identical.
    _out_dtype = jnp.result_type(zw, dz, qtke, theta)
    if _MYNN_BOULAC_FP32 and _out_dtype == jnp.float64:
        zw = zw.astype(jnp.float32)
        dz = dz.astype(jnp.float32)
        qtke = qtke.astype(jnp.float32)
        theta = theta.astype(jnp.float32)

    beta = GTR
    nz = dz.shape[-1]
    # Full interface heights zwf(kts:kte+1), length nz+1: zwf[k]=sum(dz[:k]).
    zwf = _edge_heights(dz)                 # (..., nz+1); zwf[...,0]=0
    zw_top = zwf[..., -1:]                   # zw(kte+1)
    zw_kp1 = zwf[..., 1:]                     # zw(k+1) for k=0..nz-1, length nz

    i_idx = jnp.arange(nz)                    # source level iz
    j_idx = jnp.arange(nz)                    # target level izz
    src = i_idx[:, None]                      # (nz_src, 1)
    tgt = j_idx[None, :]                      # (1, nz_tgt)

    theta_i = theta[..., :, None]             # theta(iz) broadcast over izz
    theta_j = theta[..., None, :]             # theta(izz)
    dz_j = dz[..., None, :]                   # dz(izz)
    # theta(izz+1): shift; top level uses theta(nz-1) (only used where j<=nz-2).
    theta_jp1 = jnp.concatenate((theta[..., 1:], theta[..., -1:]), axis=-1)[..., None, :]
    theta_jm1 = jnp.concatenate((theta[..., :1], theta[..., :-1]), axis=-1)[..., None, :]
    dz_jm1 = jnp.concatenate((dz[..., :1], dz[..., :-1]), axis=-1)[..., None, :]

    # ---------------- UPWARD search (dlu) ----------------
    # zup increment at level izz (valid for iz<=izz<=nz-2):
    #   d_zup = beta*(theta(izz+1)+theta(izz))*dz(izz)*0.5 - beta*theta(iz)*dz(izz)
    up_incr = (beta * (theta_jp1 + theta_j) * dz_j * 0.5
               - beta * theta_i * dz_j)
    up_valid = (tgt >= src) & (tgt <= nz - 2)
    up_incr = jnp.where(up_valid, up_incr, 0.0)
    zup = jnp.cumsum(up_incr, axis=-1)        # zup after processing level izz
    zup_inf = jnp.concatenate((jnp.zeros_like(zup[..., :1]), zup[..., :-1]), axis=-1)
    zzz_up = jnp.cumsum(jnp.where(up_valid, dz_j, 0.0), axis=-1)  # depth iz..izz

    qtke_i = qtke[..., :, None]
    bbb_up = jnp.where(jnp.abs(theta_jp1 - theta_j) > 0.0,
                       (theta_jp1 - theta_j) / dz_j, 0.0)
    rad_up = jnp.maximum((beta * (theta_j - theta_i)) ** 2
                         + 2.0 * bbb_up * beta * (qtke_i - zup_inf), 0.0)
    tl_up_b = jnp.where(bbb_up != 0.0,
                        (-beta * (theta_j - theta_i) + jnp.sqrt(rad_up))
                        / jnp.where(bbb_up != 0.0, bbb_up * beta, 1.0),
                        0.0)
    tl_up_lin = jnp.where(theta_j != theta_i,
                          (qtke_i - zup_inf) / (beta * jnp.where(theta_j != theta_i, theta_j - theta_i, 1.0)),
                          0.0)
    tl_up = jnp.where(bbb_up != 0.0, tl_up_b, tl_up_lin)
    dlu_cand = zzz_up - dz_j + tl_up
    # WRF crossing: qtke(iz) < zup .and. qtke(iz) >= zup_inf, scanning izz upward.
    up_cross = up_valid & (qtke_i < zup) & (qtke_i >= zup_inf)
    up_first = up_cross & (jnp.cumsum(up_cross.astype(jnp.int32), axis=-1) == 1)
    dlu_default = zw_top[..., None] - zw[..., :, None] - dz[..., :, None] * 0.5
    dlu = jnp.where(jnp.any(up_first, axis=-1, keepdims=True),
                    jnp.sum(jnp.where(up_first, dlu_cand, 0.0), axis=-1, keepdims=True),
                    dlu_default)[..., 0]
    # iz==kte (top) cannot integrate upward -> keeps default (handled by mask).
    dlu = jnp.where(i_idx < nz - 1, dlu, dlu_default[..., 0])

    # ---------------- DOWNWARD search (dld) ----------------
    # at level izz (valid for kts+1<=izz<=iz, scanning izz downward from iz):
    #   d_zdo = beta*theta(iz)*dz(izz-1) - beta*(theta(izz-1)+theta(izz))*dz(izz-1)*0.5
    do_incr = (beta * theta_i * dz_jm1
               - beta * (theta_jm1 + theta_j) * dz_jm1 * 0.5)
    do_valid = (tgt <= src) & (tgt >= 1)
    do_incr = jnp.where(do_valid, do_incr, 0.0)
    # cumulative scanning DOWNWARD (decreasing izz) -> reverse-cumsum from izz=iz.
    zdo = jnp.cumsum(do_incr[..., ::-1], axis=-1)[..., ::-1]
    zdo_sup = jnp.concatenate((zdo[..., 1:], jnp.zeros_like(zdo[..., :1])), axis=-1)
    zzz_do = jnp.cumsum(jnp.where(do_valid, dz_jm1, 0.0)[..., ::-1], axis=-1)[..., ::-1]

    bbb_do = jnp.where(jnp.abs(theta_j - theta_jm1) > 0.0,
                       (theta_j - theta_jm1) / dz_jm1, 0.0)
    rad_do = jnp.maximum((beta * (theta_j - theta_i)) ** 2
                         + 2.0 * bbb_do * beta * (qtke_i - zdo_sup), 0.0)
    tl_do_b = jnp.where(bbb_do != 0.0,
                        (beta * (theta_j - theta_i) + jnp.sqrt(rad_do))
                        / jnp.where(bbb_do != 0.0, bbb_do * beta, 1.0),
                        0.0)
    tl_do_lin = jnp.where(theta_j != theta_i,
                          (qtke_i - zdo_sup) / (beta * jnp.where(theta_j != theta_i, theta_j - theta_i, 1.0)),
                          0.0)
    tl_do = jnp.where(bbb_do != 0.0, tl_do_b, tl_do_lin)
    dld_cand = zzz_do - dz_jm1 + tl_do
    do_cross = do_valid & (qtke_i < zdo) & (qtke_i >= zdo_sup)
    # first crossing scanning DOWNWARD: rank by reversed cumulative count.
    do_first = do_cross & (jnp.cumsum(do_cross[..., ::-1].astype(jnp.int32), axis=-1)[..., ::-1] == 1)
    dld_default = zw[..., :, None]
    dld = jnp.where(jnp.any(do_first, axis=-1, keepdims=True),
                    jnp.sum(jnp.where(do_first, dld_cand, 0.0), axis=-1, keepdims=True),
                    dld_default)[..., 0]
    dld = jnp.where(i_idx > 0, dld, dld_default[..., 0])

    # dld(iz) = min(dld(iz), zw(iz+1)); soft Lmax limit on both.
    dld = jnp.minimum(dld, zw_kp1)
    dlu = jnp.maximum(0.1, dlu / (1.0 + dlu / NL_BOULAC_LMAX))
    dld = jnp.maximum(0.1, dld / (1.0 + dld / NL_BOULAC_LMAX))

    lb1 = jnp.minimum(dlu, dld)
    lb2 = jnp.sqrt(dlu * dld)
    # WRF copies the top level from kte-1.
    lb1 = jnp.concatenate((lb1[..., :-1], lb1[..., -2:-1]), axis=-1)
    lb2 = jnp.concatenate((lb2[..., :-1], lb2[..., -2:-1]), axis=-1)
    if _MYNN_BOULAC_FP32 and _out_dtype == jnp.float64:
        lb1 = lb1.astype(_out_dtype)
        lb2 = lb2.astype(_out_dtype)
    return lb1, lb2


def _boulac_dlu_dld_chunk(
    s0, s1, *, beta, nz, theta, dz, qtke, zw, zw_top,
    theta_jp1_row, theta_jm1_row, dz_jm1_row, j_idx,
):
    """Dense BouLac up/down search for the SOURCE rows ``[s0:s1]`` only.

    Materialises (B, s1-s0, nz) intermediates (vs (B, nz, nz) for the whole
    domain): the SOURCE axis is sliced, the TARGET axis (the inner cumsum search)
    stays full-width so no crossing information is lost.  The arithmetic is the
    VERBATIM dense-form arithmetic (``_boulac_length_dense``); this is a memory
    partition, not a numerics change.  Returns ``(dlu_chunk, dld_chunk)`` of shape
    ``(..., s1-s0)``.
    """

    src = j_idx[s0:s1][:, None]                       # (chunk, 1) source levels
    tgt = j_idx[None, :]                               # (1, nz) target levels

    theta_i = theta[..., s0:s1, None]                  # theta(iz) for chunk rows
    qtke_i = qtke[..., s0:s1, None]
    theta_j = theta[..., None, :]                       # theta(izz) full target
    dz_j = dz[..., None, :]
    theta_jp1 = theta_jp1_row[..., None, :]
    theta_jm1 = theta_jm1_row[..., None, :]
    dz_jm1 = dz_jm1_row[..., None, :]

    # ---------------- UPWARD search (dlu) ----------------
    up_incr = (beta * (theta_jp1 + theta_j) * dz_j * 0.5
               - beta * theta_i * dz_j)
    up_valid = (tgt >= src) & (tgt <= nz - 2)
    up_incr = jnp.where(up_valid, up_incr, 0.0)
    zup = jnp.cumsum(up_incr, axis=-1)
    zup_inf = jnp.concatenate((jnp.zeros_like(zup[..., :1]), zup[..., :-1]), axis=-1)
    zzz_up = jnp.cumsum(jnp.where(up_valid, dz_j, 0.0), axis=-1)

    bbb_up = jnp.where(jnp.abs(theta_jp1 - theta_j) > 0.0,
                       (theta_jp1 - theta_j) / dz_j, 0.0)
    rad_up = jnp.maximum((beta * (theta_j - theta_i)) ** 2
                         + 2.0 * bbb_up * beta * (qtke_i - zup_inf), 0.0)
    tl_up_b = jnp.where(bbb_up != 0.0,
                        (-beta * (theta_j - theta_i) + jnp.sqrt(rad_up))
                        / jnp.where(bbb_up != 0.0, bbb_up * beta, 1.0),
                        0.0)
    tl_up_lin = jnp.where(theta_j != theta_i,
                          (qtke_i - zup_inf) / (beta * jnp.where(theta_j != theta_i, theta_j - theta_i, 1.0)),
                          0.0)
    tl_up = jnp.where(bbb_up != 0.0, tl_up_b, tl_up_lin)
    dlu_cand = zzz_up - dz_j + tl_up
    up_cross = up_valid & (qtke_i < zup) & (qtke_i >= zup_inf)
    up_first = up_cross & (jnp.cumsum(up_cross.astype(jnp.int32), axis=-1) == 1)
    dlu_default = zw_top[..., None] - zw[..., s0:s1, None] - dz[..., s0:s1, None] * 0.5
    dlu = jnp.where(jnp.any(up_first, axis=-1, keepdims=True),
                    jnp.sum(jnp.where(up_first, dlu_cand, 0.0), axis=-1, keepdims=True),
                    dlu_default)[..., 0]
    dlu = jnp.where(src[:, 0] < nz - 1, dlu, dlu_default[..., 0])

    # ---------------- DOWNWARD search (dld) ----------------
    do_incr = (beta * theta_i * dz_jm1
               - beta * (theta_jm1 + theta_j) * dz_jm1 * 0.5)
    do_valid = (tgt <= src) & (tgt >= 1)
    do_incr = jnp.where(do_valid, do_incr, 0.0)
    zdo = jnp.cumsum(do_incr[..., ::-1], axis=-1)[..., ::-1]
    zdo_sup = jnp.concatenate((zdo[..., 1:], jnp.zeros_like(zdo[..., :1])), axis=-1)
    zzz_do = jnp.cumsum(jnp.where(do_valid, dz_jm1, 0.0)[..., ::-1], axis=-1)[..., ::-1]

    bbb_do = jnp.where(jnp.abs(theta_j - theta_jm1) > 0.0,
                       (theta_j - theta_jm1) / dz_jm1, 0.0)
    rad_do = jnp.maximum((beta * (theta_j - theta_i)) ** 2
                         + 2.0 * bbb_do * beta * (qtke_i - zdo_sup), 0.0)
    tl_do_b = jnp.where(bbb_do != 0.0,
                        (beta * (theta_j - theta_i) + jnp.sqrt(rad_do))
                        / jnp.where(bbb_do != 0.0, bbb_do * beta, 1.0),
                        0.0)
    tl_do_lin = jnp.where(theta_j != theta_i,
                          (qtke_i - zdo_sup) / (beta * jnp.where(theta_j != theta_i, theta_j - theta_i, 1.0)),
                          0.0)
    tl_do = jnp.where(bbb_do != 0.0, tl_do_b, tl_do_lin)
    dld_cand = zzz_do - dz_jm1 + tl_do
    do_cross = do_valid & (qtke_i < zdo) & (qtke_i >= zdo_sup)
    do_first = do_cross & (jnp.cumsum(do_cross[..., ::-1].astype(jnp.int32), axis=-1)[..., ::-1] == 1)
    dld_default = zw[..., s0:s1, None]
    dld = jnp.where(jnp.any(do_first, axis=-1, keepdims=True),
                    jnp.sum(jnp.where(do_first, dld_cand, 0.0), axis=-1, keepdims=True),
                    dld_default)[..., 0]
    dld = jnp.where(src[:, 0] > 0, dld, dld_default[..., 0])
    return dlu, dld


def _boulac_length_chunked(zw, dz, qtke, theta, chunk=None):
    """Source-chunked dense BouLac length (``GPUWRF_MYNN_BOULAC_CHUNKED=1``).

    Numerically IDENTICAL to :func:`_boulac_length_dense` (same arithmetic), but
    the dense (B, nz, nz) parcel-search matrices are never materialised whole:
    the SOURCE-level axis is processed in contiguous chunks of
    ``GPUWRF_MYNN_BOULAC_CHUNK`` (default 1 -- the MEASURED value that fits 1km
    fp64; ``_MYNN_BOULAC_CHUNK``), so each block materialises only
    (B, chunk, nz).  Peak BouLac memory drops by ~nz/chunk.  The outer loop is a
    SHORT shallow structural unroll over ceil(nz/chunk) fusion-friendly dense
    blocks (each a pure cumsum/where/reduce graph that XLA fuses well) -- NOT the
    deep data-dependent sequential chain of ``_boulac_length_onz`` that trips the
    XLA "Very slow compile" pathology.  This is the 1km-UNLOCK path (the dense
    (B,nz,nz) matrix is the ~18.8 GiB single allocation that OOMs at 147k cols).
    """

    _out_dtype = jnp.result_type(zw, dz, qtke, theta)
    if _MYNN_BOULAC_FP32 and _out_dtype == jnp.float64:
        zw = zw.astype(jnp.float32)
        dz = dz.astype(jnp.float32)
        qtke = qtke.astype(jnp.float32)
        theta = theta.astype(jnp.float32)

    beta = GTR
    nz = dz.shape[-1]
    zwf = _edge_heights(dz)                 # (..., nz+1); zwf[...,0]=0
    zw_top = zwf[..., -1:]                   # zw(kte+1)
    zw_kp1 = zwf[..., 1:]                     # zw(k+1), length nz

    j_idx = jnp.arange(nz)
    # target-shifted rows, length nz (full target axis, shared by all chunks).
    theta_jp1_row = jnp.concatenate((theta[..., 1:], theta[..., -1:]), axis=-1)
    theta_jm1_row = jnp.concatenate((theta[..., :1], theta[..., :-1]), axis=-1)
    dz_jm1_row = jnp.concatenate((dz[..., :1], dz[..., :-1]), axis=-1)

    chunk = _MYNN_BOULAC_CHUNK if chunk is None else chunk
    if chunk <= 0 or chunk >= nz:
        # Fall back to a single whole-domain block (== the dense form).
        chunk = nz

    dlu_parts = []
    dld_parts = []
    for s0 in range(0, nz, chunk):
        s1 = min(s0 + chunk, nz)
        dlu_c, dld_c = _boulac_dlu_dld_chunk(
            s0, s1, beta=beta, nz=nz, theta=theta, dz=dz, qtke=qtke,
            zw=zw, zw_top=zw_top, theta_jp1_row=theta_jp1_row,
            theta_jm1_row=theta_jm1_row, dz_jm1_row=dz_jm1_row, j_idx=j_idx,
        )
        dlu_parts.append(dlu_c)
        dld_parts.append(dld_c)
    dlu = jnp.concatenate(dlu_parts, axis=-1) if len(dlu_parts) > 1 else dlu_parts[0]
    dld = jnp.concatenate(dld_parts, axis=-1) if len(dld_parts) > 1 else dld_parts[0]

    # Shared post-processing (verbatim dense form).
    dld = jnp.minimum(dld, zw_kp1)
    dlu = jnp.maximum(0.1, dlu / (1.0 + dlu / NL_BOULAC_LMAX))
    dld = jnp.maximum(0.1, dld / (1.0 + dld / NL_BOULAC_LMAX))

    lb1 = jnp.minimum(dlu, dld)
    lb2 = jnp.sqrt(dlu * dld)
    lb1 = jnp.concatenate((lb1[..., :-1], lb1[..., -2:-1]), axis=-1)
    lb2 = jnp.concatenate((lb2[..., :-1], lb2[..., -2:-1]), axis=-1)
    if _MYNN_BOULAC_FP32 and _out_dtype == jnp.float64:
        lb1 = lb1.astype(_out_dtype)
        lb2 = lb2.astype(_out_dtype)
    return lb1, lb2


def _boulac_length_onz_production(zw, dz, qtke, theta):
    """Production ``GPUWRF_MYNN_BOULAC_ONZ`` path.

    Uses the bit-identical source-chunked dense schedule with ``chunk=1``.  The
    legacy pure scan `_boulac_length_onz` is still available for diagnostics via
    ``GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN=1``, but it is not shippable in the full
    forecast jit because of the documented XLA/ptxas pathology.
    """

    return _boulac_length_chunked(zw, dz, qtke, theta, chunk=1)


def _boulac_length(zw, dz, qtke, theta):
    """Dispatch the BouLac length scale: dense (default), chunked, or O(nz).

    The dense form is the default and stays untouched.  ``GPUWRF_MYNN_BOULAC_ONZ``
    is the v0.17 production O(nz)-working-set path: source-chunked dense with
    chunk=1, bit-identical to dense in fp64 and measured to fit the 147k-column
    1km case.  ``GPUWRF_MYNN_BOULAC_CHUNKED=1`` keeps the older tunable chunked
    path.  ``GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN=1`` re-enables the rejected pure
    scan implementation for pathology characterization only.
    """

    if _MYNN_BOULAC_ONZ:
        if _MYNN_BOULAC_ONZ_LEGACY_SCAN:
            return _boulac_length_onz(zw, dz, qtke, theta)
        return _boulac_length_onz_production(zw, dz, qtke, theta)
    if _MYNN_BOULAC_CHUNKED:
        return _boulac_length_chunked(zw, dz, qtke, theta)
    return _boulac_length_dense(zw, dz, qtke, theta)


def _mym_length_option1(state: MynnPBLColumnState, qke, dtv, fltv, ustar, dx, xland=1.0):
    """WRF NONLOCAL master length scale (``bl_mynn_mixlength==1``).

    Faithful transcription of the ``CASE (1)`` branch of WRF
    ``module_bl_mynnedmf.F:mym_length`` (lines 1753-1870). This is the WRF v4.7.1
    default and the option the v0.9.0 oracle run used (namelist.output
    ``BL_MYNN_MIXLENGTH=1``). EDMF/downdraft terms are off in the operational
    config so ``qkw_mf=0`` and the ``alp6*qkw_mf`` terms drop out.

    The blend differs from option 2 in two load-bearing ways that drove the
    ~0.37x el_pbl / ~0.33x Kh under-mixing when option 2 was used by mistake:

    * the in-PBL master length is ``el = min( sqrt(els^2/(1+els^2/elt^2)),
      elb, elf )`` (buoyancy-limited via ``elb``/``elf``, not the option-2
      ``els^2/(1+els^2/elt^2+els^2/elb_mf^2)`` harmonic form), and
    * the free atmosphere uses the **BouLac** length:
      ``el = el*(1-wt) + alp5*elBLavg*wt`` — so el stays finite (~2 m) aloft
      instead of collapsing toward zero.

    ``els`` uses the same stability-dependent (rmol) surface-layer form as
    option 2, with ``rmol`` recomputed exactly as the MYNN driver does (line 879)
    from ``fltv``/``ust``. ``dx`` feeds the scale-aware ``Psig_bl`` taper.
    """

    pblh = _get_pblh(state, qke)
    rmol = -KARMAN * GTR * fltv / jnp.maximum(ustar ** 3, 1.0e-6)
    psig_bl = _scale_aware_psig_bl(dx, pblh)
    return _mym_length_option1_with(
        state, qke, dtv, fltv, ustar, dx, xland, pblh=pblh, psig_bl=psig_bl, rmol=rmol
    )


def _mym_length_option1_with(
    state: MynnPBLColumnState, qke, dtv, fltv, ustar, dx, xland, *, pblh, psig_bl, rmol
):
    """``mym_length`` CASE(1) body with caller-supplied ``zi``/``Psig_bl``/``rmol``.

    WRF passes ``zi`` (PBLH) and ``Psig_bl`` INTO ``mym_length`` rather than
    recomputing them per call; on the regular step they come from the same
    current-qke ``GET_PBLH``/``SCALE_AWARE`` so :func:`_mym_length_option1`
    computing them inline is equivalent. ``mym_initialize`` however calls
    ``mym_length`` inside its 5-pass fixed-point iteration with ``zi``/``Psig_bl``
    FROZEN from the pre-iteration taper-seed qke and with the local
    ``flt=fltv=flq=0`` surface terms, so the cold-start path needs this split
    entry to stay faithful.
    """

    dz = state.dz
    zw = _wrf_zw(dz)
    dzk = _interface_dz(dz)
    nz = dz.shape[-1]
    idx = jnp.arange(nz)

    afk_i = dz[..., 1:] / (dz[..., 1:] + dz[..., :-1])
    abk_i = 1.0 - afk_i
    qkw_i = jnp.sqrt(jnp.maximum(qke[..., 1:] * abk_i + qke[..., :-1] * afk_i, QKEMIN))
    qkw = jnp.concatenate((jnp.sqrt(jnp.maximum(qke[..., :1], QKEMIN)), qkw_i), axis=-1)
    # CASE(1) qtke: max(0.5*qkw^2, 0.005) for k>=1; surface = max(0.5*qke, 0.5*qkemin)
    qtke = jnp.concatenate(
        (jnp.maximum(0.5 * qke[..., :1], 0.5 * QKEMIN),
         jnp.maximum(0.5 * qkw_i * qkw_i, 0.005)),
        axis=-1,
    )
    # thetaw: theta at full-sigma levels (theta(k)*abk + theta(k-1)*afk).
    thetaw_i = state.theta[..., 1:] * abk_i + state.theta[..., :-1] * afk_i
    thetaw = jnp.concatenate((state.theta[..., :1], thetaw_i), axis=-1)

    # hurricane-shear tapers (wt_u1/wt_u2); =1.0/=1.0 below the 20 m/s onset.
    # WRF lines 1758-1759: wt_u2 always scales alp3 (buoyancy enhancement) on
    # land AND water; wt_u1 only tapers el(k) over water (lines 1859-1861).
    ugrid = jnp.sqrt(state.u[..., 0] ** 2 + state.v[..., 0] ** 2)
    over_onset = jnp.minimum(1.0, jnp.maximum(0.0, ugrid - NL_UONSET) / 50.0)
    wt_u1 = 1.0 - 0.2 * over_onset
    wt_u2 = 1.0 - 0.4 * over_onset
    alp3 = NL_ALP3 * wt_u2
    # WRF land/sea branch (xland 1=land, 2=water). is_water = (xland-1.5)>=0.
    xland_col = jnp.broadcast_to(jnp.asarray(xland, dtype=qke.dtype), ugrid.shape)
    is_water = (xland_col - 1.5) >= 0.0

    pblh2 = jnp.maximum(pblh, MIN_PBLH)
    h1 = jnp.minimum(jnp.maximum(0.3 * pblh2, 300.0), MAX_PBLH_TRANSITION)
    h2 = 0.5 * h1

    # elt: qkw-weighted PBL-depth length integral (note alp1=0.23, floor qdz=0.01).
    integrate_mask = (idx >= 1) & (zw <= (pblh2 + h1)[..., None])
    qdz = jnp.minimum(jnp.maximum(qkw - QMIN, 0.01), 30.0) * dzk
    elt_num = 1.0e-5 + jnp.sum(jnp.where(integrate_mask, qdz * zw, 0.0), axis=-1)
    elt_den = 1.0e-5 + jnp.sum(jnp.where(integrate_mask, qdz, 0.0), axis=-1)
    # WRF lines 1801-1805: elt_max is land/water-dependent. Over WATER (the
    # dominant Canary/marine surface) WRF caps the PBL-depth length integral at
    # 350 m (+ a hurricane ugrid>50 m/s enhancement to <=450 m); over LAND at
    # 400 m. The previous land-only 400 m cap left el ~14% too long over water,
    # the dominant clear-sky entrainment-zone exch_h miss.
    elt_max_water = NL_ELT_MAX_WATER + 100.0 * jnp.minimum(
        1.0, jnp.maximum(0.0, ugrid - 50.0) / 25.0
    )
    elt_max = jnp.where(is_water, elt_max_water, NL_ELT_MAX)
    elt = jnp.minimum(jnp.maximum(NL_ALP1 * elt_num / elt_den, NL_ELT_MIN), elt_max)
    vsc = (GTR * elt * jnp.maximum(fltv, 0.0)) ** (1.0 / 3.0)

    # BouLac free-atmosphere length (elBLavg = lb2).
    _lb1, elblavg = _boulac_length(zw, dz, qtke, thetaw)

    bv = jnp.sqrt(jnp.maximum(GTR * dtv, 0.0))
    bv = jnp.maximum(bv, 0.001)
    stable = dtv > 0.0
    qkw_elb = jnp.maximum(qkw, NL_QKW_ELB_MIN)
    elb_stable = jnp.minimum(
        (NL_ALP2 * qkw_elb / bv) * (1.0 + alp3[..., None] * jnp.sqrt(vsc[..., None] / jnp.maximum(bv * elt[..., None], 1.0e-12))),
        zw,
    )
    elf_stable = qkw_elb / bv  # one*max(qkw,qkw_elb_min)/bv ; elb_mf=0
    # unstable -> elb=elf=1e10 so the min() picks els-based length.
    elb = jnp.where(stable, elb_stable, 1.0e10)
    elf = jnp.where(stable, elf_stable, 1.0e10)

    # surface-layer length els (rmol-dependent, WRF lines 1842-1846).
    zwrmol = zw * rmol[..., None]
    els_stable = KARMAN * zw / (1.0 + NL_CNS * jnp.minimum(zwrmol, ZMAX))
    els_unstable = KARMAN * zw * jnp.maximum(1.0 - NL_ALP4 * zwrmol, 0.0) ** 0.2
    els = jnp.where(rmol[..., None] > 0.0, els_stable, els_unstable)

    wt = 0.5 * jnp.tanh((zw - (pblh2 + h1)[..., None]) / h2[..., None]) + 0.5
    el = jnp.sqrt((els * els) / (1.0 + (els * els) / (elt[..., None] * elt[..., None])))
    el = jnp.minimum(el, elb)
    el = jnp.minimum(el, elf)
    # WRF lines 1859-1861: the wt_u1 hurricane taper on el(k) is applied OVER
    # WATER ONLY. Over land WRF leaves el untouched here. (For U<=20 m/s wt_u1=1
    # so this is a no-op at the validation step, but the branch is now faithful.)
    el = jnp.where(is_water[..., None], el * wt_u1[..., None], el)
    el = el * (1.0 - wt) + NL_ALP5 * elblavg * wt

    # scale-aware Psig_bl taper (WRF lines 2008-2011; ~1 on coarse grids).
    el_les = 0.25 * dzk
    el = el * psig_bl[..., None] + (1.0 - psig_bl[..., None]) * jnp.minimum(el_les, el)

    el = jnp.where(idx == 0, 0.0, el)
    return qkw, el, pblh


def mynn_coldstart_init_columns(
    state: MynnPBLColumnState, ustar, dx, xland, rmol_init=None
) -> tuple[jax.Array, jax.Array]:
    """WRF MYNN first-call turbulence-state initialization (cold start).

    Faithful transcription of the ``initflag>0 .and. .not.restart`` /
    ``INITIALIZE_QKE=.TRUE.`` path of ``module_bl_mynnedmf.F`` (driver block
    lines 618-747 plus ``mym_initialize`` lines 1281-1430):

    1. driver pre-seed ``qke(k)=5*ust*MAX((ust*700-zw(k))/(MAX(ust,0.01)*700),
       0.01)`` (kts..kte) — used ONLY for the initial ``GET_PBLH``;
    2. ``zi=GET_PBLH(thv, qke_seed)`` and ``Psig_bl=SCALE_AWARE(dx, zi)``, both
       FROZEN through the iteration;
    3. ``mym_level2`` stability functions from the initial profiles (frozen);
    4. pre-iteration ``qke(kts)=1.5*ust^2*(b1*pmz)^(2/3)`` (``pmz=1``), tapered
       above with the same ``MAX(...,0.01)`` form;
    5. ``lmax=5`` fixed-point passes: ``mym_length`` (with the local
       ``flt=fltv=flq=0`` and the WRF-effective ``rmol=0``; see note) ->
       ``pdk=el*qkw*(sm*gm+sh*gh)`` -> level-2 equilibrium
       ``qke(k)=MIN(MAX(b1l*(pdk(k+1)+pdk(k)),qkemin),125)^(2/3)`` for
       kts+1..kte-1 and ``qke(kts)=MAX(ust,0.02)^2*(b1*elv)^(2/3)``;
    6. final ``qke(kts)=0.5*(qke(kts)+qke(kts+1))``; ``qke(kte)=qke(kte-1)``.

    In statically unstable initial layers (``sh*gh>0`` large) the level-2
    equilibrium produces O(0.1-10 m^2/s^2) qke — orders of magnitude above the
    old taper-only seed — which is what lets WRF's step-1 MYNN mix out unstable
    nest-interpolated profiles. ``rmol`` note: WRF's ``mynnedmf`` computes its
    local ``rmol`` only AFTER the init block (line 879), so ``mym_initialize``
    receives the never-assigned local; with gfortran's zeroed first-touch stack
    this is 0.0 (neutral ``els``), verified against the v014 WRF driver hook.

    ``state`` is the MYNN column view ``(..., nz)``; ``ustar``/``xland`` are
    ``(...)`` per-column scalars; ``dx`` is the scalar grid spacing (m).
    ``rmol_init`` overrides the per-column ``rmol`` seen by the iteration's
    ``mym_length`` calls (proof/oracle use only — reproducing an emitted WRF
    stack value); production leaves it ``None`` (0.0).
    Returns ``(qke, pblh)``.
    """

    ust = jnp.asarray(ustar, dtype=state.theta.dtype)
    zw = _wrf_zw(state.dz)
    taper = jnp.maximum(
        (ust[..., None] * 700.0 - zw) / (jnp.maximum(ust[..., None], 0.01) * 700.0), 0.01
    )
    qke = 5.0 * ust[..., None] * taper
    pblh = _get_pblh(state, qke)
    psig_bl = _scale_aware_psig_bl(dx, pblh)
    _dtl, _dqw, dtv, gm, gh, sm, sh = _mym_level2(state)

    # mym_initialize preliminary qke (overwrites the driver pre-seed; NO ust
    # floor on the surface value here — WRF uses raw ust^2 at this point).
    qke0 = 1.5 * ust * ust * (B1 * 1.0) ** (2.0 / 3.0)
    qke = jnp.concatenate((qke0[..., None], qke0[..., None] * taper[..., 1:]), axis=-1)

    fltv0 = jnp.zeros_like(ust)
    rmol0 = jnp.zeros_like(ust) if rmol_init is None else jnp.asarray(rmol_init, dtype=ust.dtype)
    vkz = KARMAN * 0.5 * state.dz[..., 0]
    for _ in range(5):  # WRF lmax = 5
        qkw, el, _pblh_unused = _mym_length_option1_with(
            state, qke, dtv, fltv0, ust, dx, xland,
            pblh=pblh, psig_bl=psig_bl, rmol=rmol0,
        )
        pdk = el * qkw * (sm * gm + sh * gh)
        elv = 0.5 * (el[..., 1] + el[..., 0]) / vkz
        qke_sfc = jnp.maximum(ust, 0.02) ** 2 * (B1 * 1.0 * elv) ** (2.0 / 3.0)
        b1l = B1 * 0.25 * (el[..., 2:] + el[..., 1:-1])
        tmpq = jnp.clip(b1l * (pdk[..., 2:] + pdk[..., 1:-1]), QKEMIN, 125.0)
        qke = jnp.concatenate(
            (qke_sfc[..., None], tmpq ** (2.0 / 3.0), qke[..., -1:]), axis=-1
        )
    qke_kts = 0.5 * (qke[..., 0] + qke[..., 1])
    qke = jnp.concatenate((qke_kts[..., None], qke[..., 1:-1], qke[..., -2:-1]), axis=-1)
    return qke, pblh


def _mym_length_option2(state: MynnPBLColumnState, qke, dtv, fltv, ustar, dx):
    """WRF option-2 MYNN master length scale (``bl_mynn_mixlength==2``).

    Faithful transcription of the ``CASE (2)`` branch of WRF
    ``module_bl_mynnedmf.F:mym_length`` (lines 1872-2011), EDMF/cloud terms off
    (``qkw_mf=0`` -> ``alp6*qkw_mf`` drops out of every ``max``/``min``). The
    master length blends three scales:

    * ``els`` - surface-layer length, **stability-dependent via the inverse
      Monin-Obukhov length** ``rmol`` (WRF lines 1990-1994). WRF recomputes
      ``rmol = -karman*gtr*fltv/max(ust**3,1e-6)`` (line 879) inside the driver
      from the same ``fltv``/``ust`` the surface layer hands MYNN, so we do too.
      In an unstable (daytime) PBL ``rmol<0`` and ``els`` is *larger* than the
      neutral ``karman*z`` — the previously-missing enhancement that drove the
      ~0.37x mixing-length / ~0.33x Kh under-mixing.
    * ``elt`` - PBL-depth length from the qke-weighted height integral.
    * ``elb_mf`` - buoyancy-limited length.

    The squared harmonic blend ``el = sqrt(els^2/(1 + els^2/elt^2 +
    els^2/elb_mf^2))`` then transitions to the free-atmosphere ``elf`` via the
    ``wt`` tanh weight, and finally the scale-aware ``Psig_bl`` taper (WRF
    lines 2008-2011, SCALE_AWARE line 7512) limits ``el`` toward ``0.25*dzk`` on
    grids fine enough to resolve the eddies. ``dx`` is the grid spacing (m).
    """

    dz = state.dz
    zw = _wrf_zw(dz)
    dzk = _interface_dz(dz)
    nz = dz.shape[-1]
    idx = jnp.arange(nz)
    afk_i = dz[..., 1:] / (dz[..., 1:] + dz[..., :-1])
    abk_i = 1.0 - afk_i
    qkw_i = jnp.sqrt(jnp.maximum(qke[..., 1:] * abk_i + qke[..., :-1] * afk_i, QKEMIN))
    qkw = jnp.concatenate((jnp.sqrt(jnp.maximum(qke[..., :1], QKEMIN)), qkw_i), axis=-1)
    qtke = jnp.concatenate((jnp.maximum(0.5 * qke[..., :1], 0.5 * QKEMIN), 0.5 * qkw_i * qkw_i), axis=-1)

    pblh = _get_pblh(state, qke)
    pblh2 = jnp.maximum(pblh, MIN_PBLH)
    h1 = jnp.minimum(jnp.maximum(0.3 * pblh2, 300.0), MAX_PBLH_TRANSITION)
    h2 = 0.5 * h1
    pblh_plus_ent = jnp.maximum(pblh + h1, 100.0)

    integrate_mask = (idx >= 1) & (zw <= pblh_plus_ent[..., None])
    qdz = jnp.minimum(jnp.maximum(qkw - QMIN, 0.03), 30.0) * dzk
    elt_num = 1.0e-5 + jnp.sum(jnp.where(integrate_mask, qdz * zw, 0.0), axis=-1)
    elt_den = 1.0e-5 + jnp.sum(jnp.where(integrate_mask, qdz, 0.0), axis=-1)
    elt = jnp.minimum(jnp.maximum(LOCAL_ALP1 * elt_num / elt_den, LOCAL_ELT_MIN), LOCAL_ELT_MAX)
    vsc = (GTR * elt * jnp.maximum(fltv, 0.0)) ** (1.0 / 3.0)

    bv = jnp.maximum(jnp.sqrt(jnp.maximum(GTR * dtv, 0.0)), 0.001)
    stable = dtv > 0.0
    elb_mf_stable = (LOCAL_ALP2 * qkw / bv) * (1.0 + LOCAL_ALP3 * jnp.sqrt(vsc[..., None] / jnp.maximum(bv * elt[..., None], 1.0e-12)))
    wstar_stable = 1.25 * (GTR * pblh[..., None] * jnp.maximum(fltv[..., None], 1.0e-4)) ** (1.0 / 3.0)
    wt = 0.5 * jnp.tanh((zw - (pblh2 + h1)[..., None]) / h2[..., None]) + 0.5
    tau_stable = jnp.minimum(jnp.maximum(CTAU * wstar_stable / GRAV, 30.0), 150.0)
    tau_stable = tau_stable * (1.0 - wt) + 50.0 * wt
    elf_stable = jnp.minimum(tau_stable * jnp.sqrt(jnp.minimum(qtke, 40.0)), zw)

    wstar_unstable = wstar_stable
    tau_unstable = jnp.minimum(jnp.maximum(CTAU * wstar_unstable / GRAV, 50.0), 200.0)
    tau_unstable = tau_unstable * (1.0 - wt) + jnp.maximum(100.0, 0.25 * dzk) * wt
    elb_unstable = jnp.minimum(tau_unstable * jnp.sqrt(jnp.minimum(qtke, 40.0)), zw)
    elf_unstable = elb_unstable

    elb_mf = jnp.maximum(jnp.where(stable, elb_mf_stable, elb_unstable), 0.01)
    elf = jnp.where(stable, elf_stable, elf_unstable)
    elf = elf / (1.0 + elf / LOCAL_ELF_SOFT_MAX)

    # Surface-layer length (WRF lines 1990-1994). rmol is recomputed exactly as
    # the MYNN driver does (line 879) from the surface buoyancy flux + ustar.
    rmol = -KARMAN * GTR * fltv / jnp.maximum(ustar ** 3, 1.0e-6)
    zwrmol = zw * rmol[..., None]
    els_stable = KARMAN * zw / (1.0 + LOCAL_CNS * jnp.minimum(zwrmol, ZMAX))
    # unstable: karman*z*(1 - alp4*z*rmol)**0.2 ; rmol<0 -> base>1 -> els grows
    els_unstable = KARMAN * zw * jnp.maximum(1.0 - LOCAL_ALP4 * zwrmol, 0.0) ** 0.2
    els = jnp.where(rmol[..., None] > 0.0, els_stable, els_unstable)

    el = jnp.sqrt((els * els) / (1.0 + (els * els) / (elt[..., None] * elt[..., None]) + (els * els) / (elb_mf * elb_mf)))
    el = el * (1.0 - wt) + elf * wt

    # Scale-aware taper (WRF SCALE_AWARE + lines 2008-2011). Psig_bl~1 on coarse
    # grids; on fine grids it limits el toward the LES sub-grid scale 0.25*dzk.
    psig_bl = _scale_aware_psig_bl(dx, pblh)
    el_les = 0.25 * dzk
    el = el * psig_bl[..., None] + (1.0 - psig_bl[..., None]) * jnp.minimum(el_les, el)

    el = jnp.where(idx == 0, 0.0, el)
    return qkw, el, pblh


def _mym_turbulence(state: MynnPBLColumnState, qke, fltv, ustar, dx, xland=1.0):
    """WRF MYNN `mym_turbulence` dry level-2.5 path.

    Uses the WRF-default NONLOCAL mixing length (``bl_mynn_mixlength=1``,
    :func:`_mym_length_option1`) — the option the validation oracle ran. The
    local option-2 form (:func:`_mym_length_option2`) is retained for reference
    but is NOT the operational/validated path. ``xland`` (1=land, 2=water) feeds
    the WRF land/water branch of the mixing length."""

    dtl, dqw, dtv, gm, gh, sm20_raw, sh20_raw = _mym_level2(state)
    qkw, el, pblh = _mym_length_option1(state, qke, dtv, fltv, ustar, dx, xland)
    dzk = _interface_dz(state.dz)
    elsq = el * el
    q3sq_initial = qkw * qkw
    q2sq = B1 * elsq * (sm20_raw * gm + sh20_raw * gh)
    sh = jnp.maximum(sh20_raw, 1.0e-5)
    sm = sm20_raw

    ri = -gh / jnp.maximum(gm, 1.0e-10)
    a2fac = jnp.where(CKMOD == 1.0, 1.0 / (1.0 + jnp.maximum(ri, 0.0)), 1.0)
    prlim = jnp.where(
        ri >= 1.0,
        7.0 * ri,
        jnp.where((ri >= 0.01) & (ri <= 1.0), 6.873 * ri + 1.0 / (6.873 * ri), 5.0),
    )
    gmel = gm * elsq
    ghel = gh * elsq
    q3sq = jnp.where((elsq > 0.0) & ((q3sq_initial / jnp.maximum(elsq, 1.0e-20)) < -gh), -elsq * gh, q3sq_initial)
    q3sq = jnp.maximum(q3sq, QKEMIN)

    qdiv = jnp.sqrt(jnp.maximum(q3sq / jnp.maximum(q2sq, 1.0e-20), 0.0))
    helfand = (q2sq > 0.0) & (q3sq < q2sq)
    sh_hl = sh * qdiv
    sm_hl = sm * qdiv

    e1_hl = q3sq - E1C * ghel * a2fac * qdiv * qdiv
    e2_hl = q3sq - E2C * ghel * a2fac * qdiv * qdiv
    e3_hl = e1_hl + E3C * ghel * a2fac * a2fac * qdiv * qdiv
    e4_hl = e1_hl - E4C * ghel * a2fac * qdiv * qdiv
    eden_hl = jnp.maximum(e2_hl * e4_hl + e3_hl * E5C * gmel * qdiv * qdiv, 1.0e-20)
    del e1_hl, e2_hl, e3_hl, e4_hl, eden_hl

    e1 = q3sq - E1C * ghel * a2fac
    e2 = q3sq - E2C * ghel * a2fac
    e3 = e1 + E3C * ghel * a2fac * a2fac
    e4 = e1 - E4C * ghel * a2fac
    eden = jnp.maximum(e2 * e4 + e3 * E5C * gmel, 1.0e-20)
    sm25 = q3sq * A1 * (e3 - 3.0 * C1 * e4) / eden
    sh25 = q3sq * (A2 * a2fac) * (e2 + 3.0 * C1 * E5C * gmel) / eden
    sm = jnp.where(helfand, sm_hl, sm25)
    sh = jnp.where(helfand, sh_hl, sh25)
    sh = jnp.minimum(jnp.maximum(sh, 0.0), 4.0)
    sm = jnp.minimum(sm, prlim * jnp.maximum(sh, 0.02))
    sm = jnp.maximum(sm, 0.0)

    elq = el * qkw
    pdk = elq * (sm * gm + sh * gh)
    pdt = elq * sh * dtl * dtl
    pdq = elq * sh * dqw * dqw
    pdc = elq * sh * dtl * dqw
    tcd = _zero_like(pdk)
    qcd = _zero_like(pdk)
    dfm = jnp.where(dzk > 0.0, elq * sm / dzk, 0.0)
    dfh = jnp.where(dzk > 0.0, elq * sh / dzk, 0.0)
    dfq = dfm
    qshear = elq * sm * gm
    qbuoy = elq * sh * gh
    zeros = _zero_edge_like(state.tke)
    dfm = jnp.concatenate((zeros, dfm[..., 1:]), axis=-1)
    dfh = jnp.concatenate((zeros, dfh[..., 1:]), axis=-1)
    dfq = jnp.concatenate((zeros, dfq[..., 1:]), axis=-1)
    pdk = jnp.concatenate((zeros, pdk[..., 1:]), axis=-1)
    qshear = jnp.concatenate((zeros, qshear[..., 1:]), axis=-1)
    qbuoy = jnp.concatenate((zeros, qbuoy[..., 1:]), axis=-1)
    return {
        "qkw": qkw,
        "el": el,
        "pblh": pblh,
        "dfm": dfm,
        "dfh": dfh,
        "dfq": dfq,
        "pdk": pdk,
        "pdt": pdt,
        "pdq": pdq,
        "pdc": pdc,
        "tcd": tcd,
        "qcd": qcd,
        "qshear": qshear,
        "qbuoy": qbuoy,
    }


def _rho_interfaces(state: MynnPBLColumnState, diffusivity):
    """Builds WRF rho-weighted interface diffusion factors."""

    dz = state.dz
    rho = state.rho
    rhoz_i = (rho[..., 1:] * dz[..., :-1] + rho[..., :-1] * dz[..., 1:]) / (dz[..., :-1] + dz[..., 1:])
    rhoz_i = jnp.maximum(rhoz_i, 1.0e-4)
    rhoz = jnp.concatenate((rho[..., :1], rhoz_i, rhoz_i[..., -1:]), axis=-1)
    diff_ext = jnp.concatenate((diffusivity, diffusivity[..., -1:]), axis=-1)
    return rhoz * diff_ext


def _solve_tridiagonal(a, b, c, d):
    """Uses XLA's tridiagonal primitive for the production vertical solves."""

    return solve_tridiagonal(a, b, c, d)


def _wrf_qke_minmax(value):
    """WRF ``qke=max(x,qkemin); qke=min(qke,150)`` post-solve bounds.

    WRF applies Fortran ``MAX``/``MIN`` immediately after the MYNN ``tridiag2``
    solve (``module_bl_mynnedmf.F:3106-3107`` in the local pristine v4 source).
    Under the common Fortran comparison semantics used by WRF builds, a NaN
    solver element compares false and the finite qkemin/upper bound is selected.
    ``jnp.maximum`` propagates NaN, so use IEEE fmax/fmin to match the WRF
    intrinsic behavior while leaving every finite value's bounds unchanged.
    """

    return jnp.fmin(jnp.fmax(value, QKEMIN), 150.0)


def _phim_puhales(zet):
    """WRF Puhales-2020 momentum stability function ``phim(zet)`` (default
    ``bl_mynn_stfunc=1``, ``module_bl_mynnedmf.F:7701-7750``).

    Stable (``zet>=0``): Cheng & Brutsaert (2005) form valid to very stable z/L.
    Unstable (``zet<0``): Grachev-et-al-2000 convective blend. ``phim`` returns
    ``phi_m`` (the ``-zet`` subtraction is applied by the caller as ``pmz``)."""

    am_st = 6.1
    bm_st = 2.5
    rbm_st = 1.0 / bm_st
    am_unst = 10.0
    # ---- stable branch (zet>=0) ----
    zet_s = jnp.maximum(zet, 0.0)
    d0_s = 1.0 + zet_s ** bm_st
    d1_s = zet_s + d0_s ** rbm_st
    d11_s = 1.0 + d0_s ** (rbm_st - 1.0) * zet_s ** (bm_st - 1.0)
    d2_s = (-am_st / d1_s) * d11_s
    phi_m_st = 1.0 - zet_s * d2_s
    # ---- unstable branch (zet<0) ----
    zet_u = jnp.minimum(zet, -1.0e-12)  # keep strictly negative for the 1/zet terms
    dum0 = (1.0 - CPHM_UNST * zet_u) ** 0.25
    phi_m0 = 1.0 / dum0
    dpsi = (2.0 * jnp.log(0.5 * (1.0 + dum0)) + jnp.log(0.5 * (1.0 + dum0 * dum0))
            - 2.0 * jnp.arctan(dum0) + 1.570796)
    a0 = 1.0 - am_unst * zet_u
    y = a0 ** 0.333333
    dydz = -0.33333 * am_unst * a0 ** (-0.6666667)
    f = 0.33333 * (y * y + y + 1.0)
    dfdz = 0.3333 * dydz * (2.0 * y + 1.0)
    g = 0.57735 * (2.0 * y + 1.0)
    dgdz = 1.1547 * dydz
    psic = 1.5 * jnp.log(f) - 1.73205 * jnp.arctan(g) + 1.813799364
    dpsic = (1.5 / f) * dfdz - 1.73205 * dgdz / (1.0 + g * g)
    z2 = zet_u * zet_u
    denon = 1.0 / (1.0 + z2)
    ddenon = 2.0 * zet_u
    term1 = ((1.0 - phi_m0) / zet_u + ddenon * psic + z2 * dpsic) * denon
    term2 = -ddenon * (dpsi + z2 * psic) * denon * denon
    phi_m_unst = 1.0 - zet_u * (term1 + term2)
    return jnp.where(zet >= 0.0, phi_m_st, phi_m_unst)


def _pmz_surface(fltv, ustar, dz0):
    """WRF surface non-dimensional shear ``pmz`` for the TKE surface production
    (``module_bl_mynnedmf.F:879-897``, ``bl_mynn_stfunc=1`` default).

    ``rmol = -karman*gtr*fltv/max(ust**3,1e-6)`` (line 879); ``zet=0.5*dz1*rmol``
    clamped to [-20,20]; ``pmz = phim(zet) - zet``. In stable surface layers
    (``fltv<0`` -> ``zet>0``) ``pmz>1`` enhances the surface TKE source — the
    factor the previous kernel dropped (it assumed ``pmz=1``), which collapsed the
    surface qke over the stable nighttime/evening land columns."""

    rmol = -KARMAN * GTR * fltv / jnp.maximum(ustar ** 3, 1.0e-6)
    zet = jnp.clip(0.5 * dz0 * rmol, -20.0, 20.0)
    return _phim_puhales(zet) - zet


def _mym_predict_qke(state: MynnPBLColumnState, qke, turb, dt, ustar, flux):
    """WRF `mym_predict` dry level-2.5 qke equation."""

    dz = state.dz
    rho = state.rho
    dtz = dt / dz
    rhoinv = 1.0 / jnp.maximum(rho, 1.0e-4)
    qkw_mass = jnp.sqrt(jnp.maximum(qke, 0.0))
    df3q = SQFAC * turb["dfq"]
    kqdz = _rho_interfaces(state, df3q)
    vkz = KARMAN * 0.5 * dz[..., 0]
    # WRF line 3072: pdk1 = 2*ust**3*pmz/vkz. pmz is the surface non-dimensional
    # shear from the (Puhales-2020) stability function; pmz>1 in stable layers.
    pmz = _pmz_surface(flux.fltv, ustar, dz[..., 0])
    pdk1 = 2.0 * ustar * ustar * ustar * pmz / jnp.maximum(vkz, 1.0e-6)
    pdk = jnp.concatenate(((pdk1 - turb["pdk"][..., 1])[..., None], turb["pdk"][..., 1:]), axis=-1)

    el_pair = 0.5 * (turb["el"][..., 1:] + turb["el"][..., :-1])
    b1l = jnp.maximum(B1 * el_pair, 1.0e-6)
    bp_i = 2.0 * qkw_mass[..., :-1] / b1l
    rp_i = pdk[..., 1:] + pdk[..., :-1]
    lower = -dtz[..., :-1] * kqdz[..., :-2] * rhoinv[..., :-1]
    diag = 1.0 + dtz[..., :-1] * (kqdz[..., :-2] + kqdz[..., 1:-1]) * rhoinv[..., :-1] + bp_i * dt
    upper = -dtz[..., :-1] * kqdz[..., 1:-1] * rhoinv[..., :-1]
    rhs = rp_i * dt + qke[..., :-1]

    a = jnp.concatenate((lower, _zero_edge_like(qke)), axis=-1)
    b = jnp.concatenate((diag, jnp.ones_like(qke[..., -1:])), axis=-1)
    c = jnp.concatenate((upper, _zero_edge_like(qke)), axis=-1)
    d = jnp.concatenate((rhs, qke[..., -1:]), axis=-1)
    qke_new = _wrf_qke_minmax(_solve_tridiagonal(a, b, c, d))

    tke_up = 0.5 * qke_new
    qwt_bottom = (kqdz[..., 1] * (tke_up[..., 1] - tke_up[..., 0]) - kqdz[..., 0] * tke_up[..., 0]) / dz[..., 0]
    qwt_mid = (
        kqdz[..., 2:-1] * (tke_up[..., 2:] - tke_up[..., 1:-1])
        - kqdz[..., 1:-2] * (tke_up[..., 1:-1] - tke_up[..., :-2])
    ) / dz[..., 1:-1]
    qwt_top = (-kqdz[..., -2] * (tke_up[..., -1] - tke_up[..., -2])) / dz[..., -1]
    qwt = jnp.concatenate((qwt_bottom[..., None], qwt_mid, qwt_top[..., None]), axis=-1)
    bp = jnp.concatenate((bp_i, _zero_edge_like(qke)), axis=-1)
    qdiss = bp * tke_up
    return qke_new, qwt, qdiss, pdk


def _mym_predict_qsq(state: MynnPBLColumnState, qsq, turb, dt, s_aw=None):
    """WRF ``mym_predict`` closure-2.6 prognostic total-water variance ``qsq``.

    Transcribes module_bl_mynnedmf.F mym_predict ``IF (closure > 2.5)``
    (lines ~3185-3220 of the subroutine): implicit vertical diffusion with
    ``kmdz = rhoz*dfq`` (incl. the JOE conservation/MF stability floors when the
    EDMF ``s_aw`` is active), production ``rp = pdq(k+1)+pdq(k)`` with the WRF
    surface row ``pdq(kts)=pdq(kts+1)``, dissipation ``bp = 2*qkw/(b2*0.5*
    (el(k+1)+el(k)))``, zero-gradient top (``a(kte)=-1``), and the post-solve
    ``qsq = MAX(x, 1e-17)`` floor.  ``qsq`` feeds the mym_condensation CASE(2)
    sigma at the NEXT step (WRF call order: condensation before predict).
    """

    dz = state.dz
    dtz = dt / dz
    rhoinv = 1.0 / jnp.maximum(state.rho, 1.0e-4)
    qkw_mass = jnp.sqrt(jnp.maximum(2.0 * state.tke, 0.0))
    kmdz = _rho_interfaces(state, turb["dfq"])
    if s_aw is not None:
        kmdz = _apply_s_aw_stability_floor(kmdz, s_aw)

    # WRF pdq(kts) = pdq(kts+1) before rp (mym_predict lines 151-153).
    pdq = turb["pdq"]
    pdq = jnp.concatenate((pdq[..., 1:2], pdq[..., 1:]), axis=-1)

    el_pair = 0.5 * (turb["el"][..., 1:] + turb["el"][..., :-1])
    b2l = jnp.maximum(B2 * el_pair, 1.0e-6)
    bp_i = 2.0 * qkw_mass[..., :-1] / b2l
    rp_i = pdq[..., 1:] + pdq[..., :-1]
    lower = -dtz[..., :-1] * kmdz[..., :-2] * rhoinv[..., :-1]
    diag = 1.0 + dtz[..., :-1] * (kmdz[..., :-2] + kmdz[..., 1:-1]) * rhoinv[..., :-1] + bp_i * dt
    upper = -dtz[..., :-1] * kmdz[..., 1:-1] * rhoinv[..., :-1]
    rhs = rp_i * dt + qsq[..., :-1]

    # zero-gradient top row: a(kte)=-1, b=1, c=0, d=0 -> x(kte)=x(kte-1)
    a = jnp.concatenate((lower, -jnp.ones_like(qsq[..., -1:])), axis=-1)
    b = jnp.concatenate((diag, jnp.ones_like(qsq[..., -1:])), axis=-1)
    c = jnp.concatenate((upper, _zero_edge_like(qsq)), axis=-1)
    d = jnp.concatenate((rhs, _zero_edge_like(qsq)), axis=-1)
    return jnp.maximum(_solve_tridiagonal(a, b, c, d), 1.0e-17)


def _apply_s_aw_stability_floor(kdz, s_aw):
    """WRF MYNN-EDMF stability floor on interface K*dz arrays."""

    kdz_int = jnp.maximum(
        jnp.maximum(kdz[..., 1:-1], 0.5 * s_aw[..., 1:-1]),
        -0.5 * (s_aw[..., 1:-1] - s_aw[..., 2:]),
    )
    return jnp.concatenate((kdz[..., :1], kdz_int, kdz[..., -1:]), axis=-1)


def _diffusion_solve_with_surface(x, diffusivity, state, dt, bottom_rhs, bottom_drag=0.0,
                                  s_aw_floor=None):
    """WRF `mynn_tendencies` tridiagonal coefficients for one dry scalar."""

    dz = state.dz
    dtz = dt / dz
    rhoinv = 1.0 / jnp.maximum(state.rho, 1.0e-4)
    kdz = _rho_interfaces(state, diffusivity)
    if s_aw_floor is not None:
        kdz = _apply_s_aw_stability_floor(kdz, s_aw_floor)
    bottom_drag = jnp.zeros_like(bottom_rhs) + bottom_drag
    lower0 = -dtz[..., :1] * kdz[..., :1] * rhoinv[..., :1]
    diag0 = 1.0 + dtz[..., :1] * (kdz[..., 1:2] + kdz[..., :1] + bottom_drag[..., None]) * rhoinv[..., :1]
    upper0 = -dtz[..., :1] * kdz[..., 1:2] * rhoinv[..., :1]
    rhs0 = x[..., :1] + bottom_rhs[..., None]

    lower_i = -dtz[..., 1:-1] * kdz[..., 1:-2] * rhoinv[..., 1:-1]
    diag_i = 1.0 + dtz[..., 1:-1] * (kdz[..., 1:-2] + kdz[..., 2:-1]) * rhoinv[..., 1:-1]
    upper_i = -dtz[..., 1:-1] * kdz[..., 2:-1] * rhoinv[..., 1:-1]
    rhs_i = x[..., 1:-1]

    a = jnp.concatenate((lower0, lower_i, _zero_edge_like(x)), axis=-1)
    b = jnp.concatenate((diag0, diag_i, jnp.ones_like(x[..., -1:])), axis=-1)
    c = jnp.concatenate((upper0, upper_i, _zero_edge_like(x)), axis=-1)
    d = jnp.concatenate((rhs0, rhs_i, x[..., -1:]), axis=-1)
    return _solve_tridiagonal(a, b, c, d)


def _diffusion_solve_with_mf(x, diffusivity, state, dt, bottom_rhs, s_aw, s_awx,
                             bottom_drag=0.0):
    """WRF `mynn_tendencies` scalar solve WITH the MYNN-EDMF mass-flux terms.

    Faithful to the qv/thl blocks of WRF ``module_bl_mynnedmf.F:4316-4382`` for the
    operational config (no downdraft -> sd_*=0; env_subs=.false. -> sub_*=det_*=0;
    bl_mynn_mixqt=0 -> mix qv separately). The mass-flux additions to the implicit
    tridiagonal (s_aw on the diagonal/off-diagonals) and the explicit
    flux-divergence source (s_awx) are added to the pure-ED coefficients.

    ``s_aw`` and ``s_awx`` are interface-staggered (length nz+1), index k+1.
    They equal the WRF ``s_aw1``/``s_awphi1`` arrays already scaled by Psig_w and
    the flux limiter (see :func:`mynn_edmf.dmp_mf_columns`).
    """
    dz = state.dz
    dtz = dt / dz
    rhoinv = 1.0 / jnp.maximum(state.rho, 1.0e-4)
    kdz = _rho_interfaces(state, diffusivity)  # length nz+1 (khdz on interfaces)

    # WRF stability floor on khdz from the mass flux (lines 3990-3997), sd_aw=0:
    #   khdz(k) = max(khdz(k), 0.5*s_aw(k)); max(khdz(k), -0.5*(s_aw(k)-s_aw(k+1)))
    # for k=kts+1..kte-1. kdz here is indexed as interface k (0..nz). s_aw index k.
    kdz = _apply_s_aw_stability_floor(kdz, s_aw)

    bottom_drag = jnp.zeros_like(bottom_rhs) + bottom_drag

    # ---- k = kts (0-based 0) : WRF lines 4326-4336 (sd/sub/det = 0) ----
    half_dtz0 = 0.5 * dtz[..., :1] * rhoinv[..., :1]
    lower0 = -dtz[..., :1] * kdz[..., :1] * rhoinv[..., :1]
    diag0 = (1.0 + dtz[..., :1] * (kdz[..., 1:2] + kdz[..., :1] + bottom_drag[..., None]) * rhoinv[..., :1]
             - half_dtz0 * s_aw[..., 1:2])
    upper0 = (-dtz[..., :1] * kdz[..., 1:2] * rhoinv[..., :1]
              - half_dtz0 * s_aw[..., 1:2])
    rhs0 = (x[..., :1] + bottom_rhs[..., None]
            - dtz[..., :1] * rhoinv[..., :1] * s_awx[..., 1:2])

    # ---- interior k = kts+1..kte-1 : WRF lines 4338-4352 ----
    dtzi = dtz[..., 1:-1]
    rhoinvi = rhoinv[..., 1:-1]
    half_i = 0.5 * dtzi * rhoinvi
    s_aw_k = s_aw[..., 1:-2]      # s_aw(k)   for interior k -> interface k
    s_aw_kp1 = s_aw[..., 2:-1]    # s_aw(k+1)
    s_awx_k = s_awx[..., 1:-2]
    s_awx_kp1 = s_awx[..., 2:-1]
    lower_i = (-dtzi * kdz[..., 1:-2] * rhoinvi + half_i * s_aw_k)
    diag_i = (1.0 + dtzi * (kdz[..., 1:-2] + kdz[..., 2:-1]) * rhoinvi
              + half_i * (s_aw_k - s_aw_kp1))
    upper_i = (-dtzi * kdz[..., 2:-1] * rhoinvi - half_i * s_aw_kp1)
    rhs_i = x[..., 1:-1] + dtzi * rhoinvi * (s_awx_k - s_awx_kp1)

    a = jnp.concatenate((lower0, lower_i, _zero_edge_like(x)), axis=-1)
    b = jnp.concatenate((diag0, diag_i, jnp.ones_like(x[..., -1:])), axis=-1)
    c = jnp.concatenate((upper0, upper_i, _zero_edge_like(x)), axis=-1)
    d = jnp.concatenate((rhs0, rhs_i, x[..., -1:]), axis=-1)
    return _solve_tridiagonal(a, b, c, d)


def _apply_mean_tendencies(state: MynnPBLColumnState, turb, dt, flux, wind, rhosfc,
                           mf=None):
    """Applies WRF-style U/V/theta/qv implicit tendency solves.

    When ``mf`` (the MYNN-EDMF solver arrays from :func:`mynn_edmf.dmp_mf_columns`)
    is provided, U/V, theta(thl), and qv include the mass-flux nonlocal transport
    (``s_aw`` plus ``s_awu``/``s_awv``/``s_awthl``/``s_awqv``). WRF also applies
    the ``s_aw`` stability floor to ``kmdz``/``khdz`` before the implicit solves.
    """

    rhoinv0 = 1.0 / jnp.maximum(state.rho[..., 0], 1.0e-4)
    dtz0 = dt / state.dz[..., 0]
    bottom_drag = rhosfc * flux.ustar * flux.ustar / wind
    theta_rhs = dtz0 * rhosfc * flux.theta_flux * rhoinv0
    qv_flux = jnp.maximum(flux.qv_flux, jnp.minimum(0.9 * state.qv[..., 0] - 1.0e-8, 0.0) / jnp.maximum(dtz0, 1.0e-12))
    qv_rhs = dtz0 * rhosfc * qv_flux * rhoinv0
    thl0 = _liquid_potential_temperature(state)
    exner = _exner_from_pressure(state.p)
    _sqv, sqc, sqi, _sqw = _specific_moisture_components(state)
    if mf is None:
        u = _diffusion_solve_with_surface(
            state.u, turb["dfm"], state, dt, jnp.zeros_like(wind), bottom_drag)
        v = _diffusion_solve_with_surface(
            state.v, turb["dfm"], state, dt, jnp.zeros_like(wind), bottom_drag)
        thl = _diffusion_solve_with_surface(thl0, turb["dfh"], state, dt, theta_rhs)
        qv = _diffusion_solve_with_surface(state.qv, turb["dfh"], state, dt, qv_rhs)
    else:
        zero_wind_rhs = jnp.zeros_like(wind)
        u = _diffusion_solve_with_mf(
            state.u, turb["dfm"], state, dt, zero_wind_rhs,
            mf["s_aw"], mf["s_awu"], bottom_drag=bottom_drag)
        v = _diffusion_solve_with_mf(
            state.v, turb["dfm"], state, dt, zero_wind_rhs,
            mf["s_aw"], mf["s_awv"], bottom_drag=bottom_drag)
        # WRF solves liquid-water potential temperature (thl) with s_awthl, then
        # converts back to theta after the moisture/cloud solve. This port keeps
        # qc/qi read-only under the frozen B2 write contract, so the conversion
        # uses the input condensate specific contents.
        thl = _diffusion_solve_with_mf(
            thl0, turb["dfh"], state, dt, theta_rhs,
            mf["s_aw"], mf["s_awthl"])
        qv = _diffusion_solve_with_mf(
            state.qv, turb["dfh"], state, dt, qv_rhs,
            mf["s_aw"], mf["s_awqv"])
    theta = thl + XLVCP_MYNN / exner * sqc + XLSCP_MYNN / exner * sqi
    return u, v, theta, jnp.maximum(qv, 0.0)


def _edmf_arrays_from_state(state, flux, fltv, pblh, dt, dx):
    """Build the MYNN-EDMF mass-flux solver arrays from the column state.

    Calls the WRF-faithful :func:`mynn_edmf.dmp_mf_columns` (verified against the
    pristine WRF ``DMP_mf`` to <0.5% rel error, see ``proofs/mynn_edmf``). The
    standalone MYNN column can carry ``qv``/``qc``/``qi``/``theta``; the liquid
    potential temperature and total water arrays passed here mirror WRF's
    moisture-aware MYNN path. The surface flux struct supplies the kinematic
    fluxes WRF's main MYNN derives as
    ``flqv=qfx/rho`` (=qv_flux), ``flt=hfx/(rho*cpm)`` (=theta_flux), ``flq=flqv``.

    ``ts`` (skin temperature) feeds only the superadiabatic activation guard. The
    standalone column does not carry a reliable skin temperature, so we pass the
    ``ts<=0`` sentinel: :func:`mynn_edmf.dmp_mf_columns` then uses the physically
    equivalent buoyancy-flux activation criterion (``fltv>0`` <=> unstable surface
    layer). ``dx`` is the grid spacing (m).
    """
    from gpuwrf.physics import mynn_edmf as _edmf

    sqv, sqc, sqi, sqw = _specific_moisture_components(state)
    theta = state.theta
    thl = _liquid_potential_temperature(state)
    thv = theta * (1.0 + P608 * sqv)
    exner = (state.p / 100000.0) ** (287.0 / (3.5 * 287.0))

    # kinematic surface fluxes (already in flux struct)
    flqv = flux.qv_flux
    flq = flqv
    flt = flux.theta_flux
    ts = -jnp.ones_like(fltv)  # sentinel -> buoyancy-flux activation criterion

    zw = jnp.concatenate(
        (_zero_edge_like(state.dz), jnp.cumsum(state.dz, axis=-1)), axis=-1
    )
    xland = jnp.broadcast_to(
        jnp.asarray(flux.xland, dtype=state.theta.dtype), fltv.shape
    )

    return _edmf.dmp_mf_columns(
        sqw, sqv, sqc, state.u, state.v, state.w, theta, thl, thv, state.theta * 0.0,
        2.0 * state.tke, state.p, exner, state.rho, state.dz, zw,
        ust=flux.ustar, flt=flt, fltv=fltv, flq=flq, flqv=flqv,
        pblh=pblh, ts=ts, dx=dx, xland=xland, dt=dt,
    )


def _retrieve_exchange_coeffs(state: MynnPBLColumnState, turb):
    """WRF `retrieve_exchange_coeffs`: converts `dfm/dfh` back to K units."""

    dzk = _interface_dz(state.dz)
    km = turb["dfm"] * dzk
    kh = turb["dfh"] * dzk
    km = jnp.concatenate((_zero_edge_like(km), km[..., 1:]), axis=-1)
    kh = jnp.concatenate((_zero_edge_like(kh), kh[..., 1:]), axis=-1)
    return km, kh


def _step_mynn_pbl_impl_with_pblh(state: MynnPBLColumnState, dt: float, debug: bool,
                                  surface=None, edmf: bool = False, dx: float = 1000.0):
    """Unjitted implementation; returns the advanced state and the MYNN PBLH.

    ``edmf`` activates the MYNN-EDMF mass-flux nonlocal scalar transport (the
    ``s_awqv``/``s_awthl`` updraft flux) in the theta/qv solves. ``dx`` is the
    horizontal grid spacing (m) the mass-flux plume sizing needs.

    v0.15 SGS-cloud chain (WRF driver order, module_bl_mynnedmf.F:900-1100):
    GET_PBLH -> mym_condensation(CASE 2, sigma from the prognosed ``qsq``) ->
    DMP_mf (+ shallow-cu cldfra/qc_bl overwrite) -> SGS-aware ``thlv`` ->
    mym_turbulence -> mym_predict (qke + closure-2.6 ``qsq``) -> tendencies.
    ``GPUWRF_MYNN_SGS_CLOUD=0`` rolls back to the pre-v0.15 dry-buoyancy path.
    """

    state = _clip_state(state)
    flux, wind, fltv, rhosfc = _surface_terms(state, surface)
    qke = 2.0 * state.tke
    sgs_cloud = _MYNN_SGS_CLOUD
    mf = None
    if sgs_cloud:
        # WRF GET_PBLH runs before mym_condensation; the same value feeds
        # DMP_mf (bitwise-identical to the previous turb["pblh"] EDMF input).
        pblh0 = _get_pblh(state, qke)
        sqv, sqc, sqi, sqw = _specific_moisture_components(state)
        exner = _exner_from_pressure(state.p)
        qc_bl, qi_bl, cldfra_bl = mym_condensation_cloudpdf2(
            theta=state.theta,
            p=state.p,
            exner=exner,
            dz=state.dz,
            qw=sqw,
            qc=sqc,
            qi=sqi,
            qs=_specific_snow_content(state),
            qsq=state.qsq,
            pblh=pblh0,
        )
        if edmf:
            mf = _edmf_arrays_from_state(state, flux, fltv, pblh0, dt, dx)
            qc_bl, cldfra_bl = dmp_shallow_cu_overwrite(
                qc_bl=qc_bl,
                cldfra_bl=cldfra_bl,
                edmf_a=mf["edmf_a"],
                edmf_qc=mf["edmf_qc"],
                edmf_qt=mf["edmf_qt"],
                theta=state.theta,
                thl=_liquid_potential_temperature(state),
                qw=sqw,
                p=state.p,
                exner=exner,
                dz=state.dz,
                xland=jnp.broadcast_to(
                    jnp.asarray(flux.xland, dtype=state.theta.dtype), fltv.shape
                ),
            )
        # The freshly diagnosed SGS cloud enters the buoyancy (thlv) used by
        # mym_level2/mym_turbulence below (the WRF thlv1 rebuild).
        state = state.replace(qc_bl=qc_bl, qi_bl=qi_bl, cldfra_bl=cldfra_bl)
    turb = _mym_turbulence(state, qke, fltv, flux.ustar, dx, flux.xland)
    qke_new, _qwt, qdiss, _pdk = _mym_predict_qke(state, qke, turb, dt, flux.ustar, flux)
    if sgs_cloud:
        qsq_new = _mym_predict_qsq(
            state, state.qsq, turb, dt, s_aw=mf["s_aw"] if mf is not None else None
        )
    else:
        qsq_new = state.qsq
    if edmf and mf is None:
        mf = _edmf_arrays_from_state(state, flux, fltv, turb["pblh"], dt, dx)
    u, v, theta, qv = _apply_mean_tendencies(state, turb, dt, flux, wind, rhosfc, mf=mf)
    km, kh = _retrieve_exchange_coeffs(state, turb)
    tke = 0.5 * qke_new

    u = assert_finite(u, "mynn_u", enabled=debug)
    v = assert_finite(v, "mynn_v", enabled=debug)
    theta = assert_finite(theta, "mynn_theta", enabled=debug)
    qv = assert_physical_bounds(qv, 0.0, 0.1, "mynn_qv", enabled=debug)
    tke = assert_physical_bounds(tke, TKE_EPS, 200.0, "mynn_tke", enabled=debug)
    km = assert_finite(km, "mynn_km", enabled=debug)
    kh = assert_finite(kh, "mynn_kh", enabled=debug)
    el = assert_finite(turb["el"], "mynn_el", enabled=debug)
    del qdiss
    return (
        state.replace(u=u, v=v, theta=theta, qv=qv, tke=tke, km=km, kh=kh, el=el, qsq=qsq_new),
        turb["pblh"],
    )


def _step_mynn_pbl_impl(state: MynnPBLColumnState, dt: float, debug: bool, surface=None,
                        edmf: bool = False, dx: float = 1000.0) -> MynnPBLColumnState:
    """Unjitted implementation shared by production and stripped entry points."""

    next_state, _pblh = _step_mynn_pbl_impl_with_pblh(state, dt, debug, surface, edmf, dx)
    return next_state


def _mynn_budget_diagnostics(state: MynnPBLColumnState, dt: float, surface=None,
                             dx: float = 1000.0):
    """Returns one-step budget diagnostics for Tier-2 validation."""

    state = _clip_state(state)
    flux, wind, fltv, rhosfc = _surface_terms(state, surface)
    qke = 2.0 * state.tke
    turb = _mym_turbulence(state, qke, fltv, flux.ustar, dx, flux.xland)
    qke_new, qwt, qdiss, pdk = _mym_predict_qke(state, qke, turb, dt, flux.ustar, flux)
    u, v, theta, qv = _apply_mean_tendencies(state, turb, dt, flux, wind, rhosfc)
    km, kh = _retrieve_exchange_coeffs(state, turb)
    next_state = state.replace(u=u, v=v, theta=theta, qv=qv, tke=0.5 * qke_new, km=km, kh=kh, el=turb["el"])
    drag = rhosfc * flux.ustar * flux.ustar / wind
    kmdz = _rho_interfaces(state, turb["dfm"])
    khdz = _rho_interfaces(state, turb["dfh"])
    prod_mass = jnp.concatenate((0.5 * (pdk[..., :-1] + pdk[..., 1:]), _zero_edge_like(pdk)), axis=-1)
    return next_state, {
        "surface_u": -drag * u[..., 0],
        "surface_v": -drag * v[..., 0],
        "surface_theta": rhosfc * flux.theta_flux,
        "surface_qv": rhosfc * flux.qv_flux,
        "top_u": kmdz[..., -2] * (u[..., -1] - u[..., -2]),
        "top_v": kmdz[..., -2] * (v[..., -1] - v[..., -2]),
        "top_theta": khdz[..., -2] * (theta[..., -1] - theta[..., -2]),
        "top_qv": khdz[..., -2] * (qv[..., -1] - qv[..., -2]),
        "tke_production": prod_mass,
        "tke_transport": qwt,
        "tke_dissipation": qdiss,
    }


def _pad_columns_leaf(arr, ncol: int, padded_ncol: int):
    """Pads a flattened leading column axis by repeating the last real column."""

    arr = jnp.asarray(arr)
    if arr.ndim == 0 or arr.shape[0] != ncol or padded_ncol == ncol:
        return arr
    tail = jnp.broadcast_to(arr[-1:], (padded_ncol - ncol,) + arr.shape[1:])
    return jnp.concatenate((arr, tail), axis=0)


def _slice_columns_leaf(arr, start, tile_cols: int, padded_ncol: int):
    """Slices one fixed-size column tile from a padded leading axis."""

    arr = jnp.asarray(arr)
    if arr.ndim == 0 or arr.shape[0] != padded_ncol:
        return arr
    starts = [jnp.zeros((), dtype=start.dtype)] * arr.ndim
    starts[0] = start
    sizes = list(arr.shape)
    sizes[0] = tile_cols
    return lax.dynamic_slice(arr, starts, sizes)


def _scatter_columns_leaf(carry, tile, start):
    """Writes one result tile back into the padded accumulator."""

    starts = [jnp.zeros((), dtype=start.dtype)] * carry.ndim
    starts[0] = start
    return lax.dynamic_update_slice(carry, tile.astype(carry.dtype), starts)


def _tiled_mynn_step(state: MynnPBLColumnState, dt: float, debug: bool, surface,
                     edmf: bool, dx: float):
    """Runs the MYNN column step over fixed-size leading-column tiles.

    Engages only for a flat ``(B, nz)`` batch larger than the tile width; the
    whole-batch path is unchanged otherwise.  No MYNN op couples columns, so
    the tiled result is value-identical per column (GPU bit-identical; CPU
    tridiagonal solves may differ by ~1 ulp from batch-width SIMD codegen --
    see the module-level tiling note); the peak live set of the dense BouLac
    ``(B, nz, nz)`` intermediates shrinks to one tile.
    """

    profile = jnp.asarray(state.theta)
    tiling_active = (
        _MYNN_COLUMN_TILING
        and _MYNN_COLUMN_TILE_COLS > 0
        and profile.ndim == 2
        and int(profile.shape[0]) > _MYNN_COLUMN_TILE_COLS
    )
    if not tiling_active:
        return _step_mynn_pbl_impl_with_pblh(state, dt, debug, surface, edmf, dx)

    ncol = int(profile.shape[0])
    tile_cols = int(_MYNN_COLUMN_TILE_COLS)
    n_tiles = (ncol + tile_cols - 1) // tile_cols
    padded_ncol = n_tiles * tile_cols

    pad = partial(_pad_columns_leaf, ncol=ncol, padded_ncol=padded_ncol)
    padded_state = jax.tree_util.tree_map(pad, state)
    padded_surface = jax.tree_util.tree_map(pad, surface) if surface is not None else None
    init_state = jax.tree_util.tree_map(jnp.zeros_like, padded_state)
    init_pblh = jnp.zeros((padded_ncol,), dtype=profile.dtype)

    def body(carry, tile_index):
        start = tile_index * tile_cols
        tile_state = jax.tree_util.tree_map(
            lambda a: _slice_columns_leaf(a, start, tile_cols, padded_ncol), padded_state
        )
        tile_surface = (
            jax.tree_util.tree_map(
                lambda a: _slice_columns_leaf(a, start, tile_cols, padded_ncol),
                padded_surface,
            )
            if padded_surface is not None
            else None
        )
        out_state, out_pblh = _step_mynn_pbl_impl_with_pblh(
            tile_state, dt, debug, tile_surface, edmf, dx
        )
        carry_state, carry_pblh = carry
        next_state = jax.tree_util.tree_map(
            lambda c, t: _scatter_columns_leaf(c, t, start), carry_state, out_state
        )
        next_pblh = _scatter_columns_leaf(carry_pblh, out_pblh, start)
        return (next_state, next_pblh), None

    (out_state, out_pblh), _ = lax.scan(
        body, (init_state, init_pblh), jnp.arange(n_tiles, dtype=jnp.int32)
    )
    unpad = lambda a: a[:ncol] if a.ndim >= 1 and a.shape[0] == padded_ncol else a
    return jax.tree_util.tree_map(unpad, out_state), out_pblh[:ncol]


@partial(jax.jit, static_argnames=("dt", "debug", "edmf", "dx"))
def step_mynn_pbl_column(
    state: MynnPBLColumnState, dt: float, *, debug: bool = False, surface=None,
    edmf: bool = False, dx: float = 1000.0,
) -> MynnPBLColumnState:
    """Advances one fused MYNN2.5 column step.

    ``surface`` (a :class:`SurfaceFluxes` pytree) is the operational coupling
    hand-off from the WRF revised surface layer; ``None`` uses the standalone
    neutral-bulk stub for the analytic MYNN fixture. ``edmf=True`` activates the
    MYNN-EDMF mass-flux nonlocal scalar transport (``s_awqv``/``s_awthl``); ``dx``
    is the horizontal grid spacing (m) used by the plume sizing."""

    next_state, _pblh = _tiled_mynn_step(state, dt, debug, surface, edmf, dx)
    return next_state


@partial(jax.jit, static_argnames=("dt", "debug", "edmf", "dx"))
def step_mynn_pbl_column_with_pblh(
    state: MynnPBLColumnState, dt: float, *, debug: bool = False, surface=None,
    edmf: bool = False, dx: float = 1000.0,
):
    """Advance one MYNN column step and also return the diagnosed PBL height.

    Used by the operational ``mynn_adapter`` to emit the PBLH operational
    diagnostic (coupler_interface.md §4) without adding a prognostic State leaf.
    ``edmf``/``dx`` enable + size the EDMF mass-flux transport (see
    :func:`step_mynn_pbl_column`)."""

    return _tiled_mynn_step(state, dt, debug, surface, edmf, dx)


@partial(jax.jit, static_argnames=("dt",))
def step_mynn_pbl_column_debug_stripped(state: MynnPBLColumnState, dt: float) -> MynnPBLColumnState:
    """Hand-stripped sibling used for the debug-vs-production HLO identity proof."""

    return _step_mynn_pbl_impl(state, dt, False)
