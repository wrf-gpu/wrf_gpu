"""JAX Thompson-column source/sink subset for M5-S1."""

from __future__ import annotations

import math
import os
from functools import partial
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.physics import column_tiling
from gpuwrf.physics.thompson_constants import (
    AM_G_MP8,
    AM_I,
    AM_R,
    AM_S,
    AV_C,
    AV_I,
    AV_R,
    AV_S,
    AV_G_MP8,
    BV_C,
    BV_G_MP8,
    BV_I,
    BV_R,
    BV_S,
    ATO,
    C_CUBE,
    C_SQRD,
    CCG2_NU12,
    CCG3_NU12,
    CCG4_NU12,
    CCG5_NU12,
    CGE9,
    CGE11,
    CGG6_OVER_CGG3,
    CIE2,
    CIG3,
    CIG6,
    CIG7,
    CRE10,
    CRE11,
    CRE1,
    CRE2,
    CRE3,
    CRE6,
    CRE7,
    CRE8,
    CRE9,
    CRE12,
    CRG2,
    CRG3,
    CRG6,
    CRG7,
    CRG12,
    D0C,
    D0I,
    D0R,
    D0S,
    DS_FIRST,
    DS_LAST,
    EF_RI,
    EF_SI,
    EPS,
    FV_R,
    FV_S,
    HGFR,
    LFUS,
    LSUB,
    NBS_EFSW,
    NT_C,
    NU_C_MP8,
    OBMG,
    OBMI,
    OBMR,
    OCG1_NU12,
    OCG2_NU12,
    OIG1,
    OIG2,
    ORG1,
    ORG2,
    ORG3,
    PI,
    R1,
    R2,
    R_D,
    RHO_NOT,
    RV,
    RHO_W_RIME,
    T1_MELT_QG,
    T1_MELT_QS,
    T1_QG_QC,
    T1_QR_EV,
    T1_QR_QC,
    T1_QR_QI,
    T1_QS_QC,
    T1_QS_QI,
    T1_SUBL_QG,
    T1_SUBL_QS,
    T2_MELT_QG,
    T2_MELT_QS,
    T2_QR_EV,
    T2_QR_QI,
    T2_SUBL_QG,
    T2_SUBL_QS,
    T_0,
    TNO,
    XM0I,
)
from gpuwrf.physics.thompson_saturation import (
    cp_inverse,
    latent_heat_vaporization,
    saturation_mixing_ratio_ice,
    saturation_mixing_ratio_liquid,
)
from gpuwrf.physics.thompson_tables import (
    DR_FIRST,
    DR_LAST,
    N_EFRW_C,
    N_EFRW_R,
    N_I_TABLE,
    N_I1_TABLE,
    N_R1_TABLE,
    N_R_TABLE,
    N_TC_TABLE,
    NT_I_FIRST,
    R_R_FIRST,
    R_I_FIRST,
    THOMPSON_TABLES,
    ThompsonTableBundle,
    COLD_COLLECTION_TABLES,
    ColdCollectionTables,
)


config.update("jax_enable_x64", True)


# --- Work-precision control (ADR-007 fp32 microphysics) -----------------------
# The body can run its internal rate/integration math in fp32 while keeping the
# State storage dtype (the kernel inputs/outputs) unchanged: it casts every
# column leaf to the WORK dtype on entry and casts the result back to each leaf's
# storage dtype on exit.  Note ``p`` arrives fp64 (acoustic-locked), so without
# this explicit cast the whole body would promote to fp64 even with fp32 storage.
#
# MEASURED RESULT (proofs/thompson_perf): fp32 gives ~1.0x here -- the Thompson
# kernel is dominated (~85 %) by the sedimentation substep loop, which is
# LAUNCH/bandwidth-bound, NOT fp64-arithmetic-bound.  fp32 is therefore the WRONG
# lever for this kernel (the same finding as the dycore), and is kept as a GATED
# opt-in only.  It IS oracle-faithful: fp32 perturbs the moist outputs by <= ~1
# fp32 ULP (rel <= 9e-7), at or below the WRF oracle's own fp32 storage
# granularity (WRF stores these fields in fp32).
#
# Default = fp64 (byte-for-byte the prior behaviour).  Set GPUWRF_THOMPSON_FP32=1
# to run the rate math in fp32.
def _work_dtype():
    """Return the dtype the rate/integration math runs in (fp64 default)."""

    return jnp.float32 if os.environ.get("GPUWRF_THOMPSON_FP32", "0") == "1" else jnp.float64


# --- Column tiling (v0.15 kernel-final, VRAM ceiling) --------------------------
# The full operational Thompson step is the largest UN-tiled column-physics
# working set (proofs/perf/v015/km_bench/vram_ceiling_findings.json): at the
# 640x320 1-km target (~205k cols) its fused intermediates contribute to the
# single ~28 GiB temp arena that OOMs the 32 GiB card.  Running the identical
# body over fixed-size leading-column tiles (the SAME lax.scan pattern proven
# exact for RRTMG and MYNN) caps the per-step transient to one tile.  Tiling
# engages ONLY for grids whose flattened column count exceeds one tile, so the
# production Switzerland case (128x128 = 16384 cols == one tile) NEVER tiles and
# keeps its byte-identical untiled graph either way.
#
# DEFAULT OFF (v0.15 ship-gate decision, proofs/perf/v015/vram_mp_tiling.json):
# the multi-tile lax.scan moves XLA fusion boundaries, so the tiled result is
# NOT byte-identical to the monolithic body — it differs at the fp64 noise floor
# (worst measured 2.8e-14 K on T, 1 of 720,896 cells; reshape itself is
# byte-exact, the residual is the scan-vs-monolith FMA-contraction boundary,
# i.e. the documented Tier-P phenomenon of V0150-TIERED-IDENTITY-ADR).  That is
# trivially inside the v0.14 frozen tolerance (1.9e-14 of the 1.5 K T limit) but
# fails a STRICT byte gate, so tiling is an OPT-IN large-grid (>16384-col) VRAM
# lever, NOT a default: `GPUWRF_MP_COLUMN_TILING=1` to enable.  The default
# production graph stays byte-identical to v0.14.
_MP_COLUMN_TILING = column_tiling.env_bool("GPUWRF_MP_COLUMN_TILING", False)
_MP_COLUMN_TILE_COLS = max(0, column_tiling.env_int("GPUWRF_MP_COLUMN_TILE_COLS", 16384))


def _fp32_enabled() -> bool:
    return os.environ.get("GPUWRF_THOMPSON_FP32", "0") == "1"


def _riming_enabled() -> bool:
    """v0.15 cold-phase riming gate (default ON; the WRF-faithful path).

    The Switzerland d01 RAINNC 5.19 mm miss was PROVEN non-chaotic
    (proofs/v015/falsifier_rainnc_report.json: WRF internal variability is
    0.057 mm pooled over 72 h) and the dominant missing Thompson process set is
    cold-phase collection.  ``GPUWRF_THOMPSON_RIMING=0`` restores the pre-v0.15
    deposition-only cold growth bitwise.
    """

    return os.environ.get("GPUWRF_THOMPSON_RIMING", "1").strip().lower() not in {"0", "false", "no", "off"}


def _cold_collection_enabled() -> bool:
    """v0.15 cold-collection gate (rain-collecting-snow/graupel + Bigg freezing).

    Default ON when the bit-exact WRF cold-collection fixture
    (``thompson-cold-collection-v1.npz``) is present.  These rain->graupel sinks
    below 0 C are the dominant missing Thompson process for the January Alpine
    Switzerland RAINNC surplus (the coldmix column oracle shows the port retains
    rain WRF freezes/collects into graupel aloft -- proofs/v015/cold_collection_
    oracle/).  ``GPUWRF_THOMPSON_COLD_COLLECTION=0`` restores the riming-only
    cold growth bitwise.
    """

    if COLD_COLLECTION_TABLES is None:
        return False
    return os.environ.get("GPUWRF_THOMPSON_COLD_COLLECTION", "1").strip().lower() not in {"0", "false", "no", "off"}


def _ice_collection_enabled() -> bool:
    """WRF rain/snow collecting cloud ice, gated with the cold-collection lane.

    ``GPUWRF_THOMPSON_ICE_COLLECTION`` may override this scalar rci/sci family
    directly; otherwise the existing cold-collection flag controls it so
    ``GPUWRF_THOMPSON_COLD_COLLECTION=0`` restores the pre-v0.18 cold lane.
    """

    raw = os.environ.get("GPUWRF_THOMPSON_ICE_COLLECTION", os.environ.get("GPUWRF_THOMPSON_COLD_COLLECTION", "1"))
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _cast_state(state: "ThompsonColumnState", dtype) -> "ThompsonColumnState":
    """Cast every column leaf of ``state`` to ``dtype`` (no-op if already there)."""

    return state.replace(
        **{name: jnp.asarray(getattr(state, name)).astype(dtype) for name in ThompsonColumnState.__slots__}
    )


def _restore_state(state: "ThompsonColumnState", dtypes: dict) -> "ThompsonColumnState":
    """Cast every column leaf of ``state`` back to its original storage dtype."""

    return state.replace(
        **{name: jnp.asarray(getattr(state, name)).astype(dtypes[name]) for name in ThompsonColumnState.__slots__}
    )


@jax.tree_util.register_pytree_node_class
class ThompsonColumnState:
    """Pytree for a batch of independent Thompson columns on mass levels.

    Layout matches WRF ``mp_thompson`` column arrays (kts:kte, vertical last).
    ``Ns``/``Ng`` carry the snow/graupel number concentration the WRF prognostic
    state advances (``ns``/``ng1d``); ``dz``/``w`` are the WRF ``dzq``/``w1d``
    inputs required by the sedimentation flux (``module_mp_thompson.F:3784-3960``).
    All are optional with WRF-faithful defaults so legacy call sites that only
    pass the source/sink subset keep working.
    """

    __slots__ = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng", "T", "p", "rho", "dz", "w")

    def __init__(self, qv, qc, qr, qi, qs, qg, Ni, Nr, T, p, rho, Ns=None, Ng=None, dz=None, w=None) -> None:
        self.qv = qv
        self.qc = qc
        self.qr = qr
        self.qi = qi
        self.qs = qs
        self.qg = qg
        self.Ni = Ni
        self.Nr = Nr
        self.Ns = Ns if Ns is not None else jnp.zeros_like(qs)
        self.Ng = Ng if Ng is not None else jnp.zeros_like(qg)
        self.T = T
        self.p = p
        self.rho = rho
        # Default dz = 250 m (a representative mid-troposphere WRF layer) and
        # w = 0 so the sedimentation flux is well-defined for callers that do
        # not provide geometry; the coupler always supplies the real values.
        self.dz = dz if dz is not None else jnp.full_like(qv, 250.0)
        self.w = w if w is not None else jnp.zeros_like(qv)

    def replace(self, **updates) -> "ThompsonColumnState":
        """Returns a same-layout pytree with explicit field updates."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        """Presents column arrays as JAX leaves for JIT and scan transforms."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds the column state after JAX transformations.

        Reconstruct by ``__slots__`` name (not positionally) because the
        constructor signature order differs from the leaf order.
        """

        del aux
        return cls(**dict(zip(cls.__slots__, children, strict=True)))

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality outside JIT for cache/debug tests."""

        if not isinstance(other, ThompsonColumnState):
            return NotImplemented
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(_leaves(self), _leaves(other), strict=True)
        )

    def __hash__(self) -> int:
        """Hashes small column states outside JIT; never used in the physics hot path."""

        parts = []
        for leaf in _leaves(self):
            host = np.asarray(leaf)
            parts.append((tuple(host.shape), str(host.dtype), host.tobytes()))
        return hash(tuple(parts))


def _leaves(state: ThompsonColumnState) -> Iterable[jax.Array]:
    """Centralizes leaf iteration for equality, hashing, and byte accounting."""

    return (getattr(state, name) for name in ThompsonColumnState.__slots__)


def density_from_pressure_temperature(p, T, qv):
    """Matches mp_gt_driver's rho diagnostic at module_mp_thompson.F.pre line 1270."""

    return 0.622 * p / (R_D * T * (qv + 0.622))


def _clip_species(state: ThompsonColumnState) -> ThompsonColumnState:
    """Mirrors Thompson's non-negative hydrometeor preamble and qv floor."""

    return state.replace(
        qv=jnp.maximum(state.qv, 1.0e-10),
        qc=jnp.maximum(state.qc, 0.0),
        qr=jnp.maximum(state.qr, 0.0),
        qi=jnp.maximum(state.qi, 0.0),
        qs=jnp.maximum(state.qs, 0.0),
        qg=jnp.maximum(state.qg, 0.0),
        Ni=jnp.maximum(state.Ni, 0.0),
        Nr=jnp.maximum(state.Nr, 0.0),
    )


def _air_properties(state: ThompsonColumnState):
    """Reuses WRF thermodynamic scalars from lines 2055-2064 and 3572-3581."""

    tempc = state.T - 273.15
    diffu = 2.11e-5 * (state.T / 273.15) ** 1.94 * (101325.0 / state.p)
    visco = jnp.where(tempc >= 0.0, 1.718 + 0.0049 * tempc, 1.718 + 0.0049 * tempc - 1.2e-5 * tempc * tempc) * 1.0e-5
    tcond = (5.69 + 0.0168 * tempc) * 1.0e-5 * 418.936
    lvap = latent_heat_vaporization(state.T)
    ocp = cp_inverse(state.qv)
    rhof = jnp.sqrt(RHO_NOT / jnp.maximum(state.rho, R1))
    rhof2 = jnp.sqrt(rhof)
    vsc2 = jnp.sqrt(jnp.maximum(state.rho, R1) / jnp.maximum(visco, R1))
    return tempc, diffu, visco, tcond, lvap, ocp, rhof, rhof2, vsc2


def _clamp_rain_number(qr, Nr, rho):
    """WRF working-rain-number mvd clamp (module_mp_thompson.F:1888-1898, 3240-3250).

    WRF rebuilds the DIAGNOSTIC rain number ``nr(k)`` used for slope/intercept,
    fall speed and rate calculations so the median-volume diameter stays inside
    [D0r*0.75, 2.5 mm].  When the (qr, Nr) pair implies an mvd outside that band,
    the number is recomputed from the mass at the clamped mvd:
    ``nr = crg(2)*org3*rr*lamr**bm_r / am_r`` with ``lamr = (3+mu_r+0.672)/mvd``.
    This is a WORKING-number clamp (it never changes the rain MASS); the prognostic
    rain number is re-derived from the same band in ``_finish`` (WRF 4046-4055).

    Returns the clamped rain number ``Nr_c`` (per-kg), unchanged where qr<=R1.
    Faithful transcription: ``crg(2)=1`` (CRG2), ``org3=1/6`` (ORG3),
    ``bm_r=3``, so ``nr*rho = rr*lamr**3 / (6*am_r)`` -- the exact inverse of the
    forward ``lamr = (am_r*crg(3)*org2*nr/rr)**obmr`` used everywhere else.
    """

    active = qr > R1
    rr = jnp.maximum(qr * rho, R1)
    nr = jnp.maximum(Nr * rho, R2)
    lamr = (AM_R * CRG3 * ORG2 * nr / rr) ** OBMR
    mvd_r = (3.0 + 0.672) / lamr
    mvd_clamped = jnp.minimum(2.5e-3, jnp.maximum(D0R * 0.75, mvd_r))
    out_of_band = mvd_clamped != mvd_r
    lamr_c = (3.0 + 0.672) / mvd_clamped
    nr_c = CRG2 * ORG3 * rr * lamr_c**3.0 / AM_R
    # Only the diagnostic number is rebuilt, and only when out of the band; the
    # per-kg clamped number is nr_c/rho.  qr<=R1 columns keep their Nr untouched.
    return jnp.where(active & out_of_band, nr_c / rho, Nr)


def _rain_distribution(qr, Nr, rho):
    """Encapsulates WRF rain slope/intercept equations from lines 2210-2215.

    ``Nr`` is the WRF working rain number, mvd-clamped to [D0r*0.75, 2.5 mm] so the
    slope/intercept diagnostics match WRF's ``nr(k)`` (module_mp_thompson.F:1888-
    1898).  Every warm-rain rate and fall-speed term reads its rain slope through
    this helper, so the clamp is applied consistently with WRF.
    """

    Nr = _clamp_rain_number(qr, Nr, rho)
    rr = jnp.maximum(qr * rho, R1)
    nr = jnp.maximum(Nr * rho, R2)
    lamr = (AM_R * CRG3 * ORG2 * nr / rr) ** OBMR
    ilamr = 1.0 / lamr
    mvd_r = (3.0 + 0.672) / lamr
    n0_r = nr * ORG2 * lamr**CRE2
    active = (qr > R1) & (Nr > 0.0)
    return rr, nr, lamr, ilamr, mvd_r, n0_r, active


def _cloud_distribution(qc, rho):
    """Encapsulates the mp=8 cloud gamma terms used by Berry-Reinhardt."""

    rc = jnp.maximum(qc * rho, R1)
    lamc = (NT_C * AM_R * CCG2_NU12 * OCG1_NU12 / rc) ** OBMR
    xdc = jnp.maximum(D0C * 1.0e6, (rc / (AM_R * NT_C)) ** OBMR * 1.0e6)
    mvd_c = (3.0 + NU_C_MP8 + 0.672) / lamc
    mvd_c = jnp.maximum(D0C, jnp.minimum(mvd_c, D0R))
    return rc, lamc, xdc, mvd_c, qc > R1


def _ice_distribution(qi, Ni, rho):
    """Encapsulates WRF cloud-ice particle diameter terms from lines 2711-2715."""

    ri = jnp.maximum(qi * rho, R1)
    ni = jnp.maximum(Ni * rho, R2)
    lami_missing = CIE2 / 5.0e-6
    ni_missing = jnp.minimum(999.0e3, OIG2 * ri / AM_I * lami_missing**3.0)
    ni = jnp.where((qi > R1) & (ni <= R2), ni_missing, ni)
    lami = (AM_I * 6.0 * OIG1 * ni / ri) ** OBMI
    xdi_raw = (3.0 + 0.0 + 1.0) / lami
    lami_small = CIE2 / 5.0e-6
    ni_small = jnp.minimum(999.0e3, OIG2 * ri / AM_I * lami_small**3.0)
    lami_large = CIE2 / 300.0e-6
    ni_large = OIG2 * ri / AM_I * lami_large**3.0
    ni = jnp.where((qi > R1) & (xdi_raw < 5.0e-6), ni_small, ni)
    ni = jnp.where((qi > R1) & (xdi_raw > 300.0e-6), ni_large, ni)
    lami = (AM_I * 6.0 * OIG1 * ni / ri) ** OBMI
    ilami = 1.0 / lami
    xdi = jnp.maximum(D0I, (3.0 + 0.0 + 1.0) * ilami)
    xmi = AM_I * xdi**3.0
    return ri, ni, lami, ilami, xdi, xmi, qi > R1


def _lookup_digit_index(values, first_power: int, size: int):
    """Matches WRF's decade/digit table indexes without a search loop."""

    safe = jnp.maximum(values, jnp.finfo(jnp.asarray(values).dtype).tiny)
    decade = jnp.floor(jnp.log10(safe))
    digit = jnp.floor(safe / (10.0**decade))
    index = digit + 9.0 * (decade - float(first_power)) - 1.0
    return jnp.clip(index.astype(jnp.int32), 0, size - 1)


def _take2(table, i, j):
    """Dynamic 2-D table read lowered as a gather, not as Python indexing."""

    return jnp.take(jnp.ravel(table), i.astype(jnp.int32) * table.shape[1] + j.astype(jnp.int32))


def _take3_last(table, i, j):
    """Reads a packed 2-D table with a small last dimension."""

    base = (i.astype(jnp.int32) * table.shape[1] + j.astype(jnp.int32)) * table.shape[2]
    offsets = jnp.arange(table.shape[2], dtype=jnp.int32)
    return jnp.take(jnp.ravel(table), base[..., None] + offsets)


def _take_qrfz(table, idx_r, idx_r1, idx_tc):
    """Reads default-IN rain-freezing values from one flattened dynamic index."""

    combined = (idx_r.astype(jnp.int32) * N_R1_TABLE + idx_r1.astype(jnp.int32)) * N_TC_TABLE + idx_tc.astype(jnp.int32)
    base = combined * table.shape[1]
    offsets = jnp.arange(table.shape[1], dtype=jnp.int32)
    return jnp.take(jnp.ravel(table), base[..., None] + offsets)


def _snow_moment(order, smo2, tempc, tables: ThompsonTableBundle):
    """Evaluates the Field et al. snow-moment polynomial used by WRF."""

    sa = tables.snow_sa
    sb = tables.snow_sb
    tc2 = tempc * tempc
    order2 = order * order
    loga = (
        sa[0]
        + sa[1] * tempc
        + sa[2] * order
        + sa[3] * tempc * order
        + sa[4] * tc2
        + sa[5] * order2
        + sa[6] * tc2 * order
        + sa[7] * tempc * order2
        + sa[8] * tc2 * tempc
        + sa[9] * order2 * order
    )
    b = (
        sb[0]
        + sb[1] * tempc
        + sb[2] * order
        + sb[3] * tempc * order
        + sb[4] * tc2
        + sb[5] * order2
        + sb[6] * tc2 * order
        + sb[7] * tempc * order2
        + sb[8] * tc2 * tempc
        + sb[9] * order2 * order
    )
    return 10.0**loga * smo2**b


def _snow_moments(qs, rho, tempc, tables: ThompsonTableBundle = THOMPSON_TABLES):
    """Provides WRF snow moment inputs from module_mp_thompson.F.pre:2093-2191."""

    rs = jnp.maximum(qs * rho, R1)
    tc0 = jnp.minimum(-0.1, tempc)
    smob = rs / AM_S
    smo2 = smob
    smo0 = _snow_moment(0.0, smo2, tc0, tables)
    smo1 = _snow_moment(1.0, smo2, tc0, tables)
    smoc = _snow_moment(tables.cse[0], smo2, tc0, tables)
    smof = _snow_moment(tables.cse[15], smo2, tc0, tables)
    xds = jnp.where(qs > R1, smoc / jnp.maximum(smob, R1), 0.0)
    c_snow = C_SQRD + (tempc + 1.5) * (C_CUBE - C_SQRD) / (-30.0 + 1.5)
    c_snow = jnp.maximum(C_SQRD, jnp.minimum(c_snow, C_CUBE))
    return rs, xds, smo0, smo1, smof, c_snow, qs > R1


# WRF Field et al. snow distribution constants used by the sedimentation fall
# speed formula in module_mp_thompson.F:3711-3721.  Kept local to this file to
# respect the v0.11.0 Thompson lane ownership boundary.
_SNOW_KAP0 = 490.6
_SNOW_KAP1 = 17.46
_SNOW_LAM0 = 20.78
_SNOW_LAM1 = 3.29
_SNOW_CSE1 = 3.0
_SNOW_CSE4 = 3.54999995
_SNOW_CSE7 = 3.63569999
_SNOW_CSE10 = 4.18569994
_SNOW_CSG1 = 2.0
_SNOW_CSG4 = 3.51325202
_SNOW_CSG7 = 3.87160635
_SNOW_CSG10 = 7.61279917
_SNOW_MU_S = 0.6357


def _snow_terminal_velocity_wrf(rhof, xds, active_snow):
    """WRF snow mass terminal velocity before melt/riming adjustments.

    This is the Field two-gamma moment-ratio formula from WRF
    ``module_mp_thompson.F:3711-3721``:
    ``rhof*av_s*(t1_vts+t2_vts)/(t3_vts+t4_vts)``.  The previous kernel used
    the simpler ``av_s*xDs**bv_s`` single-slope closure, which is only a rough
    approximation of the mp=8 snow distribution even when the riming boost is
    inactive.  The current port does not carry WRF's pre-sedimentation
    ``prr_sml`` or ``vts_boost`` process flags, so this helper closes the
    load-bearing inactive-boost formulation and leaves those flags at their
    inactive WRF values.
    """

    mrat = 1.0 / jnp.maximum(xds, R1)
    ils1_fall = 1.0 / (mrat * _SNOW_LAM0 + 100.0)
    ils2_fall = 1.0 / (mrat * _SNOW_LAM1 + 100.0)
    t1_vts = _SNOW_KAP0 * _SNOW_CSG4 * ils1_fall**_SNOW_CSE4
    t2_vts = _SNOW_KAP1 * mrat**_SNOW_MU_S * _SNOW_CSG10 * ils2_fall**_SNOW_CSE10

    ils1_mass = 1.0 / (mrat * _SNOW_LAM0)
    ils2_mass = 1.0 / (mrat * _SNOW_LAM1)
    t3_vts = _SNOW_KAP0 * _SNOW_CSG1 * ils1_mass**_SNOW_CSE1
    t4_vts = _SNOW_KAP1 * mrat**_SNOW_MU_S * _SNOW_CSG7 * ils2_mass**_SNOW_CSE7
    vts = rhof * AV_S * (t1_vts + t2_vts) / jnp.maximum(t3_vts + t4_vts, R1)
    return jnp.where(active_snow, vts, 0.0)


def _default_mp8_graupel_number(qg, rho):
    """WRF mp=8 diagnostic graupel number when ``NG`` is not present.

    Thompson mp=8 does not pass the optional ``ng``/``qb`` arrays into
    ``mp_gt_driver``.  The wrapper derives a per-kg working ``ng1d`` from
    graupel mass and an empirical ``N0_exp`` relation before every column call
    (module_mp_thompson.F:1265-1276).  This helper returns that per-kg number.
    """

    rg = jnp.maximum(qg * rho, R1)
    ygra1 = jnp.log10(jnp.maximum(1.0e-9, rg))
    zans1 = jnp.clip(3.0 + (2.0 / 7.0) * (ygra1 + 8.0), 2.0, 6.0)
    n0_exp = 10.0**zans1
    lamg = (n0_exp * AM_G_MP8 * CRG3 / rg) ** 0.25
    ng_m3 = ORG3 * rg * lamg**3.0 / AM_G_MP8
    return jnp.where(qg > R1, jnp.maximum(R2, ng_m3 / rho), 0.0)


def _graupel_distribution(qg, Ng, rho):
    """Provides WRF mp=8 graupel slope/intercept terms.

    ``Ng`` is per kg.  Zero/absent ``Ng`` follows WRF's non-hail mp=8 diagnostic
    default; positive ``Ng`` is treated as a prognostic working number.
    """

    rg = jnp.maximum(qg * rho, R1)
    if Ng is None:
        Ng_eff = _default_mp8_graupel_number(qg, rho)
    else:
        Ng_eff = jnp.where(Ng > 0.0, Ng, _default_mp8_graupel_number(qg, rho))
    ng = jnp.maximum(Ng_eff * rho, R2)
    lamg = (AM_G_MP8 * CRG3 * ORG2 * ng / rg) ** OBMG
    mvd_g = (3.0 + 0.672) / lamg
    mvd_g = jnp.clip(mvd_g, D0R, 25.4e-3)
    lamg = (3.0 + 0.672) / mvd_g
    ng = ORG3 * rg * lamg**3.0 / AM_G_MP8
    ilamg = 1.0 / lamg
    n0_g = ng * ORG2 * lamg
    return rg, ng, lamg, ilamg, n0_g, qg > R1


def _reset_mp8_graupel_number(state: ThompsonColumnState) -> ThompsonColumnState:
    """Match WRF mp=8: rebuild diagnostic graupel number at column-call entry."""

    return state.replace(Ng=_default_mp8_graupel_number(state.qg, state.rho))


def _sublimation_prefactor(state: ThompsonColumnState, ssati, diffu, tcond):
    """Implements the Srivastava-Coen ventilation prefactor from lines 2450-2464."""

    otemp = 1.0 / state.T
    qvsi = saturation_mixing_ratio_ice(state.p, state.T)
    rvs = state.rho * qvsi
    rvs_p = rvs * otemp * (LSUB * otemp / RV - 1.0)
    rvs_pp = rvs * (
        otemp * (LSUB * otemp / RV - 1.0) * otemp * (LSUB * otemp / RV - 1.0)
        + (-2.0 * LSUB * otemp * otemp * otemp / RV)
        + otemp * otemp
    )
    gamsc = LSUB * diffu / tcond * rvs_p
    alphsc = 0.5 * (gamsc / (1.0 + gamsc)) * (gamsc / (1.0 + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = jnp.maximum(1.0e-9, alphsc)
    xsat = jnp.where(jnp.abs(ssati) < 1.0e-9, 0.0, ssati)
    t1_subl = 4.0 * PI * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat - 5.0 * alphsc**3 * xsat**3) / (1.0 + gamsc)
    return t1_subl, rvs


def _ice_collection_rates_from_moments(
    dt: float,
    rhof,
    ri,
    ni,
    xdi,
    xmi,
    active_ice,
    rs,
    smoe,
    rr,
    nr,
    lamr,
    mvd_r,
    n0_r,
    active_rain,
    cold=None,
):
    """WRF cloud-ice collection by snow and rain (module_mp_thompson.F:2710-2734).

    Returns WRF mass/number rates in kg m-3 s-1 or # m-3 s-1:
    ``prs_sci, pni_sci, pri_rci, pni_rci, prr_rci, pnr_rci, prg_rci``.

    WRF computes this entire cloud-ice collection family ONLY inside the
    ``if (temp(k).lt.T_0)`` cold block (module_mp_thompson.F:2554; the rci/sci
    code at 2710-2734 sits inside it).  Warm cells (T>=T_0) take the ``else``
    melt branch and leave ``prs_sci=pri_rci=prr_rci=prg_rci=0`` (initialized to 0
    at lines 1699/1720/1728/1739).  ``cold`` (a boolean mask, T<T_0) enforces
    that gate so warm-cell rain/ice never spuriously forms graupel; passing
    ``cold=None`` keeps the family unconditional only for the diagnostic wrapper.
    """

    zero = jnp.zeros_like(ri)
    if not _ice_collection_enabled():
        return zero, zero, zero, zero, zero, zero, zero
    if cold is None:
        cold = jnp.ones_like(ri, dtype=bool)

    odts = 1.0 / float(dt)
    oxmi = 1.0 / jnp.maximum(xmi, XM0I)

    sci_gate = cold & active_ice & (rs >= 1.0e-6)
    prs_sci = jnp.where(sci_gate, T1_QS_QI * rhof * EF_SI * ri * smoe, 0.0)
    pni_sci = prs_sci * oxmi

    rci_gate = cold & active_ice & active_rain & (rr >= R_R_FIRST) & (mvd_r > 4.0 * xdi)
    lamr_fv = jnp.maximum(lamr + FV_R, R1)
    pri_rci = jnp.where(rci_gate, rhof * T1_QR_QI * EF_RI * ri * n0_r * lamr_fv ** (-CRE9), 0.0)
    pnr_rci = jnp.where(rci_gate, rhof * T1_QR_QI * EF_RI * ni * n0_r * lamr_fv ** (-CRE9), 0.0)
    pnr_rci = jnp.minimum(nr * odts, pnr_rci)
    pni_rci = pri_rci * oxmi
    prr_rci = jnp.where(rci_gate, rhof * T2_QR_QI * EF_RI * ni * n0_r * lamr_fv ** (-CRE8), 0.0)
    prr_rci = jnp.minimum(rr * odts, prr_rci)
    prg_rci = pri_rci + prr_rci
    return prs_sci, pni_sci, pri_rci, pni_rci, prr_rci, pnr_rci, prg_rci


def _ice_collection_rates(state: ThompsonColumnState, dt: float, tables: ThompsonTableBundle = THOMPSON_TABLES):
    """Diagnostic wrapper for the production rci/sci rate helper."""

    _tempc, _diffu, _visco, _tcond, _lvap, _ocp, rhof, _rhof2, _vsc2 = _air_properties(state)
    ri, ni, _lami, _ilami, xdi, xmi, active_ice = _ice_distribution(state.qi, state.Ni, state.rho)
    rs, _xds, _smo0, _smo1, _smof, _c_snow, _active_snow = _snow_moments(
        state.qs, state.rho, state.T - 273.15, tables
    )
    rr, nr, lamr, _ilamr, mvd_r, n0_r, active_rain = _rain_distribution(state.qr, state.Nr, state.rho)
    tc0 = jnp.minimum(-0.1, state.T - 273.15)
    smo2 = jnp.maximum(state.qs * state.rho, R1) / AM_S
    smoe = _snow_moment(tables.cse[12], smo2, tc0, tables)
    return _ice_collection_rates_from_moments(
        dt, rhof, ri, ni, xdi, xmi, active_ice, rs, smoe, rr, nr, lamr, mvd_r, n0_r, active_rain
    )


def _finish(state: ThompsonColumnState) -> ThompsonColumnState:
    """Applies WRF final floors and number-balance constraints from lines 4033-4142."""

    qv = jnp.maximum(state.qv, 1.0e-10)
    qc = jnp.where(state.qc <= R1, 0.0, jnp.maximum(state.qc, 0.0))
    qr = jnp.where(state.qr <= R1, 0.0, jnp.maximum(state.qr, 0.0))
    qi = jnp.where(state.qi <= R1, 0.0, jnp.maximum(state.qi, 0.0))
    qs = jnp.where(state.qs <= R1, 0.0, jnp.maximum(state.qs, 0.0))
    qg = jnp.where(state.qg <= R1, 0.0, jnp.maximum(state.qg, 0.0))
    T = jnp.maximum(state.T, 50.0)
    rho = density_from_pressure_temperature(state.p, T, qv)

    ni_raw = jnp.maximum(R2 / rho, state.Ni)
    ri = jnp.maximum(qi * rho, R1)
    xni = jnp.maximum(R2, ni_raw * rho)
    lami = (AM_I * 6.0 * OIG1 * xni / ri) ** OBMI
    xdi = 4.0 / lami
    lami = jnp.where(xdi < 5.0e-6, CIE2 / 5.0e-6, lami)
    lami = jnp.where(xdi > 300.0e-6, CIE2 / 300.0e-6, lami)
    Ni = jnp.where(qi <= R1, 0.0, jnp.minimum((ri / AM_I * lami**3.0 * OIG2) / rho, 999.0e3 / rho))

    nr_raw = jnp.maximum(R2 / rho, state.Nr)
    rr = jnp.maximum(qr * rho, R1)
    xnr = jnp.maximum(R2, nr_raw * rho)
    lamr = (AM_R * CRG3 * ORG2 * xnr / rr) ** OBMR
    mvd_r = (3.0 + 0.672) / lamr
    mvd_r = jnp.minimum(2.5e-3, jnp.maximum(D0R * 0.75, mvd_r))
    lamr = (3.0 + 0.672) / mvd_r
    Nr = jnp.where(qr <= R1, 0.0, CRG2 * ORG3 * rr * lamr**3.0 / AM_R / rho)
    return state.replace(qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, Ni=Ni, Nr=Nr, T=T, rho=rho)


def _thermodynamically_admissible(state: ThompsonColumnState) -> jax.Array:
    """Mask cells where Thompson's WRF column assumptions are valid.

    The bounded-range comparisons below already exclude NaN/Inf (NaN fails every
    ordered comparison; +/-Inf fail the finite upper/lower bounds), so no explicit
    ``isfinite`` is needed — keeping production HLO free of ``is-finite`` ops per
    the debuggability-hook contract.
    """

    return (
        (state.p > 1000.0)
        & (state.p < 200000.0)
        & (state.T > 150.0)
        & (state.T < 400.0)
        & (state.rho > 0.0)
        & (state.rho < 10.0)
    )


def _select_state(mask: jax.Array, good: ThompsonColumnState, fallback: ThompsonColumnState) -> ThompsonColumnState:
    """Select per-cell fallback values without leaving the compiled path."""

    return good.replace(
        **{
            name: jnp.where(mask, getattr(good, name), getattr(fallback, name))
            for name in ThompsonColumnState.__slots__
        }
    )


def _saturation_adjustment_with_condensation(state: ThompsonColumnState, dt: float) -> tuple[ThompsonColumnState, jax.Array]:
    """Implements the 3-iteration Thompson cloud condensation adjustment."""

    del dt
    qvs = saturation_mixing_ratio_liquid(state.p, state.T)
    lvap = latent_heat_vaporization(state.T)
    ocp = cp_inverse(state.qv)
    lvt2 = lvap * lvap * ocp / RV / (state.T * state.T)
    clap = (state.qv - qvs) / (1.0 + lvt2 * qvs)
    for _ in range(3):
        expo = jnp.exp(lvt2 * clap)
        fcd = qvs * expo - state.qv + clap
        dfcd = qvs * lvt2 * expo + 1.0
        clap = clap - fcd / dfcd
    ssatw = state.qv / qvs - 1.0
    active = (ssatw > EPS) | ((ssatw < -EPS) & (state.qc > 0.0))
    clap = jnp.where(active, clap, 0.0)
    clap = jnp.where(clap < 0.0, jnp.maximum(clap, -state.qc), jnp.minimum(clap, state.qv - 1.0e-10))
    condensed_cloud = clap > EPS
    qv = state.qv - clap
    T = state.T + lvap * ocp * clap
    adjusted = state.replace(
        qv=qv,
        qc=state.qc + clap,
        T=T,
        rho=density_from_pressure_temperature(state.p, T, qv),
    )
    return adjusted, condensed_cloud


def _saturation_adjustment(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Legacy wrapper for tests that only need the adjusted state."""

    adjusted, _condensed_cloud = _saturation_adjustment_with_condensation(state, dt)
    return adjusted


def _warm_rain_collection(state: ThompsonColumnState, dt: float, tables: ThompsonTableBundle = THOMPSON_TABLES) -> ThompsonColumnState:
    """Applies WRF warm-rain autoconversion/accretion rates from lines 2242-2268."""

    tempc, diffu, _visco, tcond, lvap, ocp, rhof, rhof2, vsc2 = _air_properties(state)
    del tempc, diffu, tcond, lvap, ocp, rhof2, vsc2
    rc, lamc, xdc, mvd_c, active_cloud = _cloud_distribution(state.qc, state.rho)
    rr, nr, lamr, ilamr, mvd_r, n0_r, active_rain = _rain_distribution(state.qr, state.Nr, state.rho)

    # Berry-Reinhardt autoconversion, WRF lines 2242-2258.
    dc_g = ((CCG3_NU12 * OCG2_NU12) ** OBMR / lamc) * 1.0e6
    dc_b = jnp.maximum(xdc**3 * dc_g**3 - xdc**6, 0.0) ** (1.0 / 6.0)
    zeta1_raw = 6.25e-6 * xdc * dc_b**3 - 0.4
    zeta1 = 0.5 * (zeta1_raw + jnp.abs(zeta1_raw))
    zeta = 0.027 * rc * zeta1
    taud_raw = 0.5 * dc_b - 7.5
    taud = 0.5 * (taud_raw + jnp.abs(taud_raw)) + R1
    tau = 3.72 / jnp.maximum(rc * taud, R1)
    prr_wau = jnp.where((rc > 0.01e-3) & active_cloud, jnp.minimum(rc / float(dt), zeta / tau), 0.0)
    # WRF: pnr_wau = prr_wau / (am_r*nu_c*10.*D0r**3) (module_mp_thompson.F:2191).
    # The divisor is a strictly-positive constant (~7.85e-9), so it needs NO floor;
    # a prior ``jnp.maximum(..., R2=1e-6)`` clamp here silently REPLACED the true
    # 7.85e-9 divisor with 1e-6, shrinking the autoconversion rain-number source by
    # ~127x and starving Nr in cloud-base / autoconversion-dominated columns.
    pnr_wau = prr_wau / (AM_R * NU_C_MP8 * 10.0 * D0R**3)

    # WRF rain-collecting-cloud-water shape from t_Efrw, initialized at
    # module_mp_thompson.F.pre:4921-4977 and consumed at lines 2260-2268.
    idx_r_eff = jnp.clip(
        jnp.floor(N_EFRW_R * jnp.log(jnp.maximum(mvd_r, DR_FIRST) / DR_FIRST) / jnp.log(DR_LAST / DR_FIRST)),
        0,
        N_EFRW_R - 1,
    ).astype(jnp.int32)
    idx_c_eff = jnp.clip(jnp.floor(mvd_c * 1.0e6).astype(jnp.int32) - 1, 0, N_EFRW_C - 1)
    ef_rw = _take2(tables.t_Efrw, idx_r_eff, idx_c_eff)
    prr_rcw_raw = rhof * T1_QR_QC * ef_rw * rc * n0_r * ((lamr + FV_R) ** (-CRE9))
    prr_rcw = jnp.where(active_rain & (mvd_r > D0R) & (mvd_c > D0C), prr_rcw_raw, 0.0)
    prr_rcw = jnp.minimum(jnp.maximum(rc - prr_wau * float(dt), 0.0) / float(dt), prr_rcw)

    # Rain self-collection (Seifert 1994) + drop break-up (Verlinde & Cotton 1993),
    # WRF module_mp_thompson.F:2159-2167.  This is a rain-NUMBER-only process (no
    # mass change): Ef_rr>0 (mvd_r<1950um) -> self-collection SINK; Ef_rr<0
    # (mvd_r>1950um) -> break-up SOURCE.  ``nr``/``rr``/``mvd_r`` are the working
    # mvd-clamped slope values from ``_rain_distribution``.  The Nr tendency adds
    # ``-pnr_rcr`` (WRF line 3066), i.e. dNr = -pnr_rcr*dt/rho.
    ef_rr = 1.0 - jnp.exp(2300.0 * (mvd_r - 1950.0e-6))
    pnr_rcr = jnp.where(active_rain & (mvd_r > D0R), ef_rr * 2.0 * nr * rr, 0.0)

    autoconv = prr_wau * float(dt) / state.rho
    accretion = prr_rcw * float(dt) / state.rho
    transfer = jnp.minimum(state.qc, autoconv + accretion)
    nr_gain = pnr_wau * float(dt) / state.rho
    nr_rcr = pnr_rcr * float(dt) / state.rho
    # Floor the post-process rain number at 0 (WRF carries nrten then re-floors at
    # MAX(R2/rho, ...) in _finish; a self-collection sink must not drive Nr<0).
    Nr_new = jnp.maximum(0.0, state.Nr + nr_gain - nr_rcr)
    return state.replace(qc=state.qc - transfer, qr=state.qr + transfer, Nr=Nr_new)


def _rain_evaporation(
    state: ThompsonColumnState,
    dt: float,
    skip_evaporation: jax.Array | bool = False,
    graupel_melt: jax.Array | float = 0.0,
) -> ThompsonColumnState:
    """Applies WRF Srivastava-Coen rain evaporation from lines 3561-3638."""

    _tempc, diffu, _visco, tcond, lvap, ocp, _rhof, rhof2, vsc2 = _air_properties(state)
    # Srivastava-Coen rain evaporation, WRF lines 3561-3636.
    qvs = saturation_mixing_ratio_liquid(state.p, state.T)
    ssatw = state.qv / qvs - 1.0
    rvs = state.rho * qvs
    otemp = 1.0 / state.T
    rvs_p = rvs * otemp * (lvap * otemp / RV - 1.0)
    rvs_pp = rvs * (
        otemp * (lvap * otemp / RV - 1.0) * otemp * (lvap * otemp / RV - 1.0)
        + (-2.0 * lvap * otemp * otemp * otemp / RV)
        + otemp * otemp
    )
    gamsc = lvap * diffu / tcond * rvs_p
    alphsc = 0.5 * (gamsc / (1.0 + gamsc)) * (gamsc / (1.0 + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = jnp.maximum(1.0e-9, alphsc)
    xsat = jnp.minimum(-1.0e-9, ssatw)
    t1_evap = 2.0 * PI * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat - 5.0 * alphsc**3 * xsat**3) / (1.0 + gamsc)
    rr, nr, lamr, ilamr, _mvd_r, n0_r, active_rain = _rain_distribution(state.qr, state.Nr, state.rho)
    evap_raw = (
        t1_evap
        * diffu
        * (-ssatw)
        * n0_r
        * rvs
        * (T1_QR_EV * ilamr**CRE10 + T2_QR_EV * vsc2 * rhof2 * ((lamr + 0.5 * FV_R) ** (-CRE11)))
        / state.rho
    )
    fast_clear = (state.qv / qvs < 0.95) & (rr / state.rho <= 1.0e-8)
    evap_rate = jnp.where(fast_clear, state.qr / float(dt), evap_raw)
    tempc = state.T - 273.15
    eva_factor = jnp.minimum(1.0, 0.01 + (0.99 - 0.01) * (tempc / 20.0))
    rate_max = jnp.minimum(state.qr / float(dt), jnp.maximum(qvs - state.qv, 0.0) / float(dt))
    active = (ssatw < -EPS) & active_rain & ~jnp.asarray(skip_evaporation, dtype=bool)
    limited_rate = jnp.minimum(rate_max, evap_rate)
    limited_rate = jnp.where((jnp.asarray(graupel_melt) > 0.0) & ~fast_clear, limited_rate * eva_factor, limited_rate)
    evap = jnp.where(active, limited_rate * float(dt), 0.0)
    nr_loss = jnp.where(state.qr > 0.0, jnp.minimum(state.Nr * 0.99, state.Nr * evap / jnp.maximum(state.qr, R1)), 0.0)
    return state.replace(
        qv=state.qv + evap,
        qr=state.qr - evap,
        Nr=jnp.maximum(0.0, state.Nr - nr_loss),
        T=state.T - lvap * ocp * evap,
    )


def _warm_rain(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Preserves the legacy combined warm-rain helper for focused tests."""

    return _rain_evaporation(_warm_rain_collection(state, dt), dt)


def _instant_melt_freeze(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Applies WRF instant cloud-ice melt/cloud-water freeze from lines 4005-4031."""

    del dt
    ocp = cp_inverse(state.qv)
    lvap = latent_heat_vaporization(state.T)
    lfus2 = LSUB - lvap

    qi_melt = jnp.where(state.T > T_0, state.qi, 0.0)
    state = state.replace(
        qc=state.qc + qi_melt,
        qi=state.qi - qi_melt,
        Ni=jnp.where(qi_melt > 0.0, 0.0, state.Ni),
        T=state.T - LFUS * ocp * qi_melt,
    )

    qc_freeze = jnp.where(state.T < HGFR, state.qc, 0.0)
    return state.replace(
        qc=state.qc - qc_freeze,
        qi=state.qi + qc_freeze,
        Ni=state.Ni + qc_freeze / XM0I,
        T=state.T + lfus2 * ocp * qc_freeze,
    )


def _ice_sources_with_process_flags(
    state: ThompsonColumnState,
    dt: float,
    tables: ThompsonTableBundle = THOMPSON_TABLES,
    cold_collection_rates: tuple | None = None,
) -> tuple[ThompsonColumnState, jax.Array, jax.Array, tuple]:
    """Stages WRF-mapped ice tendencies before condensation and rain evaporation."""

    ocp = cp_inverse(state.qv)
    lvap = latent_heat_vaporization(state.T)
    lfus2 = LSUB - lvap
    if cold_collection_rates is None:
        cold_collection_rates = _zero_cold_collection_rates(state)

    # WRF rain-freezing tables are initialized by freezeH2O
    # (module_mp_thompson.F.pre:4664-4855) and consumed at lines 2658-2669.
    rr0 = jnp.maximum(state.qr * state.rho, R1)
    nr0 = jnp.maximum(state.Nr * state.rho, R2)
    rr_for_index = jnp.maximum(rr0, R_R_FIRST)
    _, _nr, lamr0, _ilamr0, _mvd_r0, _n0_r0, _active_rain0 = _rain_distribution(state.qr, state.Nr, state.rho)
    lam_exp = lamr0
    n0_exp = ORG1 * rr_for_index / AM_R * lam_exp**CRE1
    idx_r = _lookup_digit_index(rr_for_index, -6, N_R_TABLE)
    idx_r1 = _lookup_digit_index(n0_exp, 6, N_R1_TABLE)
    idx_tc = jnp.clip(jnp.floor(-(state.T - 273.15) + 0.5).astype(jnp.int32) - 1, 0, N_TC_TABLE - 1)
    qrfz = _take_qrfz(tables.qrfz, idx_r, idx_r1, idx_tc)
    table_active = (state.T < T_0) & (rr0 > R_R_FIRST) & (state.qr > R1)
    table_ice = jnp.where(table_active, qrfz[..., 0] / state.rho, 0.0)
    table_graupel = jnp.where(table_active, qrfz[..., 1] / state.rho, 0.0)
    table_ni = jnp.where(table_active, qrfz[..., 2] / state.rho, 0.0)
    table_nr = jnp.where(table_active, qrfz[..., 3] / state.rho, 0.0)
    fallback_active = (state.T < HGFR) & (state.qr > R1)
    fallback_ice = jnp.where(fallback_active & ~table_active, state.qr, 0.0)
    fallback_ni = jnp.where(fallback_active & ~table_active, nr0 / state.rho, 0.0)
    ice_freeze = table_ice + fallback_ice
    graupel_freeze = table_graupel
    frozen_total = ice_freeze + graupel_freeze
    freeze_ratio = jnp.where(frozen_total > state.qr, state.qr / jnp.maximum(frozen_total, R1), 1.0)
    ice_freeze = ice_freeze * freeze_ratio
    graupel_freeze = graupel_freeze * freeze_ratio
    table_ni = table_ni * freeze_ratio
    table_nr = table_nr * freeze_ratio
    pri_rfz_rate = ice_freeze * state.rho / float(dt)
    prg_rfz_rate = graupel_freeze * state.rho / float(dt)
    nr_loss = jnp.minimum(state.Nr, table_nr + table_ni + fallback_ni)
    cloud_freeze, cloud_ni, _pri_wfz_rate, _pni_wfz_rate = _cloud_water_freezing_rates(
        state,
        dt,
        COLD_COLLECTION_TABLES if _cold_collection_enabled() else None,
    )
    state = state.replace(
        qr=state.qr - ice_freeze - graupel_freeze,
        qc=state.qc - cloud_freeze,
        qi=state.qi + ice_freeze + cloud_freeze,
        qg=state.qg + graupel_freeze,
        Ni=state.Ni + table_ni + fallback_ni + cloud_ni,
        Nr=jnp.maximum(0.0, state.Nr - nr_loss),
        T=state.T + lfus2 * ocp * (ice_freeze + graupel_freeze + cloud_freeze),
    )
    state = state.replace(rho=density_from_pressure_temperature(state.p, state.T, state.qv))

    qvsi_freeze = saturation_mixing_ratio_ice(state.p, state.T)
    qvsw_freeze = saturation_mixing_ratio_liquid(state.p, state.T)
    ssati_freeze = state.qv / qvsi_freeze - 1.0
    ssatw_freeze = state.qv / qvsw_freeze - 1.0
    deposition_nucleation_active = (state.T < T_0) & ((ssati_freeze >= 0.25) | ((ssatw_freeze > EPS) & (state.T < 253.15)))
    xnc = jnp.minimum(250.0e3, TNO * jnp.exp(ATO * (T_0 - state.T)))
    xni = state.Ni * state.rho
    pni_inu = jnp.maximum(xnc - xni, 0.0) / float(dt)
    vapor_rate_max = jnp.maximum(0.0, (state.qv - qvsi_freeze) * state.rho / float(dt) * 0.999)
    pri_inu = jnp.where(deposition_nucleation_active, jnp.minimum(vapor_rate_max, XM0I * pni_inu), 0.0)
    pni_inu = jnp.where(deposition_nucleation_active, pri_inu / XM0I, 0.0)
    inu_mass = pri_inu * float(dt) / state.rho
    inu_number = pni_inu * float(dt) / state.rho

    tempc, diffu, _visco, tcond, _lvap, ocp, rhof, rhof2, vsc2 = _air_properties(state)
    del rhof
    qvs0 = saturation_mixing_ratio_liquid(state.p, T_0)
    del_qvs = jnp.maximum(0.0, qvs0 - state.qv)
    twet = state.T
    rs, _xds, smo0, smo1, smof, _c_snow, active_snow = _snow_moments(state.qs, state.rho, tempc, tables)
    rg, ng, lamg_g, ilamg, n0_g, active_graupel = _graupel_distribution(state.qg, state.Ng, state.rho)

    # Snow/graupel melting formula structure from WRF lines 2845-2889.
    prr_sml_rate = (tempc * tcond - 2.5e6 * diffu * del_qvs) * (T1_MELT_QS * smo1 + T2_MELT_QS * rhof2 * vsc2 * smof)
    prr_sml_rate = jnp.minimum(rs / float(dt), jnp.maximum(0.0, prr_sml_rate)) / state.rho
    snow_melt = jnp.where((state.T > T_0) & active_snow, prr_sml_rate * float(dt), 0.0)
    pnr_sml = jnp.where(rs > R1, smo0 / rs * snow_melt * state.rho * 10.0 ** (-0.25 * (twet - T_0)), 0.0)

    # WRF graupel-melt N0 override (module_mp_thompson.F:2802-2806): for very
    # sparse graupel the melt rate uses a renormalized intercept
    #   N0_melt = (1.E-4/rg) * ogg2 * lamg**cge(2,1)   when (rg*ng) < 1.E-4
    # (mp8: mu_g=0 -> cge(2,1)=1, ogg2=ORG2).  Without this the warm-cell melt
    # rate is computed from the diagnostic N0_g and under-melts the thin graupel
    # the v0.18 diagnostic graupel-number distribution now resolves, leaving a
    # spurious warm-cell graupel residual where WRF melts it fully.
    n0_melt = jnp.where((rg * ng) < 1.0e-4, (1.0e-4 / rg) * ORG2 * lamg_g, n0_g)
    prr_gml_rate = (tempc * tcond - 2.5e6 * diffu * del_qvs) * n0_melt * (
        T1_MELT_QG * ilamg**CRE10 + T2_MELT_QG * rhof2 * vsc2 * ilamg**CGE11
    )
    prr_gml_rate = jnp.minimum(rg / float(dt), jnp.maximum(0.0, prr_gml_rate)) / state.rho
    graupel_melt = jnp.where((state.T > T_0) & active_graupel, prr_gml_rate * float(dt), 0.0)
    pnr_gml = jnp.where(rg > R1, graupel_melt * ng / rg * 10.0 ** (-0.33 * (twet - T_0)), 0.0)
    state = state.replace(
        qs=state.qs - snow_melt,
        qg=state.qg - graupel_melt,
        qr=state.qr + snow_melt + graupel_melt,
        Nr=state.Nr + pnr_sml + pnr_gml,
        T=state.T - LFUS * ocp * (snow_melt + graupel_melt),
    )
    state = state.replace(rho=density_from_pressure_temperature(state.p, state.T, state.qv))

    tempc, diffu, visco, tcond, lvap2, ocp, rhof, rhof2, vsc2 = _air_properties(state)
    qvsi = saturation_mixing_ratio_ice(state.p, state.T)
    ssati = state.qv / qvsi - 1.0
    t1_subl, rvs = _sublimation_prefactor(state, ssati, diffu, tcond)
    ri, ni, _lami, ilami, xdi, xmi, active_ice = _ice_distribution(state.qi, state.Ni, state.rho)
    rs, xds, smo0, smo1, smof, c_snow, active_snow = _snow_moments(state.qs, state.rho, state.T - 273.15, tables)
    rg, ng, _lamg, ilamg, n0_g, active_graupel = _graupel_distribution(state.qg, state.Ng, state.rho)
    rr_ice, nr_ice, lamr_ice, _ilamr_ice, mvd_r_ice, n0_r_ice, active_rain_ice = _rain_distribution(
        state.qr, state.Nr, state.rho
    )
    tc0_ice = jnp.minimum(-0.1, state.T - 273.15)
    smo2_ice = jnp.maximum(state.qs * state.rho, R1) / AM_S
    smoe_ice = _snow_moment(tables.cse[12], smo2_ice, tc0_ice, tables)

    idx_i = jnp.where(ri > R_I_FIRST, _lookup_digit_index(ri, -10, N_I_TABLE), 0)
    idx_i1 = jnp.where(ni > NT_I_FIRST, _lookup_digit_index(ni, 0, N_I1_TABLE), 0)

    # Cloud-ice deposition uses tpi_ide from module_mp_thompson.F.pre:4870-4913;
    # ice-to-snow autoconversion uses tps/tni_iaus at lines 2731-2742.
    pri_ide_raw = C_CUBE * t1_subl * diffu * ssati * rvs * OIG1 * 1.0 * ni * ilami
    pri_ide_raw = jnp.where(active_ice, pri_ide_raw, 0.0)
    pri_ide_limited = jnp.where(
        pri_ide_raw < 0.0,
        jnp.maximum(-ri / float(dt), pri_ide_raw),
        jnp.minimum(pri_ide_raw, jnp.maximum(state.qv - qvsi, 0.0) * state.rho / float(dt) * 0.999),
    )
    iaus = _take3_last(tables.iaus, idx_i, idx_i1)
    tpi_ide = iaus[..., 2]
    pri_ide = jnp.where(pri_ide_limited > 0.0, tpi_ide * pri_ide_limited, pri_ide_limited)
    prs_ide = jnp.where(pri_ide_limited > 0.0, (1.0 - tpi_ide) * pri_ide_limited, 0.0)
    iau_table_mass = iaus[..., 0]
    iau_table_num = iaus[..., 1]
    iau_large = (idx_i == N_I_TABLE - 1) | (xdi > 5.0 * D0S)
    iau_small = xdi < 0.1 * D0S
    prs_iau_mass = jnp.where(iau_large, ri * 0.99, jnp.where(iau_small, 0.0, jnp.minimum(ri * 0.99, iau_table_mass)))
    pni_iau_num = jnp.where(iau_large, ni * 0.95, jnp.where(iau_small, 0.0, jnp.minimum(ni * 0.95, iau_table_num)))
    prs_iau_mass = jnp.where(active_ice, prs_iau_mass, 0.0)
    pni_iau_num = jnp.where(active_ice, pni_iau_num, 0.0)

    # WRF gates the whole cloud-ice collection family on the cold block
    # (module_mp_thompson.F:2554, ``if (temp(k).lt.T_0)``).  Use the post-melt
    # ``state.T`` (the same field the rci/sci moments above were built from).
    cold_block = state.T < T_0
    prs_sci, pni_sci, pri_rci, pni_rci, prr_rci, pnr_rci, prg_rci = _ice_collection_rates_from_moments(
        dt,
        rhof,
        ri,
        ni,
        xdi,
        xmi,
        active_ice,
        rs,
        smoe_ice,
        rr_ice,
        nr_ice,
        lamr_ice,
        mvd_r_ice,
        n0_r_ice,
        active_rain_ice,
        cold=cold_block,
    )

    prs_sde = c_snow * t1_subl * diffu * ssati * rvs * (T1_SUBL_QS * smo1 + T2_SUBL_QS * rhof2 * vsc2 * smof)
    prs_sde = jnp.where(active_snow, jnp.where(prs_sde < 0.0, jnp.maximum(-rs / float(dt), prs_sde), jnp.minimum(prs_sde, jnp.maximum(state.qv - qvsi, 0.0) * state.rho / float(dt) * 0.999)), 0.0)
    # WRF ordering (module_mp_thompson.F): the riming snow->graupel split (line
    # 2758) compares prs_scw against this PER-CELL-clamped prs_sde, BEFORE the
    # GLOBAL multi-term deposition vapor-conservation ratio (line ~2862) scales
    # it. Capture the pre-ratio value for the riming comparison so a
    # deposition-limited (ratio<1) cell does not spuriously trip riming_dom.
    prs_sde_preratio = prs_sde
    vapor_rate_max = (state.qv - qvsi) * state.rho / float(dt) * 0.999
    prg_gde = C_CUBE * t1_subl * diffu * ssati * rvs * n0_g * (T1_SUBL_QG * ilamg**CRE10 + T2_SUBL_QG * vsc2 * rhof2 * ilamg**CGE11)
    prg_gde = jnp.where(active_graupel & (ssati < -EPS), jnp.maximum(jnp.maximum(-rg / float(dt), prg_gde), vapor_rate_max), 0.0)
    deposition_sum = pri_inu + pri_ide + prs_ide + prs_sde + prg_gde
    limited = ((deposition_sum > EPS) & (deposition_sum > vapor_rate_max)) | ((deposition_sum < -EPS) & (deposition_sum < vapor_rate_max))
    deposition_denom = jnp.where(jnp.abs(deposition_sum) > R1, deposition_sum, 1.0)
    deposition_ratio = jnp.where(limited, vapor_rate_max / deposition_denom, 1.0)
    pri_ide = pri_ide * deposition_ratio
    prs_ide = prs_ide * deposition_ratio
    prs_sde = prs_sde * deposition_ratio
    prg_gde = prg_gde * deposition_ratio

    prs_iau_rate = prs_iau_mass / float(dt)
    pni_iau_rate = pni_iau_num / float(dt)
    ice_sump = pri_ide - prs_iau_rate - prs_sci - pri_rci
    ice_rate_max = -ri / float(dt)
    ice_ratio = jnp.where((ice_sump < ice_rate_max) & active_ice, ice_rate_max / jnp.minimum(ice_sump, -EPS), 1.0)
    pri_ide = pri_ide * ice_ratio
    prs_iau_rate = prs_iau_rate * ice_ratio
    prs_sci = prs_sci * ice_ratio
    pri_rci = pri_rci * ice_ratio
    prg_rci = pri_rci + prr_rci

    prr_rcs, _prs_rcs, _prg_rcs, _pnr_rcs, _png_rcs, prr_rcg, _prg_rcg, _pnr_rcg = cold_collection_rates
    rain_sump = -prg_rfz_rate - pri_rfz_rate - prr_rci + prr_rcs + prr_rcg
    rain_rate_max = -rr0 / float(dt)
    rain_ratio = jnp.where(
        (rain_sump < rain_rate_max) & _active_rain0,
        rain_rate_max / jnp.minimum(rain_sump, -EPS),
        1.0,
    )
    prr_rci = prr_rci * rain_ratio
    cold_collection_rates = _scale_cold_collection_rain_rates(cold_collection_rates, rain_ratio)

    ice_deposition = pri_ide * float(dt) / state.rho
    snow_from_ice_deposition = prs_ide * float(dt) / state.rho
    ice_to_snow = prs_iau_rate * float(dt) / state.rho
    ice_number_to_snow = pni_iau_rate * float(dt) / state.rho
    snow_collect_ice = prs_sci * float(dt) / state.rho
    rain_collect_ice = pri_rci * float(dt) / state.rho
    rain_to_graupel = prr_rci * float(dt) / state.rho
    ice_number_collected = (pni_sci + pni_rci) * float(dt) / state.rho
    rain_number_collected = pnr_rci * float(dt) / state.rho
    snow_deposition = prs_sde * float(dt) / state.rho
    graupel_deposition = prg_gde * float(dt) / state.rho
    vapor_sink = jnp.maximum(0.0, ice_deposition) + jnp.maximum(0.0, snow_from_ice_deposition) + jnp.maximum(0.0, snow_deposition) + jnp.maximum(0.0, graupel_deposition)
    vapor_source = jnp.maximum(0.0, -ice_deposition) + jnp.maximum(0.0, -snow_deposition) + jnp.maximum(0.0, -graupel_deposition)
    updated_qv = state.qv - vapor_sink + vapor_source - inu_mass
    rci_heat = jnp.where(state.T < T_0, (LSUB - lvap2) * ocp * rain_to_graupel, 0.0)
    updated_T = state.T + LSUB * ocp * (vapor_sink - vapor_source + inu_mass) + rci_heat
    updated = state.replace(
        qv=updated_qv,
        qi=state.qi + ice_deposition - ice_to_snow - snow_collect_ice - rain_collect_ice + inu_mass,
        qs=state.qs + snow_from_ice_deposition + ice_to_snow + snow_collect_ice + snow_deposition,
        qr=jnp.maximum(0.0, state.qr - rain_to_graupel),
        qg=state.qg + graupel_deposition + prg_rci * float(dt) / state.rho,
        # WRF lines 2719-2727 update pni_ide only in sublimation; positive
        # deposition partitions mass but does not create new cloud-ice number.
        Ni=jnp.maximum(
            0.0,
            state.Ni
            + inu_number
            + jnp.where(ice_deposition < 0.0, ice_deposition / jnp.maximum(xmi, XM0I), 0.0)
            - ice_number_to_snow
            - ice_number_collected,
        ),
        Nr=jnp.maximum(0.0, state.Nr - rain_number_collected),
        Ng=jnp.maximum(0.0, state.Ng + rain_number_collected),
        T=updated_T,
        rho=density_from_pressure_temperature(state.p, updated_T, updated_qv),
    )

    # ---- v0.15 cold-phase riming: snow/graupel collecting cloud water ----
    # WRF module_mp_thompson.F:2403-2440 (prs_scw via t_Efsw, prg_gcw via the
    # graupel Stokes-number efficiency), 2758-2776 (rimed-snow -> graupel
    # conversion split + the snow fall-speed boost vts_boost), 2879-2890
    # (cloud-water conservation), 3094/3100 (qs/qg tendencies), 3165-3173
    # (collected-water freezing heat, T<T_0 branch only).  Rates are computed
    # from the SAME pass fields the deposition rates used (state) and applied
    # after the deposition update, mirroring WRF's one-pass rate + joint apply.
    # Rain/snow, rain/graupel, and rain/snow collecting cloud-ice cold lanes are
    # handled elsewhere in this module; Hallett-Mossop rime splintering and the
    # bucketed graupel density/volume prognostic remain outside this local riming
    # block.  GPUWRF_THOMPSON_RIMING=0 rolls back this cloud-water riming path.
    #
    # dt gate: WRF (line 2840) keeps riming for dt>120 s but reroutes the
    # collected water to RAIN (prr_rcw += prs_scw+prg_gcw) instead of snow/
    # graupel -- the GPU's split-step structure applies warm-rain collection
    # BEFORE this point, so that late rain reroute is not expressible here.
    # ``dt`` is a STATIC Python float, so this is a compile-time branch; every
    # in-scope grid (Switzerland/Canary mp dt = 18 s, all operational dt <=
    # 60 s) takes the riming path.  A dt>120 s config would fall back to
    # deposition-only cold growth -- a documented scope limit, not a silent
    # masking of the riming-on path.
    vts_boost = jnp.ones_like(state.qs)
    if _riming_enabled() and float(dt) <= 120.0:
        odts = 1.0 / float(dt)
        rc_rime, _lamc_r, _xdc_r, mvd_c, active_cloud = _cloud_distribution(state.qc, state.rho)
        # smoe = Field snow moment at order cse(13) = bv_s+2 (WRF 2085-2101),
        # evaluated with the same smo2 = rs/am_s and tc0 convention as
        # _snow_moments.
        smoe = smoe_ice
        # Ef_sw: snow-bin index over Ds(1..100) (log bins D0s..2cm, WRF 862-872;
        # Fortran idx = 1 + INT(...) truncates toward zero, then MIN(nbs)).
        xds_pos = NBS_EFSW * jnp.log(jnp.maximum(xds, DS_FIRST) / DS_FIRST) / math.log(DS_LAST / DS_FIRST)
        idx_s_eff = jnp.clip(jnp.trunc(xds_pos), 0, NBS_EFSW - 1).astype(jnp.int32)
        idx_c_rime = jnp.clip(jnp.floor(mvd_c * 1.0e6).astype(jnp.int32) - 1, 0, N_EFRW_C - 1)
        ef_sw = _take2(tables.t_Efsw, idx_s_eff, idx_c_rime)
        scw_gate = active_snow & active_cloud & (mvd_c > D0C) & (xds > D0S)
        prs_scw = jnp.where(scw_gate, rhof * T1_QS_QC * ef_sw * rc_rime * smoe, 0.0)
        prs_scw = jnp.minimum(rc_rime * odts, prs_scw)

        # Graupel collecting cloud water (GPU single-density mp8 PSD convention,
        # mu_g=0): Stokes-number efficiency of WRF 2414-2431.
        xdg = 4.0 * ilamg
        vtg = rhof * AV_G_MP8 * CGG6_OVER_CGG3 * ilamg**BV_G_MP8
        stoke_g = mvd_c * mvd_c * vtg * RHO_W_RIME / (9.0 * jnp.maximum(visco, R1) * jnp.maximum(xdg, R1))
        ef_gw = jnp.where(
            stoke_g >= 0.4,
            jnp.where(stoke_g > 10.0, 0.77, 0.55 * jnp.log10(jnp.maximum(2.51 * stoke_g, R1))),
            0.0,
        )
        ef_gw = jnp.where(state.T > T_0, ef_gw * 0.1, ef_gw)
        gcw_gate = active_graupel & (rg >= 1.0e-6) & active_cloud & (mvd_c > D0C)
        prg_gcw = jnp.where(gcw_gate, rhof * T1_QG_QC * ef_gw * rc_rime * n0_g * ilamg**CGE9, 0.0)

        # Rimed-snow -> graupel conversion + snow fall-speed boost (WRF 2758-2776).
        # Compare against the PRE-global-ratio prs_sde (WRF runs this block before
        # the deposition vapor-conservation ratio); the per-cell rate_max clamp at
        # line 913 is already applied, matching WRF lines 2690-2692.
        riming_dom = (prs_scw > 2.0 * prs_sde_preratio) & (prs_sde_preratio > EPS)
        r_frac = jnp.minimum(30.0, prs_scw / jnp.maximum(prs_sde_preratio, EPS))
        g_frac = jnp.minimum(0.95, 0.15 + (r_frac - 2.0) * 0.028)
        vts_single = AV_S * xds**BV_S * jnp.exp(-FV_S * xds)
        const_ri = jnp.clip(-(mvd_c * 0.5e6) * vts_single / jnp.minimum(-0.1, tempc), 0.1, 10.0)
        rime_dens = (0.051 + 0.114 * const_ri - 0.0055 * const_ri * const_ri) * 1000.0
        g_frac = jnp.where(rime_dens < 150.0, 0.0, g_frac)  # A. Jensen low-density cutoff
        g_frac = jnp.where(riming_dom, g_frac, 0.0)
        vts_boost = jnp.where(riming_dom, jnp.minimum(1.5, 1.1 + (r_frac - 2.0) * 0.014), 1.0)
        prg_scw = g_frac * prs_scw
        png_scw = jnp.where(riming_dom & (rs > R1), prg_scw * smo0 / jnp.maximum(rs, R1), 0.0)
        prs_scw = prs_scw - prg_scw

        # Cloud-water conservation (WRF 2879-2890): the collected sum may not
        # deplete more cloud water than exists this step.
        collected = prs_scw + prg_scw + prg_gcw
        ratio_qc = jnp.where(collected > rc_rime * odts, rc_rime * odts / jnp.maximum(collected, EPS), 1.0)
        prs_scw = prs_scw * ratio_qc
        prg_scw = prg_scw * ratio_qc
        prg_gcw = prg_gcw * ratio_qc

        # ratio_qc already bounds the collected sum by rc_rime*odts and the
        # deposition block never touches qc, so the qc sink below equals the
        # qs+qg gain exactly (no extra clamp -> no mass creation/destruction).
        scw_total_dt = (prs_scw + prg_scw + prg_gcw) * float(dt) / state.rho
        lfus2_rime = LSUB - lvap2
        rime_heat = jnp.where(state.T < T_0, lfus2_rime * ocp * scw_total_dt, 0.0)
        rime_T = updated.T + rime_heat
        updated = updated.replace(
            qc=updated.qc - scw_total_dt,
            qs=updated.qs + prs_scw * float(dt) / state.rho,
            qg=updated.qg + (prg_scw + prg_gcw) * float(dt) / state.rho,
            Ns=jnp.maximum(0.0, updated.Ns - png_scw * float(dt) / state.rho),
            T=rime_T,
            rho=density_from_pressure_temperature(state.p, rime_T, updated.qv),
        )
    return updated, graupel_melt, vts_boost, cold_collection_rates


def _take4(table, i, j, k, m):
    """Dynamic 4-D table read (C-order) lowered as a single gather.

    ``table`` has shape (n0, n1, n2, n3); indices are int arrays broadcast to a
    common shape.  Matches WRF's flat lookup of a (d0,d1,d2,d3) Fortran array
    that this fixture stores in C-order with the SAME axis meaning.
    """

    n1, n2, n3 = table.shape[1], table.shape[2], table.shape[3]
    flat = (((i.astype(jnp.int32) * n1 + j.astype(jnp.int32)) * n2
             + k.astype(jnp.int32)) * n3 + m.astype(jnp.int32))
    return jnp.take(jnp.ravel(table), flat)


def _cloud_water_freezing_rates(state: ThompsonColumnState, dt: float, cold_tables: ColdCollectionTables | None):
    """WRF source-stage freezing of cloud water to cloud ice.

    Thompson's non-aerosol mp8 path uses fixed cloud droplet number ``Nt_c`` and
    default ice nuclei, so the runtime only needs the reduced qcfz planes
    ``(idx_c, idx_tc)`` extracted from freezeH2O.dat.
    """

    rc = jnp.maximum(state.qc * state.rho, R1)
    idx_c = _lookup_digit_index(jnp.maximum(rc, 1.0e-6), -6, 37)
    idx_tc = jnp.clip(jnp.floor(-(state.T - 273.15) + 0.5).astype(jnp.int32) - 1, 0, N_TC_TABLE - 1)
    table_active = (state.qc > R1) & (rc > 1.0e-6)
    fallback_active = (state.qc > R1) & (rc <= 1.0e-6) & (state.T < HGFR)

    if cold_tables is None:
        pri_wfz = jnp.zeros_like(state.qc)
        pni_wfz = jnp.zeros_like(state.qc)
    else:
        pri_wfz = jnp.where(table_active, _take2(cold_tables.tpi_qcfz, idx_c, idx_tc) / float(dt), 0.0)
        pri_wfz = jnp.minimum(rc / float(dt), pri_wfz)
        pni_wfz = jnp.where(table_active, _take2(cold_tables.tni_qcfz, idx_c, idx_tc) / float(dt), 0.0)
        pni_wfz = jnp.minimum(jnp.minimum(NT_C / float(dt), pri_wfz / (2.0 * XM0I)), pni_wfz)

    pri_wfz = jnp.where(fallback_active, rc / float(dt), pri_wfz)
    pni_wfz = jnp.where(fallback_active, NT_C / float(dt), pni_wfz)
    cloud_freeze = pri_wfz * float(dt) / state.rho
    cloud_ni = pni_wfz * float(dt) / state.rho
    return cloud_freeze, cloud_ni, pri_wfz, pni_wfz


def _cold_collection_rates(
    state: ThompsonColumnState,
    dt: float,
    cold_tables: ColdCollectionTables,
) -> tuple:
    """v0.15 rain-collecting-snow (qr_acr_qs) + rain-collecting-graupel
    (qr_acr_qg) below 0 C (module_mp_thompson.F:2484-2548 + tendencies
    3058-3120).

    These two table-driven collection-collision processes convert RAIN to
    GRAUPEL (and exchange snow) when rain coexists with snow/graupel below
    freezing -- the dominant missing rain sink for the January Alpine
    Switzerland RAINNC surplus.  ``twet`` is the wet-bulb temperature; for the
    sub-freezing levels that drive this case ``twet == temp`` (WRF only solves
    the LCL wet-bulb on the warm ``k_melting`` levels), so the cold branch
    (twet < T_0) is selected by ``state.T < T_0``.  The warm rcs/rcg branch
    (rain melting snow/graupel below the melting line) is the small reverse
    contribution and is included with the same ``twet`` approximation.

    Rates are read from the bit-exact Fortran tables (no recomputation); mass and
    number tendencies are wired exactly as WRF's qrten/qsten/qgten/nrten/ngten.
    """

    odts = 1.0 / float(dt)

    rho = state.rho
    rr = jnp.maximum(state.qr * rho, R1)
    rs = jnp.maximum(state.qs * rho, R1)
    rg = jnp.maximum(state.qg * rho, R1)
    nr = jnp.maximum(state.Nr * rho, R2)

    # Rain slope/intercept for idx_r / idx_r1 (WRF 2319-2333).
    _, _nr_c, lamr, _ilamr, _mvd_r, _n0r, active_rain = _rain_distribution(state.qr, state.Nr, state.rho)
    rr_idx = jnp.maximum(rr, R_R_FIRST)
    n0r_exp = ORG1 * rr_idx / AM_R * lamr**CRE1
    idx_r = _lookup_digit_index(rr_idx, -6, N_R_TABLE)
    idx_r1 = _lookup_digit_index(n0r_exp, 6, N_R1_TABLE)

    # Snow content index idx_s (WRF 2340-2349); ntb_s = 37, r_s(1)=1e-6.
    idx_s = _lookup_digit_index(jnp.maximum(rs, 1.0e-6), -6, 37)
    # Snow temperature index idx_t (WRF 2257-2259): INT((tempc-2.5)/5)-1,
    # idx_t=MAX(1,-idx_t), idx_t=MIN(idx_t,ntb_t=9). 0-based here.
    tempc = state.T - 273.15
    idx_t_f = jnp.floor((tempc - 2.5) / 5.0) - 1.0  # INT toward -inf == floor for negatives
    # WRF uses INT() (truncation toward zero); for (tempc-2.5)/5 < 0 this differs
    # from floor.  Reproduce INT (truncate toward zero) exactly.
    val_t = (tempc - 2.5) / 5.0
    idx_t_int = jnp.trunc(val_t) - 1.0
    idx_t = jnp.clip(-idx_t_int, 1.0, 9.0).astype(jnp.int32) - 1  # 0-based

    # Graupel content index idx_g + intercept index idx_g1 (WRF 2355-2380).
    # mp8 single density (am_g = AM_G_MP8, mu_g=0): cge(1,1)=bm_g+1=4 and
    # (cgg(3)*ogg2*ogg1)**bm_g == 1, but WRF's leading ``ogg1`` remains
    # 1/Gamma(mu_g+1)=1/6 (lines 2367-2368), so
    #   N0_exp = (rg/am_g) * lamg**4 / 6.
    # ``lamg`` from the shared graupel distribution helper.
    _rg_d, _ng_d, lamg, _ilamg_d, _n0g_d, _act_g = _graupel_distribution(state.qg, state.Ng, state.rho)
    idx_g = _lookup_digit_index(jnp.maximum(rg, 1.0e-6), -6, 37)
    n0g_exp = ORG1 * rg / AM_G_MP8 * lamg ** 4.0
    idx_g1 = _lookup_digit_index(jnp.maximum(n0g_exp, 1.0e2), 2, 37)

    cold = state.T < T_0
    have_rain = active_rain & (rr >= R_R_FIRST)
    have_snow = state.qs > R1
    have_graupel = state.qg > R1

    # ---- Rain collecting snow (cold branch, twet<T_0 == state.T<T_0) ----
    racs_gate = have_rain & have_snow & cold
    tmr_racs2 = _take4(cold_tables.tmr_racs2, idx_s, idx_t, idx_r1, idx_r)
    tcr_sacr2 = _take4(cold_tables.tcr_sacr2, idx_s, idx_t, idx_r1, idx_r)
    tmr_racs1 = _take4(cold_tables.tmr_racs1, idx_s, idx_t, idx_r1, idx_r)
    tcr_sacr1 = _take4(cold_tables.tcr_sacr1, idx_s, idx_t, idx_r1, idx_r)
    tcs_racs1 = _take4(cold_tables.tcs_racs1, idx_s, idx_t, idx_r1, idx_r)
    tms_sacr1 = _take4(cold_tables.tms_sacr1, idx_s, idx_t, idx_r1, idx_r)
    tnr_racs1 = _take4(cold_tables.tnr_racs1, idx_s, idx_t, idx_r1, idx_r)
    tnr_racs2 = _take4(cold_tables.tnr_racs2, idx_s, idx_t, idx_r1, idx_r)
    tnr_sacr1 = _take4(cold_tables.tnr_sacr1, idx_s, idx_t, idx_r1, idx_r)
    tnr_sacr2 = _take4(cold_tables.tnr_sacr2, idx_s, idx_t, idx_r1, idx_r)

    prr_rcs = -(tmr_racs2 + tcr_sacr2 + tmr_racs1 + tcr_sacr1)
    prs_rcs = tmr_racs2 + tcr_sacr2 - tcs_racs1 - tms_sacr1
    prg_rcs = tmr_racs1 + tcr_sacr1 + tcs_racs1 + tms_sacr1
    prr_rcs = jnp.maximum(-rr * odts, prr_rcs)
    prs_rcs = jnp.maximum(-rs * odts, prs_rcs)
    prg_rcs = jnp.minimum((rr + rs) * odts, prg_rcs)
    pnr_rcs = jnp.minimum(nr * odts, tnr_racs1 + tnr_racs2 + tnr_sacr1 + tnr_sacr2)
    png_rcs = pnr_rcs

    prr_rcs = jnp.where(racs_gate, prr_rcs, 0.0)
    prs_rcs = jnp.where(racs_gate, prs_rcs, 0.0)
    prg_rcs = jnp.where(racs_gate, prg_rcs, 0.0)
    pnr_rcs = jnp.where(racs_gate, pnr_rcs, 0.0)
    png_rcs = jnp.where(racs_gate, png_rcs, 0.0)

    # ---- Rain collecting graupel (cold branch) ----
    racg_gate = have_rain & have_graupel & cold
    tmr_racg = _take4(cold_tables.tmr_racg, idx_g1, idx_g, idx_r1, idx_r)
    tcr_gacr = _take4(cold_tables.tcr_gacr, idx_g1, idx_g, idx_r1, idx_r)
    tnr_racg = _take4(cold_tables.tnr_racg, idx_g1, idx_g, idx_r1, idx_r)
    tnr_gacr = _take4(cold_tables.tnr_gacr, idx_g1, idx_g, idx_r1, idx_r)

    prg_rcg = jnp.minimum(rr * odts, tmr_racg + tcr_gacr)
    prr_rcg = -prg_rcg
    pnr_rcg = jnp.minimum(nr * odts, tnr_racg + tnr_gacr)
    prg_rcg = jnp.where(racg_gate, prg_rcg, 0.0)
    prr_rcg = jnp.where(racg_gate, prr_rcg, 0.0)
    pnr_rcg = jnp.where(racg_gate, pnr_rcg, 0.0)

    return prr_rcs, prs_rcs, prg_rcs, pnr_rcs, png_rcs, prr_rcg, prg_rcg, pnr_rcg


def _zero_cold_collection_rates(state: ThompsonColumnState) -> tuple:
    zero = jnp.zeros_like(state.qr)
    return zero, zero, zero, zero, zero, zero, zero, zero


def _scale_cold_collection_rain_rates(rates: tuple, ratio) -> tuple:
    """Apply WRF rain-conservation scaling to rain-mass cold terms.

    WRF scales ``prr_rcs`` and ``prr_rcg`` in the rain conservation block, then
    re-enforces ``prr_rcg = -prg_rcg``.  The cold ``prs_rcs/prg_rcs`` terms are
    intentionally not scaled by that rain limiter.
    """

    prr_rcs, prs_rcs, prg_rcs, pnr_rcs, png_rcs, prr_rcg, prg_rcg, pnr_rcg = rates
    prr_rcs = prr_rcs * ratio
    prr_rcg = prr_rcg * ratio
    prg_rcg = -prr_rcg
    return prr_rcs, prs_rcs, prg_rcs, pnr_rcs, png_rcs, prr_rcg, prg_rcg, pnr_rcg


def _apply_cold_collection_rates(state: ThompsonColumnState, dt: float, rates: tuple) -> ThompsonColumnState:
    # ---- Mass / number tendency assembly (WRF 3058-3120; only the rcs/rcg
    # terms; the other tendencies are already applied by the earlier blocks). ----
    prr_rcs, prs_rcs, prg_rcs, pnr_rcs, png_rcs, prr_rcg, prg_rcg, pnr_rcg = rates
    rho = state.rho
    orho = 1.0 / rho
    d_qr = (prr_rcs + prr_rcg) * float(dt) * orho
    d_qs = prs_rcs * float(dt) * orho
    d_qg = (prg_rcs + prg_rcg) * float(dt) * orho
    d_nr = -(pnr_rcs + pnr_rcg) * float(dt) * orho
    d_ng = png_rcs * float(dt) * orho  # WRF: ngten += png_rcs (cold png_rcg=0)

    new_qr = jnp.maximum(0.0, state.qr + d_qr)
    new_qs = jnp.maximum(0.0, state.qs + d_qs)
    new_qg = jnp.maximum(0.0, state.qg + d_qg)
    new_nr = jnp.maximum(0.0, state.Nr + d_nr)
    new_ng = jnp.maximum(0.0, state.Ng + d_ng)

    return state.replace(qr=new_qr, qs=new_qs, qg=new_qg, Nr=new_nr, Ng=new_ng)


def _cold_collection(
    state: ThompsonColumnState,
    dt: float,
    cold_tables: ColdCollectionTables,
) -> ThompsonColumnState:
    return _apply_cold_collection_rates(state, dt, _cold_collection_rates(state, dt, cold_tables))


def _ice_sources(state: ThompsonColumnState, dt: float, tables: ThompsonTableBundle = THOMPSON_TABLES) -> ThompsonColumnState:
    """Legacy wrapper for callers that do not need process flags."""

    updated, _graupel_melt, _vts_boost, _cold_rates = _ice_sources_with_process_flags(state, dt, tables)
    return updated


# Static UPPER BOUND on the per-column sedimentation sub-step count. WRF chooses
# the substep count *adaptively per column* from the CFL condition
# (module_mp_thompson.F:3634-3641,3791): nstep = MAX_k INT(DT/(dzq/vt) + 1) over
# levels with a sedimenting particle, then advances exactly ``nstep`` upwind
# substeps each of size DT/nstep.  We reproduce that EXACTLY with a masked
# fixed-length scan: the loop runs ``NSED_MAX`` iterations (a JIT-static upper
# bound), but per-column iteration ``n`` is a no-op once ``n >= nstep_col`` and
# each active substep uses the per-column ``DT/nstep_col`` -- so the integration
# is bit-faithful to WRF for any column whose ``nstep <= NSED_MAX``.
#
# Why this matters (PRECIP PARITY, P1-5): the PRIOR code ran a *fixed* 64
# substeps each of DT/64 with no per-column nstep.  Over-resolving the substep
# integrates more of the falling front out the surface face within one DT than
# WRF's coarser nstep does, biasing surface precip HIGH (+13% vs the WRF
# precipitating oracle).  Matching WRF's adaptive nstep collapses that bias.
#
# Cap sizing: explicit-upwind stability needs nstep >= vt*DT/dz.  The v0.10.0
# d02/d03 wet-column scan found max nstep=2 and zero clips at cap=16, so 16 is an
# 8x margin over the observed active corpus while still covering severe-column
# estimates (~8-12).  If a pathological column ever needs nstep > NSED_MAX the
# substep is silently capped at NSED_MAX (same behavior as the old cap=64 path,
# stable but slightly under-resolved) and counted as a sed-clip fallback by the
# validation harness.  ``GPUWRF_THOMPSON_NSED`` overrides the cap.
def _nsed_substeps() -> int:
    try:
        return max(1, int(os.environ.get("GPUWRF_THOMPSON_NSED", "16")))
    except ValueError:
        return 16


# Static upper bound on per-column substeps (loop length); the EFFECTIVE substep
# count is WRF's adaptive per-column ``nstep`` (computed in ``_nstep_per_column``).
NSED_MAX = _nsed_substeps()
# Backward-compatible alias (the precip-oracle harness + perf scripts read this).
NSED_SUBSTEPS = NSED_MAX


# WRF surface-precip accumulation threshold: only accumulate the bottom-face flux
# during a substep whose UPDATED surface-layer hydrometeor density exceeds
# R1*1000 = 1e-9 kg m^-3 (module_mp_thompson.F:3817,3868,3895,3936).  On the
# precipitating oracle this gate is a no-op (the surface rain density stays well
# above 1e-9), but it is WRF-faithful and prevents trace numerical drizzle from
# accumulating in lightly-precipitating columns.
RR_SURF_THRESHOLD = R1 * 1000.0


def _nstep_per_column(vt_a, vt_b, dz, dt):
    """WRF adaptive substep count per column (module_mp_thompson.F:3634-3641).

    ``nstep = MAX_k INT(DT/(dzq(k)/vt(k)) + 1)`` taken over levels where the
    governing fall speed exceeds 1e-3 m/s; columns with no sedimenting particle
    get ``nstep = 1`` (onstep = 1, a single pass).  ``vt_a``/``vt_b`` are the two
    speeds WRF maxes for the CFL test (mass & number for rain; for ice/snow/
    graupel pass the same array twice).  Axis -1 is vertical.  Returned as a
    float per column (== WRF's REAL(nstep)); clipped to [1, NSED_MAX].
    """

    vt = jnp.maximum(vt_a, vt_b)
    dz = jnp.maximum(dz, 1.0)
    active = vt > 1.0e-3
    # INT(DT/(dz/vt) + 1.) == floor(dt*vt/dz + 1) for the positive argument here.
    cand = jnp.floor(dt * vt / dz + 1.0)
    cand = jnp.where(active, cand, 0.0)
    nstep = jnp.max(cand, axis=-1)
    nstep = jnp.clip(nstep, 1.0, float(NSED_MAX))
    return nstep


def _sed_unroll() -> int:
    """Scan-unroll factor for the sedimentation substep loop.

    The sedimentation explicit-upwind substep scan is the single largest cost of
    the Thompson kernel (~85 % of it) and is LAUNCH/bandwidth-bound:
    ``NSED_SUBSTEPS`` sequential tiny dependent steps.  ``jax.lax.scan(...,
    unroll=U)`` replicates the substep body so XLA fuses U dependent steps into
    fewer, larger kernels, cutting the launch count.  The math is identical — the
    unrolled scan inlines iterations in order (no reassociation), so the result
    is BIT-IDENTICAL to ``unroll=1``.  Measured best at U=2 on the d02 workload
    (~1.1x; higher U adds compile cost with no further speedup).  Override with
    ``GPUWRF_THOMPSON_SED_UNROLL``.
    """

    try:
        return max(1, int(os.environ.get("GPUWRF_THOMPSON_SED_UNROLL", "2")))
    except ValueError:
        return 2


def _implicit_sed_nsub() -> int:
    """Number of backward-Euler implicit-sedimentation sweeps.

    ``GPUWRF_THOMPSON_IMPLICIT_SED`` selects the EXPERIMENTAL implicit
    (backward-Euler upwind) sedimentation instead of the faithful WRF explicit
    sub-stepped upwind.  Value = number of implicit sweeps (``0`` = OFF = faithful
    explicit default; ``1`` = single full-step BE; ``2``/``4`` reduce implicit
    diffusion at proportional cost).  This is a NUMERICAL SCHEME CHANGE (more
    diffusive vertical precip) gated behind this flag; default is OFF so the
    shipped kernel is the faithful explicit scheme, byte-identical to base.
    """

    try:
        return max(0, int(os.environ.get("GPUWRF_THOMPSON_IMPLICIT_SED", "0")))
    except ValueError:
        return 0


def _sed_implicit_q(q, vt, dz, rho, dt, nsub):
    """Backward-Euler upwind sedimentation of one field in ``nsub`` implicit sweeps.

    Axis -1 is vertical, index 0 = surface, last = model top.  Sedimentation flux
    enters a layer only from the layer ABOVE (higher index), so the implicit solve
    is a top->bottom bidiagonal recurrence:
        (1 + dt_s*vt_k/dz_k) q_k' = q_k + (dt_s/(rho_k dz_k)) rho_{k+1} vt_{k+1} q_{k+1}'
    Returns (q', surface_precip_mm).  Unconditionally stable; mass-conserving up to
    the surface flux.  See proofs/thompson_perf/implicit_sedimentation_prototype.py.
    """

    acc = jnp.result_type(q.dtype, vt.dtype, rho.dtype, dz.dtype)
    q_dt = q.dtype
    q = q.astype(acc)
    dts = float(dt) / nsub
    qr0 = jnp.moveaxis(q, -1, 0)[::-1]      # (z, ...) z=0 == model top
    vtr = jnp.moveaxis(vt, -1, 0)[::-1]
    rhor = jnp.moveaxis(rho, -1, 0)[::-1]
    dzr = jnp.moveaxis(dz, -1, 0)[::-1]
    nz = qr0.shape[0]
    diag = 1.0 + dts * vtr / dzr

    def one(qcur):
        def body(inflow_mass, k):
            qk = (qcur[k] + dts / (rhor[k] * dzr[k]) * inflow_mass) / diag[k]
            return rhor[k] * vtr[k] * qk, qk
        _, qsol = jax.lax.scan(body, jnp.zeros(qcur.shape[1:], acc), jnp.arange(nz))
        surf = rhor[nz - 1] * vtr[nz - 1] * qsol[nz - 1]  # bottom (surface) flux
        return jnp.maximum(qsol, 0.0), surf

    def step(carry, _):
        qc, sacc = carry
        qsol, surf = one(qc)
        return (qsol, sacc + surf * dts), None

    (qf, sf), _ = jax.lax.scan(step, (qr0, jnp.zeros(qr0.shape[1:], acc)), None, length=nsub)
    qf = jnp.moveaxis(jnp.maximum(qf, 0.0)[::-1], 0, -1)
    return qf.astype(q_dt), sf


def _sedimentation_implicit(state: ThompsonColumnState, dt: float, nsub: int, vts_boost=None):
    """EXPERIMENTAL implicit backward-Euler sedimentation (gated, default OFF)."""

    vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num = _fall_speeds(state, vts_boost)
    dz = jnp.maximum(state.dz, 1.0)
    rho = jnp.maximum(state.rho, R1)
    qr, pr = _sed_implicit_q(state.qr, vt_r_mass, dz, rho, dt, nsub)
    Nr, _ = _sed_implicit_q(state.Nr, vt_r_num, dz, rho, dt, nsub)
    qi, pi = _sed_implicit_q(state.qi, vt_i_mass, dz, rho, dt, nsub)
    Ni, _ = _sed_implicit_q(state.Ni, vt_i_num, dz, rho, dt, nsub)
    qs, ps = _sed_implicit_q(state.qs, vt_s_mass, dz, rho, dt, nsub)
    Ns, _ = _sed_implicit_q(state.Ns, vt_s_mass, dz, rho, dt, nsub)
    qg, pg = _sed_implicit_q(state.qg, vt_g_mass, dz, rho, dt, nsub)
    Ng, _ = _sed_implicit_q(state.Ng, vt_g_num, dz, rho, dt, nsub)
    updated = state.replace(qr=qr, Nr=Nr, qi=qi, Ni=Ni, qs=qs, Ns=Ns, qg=qg, Ng=Ng)
    return updated, {"rain": pr, "snow": ps, "graupel": pg, "ice": pi}


def _rho_correction(rho):
    """WRF air-density fall-speed correction rhof = sqrt(rho_not/rho)."""

    return jnp.sqrt(RHO_NOT / jnp.maximum(rho, R1))


def _fill_down(vt, active):
    """Fill inactive layers with the fall speed from the layer above.

    WRF sets ``vtrk(k)=vtrk(k+1)`` for layers without that hydrometeor
    (module_mp_thompson.F:3630-3631,3689-3690,3729,3769-3770), processing
    top->bottom so a falling blob keeps a non-zero speed in the empty layers it
    enters.  Axis -1 is vertical with index 0 = surface, so "above" = the next
    higher index; we scan from the top (last index) toward the surface.
    """

    vt_t = jnp.moveaxis(vt, -1, 0)  # (z, ...) with z=0 surface
    act_t = jnp.moveaxis(active, -1, 0)
    nz = vt_t.shape[0]

    def body(carry, k):
        prev = carry  # fall speed of the layer above (higher index), already filled
        kk = nz - 1 - k  # iterate top (nz-1) -> bottom (0)
        cur = jnp.where(act_t[kk], vt_t[kk], prev)
        return cur, cur

    init = jnp.zeros(vt_t.shape[1:], dtype=vt_t.dtype)
    _, filled_rev = jax.lax.scan(body, init, jnp.arange(nz))
    # filled_rev[k] corresponds to physical level nz-1-k; reverse to level order.
    filled = filled_rev[::-1]
    return jnp.moveaxis(filled, 0, -1)


def _fall_speeds(state: ThompsonColumnState, vts_boost=None):
    """Mass/number terminal fall speeds per species (m/s), WRF formulas.

    Rain:    module_mp_thompson.F:3616-3628 (vtrk mass, vtnrk number).
    Ice:     module_mp_thompson.F:3678-3691 (vtik mass, vtnik number).
    Snow:    module_mp_thompson.F:3711-3721 Field two-gamma moment-ratio mass
             speed; ``vts_boost`` is the WRF riming fall-speed factor (line
             3721 ``vts*vts_boost(k)``) produced by the v0.15 riming block
             (1.0 where riming is not dominant; the melt adjustment remains at
             its neutral value).
    Graupel: module_mp_thompson.F:3758-3766 mass speed with av_g/bv_g (idx_bg1).
    """

    rho = state.rho
    rhof = _rho_correction(rho)

    act_r = state.qr > R1
    rr = jnp.maximum(state.qr * rho, R1)
    # WRF computes the rain fall speeds from the mvd-clamped working number
    # ``nr(k)`` rebuilt at module_mp_thompson.F:3240-3250, NOT from the raw
    # prognostic Nr; the same clamp the rate-stage slopes use (_rain_distribution).
    nr = jnp.maximum(_clamp_rain_number(state.qr, state.Nr, rho) * rho, R2)
    lamr = (AM_R * CRG3 * ORG2 * nr / rr) ** OBMR
    vt_r_mass = rhof * AV_R * CRG6 * ORG3 * lamr ** CRE3 * ((lamr + FV_R) ** (-CRE6))
    vt_r_num = rhof * AV_R * CRG7 / CRG12 * lamr ** CRE12 * ((lamr + FV_R) ** (-CRE7))
    vt_r_mass = _fill_down(jnp.where(act_r, vt_r_mass, 0.0), act_r)
    vt_r_num = _fill_down(jnp.where(act_r, vt_r_num, 0.0), act_r)

    act_i = state.qi > R1
    ri = jnp.maximum(state.qi * rho, R1)
    ni = jnp.maximum(state.Ni * rho, R2)
    # cig(2) = Gamma(bm_i+mu_i+1) = Gamma(4) = 6 (WRF module_mp_thompson.F:695).
    lami = (AM_I * 6.0 * OIG1 * ni / ri) ** OBMI
    ilami = 1.0 / lami
    vt_i_mass = rhof * AV_I * CIG3 * OIG2 * ilami ** BV_I
    vt_i_num = rhof * AV_I * CIG6 / CIG7 * ilami ** BV_I
    vt_i_mass = _fill_down(jnp.where(act_i, vt_i_mass, 0.0), act_i)
    vt_i_num = _fill_down(jnp.where(act_i, vt_i_num, 0.0), act_i)

    act_s = state.qs > R1
    tempc = state.T - 273.15
    _rs2, xds, _smo0, _smo1, _smof, _csnow, _act_s = _snow_moments(state.qs, rho, tempc)
    vts_raw = _snow_terminal_velocity_wrf(rhof, xds, act_s)
    if vts_boost is not None:
        # WRF line 3721: vts = vts*vts_boost(k) (riming-dominant layers fall up
        # to 1.5x faster; boost==1.0 elsewhere and with riming disabled).
        vts_raw = vts_raw * vts_boost
    vt_s_mass = _fill_down(vts_raw, act_s)

    act_g = state.qg > R1
    _rg, _ng, _lamg, ilamg, _n0_g, _active_g = _graupel_distribution(state.qg, state.Ng, rho)
    vt_g_mass = _fill_down(jnp.where(act_g, rhof * AV_G_MP8 * 6.0 * ORG3 * ilamg ** BV_G_MP8, 0.0), act_g)
    vt_g_num = _fill_down(jnp.where(act_g, rhof * AV_G_MP8 * CRG7 / CRG12 * ilamg ** BV_G_MP8, 0.0), act_g)

    return (vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num)


def _sed_one_species(q, num, vt_mass, vt_num, dz, rho, dt, nstep):
    """One species' WRF-faithful adaptive-nstep upwind sedimentation.

    Mirrors WRF module_mp_thompson.F:3790-3939: ``nstep`` explicit upwind flux
    substeps, each advancing by ``DT/nstep`` (``onstep = 1/nstep``), with surface
    accumulation gated by the updated surface-layer density (>1e-9 kg m^-3).
    Axis -1 is vertical with index 0 = surface (kts), last index = model top
    (kte); precipitation leaves through the surface (index-0) face.

    ``nstep`` is the WRF per-column adaptive substep count (a float, == WRF's
    REAL(nstep); see ``_nstep_per_column``).  The scan runs a JIT-static
    ``NSED_MAX`` iterations; iteration ``n`` is a no-op for any column where
    ``n >= nstep`` (mask), so the result is bit-faithful to WRF for nstep<=NSED_MAX
    yet keeps a static loop length.  ``dt_sub = DT/nstep`` is per-column.

    Returns (q', num', surface_precip_mm), q'/num' cast back to the input dtype.
    Accumulation runs in the result dtype of (q, num, vt, rho, dz) — fp64 here —
    so sedimentation never silently downcasts the flux integration even when the
    incoming State fields are fp32 (ADR-007 fp32-gated).
    """

    acc_dtype = jnp.result_type(q.dtype, num.dtype, vt_mass.dtype, rho.dtype, dz.dtype)
    q_dt, num_dt = q.dtype, num.dtype
    q0 = q.astype(acc_dtype)
    num0 = num.astype(acc_dtype)
    nstep = jnp.asarray(nstep, acc_dtype)               # (...,) per column
    dt_sub = jnp.asarray(dt, acc_dtype) / nstep         # (...,) onstep*DT
    nstep_col = nstep[..., None]                          # broadcast over levels
    dt_sub_col = dt_sub[..., None]
    surf_thresh = jnp.asarray(RR_SURF_THRESHOLD, acc_dtype)

    def body(carry, n):
        q_c, num_c, ppt_c = carry
        # column-level mask: this substep is real only while n < nstep_col.
        live = (n < nstep).astype(acc_dtype)             # (...,)
        live_col = live[..., None]
        rq = jnp.maximum(q_c * rho, 0.0)
        rn = jnp.maximum(num_c * rho, 0.0)
        flux_q = vt_mass * rq  # kg m^-2 s^-1 (downward, positive)
        flux_n = vt_num * rn
        # Flux INTO a layer comes from the layer above (higher index); flux OUT
        # goes to the layer below (lower index, toward the surface at index 0).
        flux_q_above = jnp.concatenate(
            [flux_q[..., 1:], jnp.zeros_like(flux_q[..., :1])], axis=-1
        )
        flux_n_above = jnp.concatenate(
            [flux_n[..., 1:], jnp.zeros_like(flux_n[..., :1])], axis=-1
        )
        dq = (flux_q_above - flux_q) / dz / rho * dt_sub_col
        dn = (flux_n_above - flux_n) / dz / rho * dt_sub_col
        q_new = jnp.maximum(q_c + dq, 0.0)
        num_new = jnp.maximum(num_c + dn, 0.0)
        # apply the update only on live substeps (dead columns hold their state).
        q_c = jnp.where(live_col > 0, q_new, q_c)
        num_c = jnp.where(live_col > 0, num_new, num_c)
        # Surface flux leaving the bottom (index 0) face this substep (pre-update
        # density, WRF sed_r(kts) = vt*rr(kts)).  WRF gates the accumulation on
        # the UPDATED surface density rr(kts) > 1e-9 kg/m3.
        rr_surf_updated = jnp.maximum(q_c[..., 0] * rho[..., 0], 0.0)
        gate = (live > 0) & (rr_surf_updated > surf_thresh)
        surf = jnp.where(gate, flux_q[..., 0] * dt_sub, 0.0)  # kg m^-2 == mm
        return (q_c, num_c, ppt_c + surf), None

    zero_ppt = jnp.zeros(q.shape[:-1], dtype=acc_dtype)
    (q_out, num_out, ppt), _ = jax.lax.scan(
        body, (q0, num0, zero_ppt), jnp.arange(NSED_MAX, dtype=acc_dtype),
        unroll=_sed_unroll(),
    )
    return q_out.astype(q_dt), num_out.astype(num_dt), ppt


def _cloud_water_fall_speed(state: ThompsonColumnState):
    """Cloud-droplet mass terminal fall speed (m/s), WRF module_mp_thompson.F:3656-3664.

    ``vtc = rhof*av_c*ccg(5,nu_c)*ocg2(nu_c)*ilamc**bv_c`` with the mp=8 default
    ``nu_c = 12`` (fixed cloud number ``NT_C``), where ``lamc`` is the cloud
    gamma slope from :func:`_cloud_distribution`.  WRF only assigns a non-zero
    ``vtck`` where ``rc > R1`` AND the local vertical velocity ``w < 0.1 m/s``
    (line 3657: ``w1d(k) .lt. 1.E-1`` — cloud water does not sediment inside an
    updraft); elsewhere ``vtck`` stays 0 (NO fill-down, unlike rain/ice/snow/
    graupel — WRF leaves vtck=0 in inactive layers).
    """

    rho = jnp.maximum(state.rho, R1)
    rhof = _rho_correction(rho)
    rc = jnp.maximum(state.qc * rho, R1)
    # lamc identical to _cloud_distribution / WRF line 3659.
    lamc = (NT_C * AM_R * CCG2_NU12 * OCG1_NU12 / rc) ** OBMR
    ilamc = 1.0 / lamc
    vtc = rhof * AV_C * CCG5_NU12 * OCG2_NU12 * ilamc ** BV_C
    active = (state.qc > R1) & (state.w < 1.0e-1)
    return jnp.where(active, vtc, 0.0)


def _sed_cloud_water(state: ThompsonColumnState, dt: float):
    """WRF cloud-water sedimentation (module_mp_thompson.F:3824-3837).

    Distinct from rain/ice/snow/graupel sedimentation in three WRF-faithful ways:
      1. SINGLE full-DT explicit-upwind pass (no nstep substepping; WRF runs the
         cloud-water update once, ``onstep`` is not applied — lines 3829-3836).
      2. Confined to BELOW 500 m AGL: ``ksed1(5)`` is the top sedimenting level,
         and the kernel-side ``below_500m`` mask reproduces the WRF height cap
         (lines 3646-3653).  Layers above stay untouched.
      3. The bottom-face cloud-water flux ``sed_c(kts)`` leaves the column but is
         NOT accumulated into any surface-precip channel in WRF (no ``pptXXX +=``
         line) — so cloud-water sedimentation is a (small) water-budget sink that
         we report separately, never as precip.

    Returns ``(qc', cloudw_surface_loss_mm)``.  Axis -1 is vertical, index 0 ==
    surface (kts); flux enters a layer from the layer ABOVE (higher index).  The
    GPU column carries a FIXED cloud number (``NT_C``), so only the qc mass is
    advected here (WRF's ``nc`` redistribution is a no-op under fixed-Nc).
    """

    rho = jnp.maximum(state.rho, R1)
    dz = jnp.maximum(state.dz, 1.0)
    acc_dtype = jnp.result_type(state.qc.dtype, rho.dtype, dz.dtype)
    qc_dt = state.qc.dtype
    qc = state.qc.astype(acc_dtype)
    rho = rho.astype(acc_dtype)
    dz = dz.astype(acc_dtype)
    vtc = _cloud_water_fall_speed(state).astype(acc_dtype)
    dt_a = jnp.asarray(dt, acc_dtype)

    # Below-500 m-AGL mask: cumulative layer thickness from the surface (index 0)
    # excluding the current layer, matching WRF's hgt_agl accumulation that stops
    # once it exceeds 500 m (lines 3648-3653).  Cloud-water sedimentation acts
    # only on levels below this cap.
    hgt_agl = jnp.cumsum(dz, axis=-1) - dz  # bottom-of-layer AGL height
    below_500m = hgt_agl < 500.0

    # sed_c(k) = vtck(k)*rc(k); rc = qc*rho.  vtck is already gated (rc>R1 & w<0.1
    # & active cloud); confine the flux to the below-500 m band.
    rc = jnp.maximum(qc * rho, 0.0)
    sed_c = jnp.where(below_500m, vtc * rc, 0.0)  # kg m^-2 s^-1 (downward)
    # Flux INTO layer k comes from the layer above (k+1, higher index).
    sed_c_above = jnp.concatenate([sed_c[..., 1:], jnp.zeros_like(sed_c[..., :1])], axis=-1)
    # rc(k) += (sed_c(k+1) - sed_c(k))*odzq*DT  (single full-DT pass, WRF 3834).
    dq = (sed_c_above - sed_c) / dz / rho * dt_a
    qc_new = jnp.where(below_500m, jnp.maximum(qc + dq, 0.0), qc)
    # Bottom-face cloud-water flux leaving the column at the surface (kg m^-2 ==
    # mm).  WRF does NOT count this as precip; we return it as a water-budget sink.
    cloudw_surface_loss = sed_c[..., 0] * dt_a
    return qc_new.astype(qc_dt), cloudw_surface_loss.astype(jnp.float64)


def _sedimentation(state: ThompsonColumnState, dt: float, vts_boost=None):
    """Faithful WRF sedimentation of rain/ice/snow/graupel + cloud water; precip mm.

    WRF module_mp_thompson.F:3784-3939.  Advects the four precipitating channels
    (rain, snow, graupel, ice) with WRF's adaptive per-species ``nstep`` substep
    scan, plus the cloud-water fall term (single full-DT pass below 500 m AGL,
    :func:`_sed_cloud_water`).  Snow/graupel numbers (Ns/Ng) follow their mass
    since the mp=8 default carries diagnostic snow number and a fixed graupel
    intercept; Ng falls with the mass flux when present.

    Returns ``(state', precip-dict-mm)``.  ``precip`` carries the four surface
    precip channels (rain/snow/graupel/ice); cloud-water sedimentation does NOT
    contribute to surface precip in WRF, so its small surface loss is reported
    under ``cloudw`` (a water-budget SINK, never summed into the precip total).
    """

    nsub = _implicit_sed_nsub()
    if nsub > 0:
        # EXPERIMENTAL implicit backward-Euler sedimentation (gated, default OFF).
        return _sedimentation_implicit(state, dt, nsub, vts_boost)

    vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num = _fall_speeds(state, vts_boost)
    dz = jnp.maximum(state.dz, 1.0)
    rho = jnp.maximum(state.rho, R1)

    # WRF chooses an INDEPENDENT adaptive substep count per species from the CFL
    # of that species' fall speed (module_mp_thompson.F:3634/3693/3732/3773).
    # Rain maxes mass & number speeds; ice/snow/graupel use their governing
    # speed.  The masked scan then runs each species at its own nstep.
    nstep_r = _nstep_per_column(vt_r_mass, vt_r_num, dz, dt)
    nstep_i = _nstep_per_column(vt_i_mass, vt_i_mass, dz, dt)
    nstep_s = _nstep_per_column(vt_s_mass, vt_s_mass, dz, dt)
    nstep_g = _nstep_per_column(vt_g_mass, vt_g_mass, dz, dt)

    # Four independent per-species substep scans. XLA already overlaps these four
    # independent scans well; the per-scan ``unroll`` (``_sed_unroll``) fuses
    # adjacent substeps to cut the launch count. (A single 4-species batched scan
    # was measured SLOWER — it serialises what XLA otherwise parallelises — so we
    # keep the per-species structure; see proofs/thompson_perf.)
    # WRF sediments the mvd-clamped working rain number (module_mp_thompson.F:
    # 3240-3250 rebuilds ``nr(k)`` before the rain fall loop at 3790-3819); the
    # clamped field is what the upwind flux advects and what becomes the new
    # prognostic (re-balanced to the same band in ``_finish``, WRF 4046-4055).
    Nr_sed = _clamp_rain_number(state.qr, state.Nr, rho)
    qr, Nr, ppt_rain = _sed_one_species(state.qr, Nr_sed, vt_r_mass, vt_r_num, dz, rho, dt, nstep_r)
    qi, Ni, ppt_ice = _sed_one_species(state.qi, state.Ni, vt_i_mass, vt_i_num, dz, rho, dt, nstep_i)
    # Snow: number tracks mass (diagnostic Ns).  Use the mass speed for both.
    qs, Ns, ppt_snow = _sed_one_species(state.qs, state.Ns, vt_s_mass, vt_s_mass, dz, rho, dt, nstep_s)
    qg, Ng, ppt_graupel = _sed_one_species(state.qg, state.Ng, vt_g_mass, vt_g_num, dz, rho, dt, nstep_g)

    # Cloud-water fall term: single full-DT pass below 500 m AGL, NOT counted as
    # surface precip (WRF module_mp_thompson.F:3824-3837).  Reported under
    # ``cloudw`` as a water-budget sink so the closure budget stays exact.
    qc, ppt_cloudw = _sed_cloud_water(state, dt)

    updated = state.replace(qc=qc, qr=qr, Nr=Nr, qi=qi, Ni=Ni, qs=qs, Ns=Ns, qg=qg, Ng=Ng)
    precip = {
        "rain": ppt_rain,
        "snow": ppt_snow,
        "graupel": ppt_graupel,
        "ice": ppt_ice,
        "cloudw": ppt_cloudw,
    }
    return updated, precip


def _debug_checks(state: ThompsonColumnState, debug: bool) -> ThompsonColumnState:
    """Threads zero-production-cost debug assertions through the public kernel."""

    qv = assert_finite(state.qv, "thompson.qv", enabled=debug)
    qc = assert_physical_bounds(state.qc, 0.0, 1.0, "thompson.qc", enabled=debug)
    qr = assert_physical_bounds(state.qr, 0.0, 1.0, "thompson.qr", enabled=debug)
    qi = assert_physical_bounds(state.qi, 0.0, 1.0, "thompson.qi", enabled=debug)
    qs = assert_physical_bounds(state.qs, 0.0, 1.0, "thompson.qs", enabled=debug)
    qg = assert_physical_bounds(state.qg, 0.0, 1.0, "thompson.qg", enabled=debug)
    Ni = assert_physical_bounds(state.Ni, 0.0, 1.0e12, "thompson.Ni", enabled=debug)
    Nr = assert_physical_bounds(state.Nr, 0.0, 1.0e12, "thompson.Nr", enabled=debug)
    T = assert_physical_bounds(state.T, 50.0, 400.0, "thompson.T", enabled=debug)
    p = assert_physical_bounds(state.p, 1.0, 120000.0, "thompson.p", enabled=debug)
    rho = assert_finite(state.rho, "thompson.rho", enabled=debug)
    return state.replace(qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, Ni=Ni, Nr=Nr, T=T, p=p, rho=rho)


def _thompson_source_sink_body(state: ThompsonColumnState, dt: float, debug: bool, *, sediment: bool):
    """Shared Thompson column body; optionally runs sedimentation.

    WRF order (module_mp_thompson.F): stage rates/tendencies (2157-3247),
    cloud cond/evap (3399-3494), rain evaporation (3500-3558),
    sedimentation (3784-3939), instant melt/freeze (3941-3967),
    final write/balance (3969-4060). Returns ``(state, precip-dict-mm)`` with
    ``pptrain/pptsnow/pptgraul/pptice`` mapped to rain/snow/graupel/ice.

    Work precision: when GPUWRF_THOMPSON_FP32=1 the rate/integration math runs
    in fp32.  Inputs are cast to the work dtype on entry and the result is cast
    back to each leaf's storage dtype on exit, so the kernel's I/O contract is
    unchanged.  Default work dtype = fp64 (no-op cast, byte-identical to the
    prior behaviour).  fp32 was measured to give ~1.0x (this kernel is
    launch/bandwidth-bound, not arithmetic-bound) -- see ``_work_dtype``.
    """

    work = _work_dtype()
    storage_dtypes = {name: jnp.asarray(getattr(state, name)).dtype for name in ThompsonColumnState.__slots__}
    state = _cast_state(state, work)

    state = _clip_species(state)
    state = _reset_mp8_graupel_number(state)
    valid = _thermodynamically_admissible(state)
    fallback = state
    state = _debug_checks(state, debug)
    state = _warm_rain_collection(state, dt)
    cold_rates = (
        _cold_collection_rates(state, dt, COLD_COLLECTION_TABLES)
        if _cold_collection_enabled()
        else _zero_cold_collection_rates(state)
    )
    state, graupel_melt, vts_boost, cold_rates = _ice_sources_with_process_flags(
        state, dt, cold_collection_rates=cold_rates
    )
    if _cold_collection_enabled():
        # rain-collecting-snow / rain-collecting-graupel below 0 C: convert rain
        # to graupel where supercooled rain meets snow/graupel (WRF 2484-2548).
        # Computed after the freeze/riming staging pass, mirroring WRF's
        # single-pass rate staging before sedimentation.
        state = _apply_cold_collection_rates(state, dt, cold_rates)
    state, cloud_condensed = _saturation_adjustment_with_condensation(state, dt)
    state = _rain_evaporation(state, dt, skip_evaporation=cloud_condensed, graupel_melt=graupel_melt)
    if sediment:
        state, precip = _sedimentation(state, dt, vts_boost=vts_boost)
    else:
        zero = jnp.zeros(state.qv.shape[:-1], dtype=state.qv.dtype)
        precip = {"rain": zero, "snow": zero, "graupel": zero, "ice": zero}
    state = _instant_melt_freeze(state, dt)
    state = _finish(state)
    state = _select_state(valid, state, fallback)
    state = _restore_state(state, storage_dtypes)
    # Precip (surface accumulation, mm) tracks the fp64 accumulators downstream;
    # keep it fp64 regardless of work dtype so the per-step sum does not lose a
    # digit before it reaches the fp64-locked rain/snow/graupel/ice accumulators.
    precip = {k: jnp.asarray(v).astype(jnp.float64) for k, v in precip.items()}
    return _debug_checks(state, debug), precip


def _step_thompson_column_impl(state: ThompsonColumnState, dt: float, debug: bool) -> ThompsonColumnState:
    """Source/sink-only Thompson body (no sedimentation), returns State.

    Kept as the historical M5 source/sink subset entry so the analytic-column
    parity/invariant fixtures (which were generated without sedimentation) stay
    valid. Operational coupling uses :func:`step_thompson_column_with_precip`.
    """

    out, _precip = _thompson_source_sink_body(state, dt, debug, sediment=False)
    return out


@partial(jax.jit, static_argnames=("dt", "debug"))
def step_thompson_column(state: ThompsonColumnState, dt: float, *, debug: bool = False) -> ThompsonColumnState:
    """Advances one Thompson source/sink column step (no sedimentation)."""

    return _step_thompson_column_impl(state, dt, debug)


def _step_thompson_column_full_impl(state: ThompsonColumnState, dt: float, debug: bool):
    """Full WRF mp_gt_driver column body including sedimentation + precip."""

    return _thompson_source_sink_body(state, dt, debug, sediment=True)


def _maybe_tiled_thompson_full(state: ThompsonColumnState, dt: float, debug: bool):
    """Runs the full Thompson step over fixed-size leading-column tiles.

    The production column view (`_thompson_column_from_state`) is laid out
    ``(ny, nx, nz)`` — the horizontal grid in the LEADING axes, vertical
    trailing.  The Thompson kernel is per-column (every op is broadcast over the
    leading axes; sedimentation moves only along the trailing vertical axis), so
    flattening the leading horizontal axes to a single column axis ``(ny*nx, nz)``
    and reshaping the outputs back is a pure execution-shape change with no math,
    no clamps, no cross-column coupling.  When that flattened width exceeds one
    tile, we run the kernel over fixed-size column tiles under ``lax.scan`` to
    cap the per-step working set to ONE tile.

    Engages only when the flattened column count exceeds one tile
    (``GPUWRF_MP_COLUMN_TILE_COLS``, default 16384) — the production 128x128 case
    is exactly one 16384-col tile and stays byte-for-byte on the untiled graph.
    No Thompson op couples columns, so the tiled result is value-identical per
    column — exact-output gate in
    ``proofs/perf/v015/km_bench/mp_tiling_identity_fit.json``; pattern + VRAM
    precedent in ``proofs/v013/rrtmg_column_tile_vram_suite.json``.
    """

    profile = jnp.asarray(state.qv)
    # The column view always carries the vertical axis trailing; everything in
    # front of it is the (flattened) horizontal column axis.
    lead_shape = tuple(profile.shape[:-1])  # e.g. (ny, nx) or (ncol,)
    ncol = 1
    for d in lead_shape:
        ncol *= int(d)
    tiling_active = (
        _MP_COLUMN_TILING
        and _MP_COLUMN_TILE_COLS > 0
        and profile.ndim >= 2
        and ncol > _MP_COLUMN_TILE_COLS
    )
    if not tiling_active:
        return _step_thompson_column_full_impl(state, dt, debug)

    # Flatten the leading horizontal axes of every per-column leaf to a single
    # column axis so the generic column tiler can scan over fixed-size tiles;
    # leaves that do not share this leading shape (scalars, per-level constants)
    # pass through unchanged.
    def _flatten_leaf(a):
        a = jnp.asarray(a)
        if a.ndim >= len(lead_shape) + 1 and tuple(a.shape[: len(lead_shape)]) == lead_shape:
            return a.reshape((ncol,) + tuple(a.shape[len(lead_shape):]))
        return a

    def _restore_leaf(a):
        a = jnp.asarray(a)
        if a.ndim >= 1 and int(a.shape[0]) == ncol:
            return a.reshape(lead_shape + tuple(a.shape[1:]))
        return a

    flat_state = jax.tree_util.tree_map(_flatten_leaf, state)
    tiled = column_tiling.tiled_column_apply(
        lambda tile: _step_thompson_column_full_impl(tile, dt, debug),
        flat_state,
        ncol=ncol,
        tile_cols=int(_MP_COLUMN_TILE_COLS),
    )
    return jax.tree_util.tree_map(_restore_leaf, tiled)


@partial(jax.jit, static_argnames=("dt", "debug"))
def step_thompson_column_with_precip(state: ThompsonColumnState, dt: float, *, debug: bool = False):
    """Advances one full Thompson column step; returns ``(State, precip-dict-mm)``.

    This is the operational coupling entry: it runs the source/sink processes
    AND faithful sedimentation, returning the surface precipitation accumulated
    over the step (mm) per channel for the State precip accumulators.  Batches
    wider than ``GPUWRF_MP_COLUMN_TILE_COLS`` run tile-by-tile (VRAM cap,
    value-identical; see :func:`_maybe_tiled_thompson_full`).
    """

    return _maybe_tiled_thompson_full(state, dt, debug)
