"""JAX Thompson-column source/sink subset for M5-S1."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

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
    AV_G_OLD,
    BV_C,
    BV_G_MP8,
    BV_G_OLD,
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
    CGE9_OLD,
    CGE11,
    CGE11_OLD,
    CGG6_OVER_CGG3,
    CGG6_OVER_CGG3_OLD,
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
    T1_QG_QC_OLD,
    T1_QR_EV,
    T1_QR_QC,
    T1_QR_QI,
    T1_QS_QC,
    T1_QS_QI,
    T1_SUBL_QG,
    T1_SUBL_QS,
    T2_MELT_QG,
    T2_MELT_QG_OLD,
    T2_MELT_QS,
    T2_QR_EV,
    T2_QR_QI,
    T2_SUBL_QG,
    T2_SUBL_QG_OLD,
    T2_SUBL_QS,
    T_0,
    TNO,
    WRF_REAL_CONSTANTS,
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


configure_jax_x64()


# --- Work-precision control (ADR-007 fp32 microphysics) -----------------------
# The body can run its internal rate/integration math in fp32 while keeping the
# State storage dtype (the kernel inputs/outputs) unchanged: it casts every
# column leaf to the WORK dtype on entry and casts the result back to each leaf's
# storage dtype on exit.  Note ``p`` arrives fp64 (acoustic-locked), so without
# this explicit cast the whole body would promote to fp64 even with fp32 storage.
#
# HISTORICAL RESULT (proofs/thompson_perf, older tree): the blanket fp32 flag
# gave ~1.0x. Current P0-tree measurements (b-thompson BT03) remove 16x device
# ops but gain only 1.31-1.37x in full-call time: rate/layout fusions dominate
# after column loops are fused. Do not infer time savings from launch counts.
# On the older fixture the fp32 flag perturbed moist outputs by <= ~1
# fp32 ULP (rel <= 9e-7), at or below the WRF oracle's own fp32 storage
# granularity (WRF stores these fields in fp32).
#
# Default = fp64 (byte-for-byte the prior behaviour).  Set GPUWRF_THOMPSON_FP32=1
# to run the rate math in fp32.
def _work_dtype():
    """Return the dtype the rate/integration math runs in (fp64 default)."""

    return jnp.float32 if (os.environ.get("GPUWRF_THOMPSON_FP32", "0") == "1" or _native_real_enabled()) else jnp.float64


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


def _native_real_enabled() -> bool:
    return os.environ.get("GPUWRF_THOMPSON_NATIVE_REAL", "0") == "1"


def _wrf_constants_enabled() -> bool:
    """GPUWRF_THOMPSON_WRF_CONSTANTS (BD94): WRF's own REAL init constants on the native REAL path; f64 legacy unchanged."""
    return os.environ.get("GPUWRF_THOMPSON_WRF_CONSTANTS", "0") == "1" and _native_real_enabled()


def _wrf(name: str, legacy: float) -> float:
    """``legacy`` (the port's f64 fold) or, under the key, WRF's REAL value of the same constant (WRF_REAL_CONSTANTS)."""
    return WRF_REAL_CONSTANTS[name] if _wrf_constants_enabled() else legacy


def _am_r() -> float:
    return _wrf("AM_R", AM_R)


def _am_i() -> float:
    return _wrf("AM_I", AM_I)


def _d0i() -> float:
    return _wrf("D0I", D0I)


def _times_cgg63(x, cgg63):
    """x * cgg(6)/cgg(3) of the graupel mass fall speed; under the key WRF's per-op x*cgg(6)*ogg3 (:2417, :3758)."""
    if _wrf_constants_enabled():
        return x * WRF_REAL_CONSTANTS["CGG6_OLD"] * WRF_REAL_CONSTANTS["OGG3"]
    return x * cgg63


def _graupel_constants():
    """(av_g, bv_g, cgg6/cgg3, t1_qg_qc, cge9, t2_subl_qg, t2_melt_qg, cge11) of the mp8 graupel fall law.

    Native REAL uses WRF's av_g_old/bv_g_old (thompson_init :459-464, before cge/cgg :758); the legacy
    path keeps the hail-aware table entry it always used.
    """
    if _native_real_enabled():
        return (AV_G_OLD, BV_G_OLD, CGG6_OVER_CGG3_OLD, _wrf("T1_QG_QC_OLD", T1_QG_QC_OLD), _wrf("CGE9_OLD", CGE9_OLD),
                T2_SUBL_QG_OLD, _wrf("T2_MELT_QG_OLD", T2_MELT_QG_OLD), CGE11_OLD)
    return (AV_G_MP8, BV_G_MP8, CGG6_OVER_CGG3, T1_QG_QC, CGE9, T2_SUBL_QG, T2_MELT_QG, CGE11)


def _dpow(x, exponent):
    """``x**exponent`` for a constant exponent.

    Native REAL with a DOUBLE operand and an integral exponent uses products
    (lax.integer_pow): within a few DOUBLE ulp of pow, but no libdevice pow at
    1/64-rate fp64 (GPUWRF_THOMPSON_DPOW_PRODUCTS=0 restores pow for A/B).
    """
    exponent = float(exponent)
    if (exponent.is_integer() and _native_real_enabled()
            and os.environ.get("GPUWRF_THOMPSON_DPOW_PRODUCTS", "1") == "1"
            and jnp.result_type(x) == jnp.float64):
        return x ** int(exponent)
    return x ** exponent



def _real_state(state):
    return _cast_state(state, jnp.float32) if _native_real_enabled() else state


def _real_incr(*values):
    """WRF's REAL tendency rounding point for DOUBLE process increments (native REAL).

    mp_thompson accumulates DOUBLE rates*orho into REAL tendencies and updates the
    REAL state once; rounding each per-process increment where it is formed keeps
    the staged state, T, qv and rho REAL. Identity off native REAL.
    """
    if not _native_real_enabled():
        return values[0] if len(values) == 1 else values
    out = tuple(jnp.asarray(v).astype(jnp.float32) for v in values)
    return out[0] if len(out) == 1 else out


def _sed_prep_fused_enabled() -> bool:
    return os.environ.get("GPUWRF_THOMPSON_SED_PREP_FUSED", "0") == "1"


def _full_column_enabled() -> bool:
    return os.environ.get("GPUWRF_THOMPSON_FULL_COLUMN", "0") == "1"



def _column_sed_enabled() -> bool:
    """Trace-time gate; changing it requires clearing the caller's JIT cache."""
    return os.environ.get("GPUWRF_THOMPSON_COLUMN_SED", "0") == "1"


def _column_sed_fp32_enabled() -> bool:
    """WRF REAL sedimentation only; DOUBLE source/rate islands stay unchanged."""
    return _column_sed_enabled() and os.environ.get("GPUWRF_THOMPSON_SED_FP32", "0") == "1"


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
    lamr = (_am_r() * CRG3 * ORG2 * nr / rr) ** OBMR
    if _native_real_enabled():
        lamr = lamr.astype(jnp.float64)
    mvd_r = (3.0 + 0.672) / lamr
    if _native_real_enabled():
        mvd_r = mvd_r.astype(jnp.float32)  # WRF REAL mvd_r (:1589).
    mvd_clamped = jnp.minimum(2.5e-3, jnp.maximum(_wrf("D0R_X075", D0R * 0.75), mvd_r))
    out_of_band = mvd_clamped != mvd_r
    lamr_c = (3.0 + 0.672) / mvd_clamped
    if _native_real_enabled():
        lamr_c = lamr_c.astype(jnp.float64)
    nr_c = CRG2 * ORG3 * rr * _dpow(lamr_c, 3.0) / _am_r()
    if _native_real_enabled():
        nr_c = nr_c.astype(jnp.float32)  # WRF REAL nr(k)
    # Only the diagnostic number is rebuilt, and only when out of the band; the
    # per-kg clamped number is nr_c/rho.  qr<=R1 columns keep their Nr untouched.
    number = jnp.where(active & out_of_band, nr_c / rho, Nr)
    return number.astype(jnp.float32) if _native_real_enabled() else number


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
    lamr = (_am_r() * CRG3 * ORG2 * nr / rr) ** OBMR
    if _native_real_enabled():
        lamr = lamr.astype(jnp.float64)
    ilamr = 1.0 / lamr
    mvd_r = (3.0 + 0.672) / lamr
    if _native_real_enabled():
        mvd_r = mvd_r.astype(jnp.float32)
    n0_r = nr * ORG2 * _dpow(lamr, CRE2)
    active = (qr > R1) & (Nr > 0.0)
    return rr, nr, lamr, ilamr, mvd_r, n0_r, active


def _cloud_distribution(qc, rho):
    """Encapsulates the mp=8 cloud gamma terms used by Berry-Reinhardt."""

    rc = jnp.maximum(qc * rho, R1)
    lamc = (_wrf("LAMC_PREFIX", NT_C * AM_R * CCG2_NU12 * OCG1_NU12) / rc) ** OBMR
    if _native_real_enabled():
        lamc = lamc.astype(jnp.float64)
    xdc = jnp.maximum(D0C * 1.0e6, (rc / (_am_r() * NT_C)) ** OBMR * 1.0e6)
    mvd_c = (3.0 + NU_C_MP8 + 0.672) / lamc
    if _native_real_enabled():
        mvd_c = mvd_c.astype(jnp.float32)  # WRF REAL mvd_c (:1589).
    mvd_c = jnp.maximum(D0C, jnp.minimum(mvd_c, D0R))
    return rc, lamc, xdc, mvd_c, qc > R1


def _ice_distribution(qi, Ni, rho):
    """Encapsulates WRF cloud-ice particle diameter terms from lines 2711-2715."""

    ri = jnp.maximum(qi * rho, R1)
    ni = jnp.maximum(Ni * rho, R2)
    lami_missing = CIE2 / 5.0e-6
    ni_missing = jnp.minimum(999.0e3, OIG2 * ri / _am_i() * _dpow(lami_missing, 3.0))
    ni = jnp.where((qi > R1) & (ni <= R2), ni_missing, ni)
    lami = (_am_i() * 6.0 * OIG1 * ni / ri) ** OBMI
    if _native_real_enabled():
        lami = lami.astype(jnp.float64)
    xdi_raw = (3.0 + 0.0 + 1.0) / lami
    if _native_real_enabled():
        xdi_raw = xdi_raw.astype(jnp.float32)
    lami_small = CIE2 / 5.0e-6
    ni_small = jnp.minimum(999.0e3, OIG2 * ri / _am_i() * _dpow(lami_small, 3.0))
    lami_large = CIE2 / 300.0e-6
    ni_large = OIG2 * ri / _am_i() * _dpow(lami_large, 3.0)
    ni = jnp.where((qi > R1) & (xdi_raw < 5.0e-6), ni_small, ni)
    ni = jnp.where((qi > R1) & (xdi_raw > 300.0e-6), ni_large, ni)
    lami = (_am_i() * 6.0 * OIG1 * ni / ri) ** OBMI
    if _native_real_enabled():
        lami = lami.astype(jnp.float64)
    ilami = 1.0 / lami
    xdi = jnp.maximum(_d0i(), (3.0 + 0.0 + 1.0) * ilami)
    if _native_real_enabled():
        xdi = xdi.astype(jnp.float32)  # WRF REAL xDi/xmi (:1599,1613).
    xmi = _am_i() * _dpow(xdi, 3.0)
    return ri, ni, lami, ilami, xdi, xmi, qi > R1


def _balance_ice_number(qi, Ni, rho):
    """WRF working-ice-number size balance (module_mp_thompson.F:3033-3055).

    The ice analogue of :func:`_clamp_rain_number`.  WRF re-balances cloud-ice
    mass against number BEFORE the working ``ri``/``ni`` pair is formed
    (3226-3234) and therefore before ice fall speeds (3678-3691) and the
    adaptive sedimentation substep count (3693-3697) are built from it.  The
    balance keeps the mass-weighted mean ice diameter inside 5-300 um and the
    number below 999e3 m^-3.

    WRF writes the balance as a tendency correction
    ``niten = (xni - ni1d*rho)*odts*orho``; since ``odts = 1/dtsave`` the
    post-tendency number density is exactly ``xni``, so returning ``xni`` is an
    exact restatement of the source, not an approximation.

    Why this is load-bearing and not cosmetic: ``lami`` is built from the
    number/mass ratio, so an active layer carrying ice mass with a near-zero
    number implies an enormous mean diameter and hence an enormous terminal
    fall speed.  WRF's ``vtik(k)=vtik(k+1)`` fill-down then carries that speed
    down into the thin near-surface layers, where ``nstep = INT(DT/(dzq/vt)+1)``
    explodes.  Balancing FIRST is what stops the unphysical speed from ever
    being formed -- the port previously applied this same algebra only in
    ``_finish``, i.e. AFTER sedimentation had already run on the raw pair.

    This is a WORKING-number balance (it never changes the ice MASS); the
    prognostic ice number is re-derived from the same band in ``_finish``
    (WRF 4023-4038).  Returns the balanced ice number (per-kg), byte-unchanged
    wherever the (qi, Ni) pair is already inside the band.
    """

    ri = jnp.maximum(qi * rho, R1)
    ni = jnp.maximum(Ni * rho, R2)
    # WRF's `if (xri .gt. R1)` test is on the mass DENSITY xri = MAX(R1, qi*rho).
    active = ri > R1
    lami = (_am_i() * 6.0 * OIG1 * ni / ri) ** OBMI
    if _native_real_enabled():
        lami = lami.astype(jnp.float64)
    xdi = CIE2 / lami
    if _native_real_enabled():
        xdi = xdi.astype(jnp.float32)  # WRF REAL xDi
    too_small = xdi < 5.0e-6
    too_large = xdi > 300.0e-6
    # cig(1) = Gamma(mu_i+1) = 1 and bm_i = 3, written as literals to match the
    # surrounding ice code (_ice_distribution, _finish).
    ni_small = jnp.minimum(999.0e3, OIG2 * ri / _am_i() * (CIE2 / 5.0e-6) ** 3.0)
    ni_large = OIG2 * ri / _am_i() * (CIE2 / 300.0e-6) ** 3.0
    ni_bal = jnp.where(too_small, ni_small, jnp.where(too_large, ni_large, ni))
    # Trailing number ceiling, WRF 3053-3055.
    over_ceiling = ni_bal > 999.0e3
    ni_bal = jnp.where(over_ceiling, 999.0e3, ni_bal)
    # Only rebuild where the band was actually violated, so in-band columns keep
    # their exact prognostic Ni (no *rho -> /rho round-trip).  WRF's inactive
    # branch (`niten = -ni1d*odts`, and the 999e3 cap on inactive layers) is
    # already carried by ``_finish``, which zeroes Ni wherever qi <= R1.
    rebuilt = active & (too_small | too_large | over_ceiling)
    return jnp.where(rebuilt, ni_bal / rho, Ni)


def _lookup_digit_index(values, first_power: int, size: int):
    """Matches WRF's decade/digit table indexes without a search loop."""

    safe = jnp.maximum(values, jnp.finfo(jnp.asarray(values).dtype).tiny)
    decade = jnp.floor(jnp.log10(safe))
    digit = jnp.floor(safe / (10.0**decade))
    if _native_real_enabled():
        # WRF forms the index in INTEGER arithmetic (exact: the floors are integers).
        index = digit.astype(jnp.int32) + 9 * (decade.astype(jnp.int32) - first_power) - 1
        return jnp.clip(index, 0, size - 1)
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

    if _native_real_enabled():
        order = jnp.asarray(order, jnp.float32)  # WRF cse is REAL (:403).
    sa = tables.snow_sa.astype(smo2.dtype) if _native_real_enabled() else tables.snow_sa
    sb = tables.snow_sb.astype(smo2.dtype) if _native_real_enabled() else tables.snow_sb
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
    order_c = tables.cse[0].astype(smo2.dtype) if _native_real_enabled() else tables.cse[0]
    smoc = _snow_moment(order_c, smo2, tc0, tables)
    order_f = tables.cse[15].astype(smo2.dtype) if _native_real_enabled() else tables.cse[15]
    smof = _snow_moment(order_f, smo2, tc0, tables)
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
    denominator = jnp.where(active_snow, t3_vts + t4_vts, jnp.ones_like(t3_vts))
    vts = rhof * AV_S * (t1_vts + t2_vts) / denominator
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
    ng_m3 = ORG3 * rg * _dpow(lamg, 3.0) / AM_G_MP8
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
    if _native_real_enabled():
        lamg = lamg.astype(jnp.float64)
    mvd_g = (3.0 + 0.672) / lamg
    if _native_real_enabled():
        mvd_g = mvd_g.astype(jnp.float32)
    mvd_g = jnp.clip(mvd_g, D0R, 25.4e-3)
    lamg = (3.0 + 0.672) / mvd_g
    if _native_real_enabled():
        lamg = lamg.astype(jnp.float64)
    ng = ORG3 * rg * _dpow(lamg, 3.0) / AM_G_MP8
    if _native_real_enabled():
        ng = ng.astype(jnp.float32)  # WRF REAL ng (:1580-1582).
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
    prs_sci = jnp.where(sci_gate, _wrf("T1_QS_QI", T1_QS_QI) * rhof * EF_SI * ri * smoe, 0.0)
    if _native_real_enabled():
        prs_sci = prs_sci.astype(jnp.float64)  # WRF DOUBLE prs_sci (pni_sci follows)
    pni_sci = prs_sci * oxmi

    rci_gate = cold & active_ice & active_rain & (rr >= R_R_FIRST) & (mvd_r > 4.0 * xdi)
    lamr_fv = jnp.maximum(lamr + FV_R, R1)
    pri_rci = jnp.where(rci_gate, rhof * _wrf("T1_QR_QI", T1_QR_QI) * EF_RI * ri * n0_r * _dpow(lamr_fv, -CRE9), 0.0)
    pnr_rci = jnp.where(rci_gate, rhof * _wrf("T1_QR_QI", T1_QR_QI) * EF_RI * ni * n0_r * _dpow(lamr_fv, -CRE9), 0.0)
    pnr_rci = jnp.minimum(nr * odts, pnr_rci)
    pni_rci = pri_rci * oxmi
    prr_rci = jnp.where(rci_gate, rhof * _wrf("T2_QR_QI", T2_QR_QI) * EF_RI * ni * n0_r * _dpow(lamr_fv, -CRE8), 0.0)
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
    # WRF caps xni at 999e3 BEFORE forming lami (module_mp_thompson.F:3043), making
    # _finish idempotent against a large finite Ni (defense-in-depth).
    xni = jnp.minimum(jnp.maximum(R2, ni_raw * rho), 999.0e3)
    lami = (_am_i() * 6.0 * OIG1 * xni / ri) ** OBMI
    xdi = 4.0 / lami
    lami = jnp.where(xdi < 5.0e-6, CIE2 / 5.0e-6, lami)
    lami = jnp.where(xdi > 300.0e-6, CIE2 / 300.0e-6, lami)
    Ni = jnp.where(qi <= R1, 0.0, jnp.minimum((ri / _am_i() * _dpow(lami, 3.0) * OIG2) / rho, 999.0e3 / rho))

    nr_raw = jnp.maximum(R2 / rho, state.Nr)
    rr = jnp.maximum(qr * rho, R1)
    xnr = jnp.maximum(R2, nr_raw * rho)
    lamr = (_am_r() * CRG3 * ORG2 * xnr / rr) ** OBMR
    mvd_r = (3.0 + 0.672) / lamr
    mvd_r = jnp.minimum(2.5e-3, jnp.maximum(_wrf("D0R_X075", D0R * 0.75), mvd_r))
    lamr = (3.0 + 0.672) / mvd_r
    Nr = jnp.where(qr <= R1, 0.0, CRG2 * ORG3 * rr * _dpow(lamr, 3.0) / _am_r() / rho)
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


def _condensation_branch(state):
    """WRF cloud condensation/evaporation branch activation (module_mp_thompson.F:3401-3402):
    ``(ssatw > eps) .or. (ssatw < -eps .and. L_qc(k))`` with ``L_qc(k) = (qc1d+qcten*DT) > R1`` (:3215-3219).
    Shared by the mp=8 and mp=28 adjustments and by :func:`_wrf_l_qc_any`, so all three use one mask."""

    qvs = saturation_mixing_ratio_liquid(state.p, state.T)
    ssatw = state.qv / qvs - 1.0
    return (ssatw > EPS) | ((ssatw < -EPS) & (state.qc > R1))


def _saturation_adjustment_with_condensation(state: ThompsonColumnState, dt: float, *, vapor=None):
    """Implements the 3-iteration Thompson cloud condensation adjustment."""

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
    active = _condensation_branch(state)   # WRF L_qc is qc > R1 (was qc > 0.0)
    clap = jnp.where(active, clap, 0.0)
    clap = jnp.where(clap < 0.0, jnp.maximum(clap, -state.qc), jnp.minimum(clap, state.qv - 1.0e-10))
    condensed_cloud = clap > EPS
    qv = state.qv - clap
    if vapor is not None:
        # WRF :3479/:3488 accumulates a REAL tendency against raw qv1d,
        # while the Newton solve above uses the floored private work array.
        qv1d, qvten = vapor
        odt = np.float32(1.0) / np.float32(dt)
        qvten = (qvten.astype(jnp.float64) - (clap * odt).astype(jnp.float64)).astype(jnp.float32)
        qv = jnp.where(qv1d < 1.0e-10, jnp.maximum(1.0e-10, qv1d + float(dt) * qvten), qv)
    T = state.T + lvap * ocp * clap
    adjusted = state.replace(
        qv=qv,
        qc=state.qc + clap,
        T=T,
        rho=density_from_pressure_temperature(state.p, T, qv),
    )
    if vapor is not None:
        return adjusted, condensed_cloud, qvten
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
    dc_g = (_wrf("DC_G_PREFIX", (CCG3_NU12 * OCG2_NU12) ** OBMR) / lamc) * 1.0e6
    if _native_real_enabled():
        dc_g = dc_g.astype(jnp.float32)  # WRF REAL Dc_g: the Berry-Reinhardt chain is REAL
    dc_b = jnp.maximum(xdc**3 * dc_g**3 - xdc**6, 0.0) ** (1.0 / 6.0)
    zeta1_raw = 6.25e-6 * xdc * dc_b**3 - 0.4
    zeta1 = 0.5 * (zeta1_raw + jnp.abs(zeta1_raw))
    zeta = 0.027 * rc * zeta1
    taud_raw = 0.5 * dc_b - 7.5
    taud = 0.5 * (taud_raw + jnp.abs(taud_raw)) + R1
    tau = 3.72 / jnp.maximum(rc * taud, R1)
    prr_wau = jnp.where((rc > 0.01e-3) & active_cloud, jnp.minimum(rc / float(dt), zeta / tau), 0.0)
    if _native_real_enabled():
        prr_wau = prr_wau.astype(jnp.float64)  # REAL zeta/tau stored in WRF's DOUBLE prr_wau
    # WRF: pnr_wau = prr_wau / (am_r*nu_c*10.*D0r**3) (module_mp_thompson.F:2191).
    # The divisor is a strictly-positive constant (~7.85e-9), so it needs NO floor;
    # a prior ``jnp.maximum(..., R2=1e-6)`` clamp here silently REPLACED the true
    # 7.85e-9 divisor with 1e-6, shrinking the autoconversion rain-number source by
    # ~127x and starving Nr in cloud-base / autoconversion-dominated columns.
    pnr_wau = prr_wau / _wrf("PNR_WAU_DEN", AM_R * NU_C_MP8 * 10.0 * D0R**3)

    # WRF rain-collecting-cloud-water shape from t_Efrw, initialized at
    # module_mp_thompson.F.pre:4921-4977 and consumed at lines 2260-2268.
    idx_r_eff = jnp.clip(
        jnp.floor(N_EFRW_R * jnp.log(jnp.maximum(mvd_r, DR_FIRST) / DR_FIRST) / jnp.log(DR_LAST / DR_FIRST)),
        0,
        N_EFRW_R - 1,
    ).astype(jnp.int32)
    idx_c_eff = jnp.clip(jnp.floor(mvd_c * 1.0e6).astype(jnp.int32) - 1, 0, N_EFRW_C - 1)
    ef_rw = _take2(tables.t_Efrw, idx_r_eff, idx_c_eff)
    if _native_real_enabled():
        ef_rw = ef_rw.astype(jnp.float32)  # WRF REAL Ef_rw read from the DOUBLE t_Efrw
    prr_rcw_raw = rhof * _wrf("T1_QR_QC", T1_QR_QC) * ef_rw * rc * n0_r * _dpow(lamr + FV_R, -CRE9)
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
    return _apply_warm_rain_rates(state, dt, prr_wau, pnr_wau, prr_rcw, pnr_rcr)


def _apply_warm_rain_rates(state, dt, prr_wau, pnr_wau, prr_rcw, pnr_rcr):
    """Warm-rain transfer and rain-number band cap (zero rates: the cap alone)."""
    autoconv = prr_wau * float(dt) / state.rho
    accretion = prr_rcw * float(dt) / state.rho
    autoconv, accretion = _real_incr(autoconv, accretion)
    transfer = jnp.minimum(state.qc, autoconv + accretion)
    nr_gain = pnr_wau * float(dt) / state.rho
    nr_rcr = pnr_rcr * float(dt) / state.rho
    nr_gain, nr_rcr = _real_incr(nr_gain, nr_rcr)
    # Floor the post-process rain number at 0 (WRF carries nrten then re-floors at
    # MAX(R2/rho, ...) in _finish; a self-collection sink must not drive Nr<0).
    Nr_new = jnp.maximum(0.0, state.Nr + nr_gain - nr_rcr)
    new_state = state.replace(qc=state.qc - transfer, qr=state.qr + transfer, Nr=Nr_new)
    # WRF re-balances the rain number to the mvd band EVERY step
    # (module_mp_thompson.F:3070-3091): cap Nr at the number implied by the
    # SMALLEST allowed mvd (D0r*0.75) for the current rain mass.  The autoconversion
    # rain-number source (pnr_wau) was floored-only, leaving the upward Nr write
    # unbounded -> Nr runaway -> overflow.  Bit-identical below the band max (same
    # CRG2, ORG3, AM_R, D0R, R1 constants _finish uses).
    _rr_cap = jnp.maximum(new_state.qr * new_state.rho, R1)
    _nr_max = CRG2 * ORG3 * _rr_cap * ((3.0 + 0.672) / (D0R * 0.75)) ** 3.0 / _am_r() / new_state.rho
    return new_state.replace(Nr=jnp.maximum(0.0, jnp.minimum(new_state.Nr, _nr_max)))


def _rain_evaporation(
    state: ThompsonColumnState,
    dt: float,
    skip_evaporation: jax.Array | bool = False,
    graupel_melt: jax.Array | float = 0.0,
    *, vapor=None,
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
        * (T1_QR_EV * _dpow(ilamr, CRE10) + T2_QR_EV * vsc2 * rhof2 * _dpow(lamr + 0.5 * FV_R, -CRE11))
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
    if vapor is not None:
        return _apply_rain_evaporation(state, evap, lvap, ocp, vapor=vapor,
                                      vapor_rate=jnp.where(active, limited_rate, 0.0), dt=dt)
    return _apply_rain_evaporation(state, evap, lvap, ocp)


def _apply_rain_evaporation(state, evap, lvap, ocp, *, vapor=None, vapor_rate=None, dt=None):
    """Rain evaporation update for an evaporated amount (zero: Nr floor only)."""
    nr_loss = jnp.where(state.qr > 0.0, jnp.minimum(state.Nr * 0.99, state.Nr * evap / jnp.maximum(state.qr, R1)), 0.0)
    if _native_real_enabled():
        heat = lvap * ocp * evap
        evap, nr_loss, heat = _real_incr(evap, nr_loss, heat)
        out = state.replace(qv=state.qv + evap, qr=state.qr - evap,
                            Nr=jnp.maximum(0.0, state.Nr - nr_loss), T=state.T - heat)
        if vapor is not None:
            # WRF :3563/:3569/:3572: DOUBLE evaporation rate -> REAL
            # cumulative tendency -> raw baseline update -> working density.
            qv1d, qvten = vapor
            qvten = (qvten.astype(jnp.float64) + vapor_rate).astype(jnp.float32)
            affected = qv1d < 1.0e-10
            qv = jnp.where(affected, jnp.maximum(1.0e-10, qv1d + float(dt) * qvten), out.qv)
            rho = jnp.where(affected, density_from_pressure_temperature(out.p, out.T, qv), out.rho)
            return out.replace(qv=qv, rho=rho), qvten
        return out
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
        # WRF caps ice number at 999e3 m^-3 (module_mp_thompson.F:3054-3055). The
        # qc_freeze/XM0I number estimate (frozen mass / minimum crystal mass) is
        # unbounded and can spike Ni far past the WRF ceiling on steep nest columns,
        # leading to a non-finite Ni downstream. Enforce the WRF invariant here too
        # (per-mass = 999e3/rho). No-op below the ceiling => bit-identical on finite cases.
        Ni=jnp.minimum(state.Ni + qc_freeze / XM0I, 999.0e3 / state.rho),
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
    n0_exp = ORG1 * rr_for_index / _am_r() * _dpow(lam_exp, CRE1)
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
    (ice_freeze, graupel_freeze, cloud_freeze, table_ni, fallback_ni, cloud_ni, nr_loss) = _real_incr(
        ice_freeze, graupel_freeze, cloud_freeze, table_ni, fallback_ni, cloud_ni, nr_loss)
    state = state.replace(
        qr=state.qr - ice_freeze - graupel_freeze,
        qc=state.qc - cloud_freeze,
        qi=state.qi + ice_freeze + cloud_freeze,
        qg=state.qg + graupel_freeze,
        # Enforce the WRF 999e3 m^-3 ice-number ceiling (mp_thompson.F:3054-3055) on
        # this freezing/nucleation update too (per-mass = 999e3/rho). No-op below the
        # ceiling => bit-identical on already-finite cases.
        Ni=jnp.minimum(state.Ni + table_ni + fallback_ni + cloud_ni, 999.0e3 / state.rho),
        Nr=jnp.maximum(0.0, state.Nr - nr_loss),
        T=state.T + lfus2 * ocp * (ice_freeze + graupel_freeze + cloud_freeze),
    )
    state = state.replace(rho=density_from_pressure_temperature(state.p, state.T, state.qv))
    state = _real_state(state)  # WRF keeps temp/qv/rho/species in REAL arrays between stages

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
    if _native_real_enabled():
        pri_inu = pri_inu.astype(jnp.float64)  # WRF DOUBLE pri_inu
    pni_inu = jnp.where(deposition_nucleation_active, pri_inu / XM0I, 0.0)
    inu_mass = pri_inu * float(dt) / state.rho
    inu_number = pni_inu * float(dt) / state.rho

    tempc, diffu, _visco, tcond, _lvap, ocp, rhof, rhof2, vsc2 = _air_properties(state)
    del rhof
    qvs0 = saturation_mixing_ratio_liquid(
        state.p, jnp.asarray(T_0, state.T.dtype) if _native_real_enabled() else T_0)  # WRF RSLF is REAL
    del_qvs = jnp.maximum(0.0, qvs0 - state.qv)
    twet = state.T
    rs, _xds, smo0, smo1, smof, _c_snow, active_snow = _snow_moments(state.qs, state.rho, tempc, tables)
    rg, ng, lamg_g, ilamg, n0_g, active_graupel = _graupel_distribution(state.qg, state.Ng, state.rho)

    # Snow/graupel melting formula structure from WRF lines 2845-2889.
    prr_sml_rate = (tempc * tcond - 2.5e6 * diffu * del_qvs) * (_wrf("T1_MELT_QS", T1_MELT_QS) * smo1 + _wrf("T2_MELT_QS", T2_MELT_QS) * rhof2 * vsc2 * smof)
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
    _av_g, _bv_g, _cgg63, _t1_qg_qc, _cge9, _t2_subl_qg, t2_melt_qg, cge11 = _graupel_constants()
    prr_gml_rate = (tempc * tcond - 2.5e6 * diffu * del_qvs) * n0_melt * (
        T1_MELT_QG * _dpow(ilamg, CRE10) + t2_melt_qg * rhof2 * vsc2 * ilamg**cge11
    )
    prr_gml_rate = jnp.minimum(rg / float(dt), jnp.maximum(0.0, prr_gml_rate)) / state.rho
    graupel_melt = jnp.where((state.T > T_0) & active_graupel, prr_gml_rate * float(dt), 0.0)
    pnr_gml = jnp.where(rg > R1, graupel_melt * ng / rg * 10.0 ** (-0.33 * (twet - T_0)), 0.0)
    snow_melt, graupel_melt, pnr_sml, pnr_gml = _real_incr(snow_melt, graupel_melt, pnr_sml, pnr_gml)
    state = state.replace(
        qs=state.qs - snow_melt,
        qg=state.qg - graupel_melt,
        qr=state.qr + snow_melt + graupel_melt,
        Nr=state.Nr + pnr_sml + pnr_gml,
        T=state.T - LFUS * ocp * (snow_melt + graupel_melt),
    )
    state = state.replace(rho=density_from_pressure_temperature(state.p, state.T, state.qv))
    # WRF re-balances the rain number to the mvd band EVERY step
    # (module_mp_thompson.F:3070-3091): nr is capped at the number implied by the
    # SMALLEST allowed mvd (D0r*0.75) for the current rain mass.  The port only
    # re-derived this in _finish, leaving the snow/graupel melt source
    # (smo0/rs * 10**...) unbounded -> Nr runaway -> overflow on steep
    # cold->warming nest columns.  Bit-identical below the band max (same CRG2,
    # ORG3, AM_R, D0R, R1 constants _finish uses).
    _rr_cap = jnp.maximum(state.qr * state.rho, R1)
    _nr_max = CRG2 * ORG3 * _rr_cap * ((3.0 + 0.672) / (D0R * 0.75)) ** 3.0 / _am_r() / state.rho
    state = state.replace(Nr=jnp.maximum(0.0, jnp.minimum(state.Nr, _nr_max)))
    state = _real_state(state)

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
    # Both sit in WRF's ``if (temp(k).lt.T_0)`` cold block (:2554-2779): cloud ice above 0 C is neither
    # deposited/sublimated nor autoconverted, it only melts instantly at the end of the step (:3945).
    cold_block = state.T < T_0
    pri_ide_raw = C_CUBE * t1_subl * diffu * ssati * rvs * OIG1 * 1.0 * ni * ilami
    pri_ide_raw = jnp.where(active_ice & cold_block, pri_ide_raw, 0.0)
    sublimation_floor = None
    if _native_real_enabled():
        # WRF bounds sublimation by DBLE(rate_max) as well (:2557 rate_max, :2656 pri_ide, :2690 prs_sde).
        sublimation_floor = ((state.qv - qvsi) * state.rho / float(dt) * 0.999).astype(jnp.float64)
    pri_ide_limited = jnp.where(
        pri_ide_raw < 0.0,
        jnp.maximum(-ri / float(dt), pri_ide_raw) if sublimation_floor is None
        else jnp.maximum(jnp.maximum(-ri / float(dt), pri_ide_raw), sublimation_floor),
        jnp.minimum(pri_ide_raw, jnp.maximum(state.qv - qvsi, 0.0) * state.rho / float(dt) * 0.999),
    )
    iaus = _take3_last(tables.iaus, idx_i, idx_i1)
    tpi_ide = iaus[..., 2]
    pri_ide = jnp.where(pri_ide_limited > 0.0, tpi_ide * pri_ide_limited, pri_ide_limited)
    prs_ide = jnp.where(pri_ide_limited > 0.0, (1.0 - tpi_ide) * pri_ide_limited, 0.0)
    iau_table_mass = iaus[..., 0]
    iau_table_num = iaus[..., 1]
    iau_large = (idx_i == N_I_TABLE - 1) | (xdi > 5.0 * D0S)
    iau_small = xdi < _wrf("D0S_X01", 0.1 * D0S)
    prs_iau_mass = jnp.where(iau_large, ri * 0.99, jnp.where(iau_small, 0.0, jnp.minimum(ri * 0.99, iau_table_mass)))
    pni_iau_num = jnp.where(iau_large, ni * 0.95, jnp.where(iau_small, 0.0, jnp.minimum(ni * 0.95, iau_table_num)))
    prs_iau_mass = jnp.where(active_ice & cold_block, prs_iau_mass, 0.0)
    pni_iau_num = jnp.where(active_ice & cold_block, pni_iau_num, 0.0)

    # WRF gates the whole cloud-ice collection family on the same cold block
    # (module_mp_thompson.F:2554, ``if (temp(k).lt.T_0)``), on the post-melt
    # ``state.T`` (the same field the rci/sci moments above were built from).
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
    prs_sde = jnp.where(active_snow, jnp.where(prs_sde < 0.0, jnp.maximum(-rs / float(dt), prs_sde) if sublimation_floor is None
                                               else jnp.maximum(jnp.maximum(-rs / float(dt), prs_sde), sublimation_floor),
                                               jnp.minimum(prs_sde, jnp.maximum(state.qv - qvsi, 0.0) * state.rho / float(dt) * 0.999)), 0.0)
    # WRF ordering (module_mp_thompson.F): the riming snow->graupel split (line
    # 2758) compares prs_scw against this PER-CELL-clamped prs_sde, BEFORE the
    # GLOBAL multi-term deposition vapor-conservation ratio (line ~2862) scales
    # it. Capture the pre-ratio value for the riming comparison so a
    # deposition-limited (ratio<1) cell does not spuriously trip riming_dom.
    if _native_real_enabled():
        prs_sde = prs_sde.astype(jnp.float64)  # WRF DOUBLE prs_sde
    prs_sde_preratio = prs_sde
    vapor_rate_max = (state.qv - qvsi) * state.rho / float(dt) * 0.999
    _av_g, _bv_g, _cgg63, _t1_qg_qc, _cge9, t2_subl_qg, _t2_melt_qg, cge11 = _graupel_constants()
    prg_gde = C_CUBE * t1_subl * diffu * ssati * rvs * n0_g * (T1_SUBL_QG * _dpow(ilamg, CRE10) + t2_subl_qg * vsc2 * rhof2 * ilamg**cge11)
    prg_gde = jnp.where(active_graupel & (ssati < -EPS), jnp.maximum(jnp.maximum(-rg / float(dt), prg_gde), vapor_rate_max), 0.0)
    deposition_sum = pri_inu + pri_ide + prs_ide + prs_sde + prg_gde
    if _native_real_enabled():
        deposition_sum = deposition_sum.astype(jnp.float32)  # WRF REAL sump; compares/ratio REAL
    limited = ((deposition_sum > EPS) & (deposition_sum > vapor_rate_max)) | ((deposition_sum < -EPS) & (deposition_sum < vapor_rate_max))
    deposition_denom = jnp.where(jnp.abs(deposition_sum) > R1, deposition_sum, 1.0)
    deposition_ratio = jnp.where(limited, vapor_rate_max / deposition_denom, 1.0)
    if _native_real_enabled():
        inu_mass = pri_inu * deposition_ratio * float(dt) / state.rho  # WRF scales pri_inu too (:2868)
    pri_ide = pri_ide * deposition_ratio
    prs_ide = prs_ide * deposition_ratio
    prs_sde = prs_sde * deposition_ratio
    prg_gde = prg_gde * deposition_ratio

    prs_iau_rate = prs_iau_mass / float(dt)
    pni_iau_rate = pni_iau_num / float(dt)
    ice_sump = pri_ide - prs_iau_rate - prs_sci - pri_rci
    if _native_real_enabled():
        ice_sump = ice_sump.astype(jnp.float32)  # WRF REAL sump
    ice_rate_max = -ri / float(dt)
    ice_ratio = jnp.where((ice_sump < ice_rate_max) & active_ice, ice_rate_max / jnp.minimum(ice_sump, -EPS), 1.0)
    pri_ide = pri_ide * ice_ratio
    prs_iau_rate = prs_iau_rate * ice_ratio
    prs_sci = prs_sci * ice_ratio
    pri_rci = pri_rci * ice_ratio
    if not _native_real_enabled():
        prg_rci = pri_rci + prr_rci
    # Native REAL keeps WRF's prg_rci, formed once before the ice/rain limiters (:2733).

    prr_rcs, _prs_rcs, _prg_rcs, _pnr_rcs, _png_rcs, prr_rcg, _prg_rcg, _pnr_rcg = cold_collection_rates
    rain_sump = -prg_rfz_rate - pri_rfz_rate - prr_rci + prr_rcs + prr_rcg
    if _native_real_enabled():
        rain_sump = rain_sump.astype(jnp.float32)  # WRF REAL sump
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
    (ice_deposition, snow_from_ice_deposition, ice_to_snow, ice_number_to_snow, snow_collect_ice,
     rain_collect_ice, rain_to_graupel, ice_number_collected, rain_number_collected, snow_deposition,
     graupel_deposition, inu_mass, inu_number) = _real_incr(
        ice_deposition, snow_from_ice_deposition, ice_to_snow, ice_number_to_snow, snow_collect_ice,
        rain_collect_ice, rain_to_graupel, ice_number_collected, rain_number_collected, snow_deposition,
        graupel_deposition, inu_mass, inu_number)
    rci_graupel = _real_incr(prg_rci * float(dt) / state.rho) if _native_real_enabled() else None
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
        qg=state.qg + graupel_deposition + (prg_rci * float(dt) / state.rho if rci_graupel is None else rci_graupel),
        # WRF lines 2719-2727 update pni_ide only in sublimation; positive
        # deposition partitions mass but does not create new cloud-ice number.
        # WRF (module_mp_thompson.F:3054-3055) also caps ice number at 999e3 m^-3
        # AFTER summing all microphysics tendencies. The port had kept only the
        # maximum(0.0,...) floor and dropped that upper ceiling on this per-step
        # update, letting Ni run away on steep-terrain inner-nest (d03) columns ->
        # float overflow -> NaN at ~step 720 (v0.21.1 root cause). Restore the WRF
        # ceiling (per-mass = 999e3/rho, matching the saturation path ~line 752).
        # The cap only fires above 999e3/rho, so already-finite cases are unchanged
        # (bit-identical, no default regression) while the d03 runaway is prevented.
        Ni=jnp.minimum(
            jnp.maximum(
                0.0,
                state.Ni
                + inu_number
                + jnp.where(ice_deposition < 0.0, ice_deposition / jnp.maximum(xmi, XM0I), 0.0)
                - ice_number_to_snow
                - ice_number_collected,
            ),
            999.0e3 / state.rho,
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
        if _native_real_enabled():
            ef_sw = ef_sw.astype(jnp.float32)  # WRF REAL Ef_sw read from the DOUBLE t_Efsw
        scw_gate = active_snow & active_cloud & (mvd_c > D0C) & (xds > D0S)
        prs_scw = jnp.where(scw_gate, rhof * _wrf("T1_QS_QC", T1_QS_QC) * ef_sw * rc_rime * smoe, 0.0)
        prs_scw = jnp.minimum(rc_rime * odts, prs_scw)
        if _native_real_enabled():
            prs_scw = prs_scw.astype(jnp.float64)  # REAL product stored in WRF's DOUBLE prs_scw

        # Graupel collecting cloud water (GPU single-density mp8 PSD convention,
        # mu_g=0): Stokes-number efficiency of WRF 2414-2431.
        xdg = 4.0 * ilamg
        av_g, bv_g, cgg63, t1_qg_qc, cge9, _t2_subl_qg, _t2_melt_qg, _cge11 = _graupel_constants()
        vtg = _times_cgg63(rhof * av_g, cgg63) * ilamg**bv_g
        if _native_real_enabled():
            xdg, vtg = xdg.astype(jnp.float32), vtg.astype(jnp.float32)  # WRF REAL xDg / vtg
        stoke_g = mvd_c * mvd_c * vtg * RHO_W_RIME / (9.0 * jnp.maximum(visco, R1) * jnp.maximum(xdg, R1))
        ef_gw = jnp.where(
            stoke_g >= 0.4,
            jnp.where(stoke_g > 10.0, 0.77, 0.55 * jnp.log10(jnp.maximum(2.51 * stoke_g, R1))),
            0.0,
        )
        ef_gw = jnp.where(state.T > T_0, ef_gw * 0.1, ef_gw)
        gcw_gate = active_graupel & (rg >= 1.0e-6) & active_cloud & (mvd_c > D0C)
        prg_gcw = jnp.where(gcw_gate, rhof * t1_qg_qc * ef_gw * rc_rime * n0_g * ilamg**cge9, 0.0)

        # Rimed-snow -> graupel conversion + snow fall-speed boost (WRF 2758-2776).
        # Compare against the PRE-global-ratio prs_sde (WRF runs this block before
        # the deposition vapor-conservation ratio); the per-cell rate_max clamp at
        # line 913 is already applied, matching WRF lines 2690-2692.
        riming_dom = (prs_scw > 2.0 * prs_sde_preratio) & (prs_sde_preratio > EPS)
        r_frac = jnp.minimum(30.0, prs_scw / jnp.maximum(prs_sde_preratio, EPS))
        if _native_real_enabled():
            r_frac = r_frac.astype(jnp.float32)  # WRF REAL r_frac (g_frac, vts_boost REAL)
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
        if _native_real_enabled():
            collected = collected.astype(jnp.float32)  # WRF REAL sump
        ratio_qc = jnp.where(collected > rc_rime * odts, rc_rime * odts / jnp.maximum(collected, EPS), 1.0)
        prs_scw = prs_scw * ratio_qc
        prg_scw = prg_scw * ratio_qc
        prg_gcw = prg_gcw * ratio_qc

        # ratio_qc already bounds the collected sum by rc_rime*odts and the
        # deposition block never touches qc, so the qc sink below equals the
        # qs+qg gain exactly (no extra clamp -> no mass creation/destruction).
        scw_total_dt = (prs_scw + prg_scw + prg_gcw) * float(dt) / state.rho
        rime_incr = None
        if _native_real_enabled():
            scw_total_dt, *rime_incr = _real_incr(
                scw_total_dt, prs_scw * float(dt) / state.rho,
                (prg_scw + prg_gcw) * float(dt) / state.rho, png_scw * float(dt) / state.rho)
        lfus2_rime = LSUB - lvap2
        rime_heat = jnp.where(state.T < T_0, lfus2_rime * ocp * scw_total_dt, 0.0)
        rime_T = updated.T + rime_heat
        updated = updated.replace(
            qc=updated.qc - scw_total_dt,
            qs=updated.qs + (prs_scw * float(dt) / state.rho if rime_incr is None else rime_incr[0]),
            qg=updated.qg + ((prg_scw + prg_gcw) * float(dt) / state.rho if rime_incr is None else rime_incr[1]),
            Ns=jnp.maximum(0.0, updated.Ns - (png_scw * float(dt) / state.rho if rime_incr is None else rime_incr[2])),
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
    n0r_exp = ORG1 * rr_idx / _am_r() * _dpow(lamr, CRE1)
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
    n0g_exp = ORG1 * rg / AM_G_MP8 * _dpow(lamg, 4.0)
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
    d_qr, d_qs, d_qg, d_nr, d_ng = _real_incr(d_qr, d_qs, d_qg, d_nr, d_ng)

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
# estimates (~8-12).  ``GPUWRF_THOMPSON_NSED`` overrides the cap.
#
# WARNING -- clipping is NOT a safe fallback.  An earlier version of this note
# claimed a capped column stayed "stable but slightly under-resolved"; that is
# false and it hid a real failure.  Clipping nstep below vt*DT/dz breaks the
# explicit upwind CFL outright, and the scheme then AMPLIFIES rather than
# under-resolves.  v0234 d01 native step 1148 needed nstep=911, was silently
# clipped to 16, and created 7.93e13x the column ice mass in a single step
# (.agent/sprints/2026-07-21-v0234-opus-late-ni-fix/).  The root cause there was
# an unphysical 856.96 m/s ice fall speed from an unbalanced ice number, fixed at
# source in ``_balance_ice_number`` so the speed is never formed -- but the cap
# itself remains a SILENT corruption path for any future column that exceeds it.
# Raising the cap is not a fix; a fail-closed check is the open follow-up.
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
    float per column (== WRF's REAL(nstep)). The reference path clips to
    [1, NSED_MAX]; the private column path retains the uncapped adaptive count.
    """

    if _column_sed_fp32_enabled():
        vt_a, vt_b, dz = (jnp.asarray(a, jnp.float32) for a in (vt_a, vt_b, dz))
    vt = jnp.maximum(vt_a, vt_b)
    dz = jnp.maximum(dz, 1.0)
    active = vt > 1.0e-3
    # INT(DT/(dz/vt) + 1.) == floor(dt*vt/dz + 1) for the positive argument here.
    cand = jnp.floor((dt / (dz / vt) if _column_sed_fp32_enabled()
                      else dt * vt / dz) + 1.0)
    cand = jnp.where(active, cand, 0.0)
    nstep = jnp.max(cand, axis=-1)
    # The private column loop has no fixed trip cap. Retain the legacy cap only
    # on the reference path, so the gated path cannot silently truncate CFL.
    nstep = (jnp.maximum(nstep, 1.0) if _column_sed_enabled()
             else jnp.clip(nstep, 1.0, float(NSED_MAX)))
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
        def body(carry, _):
            inflow_mass, k = carry
            qk = (qcur[k] + dts / (rhor[k] * dzr[k]) * inflow_mass) / diag[k]
            return (rhor[k] * vtr[k] * qk, k + jnp.asarray(1, dtype=jnp.int32)), qk
        (_, _), qsol = jax.lax.scan(
            body,
            (jnp.zeros(qcur.shape[1:], acc), jnp.asarray(0, dtype=jnp.int32)),
            None,
            length=nz,
        )
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

    if _column_sed_enabled():
        from gpuwrf.kernels.phys_thompson_sedimentation import fill_down
        work = vt.astype(jnp.float32) if _column_sed_fp32_enabled() else vt
        return fill_down(work, active).astype(vt.dtype)

    vt_t = jnp.moveaxis(vt, -1, 0)  # (z, ...) with z=0 surface
    act_t = jnp.moveaxis(active, -1, 0)
    nz = vt_t.shape[0]

    def body(carry, _):
        prev, kk = carry  # fall speed of the layer above (higher index), already filled
        cur = jnp.where(act_t[kk], vt_t[kk], prev)
        return (cur, kk - jnp.asarray(1, dtype=jnp.int32)), cur

    init = jnp.zeros(vt_t.shape[1:], dtype=vt_t.dtype)
    (_, _), filled_rev = jax.lax.scan(
        body,
        (init, jnp.asarray(nz - 1, dtype=jnp.int32)),
        None,
        length=nz,
    )
    # filled_rev[k] corresponds to physical level nz-1-k; reverse to level order.
    filled = filled_rev[::-1]
    return jnp.moveaxis(filled, 0, -1)


def _fall_speeds(state: ThompsonColumnState, vts_boost=None, *, fill_down=None,
                 tables=THOMPSON_TABLES):
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

    fill = _fill_down if fill_down is None else fill_down
    rho = state.rho
    rhof = _rho_correction(rho)

    act_r = state.qr > R1
    rr = jnp.maximum(state.qr * rho, R1)
    # WRF computes the rain fall speeds from the mvd-clamped working number
    # ``nr(k)`` rebuilt at module_mp_thompson.F:3240-3250, NOT from the raw
    # prognostic Nr; the same clamp the rate-stage slopes use (_rain_distribution).
    nr = jnp.maximum(_clamp_rain_number(state.qr, state.Nr, rho) * rho, R2)
    lamr = (_am_r() * CRG3 * ORG2 * nr / rr) ** OBMR
    if _native_real_enabled():
        lamr = lamr.astype(jnp.float64)
    vt_r_mass = rhof * AV_R * CRG6 * ORG3 * _dpow(lamr, CRE3) * _dpow(lamr + FV_R, -CRE6)
    vt_r_num = rhof * AV_R * _wrf("CRG7", CRG7) / CRG12 * lamr ** CRE12 * ((lamr + FV_R) ** (-CRE7))
    vt_r_mass = fill(jnp.where(act_r, vt_r_mass, 0.0), act_r)
    vt_r_num = fill(jnp.where(act_r, vt_r_num, 0.0), act_r)

    act_i = state.qi > R1
    ri = jnp.maximum(state.qi * rho, R1)
    # WRF forms the ice fall speeds from the SIZE-BALANCED working number
    # ``ni(k)`` built at module_mp_thompson.F:3226-3234 out of the tendencies
    # that line 3033-3055 already re-balanced into the 5-300 um band -- NOT from
    # the raw prognostic Ni.  Same working-number discipline as rain above.
    ni = jnp.maximum(_balance_ice_number(state.qi, state.Ni, rho) * rho, R2)
    # cig(2) = Gamma(bm_i+mu_i+1) = Gamma(4) = 6 (WRF module_mp_thompson.F:695).
    lami = (_am_i() * 6.0 * OIG1 * ni / ri) ** OBMI
    if _native_real_enabled():
        lami = lami.astype(jnp.float64)
    ilami = 1.0 / lami
    vt_i_mass = rhof * AV_I * CIG3 * OIG2 * _dpow(ilami, BV_I)
    vt_i_num = rhof * AV_I * _wrf("CIG6", CIG6) / CIG7 * _dpow(ilami, BV_I)
    vt_i_mass = fill(jnp.where(act_i, vt_i_mass, 0.0), act_i)
    vt_i_num = fill(jnp.where(act_i, vt_i_num, 0.0), act_i)

    act_s = state.qs > R1
    tempc = state.T - 273.15
    _rs2, xds, _smo0, _smo1, _smof, _csnow, _act_s = _snow_moments(state.qs, rho, tempc, tables)
    vts_raw = _snow_terminal_velocity_wrf(rhof, xds, act_s)
    if vts_boost is not None and _mixed_phase_wrf_enabled():
        vts_raw = _wrf_snow_speed(vts_raw, vts_boost, _rs2, rr, vt_r_mass)
    elif vts_boost is not None:
        # WRF line 3721: vts = vts*vts_boost(k) (riming-dominant layers fall up
        # to 1.5x faster; boost==1.0 elsewhere and with riming disabled).
        vts_raw = vts_raw * vts_boost
    vt_s_mass = fill(vts_raw, act_s)

    act_g = state.qg > R1
    native = _native_real_enabled()
    # Native REAL: WRF mp8 re-diagnoses the working ng from the post-process rg (:3287-3300), like nr/ni above.
    _rg, _ng, _lamg, ilamg, _n0_g, _active_g = _graupel_distribution(state.qg, None if native else state.Ng, rho)
    if native:
        av_g, bv_g, cgg63 = _graupel_constants()[:3]
        vt_g_mass = fill(jnp.where(act_g, _times_cgg63(rhof * av_g, cgg63) * ilamg ** bv_g, 0.0), act_g)  # cgg(6)*ogg3 (:3758)
    else:
        vt_g_mass = fill(jnp.where(act_g, rhof * AV_G_MP8 * 6.0 * ORG3 * ilamg ** BV_G_MP8, 0.0), act_g)
    vt_g_num = fill(jnp.where(act_g, rhof * AV_G_MP8 * CRG7 / CRG12 * ilamg ** BV_G_MP8, 0.0), act_g)

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
    Reference accumulation uses the result dtype of (q, num, vt, rho, dz).
    The private column path preserves that precision unless its explicit
    SED_FP32 gate selects WRF REAL flux work; output storage types are retained.
    """

    if _column_sed_enabled():
        from gpuwrf.kernels.phys_thompson_sedimentation import sediment_one_species
        return sediment_one_species(q, num, vt_mass, vt_num, dz, rho, dt, nstep,
                                    RR_SURF_THRESHOLD,
                                    work_dtype=jnp.float32 if _column_sed_fp32_enabled() else None)

    acc_dtype = jnp.result_type(q.dtype, num.dtype, vt_mass.dtype, rho.dtype, dz.dtype)
    q_dt, num_dt = q.dtype, num.dtype
    q0 = q.astype(acc_dtype)
    num0 = num.astype(acc_dtype)
    nstep = jnp.asarray(nstep, acc_dtype)               # (...,) per column
    dt_sub = jnp.asarray(dt, acc_dtype) / nstep         # (...,) onstep*DT
    nstep_col = nstep[..., None]                          # broadcast over levels
    dt_sub_col = dt_sub[..., None]
    surf_thresh = jnp.asarray(RR_SURF_THRESHOLD, acc_dtype)

    def body(carry, _):
        q_c, num_c, ppt_c, n = carry
        # column-level mask: this substep is real only while n < nstep_col.
        live = (n.astype(acc_dtype) < nstep).astype(acc_dtype)  # (...,)
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
        return (q_c, num_c, ppt_c + surf, n + jnp.asarray(1, dtype=jnp.int32)), None

    zero_ppt = jnp.zeros(q.shape[:-1], dtype=acc_dtype)
    (q_out, num_out, ppt, _), _ = jax.lax.scan(
        body, (q0, num0, zero_ppt, jnp.asarray(0, dtype=jnp.int32)), None,
        length=NSED_MAX,
        unroll=_sed_unroll(),
    )
    return q_out.astype(q_dt), num_out.astype(num_dt), ppt


def _wrf_cloud_sed_band(rc, dz):
    """WRF ``ksed1(5)`` cloud-sedimentation band (module_mp_thompson.F:3646-3655, :3829).

    WRF walks up from kts over k = kts..kte-1 while the height BELOW level k is <= 500 m (inclusive: ``hgt_agl`` is
    tested after adding ``dzq(k)``), setting ``ksed1(5) = k`` wherever ``rc(k) > R2``; it starts at ``ksed1(:) = 1``
    (:3598), i.e. kts.  Cloud water (and number) sediment only on kts..ksed1(5); ``vtck = 0`` above (:3607), so no
    flux enters from above ksed1(5).  ``rc`` is WRF's working cloud density ``MAX(R1, qc*rho)``; axis -1 is vertical
    (index 0 = kts).  Returns the boolean band mask ``k <= ksed1(5)``.
    """

    nz = rc.shape[-1]
    k_idx = jnp.arange(nz)
    hgt_below = jnp.cumsum(dz, axis=-1) - dz
    visited = (hgt_below <= 500.0) & (k_idx <= nz - 2)
    ksed = jnp.max(jnp.where(visited & (rc > R2), k_idx, 0), axis=-1, keepdims=True)
    return k_idx <= ksed


def _cloud_water_fall_speed(state: ThompsonColumnState, rho_rc=None, rho_f=None):
    """Cloud-droplet mass terminal fall speed (m/s), WRF module_mp_thompson.F:3656-3664.

    ``vtc = rhof*av_c*ccg(5,nu_c)*ocg2(nu_c)*ilamc**bv_c`` with the mp=8 default
    ``nu_c = 12`` (fixed cloud number ``NT_C``), where ``lamc`` is the cloud
    gamma slope from :func:`_cloud_distribution`.  WRF only assigns a non-zero
    ``vtck`` where ``rc > R1`` AND the local vertical velocity ``w < 0.1 m/s``
    (line 3657: ``w1d(k) .lt. 1.E-1`` — cloud water does not sediment inside an
    updraft); elsewhere ``vtck`` stays 0 (NO fill-down, unlike rain/ice/snow/
    graupel — WRF leaves vtck=0 in inactive layers).

    WRF density stages (P0 closure K5): ``rc`` uses the pre-condensation ``rho`` (``rho_rc``; :3217, :3484) and
    ``rhof`` the ``rho_f`` chosen by :func:`_cloud_sed_rho_stages`.  ``None`` keeps ``state.rho`` for isolated callers.
    """

    rho_rc = jnp.maximum(state.rho if rho_rc is None else rho_rc, R1)
    rhof = _rho_correction(state.rho if rho_f is None else rho_f)
    rc = jnp.maximum(state.qc * rho_rc, R1)
    # lamc identical to _cloud_distribution / WRF line 3659.
    lamc = (_wrf("LAMC_PREFIX", NT_C * AM_R * CCG2_NU12 * OCG1_NU12) / rc) ** OBMR
    if _native_real_enabled():
        lamc = lamc.astype(jnp.float64)
    ilamc = 1.0 / lamc
    vtc = rhof * AV_C * _wrf("CCG5_NU12", CCG5_NU12) * _wrf("OCG2_NU12", OCG2_NU12) * _dpow(ilamc, BV_C)
    if _native_real_enabled():
        vtc = vtc.astype(jnp.float32)  # WRF REAL vtc/sed_c (:1592,1603).
    # WRF :3657 tests the DENSITY rc(k) = MAX(R1, qc*rho) > R1 (not the mixing ratio).
    active = (rc > R1) & (state.w < 1.0e-1)
    return jnp.where(active, vtc, 0.0)


def _sed_cloud_water(state: ThompsonColumnState, dt: float, cloud_sed_on=None, cloud_rho=None):
    """WRF cloud-water sedimentation (module_mp_thompson.F:3824-3837).

    Distinct from rain/ice/snow/graupel sedimentation in three WRF-faithful ways:
      1. SINGLE full-DT explicit-upwind pass (no nstep substepping; WRF runs the
         cloud-water update once, ``onstep`` is not applied — lines 3829-3836).
      2. Confined to the WRF ``ksed1(5)`` band (:func:`_wrf_cloud_sed_band`): kts up to the highest level at most
         500 m above the surface (inclusive) with ``rc > R2`` (lines 3646-3655).  Layers above stay untouched.
      3. The bottom-face cloud-water flux ``sed_c(kts)`` leaves the column but is
         NOT accumulated into any surface-precip channel in WRF (no ``pptXXX +=``
         line) — so cloud-water sedimentation is a (small) water-budget sink that
         we report separately, never as precip.

    ``cloud_sed_on`` (per-column bool, trailing axis kept) is WRF's column gate ``ANY(L_qc)`` (:3646, :3824): both
    cloud-sedimentation blocks run only when some level carried cloud BEFORE the saturation adjustment and kept it
    through it (see :func:`_wrf_l_qc_any`).  ``None`` keeps the ungated behaviour for isolated callers.

    ``cloud_rho = (rho_rc, rho_f, rho_o)`` are WRF's density stages (:func:`_cloud_sed_rho_stages`): ``rc`` (flux and
    the ksed1(5) test) with the pre-condensation rho, ``rhof`` per ANY(L_qr), and ``orho`` of the update (:3831)
    from the current rho.  ``None`` uses ``state.rho`` for all three (isolated callers).

    Returns ``(qc', cloudw_surface_loss_mm)``.  Axis -1 is vertical, index 0 ==
    surface (kts); flux enters a layer from the layer ABOVE (higher index).  The
    GPU column carries a FIXED cloud number (``NT_C``), so only the qc mass is
    advected here (WRF's ``nc`` redistribution is a no-op under fixed-Nc).
    """

    rho_rc, rho_f, rho_o = (state.rho,) * 3 if cloud_rho is None else cloud_rho
    rho = jnp.maximum(rho_rc, R1)
    dz = jnp.maximum(state.dz, 1.0)
    acc_dtype = jnp.result_type(state.qc.dtype, rho.dtype, dz.dtype)
    qc_dt = state.qc.dtype
    qc = state.qc.astype(acc_dtype)
    rho = rho.astype(acc_dtype)
    orho_rho = jnp.maximum(rho_o, R1).astype(acc_dtype)
    dz = dz.astype(acc_dtype)
    vtc = _cloud_water_fall_speed(state, rho_rc, rho_f).astype(acc_dtype)
    if cloud_sed_on is not None:
        vtc = jnp.where(cloud_sed_on, vtc, jnp.zeros_like(vtc))
    dt_a = jnp.asarray(dt, acc_dtype)

    # WRF ksed1(5) band (:3646-3655): kts..ksed1(5) only (was: every level below 500 m, strict).
    band = _wrf_cloud_sed_band(jnp.maximum(qc * rho, R1), dz)

    # sed_c(k) = vtck(k)*rc(k); rc = qc*rho.  vtck is already gated (rc>R1 & w<0.1
    # & active cloud); confine the flux to the ksed1(5) band.
    rc = jnp.maximum(qc * rho, 0.0)
    sed_c = jnp.where(band, vtc * rc, 0.0)  # kg m^-2 s^-1 (downward)
    # Flux INTO layer k comes from the layer above (k+1, higher index).
    sed_c_above = jnp.concatenate([sed_c[..., 1:], jnp.zeros_like(sed_c[..., :1])], axis=-1)
    # rc(k) += (sed_c(k+1) - sed_c(k))*odzq*DT  (single full-DT pass, WRF 3834).
    dq = (sed_c_above - sed_c) / dz / orho_rho * dt_a   # WRF :3831-3832, orho = 1/current rho
    qc_new = jnp.where(band, jnp.maximum(qc + dq, 0.0), qc)
    # Bottom-face cloud-water flux leaving the column at the surface (kg m^-2 ==
    # mm).  WRF does NOT count this as precip; we return it as a water-budget sink.
    cloudw_surface_loss = sed_c[..., 0] * dt_a
    return qc_new.astype(qc_dt), cloudw_surface_loss.astype(jnp.float64)


def _wrf_l_qc_any(qc_pre, rho_pre, qc_post, branch):
    """WRF ``ANY(L_qc)`` at the sedimentation stage (module_mp_thompson.F).

    ``L_qc(k)`` is set ONLY from the pre-condensation cloud ``(qc1d+qcten*DT) > R1`` (:3215-3223) and is CLEARED
    only INSIDE the condensation/evaporation branch (``branch`` = :func:`_condensation_branch` on the
    pre-adjustment state, :3401-3402), when ``rc = MAX(R1, (qc1d+DT*qcten)*rho) == R1`` with the pre-update ``rho``
    (:3484-3485) -- whether or not qc actually changed.  Newly condensed cloud never sets it.  Returns the
    per-column gate with the vertical axis kept (size 1) for broadcasting.
    """

    cleared = branch & (jnp.maximum(qc_post * rho_pre, R1) == R1)
    l_qc = (qc_pre > R1) & ~cleared
    return jnp.any(l_qc, axis=-1, keepdims=True)


def _cloud_sed_rho_stages(rho_pre_cond, qr_pre_cond, state):
    """WRF density stages seen by cloud sedimentation (module_mp_thompson.F), per level; returns (rho_rc, rho_f, rho_o).

    * ``rho_rc``: ``rc(k) = (qc1d+qcten*DT)*rho`` with the pre-condensation rho (:3193, :3217; :3484 inside the
      branch, before the :3490 update) -- the flux ``vtck*rc`` and the ksed1(5) test use it.
    * ``rho_o``: the CURRENT rho, recomputed after condensation (:3490) and rain evaporation (:3572) from the updated
      temperature and qv (unchanged levels reproduce the same value); ``orho = 1/rho`` of the update (:3831).
    * ``rho_f``: ``rhof = SQRT(RHO_NOT/rho)`` from :3194 (pre-condensation rho), overwritten for every level by :3614
      from the current rho when ANY(L_qr); ``L_qr`` = pre-condensation rain ``(qr1d+qrten*DT) > R1`` (:3236-3239),
      never cleared by rain evaporation.  (The :3505 rhof inside rain evaporation only runs where L_qr, so :3614
      supersedes it in the same column.)
    """

    rho_o = density_from_pressure_temperature(state.p, state.T, state.qv)
    any_l_qr = jnp.any(qr_pre_cond > R1, axis=-1, keepdims=True)
    return rho_pre_cond, jnp.where(any_l_qr, rho_o, rho_pre_cond), rho_o


def _sedimentation(state: ThompsonColumnState, dt: float, vts_boost=None, cloud_sed_on=None, cloud_rho=None):
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
        if _sed_prep_fused_enabled():
            raise ValueError("fused prep implements only WRF explicit sedimentation")
        # EXPERIMENTAL implicit backward-Euler sedimentation (gated, default OFF).
        return _sedimentation_implicit(state, dt, nsub, vts_boost)

    if _sed_prep_fused_enabled():
        if not (_native_real_enabled() and _column_sed_fp32_enabled()):
            raise ValueError("fused sedimentation prep requires Native REAL and column fp32 sedimentation")
        from gpuwrf.kernels.phys_thompson_prep import sediment_and_finish
        # Cloud sedimentation reads the pre-precipitation state and does not
        # affect the four fall-speed channels. Its P0 gates/density stages stay
        # in the retained helper; the column kernel also performs final finish.
        qc, cloudw = _sed_cloud_water(state, dt, cloud_sed_on, cloud_rho)
        out, precip = sediment_and_finish(state.replace(qc=qc), dt, vts_boost)
        return out, {**precip, "cloudw": cloudw}

    speeds = _fall_speeds(state, vts_boost)
    return _sediment_with_speeds(state, dt, speeds, cloud_sed_on, cloud_rho)


def _sediment_with_speeds(state, dt, speeds, cloud_sed_on=None, cloud_rho=None):
    """Explicit WRF sedimentation for given fall speeds (zero speeds: no transport)."""
    if _native_real_enabled():
        speeds = tuple(v.astype(jnp.float32) for v in speeds)  # WRF REAL vtrk/vtnrk/vtik/vtsk/vtgk
    vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num = speeds
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
    # Ice follows the same rule: WRF sediments the size-balanced working number
    # ``ni(k)`` (3033-3055 -> 3226-3234 -> the 3840-3869 ice fall loop), and the
    # result is re-balanced to the same band by ``_finish`` (WRF 4023-4038).
    Ni_sed = _balance_ice_number(state.qi, state.Ni, rho)
    qi, Ni, ppt_ice = _sed_one_species(state.qi, Ni_sed, vt_i_mass, vt_i_num, dz, rho, dt, nstep_i)
    # Snow: number tracks mass (diagnostic Ns).  Use the mass speed for both.
    qs, Ns, ppt_snow = _sed_one_species(state.qs, state.Ns, vt_s_mass, vt_s_mass, dz, rho, dt, nstep_s)
    qg, Ng, ppt_graupel = _sed_one_species(state.qg, state.Ng, vt_g_mass, vt_g_num, dz, rho, dt, nstep_g)

    # Cloud-water fall term: single full-DT pass below 500 m AGL, NOT counted as
    # surface precip (WRF module_mp_thompson.F:3824-3837).  Reported under
    # ``cloudw`` as a water-budget sink so the closure budget stays exact.
    qc, ppt_cloudw = _sed_cloud_water(state, dt, cloud_sed_on, cloud_rho)

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


def _thompson_source_sink_body(state: ThompsonColumnState, dt: float, debug: bool, *, sediment: bool, raw_vapor=False):
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
    prior behaviour). The staged NATIVE_REAL policy assigns REAL work at
    process boundaries while retaining DOUBLE distributions and lookup tables.
    """

    # Oracle comparison harnesses may install a separate sedimentation scheme.
    # Preserve that scheme's own implementation and diagnostic contract.
    if sediment and _full_column_enabled() and _sedimentation.__module__ == __name__:
        if debug or not (_native_real_enabled() and _column_sed_fp32_enabled()):
            raise ValueError("full column kernel requires non-debug Native REAL/fp32 column work")
        if _implicit_sed_nsub():
            raise ValueError("full column kernel implements WRF explicit sedimentation")
        from gpuwrf.kernels.phys_thompson_full import full_column
        return full_column(state, dt)

    work = _work_dtype()
    storage_dtypes = {name: jnp.asarray(getattr(state, name)).dtype for name in ThompsonColumnState.__slots__}
    state = _cast_state(state, work)

    vapor = None
    if raw_vapor:
        # Native mp8 only. WRF :1226 retains raw qv1d; :1800 floors a
        # separate qv work array, never the original tendency baseline.
        raw_entry = _real_state(state)
        vapor = (raw_entry.qv, jnp.zeros_like(raw_entry.qv))
    state = _clip_species(state)
    if raw_vapor:
        entry = _real_state(state)
        rho = density_from_pressure_temperature(entry.p, entry.T, entry.qv)
        state = state.replace(rho=jnp.where(raw_entry.qv < 1.0e-10, rho, state.rho))
        # WRF :1990/:2020 returns before applying any vapor tendency if
        # the entire column has neither hydrometeors nor supersaturation.
        qvsi = jnp.where(entry.T <= T_0, saturation_mixing_ratio_ice(entry.p, entry.T),
                        saturation_mixing_ratio_liquid(entry.p, entry.T))
        ssati = entry.qv / qvsi - 1.0
        supersaturated = (ssati > 0.0) & (jnp.abs(ssati) >= EPS)
        no_micro = ~jnp.any((entry.qc > R1) | (entry.qr > R1) | (entry.qi > R1)
                           | (entry.qs > R1) | (entry.qg > R1) | supersaturated,
                           axis=-1, keepdims=True)
    state = _reset_mp8_graupel_number(state)
    valid = _thermodynamically_admissible(state)
    fallback = state
    state = _debug_checks(state, debug)
    if _mixed_phase_wrf_enabled():   # WRF single-pass source stage (B55 + mixed-phase fixes, end of module)
        if vapor is None:
            state, graupel_melt, vts_boost = _wrf_mixed_phase_sources(
                _real_state(state), dt, THOMPSON_TABLES, _mixed_phase_cold_tables())
        else:
            state, graupel_melt, vts_boost, qvten = _wrf_mixed_phase_sources(
                _real_state(state), dt, THOMPSON_TABLES, _mixed_phase_cold_tables(), vapor=vapor)
            vapor = (vapor[0], qvten)
    else:
        # Native REAL: WRF forms the rain-snow/rain-graupel rates from the entry rr/nr (idx_r1 :2279), not from the
        # number left by rain self-collection/break-up.
        cold_entry = _real_state(state) if _native_real_enabled() else None
        state = _warm_rain_collection(state, dt)
        state = _real_state(state)
        cold_rates = (
            _cold_collection_rates(state if cold_entry is None else cold_entry, dt, COLD_COLLECTION_TABLES)
            if _cold_collection_enabled()
            else _zero_cold_collection_rates(state)
        )
        state, graupel_melt, vts_boost, cold_rates = _ice_sources_with_process_flags(
            state, dt, cold_collection_rates=cold_rates
        )
        state = _real_state(state)
        if _cold_collection_enabled():
            # rain-collecting-snow / rain-collecting-graupel below 0 C: convert rain
            # to graupel where supercooled rain meets snow/graupel (WRF 2484-2548).
            # Computed after the freeze/riming staging pass, mirroring WRF's
            # single-pass rate staging before sedimentation.
            state = _apply_cold_collection_rates(state, dt, cold_rates)
            state = _real_state(state)
    qc_pre_cond, rho_pre_cond, branch = state.qc, state.rho, _condensation_branch(state)
    qr_pre_cond = state.qr   # WRF L_qr set point (:3236-3239)
    if vapor is None:
        state, cloud_condensed = _saturation_adjustment_with_condensation(state, dt)
    else:
        state, cloud_condensed, qvten = _saturation_adjustment_with_condensation(state, dt, vapor=vapor)
        vapor = (vapor[0], qvten)
    state = _real_state(state)
    # WRF gates cloud sedimentation on ANY(L_qc) from the PRE-condensation cloud (P0 gate-2 attribution).
    cloud_sed_on = _wrf_l_qc_any(qc_pre_cond, rho_pre_cond, state.qc, branch)
    if vapor is None:
        state = _rain_evaporation(state, dt, skip_evaporation=cloud_condensed, graupel_melt=graupel_melt)
    else:
        state, qvten = _rain_evaporation(state, dt, skip_evaporation=cloud_condensed,
                                       graupel_melt=graupel_melt, vapor=vapor)
    state = _real_state(state)
    if sediment:
        cloud_rho = _cloud_sed_rho_stages(rho_pre_cond, qr_pre_cond, state)   # WRF density stages (K5, D2)
        state, precip = _sedimentation(state, dt, vts_boost=vts_boost, cloud_sed_on=cloud_sed_on, cloud_rho=cloud_rho)
    else:
        zero = jnp.zeros(state.qv.shape[:-1], dtype=state.qv.dtype)
        precip = {"rain": zero, "snow": zero, "graupel": zero, "ice": zero}
    if not (sediment and _sed_prep_fused_enabled()):
        state = _instant_melt_freeze(state, dt)
        state = _real_state(state)
        state = _finish(state)
        state = _real_state(state)
    state = _select_state(valid, state, fallback)
    if raw_vapor:
        # WRF no_micro never copies its private floor back to qv1d (:2020,
        # :1350). Hydrometeor/number cleanup above remains in effect.
        state = state.replace(qv=jnp.where(no_micro & valid, raw_entry.qv, state.qv),
                              rho=jnp.where(no_micro & valid, raw_entry.rho, state.rho))
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

    if os.environ.get("GPUWRF_THOMPSON_COLUMN_LAYOUT", "0") == "1":
        # Keep a single column/level layout across rates and private column
        # kernels. Restore the retained coupler's original leading axes.
        shape = state.qv.shape
        flat = state.replace(**{key: jnp.reshape(getattr(state, key), (-1, shape[-1]))
                                for key in ThompsonColumnState.__slots__})
        out, precip = _thompson_source_sink_body(flat, dt, debug, sediment=True)
        out = out.replace(**{key: jnp.reshape(getattr(out, key), shape)
                             for key in ThompsonColumnState.__slots__})
        return out, {key: jnp.reshape(value, shape[:-1]) for key, value in precip.items()}
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


# ---------------------------------------------------------------------------
# GPUWRF_THOMPSON_MIXED_PHASE_WRF (default off, native REAL only): WRF's single-pass source stage
# (mp_thompson, module_mp_thompson.F:1810-3204). Every warm-rain and frozen-species rate is formed from the
# ENTRY column, then WRF's joint conservation limiters (:2853-2955), its REAL tendencies (:2971-3180) and one
# state update (:3195-3204) — instead of the staged warm -> freeze -> melt -> deposition -> riming applies above
# (B55). Adds what the staged path lacks: the wet-bulb twet (:2001-2011) for the rcs/rcg branch, Ef_gw, melt
# enhancement and pnr_sml/pnr_gml; the warm rain-snow / rain-graupel branch (:2512-2547) and the :2946 re-balance;
# the r_s(1)/r_g(1) collection guards; warm snow/graupel sublimation only while not melting (:2783-2833);
# the melt enhancement term (:2786-2789); the rime split in the cold block only (:2758); Hallett-Mossop
# splintering (:2741-2752); tcg_racg read like WRF at idx_bg1 and the cloud-freezing planes at WRF's integer-nic1
# idx_n (thompson_tables.load_wrf_cold_collection_tables); the melting-snow fall-speed blend (:3721-3724).
# ---------------------------------------------------------------------------
_OLFUS32 = np.float32(1.0) / np.float32(334000.0)   # WRF olfus = 1./(lsub - lvap0), REAL parameter


def _mixed_phase_wrf_enabled() -> bool:
    return os.environ.get("GPUWRF_THOMPSON_MIXED_PHASE_WRF", "0") == "1"


def _mixed_phase_cold_tables():
    """Cold-collection tables + effective tcg_racg as fp64 jnp arrays, built PER CALL from the host cache (a cached jax
    value would leak from one trace into the next program: F0, FINDINGS E206); the full-column kernel substitutes its
    table refs. WRF reads these tables in DOUBLE, so without x64 this refuses instead of truncating to fp32."""
    from gpuwrf.physics.thompson_tables import load_wrf_cold_collection_tables  # noqa: PLC0415
    host = load_wrf_cold_collection_tables()
    if jax.dtypes.canonicalize_dtype(jnp.float64) != jnp.float64:      # x64 off, incl. jax.enable_x64(False) scopes
        raise ValueError("GPUWRF_THOMPSON_MIXED_PHASE_WRF needs jax_enable_x64: the WRF cold-collection tables are DOUBLE")
    return type(host)(*(jnp.asarray(table, dtype=jnp.float64) for table in host))


def _wrf_melting_band(tempc):
    """k <= k_melting (:1975-1985): every level at or below the highest level with tempc > 0."""
    k = jnp.arange(tempc.shape[-1], dtype=jnp.int32)
    return k <= jnp.max(jnp.where(tempc > 0.0, k, -1), axis=-1, keepdims=True)


def _wrf_theta_e(p, t, w, tlcl):
    """theta_e (:6032-6062, Bolton 1980 eqn 43), REAL."""
    rr = w + 1.0e-8
    xx = t * (100000.0 / p) ** (0.2854 * (1.0 - 0.28 * rr))
    return xx * jnp.exp((3.376 / tlcl - 0.00254) * ((rr * 1000.0) * (1.0 + 0.81 * rr)))


def _wrf_theta_wetb(the):
    """theta_wetb (:6111-6145): REAL*8 DATA from REAL literals, DOUBLE Horner, REAL answer."""
    c = [float(np.float32(v)) for v in (-1.00922292e-10, -1.47945344e-8, -1.7303757e-6, -0.00012709,
                                        1.15849867e-6, -3.518296861e-9, 3.5741522e-12)]
    d = [float(np.float32(v)) for v in (0.0, -3.5223513e-10, -5.7250807e-8, -5.83975422e-6,
                                        4.72445163e-8, -1.13402845e-10, 8.729580402e-14)]
    x = jnp.minimum(475.0, the).astype(jnp.float64)

    def horner(a):
        return a[0] + x * (a[1] + x * (a[2] + x * (a[3] + x * (a[4] + x * (a[5] + x * a[6])))))

    return jnp.where(x <= 335.5, horner(c), horner(d)).astype(the.dtype) + 273.15


def _wrf_twet(temp, p, qv, qvs, band):
    """Wet-bulb twet (:2001-2011): min(T, compT_fr_The(theta_e(T_lcl))) on k <= k_melting where satw < 0.999.

    compT_fr_The (:6149-6175) is WRF's Newton iteration (<= 100 steps, |cor| < 0.01, theta_wetb fallback); each
    cell stops at its own convergence step, the loop ends when no needing cell is left.
    """
    need = band & (qv / qvs < 0.999)
    es = p * (qv + 1.0e-8) / (0.622 + (qv + 1.0e-8))
    esln = jnp.log(es)
    dew_t = jnp.minimum(temp - 0.001, (35.86 * esln - 4947.2325) / (esln - 23.6837))       # t_dew :6089
    tlcl = 1.0 / (1.0 / (dew_t - 56.0) + jnp.log(temp / dew_t) / 800.0) + 56.0             # t_lcl :6066
    the = jnp.where(need, _wrf_theta_e(p, temp, qv, tlcl), temp)
    guess0 = (the - 0.5 * jnp.maximum(the - 270.0, 0.0) ** 1.05) * (p / 100000.0) ** 0.2
    todo0 = jnp.where(need, 1.0, 0.0).astype(temp.dtype)

    def body(carry):
        it, guess, todo = carry
        tenu = _wrf_theta_e(p, guess, saturation_mixing_ratio_liquid(p, guess), guess)
        tenup = _wrf_theta_e(p, guess + 1.0, saturation_mixing_ratio_liquid(p, guess + 1.0), guess + 1.0)
        cor = (the - tenu) / (tenup - tenu)
        active = todo > 0.0
        guess = jnp.where(active, guess + cor, guess)
        todo = jnp.where(active & (cor < 0.01) & (-cor < 0.01), 0.0, todo)
        return it + 1, guess, todo

    def cond(carry):
        it, _guess, todo = carry
        return (it < 100) & jnp.any(todo > 0.0)

    _it, guess, todo = jax.lax.while_loop(cond, body, (jnp.int32(0), guess0, todo0))
    comp = jnp.where(todo > 0.0, _wrf_theta_wetb(the) * (p / 100000.0) ** 0.286, guess)
    return jnp.where(need, jnp.minimum(temp, comp), temp)


def _wrf_subl_prefactor(state, ssati, diffu, tcond, qvsi):
    """Srivastava-Coen prefactor (:2383-2400) with WRF's level qvsi (= qvs above 0 C, :1978-1981)."""
    otemp = 1.0 / state.T
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


def _wrf_warm_rain_rates(state, dt, tables, odts):
    """prr_wau/pnr_wau/prr_rcw/pnr_rcr (:2157-2207) from the entry column; prr_rcw = MIN(rc*odts, .) (:2205)."""
    rc, lamc, xdc, mvd_c, active_cloud = _cloud_distribution(state.qc, state.rho)
    rr, nr, lamr, _ilamr, mvd_r, n0_r, active_rain = _rain_distribution(state.qr, state.Nr, state.rho)
    rhof = _air_properties(state)[6]
    dc_g = ((_wrf("DC_G_PREFIX", (CCG3_NU12 * OCG2_NU12) ** OBMR) / lamc) * 1.0e6).astype(jnp.float32)
    dc_b = jnp.maximum(xdc**3 * dc_g**3 - xdc**6, 0.0) ** (1.0 / 6.0)
    zeta1_raw = 6.25e-6 * xdc * dc_b**3 - 0.4
    zeta = 0.027 * rc * (0.5 * (zeta1_raw + jnp.abs(zeta1_raw)))
    taud_raw = 0.5 * dc_b - 7.5
    tau = 3.72 / jnp.maximum(rc * (0.5 * (taud_raw + jnp.abs(taud_raw)) + R1), R1)
    prr_wau = jnp.where((rc > 0.01e-3) & active_cloud, jnp.minimum(rc * odts, zeta / tau), 0.0).astype(jnp.float64)
    pnr_wau = prr_wau / _wrf("PNR_WAU_DEN", AM_R * NU_C_MP8 * 10.0 * D0R**3)
    idx_r_eff = jnp.clip(jnp.floor(N_EFRW_R * jnp.log(jnp.maximum(mvd_r, DR_FIRST) / DR_FIRST)
                                   / jnp.log(DR_LAST / DR_FIRST)), 0, N_EFRW_R - 1).astype(jnp.int32)
    idx_c_eff = jnp.clip(jnp.floor(mvd_c * 1.0e6).astype(jnp.int32) - 1, 0, N_EFRW_C - 1)
    ef_rw = _take2(tables.t_Efrw, idx_r_eff, idx_c_eff).astype(jnp.float32)
    prr_rcw = rhof * _wrf("T1_QR_QC", T1_QR_QC) * ef_rw * rc * n0_r * _dpow(lamr + FV_R, -CRE9)
    prr_rcw = jnp.where(active_rain & (mvd_r > D0R) & (mvd_c > D0C), jnp.minimum(rc * odts, prr_rcw), 0.0)
    ef_rr = 1.0 - jnp.exp(2300.0 * (mvd_r - 1950.0e-6))
    pnr_rcr = jnp.where(active_rain & (mvd_r > D0R), ef_rr * 2.0 * nr * rr, 0.0)
    return prr_wau, pnr_wau, prr_rcw, pnr_rcr


def _wrf_rain_snow_graupel_rates(state, dt, cold_tables, twet, odts):
    """Rain-snow / rain-graupel collection, both twet branches (:2484-2547), WRF guards rs >= r_s(1), rg >= r_g(1)."""
    rho = state.rho
    rr = jnp.maximum(state.qr * rho, R1)
    rs = jnp.maximum(state.qs * rho, R1)
    rg = jnp.maximum(state.qg * rho, R1)
    nr = jnp.maximum(state.Nr * rho, R2)
    _, _nr_c, lamr, _ilamr, _mvd_r, _n0r, active_rain = _rain_distribution(state.qr, state.Nr, state.rho)
    rr_idx = jnp.maximum(rr, R_R_FIRST)
    idx_r = _lookup_digit_index(rr_idx, -6, N_R_TABLE)
    idx_r1 = _lookup_digit_index(ORG1 * rr_idx / _am_r() * _dpow(lamr, CRE1), 6, N_R1_TABLE)
    idx_s = _lookup_digit_index(jnp.maximum(rs, 1.0e-6), -6, 37)
    idx_t = jnp.clip(-(jnp.trunc((state.T - 273.15 - 2.5) / 5.0) - 1.0), 1.0, 9.0).astype(jnp.int32) - 1
    _rg_d, ng, lamg, _ilamg_d, _n0g_d, _act_g = _graupel_distribution(state.qg, state.Ng, state.rho)
    idx_g = _lookup_digit_index(jnp.maximum(rg, 1.0e-6), -6, 37)
    idx_g1 = _lookup_digit_index(jnp.maximum(ORG1 * rg / AM_G_MP8 * _dpow(lamg, 4.0), 1.0e2), 2, 37)
    rain = active_rain & (rr >= R_R_FIRST)
    racs = rain & (rs >= 1.0e-6)
    racg = rain & (rg >= 1.0e-6)
    cold = twet < T_0

    def s4(name):
        return _take4(getattr(cold_tables, name), idx_s, idx_t, idx_r1, idx_r)

    def g4(name):
        return _take4(getattr(cold_tables, name), idx_g1, idx_g, idx_r1, idx_r)

    rr_odts, rs_odts, rg_odts = (rr * odts).astype(jnp.float64), (rs * odts).astype(jnp.float64), (rg * odts).astype(jnp.float64)
    prr_rcs_c = jnp.maximum(-rr_odts, -(s4("tmr_racs2") + s4("tcr_sacr2") + s4("tmr_racs1") + s4("tcr_sacr1")))
    prs_rcs_c = jnp.maximum(-rs_odts, s4("tmr_racs2") + s4("tcr_sacr2") - s4("tcs_racs1") - s4("tms_sacr1"))
    prg_rcs_c = jnp.minimum(((rr + rs) * odts).astype(jnp.float64), s4("tmr_racs1") + s4("tcr_sacr1") + s4("tcs_racs1") + s4("tms_sacr1"))
    pnr_rcs_c = jnp.minimum((nr * odts).astype(jnp.float64), s4("tnr_racs1") + s4("tnr_racs2") + s4("tnr_sacr1") + s4("tnr_sacr2"))
    prs_rcs_w = jnp.maximum(-rs_odts, -s4("tcs_racs1") - s4("tms_sacr1") + s4("tmr_racs2") + s4("tcr_sacr2"))
    prr_rcs = jnp.where(racs, jnp.where(cold, prr_rcs_c, -prs_rcs_w), 0.0)
    prs_rcs = jnp.where(racs, jnp.where(cold, prs_rcs_c, prs_rcs_w), 0.0)
    prg_rcs = jnp.where(racs & cold, prg_rcs_c, 0.0)
    pnr_rcs = jnp.where(racs & cold, pnr_rcs_c, 0.0)

    prg_rcg_c = jnp.minimum(rr_odts, g4("tmr_racg") + g4("tcr_gacr"))
    pnr_rcg_c = jnp.minimum((nr * odts).astype(jnp.float64), g4("tnr_racg") + g4("tnr_gacr"))
    prr_rcg_w = jnp.minimum(rg_odts, g4("tcg_racg"))
    png_rcg_w = jnp.minimum((ng * odts).astype(jnp.float64), g4("tnr_racg"))
    prr_rcg = jnp.where(racg, jnp.where(cold, -prg_rcg_c, prr_rcg_w), 0.0)
    prg_rcg = -prr_rcg
    pnr_rcg = jnp.where(racg, jnp.where(cold, pnr_rcg_c, -1.5 * g4("tnr_gacr")), 0.0)
    png_rcg = jnp.where(racg & ~cold, png_rcg_w, 0.0)
    return prr_rcs, prs_rcs, prg_rcs, pnr_rcs, pnr_rcs, prr_rcg, prg_rcg, pnr_rcg, png_rcg


def _wrf_scale(mask, ratio, *rates):
    """rate*ratio where the WRF limiter fires (REAL ratio, DOUBLE rate)."""
    return tuple(jnp.where(mask, r * ratio, r) for r in rates)


def _wrf_mixed_phase_activity(state):
    """Per-cell (warm, cold) activity of _wrf_mixed_phase_sources, a SUPERSET of its rate gates: every warm rate needs
    cloud or rain > R1; every cold/mixed-phase/melt rate needs ice, snow or graupel > R1, cloud or rain > R1 in a cold
    cell (freezing), or a nucleation cell (ssati >= 0.25, or ssatw > EPS below 253.15 K; tested with 0.2 / 0)."""
    l_qc, l_qr = state.qc > R1, state.qr > R1
    cold = state.T < T_0
    ssati = state.qv / saturation_mixing_ratio_ice(state.p, state.T) - 1.0
    ssatw = state.qv / saturation_mixing_ratio_liquid(state.p, state.T) - 1.0
    nuc = cold & ((ssati >= 0.2) | ((ssatw > 0.0) & (state.T < 253.15)))
    ice = (state.qi > R1) | (state.qs > R1) | (state.qg > R1)
    return l_qc | l_qr, ice | (cold & (l_qc | l_qr)) | nuc


def _wrf_mixed_phase_sources(state, dt, tables, cold_tables, *, warm_rates=True, cold_rates=True, vapor=None):
    """WRF single-pass source stage on the entry column; returns (state, prr_gml > 0 proxy, vts_boost).

    warm_rates/cold_rates=False (EARLY_EXIT kernel branches, see _wrf_mixed_phase_activity) skip rate families that
    are exactly zero in that column; limiters, tendencies, number caps and the update are shared.
    """
    if not _native_real_enabled():
        raise ValueError("GPUWRF_THOMPSON_MIXED_PHASE_WRF requires GPUWRF_THOMPSON_NATIVE_REAL=1")
    if _sed_prep_fused_enabled() or _implicit_sed_nsub():
        raise ValueError("GPUWRF_THOMPSON_MIXED_PHASE_WRF implements WRF explicit sedimentation only")
    f64 = jnp.float64
    dtf = float(dt)
    odts = np.float32(1.0) / np.float32(dtf)                  # WRF REAL odts = 1./dt
    # WRF zeroes sub-R1 species and their numbers before any rate (:1820-1940).
    state = state.replace(
        qc=jnp.where(state.qc > R1, state.qc, 0.0), qr=jnp.where(state.qr > R1, state.qr, 0.0),
        qi=jnp.where(state.qi > R1, state.qi, 0.0), qs=jnp.where(state.qs > R1, state.qs, 0.0),
        qg=jnp.where(state.qg > R1, state.qg, 0.0),
        Ni=jnp.where(state.qi > R1, state.Ni, 0.0), Nr=jnp.where(state.qr > R1, state.Nr, 0.0))
    temp, qv, p, rho = state.T, state.qv, state.p, state.rho
    tempc, diffu, visco, tcond, lvap, ocp, rhof, rhof2, vsc2 = _air_properties(state)
    orho = 1.0 / rho
    qvs = saturation_mixing_ratio_liquid(p, temp)
    qvsi = jnp.where(tempc <= 0.0, saturation_mixing_ratio_ice(p, temp), qvs)
    del_qvs = jnp.maximum(0.0, saturation_mixing_ratio_liquid(p, jnp.asarray(T_0, temp.dtype)) - qv)
    ssatw = qv / qvs - 1.0
    ssatw = jnp.where(jnp.abs(ssatw) < EPS, 0.0, ssatw)
    ssati = qv / qvsi - 1.0
    ssati = jnp.where(jnp.abs(ssati) < EPS, 0.0, ssati)
    cold = temp < T_0
    warm = ~cold

    if warm_rates or cold_rates:
        rc, _lamc, _xdc, mvd_c, l_qc = _cloud_distribution(state.qc, rho)
        rr, nr, lamr, _ilamr, mvd_r, n0_r, active_rain = _rain_distribution(state.qr, state.Nr, rho)
    else:  # dry column: rc/rr only reach the limiters, masked by l_qc/l_qr (both False)
        rc, l_qc, rr = jnp.maximum(state.qc * rho, R1), state.qc > R1, jnp.maximum(state.qr * rho, R1)
    l_qr = state.qr > R1
    rate_max = (qv - qvsi) * rho * odts * 0.999                                          # REAL
    zero64 = jnp.zeros_like(temp, dtype=f64)

    # ---- warm rain (:2157-2207) ----
    if warm_rates:
        prr_wau, pnr_wau, prr_rcw, pnr_rcr = _wrf_warm_rain_rates(state, dt, tables, odts)
    else:  # no cloud/rain > R1 in the column: every warm rate is exactly 0 (EARLY_EXIT kernel branch)
        prr_wau = pnr_wau = prr_rcw = zero64
        pnr_rcr = jnp.zeros_like(temp)

    if cold_rates:
        ri, ni, _lami, ilami, xdi, xmi, l_qi = _ice_distribution(state.qi, state.Ni, rho)
        rs, xds, smo0, smo1, smof, c_snow, l_qs = _snow_moments(state.qs, rho, tempc, tables)
        rg, ng, lamg, ilamg, n0_g, l_qg = _graupel_distribution(state.qg, state.Ng, rho)
        # twet only enters snow/graupel processes (riming Ef_gw, rcs/rcg branch, melt, :2946 re-balance), all gated by
        # L_qs/L_qg: the Newton solve runs only there (WRF solves every subsaturated k <= k_melting; same values used).
        twet = _wrf_twet(temp, p, qv, qvs, _wrf_melting_band(tempc) & (l_qs | l_qg))
        smoe = _snow_moment(tables.cse[12], jnp.maximum(state.qs * rho, R1) / AM_S, jnp.minimum(-0.1, tempc), tables)
        _av_g, _bv_g, cgg63, t1_qg_qc, cge9, t2_subl_qg, t2_melt_qg, cge11 = _graupel_constants()

        # ---- rain-snow / rain-graupel (:2484-2547) ----
        prr_rcs, prs_rcs, prg_rcs, pnr_rcs, png_rcs, prr_rcg, prg_rcg, pnr_rcg, png_rcg = (
            _wrf_rain_snow_graupel_rates(state, dt, cold_tables, twet, odts))

        # ---- riming: snow / graupel collecting cloud water, all temperatures (:2403-2440) ----
        xds_pos = NBS_EFSW * jnp.log(jnp.maximum(xds, DS_FIRST) / DS_FIRST) / math.log(DS_LAST / DS_FIRST)
        idx_s_eff = jnp.clip(jnp.trunc(xds_pos), 0, NBS_EFSW - 1).astype(jnp.int32)
        idx_c_rime = jnp.clip(jnp.floor(mvd_c * 1.0e6).astype(jnp.int32) - 1, 0, N_EFRW_C - 1)
        ef_sw = _take2(tables.t_Efsw, idx_s_eff, idx_c_rime).astype(jnp.float32)
        scw_gate = l_qs & l_qc & (mvd_c > D0C) & (xds > D0S)
        prs_scw = jnp.where(scw_gate, jnp.minimum(rc * odts, rhof * _wrf("T1_QS_QC", T1_QS_QC) * ef_sw * rc * smoe), 0.0).astype(f64)
        xdg = (4.0 * ilamg).astype(jnp.float32)
        vtg = (_times_cgg63(rhof * _av_g, cgg63) * ilamg**_bv_g).astype(jnp.float32)
        stoke_g = mvd_c * mvd_c * vtg * RHO_W_RIME / (9.0 * jnp.maximum(visco, R1) * jnp.maximum(xdg, R1))
        ef_gw = jnp.where(stoke_g >= 0.4, jnp.where(stoke_g > 10.0, 0.77, 0.55 * jnp.log10(jnp.maximum(2.51 * stoke_g, R1))), 0.0)
        ef_gw = jnp.where(twet > T_0, ef_gw * 0.1, ef_gw)                                    # :2431 twet
        gcw_gate = l_qg & (rg >= 1.0e-6) & l_qc & (mvd_c > D0C)
        prg_gcw = jnp.where(gcw_gate, rhof * t1_qg_qc * ef_gw * rc * n0_g * ilamg**cge9, 0.0)

        # ---- cold block (:2554-2780) ----
        rate_max64 = rate_max.astype(f64)
        rr_idx = jnp.maximum(rr, R_R_FIRST)
        idx_r = _lookup_digit_index(rr_idx, -6, N_R_TABLE)
        idx_r1 = _lookup_digit_index(ORG1 * rr_idx / _am_r() * _dpow(lamr, CRE1), 6, N_R1_TABLE)
        idx_tc = jnp.clip(jnp.floor(-tempc + 0.5).astype(jnp.int32) - 1, 0, N_TC_TABLE - 1)
        qrfz = _take_qrfz(tables.qrfz, idx_r, idx_r1, idx_tc)
        rfz_tab = cold & (rr > R_R_FIRST)
        rfz_hom = cold & ~rfz_tab & l_qr & (temp < HGFR)
        pri_rfz = jnp.where(rfz_tab, qrfz[..., 0] * odts, jnp.where(rfz_hom, rr * odts, 0.0)).astype(f64)
        prg_rfz = jnp.where(rfz_tab, qrfz[..., 1] * odts, 0.0)
        pni_rfz = jnp.where(rfz_tab, qrfz[..., 2] * odts, jnp.where(rfz_hom, nr * odts, 0.0)).astype(f64)
        pnr_rfz = jnp.where(rfz_tab, jnp.minimum((nr * odts).astype(f64), qrfz[..., 3] * odts), 0.0)
        _cf, _cn, pri_wfz, pni_wfz = _cloud_water_freezing_rates(state, dt, cold_tables)
        pri_wfz = jnp.where(cold, pri_wfz, 0.0).astype(f64)
        pni_wfz = jnp.where(cold, pni_wfz, 0.0).astype(f64)

        nuc = cold & ((ssati >= 0.25) | ((ssatw > EPS) & (temp < 253.15)))
        xnc = jnp.minimum(250.0e3, TNO * jnp.exp(ATO * (T_0 - temp)))
        xni = (ni + (pni_rfz + pni_wfz) * dtf).astype(jnp.float32)
        pni_inu = (0.5 * (xnc - xni + jnp.abs(xnc - xni)) * odts).astype(f64)
        pri_inu = jnp.where(nuc, jnp.minimum(rate_max64, XM0I * pni_inu), 0.0)
        pni_inu = jnp.where(nuc, pri_inu / XM0I, 0.0)

        t1_subl, rvs = _wrf_subl_prefactor(state, ssati, diffu, tcond, qvsi)
        oxmi = 1.0 / xmi
        pri_ide = C_CUBE * t1_subl * diffu * ssati * rvs * OIG1 * 1.0 * ni * ilami
        ide_neg = pri_ide < 0.0
        pri_ide = jnp.where(ide_neg, jnp.maximum(jnp.maximum((-ri * odts).astype(f64), pri_ide), rate_max64),
                            jnp.minimum(pri_ide, rate_max64))
        idx_i = jnp.where(ri > R_I_FIRST, _lookup_digit_index(ri, -10, N_I_TABLE), 0)
        idx_i1 = jnp.where(ni > NT_I_FIRST, _lookup_digit_index(ni, 0, N_I1_TABLE), 0)
        iaus = _take3_last(tables.iaus, idx_i, idx_i1)
        dep_i = cold & l_qi
        pni_ide = jnp.where(dep_i & ide_neg, jnp.maximum((-ni * odts).astype(f64), pri_ide * oxmi), 0.0)
        prs_ide = jnp.where(dep_i & ~ide_neg, (1.0 - iaus[..., 2]) * pri_ide, 0.0)
        pri_ide = jnp.where(dep_i, jnp.where(ide_neg, pri_ide, iaus[..., 2] * pri_ide), 0.0)
        iau_large = (idx_i == N_I_TABLE - 1) | (xdi > 5.0 * D0S)
        iau_small = xdi < _wrf("D0S_X01", 0.1 * D0S)
        prs_iau = jnp.where(iau_large, (ri * 0.99 * odts).astype(f64), jnp.where(
            iau_small, 0.0, jnp.minimum((ri * 0.99 * odts).astype(f64), iaus[..., 0] * odts)))
        pni_iau = jnp.where(iau_large, (ni * 0.95 * odts).astype(f64), jnp.where(
            iau_small, 0.0, jnp.minimum((ni * 0.95 * odts).astype(f64), iaus[..., 1] * odts)))
        prs_iau, pni_iau = jnp.where(dep_i, prs_iau, 0.0), jnp.where(dep_i, pni_iau, 0.0)

        prs_sde_c = (c_snow * t1_subl * diffu * ssati * rvs * (T1_SUBL_QS * smo1 + T2_SUBL_QS * rhof2 * vsc2 * smof)).astype(f64)
        prs_sde_c = jnp.where(prs_sde_c < 0.0, jnp.maximum(jnp.maximum((-rs * odts).astype(f64), prs_sde_c), rate_max64),
                              jnp.minimum(prs_sde_c, rate_max64))
        prs_sde_c = jnp.where(cold & l_qs, prs_sde_c, 0.0)
        gde_base = (C_CUBE * t1_subl * diffu * ssati * rvs) * n0_g * (
            T1_SUBL_QG * _dpow(ilamg, CRE10) + t2_subl_qg * vsc2 * rhof2 * ilamg**cge11)
        prg_gde_c = jnp.where(gde_base < 0.0, jnp.maximum(jnp.maximum((-rg * odts).astype(f64), gde_base), rate_max64),
                              jnp.minimum(gde_base, rate_max64))
        gde_c = cold & l_qg & (ssati < -EPS)
        png_gde_c = jnp.where(gde_c & (prg_gde_c < 0.0), prg_gde_c * ng / rg, 0.0)
        prg_gde_c = jnp.where(gde_c, prg_gde_c, 0.0)

        prs_sci, pni_sci, pri_rci, pni_rci, prr_rci, pnr_rci, prg_rci = _ice_collection_rates_from_moments(
            dt, rhof, ri, ni, xdi, xmi, l_qi, rs, smoe, rr, nr, lamr, mvd_r, n0_r, active_rain, cold=cold)

        # Hallett-Mossop rime splinters (:2741-2752), before the rime split.
        tf = jnp.where((tempc >= -5.0) & (tempc < -3.0), 0.5 * (-3.0 - tempc),
                       jnp.where((tempc > -8.0) & (tempc < -5.0), 0.33333333 * (8.0 + tempc), 0.0))
        hm = cold & (prg_gcw > EPS) & (tempc > -8.0)
        pni_ihm = jnp.where(hm, 3.5e8 * tf * prg_gcw, 0.0)
        pri_ihm = XM0I * pni_ihm
        rime_sum = jnp.where(hm, prs_scw + prg_gcw, 1.0)
        prs_ihm = jnp.where(hm, prs_scw / rime_sum * pri_ihm, 0.0)
        prg_ihm = jnp.where(hm, prg_gcw / rime_sum * pri_ihm, 0.0)

        # Rimed snow -> graupel split (:2758-2777), cold block only, against the per-cell prs_sde.
        riming_dom = cold & (prs_scw > 2.0 * prs_sde_c) & (prs_sde_c > EPS)
        r_frac = jnp.minimum(30.0, prs_scw / jnp.where(riming_dom, prs_sde_c, 1.0)).astype(jnp.float32)
        g_frac = jnp.minimum(0.95, 0.15 + (r_frac - 2.0) * 0.028)
        vts_boost = jnp.where(riming_dom, jnp.minimum(1.5, 1.1 + (r_frac - 2.0) * 0.014), 1.0).astype(jnp.float32)
        vts_single = AV_S * xds**BV_S * jnp.exp(-FV_S * xds)
        const_ri = jnp.clip(-(mvd_c * 0.5e6) * vts_single / jnp.minimum(-0.1, tempc), 0.1, 10.0)
        rime_dens = (0.051 + 0.114 * const_ri - 0.0055 * const_ri * const_ri) * 1000.0
        g_frac = jnp.where(riming_dom & (rime_dens >= 150.0), g_frac, 0.0)
        prg_scw = g_frac * prs_scw
        png_scw = jnp.where(g_frac > 0.0, prg_scw * smo0 / rs, 0.0)
        prs_scw = jnp.where(riming_dom, (1.0 - g_frac) * prs_scw, prs_scw)

        # ---- warm block: melt, enhancement, sublimation only when not melting (:2781-2834) ----
        melt_base = tempc * tcond - 2.5e6 * diffu * del_qvs
        prr_sml = (melt_base * (_wrf("T1_MELT_QS", T1_MELT_QS) * smo1 + _wrf("T2_MELT_QS", T2_MELT_QS) * rhof2 * vsc2 * smof)).astype(f64)
        prr_sml = jnp.where(prr_sml > 0.0, prr_sml + 4218.0 * _OLFUS32 * (twet - T_0) * (prr_rcs + prs_scw), prr_sml)
        prr_sml = jnp.where(warm & l_qs, jnp.minimum((rs * odts).astype(f64), jnp.maximum(0.0, prr_sml)), 0.0)
        pnr_sml = jnp.where(prr_sml > 0.0, smo0 / rs * prr_sml * 10.0 ** (-0.25 * (twet - T_0)), 0.0)
        sde_w = warm & l_qs & ~(prr_sml > 0.0) & (ssati < 0.0)
        prs_sde_w = (C_SQRD * t1_subl * diffu * ssati * rvs * (T1_SUBL_QS * smo1 + T2_SUBL_QS * rhof2 * vsc2 * smof)).astype(f64)
        prs_sde = jnp.where(cold, prs_sde_c, jnp.where(sde_w, jnp.maximum((-rs * odts).astype(f64), prs_sde_w), 0.0))
        n0_melt = jnp.where((rg * ng) < 1.0e-4, (1.0e-4 / rg) * ORG2 * lamg, n0_g)
        prr_gml = melt_base * n0_melt * (T1_MELT_QG * _dpow(ilamg, CRE10) + t2_melt_qg * rhof2 * vsc2 * ilamg**cge11)
        prr_gml = jnp.where(warm & l_qg, jnp.minimum((rg * odts).astype(f64), jnp.maximum(0.0, prr_gml)), 0.0)
        pnr_gml = jnp.where(prr_gml > 0.0, prr_gml * ng / rg * 10.0 ** (-0.33 * (twet - T_0)), 0.0)
        gde_w = warm & l_qg & ~(prr_gml > 0.0) & (ssati < 0.0)
        prg_gde_w = jnp.maximum((-rg * odts).astype(f64), gde_base)
        prg_gde = jnp.where(cold, prg_gde_c, jnp.where(gde_w, prg_gde_w, 0.0))
        png_gde = jnp.where(cold, png_gde_c, jnp.where(gde_w, prg_gde_w * ng / rg, 0.0))
    else:  # no ice/snow/graupel > R1, no cold cloud/rain, no nucleation (_wrf_mixed_phase_activity): all 0, twet = T
        ri, l_qi = jnp.maximum(state.qi * rho, R1), state.qi > R1
        rs, l_qs = jnp.maximum(state.qs * rho, R1), state.qs > R1
        rg, l_qg = jnp.maximum(state.qg * rho, R1), state.qg > R1
        twet, vts_boost = temp, jnp.ones_like(temp)
        png_gde = png_rcg = png_rcs = png_scw = pni_iau = pni_ide = pni_ihm = pni_inu = zero64
        pni_rci = pni_rfz = pni_sci = pni_wfz = pnr_gml = pnr_rcg = pnr_rci = pnr_rcs = zero64
        pnr_rfz = pnr_sml = prg_gcw = prg_gde = prg_ihm = prg_rcg = prg_rci = prg_rcs = zero64
        prg_rfz = prg_scw = pri_ide = pri_ihm = pri_inu = pri_rci = pri_rfz = pri_wfz = zero64
        prr_gml = prr_rcg = prr_rci = prr_rcs = prr_sml = prs_iau = prs_ide = prs_ihm = zero64
        prs_rcs = prs_sci = prs_scw = prs_sde = zero64
    if dtf > 120.0:                                                                       # :2840-2844, warm block
        prr_rcw = jnp.where(warm, prr_rcw + prs_scw + prg_gcw, prr_rcw)
        prs_scw, prg_gcw = jnp.where(warm, 0.0, prs_scw), jnp.where(warm, 0.0, prg_gcw)

    # ---- conservation limiters (:2853-2955; REAL sump/ratio) ----
    real = jnp.float32
    sump = (pri_inu + pri_ide + prs_ide + prs_sde + prg_gde).astype(real)
    lim = ((sump > EPS) & (sump > rate_max)) | ((sump < -EPS) & (sump < rate_max))
    ratio = rate_max / jnp.where(lim, sump, 1.0)
    pri_inu, pri_ide, pni_ide, prs_ide, prs_sde, prg_gde = _wrf_scale(lim, ratio, pri_inu, pri_ide, pni_ide, prs_ide, prs_sde, prg_gde)
    sump = (-prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw).astype(real)
    lim = (sump < -rc * odts) & l_qc
    ratio = -rc * odts / jnp.where(lim, sump, 1.0)
    prr_wau, pri_wfz, prr_rcw, prs_scw, prg_scw, prg_gcw = _wrf_scale(lim, ratio, prr_wau, pri_wfz, prr_rcw, prs_scw, prg_scw, prg_gcw)
    sump = (pri_ide - prs_iau - prs_sci - pri_rci).astype(real)
    lim = (sump < -ri * odts) & l_qi
    ratio = -ri * odts / jnp.where(lim, sump, 1.0)
    pri_ide, prs_iau, prs_sci, pri_rci = _wrf_scale(lim, ratio, pri_ide, prs_iau, prs_sci, pri_rci)
    sump = (-prg_rfz - pri_rfz - prr_rci + prr_rcs + prr_rcg).astype(real)
    lim = (sump < -rr * odts) & l_qr
    ratio = -rr * odts / jnp.where(lim, sump, 1.0)
    prg_rfz, pri_rfz, prr_rci, prr_rcs, prr_rcg = _wrf_scale(lim, ratio, prg_rfz, pri_rfz, prr_rci, prr_rcs, prr_rcg)
    sump = (prs_sde - prs_ihm - prr_sml + prs_rcs).astype(real)
    lim = (sump < -rs * odts) & l_qs
    ratio = -rs * odts / jnp.where(lim, sump, 1.0)
    prs_sde, prs_ihm, prr_sml, prs_rcs = _wrf_scale(lim, ratio, prs_sde, prs_ihm, prr_sml, prs_rcs)
    sump = (prg_gde - prg_ihm - prr_gml + prg_rcg).astype(real)
    lim = (sump < -rg * odts) & l_qg
    ratio = -rg * odts / jnp.where(lim, sump, 1.0)
    prg_gde, prg_ihm, prr_gml, prg_rcg = _wrf_scale(lim, ratio, prg_gde, prg_ihm, prr_gml, prg_rcg)
    pri_ihm = prs_ihm + prg_ihm
    ratio = jnp.minimum(jnp.abs(prr_rcg), jnp.abs(prg_rcg)).astype(real)                 # :2943-2951
    prr_rcg = (ratio * jnp.where(prr_rcg.astype(real) < 0.0, -1.0, 1.0)).astype(f64)
    prg_rcg = -prr_rcg
    rebal = twet > T_0
    ratio = jnp.minimum(jnp.abs(prr_rcs), jnp.abs(prs_rcs)).astype(real)
    prr_rcs = jnp.where(rebal, (ratio * jnp.where(prr_rcs.astype(real) < 0.0, -1.0, 1.0)).astype(f64), prr_rcs)
    prs_rcs = jnp.where(rebal, -prr_rcs, prs_rcs)

    # ---- REAL tendencies (:2971-3180) and the one state update (:3195-3204) ----
    def ten(x):
        return (x * orho).astype(real)

    qvten = ten(-pri_inu - pri_ide - prs_ide - prs_sde - prg_gde)
    qcten = ten(-prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw)
    qiten = ten(pri_inu + pri_ihm + pri_wfz + pri_rfz + pri_ide - prs_iau - prs_sci - pri_rci)
    niten = ten(pni_inu + pni_ihm + pni_wfz + pni_rfz + pni_ide - pni_iau - pni_sci - pni_rci)
    qrten = ten(prr_wau + prr_rcw + prr_sml + prr_gml + prr_rcs + prr_rcg - prg_rfz - pri_rfz - prr_rci)
    nrten = ten(pnr_wau + pnr_sml + pnr_gml - (pnr_rfz + pnr_rcr + pnr_rcg + pnr_rcs + pnr_rci + pni_rfz))
    qsten = ten(prs_iau + prs_sde + prs_sci + prs_scw + prs_rcs + prs_ide - prs_ihm - prr_sml)
    qgten = ten(prg_scw + prg_rfz + prg_gde + prg_rcg + prg_gcw + prg_rci + prg_rcs - prg_ihm - prr_gml)
    ngten = ten(png_scw + pnr_rfz - png_rcg + pnr_rci + png_rcs + png_gde - pnr_gml)
    ni1d, nr1d, ng1d = state.Ni, state.Nr, state.Ng          # input per-kg numbers (mp8 ng1d: entry diagnostic)

    if cold_rates:
        xri = jnp.maximum(R1, (state.qi + qiten * dtf) * rho)                                # :2999-3020
        xni = jnp.maximum(R2, (ni1d + niten * dtf) * rho)
        lami = ((_am_i() * 6.0 * OIG1 * xni / xri) ** OBMI).astype(f64)
        xdi_t = ((3.0 + 0.0 + 1.0) * (1.0 / lami)).astype(real)                          # REAL xDi
        ni_small = jnp.minimum(999.0e3, OIG2 * xri / _am_i() * _dpow(CIE2 / 5.0e-6, 3.0))
        ni_large = OIG2 * xri / _am_i() * _dpow(CIE2 / 300.0e-6, 3.0)
        niten = jnp.where(xri > R1, jnp.where(xdi_t < 5.0e-6, ((ni_small - ni1d * rho) * odts * orho).astype(real),
                          jnp.where(xdi_t > 300.0e-6, ((ni_large - ni1d * rho) * odts * orho).astype(real), niten)),
                          -ni1d * odts)
        xni = jnp.maximum(0.0, (ni1d + niten * dtf) * rho)
        niten = jnp.where(xni > 999.0e3, (999.0e3 - ni1d * rho) * odts * orho, niten)
    else:  # qi = Ni = 0 (no ice > R1 in the column): xri = R1, the cap selects -ni1d*odts
        niten = -ni1d * odts

    if warm_rates or cold_rates:
        xrr = jnp.maximum(R1, (state.qr + qrten * dtf) * rho)                                # :3035-3055
        xnr = jnp.maximum(R2, (nr1d + nrten * dtf) * rho)
        mvd_t = ((3.0 + 0.672) / ((_am_r() * CRG3 * ORG2 * xnr / xrr) ** OBMR).astype(f64)).astype(real)
        mvd_c_r = jnp.minimum(2.5e-3, jnp.maximum(_wrf("D0R_X075", D0R * 0.75), mvd_t))
        xnr_c = (CRG2 * ORG3 * xrr * _dpow(((3.0 + 0.672) / mvd_c_r).astype(f64), 3.0) / _am_r()).astype(real)
        nrten = jnp.where(xrr > R1, jnp.where(mvd_c_r != mvd_t, ((xnr_c - nr1d * rho) * odts * orho).astype(real), nrten),
                          -nr1d * odts)
        qrten = jnp.where(xrr > R1, qrten, -state.qr * odts)
    else:  # dry column: qr = Nr = 0, xrr = R1
        nrten, qrten = -nr1d * odts, -state.qr * odts

    if cold_rates:
        xrg = jnp.maximum(R1, (state.qg + qgten * dtf) * rho)                                # :3074-3129
        xng = jnp.maximum(R2, (ng1d + ngten * dtf) * rho)
        mvd_t = ((3.0 + 0.672) / ((AM_G_MP8 * CRG3 * ORG2 * xng / xrg) ** OBMG).astype(f64)).astype(real)
        mvd_c_g = jnp.minimum(25.4e-3, jnp.maximum(D0R, mvd_t))
        xng_c = (ORG3 * xrg * _dpow(((3.0 + 0.672) / mvd_c_g).astype(f64), 3.0) / AM_G_MP8).astype(real)
        ngten = jnp.where(xrg > R1, jnp.where(mvd_c_g != mvd_t, ((xng_c - ng1d * rho) * odts * orho).astype(real), ngten),
                          -ng1d * odts)
        qgten = jnp.where(xrg > R1, qgten, -state.qg * odts)
    else:  # qg = 0 and no graupel source: xrg = R1
        ngten, qgten = -ng1d * odts, -state.qg * odts

    lfus2 = LSUB - lvap
    tten = jnp.where(
        cold,
        (LSUB * ocp * (pri_inu + pri_ide + prs_ide + prs_sde + prg_gde)
         + lfus2 * ocp * (pri_wfz + pri_rfz + prg_rfz + prs_scw + prg_scw + prg_gcw + prg_rcs + prs_rcs + prr_rci + prg_rcg)) * orho,
        (LFUS * ocp * (-prr_sml - prr_gml - prr_rcg - prr_rcs) + LSUB * ocp * (prs_sde + prg_gde)) * orho).astype(real)

    new_t = temp + dtf * tten
    new_qv = jnp.maximum(1.0e-10, qv + dtf * qvten)
    if vapor is not None:
        # WRF :3192 uses raw qv1d plus the REAL source tendency. All rates,
        # saturation and heat capacities above use the private working qv.
        qv1d, _ = vapor
        new_qv = jnp.where(qv1d < 1.0e-10, jnp.maximum(1.0e-10, qv1d + dtf * qvten), new_qv)
    out = state.replace(
        qv=new_qv, qc=state.qc + qcten * dtf, qr=state.qr + qrten * dtf, qi=state.qi + qiten * dtf,
        qs=state.qs + qsten * dtf, qg=state.qg + qgten * dtf, Ni=ni1d + niten * dtf, Nr=nr1d + nrten * dtf,
        Ng=ng1d + ngten * dtf, T=new_t, rho=density_from_pressure_temperature(p, new_t, new_qv))
    # Melting snow (prr_sml > 0) falls at the snow/rain blend instead of vts*vts_boost (:3721-3726); the sign
    # carries that to _fall_speeds (vts_boost >= 1 everywhere else).
    vts_boost = jnp.where(prr_sml > 0.0, -1.0, vts_boost).astype(jnp.float32)
    if vapor is not None:
        return _real_state(out), prr_gml, vts_boost, qvten
    return _real_state(out), prr_gml, vts_boost


def _wrf_snow_speed(vts, vts_boost, rs, rr, vtr):
    """Snow mass fall speed (:3721-3726, REAL): SR = rs/(rs+rr) blend with the rain speed where melting."""
    sr = rs / (rs + rr)
    return jnp.where(vts_boost < 0.0, vts * sr + (1.0 - sr) * vtr.astype(vts.dtype), vts * vts_boost)
