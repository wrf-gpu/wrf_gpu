"""JAX Thompson-column source/sink subset for M5-S1."""

from __future__ import annotations

import os
from functools import partial
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.physics.thompson_constants import (
    AM_G_MP8,
    AM_I,
    AM_R,
    AM_S,
    AV_I,
    AV_R,
    AV_S,
    AV_G_MP8,
    BV_G_MP8,
    BV_I,
    BV_R,
    BV_S,
    ATO,
    C_CUBE,
    C_SQRD,
    CCG2_NU12,
    CCG3_NU12,
    CGE11,
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
    EPS,
    FV_R,
    HGFR,
    LFUS,
    LSUB,
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
    T1_MELT_QG,
    T1_MELT_QS,
    T1_QR_EV,
    T1_QR_QC,
    T1_SUBL_QG,
    T1_SUBL_QS,
    T2_MELT_QG,
    T2_MELT_QS,
    T2_QR_EV,
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


def _fp32_enabled() -> bool:
    return os.environ.get("GPUWRF_THOMPSON_FP32", "0") == "1"


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


def _rain_distribution(qr, Nr, rho):
    """Encapsulates WRF rain slope/intercept equations from lines 2210-2215."""

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


def _graupel_distribution(qg, rho):
    """Provides the mp=8 graupel slope/intercept terms used by WRF formulas."""

    rg = jnp.maximum(qg * rho, R1)
    ng = jnp.maximum(4.0e5 * rho, R2)
    lamg = (AM_G_MP8 * CRG3 * ORG2 * ng / rg) ** (1.0 / 3.0)
    ilamg = 1.0 / lamg
    n0_g = ng * ORG2 * lamg
    return rg, ng, lamg, ilamg, n0_g, qg > R1


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
    pnr_wau = prr_wau / jnp.maximum(AM_R * NU_C_MP8 * 10.0 * D0R**3, R2)

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

    autoconv = prr_wau * float(dt) / state.rho
    accretion = prr_rcw * float(dt) / state.rho
    transfer = jnp.minimum(state.qc, autoconv + accretion)
    nr_gain = pnr_wau * float(dt) / state.rho
    return state.replace(qc=state.qc - transfer, qr=state.qr + transfer, Nr=state.Nr + nr_gain)


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
    state: ThompsonColumnState, dt: float, tables: ThompsonTableBundle = THOMPSON_TABLES
) -> tuple[ThompsonColumnState, jax.Array]:
    """Stages WRF-mapped ice tendencies before condensation and rain evaporation."""

    ocp = cp_inverse(state.qv)
    lvap = latent_heat_vaporization(state.T)
    lfus2 = LSUB - lvap

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
    nr_loss = jnp.minimum(state.Nr, table_nr + table_ni + fallback_ni)
    state = state.replace(
        qr=state.qr - ice_freeze - graupel_freeze,
        qi=state.qi + ice_freeze,
        qg=state.qg + graupel_freeze,
        Ni=state.Ni + table_ni + fallback_ni,
        Nr=jnp.maximum(0.0, state.Nr - nr_loss),
        T=state.T + lfus2 * ocp * (ice_freeze + graupel_freeze),
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
    rg, ng, _lamg, ilamg, n0_g, active_graupel = _graupel_distribution(state.qg, state.rho)

    # Snow/graupel melting formula structure from WRF lines 2845-2889.
    prr_sml_rate = (tempc * tcond - 2.5e6 * diffu * del_qvs) * (T1_MELT_QS * smo1 + T2_MELT_QS * rhof2 * vsc2 * smof)
    prr_sml_rate = jnp.minimum(rs / float(dt), jnp.maximum(0.0, prr_sml_rate)) / state.rho
    snow_melt = jnp.where((state.T > T_0) & active_snow, prr_sml_rate * float(dt), 0.0)
    pnr_sml = jnp.where(rs > R1, smo0 / rs * snow_melt * state.rho * 10.0 ** (-0.25 * (twet - T_0)), 0.0)

    prr_gml_rate = (tempc * tcond - 2.5e6 * diffu * del_qvs) * n0_g * (
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

    tempc, diffu, _visco, tcond, _lvap, ocp, rhof, rhof2, vsc2 = _air_properties(state)
    del tempc, rhof
    qvsi = saturation_mixing_ratio_ice(state.p, state.T)
    ssati = state.qv / qvsi - 1.0
    t1_subl, rvs = _sublimation_prefactor(state, ssati, diffu, tcond)
    ri, ni, _lami, ilami, xdi, xmi, active_ice = _ice_distribution(state.qi, state.Ni, state.rho)
    rs, _xds, _smo0, smo1, smof, c_snow, active_snow = _snow_moments(state.qs, state.rho, state.T - 273.15, tables)
    rg, ng, _lamg, ilamg, n0_g, active_graupel = _graupel_distribution(state.qg, state.rho)

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

    prs_sde = c_snow * t1_subl * diffu * ssati * rvs * (T1_SUBL_QS * smo1 + T2_SUBL_QS * rhof2 * vsc2 * smof)
    prs_sde = jnp.where(active_snow, jnp.where(prs_sde < 0.0, jnp.maximum(-rs / float(dt), prs_sde), jnp.minimum(prs_sde, jnp.maximum(state.qv - qvsi, 0.0) * state.rho / float(dt) * 0.999)), 0.0)
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

    ice_deposition = pri_ide * float(dt) / state.rho
    snow_from_ice_deposition = prs_ide * float(dt) / state.rho
    ice_to_snow = prs_iau_mass / state.rho
    ice_number_to_snow = pni_iau_num / state.rho
    snow_deposition = prs_sde * float(dt) / state.rho
    graupel_deposition = prg_gde * float(dt) / state.rho
    vapor_sink = jnp.maximum(0.0, ice_deposition) + jnp.maximum(0.0, snow_from_ice_deposition) + jnp.maximum(0.0, snow_deposition) + jnp.maximum(0.0, graupel_deposition)
    vapor_source = jnp.maximum(0.0, -ice_deposition) + jnp.maximum(0.0, -snow_deposition) + jnp.maximum(0.0, -graupel_deposition)
    updated_qv = state.qv - vapor_sink + vapor_source - inu_mass
    updated_T = state.T + LSUB * ocp * (vapor_sink - vapor_source + inu_mass)
    updated = state.replace(
        qv=updated_qv,
        qi=state.qi + ice_deposition - ice_to_snow + inu_mass,
        qs=state.qs + snow_from_ice_deposition + ice_to_snow + snow_deposition,
        qg=state.qg + graupel_deposition,
        # WRF lines 2719-2727 update pni_ide only in sublimation; positive
        # deposition partitions mass but does not create new cloud-ice number.
        Ni=jnp.maximum(0.0, state.Ni + inu_number + jnp.where(ice_deposition < 0.0, ice_deposition / jnp.maximum(xmi, XM0I), 0.0) - ice_number_to_snow),
        T=updated_T,
        rho=density_from_pressure_temperature(state.p, updated_T, updated_qv),
    )
    return updated, graupel_melt


def _ice_sources(state: ThompsonColumnState, dt: float, tables: ThompsonTableBundle = THOMPSON_TABLES) -> ThompsonColumnState:
    """Legacy wrapper for callers that do not need process flags."""

    updated, _graupel_melt = _ice_sources_with_process_flags(state, dt, tables)
    return updated


# Maximum sedimentation sub-steps. WRF chooses nstep per column from the CFL
# condition (module_mp_thompson.F:3634-3641); we apply a fixed-cap explicit
# sub-stepped upwind flux and damp the per-substep increment by 1/nstep so the
# scheme is stable and JIT-static.  64 sub-steps covers fast graupel at WRF's
# typical dz/dt; columns needing fewer simply repeat a converged state.
#
# CFL floor: for fall speeds up to ~20 m/s (graupel) at the domain's thinnest
# layer (dz ~ 48 m) and dt = 18 s, explicit-upwind stability needs
# nstep >= vt*dt/dz ~ 7.5, so 64 is ~8x oversampled.  ``GPUWRF_THOMPSON_NSED``
# allows tuning the substep count; any reduction is gated on the moist WRF
# oracle re-validation (it changes the time-integration accuracy, not just
# launches).  Default stays 64 (byte-identical to the prior behaviour).
def _nsed_substeps() -> int:
    try:
        return max(1, int(os.environ.get("GPUWRF_THOMPSON_NSED", "64")))
    except ValueError:
        return 64


NSED_SUBSTEPS = _nsed_substeps()


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


def _sedimentation_implicit(state: ThompsonColumnState, dt: float, nsub: int):
    """EXPERIMENTAL implicit backward-Euler sedimentation (gated, default OFF)."""

    vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num = _fall_speeds(state)
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


def _fall_speeds(state: ThompsonColumnState):
    """Mass/number terminal fall speeds per species (m/s), WRF formulas.

    Rain:    module_mp_thompson.F:3616-3628 (vtrk mass, vtnrk number).
    Ice:     module_mp_thompson.F:3678-3691 (vtik mass, vtnik number).
    Snow:    bulk mass-weighted speed from the snow slope (WRF uses the Field
             two-gamma moments; here the single-slope WRF av_s/bv_s closure on
             the snow characteristic diameter — faithful for the mp=8 default
             snow when the racs/sacr boost is inactive).
    Graupel: module_mp_thompson.F:3758-3766 mass speed with av_g/bv_g (idx_bg1).
    """

    rho = state.rho
    rhof = _rho_correction(rho)

    act_r = state.qr > R1
    rr = jnp.maximum(state.qr * rho, R1)
    nr = jnp.maximum(state.Nr * rho, R2)
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

    # Snow bulk fall speed via the Field-moment slope: xDs = smoc/smob is the
    # mass-weighted mean diameter; WRF combines av_s with the two-gamma fit.
    # We use the WRF single-mode closure vts = rhof*av_s*Ds^bv_s as a faithful
    # mp=8 approximation when riming-boost (racs) is inactive.
    act_s = state.qs > R1
    tempc = state.T - 273.15
    _rs2, xds, _smo0, _smo1, _smof, _csnow, _act_s = _snow_moments(state.qs, rho, tempc)
    vt_s_mass = _fill_down(jnp.where(act_s, rhof * AV_S * jnp.maximum(xds, D0S) ** BV_S, 0.0), act_s)

    act_g = state.qg > R1
    rg = jnp.maximum(state.qg * rho, R1)
    ng = jnp.maximum(state.Ng * rho, R2)
    ng = jnp.where(state.Ng > 0.0, ng, jnp.maximum(4.0e5 * rho, R2))
    lamg = (AM_G_MP8 * CRG3 * ORG2 * ng / rg) ** OBMG
    ilamg = 1.0 / lamg
    vt_g_mass = _fill_down(jnp.where(act_g, rhof * AV_G_MP8 * 6.0 * ORG3 * ilamg ** BV_G_MP8, 0.0), act_g)
    vt_g_num = _fill_down(jnp.where(act_g, rhof * AV_G_MP8 * CRG7 / CRG12 * ilamg ** BV_G_MP8, 0.0), act_g)

    return (vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num)


def _sed_one_species(q, num, vt_mass, vt_num, dz, rho, dt):
    """One upwind-flux sedimentation pass for a mixing ratio + its number.

    Mirrors WRF module_mp_thompson.F:3790-3939 explicit upwind flux divergence,
    sub-stepped by 1/NSED_SUBSTEPS.  Axis -1 is vertical with index 0 = surface
    (kts) and the last index = model top (kte); precipitation leaves through the
    surface face.  Returns (q', num', surface_precip_mm), with q'/num' cast back
    to the input field dtype.

    Accumulation runs in the result dtype of (q, num, vt, rho, dz) — fp64 here —
    so sedimentation never silently downcasts the flux integration even when the
    incoming State fields are fp32 (ADR-007 fp32-gated). The scan carry is held
    at that accumulation dtype to keep carry-in/carry-out types consistent.
    """

    sub = 1.0 / float(NSED_SUBSTEPS)
    dt_sub = float(dt) * sub
    acc_dtype = jnp.result_type(q.dtype, num.dtype, vt_mass.dtype, rho.dtype, dz.dtype)
    q_dt, num_dt = q.dtype, num.dtype
    q0 = q.astype(acc_dtype)
    num0 = num.astype(acc_dtype)

    def body(carry, _):
        q_c, num_c, ppt_c = carry
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
        dq = (flux_q_above - flux_q) / dz / rho * dt_sub
        dn = (flux_n_above - flux_n) / dz / rho * dt_sub
        q_new = jnp.maximum(q_c + dq, 0.0)
        num_new = jnp.maximum(num_c + dn, 0.0)
        # Surface precip = downward flux leaving the bottom (index 0) face.
        surf = flux_q[..., 0] * dt_sub  # kg m^-2 == mm
        return (q_new, num_new, ppt_c + surf), None

    zero_ppt = jnp.zeros(q.shape[:-1], dtype=acc_dtype)
    (q_out, num_out, ppt), _ = jax.lax.scan(
        body, (q0, num0, zero_ppt), None, length=NSED_SUBSTEPS, unroll=_sed_unroll()
    )
    return q_out.astype(q_dt), num_out.astype(num_dt), ppt


def _sedimentation(state: ThompsonColumnState, dt: float):
    """Faithful WRF sedimentation of rain/ice/snow/graupel; returns precip mm.

    WRF module_mp_thompson.F:3784-3939.  Cloud-water sedimentation (very small)
    is neglected, matching WRF's near-zero contribution above the lowest 500 m
    where vtc is only computed; the dominant precip channels (rain, snow,
    graupel, ice) are advected here.  Snow/graupel numbers (Ns/Ng) follow their
    mass since the mp=8 default carries diagnostic snow number and a fixed
    graupel intercept; Ng falls with the mass flux when present.
    """

    nsub = _implicit_sed_nsub()
    if nsub > 0:
        # EXPERIMENTAL implicit backward-Euler sedimentation (gated, default OFF).
        return _sedimentation_implicit(state, dt, nsub)

    vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num = _fall_speeds(state)
    dz = jnp.maximum(state.dz, 1.0)
    rho = jnp.maximum(state.rho, R1)

    # Four independent per-species substep scans. XLA already overlaps these four
    # independent scans well; the per-scan ``unroll`` (``_sed_unroll``) fuses
    # adjacent substeps to cut the launch count. (A single 4-species batched scan
    # was measured SLOWER — it serialises what XLA otherwise parallelises — so we
    # keep the per-species structure; see proofs/thompson_perf.)
    qr, Nr, ppt_rain = _sed_one_species(state.qr, state.Nr, vt_r_mass, vt_r_num, dz, rho, dt)
    qi, Ni, ppt_ice = _sed_one_species(state.qi, state.Ni, vt_i_mass, vt_i_num, dz, rho, dt)
    # Snow: number tracks mass (diagnostic Ns).  Use the mass speed for both.
    qs, Ns, ppt_snow = _sed_one_species(state.qs, state.Ns, vt_s_mass, vt_s_mass, dz, rho, dt)
    qg, Ng, ppt_graupel = _sed_one_species(state.qg, state.Ng, vt_g_mass, vt_g_num, dz, rho, dt)

    updated = state.replace(qr=qr, Nr=Nr, qi=qi, Ni=Ni, qs=qs, Ns=Ns, qg=qg, Ng=Ng)
    precip = {
        "rain": ppt_rain,
        "snow": ppt_snow,
        "graupel": ppt_graupel,
        "ice": ppt_ice,
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
    valid = _thermodynamically_admissible(state)
    fallback = state
    state = _debug_checks(state, debug)
    state = _warm_rain_collection(state, dt)
    state, graupel_melt = _ice_sources_with_process_flags(state, dt)
    state, cloud_condensed = _saturation_adjustment_with_condensation(state, dt)
    state = _rain_evaporation(state, dt, skip_evaporation=cloud_condensed, graupel_melt=graupel_melt)
    if sediment:
        state, precip = _sedimentation(state, dt)
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


@partial(jax.jit, static_argnames=("dt", "debug"))
def step_thompson_column_with_precip(state: ThompsonColumnState, dt: float, *, debug: bool = False):
    """Advances one full Thompson column step; returns ``(State, precip-dict-mm)``.

    This is the operational coupling entry: it runs the source/sink processes
    AND faithful sedimentation, returning the surface precipitation accumulated
    over the step (mm) per channel for the State precip accumulators.
    """

    return _step_thompson_column_full_impl(state, dt, debug)
