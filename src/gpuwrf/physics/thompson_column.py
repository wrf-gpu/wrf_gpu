"""JAX Thompson-column source/sink subset for M5-S1."""

from __future__ import annotations

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
    C_CUBE,
    C_SQRD,
    CCG2_NU12,
    CCG3_NU12,
    CGE11,
    CIE2,
    CRE10,
    CRE11,
    CRE2,
    CRE9,
    CRG2,
    CRG3,
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
    OBMI,
    OBMR,
    OCG1_NU12,
    OCG2_NU12,
    OIG1,
    OIG2,
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
    XM0I,
)
from gpuwrf.physics.thompson_saturation import (
    cp_inverse,
    latent_heat_vaporization,
    saturation_mixing_ratio_ice,
    saturation_mixing_ratio_liquid,
)


config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class ThompsonColumnState:
    """Pytree for a batch of independent Thompson columns on mass levels."""

    __slots__ = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "T", "p", "rho")

    def __init__(self, qv, qc, qr, qi, qs, qg, Ni, Nr, T, p, rho) -> None:
        self.qv = qv
        self.qc = qc
        self.qr = qr
        self.qi = qi
        self.qs = qs
        self.qg = qg
        self.Ni = Ni
        self.Nr = Nr
        self.T = T
        self.p = p
        self.rho = rho

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
        """Rebuilds the column state after JAX transformations."""

        del aux
        return cls(*children)

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


def _snow_moment_proxy(qs, rho, tempc):
    """Provides the snow moment inputs used by WRF deposition/melting formulas."""

    rs = jnp.maximum(qs * rho, R1)
    xds = jnp.maximum(D0S, (rs / 0.069) ** 0.5)
    smo0 = rs / jnp.maximum(0.069 * xds * xds, R1)
    smo1 = smo0 * xds
    smof = smo0 * jnp.sqrt(xds)
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


def _saturation_adjustment(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
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
    qv = state.qv - clap
    T = state.T + lvap * ocp * clap
    return state.replace(
        qv=qv,
        qc=state.qc + clap,
        T=T,
        rho=density_from_pressure_temperature(state.p, T, qv),
    )


def _warm_rain_collection(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
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

    # WRF rain-collecting-cloud-water shape, lines 2260-2268. The table
    # value t_Efrw is represented as a bounded efficiency proxy because the
    # generated lookup table is not a prognostic M5-S1 input.
    ef_rw = jnp.clip(0.55 + 0.45 * (mvd_r - D0R) / jnp.maximum(2.5e-3 - D0R, R1), 0.0, 1.0)
    prr_rcw_raw = rhof * T1_QR_QC * ef_rw * rc * n0_r * ((lamr + FV_R) ** (-CRE9))
    prr_rcw = jnp.where(active_rain & (mvd_r > D0R) & (mvd_c > D0C), prr_rcw_raw, 0.0)
    prr_rcw = jnp.minimum(jnp.maximum(rc - prr_wau * float(dt), 0.0) / float(dt), prr_rcw)

    autoconv = prr_wau * float(dt) / state.rho
    accretion = prr_rcw * float(dt) / state.rho
    transfer = jnp.minimum(state.qc, autoconv + accretion)
    nr_gain = pnr_wau * float(dt) / state.rho
    return state.replace(qc=state.qc - transfer, qr=state.qr + transfer, Nr=state.Nr + nr_gain)


def _rain_evaporation(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
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
    rate_max = jnp.minimum(state.qr / float(dt), jnp.maximum(qvs - state.qv, 0.0) / float(dt))
    evap = jnp.where((ssatw < -EPS) & active_rain, jnp.minimum(rate_max, evap_rate) * float(dt), 0.0)
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


def _ice_sources(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Stages WRF-mapped ice tendencies before condensation and rain evaporation."""

    ocp = cp_inverse(state.qv)
    lvap = latent_heat_vaporization(state.T)
    lfus2 = LSUB - lvap

    # Rain freezing fallback branch from WRF lines 2658-2669 (lookup-table branch omitted).
    rain_freeze = jnp.where(state.T < HGFR, state.qr, 0.0)
    state = state.replace(
        qr=state.qr - rain_freeze,
        qi=state.qi + rain_freeze,
        Ni=state.Ni + jnp.where(rain_freeze > 0.0, state.Nr, 0.0),
        Nr=jnp.where(rain_freeze > 0.0, 0.0, state.Nr),
        T=state.T + lfus2 * ocp * rain_freeze,
    )
    state = state.replace(rho=density_from_pressure_temperature(state.p, state.T, state.qv))

    tempc, diffu, _visco, tcond, _lvap, ocp, rhof, rhof2, vsc2 = _air_properties(state)
    del rhof
    qvs0 = saturation_mixing_ratio_liquid(state.p, T_0)
    del_qvs = jnp.maximum(0.0, qvs0 - state.qv)
    twet = jnp.minimum(state.T, T_0)
    rs, _xds, smo0, smo1, smof, _c_snow, active_snow = _snow_moment_proxy(state.qs, state.rho, tempc)
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
    ri, ni, _lami, ilami, _xdi, xmi, active_ice = _ice_distribution(state.qi, state.Ni, state.rho)
    rs, _xds, _smo0, smo1, smof, c_snow, active_snow = _snow_moment_proxy(state.qs, state.rho, state.T - 273.15)
    rg, ng, _lamg, ilamg, n0_g, active_graupel = _graupel_distribution(state.qg, state.rho)

    # Cloud-ice and aggregate deposition/sublimation, WRF lines 2709-2770.
    pri_ide = C_CUBE * t1_subl * diffu * ssati * rvs * OIG1 * 1.0 * ni * ilami
    pri_ide = jnp.where(active_ice, pri_ide, 0.0)
    pri_ide = jnp.where(pri_ide < 0.0, jnp.maximum(-ri / float(dt), pri_ide), jnp.minimum(pri_ide, jnp.maximum(state.qv - qvsi, 0.0) * state.rho / float(dt) * 0.999))
    prs_sde = c_snow * t1_subl * diffu * ssati * rvs * (T1_SUBL_QS * smo1 + T2_SUBL_QS * rhof2 * vsc2 * smof)
    prs_sde = jnp.where(active_snow, jnp.where(prs_sde < 0.0, jnp.maximum(-rs / float(dt), prs_sde), jnp.minimum(prs_sde, jnp.maximum(state.qv - qvsi, 0.0) * state.rho / float(dt) * 0.999)), 0.0)
    prg_gde = C_CUBE * t1_subl * diffu * ssati * rvs * n0_g * (T1_SUBL_QG * ilamg**CRE10 + T2_SUBL_QG * vsc2 * rhof2 * ilamg**CGE11)
    prg_gde = jnp.where(active_graupel, jnp.where(prg_gde < 0.0, jnp.maximum(-rg / float(dt), prg_gde), jnp.minimum(prg_gde, jnp.maximum(state.qv - qvsi, 0.0) * state.rho / float(dt) * 0.999)), 0.0)

    ice_deposition = pri_ide * float(dt) / state.rho
    snow_deposition = prs_sde * float(dt) / state.rho
    graupel_deposition = prg_gde * float(dt) / state.rho
    vapor_sink = jnp.maximum(0.0, ice_deposition) + jnp.maximum(0.0, snow_deposition) + jnp.maximum(0.0, graupel_deposition)
    vapor_source = jnp.maximum(0.0, -ice_deposition) + jnp.maximum(0.0, -snow_deposition) + jnp.maximum(0.0, -graupel_deposition)
    return state.replace(
        qv=state.qv - vapor_sink + vapor_source,
        qi=state.qi + ice_deposition,
        qs=state.qs + snow_deposition,
        qg=state.qg + graupel_deposition,
        # WRF lines 2719-2727 update pni_ide only in sublimation; positive
        # deposition partitions mass but does not create new cloud-ice number.
        Ni=jnp.maximum(0.0, state.Ni + jnp.where(ice_deposition < 0.0, ice_deposition / jnp.maximum(xmi, XM0I), 0.0)),
        T=state.T + LSUB * ocp * (vapor_sink - vapor_source),
        rho=density_from_pressure_temperature(state.p, state.T + LSUB * ocp * (vapor_sink - vapor_source), state.qv - vapor_sink + vapor_source),
    )


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


def _step_thompson_column_impl(state: ThompsonColumnState, dt: float, debug: bool) -> ThompsonColumnState:
    """Runs the fused source/sink Thompson body in WRF checkpoint order."""

    state = _clip_species(state)
    state = _debug_checks(state, debug)
    # WRF order: stage rates/tendencies (2917-3247), update working state
    # before condensation (3250-3273), cloud cond/evap (3456-3558),
    # rain evaporation (3561-3638), instant melt/freeze (4005-4031),
    # final write/balance (4033-4142). This source/sink subset uses the
    # same checkpoints while keeping sedimentation out of scope.
    state = _warm_rain_collection(state, dt)
    state = _ice_sources(state, dt)
    state = _saturation_adjustment(state, dt)
    state = _rain_evaporation(state, dt)
    state = _instant_melt_freeze(state, dt)
    state = _finish(state)
    return _debug_checks(state, debug)


@partial(jax.jit, static_argnames=("dt", "debug"))
def step_thompson_column(state: ThompsonColumnState, dt: float, *, debug: bool = False) -> ThompsonColumnState:
    """Advances one Thompson source/sink column step under one JAX program."""

    return _step_thompson_column_impl(state, dt, debug)
