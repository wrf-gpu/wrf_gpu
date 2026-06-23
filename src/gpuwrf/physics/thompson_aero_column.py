"""JAX aerosol-aware Thompson column microphysics (WRF mp_physics=28).

Extends the mp=8 Thompson column port (``thompson_column.py``) with the
``is_aerosol_aware`` path of WRF ``module_mp_thompson.F`` (Thompson &
Eidhammer 2014): prognostic cloud-droplet number ``Nc`` plus the two
prognostic aerosol number concentrations ``nwfa`` (water-friendly CCN) and
``nifa`` (ice-friendly IN), with

* CCN activation from the Eidhammer parcel-model lookup table
  (``activ_ncloud``; WRF lines 3413-3420 / 5178-5253),
* droplet evaporation releasing CCN with the explicit ``tnc_wev`` number
  table (WRF 3422-3475),
* variable cloud gamma shape ``nu_c = MIN(15, NINT(1000e6/nc)+2)``
  throughout the warm-rain and sedimentation cloud terms,
* DeMott (2010) dust ice nucleation replacing Cooper (WRF 2573-2631,
  ``iceDeMott``), with the rain/cloud freezing tables indexed by the live
  DeMott IN count (``idx_IN``; WRF 2579-2616),
* Koop et al. (2001) homogeneous freezing of deliquesced aerosols
  (``iceKoop``; WRF 2633-2641),
* wet scavenging of nwfa/nifa by rain, snow and graupel (``Eff_aero``;
  WRF 2210-2232, 2442-2481),
* rain evaporation returning rain-number worth of CCN (WRF 3565),
* the WRF working-unit clamps, floors and final per-kg writeback
  (WRF 1798-1825, 3211, 3976-4021), and
* the fake surface emission applied after the column step
  (``mp_gt_driver``; WRF 1317-1326).

Process subset: this builds on the same WRF process subset the validated
mp=8 kernel carries (warm rain, rain freezing, melt, deposition/sublimation,
nucleation, sedimentation, saturation adjustment).  Snow/graupel riming of
cloud water (prs_scw/prg_gcw) remains outside the subset, so its cloud-number
legs (pnc_scw/pnc_gcw) are likewise not applied — the aerosol SCAVENGING legs
of snow/graupel (pna_sca/pna_gca/pnd_scd/pnd_gcd), which are independent
aerosol sinks, ARE applied.  The BC aerosol (nbca, wif_input_opt=2) path is
not implemented; mp=28 runs with wif_input_opt=1 (nbca inert), matching the
oracle configuration.

All math fp64; one column step == one WRF ``mp_gt_driver`` call on a column.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import math
from functools import partial
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_column import (
    _air_properties,
    _clamp_rain_number,
    _fall_speeds,
    _graupel_distribution,
    _lookup_digit_index,
    _nstep_per_column,
    _rain_distribution,
    _rho_correction,
    _sed_one_species,
    _snow_moment,
    _snow_moments,
    _sublimation_prefactor,
    _take2,
    _take3_last,
    density_from_pressure_temperature,
)
from gpuwrf.physics.thompson_constants import (
    AM_I,
    AM_R,
    AV_C,
    AV_S,
    BV_C,
    BV_S,
    CGE11,
    CIE2,
    CRE1,
    CRE9,
    CRE10,
    CRE11,
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
    OBMI,
    OBMR,
    OIG1,
    OIG2,
    ORG1,
    ORG2,
    ORG3,
    PI,
    R1,
    R2,
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
    N_I1_TABLE,
    N_I_TABLE,
    N_R1_TABLE,
    N_R_TABLE,
    N_TC_TABLE,
    R_I_FIRST,
    R_R_FIRST,
    NT_I_FIRST,
    THOMPSON_TABLES,
    ThompsonTableBundle,
)
from gpuwrf.physics.thompson_aero_tables import (
    N_C_TABLE,
    N_IN_TABLE,
    NBC,
    THOMPSON_AERO_TABLES,
    ThompsonAeroTableBundle,
)

configure_jax_x64()


# --- WRF module_mp_thompson.F aerosol constants -------------------------------
NT_C_MAX = 1999.0e6          # max droplet number (line 89)
NA_CCN0 = 300.0e6            # climatological CCN profile scale (line 96)
NA_CCN1 = 50.0e6             # climatological CCN background (line 97)
NA_IN0 = 1.5e6               # climatological IN profile scale (line 94)
NA_IN1 = 0.5e6               # climatological IN background (line 95)
NWFA_FLOOR = 11.1e6          # working/writeback floor (lines 1805, 3979)
NIFA_FLOOR = NA_IN1 * 0.01   # = 5e3 (lines 1806, 3981)
AERO_CAP = 9999.0e6          # working/writeback cap (lines 1805-1806)
R_UNI = 8.314                # universal gas constant (line 206 region)
AR_VOLUME = 4.0 / 3.0 * PI * (2.5e-6) ** 3  # deliquesced-aerosol volume (Koop)
RHO_NOT0 = 101325.0 / (287.05 * 273.15)     # DeMott reference density
R_C_FIRST = 1.0e-6           # r_c(1) kg m^-3
R_S_FIRST = 1.0e-5           # r_s(1) kg m^-3
R_G_FIRST = 1.0e-5           # r_g(1) kg m^-3
# Aerosol radii sampled by the scavenging efficiency (lines 2212/2218 etc.).
DA_NWFA = 0.04e-6
DA_NIFA = 0.8e-6
# Non-hail-aware graupel fall-speed law (thompson_init lines 463-465: for
# mp=8/28 WRF overwrites av_g(idx_bg1)/bv_g(idx_bg1) with the legacy values).
AV_G_OLD = 442.0
BV_G_OLD = 0.89
MU_G = 0.0
CGE9_OLD = MU_G + BV_G_OLD + 3.0
CGG9_OLD = math.gamma(CGE9_OLD)
T1_QG_QC_OLD = PI * 0.25 * AV_G_OLD * CGG9_OLD
T1_QS_QC = PI * 0.25 * AV_S
# Eff_aero constants (lines 4971-4972).
BOLTZMAN = 1.3806503e-23
MEAN_PATH = 0.0256e-6


@jax.tree_util.register_pytree_node_class
class ThompsonAeroColumnState:
    """Batch of aerosol-aware Thompson columns (per-kg units, vertical last).

    Mirrors :class:`thompson_column.ThompsonColumnState` plus the three
    aerosol-aware prognostics: ``Nc`` (cloud droplet number, kg^-1),
    ``nwfa``/``nifa`` (water-/ice-friendly aerosol number, kg^-1).
    """

    __slots__ = (
        "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng",
        "Nc", "nwfa", "nifa", "T", "p", "rho", "dz", "w",
    )

    def __init__(self, qv, qc, qr, qi, qs, qg, Ni, Nr, Nc, nwfa, nifa, T, p, rho,
                 Ns=None, Ng=None, dz=None, w=None) -> None:
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
        self.Nc = Nc
        self.nwfa = nwfa
        self.nifa = nifa
        self.T = T
        self.p = p
        self.rho = rho
        self.dz = dz if dz is not None else jnp.full_like(qv, 250.0)
        self.w = w if w is not None else jnp.zeros_like(qv)

    def replace(self, **updates) -> "ThompsonAeroColumnState":
        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(**dict(zip(cls.__slots__, children, strict=True)))


def _leaves(state: ThompsonAeroColumnState) -> Iterable[jax.Array]:
    return (getattr(state, name) for name in ThompsonAeroColumnState.__slots__)


# ------------------------------------------------------------------------------
# Working-unit helpers
# ------------------------------------------------------------------------------


def _clip_species_aero(state: ThompsonAeroColumnState) -> ThompsonAeroColumnState:
    """Non-negative preamble incl. the aerosol/cloud-number prognostics."""

    return state.replace(
        qv=jnp.maximum(state.qv, 1.0e-10),
        qc=jnp.maximum(state.qc, 0.0),
        qr=jnp.maximum(state.qr, 0.0),
        qi=jnp.maximum(state.qi, 0.0),
        qs=jnp.maximum(state.qs, 0.0),
        qg=jnp.maximum(state.qg, 0.0),
        Ni=jnp.maximum(state.Ni, 0.0),
        Nr=jnp.maximum(state.Nr, 0.0),
        Nc=jnp.maximum(state.Nc, 0.0),
        nwfa=jnp.maximum(state.nwfa, 0.0),
        nifa=jnp.maximum(state.nifa, 0.0),
    )


def _nwfa_working(nwfa, rho):
    """Working per-m3 CCN count (WRF line 1805, aer_init_opt < 2)."""

    return jnp.maximum(NWFA_FLOOR, jnp.minimum(AERO_CAP, nwfa * rho))


def _nifa_working(nifa, rho):
    """Working per-m3 IN count (WRF line 1806, aer_init_opt < 2)."""

    return jnp.maximum(NIFA_FLOOR, jnp.minimum(AERO_CAP, nifa * rho))


def _nu_c_from_nc(nc_m3):
    """WRF ``nu_c = MIN(15, NINT(1000.E6/nc)+2)`` as a 0-based table index."""

    nint = jnp.floor(1000.0e6 / jnp.maximum(nc_m3, 2.0) + 0.5)
    return jnp.clip(nint + 2.0, 3.0, 15.0).astype(jnp.int32) - 1


def _entry_cloud_number(qc, Nc, rho, aero: ThompsonAeroTableBundle):
    """Entry-clamp the working cloud number (WRF lines 1827-1849).

    Where ``qc > R1`` the working nc (per-m3) is clamped to [2, Nt_c_max] then
    rebalanced so the mean-mass diameter stays in [D0c, 2*D0r]; elsewhere the
    working nc is 2 m^-3.  Returns the working per-m3 ``nc``.
    """

    rc = jnp.maximum(qc * rho, R1)
    nc = jnp.maximum(2.0, jnp.minimum(Nc * rho, NT_C_MAX))
    nu = _nu_c_from_nc(nc)
    ccg2 = jnp.take(aero.ccg[1], nu)
    ocg1 = jnp.take(aero.ocg1, nu)
    cce2 = jnp.take(aero.cce[1], nu)
    ccg1 = jnp.take(aero.ccg[0], nu)
    ocg2 = jnp.take(aero.ocg2, nu)
    lamc = (nc * AM_R * ccg2 * ocg1 / rc) ** OBMR
    xdc = (3.0 + nu.astype(jnp.float64) + 1.0 + 1.0) / lamc  # bm_r + nu_c + 1
    lamc = jnp.where(xdc < D0C, cce2 / D0C, lamc)
    lamc = jnp.where(xdc > D0R * 2.0, cce2 / (D0R * 2.0), lamc)
    nc_balanced = jnp.minimum(NT_C_MAX, ccg1 * ocg2 * rc / AM_R * lamc**3.0)
    return jnp.where(qc > R1, nc_balanced, 2.0)


def _cloud_distribution_aero(qc, nc_m3, rho, aero: ThompsonAeroTableBundle):
    """Variable-nu_c cloud gamma terms (WRF lines 2169-2176).

    ``nc_m3`` is the WORKING per-m3 droplet number.  Returns
    (rc, nu_idx, lamc, xdc, mvd_c, active).
    """

    rc = jnp.maximum(qc * rho, R1)
    nu = _nu_c_from_nc(nc_m3)
    nuf = nu.astype(jnp.float64) + 1.0  # nu_c value (1-based)
    ccg2 = jnp.take(aero.ccg[1], nu)
    ocg1 = jnp.take(aero.ocg1, nu)
    lamc = (nc_m3 * AM_R * ccg2 * ocg1 / rc) ** OBMR
    xdc = jnp.maximum(D0C * 1.0e6, ((rc / (AM_R * nc_m3)) ** OBMR) * 1.0e6)
    mvd_c = (3.0 + nuf + 0.672) / lamc
    mvd_c = jnp.maximum(D0C, jnp.minimum(mvd_c, D0R))
    return rc, nu, lamc, xdc, mvd_c, qc > R1


def _cloud_number_balance(qc, Nc, rho, aero: ThompsonAeroTableBundle):
    """Mass/number balance keeping xDc in [D0c, 2*D0r] (WRF 2997-3019, 4007-4021).

    Operates on per-kg (qc, Nc); returns the balanced per-kg Nc (0 where no
    cloud), capped at Nt_c_max/rho.
    """

    rc = jnp.maximum(qc * rho, R1)
    nc = jnp.maximum(2.0, Nc * rho)
    nu = _nu_c_from_nc(nc)
    ccg2 = jnp.take(aero.ccg[1], nu)
    ocg1 = jnp.take(aero.ocg1, nu)
    cce2 = jnp.take(aero.cce[1], nu)
    ccg1 = jnp.take(aero.ccg[0], nu)
    ocg2 = jnp.take(aero.ocg2, nu)
    lamc = (nc * AM_R * ccg2 * ocg1 / rc) ** OBMR
    xdc = (3.0 + nu.astype(jnp.float64) + 1.0 + 1.0) / lamc
    out_low = xdc < D0C
    out_high = xdc > D0R * 2.0
    lamc_fix = jnp.where(out_low, cce2 / D0C, cce2 / (D0R * 2.0))
    nc_fix = ccg1 * ocg2 * rc / AM_R * lamc_fix**3.0
    nc_new = jnp.where(out_low | out_high, nc_fix, nc)
    nc_new = jnp.minimum(nc_new, NT_C_MAX)
    return jnp.where(qc > R1, nc_new / rho, 0.0)


# ------------------------------------------------------------------------------
# Aerosol physics closed forms
# ------------------------------------------------------------------------------


def _ice_demott(tempc, rho, nifa_m3):
    """DeMott et al. (2010) IN count, per m3 (WRF ``iceDeMott``, 5447-5514)."""

    nifa_cc = jnp.maximum(0.5, nifa_m3 * RHO_NOT0 * 1.0e-6 / rho)
    neg_tc = jnp.maximum(-tempc, 1.0e-12)
    xni = (5.94e-5 * neg_tc**3.33) * (nifa_cc ** ((-0.0264 * tempc) + 0.0033))
    xni = xni * rho / RHO_NOT0 * 1000.0
    return jnp.maximum(0.0, xni)


def _ice_koop(temp, qv, qvs, naero_m3, dt):
    """Koop et al. (2001) homogeneous freezing count, per m3 (WRF 5521-5546)."""

    satw = qv / qvs
    mu_diff = 210368.0 + 131.438 * temp - 3.32373e6 / temp - 41729.1 * jnp.log(temp)
    a_w_i = jnp.exp(mu_diff / (R_UNI * temp))
    delta_aw = satw - a_w_i
    log_j = -906.7 + 8502.0 * delta_aw - 26924.0 * delta_aw**2 + 29180.0 * delta_aw**3
    log_j = jnp.minimum(20.0, log_j)
    j_rate = 10.0**log_j
    prob_h = jnp.minimum(1.0 - jnp.exp(-j_rate * AR_VOLUME * dt), 1.0)
    xni = jnp.where(prob_h > 0.0, jnp.minimum(prob_h * naero_m3, 1000.0e3), 0.0)
    return jnp.maximum(0.0, xni)


def _eff_aero(d, da, visco, rho, temp, species: str):
    """Aerosol collection efficiency (WRF ``Eff_aero``, 4965-5001)."""

    if species == "r":
        vt = -0.1021 + 4.932e3 * d - 0.9551e6 * d * d + 0.07934e9 * d**3 - 0.002362e12 * d**4
    elif species == "s":
        vt = AV_S * d**BV_S
    elif species == "g":
        vt = AV_G_OLD * d**BV_G_OLD
    else:  # pragma: no cover - transcription guard
        raise ValueError(species)
    cc = 1.0 + 2.0 * MEAN_PATH / da * (1.257 + 0.4 * math.exp(-0.55 * da / MEAN_PATH))
    diff = BOLTZMAN * temp * cc / (3.0 * PI * visco * da)
    re = 0.5 * rho * d * vt / visco
    sc = visco / (rho * diff)
    st = da * da * vt * 1000.0 / (9.0 * visco * d)
    aval = 1.0 + jnp.log(1.0 + re)
    st2 = (1.2 + 1.0 / 12.0 * aval) / (1.0 + aval)
    eff = 4.0 / (re * sc) * (1.0 + 0.4 * jnp.sqrt(re) * sc ** (1.0 / 3.0) + 0.16 * jnp.sqrt(re) * jnp.sqrt(sc)) \
        + 4.0 * da / d * (0.02 + da / d * (1.0 + 2.0 * jnp.sqrt(re)))
    eff = jnp.where(st > st2, eff + ((st - st2) / (st - st2 + 0.666667)) ** 1.5, eff)
    return jnp.maximum(1.0e-5, jnp.minimum(eff, 1.0))


def _activ_ncloud(temp, w, nccn_m3, aero: ThompsonAeroTableBundle):
    """Activated droplet count from the CCN table (WRF ``activ_ncloud``)."""

    ta_na = aero.ta_na
    ta_ww = aero.ta_ww
    n_local = nccn_m3 * 1.0e-6
    n_local = jnp.where(n_local >= ta_na[-1], ta_na[-1] - 1.0, n_local)
    n_local = jnp.where(n_local <= ta_na[0], ta_na[0] + 1.0, n_local)
    w_local = jnp.where(w >= ta_ww[-1], ta_ww[-1] - 1.0, w)
    w_local = jnp.where(w_local <= ta_ww[0], ta_ww[0] + 0.001, w_local)

    i_hi = jnp.clip(jnp.searchsorted(ta_na, n_local, side="right"), 1, ta_na.shape[0] - 1)
    j_hi = jnp.clip(jnp.searchsorted(ta_ww, w_local, side="right"), 1, ta_ww.shape[0] - 1)
    x1 = jnp.log(jnp.take(ta_na, i_hi - 1))
    x2 = jnp.log(jnp.take(ta_na, i_hi))
    y1 = jnp.log(jnp.take(ta_ww, j_hi - 1))
    y2 = jnp.log(jnp.take(ta_ww, j_hi))
    k_t = jnp.clip(jnp.floor((temp - 243.15) * 0.1 + 0.5).astype(jnp.int32), 0, aero.ccn_act.shape[2] - 1)

    table = aero.ccn_act  # (na, ww, tk)
    n_ww = table.shape[1]
    n_tk = table.shape[2]
    flat = jnp.ravel(table)

    def take3(ii, jj):
        return jnp.take(flat, (ii * n_ww + jj) * n_tk + k_t)

    a = take3(i_hi - 1, j_hi - 1)
    b = take3(i_hi, j_hi - 1)
    c = take3(i_hi, j_hi)
    d = take3(i_hi - 1, j_hi)
    t = (jnp.log(n_local) - x1) / (x2 - x1)
    u = (jnp.log(w_local) - y1) / (y2 - y1)
    fraction = (1.0 - t) * (1.0 - u) * a + t * (1.0 - u) * b + t * u * c + (1.0 - t) * u * d
    return nccn_m3 * fraction


def _idx_in_from_xni(xni):
    """IN-count table index from the live DeMott count (WRF 2579-2592)."""

    idx = _lookup_digit_index(jnp.maximum(xni, 1.0), 0, N_IN_TABLE)
    return jnp.where(xni > 1.0, idx, 0)


def _idx_c_from_rc(rc):
    """Cloud-mass table index (WRF 3450-3463), 0-based."""

    return jnp.where(rc > R_C_FIRST, _lookup_digit_index(rc, -6, N_C_TABLE), 0)


def _idx_n_from_nc(nc_m3, aero: ThompsonAeroTableBundle):
    """Droplet-number bin index (WRF 3447-3448: NINT(1+nbc*log(nc/t_Nc1)/nic1))."""

    val = 1.0 + NBC * jnp.log(jnp.maximum(nc_m3, 1.0) / aero.t_nc1) / aero.nic1
    return jnp.clip(jnp.floor(val + 0.5).astype(jnp.int32), 1, NBC) - 1


def _take_qcfz(aero: ThompsonAeroTableBundle, idx_c, idx_n, idx_tc, idx_in):
    """Gather (tpi_qcfz, tni_qcfz) at one flattened dynamic index."""

    combined = ((idx_c.astype(jnp.int32) * NBC + idx_n.astype(jnp.int32)) * N_TC_TABLE
                + idx_tc.astype(jnp.int32)) * N_IN_TABLE + idx_in.astype(jnp.int32)
    base = combined * 2
    offsets = jnp.arange(2, dtype=jnp.int32)
    return jnp.take(jnp.ravel(aero.qcfz), base[..., None] + offsets)


def _take_qrfz4(aero: ThompsonAeroTableBundle, idx_r, idx_r1, idx_tc, idx_in):
    """Gather the 4 rain-freezing tables at the live IN index."""

    combined = ((idx_r.astype(jnp.int32) * N_R1_TABLE + idx_r1.astype(jnp.int32)) * N_TC_TABLE
                + idx_tc.astype(jnp.int32)) * N_IN_TABLE + idx_in.astype(jnp.int32)
    base = combined * 4
    offsets = jnp.arange(4, dtype=jnp.int32)
    return jnp.take(jnp.ravel(aero.qrfz4), base[..., None] + offsets)


# ------------------------------------------------------------------------------
# Process stages (sequential structure mirroring the validated mp=8 kernel)
# ------------------------------------------------------------------------------


def _warm_rain_collection_aero(
    state: ThompsonAeroColumnState,
    dt: float,
    tables: ThompsonTableBundle = THOMPSON_TABLES,
    aero: ThompsonAeroTableBundle = THOMPSON_AERO_TABLES,
) -> ThompsonAeroColumnState:
    """Warm-rain processes + rain wet scavenging (WRF 2157-2234).

    Berry-Reinhardt autoconversion and rain-collecting-cloud with the
    VARIABLE nu_c cloud gamma (prr_wau/pnr_wau/pnc_wau, prr_rcw/pnc_rcw),
    rain self-collection/breakup (pnr_rcr), and rain collecting aerosols
    (pna_rca on nwfa, pnd_rcd on nifa).
    """

    odts = 1.0 / float(dt)
    _tempc, _diffu, visco, _tcond, _lvap, _ocp, rhof, _rhof2, _vsc2 = _air_properties(state)
    nc_m3 = _entry_cloud_number(state.qc, state.Nc, state.rho, aero)
    rc, nu, lamc, xdc, mvd_c, active_cloud = _cloud_distribution_aero(state.qc, nc_m3, state.rho, aero)
    rr, nr, lamr, ilamr, mvd_r, n0_r, active_rain = _rain_distribution(state.qr, state.Nr, state.rho)
    nwfa_m3 = _nwfa_working(state.nwfa, state.rho)
    nifa_m3 = _nifa_working(state.nifa, state.rho)
    nuf = nu.astype(jnp.float64) + 1.0

    # Berry-Reinhardt autoconversion (WRF 2178-2193), variable nu_c.
    ccg3 = jnp.take(aero.ccg[2], nu)
    ocg2 = jnp.take(aero.ocg2, nu)
    dc_g = ((ccg3 * ocg2) ** OBMR / lamc) * 1.0e6
    dc_b = jnp.maximum(xdc**3 * dc_g**3 - xdc**6, 0.0) ** (1.0 / 6.0)
    zeta1_raw = 6.25e-6 * xdc * dc_b**3 - 0.4
    zeta1 = 0.5 * (zeta1_raw + jnp.abs(zeta1_raw))
    zeta = 0.027 * rc * zeta1
    taud_raw = 0.5 * dc_b - 7.5
    taud = 0.5 * (taud_raw + jnp.abs(taud_raw)) + R1
    tau = 3.72 / jnp.maximum(rc * taud, R1)
    prr_wau = jnp.where((rc > 0.01e-3) & active_cloud, jnp.minimum(rc * odts, zeta / tau), 0.0)
    pnr_wau = prr_wau / (AM_R * nuf * 10.0 * D0R**3)
    pnc_wau = jnp.minimum(nc_m3 * odts, prr_wau / (AM_R * mvd_c**3))

    # Rain collecting cloud water (WRF 2196-2208).
    idx_r_eff = jnp.clip(
        jnp.floor(N_EFRW_R * jnp.log(jnp.maximum(mvd_r, DR_FIRST) / DR_FIRST) / jnp.log(DR_LAST / DR_FIRST)),
        0,
        N_EFRW_R - 1,
    ).astype(jnp.int32)
    idx_c_eff = jnp.clip(jnp.floor(mvd_c * 1.0e6).astype(jnp.int32) - 1, 0, N_EFRW_C - 1)
    ef_rw = _take2(tables.t_Efrw, idx_r_eff, idx_c_eff)
    rcw_geom = rhof * T1_QR_QC * n0_r * ((lamr + FV_R) ** (-CRE9))
    rain_collects = active_rain & (mvd_r > D0R) & (mvd_c > D0C)
    prr_rcw = jnp.where(rain_collects, ef_rw * rcw_geom * rc, 0.0)
    prr_rcw = jnp.minimum(jnp.maximum(rc - prr_wau * float(dt), 0.0) * odts, prr_rcw)
    pnc_rcw = jnp.where(rain_collects, jnp.minimum(nc_m3 * odts, ef_rw * rcw_geom * nc_m3), 0.0)

    # Rain self-collection / break-up (WRF 2161-2167).
    ef_rr = 1.0 - jnp.exp(2300.0 * (mvd_r - 1950.0e-6))
    pnr_rcr = jnp.where(active_rain & (mvd_r > D0R), ef_rr * 2.0 * nr * rr, 0.0)

    # Rain collecting aerosols (WRF 2210-2232): the SAME collection geometry
    # with the aerosol-size efficiency; capped at the working count per step.
    scavenges = active_rain & (mvd_r > D0R)
    ef_ra_w = _eff_aero(mvd_r, DA_NWFA, visco, state.rho, state.T, "r")
    ef_ra_i = _eff_aero(mvd_r, DA_NIFA, visco, state.rho, state.T, "r")
    pna_rca = jnp.where(scavenges, jnp.minimum(nwfa_m3 * odts, ef_ra_w * rcw_geom * nwfa_m3), 0.0)
    pnd_rcd = jnp.where(scavenges, jnp.minimum(nifa_m3 * odts, ef_ra_i * rcw_geom * nifa_m3), 0.0)

    # Apply (per-kg).  Mass transfer cloud->rain capped at the available qc
    # (the WRF cloud-water conservation limiter, 2877-2890, reduced to the
    # subset's two cloud sinks).
    autoconv = prr_wau * float(dt) / state.rho
    accretion = prr_rcw * float(dt) / state.rho
    transfer = jnp.minimum(state.qc, autoconv + accretion)
    nc_loss = jnp.minimum(state.Nc, (pnc_wau + pnc_rcw) * float(dt) / state.rho)
    nr_gain = pnr_wau * float(dt) / state.rho
    nr_rcr = pnr_rcr * float(dt) / state.rho
    return state.replace(
        qc=state.qc - transfer,
        qr=state.qr + transfer,
        Nc=jnp.maximum(0.0, state.Nc - nc_loss),
        Nr=jnp.maximum(0.0, state.Nr + nr_gain - nr_rcr),
        nwfa=jnp.maximum(0.0, state.nwfa - pna_rca * float(dt) / state.rho),
        nifa=jnp.maximum(0.0, state.nifa - pnd_rcd * float(dt) / state.rho),
    )


def _snow_graupel_scavenging(
    state: ThompsonAeroColumnState,
    dt: float,
    tables: ThompsonTableBundle = THOMPSON_TABLES,
) -> ThompsonAeroColumnState:
    """Snow/graupel collecting aerosols (WRF 2442-2481).

    Independent aerosol sinks (pna_sca/pnd_scd via the snow ``smoe`` moment;
    pna_gca/pnd_gcd via the graupel distribution).  Active below/above 0C
    alike (WRF computes them inside the frozen-species block whenever rs/rg
    exceed the first table mass).
    """

    odts = 1.0 / float(dt)
    tempc = state.T - 273.15
    _t, _diffu, visco, _tcond, _lvap, _ocp, rhof, _rhof2, _vsc2 = _air_properties(state)
    nwfa_m3 = _nwfa_working(state.nwfa, state.rho)
    nifa_m3 = _nifa_working(state.nifa, state.rho)

    rs = jnp.maximum(state.qs * state.rho, R1)
    smo2 = rs / 0.069  # smob = rs/am_s (am_s = 0.069)
    tc0 = jnp.minimum(-0.1, tempc)
    smoe = _snow_moment(tables.cse[12], smo2, tc0, tables)
    smoc = _snow_moment(tables.cse[0], smo2, tc0, tables)
    xds = smoc / jnp.maximum(smo2, R1)
    snow_active = rs > R_S_FIRST
    ef_sa_w = _eff_aero(jnp.maximum(xds, 1.0e-6), DA_NWFA, visco, state.rho, state.T, "s")
    ef_sa_i = _eff_aero(jnp.maximum(xds, 1.0e-6), DA_NIFA, visco, state.rho, state.T, "s")
    pna_sca = jnp.where(snow_active, jnp.minimum(nwfa_m3 * odts, rhof * T1_QS_QC * ef_sa_w * nwfa_m3 * smoe), 0.0)
    pnd_scd = jnp.where(snow_active, jnp.minimum(nifa_m3 * odts, rhof * T1_QS_QC * ef_sa_i * nifa_m3 * smoe), 0.0)

    rg, _ng, _lamg, ilamg, n0_g, _act_g = _graupel_distribution(state.qg, state.Ng, state.rho)
    graupel_active = rg > R_G_FIRST
    xdg = (3.0 + MU_G + 1.0) * ilamg
    geom_g = rhof * T1_QG_QC_OLD * n0_g * ilamg**CGE9_OLD
    ef_ga_w = _eff_aero(jnp.maximum(xdg, 1.0e-6), DA_NWFA, visco, state.rho, state.T, "g")
    ef_ga_i = _eff_aero(jnp.maximum(xdg, 1.0e-6), DA_NIFA, visco, state.rho, state.T, "g")
    pna_gca = jnp.where(graupel_active, jnp.minimum(nwfa_m3 * odts, ef_ga_w * geom_g * nwfa_m3), 0.0)
    pnd_gcd = jnp.where(graupel_active, jnp.minimum(nifa_m3 * odts, ef_ga_i * geom_g * nifa_m3), 0.0)

    return state.replace(
        nwfa=jnp.maximum(0.0, state.nwfa - (pna_sca + pna_gca) * float(dt) / state.rho),
        nifa=jnp.maximum(0.0, state.nifa - (pnd_scd + pnd_gcd) * float(dt) / state.rho),
    )


def _ice_sources_aero(
    state: ThompsonAeroColumnState,
    dt: float,
    tables: ThompsonTableBundle = THOMPSON_TABLES,
    aero: ThompsonAeroTableBundle = THOMPSON_AERO_TABLES,
):
    """Frozen-species sources with the aerosol-aware nucleation path.

    Adapted from the validated mp=8 ``_ice_sources_with_process_flags`` with:
      * rain freezing tables indexed by the LIVE DeMott IN count
        (idx_IN; WRF 2573-2604),
      * cloud-water heterogeneous freezing pri_wfz/pni_wfz via the qcfz
        tables (WRF 2607-2616) — a cloud mass+number sink the fixed-Nc mp=8
        subset could not carry,
      * deposition nucleation count from DeMott instead of Cooper
        (WRF 2618-2631), with the nifa sink (dustyIce; WRF 2974-2975),
      * Koop homogeneous freezing of deliquesced aerosols pri_iha/pni_iha
        (WRF 2633-2641) with the nwfa sink (WRF 2963-2965).
    Returns ``(state, graupel_melt)`` like the mp=8 stage.
    """

    odts = 1.0 / float(dt)
    ocp = cp_inverse(state.qv)
    lvap = latent_heat_vaporization(state.T)
    lfus2 = LSUB - lvap

    tempc0 = state.T - 273.15
    below_freezing = state.T < T_0
    nifa_m3 = _nifa_working(state.nifa, state.rho)
    nwfa_m3 = _nwfa_working(state.nwfa, state.rho)
    nc_m3 = _entry_cloud_number(state.qc, state.Nc, state.rho, aero)

    # Live DeMott IN count -> freezing-table IN index (WRF 2573-2592).
    xni_demott = _ice_demott(jnp.minimum(tempc0, -1.0e-6), state.rho, nifa_m3)
    idx_in = jnp.where(below_freezing, _idx_in_from_xni(xni_demott), 0)
    idx_tc = jnp.clip(jnp.floor(-tempc0 + 0.5).astype(jnp.int32) - 1, 0, N_TC_TABLE - 1)

    # --- Rain freezing (WRF 2594-2605) at the live IN index. ---
    rr0 = jnp.maximum(state.qr * state.rho, R1)
    nr0 = jnp.maximum(state.Nr * state.rho, R2)
    rr_for_index = jnp.maximum(rr0, R_R_FIRST)
    _, _nr, lamr0, _il, _mv, _n0, _ar = _rain_distribution(state.qr, state.Nr, state.rho)
    n0_exp = ORG1 * rr_for_index / AM_R * lamr0**CRE1
    idx_r = _lookup_digit_index(rr_for_index, -6, N_R_TABLE)
    idx_r1 = _lookup_digit_index(n0_exp, 6, N_R1_TABLE)
    qrfz = _take_qrfz4(aero, idx_r, idx_r1, idx_tc, idx_in)
    table_active = below_freezing & (rr0 > R_R_FIRST) & (state.qr > R1)
    table_ice = jnp.where(table_active, qrfz[..., 0] / state.rho, 0.0)
    table_graupel = jnp.where(table_active, qrfz[..., 1] / state.rho, 0.0)
    table_ni = jnp.where(table_active, qrfz[..., 2] / state.rho, 0.0)
    table_nr = jnp.where(table_active, jnp.minimum(qrfz[..., 3], nr0) / state.rho, 0.0)
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
    pni_rfz_m3 = (table_ni + fallback_ni) * state.rho * odts  # for the inu threshold

    # --- Cloud-water heterogeneous freezing via qcfz (WRF 2607-2616). ---
    rc0 = jnp.maximum(state.qc * state.rho, R1)
    idx_c = _idx_c_from_rc(rc0)
    idx_n = _idx_n_from_nc(nc_m3, aero)
    qcfz = _take_qcfz(aero, idx_c, idx_n, idx_tc, idx_in)
    wfz_table_active = below_freezing & (rc0 > R_C_FIRST)
    pri_wfz = jnp.where(wfz_table_active, jnp.minimum(rc0 * odts, qcfz[..., 0] * odts), 0.0)
    pni_wfz = jnp.where(
        wfz_table_active,
        jnp.minimum(jnp.minimum(nc_m3 * odts, pri_wfz / (2.0 * XM0I)), qcfz[..., 1] * odts),
        0.0,
    )
    # (The <HGFR instant branch for sub-table qc is handled by the final
    # instant-freeze stage, as in the mp=8 kernel.)
    wfz_mass = pri_wfz * float(dt) / state.rho
    wfz_mass = jnp.minimum(wfz_mass, state.qc)
    wfz_num = jnp.minimum(pni_wfz * float(dt) / state.rho, state.Nc)

    nr_loss = jnp.minimum(state.Nr, table_nr + table_ni + fallback_ni)
    state = state.replace(
        qr=state.qr - ice_freeze - graupel_freeze,
        qc=state.qc - wfz_mass,
        qi=state.qi + ice_freeze + wfz_mass,
        qg=state.qg + graupel_freeze,
        Ni=state.Ni + table_ni + fallback_ni + wfz_num,
        Nr=jnp.maximum(0.0, state.Nr - nr_loss),
        Nc=jnp.maximum(0.0, state.Nc - wfz_num),
        T=state.T + lfus2 * ocp * (ice_freeze + graupel_freeze + wfz_mass),
    )
    state = state.replace(rho=density_from_pressure_temperature(state.p, state.T, state.qv))

    # --- Deposition nucleation: DeMott count replaces Cooper (WRF 2618-2631). ---
    qvsi_freeze = saturation_mixing_ratio_ice(state.p, state.T)
    qvsw_freeze = saturation_mixing_ratio_liquid(state.p, state.T)
    ssati_freeze = state.qv / qvsi_freeze - 1.0
    ssatw_freeze = state.qv / qvsw_freeze - 1.0
    deposition_nucleation_active = below_freezing & (
        (ssati_freeze >= 0.25) | ((ssatw_freeze > EPS) & (state.T < 253.15))
    )
    xnc_inu = _ice_demott(jnp.minimum(state.T - 273.15, -1.0e-6), state.rho, nifa_m3)
    xni = state.Ni * state.rho + (pni_rfz_m3 + pni_wfz) * float(dt)
    pni_inu = jnp.maximum(xnc_inu - xni, 0.0) * odts
    vapor_rate_max = jnp.maximum(0.0, (state.qv - qvsi_freeze) * state.rho * odts * 0.999)
    pri_inu = jnp.where(deposition_nucleation_active, jnp.minimum(vapor_rate_max, XM0I * pni_inu), 0.0)
    pni_inu = jnp.where(deposition_nucleation_active, pri_inu / XM0I, 0.0)
    inu_mass = pri_inu * float(dt) / state.rho
    inu_number = pni_inu * float(dt) / state.rho
    # dustyIce nifa sink (WRF 2974-2975).
    nifa_sink_inu = pni_inu * float(dt) / state.rho

    # --- Koop homogeneous freezing of deliquesced aerosols (WRF 2633-2641). ---
    rs_for_ns = jnp.maximum(state.qs * state.rho, R1)
    tc0 = jnp.minimum(-0.1, state.T - 273.15)
    smo0_ns = _snow_moment(0.0, rs_for_ns / 0.069, tc0, tables)
    ns_m3 = jnp.where(state.qs > R1, smo0_ns, 0.0)
    xni_tot = ns_m3 + state.Ni * state.rho + (pni_rfz_m3 + pni_wfz + pni_inu) * float(dt)
    qvs_w = saturation_mixing_ratio_liquid(state.p, state.T)
    koop_active = (xni_tot <= 999.0e3) & (state.T < 238.0) & (ssati_freeze >= 0.4)
    xnc_koop = _ice_koop(state.T, state.qv, qvs_w, nwfa_m3, float(dt))
    pni_iha = jnp.where(koop_active, xnc_koop * odts, 0.0)
    pri_iha = jnp.minimum(vapor_rate_max, XM0I * 0.1 * pni_iha)
    pni_iha = pri_iha / (XM0I * 0.1)
    iha_mass = pri_iha * float(dt) / state.rho
    iha_number = pni_iha * float(dt) / state.rho
    nwfa_sink_iha = pni_iha * float(dt) / state.rho

    # --- Melt + deposition/sublimation: identical structure to mp=8. ---
    tempc, diffu, _visco, tcond, _lvap, ocp, rhof, rhof2, vsc2 = _air_properties(state)
    del rhof
    qvs0 = saturation_mixing_ratio_liquid(state.p, T_0)
    del_qvs = jnp.maximum(0.0, qvs0 - state.qv)
    twet = state.T
    rs, _xds, smo0, smo1, smof, _c_snow, active_snow = _snow_moments(state.qs, state.rho, tempc, tables)
    rg, ng, _lamg, ilamg, n0_g, active_graupel = _graupel_distribution(state.qg, state.Ng, state.rho)

    prr_sml_rate = (tempc * tcond - 2.5e6 * diffu * del_qvs) * (T1_MELT_QS * smo1 + T2_MELT_QS * rhof2 * vsc2 * smof)
    prr_sml_rate = jnp.minimum(rs * odts, jnp.maximum(0.0, prr_sml_rate)) / state.rho
    snow_melt = jnp.where((state.T > T_0) & active_snow, prr_sml_rate * float(dt), 0.0)
    pnr_sml = jnp.where(rs > R1, smo0 / rs * snow_melt * state.rho * 10.0 ** (-0.25 * (twet - T_0)), 0.0)

    prg_gml_rate = (tempc * tcond - 2.5e6 * diffu * del_qvs) * n0_g * (
        T1_MELT_QG * ilamg**CRE10 + T2_MELT_QG * rhof2 * vsc2 * ilamg**CGE11
    )
    prg_gml_rate = jnp.minimum(rg * odts, jnp.maximum(0.0, prg_gml_rate)) / state.rho
    graupel_melt = jnp.where((state.T > T_0) & active_graupel, prg_gml_rate * float(dt), 0.0)
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
    ri, ni, _lami, ilami, xdi, xmi, active_ice = _ice_distribution_local(state.qi, state.Ni, state.rho)
    rs, _xds, _smo0, smo1, smof, c_snow, active_snow = _snow_moments(state.qs, state.rho, state.T - 273.15, tables)
    rg, ng, _lamg, ilamg, n0_g, active_graupel = _graupel_distribution(state.qg, state.Ng, state.rho)

    idx_i = jnp.where(ri > R_I_FIRST, _lookup_digit_index(ri, -10, N_I_TABLE), 0)
    idx_i1 = jnp.where(ni > NT_I_FIRST, _lookup_digit_index(ni, 0, N_I1_TABLE), 0)

    pri_ide_raw = 0.5 * t1_subl * diffu * ssati * rvs * OIG1 * 1.0 * ni * ilami
    pri_ide_raw = jnp.where(active_ice, pri_ide_raw, 0.0)
    pri_ide_limited = jnp.where(
        pri_ide_raw < 0.0,
        jnp.maximum(-ri * odts, pri_ide_raw),
        jnp.minimum(pri_ide_raw, jnp.maximum(state.qv - qvsi, 0.0) * state.rho * odts * 0.999),
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
    prs_sde = jnp.where(
        active_snow,
        jnp.where(
            prs_sde < 0.0,
            jnp.maximum(-rs * odts, prs_sde),
            jnp.minimum(prs_sde, jnp.maximum(state.qv - qvsi, 0.0) * state.rho * odts * 0.999),
        ),
        0.0,
    )
    vapor_rate_max2 = (state.qv - qvsi) * state.rho * odts * 0.999
    prg_gde = 0.5 * t1_subl * diffu * ssati * rvs * n0_g * (
        T1_SUBL_QG * ilamg**CRE10 + T2_SUBL_QG * vsc2 * rhof2 * ilamg**CGE11
    )
    prg_gde = jnp.where(active_graupel & (ssati < -EPS), jnp.maximum(jnp.maximum(-rg * odts, prg_gde), vapor_rate_max2), 0.0)
    # Vapor limiter (WRF 2862-2876) — includes pri_inu and pri_iha; here the
    # nucleation terms were already applied above, so the limiter covers the
    # remaining deposition family exactly as the mp=8 kernel does.
    deposition_sum = pri_ide + prs_ide + prs_sde + prg_gde
    limited = ((deposition_sum > EPS) & (deposition_sum > vapor_rate_max2)) | (
        (deposition_sum < -EPS) & (deposition_sum < vapor_rate_max2)
    )
    deposition_denom = jnp.where(jnp.abs(deposition_sum) > R1, deposition_sum, 1.0)
    deposition_ratio = jnp.where(limited, vapor_rate_max2 / deposition_denom, 1.0)
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
    vapor_sink = (
        jnp.maximum(0.0, ice_deposition)
        + jnp.maximum(0.0, snow_from_ice_deposition)
        + jnp.maximum(0.0, snow_deposition)
        + jnp.maximum(0.0, graupel_deposition)
    )
    vapor_source = jnp.maximum(0.0, -ice_deposition) + jnp.maximum(0.0, -snow_deposition) + jnp.maximum(0.0, -graupel_deposition)
    updated_qv = state.qv - vapor_sink + vapor_source - inu_mass - iha_mass
    updated_T = state.T + LSUB * ocp * (vapor_sink - vapor_source + inu_mass + iha_mass)
    updated = state.replace(
        qv=updated_qv,
        qi=state.qi + ice_deposition - ice_to_snow + inu_mass + iha_mass,
        qs=state.qs + snow_from_ice_deposition + ice_to_snow + snow_deposition,
        qg=state.qg + graupel_deposition,
        Ni=jnp.maximum(
            0.0,
            state.Ni
            + inu_number
            + iha_number
            + jnp.where(ice_deposition < 0.0, ice_deposition / jnp.maximum(xmi, XM0I), 0.0)
            - ice_number_to_snow,
        ),
        nifa=jnp.maximum(0.0, state.nifa - nifa_sink_inu),
        nwfa=jnp.maximum(0.0, state.nwfa - nwfa_sink_iha),
        T=updated_T,
        rho=density_from_pressure_temperature(state.p, updated_T, updated_qv),
    )
    return updated, graupel_melt


def _ice_distribution_local(qi, Ni, rho):
    ri = jnp.maximum(qi * rho, R1)
    ni = jnp.maximum(Ni * rho, R2)
    lami = (AM_I * 6.0 * OIG1 * ni / ri) ** OBMI
    ilami = 1.0 / lami
    xdi = jnp.maximum(D0I, (3.0 + 0.0 + 1.0) * ilami)
    xmi = AM_I * xdi**3.0
    return ri, ni, lami, ilami, xdi, xmi, qi > R1


def _saturation_adjustment_aero(
    state: ThompsonAeroColumnState,
    dt: float,
    aero: ThompsonAeroTableBundle = THOMPSON_AERO_TABLES,
) -> tuple[ThompsonAeroColumnState, jax.Array]:
    """Cloud condensation/evaporation with explicit CCN (WRF 3399-3494).

    Condensation: droplet nucleation from the parcel-model activation table
    (xnc = activ_ncloud(T, w, nwfa); pnc_wcd source on Nc, equal sink on
    nwfa).  Evaporation: droplet number loss from the explicit ``tnc_wev``
    bin table (drops smaller than Dc_star evaporate entirely), the lost
    number RETURNS to nwfa.  Full-evaporation branch zeroes the cloud.
    """

    odt = 1.0 / float(dt)
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

    rc = jnp.maximum(state.qc * state.rho, R1)
    nc_m3 = _entry_cloud_number(state.qc, state.Nc, state.rho, aero)
    xrc = rc + clap * state.rho

    # --- Droplet nucleation (WRF 3413-3420). ---
    nwfa_m3 = _nwfa_working(state.nwfa, state.rho)
    xnc_act = jnp.maximum(2.0, _activ_ncloud(state.T, state.w, nwfa_m3, aero))
    pnc_wcd_pos = 0.5 * (xnc_act - nc_m3 + jnp.abs(xnc_act - nc_m3)) * odt / state.rho  # per kg s^-1

    # --- Droplet evaporation number loss (WRF 3422-3470). ---
    tempc = state.T - 273.15
    otemp = 1.0 / state.T
    rvs = state.rho * qvs
    rvs_p = rvs * otemp * (lvap * otemp / RV - 1.0)
    rvs_pp = rvs * (
        otemp * (lvap * otemp / RV - 1.0) * otemp * (lvap * otemp / RV - 1.0)
        + (-2.0 * lvap * otemp**3 / RV)
        + otemp * otemp
    )
    diffu = 2.11e-5 * (state.T / 273.15) ** 1.94 * (101325.0 / state.p)
    tcond = (5.69 + 0.0168 * tempc) * 1.0e-5 * 418.936
    gamsc = lvap * diffu / tcond * rvs_p
    alphsc = 0.5 * (gamsc / (1.0 + gamsc)) ** 2 * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = jnp.maximum(1.0e-9, alphsc)
    xsat = jnp.where(jnp.abs(ssatw) < 1.0e-9, 0.0, ssatw)
    t1_evap = 2.0 * PI * (1.0 - alphsc * xsat + 2.0 * alphsc**2 * xsat**2 - 5.0 * alphsc**3 * xsat**3) / (1.0 + gamsc)
    dc_star_sq = jnp.maximum(0.0, -2.0 * float(dt) * t1_evap / (2.0 * PI) * 4.0 * diffu * ssatw * rvs / 1000.0)
    dc_star = jnp.sqrt(dc_star_sq)
    idx_d = jnp.clip(jnp.floor(1.0e6 * dc_star).astype(jnp.int32), 1, NBC) - 1
    idx_n = _idx_n_from_nc(nc_m3, aero)
    idx_c = _idx_c_from_rc(rc)
    wev_flat_idx = (idx_d.astype(jnp.int32) * N_C_TABLE + idx_c.astype(jnp.int32)) * NBC + idx_n.astype(jnp.int32)
    tnc_wev = jnp.take(aero.tnc_wev, wev_flat_idx)
    pnc_wcd_neg = jnp.maximum(-nc_m3 * 0.99 / state.rho * odt, -tnc_wev / state.rho * odt)

    # Branch select (WRF 3409-3475).
    cond_branch = clap > EPS
    evap_branch = (clap < -EPS) & (ssatw < -1.0e-6)
    full_evap = xrc <= R1
    prw_vcd = jnp.where(full_evap, -rc / state.rho * odt, clap * odt)
    prw_vcd = jnp.where(~full_evap & (clap < 0.0), jnp.maximum(-rc * 0.99 / state.rho * odt, prw_vcd), prw_vcd)
    pnc_wcd = jnp.where(
        full_evap,
        -nc_m3 / state.rho * odt,
        jnp.where(cond_branch, pnc_wcd_pos, jnp.where(evap_branch, pnc_wcd_neg, 0.0)),
    )

    prw_vcd = jnp.where(active, prw_vcd, 0.0)
    pnc_wcd = jnp.where(active, pnc_wcd, 0.0)
    # Bound the condensate by available vapor (mp=8 kernel guard).
    delta_q = jnp.clip(prw_vcd * float(dt), -state.qc, state.qv - 1.0e-10)
    delta_nc = pnc_wcd * float(dt)
    qv = state.qv - delta_q
    T = state.T + lvap * ocp * delta_q
    nc_new = jnp.maximum(0.0, state.Nc + delta_nc)
    nwfa_new = jnp.maximum(0.0, state.nwfa - delta_nc)  # WRF 3482: nwfaten -= pnc_wcd
    condensed_cloud = delta_q > EPS
    adjusted = state.replace(
        qv=qv,
        qc=state.qc + delta_q,
        Nc=nc_new,
        nwfa=nwfa_new,
        T=T,
        rho=density_from_pressure_temperature(state.p, T, qv),
    )
    return adjusted, condensed_cloud


def _rain_evaporation_aero(
    state: ThompsonAeroColumnState,
    dt: float,
    skip_evaporation: jax.Array | bool = False,
    graupel_melt: jax.Array | float = 0.0,
) -> ThompsonAeroColumnState:
    """Rain evaporation (WRF 3500-3574) + CCN return ``nwfaten += pnr_rev``."""

    _tempc, diffu, _visco, tcond, lvap, ocp, _rhof, rhof2, vsc2 = _air_properties(state)
    qvs = saturation_mixing_ratio_liquid(state.p, state.T)
    ssatw = state.qv / qvs - 1.0
    rvs = state.rho * qvs
    otemp = 1.0 / state.T
    rvs_p = rvs * otemp * (lvap * otemp / RV - 1.0)
    rvs_pp = rvs * (
        otemp * (lvap * otemp / RV - 1.0) * otemp * (lvap * otemp / RV - 1.0)
        + (-2.0 * lvap * otemp**3 / RV)
        + otemp * otemp
    )
    gamsc = lvap * diffu / tcond * rvs_p
    alphsc = 0.5 * (gamsc / (1.0 + gamsc)) ** 2 * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = jnp.maximum(1.0e-9, alphsc)
    xsat = jnp.minimum(-1.0e-9, ssatw)
    t1_evap = 2.0 * PI * (1.0 - alphsc * xsat + 2.0 * alphsc**2 * xsat**2 - 5.0 * alphsc**3 * xsat**3) / (1.0 + gamsc)
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
    # pnr_rev (WRF 3559-3560): number of fully evaporated drops, returned as CCN.
    nr_loss = jnp.where(state.qr > 0.0, jnp.minimum(state.Nr * 0.99, state.Nr * evap / jnp.maximum(state.qr, R1)), 0.0)
    return state.replace(
        qv=state.qv + evap,
        qr=state.qr - evap,
        Nr=jnp.maximum(0.0, state.Nr - nr_loss),
        nwfa=state.nwfa + nr_loss,  # WRF 3565: nwfaten += pnr_rev
        T=state.T - lvap * ocp * evap,
    )


def _cloud_water_fall_speeds_aero(state: ThompsonAeroColumnState, aero: ThompsonAeroTableBundle):
    """Cloud mass/number fall speeds with variable nu_c (WRF 3655-3665)."""

    rho = jnp.maximum(state.rho, R1)
    rhof = _rho_correction(rho)
    rc = jnp.maximum(state.qc * rho, R1)
    nc_m3 = _entry_cloud_number(state.qc, state.Nc, rho, aero)
    nu = _nu_c_from_nc(nc_m3)
    ccg2 = jnp.take(aero.ccg[1], nu)
    ocg1 = jnp.take(aero.ocg1, nu)
    ccg5 = jnp.take(aero.ccg[4], nu)
    ocg2 = jnp.take(aero.ocg2, nu)
    ccg4 = jnp.take(aero.ccg[3], nu)
    lamc = (nc_m3 * AM_R * ccg2 * ocg1 / rc) ** OBMR
    ilamc = 1.0 / lamc
    vtc = rhof * AV_C * ccg5 * ocg2 * ilamc**BV_C
    vtnc = rhof * AV_C * ccg4 * ocg1 * ilamc**BV_C
    active = (state.qc > R1) & (state.w < 1.0e-1)
    return jnp.where(active, vtc, 0.0), jnp.where(active, vtnc, 0.0)


def _sed_cloud_water_aero(state: ThompsonAeroColumnState, dt: float, aero: ThompsonAeroTableBundle):
    """Cloud water + droplet-number sedimentation below 500 m AGL (WRF 3824-3837)."""

    rho = jnp.maximum(state.rho, R1)
    dz = jnp.maximum(state.dz, 1.0)
    vtc, vtnc = _cloud_water_fall_speeds_aero(state, aero)
    dt_a = jnp.asarray(dt, state.qc.dtype)

    hgt_agl = jnp.cumsum(dz, axis=-1) - dz
    below_500m = hgt_agl < 500.0

    rc = jnp.maximum(state.qc * rho, 0.0)
    nc = jnp.maximum(state.Nc * rho, 0.0)
    sed_c = jnp.where(below_500m, vtc * rc, 0.0)
    sed_n = jnp.where(below_500m, vtnc * nc, 0.0)
    sed_c_above = jnp.concatenate([sed_c[..., 1:], jnp.zeros_like(sed_c[..., :1])], axis=-1)
    sed_n_above = jnp.concatenate([sed_n[..., 1:], jnp.zeros_like(sed_n[..., :1])], axis=-1)
    dq = (sed_c_above - sed_c) / dz / rho * dt_a
    dn = (sed_n_above - sed_n) / dz / rho * dt_a
    qc_new = jnp.where(below_500m, jnp.maximum(state.qc + dq, 0.0), state.qc)
    nc_new = jnp.where(below_500m, jnp.maximum(state.Nc + dn, 0.0), state.Nc)
    cloudw_surface_loss = sed_c[..., 0] * dt_a
    return qc_new, nc_new, cloudw_surface_loss.astype(jnp.float64)


def _sedimentation_aero(state: ThompsonAeroColumnState, dt: float, aero: ThompsonAeroTableBundle):
    """Four-species WRF sedimentation (reused from mp=8) + aero cloud channel."""

    vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num = _fall_speeds(state)
    dz = jnp.maximum(state.dz, 1.0)
    rho = jnp.maximum(state.rho, R1)

    nstep_r = _nstep_per_column(vt_r_mass, vt_r_num, dz, dt)
    nstep_i = _nstep_per_column(vt_i_mass, vt_i_mass, dz, dt)
    nstep_s = _nstep_per_column(vt_s_mass, vt_s_mass, dz, dt)
    nstep_g = _nstep_per_column(vt_g_mass, vt_g_mass, dz, dt)

    Nr_sed = _clamp_rain_number(state.qr, state.Nr, rho)
    qr, Nr, ppt_rain = _sed_one_species(state.qr, Nr_sed, vt_r_mass, vt_r_num, dz, rho, dt, nstep_r)
    qi, Ni, ppt_ice = _sed_one_species(state.qi, state.Ni, vt_i_mass, vt_i_num, dz, rho, dt, nstep_i)
    qs, Ns, ppt_snow = _sed_one_species(state.qs, state.Ns, vt_s_mass, vt_s_mass, dz, rho, dt, nstep_s)
    qg, Ng, ppt_graupel = _sed_one_species(state.qg, state.Ng, vt_g_mass, vt_g_num, dz, rho, dt, nstep_g)

    qc, Nc, ppt_cloudw = _sed_cloud_water_aero(state, dt, aero)

    updated = state.replace(qc=qc, Nc=Nc, qr=qr, Nr=Nr, qi=qi, Ni=Ni, qs=qs, Ns=Ns, qg=qg, Ng=Ng)
    precip = {
        "rain": ppt_rain,
        "snow": ppt_snow,
        "graupel": ppt_graupel,
        "ice": ppt_ice,
        "cloudw": ppt_cloudw,
    }
    return updated, precip


def _instant_melt_freeze_aero(state: ThompsonAeroColumnState, dt: float) -> ThompsonAeroColumnState:
    """Instant qi melt above 0C / qc freeze below HGFR with number transfers
    (WRF 3945-3967: melt moves Ni into Nc; freeze moves Nc into Ni)."""

    del dt
    ocp = cp_inverse(state.qv)
    lvap = latent_heat_vaporization(state.T)
    lfus2 = LSUB - lvap

    melt = (state.T > T_0) & (state.qi > 0.0)
    qi_melt = jnp.where(melt, state.qi, 0.0)
    ni_melt = jnp.where(melt, state.Ni, 0.0)
    state = state.replace(
        qc=state.qc + qi_melt,
        qi=state.qi - qi_melt,
        Nc=state.Nc + ni_melt,
        Ni=jnp.where(melt, 0.0, state.Ni),
        T=state.T - LFUS * ocp * qi_melt,
    )

    freeze = (state.T < HGFR) & (state.qc > 0.0)
    qc_freeze = jnp.where(freeze, state.qc, 0.0)
    nc_freeze = jnp.where(freeze, state.Nc, 0.0)
    return state.replace(
        qc=state.qc - qc_freeze,
        qi=state.qi + qc_freeze,
        Ni=state.Ni + nc_freeze,
        Nc=jnp.where(freeze, 0.0, state.Nc),
        T=state.T + lfus2 * ocp * qc_freeze,
    )


def _finish_aero(state: ThompsonAeroColumnState, aero: ThompsonAeroTableBundle) -> ThompsonAeroColumnState:
    """Final floors/balances incl. the aerosol writeback (WRF 3972-4055).

    qc/nc: floor nc at 2/rho, cap Nt_c_max, zero with the cloud, then the
    final lamc rebalance (WRF 4007-4021).  nwfa/nifa: the per-kg writeback
    clamps (WRF 3979-3982, aer_init_opt<2).  Ice/rain identical to mp=8.
    """

    qv = jnp.maximum(state.qv, 1.0e-10)
    qc = jnp.where(state.qc <= R1, 0.0, jnp.maximum(state.qc, 0.0))
    qr = jnp.where(state.qr <= R1, 0.0, jnp.maximum(state.qr, 0.0))
    qi = jnp.where(state.qi <= R1, 0.0, jnp.maximum(state.qi, 0.0))
    qs = jnp.where(state.qs <= R1, 0.0, jnp.maximum(state.qs, 0.0))
    qg = jnp.where(state.qg <= R1, 0.0, jnp.maximum(state.qg, 0.0))
    T = jnp.maximum(state.T, 50.0)
    rho = density_from_pressure_temperature(state.p, T, qv)

    # Cloud number writeback (WRF 3976, 4007-4021).
    nc = jnp.maximum(2.0 / rho, jnp.minimum(state.Nc, NT_C_MAX))
    Nc = jnp.where(qc <= R1, 0.0, _cloud_number_balance(qc, nc, rho, aero))

    # Aerosols: per-kg clamps (WRF 3979-3982).
    nwfa = jnp.maximum(NWFA_FLOOR, jnp.minimum(AERO_CAP, state.nwfa))
    nifa = jnp.maximum(NIFA_FLOOR, jnp.minimum(AERO_CAP, state.nifa))

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
    return state.replace(
        qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,
        Ni=Ni, Nr=Nr, Nc=Nc, nwfa=nwfa, nifa=nifa, T=T, rho=rho,
    )


def _thermo_admissible(state: ThompsonAeroColumnState) -> jax.Array:
    return (
        (state.p > 1000.0)
        & (state.p < 200000.0)
        & (state.T > 150.0)
        & (state.T < 400.0)
        & (state.rho > 0.0)
        & (state.rho < 10.0)
    )


def _select_state(mask, good, fallback):
    return good.replace(
        **{
            name: jnp.where(mask, getattr(good, name), getattr(fallback, name))
            for name in ThompsonAeroColumnState.__slots__
        }
    )


def apply_surface_aerosol_emission(state: ThompsonAeroColumnState, nwfa2d, nifa2d, dt: float) -> ThompsonAeroColumnState:
    """Fake surface aerosol emission AFTER the column step (WRF 1317-1326).

    ``nwfa2d``/``nifa2d`` are per-kg-per-second number tendencies on the
    lowest model level (shape = column batch shape).
    """

    nwfa = state.nwfa.at[..., 0].add(jnp.asarray(nwfa2d) * float(dt))
    nifa = state.nifa.at[..., 0].add(jnp.asarray(nifa2d) * float(dt))
    return state.replace(nwfa=nwfa, nifa=nifa)


def _no_micro_column_skip(raw: ThompsonAeroColumnState):
    """WRF per-column ``no_micro`` skip (module_mp_thompson.F:1646-2020).

    A column with NO hydrometeor above R1 at ANY level AND no ice
    supersaturation anywhere hits ``if (no_micro) return`` BEFORE any
    process, floor or aerosol writeback runs.  Such columns leave
    ``mp_thompson`` with only the ENTRY zeroing applied (sub-R1 species and
    their numbers zeroed; qv, T, nwfa, nifa untouched).  Returns
    ``(active_col, skipped_state)``; ``active_col`` has the batch shape.
    """

    qv_w = jnp.maximum(raw.qv, 1.0e-10)
    qvs = saturation_mixing_ratio_liquid(raw.p, raw.T)
    qvsi = jnp.where(raw.T - 273.15 <= 0.0, saturation_mixing_ratio_ice(raw.p, raw.T), qvs)
    ssati = qv_w / qvsi - 1.0
    ssati = jnp.where(jnp.abs(ssati) < EPS, 0.0, ssati)
    hydro = (raw.qc > R1) | (raw.qi > R1) | (raw.qr > R1) | (raw.qs > R1) | (raw.qg > R1)
    active_col = jnp.any(hydro | (ssati > 0.0), axis=-1)
    skipped = raw.replace(
        qc=jnp.where(raw.qc > R1, raw.qc, 0.0),
        qi=jnp.where(raw.qi > R1, raw.qi, 0.0),
        qr=jnp.where(raw.qr > R1, raw.qr, 0.0),
        qs=jnp.where(raw.qs > R1, raw.qs, 0.0),
        qg=jnp.where(raw.qg > R1, raw.qg, 0.0),
        Nc=jnp.where(raw.qc > R1, raw.Nc, 0.0),
        Ni=jnp.where(raw.qi > R1, raw.Ni, 0.0),
        Nr=jnp.where(raw.qr > R1, raw.Nr, 0.0),
    )
    return active_col, skipped


def _thompson_aero_body(state: ThompsonAeroColumnState, dt: float, debug: bool):
    """One aerosol-aware Thompson column step (sequential mp=8 structure)."""

    del debug
    active_col, skipped = _no_micro_column_skip(state)
    state = _clip_species_aero(state)
    valid = _thermo_admissible(state)
    fallback = state
    state = _warm_rain_collection_aero(state, dt)
    state = _snow_graupel_scavenging(state, dt)
    state, graupel_melt = _ice_sources_aero(state, dt)
    state, cloud_condensed = _saturation_adjustment_aero(state, dt)
    state = _rain_evaporation_aero(state, dt, skip_evaporation=cloud_condensed, graupel_melt=graupel_melt)
    state, precip = _sedimentation_aero(state, dt, THOMPSON_AERO_TABLES)
    state = _instant_melt_freeze_aero(state, dt)
    state = _finish_aero(state, THOMPSON_AERO_TABLES)
    state = _select_state(valid, state, fallback)
    # WRF no_micro early-return: skipped columns keep their entry state.
    state = _select_state(active_col[..., None], state, skipped)
    precip = {
        k: jnp.where(active_col, jnp.asarray(v), 0.0).astype(jnp.float64)
        for k, v in precip.items()
    }
    return state, precip


@partial(jax.jit, static_argnames=("dt", "debug"))
def step_thompson_aero_column_with_precip(
    state: ThompsonAeroColumnState, dt: float, *, debug: bool = False
):
    """Advance one aerosol-aware Thompson column step; returns (state, precip mm)."""

    return _thompson_aero_body(state, dt, debug)


# ------------------------------------------------------------------------------
# Aerosol-aware initialization (thompson_init climatological profiles)
# ------------------------------------------------------------------------------


def climatological_aerosol_profiles(hgt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WRF ``thompson_init`` self-init aerosol profiles + surface emission.

    ``hgt`` is the 3-D mass-level height field with vertical LAST (m MSL).
    Mirrors module_mp_thompson.F:493-558 (use_aero_icbc=.false. path):
    boundary-layer-following exponentials for nwfa/nifa (per kg) and the
    fake surface emission ``nwfa2d`` (per kg per second).  ``nifa2d`` is
    zero in WRF for this path (never assigned by thompson_init).

    Returns (nwfa, nifa, nwfa2d).
    """

    hgt = np.asarray(hgt, dtype=np.float64)
    h0 = hgt[..., :1]
    h_01 = np.where(
        h0 <= 1000.0, 0.8, np.where(h0 >= 2500.0, 0.01, 0.8 * np.cos(h0 * 0.001 - 1.0))
    )
    ni_ccn3 = -1.0 * np.log(NA_CCN1 / NA_CCN0) / h_01
    ni_in3 = -1.0 * np.log(NA_IN1 / NA_IN0) / h_01
    dz_agl = hgt - h0
    # Level 1 uses the LEVEL-2 height offset (WRF lines 508, 546).
    dz1 = hgt[..., 1:2] - h0
    dz_eff = np.concatenate([dz1, dz_agl[..., 1:]], axis=-1)
    nwfa = NA_CCN1 + NA_CCN0 * np.exp(-(dz_eff / 1000.0) * ni_ccn3)
    nifa = NA_IN1 + NA_IN0 * np.exp(-(dz_eff / 1000.0) * ni_in3)
    z1 = np.maximum(dz1[..., 0], 1.0)
    nwfa2d = nwfa[..., 0] * 0.000196 * (50.0 / z1)
    return nwfa, nifa, nwfa2d
