"""Faithful fp64 NumPy transcription of WRF ``SFCLAY1D_mynn``.

This is a LITERAL, line-referenced port of the pristine surface-layer routine in
``/home/enric/src/wrf_pristine/WRF/phys/module_sf_mynn.F`` (sha256 recorded in
``oracle_source_sha256.txt``) for the default Canary configuration:

  ISFFLX=1, isftcflx=0, iz0tlnd=0, spp_pbl=0, COARE_OPT=3.0, psi_opt=0 (CB05),
  no snow (SNOWH<0.1), itimestep>1 (warm replay step).

It carries NO empirical repair: it is the algorithm WRF runs, in fp64, so that
comparing it against the fp32 Fortran oracle isolates fp roundoff, and comparing
the production ``surface_layer.py`` against THIS isolates port divergence from
algorithm behaviour.

Vectorized over a 1-D array of columns. All math in float64.
"""

from __future__ import annotations

import numpy as np

# The CB05 psi functions are evaluated for both stable & unstable branches on every
# cell and the wrong-sign branch is masked by np.where; the masked branch can take
# fractional powers of negatives -> NaN that is discarded. Silence those warnings.
np.seterr(invalid="ignore")

# --- WRF physical constants exactly as passed into SFCLAY_mynn ---
CP = 1004.5            # cp = 7*r_d/2
G = 9.81
R_D = 287.0
R_V = 461.6
ROVCP = R_D / CP
XLV = 2.5e6
KARMAN = 0.4
SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15
EP1 = R_V / R_D - 1.0   # 0.608362...  (ep_1, virtual-temp constant)
EP2 = R_D / R_V         # 0.621750...  (ep_2)
EP3 = 1.0 - EP2         # ep_3
P1000 = 100000.0
PRT = 1.0

# module-level MYNN parameters (module_sf_mynn.F:82-86)
WMIN = 0.1
VCONVC = 1.25
COARE_OPT = 3.0
CZIL = 0.085           # zilitinkevich_1995 land default (iz0tlnd<=1)

# --- CB05 integrated similarity tables (module_sf_mynn.F:2058-2069, psi_opt=0) ---
_N = 1000


def _psim_stable_full(z):
    return -6.1 * np.log(z + (1.0 + z ** 2.5) ** (1.0 / 2.5))


def _psih_stable_full(z):
    return -5.3 * np.log(z + (1.0 + z ** 1.1) ** (1.0 / 1.1))


def _psim_unstable_full(z):
    x = (1.0 - 16.0 * z) ** 0.25
    psimk = 2.0 * np.log(0.5 * (1.0 + x)) + np.log(0.5 * (1.0 + x * x)) - 2.0 * np.arctan(x) + 2.0 * np.arctan(1.0)
    ym = (1.0 - 10.0 * z) ** 0.33
    psimc = (3.0 / 2.0) * np.log((ym ** 2.0 + ym + 1.0) / 3.0) - np.sqrt(3.0) * np.arctan((2.0 * ym + 1.0) / np.sqrt(3.0)) + 4.0 * np.arctan(1.0) / np.sqrt(3.0)
    return (psimk + z ** 2 * psimc) / (1.0 + z ** 2.0)


def _psih_unstable_full(z):
    y = (1.0 - 16.0 * z) ** 0.5
    psihk = 2.0 * np.log((1.0 + y) / 2.0)
    yh = (1.0 - 34.0 * z) ** 0.33
    psihc = (3.0 / 2.0) * np.log((yh ** 2.0 + yh + 1.0) / 3.0) - np.sqrt(3.0) * np.arctan((2.0 * yh + 1.0) / np.sqrt(3.0)) + 4.0 * np.arctan(1.0) / np.sqrt(3.0)
    return (psihk + z ** 2 * psihc) / (1.0 + z ** 2.0)


# Build node tables (module_sf_mynn.F:2059-2069). float64 to match an fp64 ref;
# the fp32 oracle builds them in REAL*4 -- the residual between the two is the
# fp32-roundoff floor we quantify, not a bug.
_n = np.arange(0, _N + 1, dtype=np.float64)
_PSIM_STAB = _psim_stable_full(_n * 0.01)
_PSIH_STAB = _psih_stable_full(_n * 0.01)
_PSIM_UNSTAB = _psim_unstable_full(-_n * 0.01)
_PSIH_UNSTAB = _psih_unstable_full(-_n * 0.01)


def _lookup(table, full_fn, zolf, coord):
    """psi*_stable/unstable lookup, module_sf_mynn.F:2197-2271.

    nzol = int(coord); rzol = coord-nzol; if nzol+1<=1000 interpolate else full.
    Fortran ``int`` truncates toward zero; for the stable branch coord>=0, for the
    unstable branch coord = -zolf*100 >= 0, so a floor on the magnitude matches.
    """
    nzol = np.trunc(coord).astype(np.int64)
    rzol = coord - nzol
    in_table = (nzol + 1) <= _N
    nzol_c = np.clip(nzol, 0, _N - 1)
    base = table[nzol_c]
    nxt = table[np.clip(nzol_c + 1, 0, _N)]
    interp = base + rzol * (nxt - base)
    return np.where(in_table, interp, full_fn(zolf))


def psim_stable(z):
    return _lookup(_PSIM_STAB, _psim_stable_full, z, z * 100.0)


def psih_stable(z):
    return _lookup(_PSIH_STAB, _psih_stable_full, z, z * 100.0)


def psim_unstable(z):
    return _lookup(_PSIM_UNSTAB, _psim_unstable_full, z, -z * 100.0)


def psih_unstable(z):
    return _lookup(_PSIH_UNSTAB, _psih_unstable_full, z, -z * 100.0)


def li_etal_2010(rib, zaz0, z0zt):
    """Li et al. (2010) z/L, module_sf_mynn.F:1831-1890. Vectorized, branch-free."""
    au11, bu11, bu12 = 0.045, 0.003, 0.0059
    bu21, bu22, bu31, bu32, bu33 = -0.0828, 0.8845, 0.1739, -0.9213, -0.1057
    aw11, aw12, aw21, aw22 = 0.5738, -0.4399, -4.901, 52.50
    bw11, bw12, bw21, bw22 = -0.0539, 1.540, -0.669, -3.282
    as11, as21, bs11, bs21, bs22 = 0.7529, 14.94, 0.1569, -0.3091, -1.303
    zaz02 = np.clip(zaz0, 100.0, 100000.0)
    z0zt2 = np.clip(z0zt, 0.5, 100.0)
    alfa = np.log(zaz02)
    beta = np.log(z0zt2)
    zl_uns = au11 * alfa * rib ** 2 + ((bu11 * beta + bu12) * alfa ** 2 + (bu21 * beta + bu22) * alfa + (bu31 * beta ** 2 + bu32 * beta + bu33)) * rib
    zl_uns = np.clip(zl_uns, -15.0, 0.0)
    zl_w = ((aw11 * beta + aw12) * alfa + (aw21 * beta + aw22)) * rib ** 2 + ((bw11 * beta + bw12) * alfa + (bw21 * beta + bw22)) * rib
    zl_w = np.clip(zl_w, 0.0, 4.0)
    zl_s = (as11 * alfa + as21) * rib + bs11 * alfa + bs21 * beta + bs22
    zl_s = np.clip(zl_s, 1.0, 20.0)
    return np.where(rib <= 0.0, zl_uns, np.where(rib <= 0.2, zl_w, zl_s))


def zolrib(ri, za, z0, zt, logz0, logzt, zol1):
    """Brute-force z/L fixed-point, module_sf_mynn.F:1984-2048.

    Faithful: seeds zolold with zol1 on n==1 (the WRF first guess passed in), then
    iterates zolrib = ri*psix2**2/psit2 up to nmax=20, early-stop |dz|<=0.01, with
    Li_etal_2010 fallback on non-convergence. ``zol1`` MUST be the WRF first guess
    (MOL-based for warm steps), because the fixed-point can land on different roots
    from different seeds for the same ri (it is NOT globally contractive).
    """
    ri = np.asarray(ri, dtype=np.float64)
    unstable = ri < 0.0
    zol1 = np.where(zol1 * ri < 0.0, 0.0, zol1)  # WRONG-QUADRANT guard (line 1998)
    # Fortran init (lines 2003-2010): zolrib/zolold sentinels, n=1.
    zolrib_v = np.where(unstable, -66666.0, 66666.0)
    n = 1
    # ``active`` mirrors the Fortran while-head:  abs(zolold-zolrib) > 0.01 .and. n<nmax
    # On entry n==1 so the head is True for every column (sentinels differ by huge).
    active = np.ones_like(ri, dtype=bool)
    converged = np.zeros_like(ri, dtype=bool)
    while n < 20:
        # zolold = zol1 (n==1) else previous zolrib
        zolold_use = zol1 if n == 1 else zolrib_v
        zol20 = zolold_use * z0 / za
        zol3 = zolold_use + zol20
        zolt = zolold_use * zt / za
        psit2_u = np.maximum(logzt - (psih_unstable(zol3) - psih_unstable(zolt)), 1.0)
        psix2_u = np.maximum(logz0 - (psim_unstable(zol3) - psim_unstable(zol20)), 1.0)
        psit2_s = np.maximum(logzt - (psih_stable(zol3) - psih_stable(zolt)), 1.0)
        psix2_s = np.maximum(logz0 - (psim_stable(zol3) - psim_stable(zol20)), 1.0)
        psit2 = np.where(unstable, psit2_u, psit2_s)
        psix2 = np.where(unstable, psix2_u, psix2_s)
        new = ri * psix2 ** 2 / psit2
        # Only columns still active (head was True at this n) update zolrib.
        zolrib_v = np.where(active, new, zolrib_v)
        # Next-iteration head test: abs(zolold_of_next - zolrib) where zolold_of_next
        # == this zolrib. The Fortran evaluates abs(zolold - zolrib) with the NEW
        # zolrib and the zolold set THIS pass -> abs(zolold_use - new).
        head = np.abs(zolold_use - zolrib_v) > 0.01
        # a column that was active and whose head is now False has converged.
        converged = converged | (active & ~head)
        active = active & head
        n += 1
    zol_fallback = li_etal_2010(ri, za / z0, z0 / zt)
    # Fortran: if (n==nmax .and. abs(zolold-zolrib)>0.01) use fallback. i.e. columns
    # that never satisfied the head within nmax -> fallback.
    return np.where(converged, zolrib_v, zol_fallback)


def sfclay1d_mynn(inp):
    """Run faithful MYNN-SL over columns. ``inp`` is a dict of 1-D float64 arrays.

    Required keys: u, v, t1d, qv, p1d, dz8w, rho, mavail, pblh, xland, tsk, psfcpa,
    snowh, znt, ust, mol, qsfc, hfx, qfx, dx, u1d2, v1d2, dz2w.
    Returns a dict of output fields matching the oracle.
    """
    g = {k: np.asarray(v, dtype=np.float64) for k, v in inp.items()}
    U1D, V1D, T1D, QV1D, P1D = g["u"], g["v"], g["t1d"], g["qv"], g["p1d"]
    dz8w, RHO1D = g["dz8w"], g["rho"]
    U1D2, V1D2, dz2w1d = g["u1d2"], g["v1d2"], g["dz2w"]
    MAVAIL, PBLH, XLAND, TSK, PSFCPA = g["mavail"], g["pblh"], g["xland"], g["tsk"], g["psfcpa"]
    SNOWH, ZNT0, UST_in, MOL_in, QSFC_in = g["snowh"], g["znt"], g["ust"], g["mol"], g["qsfc"]
    HFX_in, QFX_in, DX = g["hfx"], g["qfx"], float(g["dx"][0])

    # --- preliminaries (module_sf_mynn.F:499-553) ---
    PSFC = PSFCPA / 1000.0
    THGB = TSK * (100.0 / PSFC) ** ROVCP
    PL = P1D / 1000.0
    THCON = (100.0 / PL) ** ROVCP
    TH1D = T1D * THCON
    TC1D = T1D - 273.15
    QVSH = QV1D / (1.0 + QV1D)               # specific humidity
    TVCON = 1.0 + EP1 * QVSH
    THV1D = TH1D * TVCON
    TV1D = T1D * TVCON
    ZA = 0.5 * dz8w
    ZA2 = dz8w + 0.5 * dz2w1d
    GOVRTH = G / TH1D

    is_water = (XLAND - 1.5) >= 0.0
    is_land = ~is_water

    # QSFC / QSFCMR (module_sf_mynn.F:522-537)
    E1g = np.where(
        TSK < 273.15,
        SVP1 * np.exp(4648.0 * (1.0 / 273.15 - 1.0 / TSK) - 11.64 * np.log(273.15 / TSK) + 0.02265 * (273.15 - TSK)),
        SVP1 * np.exp(SVP2 * (TSK - SVPT0) / (TSK - SVP3)),
    )
    recompute_q = is_water | (QSFC_in <= 0.0)
    QSFC = np.where(recompute_q, EP2 * E1g / (PSFC - EP3 * E1g), QSFC_in)        # spec hum
    QSFCMR = np.where(recompute_q, EP2 * E1g / (PSFC - E1g), QSFC_in / (1.0 - QSFC_in))  # mixing ratio

    CPM = CP * (1.0 + 0.84 * QV1D)

    # WSPD, BR (module_sf_mynn.F:555-607)
    WSPD0 = np.sqrt(U1D * U1D + V1D * V1D)
    THVGB = THGB * (1.0 + EP1 * QSFC)
    DTHVDZ = THV1D - THVGB
    fluxc = np.maximum(HFX_in / RHO1D / CP + EP1 * THVGB * QFX_in / RHO1D, 0.0)
    wstar_water = VCONVC * (G / TSK * PBLH * fluxc) ** 0.33
    wstar_land = VCONVC * (G / TSK * np.minimum(1.5 * PBLH, 4000.0) * fluxc) ** 0.33
    WSTAR = np.where(is_water, wstar_water, wstar_land)
    VSGD = 0.32 * (np.maximum(DX / 5000.0 - 1.0, 0.0)) ** 0.33
    WSPD = np.sqrt(WSPD0 * WSPD0 + WSTAR * WSTAR + VSGD * VSGD)
    WSPD = np.maximum(WSPD, WMIN)
    BR = GOVRTH * ZA * DTHVDZ / (WSPD * WSPD)
    # itimestep>1 clamp (line 597-600)
    BR = np.clip(BR, -4.0, 4.0)

    # VISC, restar, z_t/z_q (module_sf_mynn.F:620-752)
    VISC = 1.326e-5 * (1.0 + 6.542e-3 * TC1D + 8.301e-6 * TC1D ** 2 - 4.84e-9 * TC1D ** 3)
    ZNTstoch = ZNT0  # spp_pbl=0
    restar = np.maximum(UST_in * ZNTstoch / VISC, 0.1)

    # WATER z0 via charnock_1955 (COARE_OPT=3.0); then fairall_etal_2003 z_t/z_q.
    wsp10m = WSPD * np.log(10.0 / 1e-4) / np.log(ZA / 1e-4)
    CZC = 0.011 + 0.007 * np.clip((wsp10m - 10.0) / 8.0, 0.0, 1.0)
    znt_water = np.clip(CZC * UST_in * UST_in / G + 0.11 * VISC / np.maximum(UST_in, 0.05), 1.27e-7, 2.85e-3)
    ZNTstoch = np.where(is_water, znt_water, ZNTstoch)
    # restar recomputed with NEW (water) znt (line 675)
    restar = np.maximum(UST_in * ZNTstoch / VISC, 0.1)

    # fairall_etal_2003 z_t (water): Zt=5.5e-5*restar^-0.6 clip [2e-9,1e-4], Zq=Zt
    zt_water = np.clip(5.5e-5 * restar ** (-0.60), 2.0e-9, 1.0e-4)
    # zilitinkevich_1995 z_t (land, iz0tlnd=0): Zt = z0*exp(-k*CZIL*sqrt(restar)),
    # MIN(Zt, 0.75*z0). Zq = same.
    zt_land = np.minimum(ZNTstoch * np.exp(-KARMAN * CZIL * np.sqrt(restar)), 0.75 * ZNTstoch)
    z_t = np.where(is_water, zt_water, zt_land)
    z_q = z_t
    zratio = ZNTstoch / z_t

    GZ1OZ0 = np.log((ZA + ZNTstoch) / ZNTstoch)
    GZ1OZt = np.log((ZA + ZNTstoch) / z_t)
    GZ2OZ0 = np.log((2.0 + ZNTstoch) / ZNTstoch)
    GZ2OZt = np.log((2.0 + ZNTstoch) / z_t)
    GZ10OZ0 = np.log((10.0 + ZNTstoch) / ZNTstoch)
    GZ10OZt = np.log((10.0 + ZNTstoch) / z_t)

    # --- z/L solve + PSI (module_sf_mynn.F:783-937) ---
    stable = BR > 0.0
    neutral = BR == 0.0
    unstable = BR < 0.0

    # first guess (itimestep>1): ZOL = ZA*k*g*MOL/(TH1D*max(ust^2,...)) clamped.
    zol_guess_s = np.clip(ZA * KARMAN * G * MOL_in / (TH1D * np.maximum(UST_in ** 2, 0.0001)), 0.0, 20.0)
    zol_guess_u = np.clip(ZA * KARMAN * G * MOL_in / (TH1D * np.maximum(UST_in ** 2, 0.001)), -20.0, 0.0)
    zol1 = np.where(stable, zol_guess_s, np.where(unstable, zol_guess_u, 0.0))

    zol_solved = zolrib(BR, ZA, ZNTstoch, z_t, GZ1OZ0, GZ1OZt, zol1)
    ZOL = np.where(stable, np.clip(zol_solved, 0.0, 20.0), np.where(unstable, np.clip(zol_solved, -20.0, 0.0), 0.0))

    zolzt = ZOL * z_t / ZA
    zolz0 = ZOL * ZNTstoch / ZA
    zolza = ZOL * (ZA + ZNTstoch) / ZA
    zol10 = ZOL * (10.0 + ZNTstoch) / ZA
    zol2 = ZOL * (2.0 + ZNTstoch) / ZA

    # stable PSI (lines 823-840: water & land identical)
    psim_s = psim_stable(zolza) - psim_stable(zolz0)
    psih_s = psih_stable(zolza) - psih_stable(zolzt)
    psim10_s = psim_stable(zol10) - psim_stable(zolz0)
    psih10_s = psih_stable(zol10) - psih_stable(zolz0)
    psih2_s = psih_stable(zol2) - psih_stable(zolz0)
    # unstable PSI (lines 907-922)
    psim_u = psim_unstable(zolza) - psim_unstable(zolz0)
    psih_u = psih_unstable(zolza) - psih_unstable(zolzt)
    psim10_u = psim_unstable(zol10) - psim_unstable(zolz0)
    psih10_u = psih_unstable(zol10) - psih_unstable(zolz0)
    psih2_u = psih_unstable(zol2) - psih_unstable(zolz0)

    zeros = np.zeros_like(BR)
    PSIM = np.where(stable, psim_s, np.where(unstable, psim_u, zeros))
    PSIH = np.where(stable, psih_s, np.where(unstable, psih_u, zeros))
    PSIM10 = np.where(stable, psim10_s, np.where(unstable, psim10_u, zeros))
    PSIH10 = np.where(stable, psih10_s, np.where(unstable, psih10_u, zeros))
    PSIH2 = np.where(stable, psih2_s, np.where(unstable, psih2_u, zeros))

    # caps ONLY in the unstable block (lines 931-935)
    PSIH = np.where(unstable, np.minimum(PSIH, 0.9 * GZ1OZt), PSIH)
    PSIM = np.where(unstable, np.minimum(PSIM, 0.9 * GZ1OZ0), PSIM)
    PSIH2 = np.where(unstable, np.minimum(PSIH2, 0.9 * GZ2OZt), PSIH2)
    PSIM10 = np.where(unstable, np.minimum(PSIM10, 0.9 * GZ10OZ0), PSIM10)
    PSIH10 = np.where(unstable, np.minimum(PSIH10, 0.9 * GZ10OZt), PSIH10)

    REGIME = np.where(stable & (BR > 0.2), 1.0, np.where(stable, 2.0, np.where(neutral, 3.0, 4.0)))
    RMOL = ZOL / ZA

    # --- ustar (module_sf_mynn.F:945-962) ---
    PSIX = GZ1OZ0 - PSIM
    PSIX10 = GZ10OZ0 - PSIM10
    UST = 0.5 * UST_in + 0.5 * KARMAN * WSPD / PSIX
    UST = np.where(is_land, np.maximum(UST, 0.005), UST)

    # --- resistances (recomputed in flux block, module_sf_mynn.F:1017-1025) ---
    PSIT = np.maximum(GZ1OZt - PSIH, 1.0)
    PSIT2 = np.maximum(GZ2OZt - PSIH2, 1.0)
    PSIQ = np.maximum(np.log((ZA + z_q) / z_q) - PSIH, 1.0)
    PSIQ2 = np.maximum(np.log((2.0 + z_q) / z_q) - PSIH2, 1.0)
    PSIQ10 = np.maximum(np.log((10.0 + z_q) / z_q) - PSIH10, 1.0)

    # MOL, qstar (lines 981-989). NOTE MOL uses GZ1OZt-PSIH from the FIRST resistance
    # block (line 972, identical numerics to flux block) and DTG = THV1D-THVGB.
    DTG_v = THV1D - THVGB
    MOL = KARMAN * DTG_v / PSIT / PRT
    DQG = (QVSH - QSFC) * 1000.0
    qstar = KARMAN * DQG / PSIQ / PRT

    # --- fluxes (module_sf_mynn.F:1051-1091) ---
    FLQC = RHO1D * MAVAIL * UST * KARMAN / PSIQ
    FLHC = RHO1D * CPM * UST * KARMAN / PSIT
    QFX = FLQC * (QSFCMR - QV1D)
    QFX = np.maximum(QFX, -0.02)
    LH = XLV * QFX
    HFX = FLHC * (THGB - TH1D)
    HFX = np.where(is_land, np.maximum(HFX, -250.0), HFX)
    CHS = UST * KARMAN / PSIT
    CH = FLHC / (CPM * RHO1D)
    CQS2 = UST * KARMAN / PSIQ2
    CHS2 = UST * KARMAN / PSIT2

    # --- diagnostics (module_sf_mynn.F:1108-1149) ---
    neutral_log = np.log(10.0 / ZNTstoch) / np.log(ZA / ZNTstoch)
    # za>=13 branch dominates Canary (~25.7 m). Faithful full branch:
    use_2nd = (ZA <= 7.0) & (ZA2 > 7.0) & (ZA2 < 13.0)
    use_log = ((ZA <= 7.0) & ~use_2nd) | ((ZA > 7.0) & (ZA < 13.0))
    ratio = np.where(use_log, neutral_log, PSIX10 / PSIX)
    U10 = np.where(use_2nd, U1D2, U1D * ratio)
    V10 = np.where(use_2nd, V1D2, V1D * ratio)

    DTG = TH1D - THGB
    TH2 = THGB + DTG * PSIT2 / PSIT
    th2_lin = THGB + 2.0 * (TH1D - THGB) / ZA
    warm = TH1D > THGB
    out_warm = warm & ((TH2 < THGB) | (TH2 > TH1D))
    out_cold = (~warm) & ((TH2 > THGB) | (TH2 < TH1D))
    TH2 = np.where(out_warm | out_cold, th2_lin, TH2)
    T2 = TH2 * (PSFC / 100.0) ** ROVCP
    Q2 = QSFCMR + (QV1D - QSFCMR) * PSIQ2 / PSIQ
    Q2 = np.maximum(Q2, np.minimum(QSFCMR, QV1D))
    Q2 = np.minimum(Q2, 1.05 * QV1D)

    return dict(
        ust=UST, mol=MOL, rmol=RMOL, zol=ZOL, regime=REGIME, psim=PSIM, psih=PSIH, br=BR,
        flhc=FLHC, flqc=FLQC, hfx=HFX, qfx=QFX, lh=LH, qsfc=QSFC, qgh=np.zeros_like(BR),
        chs=CHS, chs2=CHS2, cqs2=CQS2, ch=CH, wspd=WSPD, gz1oz0=GZ1OZ0,
        u10=U10, v10=V10, th2=TH2, t2=T2, q2=Q2, cpm=CPM, wstar=WSTAR, qstar=qstar, znt=ZNTstoch,
        z_t=z_t, psit=PSIT, psit2=PSIT2, psiq=PSIQ, psix=PSIX, restar=restar,
    )
