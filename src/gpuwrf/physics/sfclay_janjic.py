"""WRF-faithful JAX port of the Janjic Eta surface layer (``sf_sfclay_physics=2``).

This is a 1:1 transcription of the single-column path through the UNMODIFIED
pristine WRF ``phys/module_sf_myjsfc.F`` (``MYJSFC`` setup + ``SFCDIF`` Monin-
Obukhov solver + ``MYJSFCINIT`` integral-function table). It is a column
endpoint that plugs into the frozen ``PhysicsStepResult`` exactly like the
existing ``sfclay_revised_mm5`` adapter.

Pairing: ``sf_sfclay_physics=2`` MUST pair with ``bl_pbl_physics=2`` (MYJ PBL).
The exchange coefficients/fluxes produced here are the surface coupling the MYJ
PBL consumes; the dispatcher / namelist_check enforce the pairing.

Allocation-free, ``jax.vmap``-batchable over columns; the ``KZTM=10001`` PSIM/
PSIH lookup tables are built once at import as static device arrays. fp64
throughout for savepoint parity. NTSD>1 (warm-start) path, matching how the
operational loop calls the scheme after step 1.
"""

from __future__ import annotations

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import (
    PhysicsDiagnostics,
    PhysicsStepResult,
    PhysicsTendency,
)
from gpuwrf.physics import myj_constants as C


# --- MYJSFCINIT: Paulson(1970)/Holtslag-de Bruin(1988) integral functions -----
def _build_psi_tables() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(PSIM, PSIH)`` length-``KZTM`` tables. Sea (range 1) and land
    (range 2) ranges are identical in WRF (ZTMIN1==ZTMIN2, DZETA1==DZETA2)."""

    k = np.arange(C.KZTM, dtype=np.float64)
    zeta = C.SFC_ZTMIN1 + k * C.SFC_DZETA1
    unstable = zeta < 0.0
    x = np.sqrt(np.sqrt(1.0 - 16.0 * np.where(unstable, zeta, 0.0)))
    psim_u = -2.0 * np.log((x + 1.0) / 2.0) - np.log((x * x + 1.0) / 2.0) + 2.0 * np.arctan(x) - C.PIHF
    psih_u = -2.0 * np.log((x * x + 1.0) / 2.0)
    z = np.where(unstable, 0.0, zeta)
    psi_s = 0.7 * z + 0.75 * z * (6.0 - 0.35 * z) * np.exp(-0.35 * z)
    psim = np.where(unstable, psim_u, psi_s)
    psih = np.where(unstable, psih_u, psi_s)
    return psim, psih


_PSIM_NP, _PSIH_NP = _build_psi_tables()
_PSIM = jnp.asarray(_PSIM_NP, dtype=jnp.float64)
_PSIH = jnp.asarray(_PSIH_NP, dtype=jnp.float64)


def _interp_psi(table, zeta, ztmin, dzeta):
    """WRF table interpolation: ``RZ=(zeta-ZTMIN)/DZETA; K=INT(RZ);
    RDZT=RZ-REAL(K); K=MIN(MAX(K,0),KZTM2); PSI=(T[K+2]-T[K+1])*RDZT+T[K+1]``.
    Fortran ``T[K+1]`` -> 0-based ``table[K]`` (K is the truncated index)."""

    rz = (zeta - ztmin) / dzeta
    k = jnp.floor(rz).astype(jnp.int32)
    rdzt = rz - k.astype(jnp.float64)
    k = jnp.clip(k, 0, C.KZTM2)
    lo = table[k]
    hi = table[k + 1]
    return (hi - lo) * rdzt + lo


def _myj_pblh(q2, dz):
    """MYJSFC PBL height from the TKE column (bottom-up index 0 = surface).

    WRF (top-down ``K=1..LMH``) scans ``DO K=LMH-1,1,-1`` for the first level
    with ``2*TKE <= EPSQ2*FH`` and sets ``PBLH = ZHK(LPBL) - ZHK(LMH+1)``. In
    bottom-up terms: find the LOWEST layer ``j>=1`` whose ``2*q2[j]`` is at the
    floor; ``PBLH`` is the height of that layer's TOP interface. If no level
    floors, ``LPBL=1`` (top-down) and ``PBLH`` = full column-top height.
    """

    nz = q2.shape[0]
    # interface heights: iface[0]=0 (surface), iface[k]=sum(dz[:k])
    iface = jnp.concatenate([jnp.zeros((1,), q2.dtype), jnp.cumsum(dz)])
    thresh = C.EPSQ2 * C.FH
    floored = (2.0 * q2) <= thresh                      # bottom-up per layer
    # consider layers j=1..nz-1 (WRF excludes the bottom layer and the top)
    idx = jnp.arange(nz)
    eligible = floored & (idx >= 1) & (idx <= nz - 1)
    any_floor = jnp.any(eligible)
    # lowest eligible layer (smallest bottom-up index)
    first_j = jnp.argmax(eligible.astype(jnp.int32))    # first True; 0 if none
    pblh_floor = iface[first_j + 1]                     # top interface of layer
    pblh_full = iface[nz]                               # full column top
    return jnp.where(any_floor, pblh_floor, pblh_full)


def _sfcdif_column(
    *,
    seamask, ths, qs_in, psfc, tz0, tsk, thz0_in, qz0_in, uz0_in, vz0_in,
    ustar_in, z0_in, z0base, akhs_in, akms_in, ulow, vlow, tlow, thlow,
    thelow, qlow, cwmlow, zsl, plow, pblh, wetm,
):
    """1:1 transcription of WRF ``SFCDIF`` (NTSD>1 path) for one column.

    Returns a dict of every SFCDIF output: exchange coeffs (AKHS/AKMS),
    surface roughness Z0, fluxes (HFX/QFX/LH/FLHC/FLQC), updated INOUT surface
    state (USTAR/THZ0/QZ0/UZ0/VZ0/QSFC), RMOL/RIB, and 2m/10m diagnostics.
    The land/sea diagnostic block reads the last iteration's geometry directly.
    """

    rdz = 1.0 / zsl
    cxchl = C.EXCML * rdz
    cxchs = C.EXCMS * rdz
    btgx = C.G / thlow
    elfc = C.VKARMAN * btgx
    btgh = jnp.where(pblh > 1000.0, btgx * pblh, btgx * 1000.0)
    is_sea = seamask > 0.5
    cxch = jnp.where(is_sea, cxchs, cxchl)
    ztmin = jnp.where(is_sea, C.SFC_ZTMIN1, C.SFC_ZTMIN2)
    ztmax = jnp.where(is_sea, C.SFC_ZTMAX1, C.SFC_ZTMAX2)
    fhh = jnp.where(is_sea, C.FH01, C.FH02)

    # CZIL/ZILFC/ZZIL for land are constant across iterations; precompute with
    # the land-branch RIB (which uses THZ0=THS, QZ0=QS, UZ0=VZ0=0).
    thz0_land = ths
    qz0_land = qs_in
    tem_l = (tlow + tz0) * 0.5
    thm_l = (thelow + thz0_land) * 0.5
    a_l = thm_l * C.P608
    b_l = (C.ELOCP / tem_l - 1.0 - C.P608) * thm_l
    dthv_l = ((thelow - thz0_land) * ((qlow + qz0_land + cwmlow) * (0.5 * C.P608) + 1.0)
              + (qlow - qz0_land + cwmlow) * a_l + cwmlow * b_l)
    du2_l = jnp.maximum((ulow - 0.0) ** 2 + (vlow - 0.0) ** 2, C.EPSU2)
    rib_l = btgx * dthv_l * zsl / du2_l
    zu_l = z0_in
    zslu_l = zsl + zu_l
    rzsu_l = zslu_l / zu_l
    rlogu_l = jnp.log(rzsu_l)
    zslt_l = zsl + zu_l
    zilfc = -0.1 * C.VKARMAN * C.SQVISC
    czetmax = 10.0
    zzil = jnp.where(
        dthv_l > 0.0,
        jnp.where(rib_l < C.RIC,
                  zilfc * (1.0 + (rib_l / C.RIC) ** 2 * czetmax),
                  zilfc * (1.0 + czetmax)),
        zilfc,
    )

    def body(carry, _):
        (ustar, uz0, vz0, thz0, qz0, qs, akhs, akms, z0) = carry

        # ---- SEA geometry (Janjic viscous sublayer, NTSD>1 path) ----
        z0_sea = jnp.maximum(C.USTFC * ustar * ustar, 1.59e-5)
        # below USTR
        zu_a = C.FZU1 * jnp.sqrt(jnp.sqrt(z0_sea * ustar * C.RVISC)) / ustar
        wght = akms * zu_a * C.RVISC
        rwgh = wght / (wght + 1.0)
        uz0_a = (ulow * rwgh + uz0) * 0.5
        vz0_a = (vlow * rwgh + vz0) * 0.5
        zt_a = C.FZT1 * zu_a
        zq_a = C.FZQ1 * zt_a
        wghtt_a = akhs * zt_a * C.RTVISC
        wghtq_a = akhs * zq_a * C.RQVISC
        thz0_a = ((wghtt_a * thlow + ths) / (wghtt_a + 1.0) + thz0) * 0.5
        qz0_a = ((wghtq_a * qlow + qs) / (wghtq_a + 1.0) + qz0) * 0.5
        # USTR <= ustar < USTC
        zu_b = z0_sea
        zt_b = C.FZT2 * jnp.sqrt(jnp.sqrt(z0_sea * ustar * C.RVISC)) / ustar
        zq_b = C.FZQ2 * zt_b
        wghtt_b = akhs * zt_b * C.RTVISC
        wghtq_b = akhs * zq_b * C.RQVISC
        thz0_b = ((wghtt_b * thlow + ths) / (wghtt_b + 1.0) + thz0) * 0.5
        qz0_b = ((wghtq_b * qlow + qs) / (wghtq_b + 1.0) + qz0) * 0.5
        # ustar >= USTC
        zu_c = z0_sea
        zt_c = z0_sea
        thz0_c = ths
        qz0_c = qs

        below_ustr = ustar < C.USTR
        below_ustc = ustar < C.USTC
        sel_a = below_ustc & below_ustr
        sel_b = below_ustc & (~below_ustr)

        zu_sea = jnp.where(sel_a, zu_a, jnp.where(sel_b, zu_b, zu_c))
        zt_sea = jnp.where(sel_a, zt_a, jnp.where(sel_b, zt_b, zt_c))
        uz0_sea = jnp.where(sel_a, uz0_a, jnp.zeros_like(uz0))
        vz0_sea = jnp.where(sel_a, vz0_a, jnp.zeros_like(vz0))
        thz0_sea = jnp.where(sel_a, thz0_a, jnp.where(sel_b, thz0_b, thz0_c))
        qz0_sea = jnp.where(sel_a, qz0_a, jnp.where(sel_b, qz0_b, qz0_c))

        tem_s = (tlow + tz0) * 0.5
        thm_s = (thelow + thz0_sea) * 0.5
        a_s = thm_s * C.P608
        b_s = (C.ELOCP / tem_s - 1.0 - C.P608) * thm_s
        dthv_s = ((thelow - thz0_sea) * ((qlow + qz0_sea + cwmlow) * (0.5 * C.P608) + 1.0)
                  + (qlow - qz0_sea + cwmlow) * a_s + cwmlow * b_s)
        du2_s = jnp.maximum((ulow - uz0_sea) ** 2 + (vlow - vz0_sea) ** 2, C.EPSU2)
        rib_s = btgx * dthv_s * zsl / du2_s
        rzsu_s = (zsl + zu_sea) / zu_sea
        rzst_s = (zsl + zt_sea) / zt_sea
        rlogu_s = jnp.log(rzsu_s)
        rlogt_s = jnp.log(rzst_s)

        # ---- LAND geometry (Zilitinkevich ZT) ----
        zt_land = jnp.maximum(jnp.exp(zzil * jnp.sqrt(ustar * z0base)) * z0base, C.EPSZT)
        rzst_l = zslt_l / zt_land
        rlogt_l = jnp.log(rzst_l)

        # ---- branch-selected geometry ----
        zu = jnp.where(is_sea, zu_sea, zu_l)
        zt = jnp.where(is_sea, zt_sea, zt_land)
        uz0_new = jnp.where(is_sea, uz0_sea, jnp.zeros_like(uz0))
        vz0_new = jnp.where(is_sea, vz0_sea, jnp.zeros_like(vz0))
        thz0_new = jnp.where(is_sea, thz0_sea, thz0_land)
        qz0_new = jnp.where(is_sea, qz0_sea, qz0_land)
        dthv = jnp.where(is_sea, dthv_s, dthv_l)
        du2 = jnp.where(is_sea, du2_s, du2_l)
        rib = jnp.where(is_sea, rib_s, rib_l)
        rzsu = jnp.where(is_sea, rzsu_s, rzsu_l)
        rzst = jnp.where(is_sea, rzst_s, rzst_l)
        rlogu = jnp.where(is_sea, rlogu_s, rlogu_l)
        rlogt = jnp.where(is_sea, rlogt_s, rlogt_l)
        zslu = zsl + zu
        # WRF's land branch has u,v,t at the same nominal level for the
        # similarity lookup: ZSLT=ZSL+ZU, even though the land thermal roughness
        # length ZT is Zilitinkevich-adjusted.
        zslt = jnp.where(is_sea, zsl + zt, zslt_l)

        rlmo = elfc * akhs * dthv / ustar ** 3
        zetalu = jnp.clip(zslu * rlmo, ztmin, ztmax)
        zetalt = jnp.clip(zslt * rlmo, ztmin, ztmax)
        zetau = jnp.clip(zu * rlmo, ztmin / rzsu, ztmax / rzsu)
        zetat = jnp.clip(zt * rlmo, ztmin / rzst, ztmax / rzst)

        # tables are identical for sea/land ranges; use range-1 constants
        psmz = _interp_psi(_PSIM, zetau, C.SFC_ZTMIN1, C.SFC_DZETA1)
        psmzl = _interp_psi(_PSIM, zetalu, C.SFC_ZTMIN1, C.SFC_DZETA1)
        simm = psmzl - psmz + rlogu
        pshz = _interp_psi(_PSIH, zetat, C.SFC_ZTMIN1, C.SFC_DZETA1)
        pshzl = _interp_psi(_PSIH, zetalt, C.SFC_ZTMIN1, C.SFC_DZETA1)
        simh = (pshzl - pshz + rlogt) * fhh

        ustark = ustar * C.VKARMAN
        akms_new = jnp.maximum(ustark / simm, cxch)
        akhs_new = jnp.maximum(ustark / simh, cxch)
        wstar2 = jnp.where(dthv <= 0.0, C.WWST2 * jnp.abs(btgh * akhs_new * dthv) ** (2.0 / 3.0), 0.0)
        ustar_new = jnp.maximum(jnp.sqrt(akms_new * jnp.sqrt(du2 + wstar2)), C.EPSUST)

        # QS only changes over sea via the viscous-sublayer THZ0/QZ0 update; the
        # prognostic land QS stays fixed (handled outside). Pass qs through.
        carry_out = (ustar_new, uz0_new, vz0_new, thz0_new, qz0_new, qs,
                     akhs_new, akms_new, z0_sea)
        # diagnostic geometry (last-iteration values are consumed post-loop).
        # WRF computes USTARK=USTAR*VKARMAN INSIDE the loop and never recomputes
        # it after the final USTAR update, so the 2m/10m AKMS10/AKHS02/AKHS10
        # use the LAST-ITERATION-INPUT ustar (here ``ustark`` below), not the
        # converged USTAR. UZ0/VZ0 likewise use the converged carry-out values.
        diag = (zu, zt, rlmo, psmz, pshz, rib, du2, wstar2, ustark)
        return carry_out, diag

    carry0 = (ustar_in, uz0_in, vz0_in, thz0_in, qz0_in, qs_in,
              akhs_in, akms_in, z0_in)
    carry, hist = jax.lax.scan(body, carry0, None, length=C.SFC_ITRMX)
    (ustar, uz0, vz0, thz0, qz0, qs, akhs, akms, z0_sea_final) = carry
    # Z0 is updated only in the SEA branch; land roughness is unchanged.
    z0 = jnp.where(is_sea, z0_sea_final, z0_in)
    zu = hist[0][-1]; zt = hist[1][-1]; rlmo = hist[2][-1]
    psmz = hist[3][-1]; pshz = hist[4][-1]; rib = hist[5][-1]
    ustark = hist[8][-1]   # USTAR*VKARMAN from the last iteration (NOT converged)

    # ---- DIAGNOSTIC BLOCK (2m/10m + WRF driver arrays), CT=0 ----
    umflx = akms * (ulow - uz0)
    vmflx = akms * (vlow - vz0)
    hsflx = akhs * (thlow - thz0)
    hlflx = akhs * (qlow - qz0)

    zu10 = zu + 10.0
    zt02 = zt + 2.0
    zt10 = zt + 10.0
    rlnu10 = jnp.log(zu10 / zu)
    rlnt02 = jnp.log(zt02 / zt)
    rlnt10 = jnp.log(zt10 / zt)
    ztau10 = jnp.clip(zu10 * rlmo, ztmin, ztmax)
    ztat02 = jnp.clip(zt02 * rlmo, ztmin, ztmax)
    ztat10 = jnp.clip(zt10 * rlmo, ztmin, ztmax)

    psm10 = _interp_psi(_PSIM, ztau10, C.SFC_ZTMIN1, C.SFC_DZETA1)
    simm10 = psm10 - psmz + rlnu10
    psh02 = _interp_psi(_PSIH, ztat02, C.SFC_ZTMIN1, C.SFC_DZETA1)
    simh02 = (psh02 - pshz + rlnt02) * fhh
    psh10 = _interp_psi(_PSIH, ztat10, C.SFC_ZTMIN1, C.SFC_DZETA1)
    simh10 = (psh10 - pshz + rlnt10) * fhh

    akms10 = jnp.maximum(ustark / simm10, cxch)
    akhs02 = jnp.maximum(ustark / simh02, cxch)
    akhs10 = jnp.maximum(ustark / simh10, cxch)

    u10 = umflx / akms10 + uz0
    v10 = vmflx / akms10 + vz0
    th02 = hsflx / akhs02 + thz0
    bracket02 = (((thlow > thz0) & ((th02 < thz0) | (th02 > thlow)))
                 | ((thlow < thz0) & ((th02 > thz0) | (th02 < thlow))))
    th02 = jnp.where(bracket02, thz0 + 2.0 * rdz * (thlow - thz0), th02)
    th10 = hsflx / akhs10 + thz0
    bracket10 = (((thlow > thz0) & ((th10 < thz0) | (th10 > thlow)))
                 | ((thlow < thz0) & ((th10 > thz0) | (th10 < thlow))))
    th10 = jnp.where(bracket10, thz0 + 10.0 * rdz * (thlow - thz0), th10)
    q02 = hlflx / akhs02 + qz0
    q10 = hlflx / akhs10 + qz0
    pshltr = psfc * jnp.exp(-0.068283 / tlow)

    # equivalent-Z0 10m winds (land only; sea keeps U10/V10)
    zuuz = jnp.minimum(zu * 0.50, 0.18)
    zu_e = jnp.maximum(zu * 0.35, zuuz)
    zu10_e = zu_e + 10.0
    rlnu10_e = jnp.log(zu10_e / zu_e)
    ztau10_e = jnp.clip(zu10_e * rlmo, C.SFC_ZTMIN2, C.SFC_ZTMAX2)
    psm10_e = _interp_psi(_PSIM, ztau10_e, C.SFC_ZTMIN2, C.SFC_DZETA2)
    simm10_e = psm10_e - psmz + rlnu10_e
    ekms10 = jnp.maximum(ustark / simm10_e, cxchl)
    u10e = jnp.where(is_sea, u10, umflx / ekms10 + uz0)
    v10e = jnp.where(is_sea, v10, vmflx / ekms10 + vz0)

    rlow = plow / (C.R_D * tlow)
    chs = akhs
    chs2 = akhs02
    cqs2 = akhs02
    hfx = -rlow * C.CP * hsflx
    qfx = -rlow * hlflx * wetm
    flx_lh = C.XLV * qfx
    flhc = rlow * C.CP * akhs
    flqc = rlow * akhs * wetm
    qgh = ((1.0 - seamask) * C.PQ0 + seamask * C.PQ0SEA) / plow * jnp.exp(C.A2S * (tlow - C.A3S) / (tlow - C.A4S))
    qgh = qgh / (1.0 - qgh)
    cpm = C.CP * (1.0 + 0.8 * qlow)

    qs_sea = C.PQ0SEA / psfc * jnp.exp(C.A2S * (tsk - C.A3S) / (tsk - C.A4S))
    qs_sea = qs_sea / (1.0 - qs_sea)
    qsfc = jnp.where(is_sea, qs_sea, qs)

    return {
        "ustar": ustar, "znt": z0, "akhs": akhs, "akms": akms, "rmol": rlmo,
        "rib": rib, "chs": chs, "chs2": chs2, "cqs2": cqs2, "hfx": hfx,
        "qfx": qfx, "flx_lh": flx_lh, "flhc": flhc, "flqc": flqc, "qgh": qgh,
        "cpm": cpm, "ct": jnp.zeros_like(ustar), "qsfc": qsfc, "thz0": thz0,
        "qz0": qz0, "uz0": uz0, "vz0": vz0, "u10": u10, "v10": v10, "th02": th02,
        "th10": th10, "tshltr": th02, "q02": q02, "qshltr": q02, "q10": q10,
        "pshltr": pshltr, "u10e": u10e, "v10e": v10e,
        # also expose AKHS02/10 for PBL diagnostics if needed downstream
        "akhs02": akhs02, "akhs10": akhs10, "akms10": akms10,
    }


def myjsfc_column(
    *, u, v, temperature, theta, qv, qc, p_mid, dz, q2, tsk, xland, z0base,
    psfc, znt, ustar, mavail, dt=60.0, qsfc=None, thz0=None, qz0=None,
    uz0=0.0, vz0=0.0, pblh=1000.0,
):
    """Run the Janjic surface layer on one column; returns the SFCDIF dict.

    ``u/v/temperature/theta/qv/qc/p_mid/dz/q2`` are length-``nz`` columns ordered
    bottom-up (index 0 = lowest model layer). ``q2`` is TKE (m^2/s^2); MYJSFC
    converts to ``2*q2``. Surface scalars are floats/0-d arrays. ``thz0/qz0/qsfc``
    default to lowest-layer values (warm-start seed) when not supplied.
    """

    f = lambda a: jnp.asarray(a, jnp.float64)
    u = f(u); v = f(v); temperature = f(temperature); theta = f(theta)
    qv = f(qv); qc = f(qc); p_mid = f(p_mid); dz = f(dz); q2 = f(q2)

    # --- PBL height from the TKE profile (MYJSFC, used by the Beljaars BTGH) ---
    # WRF (top-down) scans from LMH-1 upward for the first level with 2*TKE <=
    # EPSQ2*FH; PBLH = ZHK(LPBL) - ZHK(LMH+1). In bottom-up indexing (0=bottom)
    # the interface heights are the cumulative DZ from the surface, and we take
    # the LOWEST interface (closest to ground) above which 2*TKE first drops to
    # the floor. EPSQ2*FH with EPSQ2=0.2 floors at the WRF threshold.
    pblh = _myj_pblh(q2, dz)

    # lowest model layer (LMH path; LOWLYR=1 => bottom layer)
    tlow = temperature[0]
    thlow = theta[0]
    ratiomx = qv[0]
    qlow = ratiomx / (1.0 + ratiomx)
    cwmlow = qc[0]
    thelow = (cwmlow * (-C.ELOCP / tlow) + 1.0) * thlow
    ulow = u[0]
    vlow = v[0]
    plow = p_mid[0]
    zsl = dz[0] * 0.5
    psfc_v = f(psfc)
    apesfc = (psfc_v / C.P1000MB) ** C.CAPA
    thsk = f(tsk) / apesfc
    seamask = f(xland) - 1.0
    tz0 = (f(thz0) if thz0 is not None else thlow) * apesfc

    # QSFC is a WATER-VAPOR MIXING RATIO (WRF stores it as a mixing ratio); the
    # warm-start seed = lowest-layer mixing ratio qv[0]. QZ0 is a SPECIFIC
    # humidity, seeded from qlow. The land path uses QZ0=QS=QSFC (mixing ratio).
    thz0_in = f(thz0) if thz0 is not None else thlow
    qz0_in = f(qz0) if qz0 is not None else qlow
    qsfc_in = f(qsfc) if qsfc is not None else qv[0]

    out = _sfcdif_column(
        seamask=seamask, ths=thsk, qs_in=qsfc_in, psfc=psfc_v, tz0=tz0,
        tsk=f(tsk), thz0_in=thz0_in, qz0_in=qz0_in, ustar_in=f(ustar),
        z0_in=f(znt), z0base=f(z0base), akhs_in=f(0.0), akms_in=f(0.0),
        uz0_in=f(uz0), vz0_in=f(vz0),
        ulow=ulow, vlow=vlow, tlow=tlow, thlow=thlow, thelow=thelow,
        qlow=qlow, cwmlow=cwmlow, zsl=zsl, plow=plow, pblh=pblh,
        wetm=f(mavail),
    )

    # --- MYJSFC post-SFCDIF: remove 2m/10m supersaturation + convert QSHLTR/Q10
    # from specific humidity to mixing ratio (lines 324-349 of module_sf_myjsfc.F)
    rapa = apesfc
    th02p = out["tshltr"]
    th10p = out["th10"]
    rapa02 = rapa - C.GOCP02 / th02p
    rapa10 = rapa - C.GOCP10 / th10p
    t02p = th02p * rapa02
    t10p = th10p * rapa10
    p02p = (rapa02 ** C.RCAP) * C.P1000MB
    p10p = (rapa10 ** C.RCAP) * C.P1000MB
    qs02 = C.PQ0 / p02p * jnp.exp(C.A2 * (t02p - C.A3) / (t02p - C.A4))
    qs10 = C.PQ0 / p10p * jnp.exp(C.A2 * (t10p - C.A3) / (t10p - C.A4))
    qshltr = jnp.minimum(out["qshltr"], qs02)
    q10 = jnp.minimum(out["q10"], qs10)
    out["qshltr"] = qshltr
    out["q10"] = q10
    out["q02"] = qshltr / (1.0 - qshltr)        # specific humidity -> mixing ratio
    out["t02"] = out["th02"] * apesfc
    out["pblh"] = pblh
    return out


def step_janjic_sfclay_column(
    u, v, temperature, qv, pressure, exner, dz, q2, *, tsk, xland, z0base, psfc,
    znt, ustar, mavail, theta=None, qc=None, dt=60.0, qsfc=None, thz0=None,
    qz0=None, uz0=0.0, vz0=0.0, pblh=1000.0,
) -> PhysicsStepResult:
    """Janjic Eta surface-layer column endpoint -> frozen ``PhysicsStepResult``.

    Mirrors the ``step_sfclay_revised_mm5_column`` calling convention: profile
    columns ordered bottom-up, ``exner`` is the mass-level Exner column, surface
    fields are scalars. ``theta`` defaults to ``temperature/exner`` and ``qc``
    to zero when not supplied.
    """

    f = lambda a: jnp.asarray(a, jnp.float64)
    temperature = f(temperature)
    exner = f(exner)
    theta_c = f(theta) if theta is not None else temperature / exner
    qc_c = f(qc) if qc is not None else jnp.zeros_like(temperature)

    out = myjsfc_column(
        u=u, v=v, temperature=temperature, theta=theta_c, qv=qv, qc=qc_c,
        p_mid=pressure, dz=dz, q2=q2, tsk=tsk, xland=xland, z0base=z0base,
        psfc=psfc, znt=znt, ustar=ustar, mavail=mavail, dt=dt, qsfc=qsfc,
        thz0=thz0, qz0=qz0, uz0=uz0, vz0=vz0, pblh=pblh,
    )

    # theta_flux/qv_flux (kinematic) consistent with the frozen sfclay contract:
    # HFX = -rho*cp*akhs*(thlow-thz0); theta_flux = HFX/(rho*cp).
    rhosfc = out["flhc"] / (C.CP * jnp.maximum(out["akhs"], 1.0e-30))
    theta_flux = out["hfx"] / jnp.maximum(rhosfc * C.CP, 1.0e-12)
    qv_flux = out["qfx"] / jnp.maximum(rhosfc, 1.0e-12)
    wspd = jnp.sqrt(f(u)[0] ** 2 + f(v)[0] ** 2)
    tau_u = out["akms"] * f(u)[0]
    tau_v = out["akms"] * f(v)[0]
    fltv = theta_flux  # placeholder consistent with contract leaf set

    tendency = PhysicsTendency(
        state_replacements={
            "ustar": out["ustar"],
            "theta_flux": theta_flux,
            "qv_flux": qv_flux,
            "tau_u": tau_u,
            "tau_v": tau_v,
            "rhosfc": rhosfc,
            "fltv": fltv,
        },
        diagnostics={
            "HFX": out["hfx"], "QFX": out["qfx"], "LH": out["flx_lh"],
            "T2": out["t02"], "TH2": out["th02"],
            "Q2": out["q02"], "U10": out["u10"], "V10": out["v10"],
            "ZNT": out["znt"], "UST": out["ustar"],
        },
    )
    tendency.validate_keys()
    diagnostics = PhysicsDiagnostics(
        surface_layer={
            "UST": out["ustar"], "ZNT": out["znt"], "AKHS": out["akhs"],
            "AKMS": out["akms"], "RMOL": out["rmol"], "RIB": out["rib"],
            "CHS": out["chs"], "CHS2": out["chs2"], "CQS2": out["cqs2"],
            "HFX": out["hfx"], "QFX": out["qfx"], "LH": out["flx_lh"],
            "FLHC": out["flhc"], "FLQC": out["flqc"], "QGH": out["qgh"],
            "CPM": out["cpm"], "QSFC": out["qsfc"], "THZ0": out["thz0"],
            "QZ0": out["qz0"], "UZ0": out["uz0"], "VZ0": out["vz0"],
            "PBLH": out["pblh"], "U10": out["u10"], "V10": out["v10"], "T02": out["t02"],
            "TH02": out["th02"], "TSHLTR": out["tshltr"], "TH10": out["th10"],
            "Q02": out["q02"], "QSHLTR": out["qshltr"], "Q10": out["q10"],
            "PSHLTR": out["pshltr"], "U10E": out["u10e"], "V10E": out["v10e"],
            "CT": out["ct"],
        }
    )
    return PhysicsStepResult(tendency=tendency, diagnostics=diagnostics)


__all__ = [
    "myjsfc_column",
    "step_janjic_sfclay_column",
]
