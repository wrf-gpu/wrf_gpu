"""WRF-faithful single-column MYJ PBL port (``bl_pbl_physics=2``).

This module transcribes the ARW single-column path through unmodified WRF
``phys/module_bl_myjpbl.F`` for the v0.6.0 savepoint gate.  It expects the
paired Janjic Eta surface layer (``sf_sfclay_physics=2``) to have already
produced the MYJ coupling scalars: ``USTAR``, ``AKHS``, ``AKMS``, ``THZ0``,
``QZ0``, ``UZ0``, ``VZ0``, ``QSFC``, ``CHKLOWQ``, and ``ELFLX``.

Public inputs are bottom-up WRF mass-level columns.  The implementation uses
Fortran-like 1-based top-down work arrays internally because MYJ indexes from
model top to surface; this avoids the usual off-by-one mistakes in ``LPBL``,
``EL_MYJ``, and ``EXCH_H``.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import (
    PhysicsCarry,
    PhysicsDiagnostics,
    PhysicsStepResult,
    PhysicsTendency,
)
from gpuwrf.physics import myj_constants as C


configure_jax_x64()


def _as1d(value, *, length: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} length {arr.shape[0]} does not match {length}")
    return arr.copy()


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def _one_based_top_down(bottom_up: np.ndarray) -> np.ndarray:
    """Return a 1-based work array, top level at index 1."""

    n = bottom_up.shape[0]
    out = np.zeros(n + 2, dtype=np.float64)
    out[1 : n + 1] = bottom_up[::-1]
    return out


def _one_based_interfaces(dz_bottom_up: np.ndarray, ht: float) -> np.ndarray:
    """Fortran ``ZINT``/``ZHK`` interface heights, top-down and 1-based."""

    bottom = np.concatenate(([float(ht)], float(ht) + np.cumsum(dz_bottom_up, dtype=np.float64)))
    z_top_down = bottom[::-1]
    out = np.zeros(dz_bottom_up.shape[0] + 2, dtype=np.float64)
    out[1 : dz_bottom_up.shape[0] + 2] = z_top_down
    return out


def _mixlen(
    lmh: int,
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    the: np.ndarray,
    q: np.ndarray,
    cwm: np.ndarray,
    q2: np.ndarray,
    z: np.ndarray,
    ct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, float, float]:
    """Transcription of WRF ``MIXLEN``."""

    gm = np.zeros(lmh + 2, dtype=np.float64)
    gh = np.zeros(lmh + 2, dtype=np.float64)
    el = np.zeros(lmh + 2, dtype=np.float64)
    q1 = np.zeros(lmh + 2, dtype=np.float64)
    dth = np.zeros(lmh + 2, dtype=np.float64)
    elm = np.zeros(lmh + 2, dtype=np.float64)
    rel = np.zeros(lmh + 2, dtype=np.float64)

    lpbl = lmh
    found = False
    for k in range(lmh - 1, 0, -1):
        if q2[k] <= C.EPSQ2 * C.FH:
            lpbl = k
            found = True
            break
    if not found:
        lpbl = 1
    pblh = z[lpbl + 1] - z[lmh + 1]

    for k in range(1, lmh):
        dth[k] = the[k] - the[k + 1]

    for k in range(lmh - 2, 0, -1):
        if dth[k] > 0.0 and dth[k + 1] <= 0.0:
            dth[k] = dth[k] + ct
            break
    ct_out = 0.0

    for k in range(1, lmh):
        rdz = 2.0 / (z[k] - z[k + 2])
        gml = ((u[k] - u[k + 1]) ** 2 + (v[k] - v[k + 1]) ** 2) * rdz * rdz
        gm[k] = max(gml, C.EPSGM)

        tem = (t[k] + t[k + 1]) * 0.5
        thm = (the[k] + the[k + 1]) * 0.5
        a = thm * C.P608
        b = (C.ELOCP / tem - 1.0 - C.P608) * thm
        ghl = (
            dth[k] * ((q[k] + q[k + 1] + cwm[k] + cwm[k + 1]) * (0.5 * C.P608) + 1.0)
            + (q[k] - q[k + 1] + cwm[k] - cwm[k + 1]) * a
            + (cwm[k] - cwm[k + 1]) * b
        ) * rdz
        if abs(ghl) <= C.EPSGH:
            ghl = C.EPSGH
        gh[k] = ghl

    lmxl = lmh
    for k in range(1, lmh):
        gml = gm[k]
        ghl = gh[k]
        if ghl >= C.EPSGH:
            if gml / ghl <= C.REQU:
                elm[k] = C.EPSL
                lmxl = k
            else:
                aubr = (C.AUBM * gml + C.AUBH * ghl) * ghl
                bubr = C.BUBM * gml + C.BUBH * ghl
                qol2st = (-0.5 * bubr + np.sqrt(bubr * bubr * 0.25 - aubr * C.CUBR)) * C.RCUBR
                eloq2x = 1.0 / qol2st
                elm[k] = max(np.sqrt(eloq2x * q2[k]), C.EPSL)
        else:
            aden = (C.ADNM * gml + C.ADNH * ghl) * ghl
            bden = C.BDNM * gml + C.BDNH * ghl
            qol2un = -0.5 * bden + np.sqrt(bden * bden * 0.25 - aden)
            eloq2x = 1.0 / (qol2un + C.EPSRU)
            elm[k] = max(np.sqrt(eloq2x * q2[k]), C.EPSL)

    if elm[lmh - 1] == C.EPSL:
        lmxl = lmh
    mixht = z[lmxl] - z[lmh + 1]

    for k in range(lpbl, lmh + 1):
        q1[k] = np.sqrt(q2[k])

    szq = 0.0
    sq = 0.0
    for k in range(1, lmh):
        qdzl = (q1[k] + q1[k + 1]) * (z[k + 1] - z[k + 2])
        szq = (z[k + 1] + z[k + 2] - z[lmh + 1] - z[lmh + 1]) * qdzl + szq
        sq = qdzl + sq

    el0 = min(C.ALPH * szq * 0.5 / sq, C.EL0MAX)
    el0 = max(el0, C.EL0MIN)

    lpblm = max(lpbl - 1, 1)
    for k in range(1, lpblm + 1):
        el[k] = min((z[k] - z[k + 2]) * C.ELFC, elm[k])
        rel[k] = el[k] / elm[k]

    if lpbl < lmh:
        for k in range(lpbl, lmh):
            vkrmz = (z[k + 1] - z[lmh + 1]) * C.VKARMAN
            el[k] = min(vkrmz / (vkrmz / el0 + 1.0), elm[k])
            rel[k] = el[k] / elm[k]

    for k in range(lpbl + 1, lmh - 1):
        srel = min(((rel[k - 1] + rel[k + 1]) * 0.5 + rel[k]) * 0.5, rel[k])
        el[k] = max(srel * elm[k], C.EPSL)

    return gm, gh, el, pblh, lpbl, lmxl, ct_out, mixht


def _prodq2(lmh: int, dtturbl: float, ustar: float, gm: np.ndarray, gh: np.ndarray, el: np.ndarray, q2: np.ndarray) -> None:
    """Transcription of WRF ``PRODQ2``; mutates ``el`` and ``q2``."""

    for k in range(1, lmh):
        gml = gm[k]
        ghl = gh[k]
        aequ = (C.AEQM * gml + C.AEQH * ghl) * ghl
        bequ = C.BEQM * gml + C.BEQH * ghl
        eqol2 = -0.5 * bequ + np.sqrt(bequ * bequ * 0.25 - aequ)

        if (
            (gml + ghl * ghl <= C.EPSTRB)
            or (ghl >= C.EPSGH and gml / ghl <= C.REQU)
            or (eqol2 <= C.EPS2)
        ):
            q2[k] = C.EPSQ2
            el[k] = C.EPSL
            continue

        anum = (C.ANMM * gml + C.ANMH * ghl) * ghl
        bnum = C.BNMM * gml + C.BNMH * ghl
        aden = (C.ADNM * gml + C.ADNH * ghl) * ghl
        bden = C.BDNM * gml + C.BDNH * ghl
        cden = 1.0
        arhs = -(anum * bden - bnum * aden) * 2.0
        brhs = -anum * 4.0
        crhs = -bnum * 2.0
        dloq1 = el[k] / np.sqrt(q2[k])

        eloq21 = 1.0 / eqol2
        eloq11 = np.sqrt(eloq21)
        eloq31 = eloq21 * eloq11
        eloq41 = eloq21 * eloq21
        eloq51 = eloq21 * eloq31
        rden1 = 1.0 / (aden * eloq41 + bden * eloq21 + cden)
        rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11) * rden1 * rden1
        eloq12 = eloq11 + (dloq1 - eloq11) * np.exp(rhsp1 * dtturbl)
        eloq12 = max(eloq12, C.EPS1)

        eloq22 = eloq12 * eloq12
        eloq32 = eloq22 * eloq12
        eloq42 = eloq22 * eloq22
        eloq52 = eloq22 * eloq32
        rden2 = 1.0 / (aden * eloq42 + bden * eloq22 + cden)
        rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + C.RB1
        rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12) * rden2 * rden2
        rhst2 = rhs2 / rhsp2
        eloq13 = eloq12 - rhst2 + (rhst2 + dloq1 - eloq12) * np.exp(rhsp2 * dtturbl)
        eloq13 = max(eloq13, C.EPS1)

        eloqn = eloq13
        if eloqn > C.EPS1:
            q2[k] = el[k] * el[k] / (eloqn * eloqn)
            q2[k] = max(q2[k], C.EPSQ2)
            if q2[k] == C.EPSQ2:
                el[k] = C.EPSL
        else:
            q2[k] = C.EPSQ2
            el[k] = C.EPSL

    q2[lmh] = max(C.B1 ** (2.0 / 3.0) * ustar * ustar, C.EPSQ2)


def _difcof(
    lmh: int,
    gm: np.ndarray,
    gh: np.ndarray,
    el: np.ndarray,
    q2: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transcription of WRF ``DIFCOF``."""

    akm = np.zeros(lmh + 2, dtype=np.float64)
    akh = np.zeros(lmh + 2, dtype=np.float64)
    for k in range(1, lmh):
        ell = el[k]
        eloq2 = ell * ell / q2[k]
        eloq4 = eloq2 * eloq2
        gml = gm[k]
        ghl = gh[k]
        aden = (C.ADNM * gml + C.ADNH * ghl) * ghl
        bden = C.BDNM * gml + C.BDNH * ghl
        cden = 1.0
        besm = C.BSMH * ghl
        besh = C.BSHM * gml + C.BSHH * ghl
        rden = 1.0 / (aden * eloq4 + bden * eloq2 + cden)
        esm = (besm * eloq2 + C.CESM) * rden
        esh = (besh * eloq2 + C.CESH) * rden
        rdz = 2.0 / (z[k] - z[k + 2])
        q1l = np.sqrt(q2[k])
        elqdz = ell * q1l * rdz
        akm[k] = elqdz * esm
        akh[k] = elqdz * esh
    return akm, akh


def _vdifq(lmh: int, dtdif: float, q2: np.ndarray, el: np.ndarray, z: np.ndarray) -> None:
    """Transcription of WRF ``VDIFQ``; mutates ``q2``."""

    akq = np.zeros(lmh + 2, dtype=np.float64)
    cm = np.zeros(lmh + 2, dtype=np.float64)
    cr = np.zeros(lmh + 2, dtype=np.float64)
    dtoz = np.zeros(lmh + 2, dtype=np.float64)
    rsq2 = np.zeros(lmh + 2, dtype=np.float64)
    esqhf = 0.5 * C.ESQ

    for k in range(1, lmh - 1):
        dtoz[k] = (dtdif + dtdif) / (z[k] - z[k + 2])
        akq[k] = (
            np.sqrt((q2[k] + q2[k + 1]) * 0.5)
            * (el[k] + el[k + 1])
            * esqhf
            / (z[k + 1] - z[k + 2])
        )
        cr[k] = -dtoz[k] * akq[k]

    cm[1] = dtoz[1] * akq[1] + 1.0
    rsq2[1] = q2[1]

    for k in range(2, lmh - 1):
        cf = -dtoz[k] * akq[k - 1] / cm[k - 1]
        cm[k] = -cr[k - 1] * cf + (akq[k - 1] + akq[k]) * dtoz[k] + 1.0
        rsq2[k] = -rsq2[k - 1] * cf + q2[k]

    dtozs = (dtdif + dtdif) / (z[lmh - 1] - z[lmh + 1])
    akqs = (
        np.sqrt((q2[lmh - 1] + q2[lmh]) * 0.5)
        * (el[lmh - 1] + C.ELZ0)
        * esqhf
        / (z[lmh] - z[lmh + 1])
    )
    cf = -dtozs * akq[lmh - 2] / cm[lmh - 2]
    q2[lmh - 1] = (
        dtozs * akqs * q2[lmh] - rsq2[lmh - 2] * cf + q2[lmh - 1]
    ) / ((akq[lmh - 2] + akqs) * dtozs - cr[lmh - 2] * cf + 1.0)

    for k in range(lmh - 2, 0, -1):
        q2[k] = (-cr[k] * q2[k + 1] + rsq2[k]) / cm[k]


def _vdifh(
    dtdif: float,
    lmh: int,
    lpbl: int,
    sz0: np.ndarray,
    rkhs: float,
    clow: np.ndarray,
    cts: np.ndarray,
    species: np.ndarray,
    nspec: int,
    rkh: np.ndarray,
    z: np.ndarray,
    rho: np.ndarray,
) -> None:
    """Transcription of WRF ``VDIFH``; mutates ``species``."""

    cm = np.zeros(lmh + 2, dtype=np.float64)
    cr = np.zeros(lmh + 2, dtype=np.float64)
    dtoz = np.zeros(lmh + 2, dtype=np.float64)
    rkct = np.zeros((8, lmh + 2), dtype=np.float64)
    rss = np.zeros((8, lmh + 2), dtype=np.float64)

    for k in range(1, lmh):
        dtoz[k] = dtdif / (z[k] - z[k + 1])
        cr[k] = -dtoz[k] * rkh[k]
        if k < lpbl:
            rkct[1 : nspec + 1, k] = 0.0
        else:
            rkhz = rkh[k] * (z[k] - z[k + 2])
            for m in range(1, nspec + 1):
                rkct[m, k] = rkhz * cts[m] * 0.5

    rhok = rho[1]
    cm[1] = dtoz[1] * rkh[1] + rhok
    for m in range(1, nspec + 1):
        rss[m, 1] = -rkct[m, 1] * dtoz[1] + species[m, 1] * rhok

    for k in range(2, lmh):
        dtozl = dtoz[k]
        cf = -dtozl * rkh[k - 1] / cm[k - 1]
        rhok = rho[k]
        cm[k] = -cr[k - 1] * cf + (rkh[k - 1] + rkh[k]) * dtozl + rhok
        for m in range(1, nspec + 1):
            rss[m, k] = (
                -rss[m, k - 1] * cf
                + (rkct[m, k - 1] - rkct[m, k]) * dtozl
                + species[m, k] * rhok
            )

    dtozs = dtdif / (z[lmh] - z[lmh + 1])
    rkhh = rkh[lmh - 1]
    cf = -dtozs * rkhh / cm[lmh - 1]
    cmb = cr[lmh - 1] * cf
    rhok = rho[lmh]

    for m in range(1, nspec + 1):
        rkss = rkhs * clow[m]
        cmsb = -cmb + (rkhh + rkss) * dtozs + rhok
        rssb = -rss[m, lmh - 1] * cf + rkct[m, lmh - 1] * dtozs + species[m, lmh] * rhok
        species[m, lmh] = (dtozs * rkss * sz0[m] + rssb) / cmsb

    for k in range(lmh - 1, 0, -1):
        rcml = 1.0 / cm[k]
        for m in range(1, nspec + 1):
            species[m, k] = (-cr[k] * species[m, k + 1] + rss[m, k]) * rcml


def _vdifv(
    lmh: int,
    dtdif: float,
    uz0: float,
    vz0: float,
    rkms: float,
    u: np.ndarray,
    v: np.ndarray,
    rkm: np.ndarray,
    z: np.ndarray,
    rho: np.ndarray,
) -> None:
    """Transcription of WRF ``VDIFV``; mutates ``u`` and ``v``."""

    cm = np.zeros(lmh + 2, dtype=np.float64)
    cr = np.zeros(lmh + 2, dtype=np.float64)
    dtoz = np.zeros(lmh + 2, dtype=np.float64)
    rsu = np.zeros(lmh + 2, dtype=np.float64)
    rsv = np.zeros(lmh + 2, dtype=np.float64)

    for k in range(1, lmh):
        dtoz[k] = dtdif / (z[k] - z[k + 1])
        cr[k] = -dtoz[k] * rkm[k]

    rhok = rho[1]
    cm[1] = dtoz[1] * rkm[1] + rhok
    rsu[1] = u[1] * rhok
    rsv[1] = v[1] * rhok

    for k in range(2, lmh):
        dtozl = dtoz[k]
        cf = -dtozl * rkm[k - 1] / cm[k - 1]
        rhok = rho[k]
        cm[k] = -cr[k - 1] * cf + (rkm[k - 1] + rkm[k]) * dtozl + rhok
        rsu[k] = -rsu[k - 1] * cf + u[k] * rhok
        rsv[k] = -rsv[k - 1] * cf + v[k] * rhok

    dtozs = dtdif / (z[lmh] - z[lmh + 1])
    rkmh = rkm[lmh - 1]
    cf = -dtozs * rkmh / cm[lmh - 1]
    rhok = rho[lmh]
    rcmvb = 1.0 / ((rkmh + rkms) * dtozs - cr[lmh - 1] * cf + rhok)
    dtozak = dtozs * rkms
    u[lmh] = (dtozak * uz0 - rsu[lmh - 1] * cf + u[lmh] * rhok) * rcmvb
    v[lmh] = (dtozak * vz0 - rsv[lmh - 1] * cf + v[lmh] * rhok) * rcmvb

    for k in range(lmh - 1, 0, -1):
        rcml = 1.0 / cm[k]
        u[k] = (-cr[k] * u[k + 1] + rsu[k]) * rcml
        v[k] = (-cr[k] * v[k + 1] + rsv[k]) * rcml


def _bottom_up_from_top(work: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, n + 1):
        kflip = n + 1 - k
        out[k - 1] = work[kflip]
    return out


def myjpbl_column(
    *,
    u,
    v,
    temperature,
    theta,
    qv,
    qc,
    p_mid,
    p_int,
    exner,
    dz,
    tke,
    tsk,
    xland,
    ustar,
    znt,
    akhs,
    akms,
    chklowq,
    elflx,
    thz0,
    qz0,
    uz0,
    vz0,
    qsfc,
    ct=0.0,
    snow=0.0,
    sice=0.0,
    dt=60.0,
    stepbl=1,
    ht=0.0,
) -> dict[str, np.ndarray | float | int]:
    """Run one MYJ PBL column and return WRF-savepoint fields.

    All profile inputs are bottom-up WRF mass-level arrays.  ``tke`` is
    ``TKE_MYJ`` (0.5*q**2), not q**2.
    """

    u_b = _as1d(u, name="u")
    n = u_b.shape[0]
    v_b = _as1d(v, length=n, name="v")
    t_b = _as1d(temperature, length=n, name="temperature")
    th_b = _as1d(theta, length=n, name="theta")
    qv_b = _as1d(qv, length=n, name="qv")
    qc_b = _as1d(qc, length=n, name="qc")
    p_b = _as1d(p_mid, length=n, name="p_mid")
    pint_b = _as1d(p_int, length=n + 1, name="p_int")
    exner_b = _as1d(exner, length=n, name="exner")
    dz_b = _as1d(dz, length=n, name="dz")
    tke_b = _as1d(tke, length=n, name="tke")

    lmh = n
    dtturbl = float(dt) * int(stepbl)
    rdtturbl = 1.0 / dtturbl

    u_top = _one_based_top_down(u_b)
    v_top = _one_based_top_down(v_b)
    t_top = _one_based_top_down(t_b)
    th_top = _one_based_top_down(th_b)
    qv_top_mix = _one_based_top_down(qv_b)
    qc_top = _one_based_top_down(qc_b)
    p_top = _one_based_top_down(p_b)
    z = _one_based_interfaces(dz_b, _scalar(ht))
    q_top = np.zeros(n + 2, dtype=np.float64)
    cwm = qc_top.copy()
    the = np.zeros(n + 2, dtype=np.float64)
    q2 = _one_based_top_down(2.0 * tke_b)

    for k in range(1, n + 1):
        q_top[k] = qv_top_mix[k] / (1.0 + qv_top_mix[k])
        the[k] = (cwm[k] * (-C.ELOCP / t_top[k]) + 1.0) * th_top[k]

    gm, gh, el, pblh, lpbl, lmxl, ct_work, mixht = _mixlen(
        lmh, u_top, v_top, t_top, the, q_top, cwm, q2, z, _scalar(ct)
    )
    _prodq2(lmh, dtturbl, _scalar(ustar), gm, gh, el, q2)
    kpbl = n - lpbl + 1
    akm, akh = _difcof(lmh, gm, gh, el, q2, z)

    exch_h = np.zeros(n, dtype=np.float64)
    exch_m = np.zeros(n, dtype=np.float64)
    for k_b in range(1, n):
        kflip = n - k_b
        deltaz = 0.5 * (z[kflip] - z[kflip + 2])
        exch_h[k_b - 1] = akh[kflip] * deltaz
        exch_m[k_b - 1] = akm[kflip] * deltaz

    _vdifq(lmh, dtturbl, q2, el, z)

    tke_out = np.zeros(n, dtype=np.float64)
    el_out = np.zeros(n, dtype=np.float64)
    for k_b in range(1, n + 1):
        kflip = n + 1 - k_b
        q2[kflip] = max(q2[kflip], C.EPSQ2)
        tke_out[k_b - 1] = 0.5 * q2[kflip]
        if kflip < n:
            el_out[k_b - 1] = el[kflip]

    thsk = _scalar(tsk) * (1.0e5 / pint_b[0]) ** C.CAPA
    rho_top = np.zeros(n + 2, dtype=np.float64)
    species = np.zeros((8, n + 2), dtype=np.float64)
    qci = np.zeros(n + 2, dtype=np.float64)
    nspec = 4
    for k in range(1, n + 1):
        species[1, k] = the[k]
        species[2, k] = q_top[k]
        species[3, k] = qc_top[k]
        species[4, k] = qci[k]
        rho_top[k] = p_top[k] / (C.R_D * t_top[k] * (1.0 + C.P608 * q_top[k] - cwm[k]))

    akh_dens = np.zeros(n + 2, dtype=np.float64)
    for k in range(1, n):
        akh_dens[k] = akh[k] * 0.5 * (rho_top[k] + rho_top[k + 1])

    seamask = _scalar(xland) - 1.0
    thz0_work = (1.0 - seamask) * thsk + seamask * _scalar(thz0)
    akhs_dens = _scalar(akhs) * rho_top[n]
    qsfc_work = _scalar(qsfc)
    if seamask < 0.5:
        qfc1 = C.XLV * _scalar(chklowq) * akhs_dens
        if _scalar(snow) > 0.0 or _scalar(sice) > 0.5:
            qfc1 = qfc1 * C.RLIVWV
        if qfc1 > 0.0:
            qsfc_work = q_top[n] + _scalar(elflx) / qfc1
    else:
        exnsfc = (1.0e5 / pint_b[0]) ** C.CAPA
        qsfc_work = C.PQ0SEA / pint_b[0] * np.exp(
            C.A2 * (thsk - C.A3 * exnsfc) / (thsk - C.A4 * exnsfc)
        )
    qz0_work = (1.0 - seamask) * qsfc_work + seamask * _scalar(qz0)

    sz0 = np.zeros(8, dtype=np.float64)
    clow = np.zeros(8, dtype=np.float64)
    cts = np.zeros(8, dtype=np.float64)
    sz0[1] = thz0_work
    sz0[2] = qz0_work
    clow[1] = 1.0
    clow[2] = _scalar(chklowq)
    cts[1] = ct_work

    _vdifh(dtturbl, lmh, lpbl, sz0, akhs_dens, clow, cts, species, nspec, akh_dens, z, rho_top)

    the_new = species[1].copy()
    q_new = species[2].copy()
    qc_new = species[3].copy()
    cwm_new = qc_new.copy()
    rthblten = np.zeros(n, dtype=np.float64)
    rqvblten = np.zeros(n, dtype=np.float64)
    for k_b in range(1, n + 1):
        kflip = n + 1 - k_b
        thnew = the_new[kflip] + cwm_new[kflip] * C.ELOCP / exner_b[k_b - 1]
        dtdt = (thnew - th_b[k_b - 1]) * rdtturbl
        qold = qv_b[k_b - 1] / (1.0 + qv_b[k_b - 1])
        dqdt = (q_new[kflip] - qold) * rdtturbl
        rthblten[k_b - 1] = dtdt
        rqvblten[k_b - 1] = dqdt / (1.0 - q_new[kflip]) ** 2

    akm_dens = np.zeros(n + 2, dtype=np.float64)
    for k in range(1, n):
        akm_dens[k] = akm[k] * 0.5 * (rho_top[k] + rho_top[k + 1])
    uk = u_top.copy()
    vk = v_top.copy()
    akms_dens = _scalar(akms) * rho_top[n]
    _vdifv(lmh, dtturbl, _scalar(uz0), _scalar(vz0), akms_dens, uk, vk, akm_dens, z, rho_top)

    rub = np.zeros(n, dtype=np.float64)
    rvb = np.zeros(n, dtype=np.float64)
    for k_b in range(1, n + 1):
        kflip = n + 1 - k_b
        rub[k_b - 1] = (uk[kflip] - u_b[k_b - 1]) * rdtturbl
        rvb[k_b - 1] = (vk[kflip] - v_b[k_b - 1]) * rdtturbl

    return {
        "TKE_MYJ": tke_out,
        "EXCH_H": exch_h,
        "EXCH_M": exch_m,
        "EL_MYJ": el_out,
        "RUBLTEN": rub,
        "RVBLTEN": rvb,
        "RTHBLTEN": rthblten,
        "RQVBLTEN": rqvblten,
        "PBLH": float(pblh),
        "KPBL": int(kpbl),
        "MIXHT": float(mixht),
        "AKH": _bottom_up_from_top(akh, n),
        "AKM": _bottom_up_from_top(akm, n),
        "QSFC": float(qsfc_work),
        "THZ0": float(thz0_work),
        "QZ0": float(qz0_work),
    }


def step_myj_pbl_column(
    u,
    v,
    temperature,
    theta,
    qv,
    qc,
    pressure,
    pressure_interface,
    exner,
    dz,
    tke,
    *,
    tsk,
    xland,
    ustar,
    znt,
    akhs,
    akms,
    chklowq,
    elflx,
    thz0,
    qz0,
    uz0,
    vz0,
    qsfc,
    ct=0.0,
    snow=0.0,
    sice=0.0,
    dt=60.0,
    stepbl=1,
    ht=0.0,
) -> PhysicsStepResult:
    """MYJ PBL column endpoint -> frozen ``PhysicsStepResult``."""

    out = myjpbl_column(
        u=u,
        v=v,
        temperature=temperature,
        theta=theta,
        qv=qv,
        qc=qc,
        p_mid=pressure,
        p_int=pressure_interface,
        exner=exner,
        dz=dz,
        tke=tke,
        tsk=tsk,
        xland=xland,
        ustar=ustar,
        znt=znt,
        akhs=akhs,
        akms=akms,
        chklowq=chklowq,
        elflx=elflx,
        thz0=thz0,
        qz0=qz0,
        uz0=uz0,
        vz0=vz0,
        qsfc=qsfc,
        ct=ct,
        snow=snow,
        sice=sice,
        dt=dt,
        stepbl=stepbl,
        ht=ht,
    )
    tendency = PhysicsTendency(
        state_tendencies={
            "u": jnp.asarray(out["RUBLTEN"], dtype=jnp.float64),
            "v": jnp.asarray(out["RVBLTEN"], dtype=jnp.float64),
            "theta": jnp.asarray(out["RTHBLTEN"], dtype=jnp.float64),
            "qv": jnp.asarray(out["RQVBLTEN"], dtype=jnp.float64),
        },
    )
    tendency.validate_keys()
    carry = PhysicsCarry(
        pbl={
            "tke_pbl": jnp.asarray(out["TKE_MYJ"], dtype=jnp.float64),
            "el_pbl": jnp.asarray(out["EL_MYJ"], dtype=jnp.float64),
        }
    )
    diagnostics = PhysicsDiagnostics(
        pbl={
            "pblh": jnp.asarray(out["PBLH"], dtype=jnp.float64),
            "kpbl": jnp.asarray(out["KPBL"], dtype=jnp.int32),
            "mixht": jnp.asarray(out["MIXHT"], dtype=jnp.float64),
            "tke_pbl": jnp.asarray(out["TKE_MYJ"], dtype=jnp.float64),
            "tke_myj": jnp.asarray(out["TKE_MYJ"], dtype=jnp.float64),
            "exch_h": jnp.asarray(out["EXCH_H"], dtype=jnp.float64),
            "exch_m": jnp.asarray(out["EXCH_M"], dtype=jnp.float64),
            "el_pbl": jnp.asarray(out["EL_MYJ"], dtype=jnp.float64),
            "el_myj": jnp.asarray(out["EL_MYJ"], dtype=jnp.float64),
            "akh": jnp.asarray(out["AKH"], dtype=jnp.float64),
            "akm": jnp.asarray(out["AKM"], dtype=jnp.float64),
        }
    )
    return PhysicsStepResult(tendency=tendency, carry=carry, diagnostics=diagnostics)


__all__ = ["myjpbl_column", "step_myj_pbl_column"]
