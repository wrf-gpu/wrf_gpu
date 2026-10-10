"""WRF CAM radiation driver (``ra_lw_physics = 3`` / ``ra_sw_physics = 3``): the ``camrad`` wrapper as JAX columns.

Pristine WRF V4.7.1 ``phys/module_ra_cam.F`` ``camrad`` -> ``radctl``; the radiation_driver calls camrad TWICE per
radiation step (CAMLWSCHEME: ``dolw=.true., dosw=.false.``; CAMSWSCHEME: ``dolw=.false., dosw=.true.``).  This module
reproduces both calls on flat WRF-order columns (index 0 = lowest level) and returns WRF's REAL outputs.

Operands are what WRF's radiation_driver passes: ``t_phy``, ``p`` / ``p8w`` (hydrostatic p_hyd / p_hyd_w on the
operational path), ``pi_phy``, moisture (Thompson/WSM branch: qice = qi + qs), ``CLDFRA``, surface fields, ``coszen``
(calc_coszen at xtime + radt/2) and the solar constant (radconst).  The CAM monthly ozone (``oznini``) is a function of
latitude only and is passed in pre-interpolated (``ozmixm``, ``pin``), like WRF's resident ozmixm/pin grid arrays.

Held absorptivities: ``doabsems`` (radiation_driver: every ``cam_abs_freq_s``) recomputes the non-adjacent/nearest-layer
absorptivities and the emissivity; otherwise the HELD REAL arrays (WRF state abstot/absnxt/emstot) are used.  Outputs
always include the REAL arrays to hold.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import ra_cam_lw as lw
from gpuwrf.physics.ra_cam_common import (
    CONST,
    CamAbsTables,
    camrad_prepare,
    lit,
    oznint,
    param_cldoptics_calc,
    radinp,
    radozn,
    trcmix_clwrf,
)
from gpuwrf.physics.ra_cam_lw_abs import radabs


class CamGases(NamedTuple):
    """CLWRF (ghg_input = 1) volume mixing ratios as ``read_CAMgases(yr, julian, .false., "CAM")`` returns them (r8)."""

    co2vmr: jnp.ndarray
    n2ovmr: jnp.ndarray
    ch4vmr: jnp.ndarray
    f11vmr: jnp.ndarray
    f12vmr: jnp.ndarray


class CamHeld(NamedTuple):
    """WRF REAL state arrays abstot_3d / absnxt_3d / emstot_3d of one column set (CAM vertical order)."""

    abstot: jnp.ndarray  # (ncol, pverp, pverp) float32
    absnxt: jnp.ndarray  # (ncol, pver, 4) float32
    emstot: jnp.ndarray  # (ncol, pverp) float32


class CamLwOut(NamedTuple):
    rthratenlw: jnp.ndarray   # (ncol, nz) WRF order, REAL, K/s (theta tendency)
    rthratenlwc: jnp.ndarray
    glw: jnp.ndarray
    olr: jnp.ndarray
    lwcf: jnp.ndarray
    lwupt: jnp.ndarray
    lwuptc: jnp.ndarray
    lwdnt: jnp.ndarray
    lwdntc: jnp.ndarray
    lwupb: jnp.ndarray
    lwupbc: jnp.ndarray
    lwdnb: jnp.ndarray
    lwdnbc: jnp.ndarray
    cemiss: jnp.ndarray       # (ncol, nz) WRF order, REAL
    held: CamHeld


def _real(x):
    return jnp.asarray(x).astype(jnp.float32)


def _theta_tendency(q_cam, pi_phy):
    """camrad: RTHRATEN(k) = REAL(1.e4*q(kk)/(cpair*pi_phy(k))), kk the CAM index of WRF level k."""

    return _real(1.0e4 * q_cam[:, ::-1] / (CONST.cpair * jnp.asarray(pi_phy, jnp.float32).astype(jnp.float64)))


def camrad_lw(*, t_phy, p_phy, p8w, pi_phy, qv, qc, qi, qs, cldfra, xland, xice, snow, emiss, tsk, xlat, coszen,
              albedo, julian, gases: CamGases, ozmixm, pin, tables: CamAbsTables, doabsems=True,
              held: CamHeld | None = None) -> CamLwOut:
    """CAMLWSCHEME camrad call (dolw = .true., dosw = .false.) on ``(ncol, nz)`` WRF-order REAL operands."""

    cols = camrad_prepare(t_phy, p_phy, p8w, qv, qc, qi, qs, cldfra, xland, xice, snow, emiss, tsk, xlat, coszen, albedo)
    optics = param_cldoptics_calc(cols)
    o3vmr = radozn(cols.pmid, pin, oznint(ozmixm, julian))
    pbr, pnm, _eccf, _o3mmr = radinp(cols.pmid, cols.pint, o3vmr)
    as_col = lambda g: jnp.reshape(jnp.asarray(g, jnp.float64), (-1, 1))
    n2o, ch4, cfc11, cfc12 = trcmix_clwrf(cols.pmid, cols.clat, as_col(gases.n2ovmr), as_col(gases.ch4vmr),
                                          as_col(gases.f11vmr), as_col(gases.f12vmr))
    co2mmr = jnp.asarray(gases.co2vmr, jnp.float64) * CONST.mwco2 / CONST.mwdry
    tp = lw.radtpl(cols.t, cols.lwups, cols.q1, pnm, cols.lnpmid, cols.lnpint)
    if doabsems:
        plol, plos = lw.radoz2(o3vmr, pnm)
        tr = lw.trcpth(cols.t, pnm, cfc11, cfc12, n2o, ch4, cols.q1, jnp.broadcast_to(co2mmr, cols.t.shape[:1]))
        em = lw.radems(tp, pnm, plol, plos, tr, tables)
        abstot, absnxt = radabs(pbr, pnm, em.co2em, em.co2eml, tp.tplnka, tp.s2c, tp.tcg, tp.w, em.h2otr, tp.plco2,
                                tp.plh2o, em.co2t, tp.tint, tp.tlayr, plol, plos, cols.lnpmid, cols.lnpint, tr.ucfc11,
                                tr.ucfc12, tr.un2o0, tr.un2o1, tr.uch4, tr.uco211, tr.uco212, tr.uco213, tr.uco221,
                                tr.uco222, tr.uco223, tr.uptype, tr.bn2o0, tr.bn2o1, tr.bch4, em.abplnk1, em.abplnk2,
                                tp.plh2ob, tp.wb, tables)
        emstot = em.emstot
        new_held = CamHeld(_real(abstot), _real(absnxt), _real(emstot))
    else:
        if held is None:
            raise ValueError("camrad_lw(doabsems=False) needs the held REAL abstot/absnxt/emstot")
        abstot, absnxt, emstot = (jnp.asarray(a, jnp.float32).astype(jnp.float64) for a in held)
        new_held = held
    fx = lw.radclwmx_fluxes(abstot, absnxt, emstot, tp, cols.lwups, pbr, pnm, cols.cld, optics.emis, optics.pmxrgn,
                            optics.nmxrgn)
    k3 = lit(1.e-3)
    flut, flutc, flwds = fx.flut * k3, fx.flutc * k3, fx.flwds * k3
    flup, flupc, fldn, fldnc = fx.flup * k3, fx.flupc * k3, fx.fldn * k3, fx.fldnc * k3
    return CamLwOut(
        rthratenlw=_theta_tendency(fx.qrl, pi_phy),
        rthratenlwc=_theta_tendency(fx.qrlcs, pi_phy),
        glw=_real(flwds),
        olr=_real(flut),
        lwcf=_real(flutc - flut),
        lwupt=_real(flup[:, 0]),
        lwuptc=_real(flupc[:, 0]),
        lwdnt=_real(fldn[:, 0]),
        lwdntc=_real(fldnc[:, 0]),
        lwupb=_real(flup[:, -1]),
        lwupbc=_real(flupc[:, -1]),
        lwdnb=_real(fldn[:, -1]),
        lwdnbc=_real(fldnc[:, -1]),
        cemiss=_real(optics.emis[:, ::-1]),
        held=new_held,
    )


# aerosol_init (module_ra_cam_support.F:3602): REAL hybrid interface coefficients of the uniform CAM aerosol climatology.
HYBI = np.asarray([0, 0.0065700002014637, 0.0138600002974272, 0.023089999333024, 0.0346900001168251,
                   0.0491999983787537, 0.0672300010919571, 0.0894500017166138, 0.116539999842644, 0.149159997701645,
                   0.187830001115799, 0.232859998941422, 0.284209996461868, 0.341369986534119, 0.403340011835098,
                   0.468600004911423, 0.535290002822876, 0.601350009441376, 0.66482001543045, 0.724009990692139,
                   0.777729988098145, 0.825269997119904, 0.866419970989227, 0.901350021362305, 0.930540025234222,
                   0.954590022563934, 0.974179983139038, 0.990000009536743, 1], np.float32)


def cam_mxaerl(znu, p_top_pa) -> int:
    """aerosol_init: number of mass levels with shalf*1e5 + pptop >= 9e4 (REAL; shalf = ZNU, pptop = p_top/1000
    from z2sigma's eta branch -- WRF's unit mix kept), at least 1."""

    znu = np.asarray(znu, np.float32)
    pptop = np.float32(np.float32(p_top_pa) / np.float32(1000.0))
    return max(int(np.sum(znu * np.float32(1.0e5) + pptop >= np.float32(9.0e4))), 1)


def znu_from_znw(znw) -> np.ndarray:
    """real.exe ZNU = 0.5*(ZNW(k+1) + ZNW(k)) in REAL."""

    znw = np.asarray(znw, np.float32)
    return (np.float32(0.5) * (znw[1:] + znw[:-1])).astype(np.float32)


class CamSwOut(NamedTuple):
    rthratensw: jnp.ndarray   # (ncol, nz) WRF order, REAL, K/s (theta tendency)
    rthratenswc: jnp.ndarray
    gsw: jnp.ndarray          # net surface SW (fsns)
    swdown: jnp.ndarray       # radiation_driver SWDOWN = GSW/(1 - ALBEDO)
    swcf: jnp.ndarray
    coszr: jnp.ndarray
    swddir: jnp.ndarray
    swddni: jnp.ndarray
    swddif: jnp.ndarray
    swupt: jnp.ndarray
    swuptc: jnp.ndarray
    swdnt: jnp.ndarray
    swdntc: jnp.ndarray
    swupb: jnp.ndarray
    swupbc: jnp.ndarray
    swdnb: jnp.ndarray
    swdnbc: jnp.ndarray
    taucldc: jnp.ndarray      # (ncol, nz) WRF order, REAL
    taucldi: jnp.ndarray


def camrad_sw(*, t_phy, p_phy, p8w, pi_phy, qv, qc, qi, qs, cldfra, xland, xice, snow, emiss, tsk, xlat, coszen,
              albedo, julian, solcon, co2vmr, ozmixm, pin, mxaerl, aeropt_tables=None) -> CamSwOut:
    """CAMSWSCHEME camrad call (dolw = .false., dosw = .true.) on ``(ncol, nz)`` WRF-order REAL operands."""

    from gpuwrf.physics.ra_cam_sw import cam_sw_radctl

    cols = camrad_prepare(t_phy, p_phy, p8w, qv, qc, qi, qs, cldfra, xland, xice, snow, emiss, tsk, xlat, coszen, albedo)
    optics = param_cldoptics_calc(cols)
    o3vmr = radozn(cols.pmid, pin, oznint(ozmixm, julian))
    co2mmr = jnp.asarray(co2vmr, jnp.float64) * CONST.mwco2 / CONST.mwdry
    sw = cam_sw_radctl(q1=cols.q1, qliq=cols.qliq, qice=cols.qice, cld=cols.cld, pmid=cols.pmid, pint=cols.pint,
                       t=cols.t, cicewp=optics.cicewp, cliqwp=optics.cliqwp, rel=optics.rel, rei=optics.rei,
                       pmxrgn=optics.pmxrgn, nmxrgn=optics.nmxrgn, o3vmr=o3vmr, coszrs=cols.coszrs,
                       solcon=jnp.asarray(solcon, jnp.float32).astype(jnp.float64), albedo=cols.albedo,
                       julian=jnp.broadcast_to(jnp.asarray(julian, jnp.float32), cols.t.shape[:1]), mxaerl=mxaerl,
                       m_hybi=HYBI, co2mmr=co2mmr, landfrac=cols.landfrac,
                       tables=aeropt_tables)
    gsw = _real(sw["fsns"])
    alb = jnp.asarray(albedo, jnp.float32)
    return CamSwOut(
        rthratensw=_theta_tendency(sw["qrs"], pi_phy),
        rthratenswc=_theta_tendency(sw["qrscs"], pi_phy),
        gsw=gsw,
        swdown=gsw / (jnp.float32(1.0) - alb),
        swcf=_real(sw["swcftoa"]),
        coszr=_real(cols.coszrs),
        swddir=_real(sw["fsdsdir"]),
        swddni=_real(sw["fsdsdir"] / cols.coszrs),
        swddif=_real(sw["fsdsdif"]),
        swupt=_real(sw["fsup"][:, 0]),
        swuptc=_real(sw["fsupc"][:, 0]),
        swdnt=_real(sw["fsdn"][:, 0]),
        swdntc=_real(sw["fsdnc"][:, 0]),
        swupb=_real(sw["fsup"][:, -1]),
        swupbc=_real(sw["fsupc"][:, -1]),
        swdnb=_real(sw["fsdn"][:, -1]),
        swdnbc=_real(sw["fsdnc"][:, -1]),
        taucldc=_real(sw["tauxcl"][:, ::-1]),
        taucldi=_real(sw["tauxci"][:, ::-1]),
    )


__all__ = ["CamGases", "CamHeld", "CamLwOut", "CamSwOut", "HYBI", "cam_mxaerl", "camrad_lw", "camrad_sw",
           "znu_from_znw"]
