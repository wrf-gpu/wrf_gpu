"""S1 (Opus) — vertical eta/hybrid coordinate + 1D coefficients.

FROZEN ENTRY SIGNATURE (do not change without manager sign-off). Implements the
real.exe ``compute_eta`` (module_initialize_real.F:7567-7894, driven at :1590)
and ``compute_vcoord_1d_coeffs`` (nest_init_utils.F:1033-1184, driven at :1601).

Algorithm (faithful spec for the implementer):
  1. Build full eta levels ``znw`` (len nz+1), ``znw[0]=1.0`` -> ``znw[nz]=0.0``.
     If ``config.eta_levels`` is given, use them; else auto-generate via the
     ``compute_eta``/``levels`` stretch (module_initialize_real.F:7897 ``levels``;
     auto_levels_opt=2 uses the dzbot/dzstretch_s/dzstretch_u/max_dz stretch).
  2. Half levels ``znu[k]=0.5*(znw[k]+znw[k+1])`` (real.exe:3735).
  3. ``dnw[k]=znw[k+1]-znw[k]``, ``rdnw=1/dnw`` (real.exe:3733-3734);
     ``dn[k]=0.5*(dnw[k]+dnw[k-1])``, ``rdn=1/dn`` (real.exe:3742-3743);
     ``fnp[k]=0.5*dnw[k]/dn[k]``, ``fnm[k]=0.5*dnw[k-1]/dn[k]`` (real.exe:3744-3745).
  4. Extrapolation coeffs cof1/cof2 -> cf1/cf2/cf3/cfn/cfn1 (real.exe:3750-3758).
  5. Hybrid coefficients C3F per ``hybrid_opt`` (nest_init_utils.F:1071-1089):
       hybrid_opt 0/1: C3F[k]=znw[k]
       hybrid_opt 2  : Klemp polynomial with B1..B5 in terms of ``etac``
         B1=2*etac^2*(1-etac); B2=-etac*(4-3*etac-etac^3); B3=2*(1-etac^3);
         B4=-(1-etac^2); B5=(1-etac)^4;
         C3F=(B1+B2*znw+B3*znw^2+B4*znw^3)/B5, clamped: C3F=1 where znw>=etac
         region per WRF, C3F=0 at top. (Confirm exact branch vs the .F at impl.)
     Then C4F[k]=(znw[k]-C3F[k])*(p1000mb-p_top)  (nest_init_utils.F:1078).
     C1F/C2F and the half-level C1H/C2H/C3H/C4H follow nest_init_utils.F:1090+
     (C1H half-average of C3F, C2H=(1-C1H)*(p00-p_top), etc — copy faithfully).

Oracle: wrfinput ZNU/ZNW/C1H/C2H/C3H/C4H/C1F/.../P_TOP for d01/d02/d03 of the
≥10 cases; tolerances ``types.WRFINPUT_TOLS`` (1D-coord rows). Compare against
the real.exe values at the SAME nz/p_top/hybrid_opt/etac.

FILE OWNERSHIP: this file + base_state.py + hydrostatic.py + vinterp.py are S1's
exclusive files. Do not edit types.py, driver.py, or any S2/S3 file.
"""

from __future__ import annotations

import numpy as np

from gpuwrf.init.real_init.types import G, P1000MB, R_D
from gpuwrf.init.real_init.types import RealInitConfig, VerticalCoord1D


def compute_vertical_coord(config: RealInitConfig) -> VerticalCoord1D:
    """Builds the 1D vertical-coordinate + hybrid-coefficient arrays.

    Pure function of the namelist scalars in ``config`` (nz, p_top, hybrid_opt,
    etac, eta_levels/auto params). Returns a fully-populated
    :class:`VerticalCoord1D`. No domain horizontal fields are needed (terrain
    enters only the base-state/hydrostatic lanes via C1/C2/C3/C4).

    Implementer: fill per the algorithm in this module's docstring; validate
    against the wrfinput 1D-coord oracle before claiming the lane done.
    """

    znw = _compute_znw(config)
    nz = config.nz

    dnw = znw[1:] - znw[:-1]
    rdnw = 1.0 / dnw
    znu = 0.5 * (znw[1:] + znw[:-1])

    dn = np.zeros(nz, dtype=np.float64)
    rdn = np.zeros(nz, dtype=np.float64)
    fnp = np.zeros(nz, dtype=np.float64)
    fnm = np.zeros(nz, dtype=np.float64)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    rdn[1:] = 1.0 / dn[1:]
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]

    cof1 = (2.0 * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
    cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
    cf1 = float(fnp[1] + cof1)
    cf2 = float(fnm[1] - cof1 - cof2)
    cf3 = float(cof2)
    cfn = float((0.5 * dnw[-1] + dn[-1]) / dn[-1])
    cfn1 = float(-0.5 * dnw[-1] / dn[-1])

    c3f = _compute_c3f(config.hybrid_opt, config.etac, znw)
    c4f = (znw - c3f) * (P1000MB - config.p_top_pa)
    c3h = 0.5 * (c3f[1:] + c3f[:-1])
    c4h = (znu - c3h) * (P1000MB - config.p_top_pa)

    c1f = np.zeros(nz + 1, dtype=np.float64)
    c1f[1:nz] = (c3h[1:] - c3h[:-1]) / (znu[1:] - znu[:-1])
    c1f[0] = 1.0
    c1f[nz] = 1.0 if config.hybrid_opt in (0, 1) else 0.0
    c2f = (1.0 - c1f) * (P1000MB - config.p_top_pa)

    c1h = (c3f[1:] - c3f[:-1]) / (znw[1:] - znw[:-1])
    c2h = (1.0 - c1h) * (P1000MB - config.p_top_pa)

    return VerticalCoord1D(
        znw=znw,
        znu=znu,
        dnw=dnw,
        rdnw=rdnw,
        dn=dn,
        rdn=rdn,
        fnp=fnp,
        fnm=fnm,
        c1f=c1f,
        c2f=c2f,
        c3f=c3f,
        c4f=c4f,
        c1h=c1h,
        c2h=c2h,
        c3h=c3h,
        c4h=c4h,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        cfn=cfn,
        cfn1=cfn1,
        p_top_pa=float(config.p_top_pa),
    )


def _compute_znw(config: RealInitConfig) -> np.ndarray:
    if config.eta_levels:
        eta = np.asarray(config.eta_levels, dtype=np.float64)
        if eta.shape != (config.nz + 1,):
            raise ValueError(
                f"eta_levels length must be nz+1={config.nz + 1}; got {eta.shape}"
            )
        if np.isclose(eta[0], 1.0) and np.isclose(eta[-1], 0.0):
            znw = eta.copy()
        elif np.isclose(eta[0], 0.0) and np.isclose(eta[-1], 1.0):
            znw = eta[::-1].copy()
        else:
            raise ValueError("eta_levels must run 1->0 or 0->1")
        if np.any(znw[:-1] <= znw[1:]):
            raise ValueError("eta_levels must be strictly decreasing in WRF order")
        znw[0] = 1.0
        znw[-1] = 0.0
        return znw

    if config.auto_levels_opt != 2:
        raise ValueError(
            "v0.4.0 S1 implements explicit eta levels and WRF auto_levels_opt=2; "
            f"got auto_levels_opt={config.auto_levels_opt}"
        )
    return _levels_auto_opt2(
        config.nz,
        float(config.p_top_pa),
        float(config.max_dz),
        float(config.dzbot),
        float(config.dzstretch_s),
        float(config.dzstretch_u),
    )


def _levels_auto_opt2(
    nlev: int,
    ptop: float,
    dzmax: float,
    dzbot: float,
    dzstretch_s: float,
    dzstretch_u: float,
) -> np.ndarray:
    """WRF ``levels`` helper for ``auto_levels_opt=2``.

    The Fortran routine uses one-based ``zup``/``pup`` arrays and returns eta
    indexed 0:nlev, with eta[0]=1 at the surface and eta[nlev]=0 at the top.
    """

    tt = 290.0
    ztop = R_D * tt / G * np.log(1.0e5 / ptop)
    dz = dzbot
    zup = np.zeros(nlev, dtype=np.float64)
    eta = np.zeros(nlev + 1, dtype=np.float64)
    zup[0] = dz
    pup = 1.0e5 * np.exp(-G * zup[0] / R_D / tt)
    eta[0] = 1.0
    eta[1] = (pup - ptop) / (1.0e5 - ptop)
    isave = 1

    for fortran_i in range(1, nlev):
        stretch = dzstretch_u + (dzstretch_s - dzstretch_u) * max(
            (dzmax * 0.5 - dz) / (dzmax * 0.5), 0.0
        )
        dz = stretch * dz
        dztest = (ztop - zup[isave - 1]) / (nlev - isave)
        if dztest < dz:
            break
        isave = fortran_i + 1
        zup[fortran_i] = zup[fortran_i - 1] + dz
        pup = 1.0e5 * np.exp(-G * zup[fortran_i] / R_D / tt)
        eta[fortran_i + 1] = (pup - ptop) / (1.0e5 - ptop)
    else:
        raise ValueError("not enough eta levels to reach p_top")

    dz = (ztop - zup[isave - 1]) / (nlev - isave)
    if dz > 1.5 * dzmax:
        raise ValueError("upper eta levels may be too thick")
    for fortran_i in range(isave, nlev):
        zup[fortran_i] = zup[fortran_i - 1] + dz
        pup = 1.0e5 * np.exp(-G * zup[fortran_i] / R_D / tt)
        eta[fortran_i + 1] = (pup - ptop) / (1.0e5 - ptop)
    eta[nlev] = 0.0
    return eta


def _compute_c3f(hybrid_opt: int, etac: float, znw: np.ndarray) -> np.ndarray:
    if hybrid_opt in (0, 1):
        return znw.copy()
    if hybrid_opt != 2:
        raise ValueError(f"unsupported hybrid_opt={hybrid_opt}")

    b1 = 2.0 * etac**2 * (1.0 - etac)
    b2 = -etac * (4.0 - 3.0 * etac - etac**3)
    b3 = 2.0 * (1.0 - etac**3)
    b4 = -(1.0 - etac**2)
    b5 = (1.0 - etac) ** 4
    c3f = (b1 + b2 * znw + b3 * znw**2 + b4 * znw**3) / b5
    c3f = np.where(znw < etac, 0.0, c3f)
    c3f[0] = 1.0
    c3f[-1] = 0.0
    return c3f
