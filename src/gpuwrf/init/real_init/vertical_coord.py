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

    raise NotImplementedError("v0.4.0 S1 (Opus): compute_vertical_coord — frozen stub")
