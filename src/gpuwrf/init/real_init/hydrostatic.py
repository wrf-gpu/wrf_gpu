"""S1 (Opus) — hydrostatic balance: perturbation P / PH / AL / ALT.

FROZEN ENTRY SIGNATURE. Implements the real.exe final hydrostatic-balance block
(module_initialize_real.F:3876-4044): given the interpolated dynamics + the dry
base state, produce the perturbation pressure, geopotential, and inverse density
that make the column hydrostatically consistent with the WRF model equations.

Algorithm (faithful spec):
  t_2 := theta - t0 already (perturbation theta).
  MU_2 = MU0 - MUB                                                 (:3881)
  Top-down integration of perturbation dry pressure p (real.exe:3902-3958):
    at k=top:  qtot=sum(moist); qvf2=1/(1+qtot); qvf1=qtot*qvf2;
       p[top] = -0.5*((c1f*MU_2)+qvf1*(c1f*MUB+c2f))/rdnw/qvf2     (:3935)
    downward: p[k]=p[k+1] - ((c1f*MU_2)+qvf1*(c1f*MUB+c2f))/qvf2/rdn[k+1] (:3953)
    alt[k]=(r_d/p1000mb)*(t_2[k]+t0)*qvf*((p+pb)/p1000mb)^cvpm     (:3937/:3955)
    al[k]=alt[k]-alb[k]; p_hyd[k]=p[k]+pb[k]                       (:3939-3940)
  Then geopotential ph_2 from hydrostatic eq (real.exe:3965-3994), the
  hybrid_opt branch using c1h/c2h/c3h/c4h + the dZ=-al*p*dlog(p) form (:3974-3987).
  (Optionally re-diagnose al from ph_2 via :4001-4013 and recompute p :4031-4036
  to match the model's exact post-substep relation; reproduce whichever branch
  real.exe takes for this config — confirm at impl by reading :3960-4040.)

Oracle: wrfinput P / PH / AL(diag) / T / MU(=MU_2); tols ``WRFINPUT_TOLS``. This
is THE hour-0-critical check: a small PH/P error here propagates into every
downstream forecast comparison.

FILE OWNERSHIP: S1 exclusive.
"""

from __future__ import annotations

from gpuwrf.init.real_init.types import (
    BaseStateColumns,
    DynamicsInit,
    RealInitConfig,
    VerticalCoord1D,
)


def balance(
    config: RealInitConfig,
    vcoord: VerticalCoord1D,
    base: BaseStateColumns,
    dyn_seed: DynamicsInit,
) -> DynamicsInit:
    """Returns ``dyn_seed`` with hydrostatically-balanced p/ph/al/alt/p_hyd/mu.

    Takes the pre-balance interpolated dynamics from :func:`vinterp.vertical_interpolate`
    plus the dry base state from :func:`base_state.compute_base_state`, and fills
    the final perturbation pressure, geopotential, and inverse densities.
    """

    raise NotImplementedError("v0.4.0 S1 (Opus): hydrostatic.balance — frozen stub")
