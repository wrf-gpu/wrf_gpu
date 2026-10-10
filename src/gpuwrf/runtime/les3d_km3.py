"""Operational adapter for WRF ``diff_opt=2`` / ``km_opt=3`` (3-D Smagorinsky).

Builds the WRF RK1-frozen diffusion bundle from a time-t :class:`State` with the
literal operator in :mod:`gpuwrf.dynamics.les3d_smagorinsky`, mirroring the
caller wiring of ``module_first_rk_step_part2.F`` (diff_opt=2 block):

* ``thp`` = ``grid%t_2`` (theta_m - T0 with use_theta_m=1), ``th_phy``/``t_phy``/
  ``p_phy``/``p8w``/``t8w``/``rho`` from the port's ``phy_prep`` transcriptions,
  ``ph``/``phb`` split as WRF stores them, the full moist + other-scalar set.
* ``vertical_diffusion_2`` only when ``bl_pbl_physics == 0`` (WRF gate), with the
  surface stress/heat/moisture fluxes of ``isfflx`` from ``ustar``/``hfx``/``qfx``.
* Dry outputs are returned already folded the way ``rk_addtend_dry`` consumes
  them (``ru_tendf/msfuy``, ``rv_tendf*(1/msfvx)``, ``rw_tendf/msfty``,
  ``t_tendf/msfty``); moist/scalar increments are the unscaled ``moist_tend``/
  ``scalar_tend`` (``sc_tend``) contributions reused by every ``rk_update_scalar``.

CPU-oracle-qualified against pristine WRF REAL4 (proofs/o1_smag3d); GPU and
coupled-forecast qualification pending.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State
from gpuwrf.dynamics.les3d_smagorinsky import Les3dConfig, Les3dInputs, les3d_smagorinsky_tendencies

# WRF moist array order (qv, qc, qr, qi, qs, qg) followed by the other scalars.
LES3D_MOIST_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
LES3D_SCALAR_SPECIES = ("Ni", "Nr")
_T0 = 300.0


class Les3dKm3Bundle(NamedTuple):
    """RK1-frozen km_opt=3 bundle in the operational large-step space."""

    u_t: jax.Array
    v_t: jax.Array
    w_t: jax.Array
    theta_t: jax.Array
    scalar_sc: dict


def les3d_km3_active(namelist) -> bool:
    return int(namelist.diff_opt) == 2 and int(namelist.km_opt) == 3


def _boundary(namelist) -> str:
    from gpuwrf.runtime.operational_mode import _acoustic_lateral_bc_flags

    periodic, specified, nested = _acoustic_lateral_bc_flags(namelist)
    if specified or nested:
        return "specified"
    if periodic:
        return "periodic"
    raise ValueError("km_opt=3 supports specified/nested or doubly periodic lateral boundaries")


def les3d_km3_forward_tendencies(reference: State, namelist, *, base_state=None) -> Les3dKm3Bundle:
    """WRF diff_opt=2/km_opt=3 RK1 bundle from the time-t ``reference`` state."""

    from gpuwrf.coupling.physics_couplers import (
        _dry_temperature_from_state,
        _dry_theta_view,
        _wrf_phy_prep_p8w,
        _wrf_phy_prep_rho_from_state,
        _wrf_phy_prep_temperature_interfaces,
    )

    metrics = namelist.metrics
    dtype = jnp.asarray(reference.theta).dtype
    a = lambda x: jnp.asarray(x, dtype=dtype)  # noqa: E731
    dx = np.float32(namelist.grid.projection.dx_m)
    dy = np.float32(namelist.grid.projection.dy_m)
    ph_total = a(reference.ph_total)
    ph = a(reference.ph_perturbation)
    phb = a(base_state.phb) if base_state is not None and getattr(base_state, "phb", None) is not None else ph_total - ph
    p_phy = a(reference.p_total)
    th_phy = a(_dry_theta_view(reference))
    t_phy = a(_dry_temperature_from_state(reference))
    z_at_w = ph_total / np.float32(9.81)
    p8w = a(_wrf_phy_prep_p8w(p_phy, z_at_w, metrics.fnm, metrics.fnp))
    t8w = a(_wrf_phy_prep_temperature_interfaces(t_phy, reference, metrics))
    rho = a(_wrf_phy_prep_rho_from_state(reference, metrics, output_dtype=dtype))
    names = [n for n in LES3D_MOIST_SPECIES + LES3D_SCALAR_SPECIES if getattr(reference, n, None) is not None]
    moist = tuple(a(getattr(reference, n)) for n in names)
    pos = {n: i for i, n in enumerate(names)}
    vertical = int(namelist.bl_pbl_physics) == 0
    ustar = getattr(reference, "ustar", None)
    hfx = getattr(reference, "hfx", None)
    qfx = getattr(reference, "qfx", None)
    if vertical and int(getattr(namelist, "isfflx", 1)) in (1, 2) and (ustar is None or hfx is None or qfx is None):
        raise ValueError("km_opt=3 with bl_pbl_physics=0 and isfflx=1/2 needs State ustar/hfx/qfx")
    scalar = lambda x: jnp.asarray(x, dtype=dtype).reshape(-1)[0]  # noqa: E731  (trace-safe)
    inp = Les3dInputs(
        u=a(reference.u), v=a(reference.v), w=a(reference.w),
        thp=a(reference.theta) - np.float32(_T0), th_phy=th_phy, t_phy=t_phy, p_phy=p_phy, p8w=p8w, t8w=t8w,
        ph=ph, phb=phb, rho=rho, moist=moist,
        msftx=a(metrics.msftx), msfty=a(metrics.msfty), msfux=a(metrics.msfux), msfuy=a(metrics.msfuy),
        msfvx=a(metrics.msfvx), msfvy=a(metrics.msfvy),
        fnm=a(metrics.fnm), fnp=a(metrics.fnp), dn=a(metrics.dn), dnw=a(metrics.dnw),
        cf1=scalar(metrics.cf1), cf2=scalar(metrics.cf2), cf3=scalar(metrics.cf3),
        rdx=np.float32(1.0) / dx, rdy=np.float32(1.0) / dy, dx=dx, dy=dy, dt=np.float32(namelist.dt_s),
        ust=None if ustar is None else a(ustar), hfx=None if hfx is None else a(hfx),
        qfx=None if qfx is None else a(qfx),
    )
    cfg = Les3dConfig(
        boundary=_boundary(namelist),
        isotropic=int(getattr(namelist, "mix_isotropic", 0)),
        c_s=float(namelist.c_s),
        mix_upper_bound=float(getattr(namelist, "mix_upper_bound", 0.1)),
        vertical=vertical,
        isfflx=int(getattr(namelist, "isfflx", 1)),
        tke_drag_coefficient=float(getattr(namelist, "tke_drag_coefficient", 0.0)),
        tke_heat_flux=float(getattr(namelist, "tke_heat_flux", 0.0)),
        use_theta_m=1,
        mix_full_fields=False,
        iqv=pos["qv"],
        iqc=pos.get("qc"),
        iqi=pos.get("qi"),
    )
    out = les3d_smagorinsky_tendencies(inp, cfg)
    msfuy = a(metrics.msfuy)[None]
    msfvx_inv = (np.float32(1.0) / a(metrics.msfvx))[None]
    msfty = a(metrics.msfty)[None]
    return Les3dKm3Bundle(
        u_t=out.ru_tendf / msfuy,
        v_t=out.rv_tendf * msfvx_inv,
        w_t=out.rw_tendf / msfty,
        theta_t=out.t_tendf / msfty,
        scalar_sc={n: out.moist_tendf[i] for n, i in pos.items()},
    )


__all__ = ["LES3D_MOIST_SPECIES", "LES3D_SCALAR_SPECIES", "Les3dKm3Bundle", "les3d_km3_active",
           "les3d_km3_forward_tendencies"]
