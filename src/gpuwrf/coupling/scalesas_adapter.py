"""cu_physics=4 (scale-aware GFS SAS, ARW) State coupling: WRF driver inputs, cadence, held rates.

WRF call chain mirrored here (WRF 4.7.1):
  * ``first_rk_step_part1`` -> ``cumulus_driver`` runs the scheme when ``itimestep==1`` or
    ``MOD(itimestep, STEPCU)==0`` (adaptive dt is fatal for SAS); otherwise the previous
    ``RTHCUTEN/RQVCUTEN/RQCCUTEN/RQICUTEN`` and ``PRATEC`` stay in the grid (held rates).
  * ``CU_SCALESAS`` reads the ``phy_prep`` arrays: ``t_phy`` (dry theta * pi_phy), ``p_hyd`` /
    ``p_hyd_w`` (hydrostatic mass/face pressure, ``PSFC=P8W(kms)``), ``pi_phy``, ``rho``
    (``(1+qv)/alt``), ``dz8w``, mass-point ``u_phy/v_phy``, face ``w_2``, ``XLAND``,
    ``DX2D=dx`` (ARW ``compute_2d_dx_area`` ``#if 0`` branch) and ``DY`` -- see the DY note in
    ``physics.cumulus_scalesas`` (ARW never passes ``DYNMM``; the grid ``dy`` is used).
  * ``phy_cu_ten`` (SAS case) adds only RTHCUTEN/RQVCUTEN/RQCCUTEN/RQICUTEN (no momentum);
    ``RAINC += PRATEC*DT`` every step (``module_physics_addtendc.F:2291``).

Held rates use the KF carry layout ``(RTH, RQV, RQC, RQR, RQI, RQS, PRATEC)`` (``RQR=RQS=0`` for
SAS) so every existing ``cumulus_tendencies`` consumer (Noah-MP convective precipitation reads
slot 6, REAL carry casting, checkpoints) sees the same structure.

Coupling caveat (same as the default KF path): the held rates are applied as step-entry
increments ``x += dt*R`` (theta through WRF ``conv_t_tendf_to_moist`` at the time-n state)
instead of entering the RK3 ``rt_tendf``/``moist_tendf`` sums.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import (
    _dry_theta_view,
    _output_dtype,
    _theta_m_tendency_from_dry,
    _u_mass,
    _v_mass,
    _wrf_hydrostatic_pressure_profiles_from_state,
    _wrf_phy_prep_rho_from_state,
)
from gpuwrf.physics.cumulus_scalesas import cu_scalesas_columns

_REAL = jnp.float32
_G_WRF = 9.81      # module_model_constants g (phy_prep z_at_w = (ph+phb)/g)
_RCP_WRF = 2.0 / 7.0  # r_d/cp with cp = 7*r_d/2


def scalesas_rates(state, dt: float, grid=None, *, stepcu: int = 1):
    """Run ``CU_SCALESAS`` on every column of ``state``; return WRF REAL driver outputs.

    Returns ``(RTHCUTEN, RQVCUTEN, RQCCUTEN, RQICUTEN, PRATEC)`` with the 3-D rates shaped
    ``(nz, ny, nx)`` and ``PRATEC`` ``(ny, nx)`` (mm/s), all float32 like WRF.
    """

    nz, ny, nx = state.theta.shape
    theta_dry = jnp.asarray(_dry_theta_view(state), _REAL)
    p_full = jnp.asarray(state.p, _REAL)
    pi_phy = (p_full / jnp.asarray(1.0e5, _REAL)) ** jnp.asarray(_RCP_WRF, _REAL)
    t_phy = theta_dry * pi_phy
    z_at_w = jnp.asarray(state.ph, _REAL) / jnp.asarray(_G_WRF, _REAL)
    dz8w = z_at_w[1:] - z_at_w[:-1]
    metrics = getattr(grid, "metrics", None)
    if metrics is not None:
        p_hyd, p_hyd_w, _psfc = _wrf_hydrostatic_pressure_profiles_from_state(state, metrics, output_dtype=_REAL)
        psfc = jnp.asarray(p_hyd_w, _REAL)[0]
        rho = _wrf_phy_prep_rho_from_state(state, metrics, output_dtype=_REAL)
    else:  # analytic callers without hybrid metrics: full pressure, EOS total density
        p_hyd = p_full
        psfc = p_full[0] + 0.5 * (p_full[0] - p_full[1])
        qv = jnp.asarray(state.qv, _REAL)
        rho = p_full / (jnp.asarray(287.0, _REAL) * t_phy * (1.0 + jnp.asarray(0.608, _REAL) * qv)) * (1.0 + qv)
    w = jnp.asarray(state.w, _REAL)
    if w.shape[0] == nz:  # mass-level w fallback: pad a top face
        w = jnp.concatenate([w, w[-1:]], axis=0)
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    dy = float(grid.projection.dy_m) if grid is not None else dx

    def cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(jnp.asarray(field3d, _REAL), 0, -1).reshape(ny * nx, field3d.shape[0])

    out = cu_scalesas_columns(
        cols(t_phy), cols(state.qv), cols(state.qc), cols(state.qi), cols(p_hyd), cols(pi_phy),
        cols(rho), cols(dz8w), cols(_u_mass(state)), cols(_v_mass(state)), cols(w),
        jnp.asarray(psfc, _REAL).reshape(ny * nx), jnp.asarray(state.xland, _REAL).reshape(ny * nx),
        jnp.full((ny * nx,), dx, _REAL), dy, float(dt), int(stepcu), literals="r4",
    )

    def back(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    return (back(out["RTHCUTEN"]), back(out["RQVCUTEN"]), back(out["RQCCUTEN"]),
            back(out["RQICUTEN"]), out["PRATEC"].reshape(ny, nx))


def _held_layout(rates):
    """(RTH, RQV, RQC, RQI, PRATEC) -> KF carry layout (RTH, RQV, RQC, RQR, RQI, RQS, PRATEC)."""

    rth, rqv, rqc, rqi, pratec = rates
    zero3 = jnp.zeros_like(rth)
    return (rth, rqv, rqc, zero3, rqi, zero3, pratec)


def initial_scalesas_tendencies(state):
    """Held WRF R*CUTEN/PRATEC seed (scalesasinit zeroes them on a cold start), KF layout."""

    zero3 = jnp.zeros(state.theta.shape, _REAL)
    return (zero3,) * 6 + (jnp.zeros(state.theta.shape[1:], _REAL),)


def apply_scalesas_rates(state, rates, dt: float, entry_state=None):
    """``x += dt*R`` for held SAS rates (KF layout); theta via ``conv_t_tendf_to_moist`` at time n."""

    entry = state if entry_state is None else entry_state
    rth, rqv, rqc, _rqr, rqi, _rqs, pratec = rates
    theta_dtype = jnp.asarray(state.theta).dtype
    dt_f = float(dt)
    theta_next = (
        jnp.asarray(state.theta, theta_dtype)
        + dt_f * _theta_m_tendency_from_dry(rth, rqv, entry.theta, entry.qv, theta_dtype)
    ).astype(_output_dtype(state, "theta"))
    return state.replace(
        theta=theta_next,
        qv=(state.qv + dt_f * rqv).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * rqc).astype(_output_dtype(state, "qc")),
        qi=(state.qi + dt_f * rqi).astype(_output_dtype(state, "qi")),
        rainc_acc=(state.rainc_acc + dt_f * jnp.asarray(pratec, state.rainc_acc.dtype)).astype(
            _output_dtype(state, "rainc_acc")),
    )


def scalesas_cadence_step(entry_state, state, held, dt: float, grid, *, stepcu: int, itimestep):
    """WRF cumulus_driver gate on the one-based ``itimestep``; returns ``(state, held_rates)``.

    ``entry_state`` is the time-n physics entry (what every WRF physics driver reads);
    ``state`` is the slot input the held rates are applied to. ``held=None`` (carry without a
    seeded cumulus leaf, e.g. single-step harnesses) runs the scheme unconditionally.
    """

    if held is None:
        rates = _held_layout(scalesas_rates(entry_state, dt, grid, stepcu=stepcu))
        return apply_scalesas_rates(state, rates, dt, entry_state=entry_state), None

    stepcu = max(1, int(stepcu))
    step = jnp.asarray(itimestep, dtype=jnp.int32)
    run_cu = (step == 1) | (jnp.mod(step, stepcu) == 0)
    rates = jax.lax.cond(
        run_cu,
        lambda _u: _held_layout(scalesas_rates(entry_state, dt, grid, stepcu=stepcu)),
        lambda _u: tuple(jnp.asarray(h, _REAL) for h in held),
        None,
    )
    return apply_scalesas_rates(state, rates, dt, entry_state=entry_state), rates


__all__ = [
    "scalesas_rates",
    "initial_scalesas_tendencies",
    "apply_scalesas_rates",
    "scalesas_cadence_step",
]
