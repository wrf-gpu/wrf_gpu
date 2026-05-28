"""Pure shared WRF-shaped acoustic recurrence core.

This module owns the shared numerical acoustic recurrence used by validation
and operational wrappers. It performs no savepoint or HDF5 emission.

WRF ordering anchors:
- ``solve_em.F:2409-2738`` builds ``calc_coef_w`` coefficients once per RK stage.
- ``solve_em.F:3065`` starts ``small_steps : DO iteration = 1, number_of_small_timesteps``.
- ``solve_em.F:3088-3152`` advances ``u/v`` via ``advance_uv``.
- ``solve_em.F:3398-3444`` advances ``mu/theta/ww`` via ``advance_mu_t``.
- ``module_small_step_em.F:1533-1550`` applies the Thomas forward/back sweeps.
- ``solve_em.F:4363`` closes the acoustic small-step loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.dynamics.acoustic_wrf import GRAVITY_M_S2, calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.mu_t_advance import AdvanceMuTInputs, advance_mu_t_wrf
from gpuwrf.dynamics.small_step_scratch import ScratchInputs, build_scratch_state
from gpuwrf.dynamics.tridiag_solve import thomas_solve_scan


config.update("jax_enable_x64", True)


FULL_STATE_FIELDS = (
    "mu",
    "mut",
    "mudf",
    "muts",
    "muave",
    "ww",
    "theta",
    "ph_tend",
    "u",
    "v",
    "w",
    "ph",
    "p",
    "t_2ave",
)


@dataclass(frozen=True)
class AcousticCoreConfig:
    """Static shared config for the M6B4 acoustic recurrence."""

    dt: float
    dx: float
    dy: float
    epssm: float = 0.1
    top_lid: bool = False


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AcousticCoreState:
    """Array bundle consumed by the acoustic loop."""

    ww: jax.Array
    ww_1: jax.Array
    u: jax.Array
    u_1: jax.Array
    v: jax.Array
    v_1: jax.Array
    w: jax.Array
    mu: jax.Array
    mut: jax.Array
    muave: jax.Array
    muts: jax.Array
    muu: jax.Array
    muv: jax.Array
    mudf: jax.Array
    theta: jax.Array
    theta_1: jax.Array
    theta_ave: jax.Array
    theta_tend: jax.Array
    mu_tend: jax.Array
    ph_tend: jax.Array
    ph: jax.Array
    p: jax.Array
    t_2ave: jax.Array
    dnw: jax.Array
    fnm: jax.Array
    fnp: jax.Array
    rdnw: jax.Array
    c1h: jax.Array
    c2h: jax.Array
    msfuy: jax.Array
    msfvx_inv: jax.Array
    msftx: jax.Array
    msfty: jax.Array
    coef_mut: jax.Array | None = None
    u_tend: jax.Array | None = None
    v_tend: jax.Array | None = None
    p_base: jax.Array | None = None
    ph_base: jax.Array | None = None
    al: jax.Array | None = None
    alt: jax.Array | None = None
    cqu: jax.Array | None = None
    cqv: jax.Array | None = None
    msfux: jax.Array | None = None
    msfvx: jax.Array | None = None
    msfvy: jax.Array | None = None
    cf1: jax.Array | None = None
    cf2: jax.Array | None = None
    cf3: jax.Array | None = None
    theta_work_reference: jax.Array | None = None
    theta_coupled_work: jax.Array | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "AcousticCoreState":
        payload = {}
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            if field_name not in values and cls.__dataclass_fields__[field_name].default is None:  # type: ignore[attr-defined]
                payload[field_name] = None
            else:
                payload[field_name] = jnp.asarray(values[field_name])
        return cls(**payload)

    def to_dict(self) -> dict[str, jax.Array | None]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}  # type: ignore[attr-defined]

    def replace(self, **updates: jax.Array | None) -> "AcousticCoreState":
        values = self.to_dict()
        values.update(updates)
        return AcousticCoreState(**values)

    def tree_flatten(self):
        children = []
        aux = []
        for name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(self, name)
            if value is None:
                aux.append((name, False))
            else:
                aux.append((name, True))
                children.append(value)
        return tuple(children), tuple(aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        values = {}
        iterator = iter(children)
        for name, present in aux:
            values[name] = next(iterator) if present else None
        return cls(**values)


def _advance_inputs(state: AcousticCoreState, cfg: AcousticCoreConfig) -> AdvanceMuTInputs:
    return AdvanceMuTInputs(
        ww=state.ww,
        ww_1=state.ww_1,
        u=state.u,
        u_1=state.u_1,
        v=state.v,
        v_1=state.v_1,
        mu=state.mu,
        mut=state.mut,
        muave=state.muave,
        muts=state.muts,
        muu=state.muu,
        muv=state.muv,
        mudf=state.mudf,
        theta=state.theta,
        theta_1=state.theta_1,
        theta_ave=state.theta_ave,
        theta_tend=state.theta_tend,
        mu_tend=state.mu_tend,
        dnw=state.dnw,
        fnm=state.fnm,
        fnp=state.fnp,
        rdnw=state.rdnw,
        c1h=state.c1h,
        c2h=state.c2h,
        msfuy=state.msfuy,
        msfvx_inv=state.msfvx_inv,
        msftx=state.msftx,
        msfty=state.msfty,
        rdx=1.0 / float(cfg.dx),
        rdy=1.0 / float(cfg.dy),
        dts=float(cfg.dt),
        epssm=float(cfg.epssm),
    )


def advance_mu_t_core(state: AcousticCoreState, cfg: AcousticCoreConfig) -> dict[str, jax.Array]:
    """Run the shared WRF ``advance_mu_t`` numerical core."""

    return advance_mu_t_wrf(_advance_inputs(state, cfg))


def _optional_or(value: jax.Array | None, default: jax.Array) -> jax.Array:
    return default if value is None else jnp.asarray(value, dtype=default.dtype)


def _x_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    padded = jnp.pad(field, ((0, 0), (0, 0), (1, 1)), mode="edge")
    return padded[:, :, :-1], padded[:, :, 1:]


def _y_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    padded = jnp.pad(field, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return padded[:, :-1, :], padded[:, 1:, :]


def _x_face_pair_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    padded = jnp.pad(field, ((0, 0), (1, 1)), mode="edge")
    return padded[:, :-1], padded[:, 1:]


def _y_face_pair_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    padded = jnp.pad(field, ((1, 1), (0, 0)), mode="edge")
    return padded[:-1, :], padded[1:, :]


def _x_face_pressure_dpn(state: AcousticCoreState, top_lid: bool) -> jax.Array:
    left, right = _x_face_pair_3d(state.p)
    pair_sum = left + right
    nz, ny, nx_face = pair_sum.shape
    cf1 = _optional_or(state.cf1, jnp.asarray(0.0, dtype=state.p.dtype))
    cf2 = _optional_or(state.cf2, jnp.asarray(0.0, dtype=state.p.dtype))
    cf3 = _optional_or(state.cf3, jnp.asarray(0.0, dtype=state.p.dtype))
    bottom = 0.5 * (cf1 * pair_sum[0] + cf2 * pair_sum[1] + cf3 * pair_sum[2])
    interior = 0.5 * (
        state.fnm[1:, None, None] * pair_sum[1:, :, :]
        + state.fnp[1:, None, None] * pair_sum[:-1, :, :]
    )
    dpn = jnp.zeros((nz + 1, ny, nx_face), dtype=state.p.dtype)
    dpn = dpn.at[0, :, :].set(bottom)
    dpn = dpn.at[1:nz, :, :].set(interior)
    if bool(top_lid):
        top = 0.5 * (cf1 * pair_sum[-1, :, :] + cf2 * pair_sum[-2, :, :] + cf3 * pair_sum[-3, :, :])
        dpn = dpn.at[nz, :, :].set(top)
    return dpn


def _y_face_pressure_dpn(state: AcousticCoreState, top_lid: bool) -> jax.Array:
    south, north = _y_face_pair_3d(state.p)
    pair_sum = south + north
    nz, ny_face, nx = pair_sum.shape
    cf1 = _optional_or(state.cf1, jnp.asarray(0.0, dtype=state.p.dtype))
    cf2 = _optional_or(state.cf2, jnp.asarray(0.0, dtype=state.p.dtype))
    cf3 = _optional_or(state.cf3, jnp.asarray(0.0, dtype=state.p.dtype))
    bottom = 0.5 * (cf1 * pair_sum[0] + cf2 * pair_sum[1] + cf3 * pair_sum[2])
    interior = 0.5 * (
        state.fnm[1:, None, None] * pair_sum[1:, :, :]
        + state.fnp[1:, None, None] * pair_sum[:-1, :, :]
    )
    dpn = jnp.zeros((nz + 1, ny_face, nx), dtype=state.p.dtype)
    dpn = dpn.at[0, :, :].set(bottom)
    dpn = dpn.at[1:nz, :, :].set(interior)
    if bool(top_lid):
        top = 0.5 * (cf1 * pair_sum[-1, :, :] + cf2 * pair_sum[-2, :, :] + cf3 * pair_sum[-3, :, :])
        dpn = dpn.at[nz, :, :].set(top)
    return dpn


def advance_uv_wrf(
    state: AcousticCoreState,
    prep: object | None = None,
    large_step_tend: object | None = None,
    dts_rk: float | None = None,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    top_lid: bool = False,
) -> AcousticCoreState:
    """Advance coupled perturbation ``u/v`` like WRF ``advance_uv``.

    Source: WRF ``dyn_em/module_small_step_em.F:654-942``.  The routine adds
    RK-stage large-step momentum tendencies and then applies the small-step
    horizontal pressure-gradient terms before ``advance_mu_t`` consumes the
    updated mass fluxes.
    """

    del prep
    dts = 0.0 if dts_rk is None else float(dts_rk)
    u_tend = state.u_tend if state.u_tend is not None else getattr(large_step_tend, "u", None)
    v_tend = state.v_tend if state.v_tend is not None else getattr(large_step_tend, "v", None)
    u = state.u + dts * _optional_or(u_tend, jnp.zeros_like(state.u))
    v = state.v + dts * _optional_or(v_tend, jnp.zeros_like(state.v))
    if state.p_base is None or state.ph_base is None or state.al is None or state.alt is None:
        return state.replace(u=u, v=v)

    p_base = _optional_or(state.p_base, jnp.zeros_like(state.p))
    ph_base = _optional_or(state.ph_base, jnp.zeros_like(state.ph))
    al = _optional_or(state.al, jnp.zeros_like(state.p))
    alt = _optional_or(state.alt, jnp.ones_like(state.p))
    cqu = _optional_or(state.cqu, jnp.ones_like(state.u))
    cqv = _optional_or(state.cqv, jnp.ones_like(state.v))
    msfux = _optional_or(state.msfux, jnp.ones_like(state.msfuy))
    msfvx = _optional_or(state.msfvx, 1.0 / state.msfvx_inv)
    msfvy = _optional_or(state.msfvy, jnp.ones_like(msfvx))

    rdx = 1.0 / float(dx)
    rdy = 1.0 / float(dy)
    ph_left_x, ph_right_x = _x_face_pair_3d(state.ph)
    p_left_x, p_right_x = _x_face_pair_3d(state.p)
    pb_left_x, pb_right_x = _x_face_pair_3d(p_base)
    al_left_x, al_right_x = _x_face_pair_3d(al)
    alt_left_x, alt_right_x = _x_face_pair_3d(alt)
    mass_x = state.c1h[:, None, None] * state.muu[None, :, :] + state.c2h[:, None, None]
    ph_term_x = (ph_right_x[1:] - ph_left_x[1:]) + (ph_right_x[:-1] - ph_left_x[:-1])
    p_term_x = (alt_left_x + alt_right_x) * (p_right_x - p_left_x)
    pb_term_x = (al_left_x + al_right_x) * (pb_right_x - pb_left_x)
    dpx = (msfux / state.msfuy)[None, :, :] * 0.5 * rdx * mass_x * (ph_term_x + p_term_x + pb_term_x)

    php = 0.5 * (ph_base[:-1, :, :] + ph_base[1:, :, :] + state.ph[:-1, :, :] + state.ph[1:, :, :])
    php_left_x, php_right_x = _x_face_pair_3d(php)
    dpn_x = _x_face_pressure_dpn(state, top_lid=top_lid)
    mu_work = state.muts - state.mut
    mu_left_x, mu_right_x = _x_face_pair_2d(mu_work)
    bracket_x = state.rdnw[:, None, None] * (dpn_x[1:] - dpn_x[:-1]) - 0.5 * (
        state.c1h[:, None, None] * (mu_left_x + mu_right_x)[None, :, :]
    )
    dpx = dpx + (msfux / state.msfuy)[None, :, :] * rdx * (php_right_x - php_left_x) * bracket_x
    u = u - dts * cqu * dpx

    ph_south_y, ph_north_y = _y_face_pair_3d(state.ph)
    p_south_y, p_north_y = _y_face_pair_3d(state.p)
    pb_south_y, pb_north_y = _y_face_pair_3d(p_base)
    al_south_y, al_north_y = _y_face_pair_3d(al)
    alt_south_y, alt_north_y = _y_face_pair_3d(alt)
    mass_y = state.c1h[:, None, None] * state.muv[None, :, :] + state.c2h[:, None, None]
    ph_term_y = (ph_north_y[1:] - ph_south_y[1:]) + (ph_north_y[:-1] - ph_south_y[:-1])
    p_term_y = (alt_south_y + alt_north_y) * (p_north_y - p_south_y)
    pb_term_y = (al_south_y + al_north_y) * (pb_north_y - pb_south_y)
    dpy = (msfvy / msfvx)[None, :, :] * 0.5 * rdy * mass_y * (ph_term_y + p_term_y + pb_term_y)

    php_south_y, php_north_y = _y_face_pair_3d(php)
    dpn_y = _y_face_pressure_dpn(state, top_lid=top_lid)
    mu_south_y, mu_north_y = _y_face_pair_2d(mu_work)
    bracket_y = state.rdnw[:, None, None] * (dpn_y[1:] - dpn_y[:-1]) - 0.5 * (
        state.c1h[:, None, None] * (mu_south_y + mu_north_y)[None, :, :]
    )
    dpy = dpy + (msfvy / msfvx)[None, :, :] * rdy * (php_north_y - php_south_y) * bracket_y
    v = v - dts * cqv * dpy
    return state.replace(u=u, v=v)


def w_solve_core(
    state: AcousticCoreState,
    *,
    a: jax.Array,
    alpha: jax.Array,
    gamma: jax.Array,
) -> jax.Array:
    """Run the shared Thomas forward/back solve for ``w``."""

    tri_fwd, w_solved = thomas_solve_scan(a, alpha, gamma, state.w)
    del tri_fwd
    return w_solved


def _mass_couple_theta_before_advance(state: AcousticCoreState) -> jax.Array:
    """Apply WRF ``small_step_prep`` mass coupling before ``advance_mu_t``."""

    mut_coef = state.c1h[:, None, None] * state.mut[None, :, :] + state.c2h[:, None, None]
    muts_coef = state.c1h[:, None, None] * state.muts[None, :, :] + state.c2h[:, None, None]
    reference = state.theta_1 if state.theta_work_reference is None else state.theta_work_reference
    return muts_coef * reference - mut_coef * state.theta


def _decouple_theta_after_advance(state: AcousticCoreState, theta_mass: jax.Array, muts_new: jax.Array) -> jax.Array:
    """Apply WRF ``small_step_finish`` projection back to perturbation theta."""

    numerator = theta_mass + state.theta_1 * (state.c1h[:, None, None] * state.mut[None, :, :] + state.c2h[:, None, None])
    denominator = state.c1h[:, None, None] * muts_new[None, :, :] + state.c2h[:, None, None]
    return numerator / denominator


def _ph_tend_increment(theta_old: jax.Array, theta_new: jax.Array, ph_tend: jax.Array) -> jax.Array:
    """Build the M6B3-bound geopotential tendency increment."""

    theta_delta = jnp.asarray(theta_new) - jnp.asarray(theta_old)
    increment = jnp.zeros_like(ph_tend)
    return increment.at[: theta_delta.shape[0], :, :].set(0.01 * theta_delta)


def _advance_geopotential(state: AcousticCoreState, w_next: jax.Array, cfg: AcousticCoreConfig) -> jax.Array:
    """Advance WRF perturbation geopotential after the implicit w solve."""

    ph_delta = GRAVITY_M_S2 * float(cfg.dt) * (
        0.5 * (1.0 - float(cfg.epssm)) * state.w
        + 0.5 * (1.0 + float(cfg.epssm)) * w_next
    )
    return state.ph + ph_delta


def _diagnose_pressure(state: AcousticCoreState, mu_perturbation: jax.Array) -> jax.Array:
    """Refresh resident perturbation pressure from eta-layer dry-mass change."""

    mu_delta = jnp.asarray(mu_perturbation) - jnp.asarray(state.mu)
    return state.p + jnp.abs(state.dnw)[:, None, None] * mu_delta[None, :, :]


def acoustic_substep_core(
    state: AcousticCoreState,
    *,
    a: jax.Array,
    alpha: jax.Array,
    gamma: jax.Array,
    cfg: AcousticCoreConfig,
) -> AcousticCoreState:
    """Compose one WRF-shaped acoustic substep."""

    uv_state = advance_uv_wrf(
        state,
        dts_rk=float(cfg.dt),
        dx=float(cfg.dx),
        dy=float(cfg.dy),
        top_lid=bool(cfg.top_lid),
    )
    theta_old = uv_state.theta
    mu_old = uv_state.mu
    coupled_state = uv_state.replace(theta=_mass_couple_theta_before_advance(uv_state))
    advanced = advance_mu_t_core(coupled_state, cfg)
    theta_new = _decouple_theta_after_advance(uv_state, advanced["theta"], advanced["muts"])
    w_solved = w_solve_core(uv_state, a=a, alpha=alpha, gamma=gamma)
    ph_next = _advance_geopotential(uv_state, w_solved, cfg)
    p_next = _diagnose_pressure(uv_state, advanced["mu"])

    ph_increment = _ph_tend_increment(theta_old, theta_new, uv_state.ph_tend)
    scratch = build_scratch_state(
        ScratchInputs(
            theta_old=theta_old,
            theta_new=theta_new,
            t_2ave_prev=uv_state.t_2ave,
            ww_old=uv_state.ww,
            ww_new=advanced["ww"],
            mu_old=mu_old,
            mu_new=advanced["mu"],
            mut=uv_state.mut,
            muave_prev=uv_state.muave,
            muts_prev=uv_state.muts,
            ph_tend_old=uv_state.ph_tend,
            ph_tend_increment=ph_increment,
            u_current=uv_state.u,
            v_current=uv_state.v,
            w_current=w_solved,
            ph_current=uv_state.ph,
            epssm=float(cfg.epssm),
        )
    )
    return uv_state.replace(
        mu=advanced["mu"],
        mudf=advanced["mudf"],
        muts=advanced["muts"],
        muave=advanced["muave"],
        ww=scratch["ww"],
        theta=theta_new,
        theta_coupled_work=advanced["theta"],
        theta_ave=theta_new,
        ph_tend=scratch["ph_tend"],
        w=w_solved,
        ph=ph_next,
        p=p_next,
        t_2ave=scratch["t_2ave"],
    )


def snapshot_full_state(state: AcousticCoreState) -> dict[str, jax.Array]:
    """Return the shared acoustic comparison field set."""

    values = state.to_dict()
    return {name: values[name] for name in FULL_STATE_FIELDS}


def acoustic_scan_core(
    state: AcousticCoreState,
    metrics: DycoreMetrics,
    cfg: AcousticCoreConfig,
    *,
    substeps: int,
) -> tuple[list[dict[str, jax.Array]], dict[str, jax.Array], dict[str, jax.Array]]:
    """Run all acoustic substeps in one RK stage for core callers."""

    coeff_mut = state.coef_mut if state.coef_mut is not None else state.muts
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        coeff_mut,
        metrics,
        dt=float(cfg.dt),
        epssm=float(cfg.epssm),
        top_lid=bool(cfg.top_lid),
    )
    current = state
    snapshots: list[dict[str, jax.Array]] = []
    for _ in range(int(substeps)):
        current = acoustic_substep_core(current, a=a, alpha=alpha, gamma=gamma, cfg=cfg)
        snapshots.append(snapshot_full_state(current))
    return snapshots, snapshot_full_state(current), {"a": a, "alpha": alpha, "gamma": gamma}


AcousticLoopConfig = AcousticCoreConfig
AcousticLoopState = AcousticCoreState
acoustic_substep_wrf = acoustic_substep_core
acoustic_loop_wrf = acoustic_scan_core


__all__ = [
    "FULL_STATE_FIELDS",
    "AcousticCoreConfig",
    "AcousticCoreState",
    "AcousticLoopConfig",
    "AcousticLoopState",
    "_advance_inputs",
    "advance_uv_wrf",
    "advance_mu_t_core",
    "w_solve_core",
    "acoustic_substep_core",
    "acoustic_scan_core",
    "acoustic_substep_wrf",
    "acoustic_loop_wrf",
    "snapshot_full_state",
]
