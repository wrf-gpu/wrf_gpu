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
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.advance_w import (
    GRAVITY_M_S2,
    W_ALPHA,
    W_BETA,
    advance_w_wrf,
    dry_cqw,
    pg_buoy_w_dry,
)
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_step
from gpuwrf.dynamics.mu_t_advance import AdvanceMuTInputs, advance_mu_t_wrf
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
    """Static shared config for the M6B4 acoustic recurrence.

    Damping fields (Block 1) carry the WRF namelist damping controls into the
    acoustic small-step.  Defaults are OFF so existing callers/tests keep the
    bare-core behaviour; the operational path sets them from the Gen2 namelist
    (``w_damping=1``, ``damp_opt=3``, ``zdamp=5000``, ``dampcoef=0.2``).
    """

    dt: float
    dx: float
    dy: float
    epssm: float = 0.1
    top_lid: bool = False
    # WRF damping (module_small_step_em.F:1559-1572, module_big_step_utilities_em.F:2766-2770)
    w_damping: int = 0
    damp_opt: int = 0
    dampcoef: float = 0.0
    zdamp: float = 5000.0
    w_alpha: float = W_ALPHA
    w_crit_cfl: float = W_BETA


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
    # WRF advance_w inputs (F7 acoustic core)
    c2a: jax.Array | None = None
    cqw: jax.Array | None = None
    c1f: jax.Array | None = None
    c2f: jax.Array | None = None
    rdn: jax.Array | None = None
    phb: jax.Array | None = None
    ph_1: jax.Array | None = None
    ht: jax.Array | None = None
    pm1: jax.Array | None = None
    ru_m: jax.Array | None = None
    rv_m: jax.Array | None = None
    ww_m: jax.Array | None = None
    # Large-step ABSOLUTE perturbation pressure for the pg_buoy_w buoyancy source
    # (WRF rk_step_prep diagnostic p'; module_em.F:184-225 -> rk_tendency pg_buoy_w
    # :1354-1368).  The acoustic small-step ``p`` is a delta-from-reference and is
    # ~0 for a static balanced perturbation, so the buoyancy must use this
    # absolute p' once per RK stage rather than the substep delta.
    p_buoy: jax.Array | None = None
    # Uncoupled physical perturbation w from small_step_prep (WRF w_save, :272),
    # required by the damp_opt=3 implicit Rayleigh damping in advance_w.
    w_save: jax.Array | None = None
    # F7G: the large-step vertical PGF/buoyancy tendency ``rw_tend`` built ONCE per
    # RK stage from the stage ``grid%p``/``mu`` (WRF module_em.F:1361-1368 ->
    # pg_buoy_w, module_big_step_utilities_em.F:2553-2572) and carried UNCHANGED
    # through every acoustic substep.  WRF does NOT recompute pg_buoy_w from the
    # live small-step ``calc_p_rho`` work pressure each substep; that was the F7F
    # workaround (gpt-council-findings.md §2/§3.3).  When None, the substep falls
    # back to the legacy per-substep recompute (bare-core/oracle callers only).
    rw_tend_pg_buoy: jax.Array | None = None

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
    emdiv: float = 0.0,
) -> AcousticCoreState:
    """Advance coupled perturbation ``u/v`` like WRF ``advance_uv``.

    Source: WRF ``dyn_em/module_small_step_em.F:654-942``.  The routine adds
    RK-stage large-step momentum tendencies and then applies the small-step
    horizontal pressure-gradient terms before ``advance_mu_t`` consumes the
    updated mass fluxes.  The external-mode divergence-damping term
    (``mudf``/``emdiv``; WRF ``:808-810``, ``:866-869``, ``:879-880``,
    ``:940-942``) is added when ``emdiv > 0`` and ``state.mudf`` is the WRF
    in-loop divergence-damping mass tendency from ``advance_mu_t``.
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
    if float(emdiv) != 0.0:
        # WRF :808-810, :868 -- mudf_xy = -emdiv*dx*(mudf(i)-mudf(i-1))/msfuy ;
        # u += c1h(k)*mudf_xy.  mudf is a (ny, nx) mass tendency from advance_mu_t.
        mudf_l_x, mudf_r_x = _x_face_pair_2d(state.mudf)
        mudf_xy_u = -float(emdiv) * float(dx) * (mudf_r_x - mudf_l_x) / state.msfuy
        u = u + state.c1h[:, None, None] * mudf_xy_u[None, :, :]

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
    if float(emdiv) != 0.0:
        # WRF :879-880, :942 -- mudf_xy = -emdiv*dy*(mudf(j)-mudf(j-1))*msfvx_inv ;
        # v += c1h(k)*mudf_xy.
        mudf_s_y, mudf_n_y = _y_face_pair_2d(state.mudf)
        msfvx_inv = state.msfvx_inv
        mudf_xy_v = -float(emdiv) * float(dy) * (mudf_n_y - mudf_s_y) * msfvx_inv
        v = v + state.c1h[:, None, None] * mudf_xy_v[None, :, :]
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


def _decouple_theta_for_finish(state: AcousticCoreState, theta_mass: jax.Array, muts_new: jax.Array) -> jax.Array:
    """Project coupled theta work back to perturbation theta (diagnostic view).

    This mirrors WRF ``small_step_finish`` theta reconstruction for the
    operational physical-theta carry, but the canonical decouple happens inside
    :func:`gpuwrf.dynamics.core.small_step_finish.small_step_finish_wrf`.
    """

    numerator = theta_mass + state.theta_1 * (state.c1h[:, None, None] * state.mut[None, :, :] + state.c2h[:, None, None])
    denominator = state.c1h[:, None, None] * muts_new[None, :, :] + state.c2h[:, None, None]
    return numerator / denominator


# Back-compat alias for callers/tests that referenced the old name.
_decouple_theta_after_advance = _decouple_theta_for_finish


def acoustic_substep_core(
    state: AcousticCoreState,
    *,
    a: jax.Array,
    alpha: jax.Array,
    gamma: jax.Array,
    cfg: AcousticCoreConfig,
    cqw: jax.Array | None = None,
    emdiv: float = 0.01,
    smdiv: float = 0.1,
) -> AcousticCoreState:
    """Compose one WRF-faithful acoustic substep.

    WRF cadence (``solve_em.F:3065-4206``): ``advance_uv`` -> ``advance_mu_t``
    -> ``advance_w`` -> ``sumflux`` -> ``calc_p_rho(step=iteration)``.  All of
    ``u``, ``v``, ``w``, ``ph``, ``theta`` are the *coupled* small-step work
    arrays; ``mu`` is the perturbation dry-mass work array.  ``a/alpha/gamma``
    are the ``calc_coef_w`` coefficients built once per RK stage with real
    ``c2a``/``cqw``.
    """

    # --- 1. advance_uv (with external-mode divergence damping) ---
    uv_state = advance_uv_wrf(
        state,
        dts_rk=float(cfg.dt),
        dx=float(cfg.dx),
        dy=float(cfg.dy),
        top_lid=bool(cfg.top_lid),
        emdiv=float(emdiv),
    )

    # --- 2. advance_mu_t (coupled theta + mu/muts/muave/mudf/ww) ---
    # WRF couples the work theta ``t_2`` ONCE per RK stage in small_step_prep
    # (module_small_step_em.F:263) and then advances that PERSISTENT coupled
    # array in place across every acoustic substep (advance_mu_t,
    # :1141-1172), decoupling ONLY once at the end (small_step_finish).  The
    # previous code re-coupled from the (nearly static) perturbation theta every
    # substep (``_mass_couple_theta_before_advance``) and decoupled every
    # substep, which RESET the work theta each substep and discarded the
    # accumulated large-step tendency + vertical/horizontal transport — the warm
    # bubble's theta then advanced only ~1 substep worth per full step (≈1/N_sound
    # too slow; F7K WRF-diff: integrated dtheta == 0.1× the correct rate, exactly
    # 1/acoustic_substeps).  Advance the carried ``theta_coupled_work`` instead so
    # the work theta accumulates across substeps exactly as WRF ``t_2``.
    coupled_state = uv_state.replace(theta=uv_state.theta_coupled_work)
    advanced = advance_mu_t_core(coupled_state, cfg)
    theta_coupled = advanced["theta"]
    ww_new = advanced["ww"]
    muave_new = advanced["muave"]
    muts_new = advanced["muts"]
    mu_new = advanced["mu"]
    mudf_new = advanced["mudf"]

    # Refresh advance_uv divergence damping bookkeeping field after advance_mu_t
    # (mudf was used by THIS substep's advance_uv from the previous mudf state).
    state_for_w = uv_state.replace(
        mu=mu_new, muts=muts_new, muave=muave_new, ww=ww_new, mudf=mudf_new, theta=theta_coupled
    )

    # --- 3. advance_w (implicit w + geopotential), real RHS ---
    nz = int(uv_state.theta.shape[0])
    ny = int(uv_state.theta.shape[1])
    nx = int(uv_state.theta.shape[2])
    cqw_field = cqw if cqw is not None else (uv_state.cqw if uv_state.cqw is not None else dry_cqw(nz, ny, nx, dtype=uv_state.theta.dtype))
    c2a_field = uv_state.c2a if uv_state.c2a is not None else jnp.ones_like(uv_state.theta)
    alt_field = uv_state.alt if uv_state.alt is not None else jnp.ones_like(uv_state.theta)
    phb_field = uv_state.phb if uv_state.phb is not None else jnp.zeros_like(uv_state.ph)
    ph_1_field = uv_state.ph_1 if uv_state.ph_1 is not None else jnp.zeros_like(uv_state.ph)
    ht_field = uv_state.ht if uv_state.ht is not None else jnp.zeros((ny, nx), dtype=uv_state.theta.dtype)
    c1f_field = uv_state.c1f if uv_state.c1f is not None else jnp.zeros((nz + 1,), dtype=uv_state.theta.dtype)
    c2f_field = uv_state.c2f if uv_state.c2f is not None else jnp.zeros((nz + 1,), dtype=uv_state.theta.dtype)
    rdn_field = uv_state.rdn if uv_state.rdn is not None else uv_state.rdnw
    cf1 = _optional_or(uv_state.cf1, jnp.asarray(0.0, dtype=uv_state.theta.dtype))
    cf2 = _optional_or(uv_state.cf2, jnp.asarray(0.0, dtype=uv_state.theta.dtype))
    cf3 = _optional_or(uv_state.cf3, jnp.asarray(0.0, dtype=uv_state.theta.dtype))
    msfux = _optional_or(uv_state.msfux, jnp.ones_like(uv_state.msfuy))
    msfvx = _optional_or(uv_state.msfvx, 1.0 / uv_state.msfvx_inv)

    mu_work = muts_new - uv_state.mut  # WRF perturbation dry-mass work array
    # F7G: WRF builds the large-step vertical PGF/buoyancy ``rw_tend`` via
    # pg_buoy_w ONCE per RK stage from the stage ``grid%p``/``mu`` in rk_tendency
    # (module_em.F:1361-1368) and carries it UNCHANGED through all acoustic
    # substeps.  When the caller supplies that stage array (``rw_tend_pg_buoy``),
    # use it verbatim -- do NOT recompute from the live small-step ``calc_p_rho``
    # work pressure each substep (that was the refuted F7F workaround;
    # gpt-council-findings.md §2/§3.3).  The legacy per-substep recompute is kept
    # only for bare-core/oracle callers that do not stage rw_tend.
    if uv_state.rw_tend_pg_buoy is not None:
        rw_tend = uv_state.rw_tend_pg_buoy
    else:
        p_for_buoy = uv_state.p_buoy if uv_state.p_buoy is not None else uv_state.p
        rw_tend = pg_buoy_w_dry(
            p_for_buoy,
            mu_work,
            c1f=c1f_field,
            rdnw=uv_state.rdnw,
            rdn=rdn_field,
            msfty=uv_state.msfty,
            gravity=GRAVITY_M_S2,
        )

    w_solved, ph_next, t_2ave_next = advance_w_wrf(
        w=uv_state.w,
        rw_tend=rw_tend,
        ww=ww_new,
        u=uv_state.u,
        v=uv_state.v,
        mu_work=mu_work,
        mut=uv_state.mut,
        muave=muave_new,
        muts=muts_new,
        t_2ave=uv_state.t_2ave,
        t_2=theta_coupled,
        t_1=uv_state.theta_1,
        ph=uv_state.ph,
        ph_1=ph_1_field,
        phb=phb_field,
        ph_tend=uv_state.ph_tend,
        ht=ht_field,
        c2a=c2a_field,
        cqw=cqw_field,
        alt=alt_field,
        a=a,
        alpha=alpha,
        gamma=gamma,
        c1h=uv_state.c1h,
        c2h=uv_state.c2h,
        c1f=c1f_field,
        c2f=c2f_field,
        rdnw=uv_state.rdnw,
        rdn=rdn_field,
        fnm=uv_state.fnm,
        fnp=uv_state.fnp,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        msftx=uv_state.msftx,
        msfty=uv_state.msfty,
        rdx=1.0 / float(cfg.dx),
        rdy=1.0 / float(cfg.dy),
        dts=float(cfg.dt),
        epssm=float(cfg.epssm),
        top_lid=bool(cfg.top_lid),
        gravity=GRAVITY_M_S2,
        w_save=uv_state.w_save,
        damp_opt=int(cfg.damp_opt),
        dampcoef=float(cfg.dampcoef),
        zdamp=float(cfg.zdamp),
        w_damping=int(cfg.w_damping),
        w_alpha=float(cfg.w_alpha),
        w_crit_cfl=float(cfg.w_crit_cfl),
    )

    # --- 4. sumflux accumulators (Sprint B consumer); WRF solve_em.F:4048-4093 ---
    ru_m = uv_state.ru_m if uv_state.ru_m is not None else jnp.zeros_like(uv_state.u)
    rv_m = uv_state.rv_m if uv_state.rv_m is not None else jnp.zeros_like(uv_state.v)
    ww_m = uv_state.ww_m if uv_state.ww_m is not None else jnp.zeros_like(uv_state.ww)
    ru_m = ru_m + uv_state.u
    rv_m = rv_m + uv_state.v
    ww_m = ww_m + ww_new

    # --- 5. calc_p_rho(step=iteration): smdiv pressure memory ---
    # WRF solve_em.F:4164-4171 passes the *live* ``grid%muts`` (refreshed by
    # advance_mu_t this substep) as the ``Mut`` denominator -- NOT the
    # stage-entry ``grid%mut`` (=uv_state.mut).  Feeding the base/stage mass here
    # was the broken-restoring-loop bug (gpt-findings.md §3.2).
    pm1 = uv_state.pm1 if uv_state.pm1 is not None else uv_state.p
    p_rho = calc_p_rho_step(
        mu_work=mu_work,
        muts_total=muts_new,
        ph_work=ph_next,
        theta_work=theta_coupled,
        theta_1=uv_state.theta_1,
        c2a=c2a_field,
        alt=alt_field,
        c1h=uv_state.c1h,
        c2h=uv_state.c2h,
        rdnw=uv_state.rdnw,
        pm1=pm1,
        smdiv=float(smdiv),
        t0=300.0,
    )

    # Physical-theta diagnostic view for the operational carry / audit budget.
    theta_phys = _decouple_theta_for_finish(uv_state, theta_coupled, muts_new)

    return uv_state.replace(
        mu=mu_new,
        mudf=mudf_new,
        muts=muts_new,
        muave=muave_new,
        ww=ww_new,
        theta=theta_phys,
        theta_coupled_work=theta_coupled,
        theta_ave=theta_phys,
        w=w_solved,
        ph=ph_next,
        p=p_rho.p,
        al=p_rho.al,
        pm1=p_rho.pm1,
        t_2ave=t_2ave_next,
        ru_m=ru_m,
        rv_m=rv_m,
        ww_m=ww_m,
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

    # WRF calc_coef_w uses the FULL dry mass ``mut`` (solve_em.F:2676-2681),
    # not the small-step work array ``muts``; real ``c2a``/``cqw`` are required.
    nz = int(state.theta.shape[0])
    ny = int(state.theta.shape[1])
    nx = int(state.theta.shape[2])
    cqw_field = state.cqw if state.cqw is not None else dry_cqw(nz, ny, nx, dtype=state.theta.dtype)
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        state.mut,
        metrics,
        dt=float(cfg.dt),
        epssm=float(cfg.epssm),
        top_lid=bool(cfg.top_lid),
        cqw=cqw_field,
        c2a=state.c2a,
    )
    current = state
    snapshots: list[dict[str, jax.Array]] = []
    for _ in range(int(substeps)):
        current = acoustic_substep_core(current, a=a, alpha=alpha, gamma=gamma, cfg=cfg, cqw=cqw_field)
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
