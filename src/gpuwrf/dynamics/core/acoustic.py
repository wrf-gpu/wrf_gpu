"""Pure shared WRF-shaped acoustic recurrence core.

This module owns the shared numerical acoustic recurrence used by validation
and operational wrappers. It performs no savepoint or HDF5 emission.

WRF ordering anchors:
- ``solve_em.F:2409-2738`` builds ``calc_coef_w`` coefficients once per RK stage.
- ``solve_em.F:3065`` starts ``small_steps : DO iteration = 1, number_of_small_timesteps``.
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

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "AcousticCoreState":
        payload = {}
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            if field_name == "coef_mut" and field_name not in values:
                payload[field_name] = None
            else:
                payload[field_name] = jnp.asarray(values[field_name])
        return cls(**payload)

    def to_dict(self) -> dict[str, jax.Array]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "coef_mut"}  # type: ignore[attr-defined]

    def replace(self, **updates: jax.Array) -> "AcousticCoreState":
        values = self.to_dict()
        values["coef_mut"] = self.coef_mut
        values.update(updates)
        return AcousticCoreState(**values)


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


def _ph_tend_increment(theta_old: jax.Array, theta_new: jax.Array, ph_tend: jax.Array) -> jax.Array:
    """Build the M6B3-bound geopotential tendency increment."""

    theta_delta = jnp.asarray(theta_new) - jnp.asarray(theta_old)
    increment = jnp.zeros_like(ph_tend)
    return increment.at[: theta_delta.shape[0], :, :].set(0.01 * theta_delta)


def acoustic_substep_core(
    state: AcousticCoreState,
    *,
    a: jax.Array,
    alpha: jax.Array,
    gamma: jax.Array,
    cfg: AcousticCoreConfig,
) -> AcousticCoreState:
    """Compose one WRF-shaped acoustic substep."""

    theta_old = state.theta
    mu_old = state.mu
    advanced = advance_mu_t_core(state, cfg)
    w_solved = w_solve_core(state, a=a, alpha=alpha, gamma=gamma)

    ph_increment = _ph_tend_increment(theta_old, advanced["theta"], state.ph_tend)
    scratch = build_scratch_state(
        ScratchInputs(
            theta_old=theta_old,
            theta_new=advanced["theta"],
            t_2ave_prev=state.t_2ave,
            ww_old=state.ww,
            ww_new=advanced["ww"],
            mu_old=mu_old,
            mu_new=advanced["mu"],
            mut=state.mut,
            muave_prev=state.muave,
            muts_prev=state.muts,
            ph_tend_old=state.ph_tend,
            ph_tend_increment=ph_increment,
            u_current=state.u,
            v_current=state.v,
            w_current=w_solved,
            ph_current=state.ph,
            epssm=float(cfg.epssm),
        )
    )
    return state.replace(
        mu=advanced["mu"],
        mudf=advanced["mudf"],
        muts=advanced["muts"],
        muave=advanced["muave"],
        ww=scratch["ww"],
        theta=advanced["theta"],
        theta_ave=advanced["theta"],
        ph_tend=scratch["ph_tend"],
        w=w_solved,
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
    "advance_mu_t_core",
    "w_solve_core",
    "acoustic_substep_core",
    "acoustic_scan_core",
    "acoustic_substep_wrf",
    "acoustic_loop_wrf",
    "snapshot_full_state",
]
