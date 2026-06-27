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

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass
from typing import Any

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.contracts.precision import force_fp64_island
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.advance_w import (
    GRAVITY_M_S2,
    W_ALPHA,
    W_BETA,
    advance_w_wrf,
    dry_cqw,
    pg_buoy_w_dry,
)
from gpuwrf.coupling.boundary_apply import (
    DEFAULT_BOUNDARY_CONFIG,
    apply_normal_bdy_work,
    spec_bdyupdate_ph_inloop,
)
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_step
from gpuwrf.dynamics.mu_t_advance import AdvanceMuTInputs, advance_mu_t_wrf
from gpuwrf.dynamics.tridiag_solve import thomas_solve_scan


configure_jax_x64()

_SHARDED_HALO_CONTEXT: tuple[Any, int] | None = None


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
    # WIND-FIX: full model timestep ``grid%dt`` (the acoustic ``dt`` field is the
    # substep ``dts``).  Used only to scale the in-loop NORMAL-momentum relaxation
    # weight to a per-substep increment (boundary_apply.apply_normal_bdy_work).
    # Defaults to ``dt`` so bare-core/oracle callers are unaffected.
    dt_full: float | None = None
    # WRF lateral BC flags for ``advance_mu_t`` active loop bounds
    # (module_small_step_em.F:1048-1063). Defaults preserve the existing periodic
    # idealized/oracle path exactly.
    periodic_x: bool = True
    specified: bool = False
    nested: bool = False
    normal_bdy_relax_strength: float | None = None
    # v0.14 SPECIFIED WRF cadence: apply WRF ``zero_grad_bdy`` to the w work
    # array's spec zone after advance_w every substep (solve_em.F:1601-1607,
    # specified domains copy the nearest interior w into the ring; nested uses
    # spec_bdyupdate instead).  Default OFF -> existing paths unchanged.
    spec_w_zero_grad: bool = False
    spec_zone: int = 1


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
    # WIND-FIX (2026-05-30): coupled small-step WORK-array boundary targets for the
    # NORMAL momentum (u at W/E, v at S/N), built ONCE per RK stage from the
    # interpolated decoupled wrfbdy leaf so small_step_finish reconstructs the
    # boundary velocity.  When present, ``advance_uv_wrf`` reproduces WRF's
    # spec_zone freeze + relax_bdy_dry nudge on the normal momentum INSIDE the
    # acoustic loop (the end-of-step coupling-layer nudge cannot remove the in-loop
    # spike -- proofs/wind/WIND_SKILL_ROOT_CAUSE.md §4b).  None for idealized /
    # oracle / bare-core callers (no lateral boundary), leaving the bare PGF
    # advance untouched so the idealized dycore gates are unaffected.
    u_work_bdy: jax.Array | None = None
    v_work_bdy: jax.Array | None = None
    # P0-6 (2026-06-01): NESTED-child ph' boundary forcing for the d03 T2 Exner
    # bias.  The relaxation-zone half is folded into ``ph_tend``/``rw_tend_pg_buoy``
    # at staging time (so it flows through advance_w coupled with w, WRF-faithful);
    # these two fields stage the in-loop SPEC-ZONE (outermost row) ph update applied
    # AFTER advance_w_wrf every substep (WRF spec_bdyupdate_ph, solve_em.F:1587).
    # ``ph_bdy_target`` is the time-interpolated PARENT perturbation-geopotential
    # leaf (frozen per stage); ``ph_save_for_spec`` is the stage-entry uncoupled ph'
    # (prep.ph_save).  Both None for idealized / d02 self-replay / bare-core callers
    # -> the spec update is skipped, so those paths are byte-for-byte unchanged.
    ph_bdy_target: jax.Array | None = None
    ph_save_for_spec: jax.Array | None = None
    # SPLIT-EXPLICIT FIX (v0.4.0 r5, agy cadence audit
    # .agent/reviews/2026-06-03-agy-v040-acoustic-cadence.md): WRF builds the
    # mass-point geopotential ``php`` ONCE per RK stage in ``rk_step_prep``
    # (calc_php; module_em.F:181) from the STAGE-ENTRY ``grid%ph_2`` and holds it
    # STAGE-CONSTANT, passing it INTENT(IN) to ``advance_uv`` every acoustic
    # substep (solve_em.F:1282; advance_uv 4th PGF term :861/:935).  When this
    # stage-constant array is supplied (operational prep path), advance_uv_wrf uses
    # it for the 4th-term geopotential gradient INSTEAD of re-diagnosing ``php``
    # from the live, substep-updated ``state.ph`` (the split-explicit violation
    # that rectified into a slow column-wide westerly force).  None for
    # bare-core/oracle callers that never stage it; those keep the legacy
    # per-substep recompute (single-substep / analytic-oracle usage only, where
    # there is no multi-substep rectification to expose).
    php_stage: jax.Array | None = None
    # v0.14 SPECIFIED-domain WRF cadence (stage3/wrapper sprint): in-loop
    # spec-zone (ring-0) pins applied AFTER advance_mu_t every acoustic substep
    # (WRF spec_bdyupdate for t_2/mu_2/muts, solve_em.F:1462-1490; WRF's
    # advance_mu_t excludes the ring, then the spec update walks it along the
    # wrfbdy trajectory).  Targets are the STAGE-END interpolated leaf values in
    # each array's own work convention:
    #   mu_spec_target     physical mu' carry pin (= leaf mu')
    #   muts_spec_target   mub + leaf mu'
    #   muave_spec_target  pinned mu work delta (muts_pin - mut)
    #   theta_spec_target  coupled work theta pin (mass_pin*theta'_t - mass_cur*t_save)
    # All None unless the specified WRF-cadence flag is on -> every other path
    # is byte-for-byte unchanged.
    mu_spec_target: jax.Array | None = None
    muts_spec_target: jax.Array | None = None
    muave_spec_target: jax.Array | None = None
    theta_spec_target: jax.Array | None = None
    # Ring-0 TANGENTIAL momentum work pins (WRF spec_bdyupdate covers the u S/N
    # rows and v W/E columns too; the WIND-FIX normal targets only own the
    # normal faces).  Applied in advance_uv_wrf after apply_normal_bdy_work so
    # the y-side rows own the corners (WRF b_limit convention).
    u_spec_tan_target: jax.Array | None = None
    v_spec_tan_target: jax.Array | None = None

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


def _maybe_exchange_sharded_acoustic_halos(state: AcousticCoreState) -> AcousticCoreState:
    """Refresh x halos for acoustic scratch fields when an opt-in pmap context is active."""

    context = _SHARDED_HALO_CONTEXT
    if context is None:
        return state
    sharding, width = context
    if not bool(getattr(sharding, "enabled", False)):
        return state
    if getattr(sharding, "axis", "x") != "x":
        raise NotImplementedError("acoustic sharded halo exchange currently supports x-axis decomposition only")

    from gpuwrf.runtime.sharding import exchange_periodic_halo_x, exchange_periodic_halo_x_face

    local_nx = int(state.theta.shape[-1])
    updates: dict[str, jax.Array] = {}
    for name in state.__dataclass_fields__:  # type: ignore[attr-defined]
        value = getattr(state, name)
        if value is None or not hasattr(value, "shape") or getattr(value, "ndim", 0) == 0:
            continue
        last_dim = int(value.shape[-1])
        if last_dim == local_nx + 1:
            updates[name] = exchange_periodic_halo_x_face(
                value,
                width=int(width),
                num_partitions=int(sharding.resolved_partitions()),
                axis_name=str(sharding.axis_name),
            )
        elif last_dim == local_nx:
            updates[name] = exchange_periodic_halo_x(
                value,
                width=int(width),
                num_partitions=int(sharding.resolved_partitions()),
                axis_name=str(sharding.axis_name),
            )
    return state.replace(**updates) if updates else state


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
        periodic_x=bool(cfg.periodic_x),
        specified=bool(cfg.specified),
        nested=bool(cfg.nested),
    )


def advance_mu_t_core(state: AcousticCoreState, cfg: AcousticCoreConfig) -> dict[str, jax.Array]:
    """Run the shared WRF ``advance_mu_t`` numerical core."""

    return advance_mu_t_wrf(_advance_inputs(state, cfg))


def _optional_or(value: jax.Array | None, default: jax.Array) -> jax.Array:
    return default if value is None else jnp.asarray(value, dtype=default.dtype)


def _specified_w_zero_grad_work(
    w_work: jax.Array,
    *,
    w_save: jax.Array | None,
    mut: jax.Array,
    muts: jax.Array,
    c1f: jax.Array,
    c2f: jax.Array,
    msfty: jax.Array,
    spec_zone: int,
) -> jax.Array:
    """Apply specified-domain zero-gradient to finished physical W via work space.

    The acoustic carry stores the WRF small-step work variable.  The physical
    state written by ``small_step_finish_wrf`` is reconstructed as

      W = (msfty * w_work + w_save * (c1f*mut+c2f)) / (c1f*muts+c2f)

    so copying only the work row does not copy the finished physical W when
    ``w_save`` or dry mass differs between the outer row and the nearest
    interior row.  WRF's specified-domain ``zero_grad_bdy`` is a boundary copy,
    not a limiter; this helper computes the work value whose finish
    reconstruction equals the nearest interior physical W.
    """

    def _source_index_array(indices, low: int, high: int):
        return jnp.asarray(
            [min(max(int(idx), low), high) for idx in indices],
            dtype=jnp.int32,
        )

    if w_save is None:
        out = w_work
        y_len = int(w_work.shape[1])
        x_len = int(w_work.shape[2])
        for b in range(int(spec_zone)):
            x_inner = _source_index_array(
                range(x_len),
                int(spec_zone),
                x_len - 1 - int(spec_zone),
            )
            y_inner = _source_index_array(
                range(b + 1, y_len - 1 - b),
                int(spec_zone),
                y_len - 1 - int(spec_zone),
            )
            out = out.at[:, b, :].set(out[:, int(spec_zone), x_inner])
            out = out.at[:, y_len - 1 - b, :].set(
                out[:, y_len - 1 - int(spec_zone), x_inner]
            )
            out = out.at[:, b + 1 : y_len - 1 - b, b].set(out[:, y_inner, int(spec_zone)])
            out = out.at[:, b + 1 : y_len - 1 - b, x_len - 1 - b].set(
                out[:, y_inner, x_len - 1 - int(spec_zone)]
            )
        return out

    mass_cur = c1f[:, None, None] * mut[None, :, :] + c2f[:, None, None]
    mass_stage = c1f[:, None, None] * muts[None, :, :] + c2f[:, None, None]
    physical = (msfty[None, :, :] * w_work + w_save * mass_cur) / mass_stage

    def _work_for_physical(
        target_physical,
        row_or_col_mass_stage,
        row_or_col_w_save,
        row_or_col_mass_cur,
        map_factor,
    ):
        return (
            target_physical * row_or_col_mass_stage
            - row_or_col_w_save * row_or_col_mass_cur
        ) / map_factor

    out = w_work
    y_len = int(w_work.shape[1])
    x_len = int(w_work.shape[2])
    for b in range(int(spec_zone)):
        south = b
        north = y_len - 1 - b
        south_src = int(spec_zone)
        north_src = y_len - 1 - int(spec_zone)
        west = b
        east = x_len - 1 - b
        west_src = int(spec_zone)
        east_src = x_len - 1 - int(spec_zone)
        x_inner = _source_index_array(
            range(x_len),
            int(spec_zone),
            x_len - 1 - int(spec_zone),
        )

        # Y sides own corners in WRF zero_grad_bdy and use the nearest interior
        # source column at corners (module_bc.F zero_grad_bdy i_inner).
        out = out.at[:, south, :].set(
            _work_for_physical(
                physical[:, south_src, x_inner],
                mass_stage[:, south, :],
                w_save[:, south, :],
                mass_cur[:, south, :],
                msfty[south, :][None, :],
            )
        )
        out = out.at[:, north, :].set(
            _work_for_physical(
                physical[:, north_src, x_inner],
                mass_stage[:, north, :],
                w_save[:, north, :],
                mass_cur[:, north, :],
                msfty[north, :][None, :],
            )
        )

        rows = slice(b + 1, y_len - 1 - b)
        y_inner = _source_index_array(
            range(b + 1, y_len - 1 - b),
            int(spec_zone),
            y_len - 1 - int(spec_zone),
        )
        out = out.at[:, rows, west].set(
            _work_for_physical(
                physical[:, y_inner, west_src],
                mass_stage[:, rows, west],
                w_save[:, rows, west],
                mass_cur[:, rows, west],
                msfty[rows, west][None, :],
            )
        )
        out = out.at[:, rows, east].set(
            _work_for_physical(
                physical[:, y_inner, east_src],
                mass_stage[:, rows, east],
                w_save[:, rows, east],
                mass_cur[:, rows, east],
                msfty[rows, east][None, :],
            )
        )
    return out


def _maybe_sharded_x_edge_pair(
    field: jax.Array,
    left: jax.Array,
    right: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Make x-face pairs match global-domain edge padding under opt-in x sharding."""

    context = _SHARDED_HALO_CONTEXT
    if context is None:
        return left, right
    sharding, width = context
    if not bool(getattr(sharding, "enabled", False)):
        return left, right
    if getattr(sharding, "axis", "x") != "x":
        raise NotImplementedError("acoustic sharded edge-pair correction supports x-axis decomposition only")
    h = int(width)
    owned = int(field.shape[-1]) - 2 * h
    if owned < 1:
        raise ValueError("haloed x field has no owned cells")
    rank = jax.lax.axis_index(str(sharding.axis_name))
    start = rank * owned
    global_nx = owned * int(sharding.resolved_partitions())
    west_face = h
    east_face = h + owned
    is_first = start == 0
    is_last = start + owned == global_nx
    left = left.at[..., west_face].set(jnp.where(is_first, right[..., west_face], left[..., west_face]))
    right = right.at[..., east_face].set(jnp.where(is_last, left[..., east_face], right[..., east_face]))
    return left, right


# v0.10.0 Wave-A (Opus#6): the edge-pad-then-slice face pairs materialised a full
# padded copy of every large 3D field every substep (a memory-op kernel + extra
# HBM each).  ``jnp.pad(field, ((0,0),(0,0),(1,1)), mode="edge")`` followed by
# ``padded[...,:-1] , padded[...,1:]`` is exactly:
#   left  = [f[0], f[0], f[1], ..., f[-1]]  = concat(f[...,:1], f)
#   right = [f[0], f[1], ..., f[-1], f[-1]] = concat(f, f[...,-1:])
# so the concatenate form is BIT-IDENTICAL (same values, no full padded array).
def _x_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    left = jnp.concatenate([field[:, :, :1], field], axis=2)
    right = jnp.concatenate([field, field[:, :, -1:]], axis=2)
    return _maybe_sharded_x_edge_pair(field, left, right)


def _y_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    south = jnp.concatenate([field[:, :1, :], field], axis=1)
    north = jnp.concatenate([field, field[:, -1:, :]], axis=1)
    return south, north


def _x_face_pair_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    left = jnp.concatenate([field[:, :1], field], axis=1)
    right = jnp.concatenate([field, field[:, -1:]], axis=1)
    return _maybe_sharded_x_edge_pair(field, left, right)


def _y_face_pair_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    south = jnp.concatenate([field[:1, :], field], axis=0)
    north = jnp.concatenate([field, field[-1:, :]], axis=0)
    return south, north


# v0.10.0 Wave-A (Opus#7): build dpn with a single concatenate of
# [bottom, interior, top] instead of allocating zeros + 2-3 dynamic-update-slice
# scatters per substep.  The interior block spans WRF k=2..kde-1 (Python 1..nz-1),
# the bottom face is k=1, the top face is k=kde (= zeros when not top_lid, exactly
# as the zeros-init left index nz untouched).  BIT-IDENTICAL (same values).
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
    if bool(top_lid):
        top = 0.5 * (cf1 * pair_sum[-1, :, :] + cf2 * pair_sum[-2, :, :] + cf3 * pair_sum[-3, :, :])
    else:
        top = jnp.zeros((ny, nx_face), dtype=state.p.dtype)
    return jnp.concatenate([bottom[None, :, :], interior, top[None, :, :]], axis=0)


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
    if bool(top_lid):
        top = 0.5 * (cf1 * pair_sum[-1, :, :] + cf2 * pair_sum[-2, :, :] + cf3 * pair_sum[-3, :, :])
    else:
        top = jnp.zeros((ny_face, nx), dtype=state.p.dtype)
    return jnp.concatenate([bottom[None, :, :], interior, top[None, :, :]], axis=0)


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
    dt_full: float | None = None,
    normal_bdy_relax_strength: float | None = None,
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

    # v0.20 S2 intrinsic fp64-island lock: the horizontal PGF brackets below take
    # differences of large nearly-equal pressure / geopotential columns (p, ph,
    # the base-pressure gradient pb, the inverse densities al/alt, the dpn vertical
    # average, the php 4th-term geopotential). Widen these cancellation-sensitive
    # inputs to fp64 IN-OPERATOR so a later fp32 storage downcast cannot
    # contaminate the gradient. No-op (bit-identical) on fp64_default: every input
    # is already fp64, so force_fp64_island returns each array unchanged (no
    # convert op) and this replace re-wraps identical leaves.
    state = state.replace(
        p=force_fp64_island(state.p),
        ph=force_fp64_island(state.ph),
        p_base=force_fp64_island(state.p_base),
        ph_base=force_fp64_island(state.ph_base),
        al=force_fp64_island(state.al),
        alt=force_fp64_island(state.alt),
    )

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

    # SPLIT-EXPLICIT FIX (v0.4.0 r5): the 4th PGF term's mass-point geopotential
    # ``php`` is STAGE-CONSTANT in WRF (calc_php in rk_step_prep, passed INTENT(IN)
    # to advance_uv every substep -- module_em.F:181, advance_uv :861/:935).  Use
    # the frozen stage ``php_stage`` from small_step_prep when supplied; otherwise
    # (bare-core / analytic-oracle single-substep callers) fall back to the legacy
    # diagnosis from the live geopotential, which is harmless there because no
    # multi-substep rectification exists.  The FIRST-3-terms gradient above still
    # uses the LIVE ``state.ph`` (WRF :828-831 uses live ``ph``), so only the
    # 4th-term geopotential gradient is frozen -- exactly matching WRF's split.
    if state.php_stage is not None:
        php = state.php_stage
    else:
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

    # WIND-FIX: reproduce WRF's spec_zone-restricted advance_uv + relax_bdy_dry on
    # the NORMAL momentum work array INSIDE the acoustic loop (boundary_apply
    # comment block; WRF module_small_step_em.F:734-942 loop bounds +
    # solve_em.F:1346-1364 spec_bdyupdate + module_bc_em.F:161-346 relax_bdy_dry).
    # The bare PGF above freely advanced the outer spec/relax normal faces (the
    # source of the +/-7..17 m/s blow-up); here we freeze the spec face to the
    # boundary work target and relax the relax-zone faces toward it.  Skipped when
    # no boundary target is staged (idealized/oracle/bare-core), so those paths and
    # the idealized dycore gates are byte-for-byte unaffected.
    if state.u_work_bdy is not None and state.v_work_bdy is not None:
        u, v = apply_normal_bdy_work(
            u, v, state.u_work_bdy, state.v_work_bdy,
            dts, float(dt_full) if dt_full is not None else dts,
            config=DEFAULT_BOUNDARY_CONFIG,
            relax_strength=normal_bdy_relax_strength,
        )
    # v0.14 SPECIFIED WRF cadence: ring-0 TANGENTIAL momentum pins (WRF
    # spec_bdyupdate(u,'u') y-side rows / spec_bdyupdate(v,'v') x-side columns,
    # solve_em.F:1346-1364 inside the small-step loop).  Applied AFTER the
    # normal-face treatment so the u S/N rows own the corners (WRF y-side
    # b_limit=0 full range; the v x-side columns trim the corner rows, WRF j in
    # [jds+1, jde-1]).  None unless the specified-cadence flag staged targets.
    if state.u_spec_tan_target is not None and state.v_spec_tan_target is not None:
        spec_zone = int(DEFAULT_BOUNDARY_CONFIG.spec_zone)
        ny_u = int(u.shape[1])
        ny_v = int(v.shape[1])
        for b in range(spec_zone):
            u = u.at[:, b, :].set(state.u_spec_tan_target[:, b, :])
            u = u.at[:, ny_u - 1 - b, :].set(state.u_spec_tan_target[:, ny_u - 1 - b, :])
            v = v.at[:, 1:-1, b].set(state.v_spec_tan_target[:, 1:-1, b])
            v = v.at[:, 1:-1, v.shape[2] - 1 - b].set(state.v_spec_tan_target[:, 1:-1, v.shape[2] - 1 - b])
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

    state = _maybe_exchange_sharded_acoustic_halos(state)

    # --- 1. advance_uv (with external-mode divergence damping) ---
    uv_state = advance_uv_wrf(
        state,
        dts_rk=float(cfg.dt),
        dx=float(cfg.dx),
        dy=float(cfg.dy),
        top_lid=bool(cfg.top_lid),
        emdiv=float(emdiv),
        dt_full=(float(cfg.dt_full) if cfg.dt_full is not None else float(cfg.dt)),
        normal_bdy_relax_strength=cfg.normal_bdy_relax_strength,
    )
    uv_state = _maybe_exchange_sharded_acoustic_halos(uv_state)

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

    # v0.14 SPECIFIED WRF cadence: spec-zone (ring-0) mass/theta pins (WRF
    # spec_bdyupdate for t_2/mu_2/muts after advance_mu_t, solve_em.F:1462-1490;
    # WRF's advance_mu_t excludes the ring -- _advance_mu_t_specified_or_nested
    # already mirrors that -- and the spec update then walks the ring along the
    # wrfbdy trajectory).  ``muave`` is pinned to the steady pinned work delta
    # (the epssm-weighted average of a constant).  None targets (every path
    # except the specified-cadence flag) leave all four arrays untouched.
    if state.mu_spec_target is not None:
        def _pin_ring(field, target):
            out = field
            for b in range(int(DEFAULT_BOUNDARY_CONFIG.spec_zone)):
                out = out.at[..., b, :].set(target[..., b, :])
                out = out.at[..., field.shape[-2] - 1 - b, :].set(target[..., field.shape[-2] - 1 - b, :])
                out = out.at[..., :, b].set(target[..., :, b])
                out = out.at[..., :, field.shape[-1] - 1 - b].set(target[..., :, field.shape[-1] - 1 - b])
            return out

        mu_new = _pin_ring(mu_new, state.mu_spec_target)
        muts_new = _pin_ring(muts_new, state.muts_spec_target)
        muave_new = _pin_ring(muave_new, state.muave_spec_target)
        theta_coupled = _pin_ring(theta_coupled, state.theta_spec_target)

    # Refresh advance_uv divergence damping bookkeeping field after advance_mu_t
    # (mudf was used by THIS substep's advance_uv from the previous mudf state).
    state_for_w = uv_state.replace(
        mu=mu_new, muts=muts_new, muave=muave_new, ww=ww_new, mudf=mudf_new, theta=theta_coupled
    )
    state_for_w = _maybe_exchange_sharded_acoustic_halos(state_for_w)
    theta_coupled = state_for_w.theta
    ww_new = state_for_w.ww
    muave_new = state_for_w.muave
    muts_new = state_for_w.muts
    mu_new = state_for_w.mu
    mudf_new = state_for_w.mudf

    # --- 3. advance_w (implicit w + geopotential), real RHS ---
    nz = int(state_for_w.theta.shape[0])
    ny = int(state_for_w.theta.shape[1])
    nx = int(state_for_w.theta.shape[2])
    cqw_field = cqw if cqw is not None else (state_for_w.cqw if state_for_w.cqw is not None else dry_cqw(nz, ny, nx, dtype=state_for_w.theta.dtype))
    c2a_field = state_for_w.c2a if state_for_w.c2a is not None else jnp.ones_like(state_for_w.theta)
    alt_field = state_for_w.alt if state_for_w.alt is not None else jnp.ones_like(state_for_w.theta)
    phb_field = state_for_w.phb if state_for_w.phb is not None else jnp.zeros_like(state_for_w.ph)
    ph_1_field = state_for_w.ph_1 if state_for_w.ph_1 is not None else jnp.zeros_like(state_for_w.ph)
    ht_field = state_for_w.ht if state_for_w.ht is not None else jnp.zeros((ny, nx), dtype=state_for_w.theta.dtype)
    c1f_field = state_for_w.c1f if state_for_w.c1f is not None else jnp.zeros((nz + 1,), dtype=state_for_w.theta.dtype)
    c2f_field = state_for_w.c2f if state_for_w.c2f is not None else jnp.zeros((nz + 1,), dtype=state_for_w.theta.dtype)
    rdn_field = state_for_w.rdn if state_for_w.rdn is not None else state_for_w.rdnw
    cf1 = _optional_or(state_for_w.cf1, jnp.asarray(0.0, dtype=state_for_w.theta.dtype))
    cf2 = _optional_or(state_for_w.cf2, jnp.asarray(0.0, dtype=state_for_w.theta.dtype))
    cf3 = _optional_or(state_for_w.cf3, jnp.asarray(0.0, dtype=state_for_w.theta.dtype))
    msfux = _optional_or(state_for_w.msfux, jnp.ones_like(state_for_w.msfuy))
    msfvx = _optional_or(state_for_w.msfvx, 1.0 / state_for_w.msfvx_inv)

    mu_work = muts_new - state_for_w.mut  # WRF perturbation dry-mass work array
    # F7G: WRF builds the large-step vertical PGF/buoyancy ``rw_tend`` via
    # pg_buoy_w ONCE per RK stage from the stage ``grid%p``/``mu`` in rk_tendency
    # (module_em.F:1361-1368) and carries it UNCHANGED through all acoustic
    # substeps.  When the caller supplies that stage array (``rw_tend_pg_buoy``),
    # use it verbatim -- do NOT recompute from the live small-step ``calc_p_rho``
    # work pressure each substep (that was the refuted F7F workaround;
    # gpt-council-findings.md §2/§3.3).  The legacy per-substep recompute is kept
    # only for bare-core/oracle callers that do not stage rw_tend.
    if state_for_w.rw_tend_pg_buoy is not None:
        rw_tend = state_for_w.rw_tend_pg_buoy
    else:
        p_for_buoy = state_for_w.p_buoy if state_for_w.p_buoy is not None else state_for_w.p
        rw_tend = pg_buoy_w_dry(
            p_for_buoy,
            mu_work,
            c1f=c1f_field,
            rdnw=state_for_w.rdnw,
            rdn=rdn_field,
            msfty=state_for_w.msfty,
            gravity=GRAVITY_M_S2,
        )

    w_solved, ph_next, t_2ave_next = advance_w_wrf(
        w=state_for_w.w,
        rw_tend=rw_tend,
        ww=ww_new,
        # advance_w uses u/v ONLY for the kinematic terrain-following surface w BC
        # (advance_w.py:274-303; WRF module_small_step_em.F:1384, "w=mx*u*dz/dx+my*v*dz/dy").
        #
        # HONEST DEVIATION FROM WRF (do NOT "fix" by reverting to coupled u/v):
        # WRF feeds the COUPLED prognostic winds grid%u_2/grid%v_2 here. They are
        # NOT physical m/s at this point: inside the acoustic loop grid%u_2/v_2 are
        # the mass-coupled work arrays ((c1h*muu+c2h)*u/msf, ~1e4-1e5x the physical
        # wind) and stay coupled through the whole small-step loop (advance call at
        # solve_em.F:1500-1501; coupling held through module_small_step_em.F:805) --
        # WRF only decouples them back to physical m/s afterwards, in
        # small_step_finish_em.  So WRF's surface-w BC extrapolation operates on the
        # COUPLED winds, consistent with its later decoupling of w.
        #
        # We DELIBERATELY feed the DECOUPLED stage winds u_1/v_1 (physical m/s)
        # instead, as a STABILITY TRADE-OFF. Feeding the coupled winds through the
        # cf1/cf2/cf3 surface extrapolation (which has no mass-factor division here)
        # reintroduced a spurious ~73 m/s k0-only surface-w mode over the steepest
        # Canary volcanic cells (vs 1.4 m/s @ k1 -- a linearly-ramping k0 artifact
        # that detonated once MYNN sustained a near-surface wind there; global
        # blow-up; proofs/stability/ + proofs/m19 terrain-w localization,
        # 2026-05-30). The decoupled feed suppresses that mode and keeps the run
        # stable, at the cost of an under-energetic terrain-w (mountain-wave W ~2-3x
        # weak vs CPU-WRF; see proofs/m19 terrain-w resolution). This is a KNOWN
        # MINOR ACCURACY ITEM to tighten in M20 (e.g. correct mass-factor handling
        # of the coupled surface BC so the coupled feed is stable), NOT a WRF match.
        u=state_for_w.u_1,
        v=state_for_w.v_1,
        mu_work=mu_work,
        mut=state_for_w.mut,
        muave=muave_new,
        muts=muts_new,
        t_2ave=state_for_w.t_2ave,
        t_2=theta_coupled,
        t_1=state_for_w.theta_1,
        ph=state_for_w.ph,
        ph_1=ph_1_field,
        phb=phb_field,
        ph_tend=state_for_w.ph_tend,
        ht=ht_field,
        c2a=c2a_field,
        cqw=cqw_field,
        alt=alt_field,
        a=a,
        alpha=alpha,
        gamma=gamma,
        c1h=state_for_w.c1h,
        c2h=state_for_w.c2h,
        c1f=c1f_field,
        c2f=c2f_field,
        rdnw=state_for_w.rdnw,
        rdn=rdn_field,
        fnm=state_for_w.fnm,
        fnp=state_for_w.fnp,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        msftx=state_for_w.msftx,
        msfty=state_for_w.msfty,
        rdx=1.0 / float(cfg.dx),
        rdy=1.0 / float(cfg.dy),
        dts=float(cfg.dt),
        epssm=float(cfg.epssm),
        top_lid=bool(cfg.top_lid),
        gravity=GRAVITY_M_S2,
        w_save=state_for_w.w_save,
        damp_opt=int(cfg.damp_opt),
        dampcoef=float(cfg.dampcoef),
        zdamp=float(cfg.zdamp),
        w_damping=int(cfg.w_damping),
        w_alpha=float(cfg.w_alpha),
        w_crit_cfl=float(cfg.w_crit_cfl),
    )

    # --- 3b. NESTED ph' spec-zone boundary update (WRF spec_bdyupdate_ph) ---
    # WRF applies spec_bdyupdate_ph to the COUPLED ph_2 in the outermost spec_zone
    # row every acoustic substep AFTER advance_w and BEFORE calc_p_rho
    # (solve_em.F:1587-1597).  Our ph work array (``ph_next``) is the uncoupled
    # perturbation-delta ``ph'_ref - ph'``; the spec-zone delta is pinned so the
    # reconstructed spec-zone ph' equals the parent boundary leaf.  The relaxation
    # zone is owned by the ``ph_tend`` path inside advance_w above (it has already
    # been folded into the carried tendency once per RK stage).  Skipped (None
    # targets) for idealized / d02 self-replay / bare-core callers, leaving those
    # paths byte-for-byte unchanged.
    if state_for_w.ph_bdy_target is not None and state_for_w.ph_save_for_spec is not None:
        ph_next = spec_bdyupdate_ph_inloop(
            ph_next,
            state_for_w.ph_bdy_target,
            state_for_w.ph_save_for_spec,
            mu_tend=None,
            muts=muts_new,
            c1f=c1f_field,
            c2f=c2f_field,
            dts=float(cfg.dt),
            config=DEFAULT_BOUNDARY_CONFIG,
        )

    # --- 3c. SPECIFIED w spec-zone zero-gradient (WRF zero_grad_bdy) ---
    # WRF solve_em.F:1601-1607: for SPECIFIED domains the spec-zone w_2 copies
    # the nearest interior value every substep after advance_w (nested domains
    # use spec_bdyupdate instead).  The y-side rows own the corners (WRF y-side
    # b_limit=0 full range; x-side trims j to [jds+1, jde-2]).  ``w_solved`` is
    # the small-step work variable, so compute the work value whose
    # small_step_finish reconstruction has the WRF zero-gradient physical W.
    # Default OFF.
    if bool(cfg.spec_w_zero_grad):
        w_solved = _specified_w_zero_grad_work(
            w_solved,
            w_save=state_for_w.w_save,
            mut=state_for_w.mut,
            muts=muts_new,
            c1f=c1f_field,
            c2f=c2f_field,
            msfty=state_for_w.msfty,
            spec_zone=int(cfg.spec_zone),
        )

    # --- 4. sumflux accumulators (Sprint B consumer); WRF solve_em.F:4048-4093 ---
    state_for_pressure = state_for_w.replace(w=w_solved, ph=ph_next, t_2ave=t_2ave_next)
    state_for_pressure = _maybe_exchange_sharded_acoustic_halos(state_for_pressure)
    w_solved = state_for_pressure.w
    ph_next = state_for_pressure.ph
    t_2ave_next = state_for_pressure.t_2ave

    ru_m = state_for_pressure.ru_m if state_for_pressure.ru_m is not None else jnp.zeros_like(state_for_pressure.u)
    rv_m = state_for_pressure.rv_m if state_for_pressure.rv_m is not None else jnp.zeros_like(state_for_pressure.v)
    ww_m = state_for_pressure.ww_m if state_for_pressure.ww_m is not None else jnp.zeros_like(state_for_pressure.ww)
    ru_m = ru_m + state_for_pressure.u
    rv_m = rv_m + state_for_pressure.v
    ww_m = ww_m + ww_new

    # --- 5. calc_p_rho(step=iteration): smdiv pressure memory ---
    # WRF solve_em.F:4164-4171 passes the *live* ``grid%muts`` (refreshed by
    # advance_mu_t this substep) as the ``Mut`` denominator -- NOT the
    # stage-entry ``grid%mut`` (=uv_state.mut).  Feeding the base/stage mass here
    # was the broken-restoring-loop bug (gpt-findings.md §3.2).
    pm1 = state_for_pressure.pm1 if state_for_pressure.pm1 is not None else state_for_pressure.p
    p_rho = calc_p_rho_step(
        mu_work=mu_work,
        muts_total=muts_new,
        ph_work=ph_next,
        theta_work=theta_coupled,
        theta_1=state_for_pressure.theta_1,
        c2a=c2a_field,
        alt=alt_field,
        c1h=state_for_pressure.c1h,
        c2h=state_for_pressure.c2h,
        rdnw=state_for_pressure.rdnw,
        pm1=pm1,
        smdiv=float(smdiv),
        t0=300.0,
    )

    # Physical-theta diagnostic view for the operational carry / audit budget.
    theta_phys = _decouple_theta_for_finish(state_for_pressure, theta_coupled, muts_new)

    result = state_for_pressure.replace(
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
    return _maybe_exchange_sharded_acoustic_halos(result)


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
