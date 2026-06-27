"""Device-side lateral-boundary forcing for the M6/ADR-023 d02 forecast driver.

This module reproduces the WRF v4 specified + relaxation-zone lateral boundary
update (``spec_bdytend`` / ``relax_bdytend`` in ``share/module_bc.F`` and the
weight table ``lbc_fcx_gcx`` in ``dyn_em/module_bc_em.F``) as a pure
``State -> State`` adapter, applied once per operational/replay timestep after
the dycore + physics block.

WRF subtleties faithfully reproduced here
------------------------------------------
* **Specified (outer) zone** of width ``spec_zone`` (the outermost row/column,
  ``b_dist=0``) is overwritten with the time-interpolated boundary value
  (WRF ``spec_bdytend`` drives the field tendency toward the boundary value; in
  the side-history replay we set the value directly).
* **Relaxation zone** of width ``relax_zone`` (rows ``b_dist = spec_zone ..
  relax_zone-1``) is nudged by
  ``field += fcx(b_dist)*fls0 - gcx(b_dist)*(fls1+fls2+fls3+fls4-4*fls0)``
  where ``fls0`` is the residual ``bdy - field`` at the row, ``fls1/fls2`` are
  the *tangential* residual neighbours (clamped at the side ends like WRF's
  ``max(i-1,ibs)``/``min(i+1,ibe)``), and ``fls3/fls4`` are the *normal*
  residual neighbours that use the boundary strips at width ``b_dist-1`` and
  ``b_dist+1`` against the adjacent interior rows -- the discrete Laplacian
  smoothing of the boundary residual, exactly as WRF computes it.
* **Corner trimming**: WRF's Y-boundary relaxation runs the tangential index
  ``i`` only over ``[b_limit+ibs , ibe-b_limit]`` with ``b_limit=b_dist`` so the
  diagonal corners are owned by the X (W/E) boundaries; the X-boundary
  relaxation trims its tangential index ``j`` to ``[b_dist+jbs+1 ,
  jbe-b_dist-1]``. We reproduce both via per-row tangential slicing so a
  relaxation-zone cell is updated by exactly one side, matching WRF.
* **Weights** ``fcx``/``gcx`` follow ``lbc_fcx_gcx`` for the
  ``specified``/``nested`` branches (identical linear taper; the optional
  ``spec_exp`` sponge multiplies both). We fold WRF's per-RK ``dt`` factor into
  the weights so a single per-step value update equals one WRF tendency step.

Known, documented departure from bit-exact WRF
----------------------------------------------
WRF relaxes the *mass-coupled* variables (``ru = c1*mu*u``, ``t = mu*theta``,
mass-weighted ``ph``/``w``). The Gen2 replay forces with *decoupled* wrfout
side-history (raw ``U``/``V``/``T``/...), so we relax the decoupled fields
against the decoupled boundary leaves. For the slowly-varying outer strip this
is an O(mu') approximation, not bit-exact WRF. It is the honest choice for a
side-history replay whose boundary data are themselves decoupled wrfout fields;
the residual it introduces is bounded by the column-mass perturbation and is
quantified in ``proofs/b4/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.contracts.state import State


SIDES = ("W", "E", "S", "N")
SIDE_INDEX = {name: index for index, name in enumerate(SIDES)}

# WRF equation-of-state / hydrostatic-integration constants.  Mirror the dycore
# (acoustic_wrf.R_D/CP_D/P0_PA/CVPM) so the boundary-ring geopotential rebuild is
# the EXACT inverse of ``diagnose_pressure_al_alt`` / WRF ``calc_p_rho_phi``.
_R_D = 287.0
_CP_D = 7.0 * _R_D / 2.0  # WRF cp = 1004.5 exactly (mirrors acoustic_wrf.CP_D)
_P0_PA = 100000.0
_CVPM = -(_CP_D - _R_D) / _CP_D
_THETA_BASE_OFFSET = 300.0  # WRF perturbation-theta reference t0 (operational_mode._theta_base_offset)


@dataclass(frozen=True)
class BoundaryConfig:
    """WRF lateral-boundary control values for the pinned Gen2 d02 run.

    Mirrors the d01 ``namelist.input`` (``spec_bdy_width=5``, ``spec_zone=1``,
    ``relax_zone=4``, ``spec_exp=0``). ``update_cadence_s`` is the wrfbdy /
    side-history interval (hourly for the Gen2 corpus).
    """

    spec_bdy_width: int = 5
    spec_zone: int = 1
    relax_zone: int = 4
    update_cadence_s: float = 3600.0
    spec_exp: float = 0.0
    # Whether to overwrite the prognostic geopotential ``ph`` in the boundary
    # strip from the ``ph_bdy`` parent leaf.  This is correct ONLY when the
    # boundary leaves are SELF-CONSISTENT with the child column (the d02 self-
    # replay path, where the strips are the child's own hourly wrfout history).
    # For a NESTED parent->child boundary the ``ph_bdy`` strip is the PARENT's
    # perturbation geopotential bilinearly interpolated onto the finer child
    # grid; that interpolated value is NOT hydrostatically consistent with the
    # child's own mu/theta column, so overwriting ``ph`` in the ring feeds an
    # inconsistent geopotential directly into the acoustic w-ph solve
    # (small_step_prep ph_1/ph_save/ph_work), pumping spurious vertical motion
    # that warms the interior (root cause of the d03 1km +6.8 K T2 bias --
    # the ph forcing alone reproduces +2.84 K/10 min while every other forced
    # field stays within +/-0.13 K).  WRF does not independently overwrite the
    # nest geopotential from the interpolated parent field.  When
    # ``force_geopotential=False`` AND a ``metrics`` object is supplied,
    # :func:`apply_lateral_boundaries` does NOT leave ``ph'`` free either (which
    # let the forced-mu'/theta' boundary ring drift out of hydrostatic balance
    # and inflated the diagnosed perturbation pressure ~+2.6 kPa, warming d03 T2
    # ~+2 K via the Exner conversion -- see the 2026-06-01 t2bias reviews).
    # Instead it RE-DERIVES a hydrostatically self-consistent ``ph'`` in the ring
    # from the forced mu'/theta'/qv' (the exact inverse of the dycore al
    # diagnostic).  ``True`` preserves the validated d02 self-replay path
    # byte-for-byte.
    force_geopotential: bool = True
    # P0-6 (2026-06-01) in-acoustic-loop NESTED ph'/w boundary forcing toggles
    # (active only when ``force_geopotential`` is False AND a real lateral boundary
    # is running).  Each maps to one WRF mechanism so the d03 short-run sweep can
    # isolate which is WRF-faithful AND stable for this decoupled side-history
    # replay:
    #   * ``nested_ph_relax``: relax-zone ph' tendency folded into ph_tend ->
    #     advance_w (WRF relax_bdy_dry ph + rk_addtend_dry).  Primary fix for the
    #     +2.6 kPa diagnostic-pressure / Exner T2 bias.
    #   * ``nested_w_relax``: relax-zone w tendency folded into rw_tend (WRF nests
    #     only).  Risky here: the parent 3km w leaf interpolated to the 1km child is
    #     a poor child-scale target and can pump interior vertical motion.
    #   * ``nested_ph_spec``: in-loop spec-zone (outer row) ph' pin after advance_w
    #     (WRF spec_bdyupdate_ph).
    #
    # DEFAULT OFF (2026-06-01 short-d03 sweep, STOP+report): the in-loop forcing is
    # dynamically STABLE (finite, no blow-up -- a real advance over the prior
    # end-of-step attempt) and WRF-faithful in mechanism, but toward the DECOUPLED
    # hourly parent leaf it injects spurious vertical motion -- interior max|W|
    # ~13 m/s vs CPU-WRF corpus 7.2 m/s and the free-drift GPU baseline 6.2 m/s --
    # which adiabatically warms the interior theta (+5..+9 K) and makes d03 T2 WORSE
    # (RMSE 1.94 -> 5..8 K at hour 1) even though it collapses the +2.7 kPa psfc
    # error ~50%.  ANY ph' forcing variant (relax / spec / both / +w) shows this
    # w-pump.  Root cause is architectural: WRF's child ph is re-synced to the
    # parent every parent step (med_nest_force), so the child interior never drifts
    # 2.6 kPa low and the ring constraint stays small; our decoupled hourly
    # side-history replay has no such re-sync, so the child ph free-drifts and the
    # ring relax becomes a large sustained forcing that the implicit (w,ph) solve
    # balances with spurious w.  See .agent/reviews/2026-06-01-opus-d03-phfix-
    # INLOOP-findings.md.  Left ON-able for the follow-up (true nested-init re-sync
    # or a mass-coupled child-prognostic-ph path), default OFF so the d03 replay
    # stays on the validated-stable free-drift baseline.
    nested_ph_relax: bool = False
    nested_w_relax: bool = False
    nested_ph_spec: bool = False
    # In-acoustic normal-momentum relaxation strength. ``None`` preserves the
    # legacy calibrated replay default (``NORMAL_BDY_RELAX_STRENGTH``). Native
    # standalone wrfbdy roots set this to 1.0, WRF's own relax_bdy_dry strength.
    normal_bdy_relax_strength: float | None = None


DEFAULT_BOUNDARY_CONFIG = BoundaryConfig()


def _apply_3d_spec_only(field, boundary, lead_seconds, config: BoundaryConfig):
    """Spec-zone-only (ring-0) hard set from the time-interpolated leaf.

    The relax-zone half of :func:`_apply_3d` is intentionally skipped -- on the
    WRF specified cadence the relax zone is owned by the per-stage
    ``relax_bdy_dry`` tendencies inside the RK/acoustic loop, and WRF has no
    end-of-step relax-zone value write.
    """

    forcing = interpolate_boundary_leaf(boundary, lead_seconds, config.update_cadence_s)
    out = field
    for side in SIDES:
        out = _apply_side_spec(out, forcing, side, config)
    return out


def apply_lateral_boundaries(
    state: State,
    lead_seconds,
    dt_s: float,
    config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
    metrics: DycoreMetrics | None = None,
    *,
    dry_spec_only: bool = False,
) -> State:
    """Apply the WRF specified outer zone + relaxation-zone nudging in-place.

    Reads the ``*_bdy`` leaves (shape ``(time, side, bdy_width, z, side_len)``)
    and the corresponding interior fields; writes ``u,v,w,theta,qv`` plus any
    optional hydrometeor/number scalar boundary leaves and the ``p/ph/mu``
    total+perturbation triples in the boundary strip only. The interior beyond
    ``relax_zone`` is untouched. Field dtypes are preserved, so an fp64
    (``force_fp64``) state stays fp64 and an operational fp32-gated state stays
    fp32-gated.

    ``metrics`` (the dycore :class:`DycoreMetrics`) is required ONLY for the
    nested ``force_geopotential=False`` branch, where the boundary-ring
    perturbation geopotential is RE-DERIVED hydrostatically from the forced
    mu'/theta'/qv' (see :func:`_hydrostatic_ph_perturbation`).  When it is
    ``None`` (idealized / d02 self-replay callers) the geopotential ring is left
    exactly as before, so those paths stay bit-for-bit unchanged.
    """

    def _apply_optional_scalar(field_name: str):
        leaf = getattr(state, f"{field_name}_bdy", None)
        field = getattr(state, field_name)
        if leaf is None:
            return field
        return jnp.maximum(_apply_3d(field, leaf, lead_seconds, dt_s, config), 0.0)

    if dry_spec_only:
        # WRF SPECIFIED cadence (v0.14 stage3/wrapper sprint): the in-loop spec
        # pins + per-stage relax tendencies own the dry boundary band; this
        # end-of-step pass only re-syncs the ring-0 spec values to the
        # lead-time leaf (idempotent against the in-loop pins; also covers
        # fields the loop does not pin, e.g. the tangential winds' carry) and
        # keeps the FULL moisture handling (WRF moist spec+relax lives in
        # rk_update_scalar, which this codebase still applies end-of-step).
        # The diagnostic p'/pb are NEVER forced (WRF does not force p; ring
        # values stay the last calc_p_rho EOS diagnosis of the pinned fields).
        _spec3 = lambda field, leaf: _apply_3d_spec_only(field, leaf, lead_seconds, config)
        u = _spec3(state.u, state.u_bdy)
        v = _spec3(state.v, state.v_bdy)
        # WRF specified domains do not leaf-pin w at end-of-step; zero_grad_bdy
        # inside the acoustic loop owns the specified w ring.
        w = state.w
        theta = _spec3(state.theta, state.theta_bdy)
        qv = jnp.maximum(_apply_3d(state.qv, state.qv_bdy, lead_seconds, dt_s, config), 0.0)
        qc = _apply_optional_scalar("qc")
        qr = _apply_optional_scalar("qr")
        qi = _apply_optional_scalar("qi")
        qs = _apply_optional_scalar("qs")
        qg = _apply_optional_scalar("qg")
        Ni = _apply_optional_scalar("Ni")
        Nr = _apply_optional_scalar("Nr")
        mu_perturbation = _spec3(state.mu_perturbation[None, :, :], state.mu_bdy)[0]
        mub = _spec3(_base_mu(state)[None, :, :], state.mub_bdy)[0]
        ph_perturbation = _spec3(state.ph_perturbation, state.ph_bdy)
        phb = _spec3(_base_geopotential(state), state.phb_bdy)
        return state.replace(
            u=u,
            v=v,
            w=w,
            theta=theta,
            qv=qv,
            qc=qc,
            qr=qr,
            qi=qi,
            qs=qs,
            qg=qg,
            Ni=Ni,
            Nr=Nr,
            ph_total=phb + ph_perturbation,
            ph_perturbation=ph_perturbation,
            mu_total=mub + mu_perturbation,
            mu_perturbation=mu_perturbation,
        )

    u = _apply_3d(state.u, state.u_bdy, lead_seconds, dt_s, config)
    v = _apply_3d(state.v, state.v_bdy, lead_seconds, dt_s, config)
    w = _apply_3d(state.w, state.w_bdy, lead_seconds, dt_s, config)
    theta = _apply_3d(state.theta, state.theta_bdy, lead_seconds, dt_s, config)
    qv = jnp.maximum(_apply_3d(state.qv, state.qv_bdy, lead_seconds, dt_s, config), 0.0)
    qc = _apply_optional_scalar("qc")
    qr = _apply_optional_scalar("qr")
    qi = _apply_optional_scalar("qi")
    qs = _apply_optional_scalar("qs")
    qg = _apply_optional_scalar("qg")
    Ni = _apply_optional_scalar("Ni")
    Nr = _apply_optional_scalar("Nr")
    p_perturbation = _apply_3d(state.p_perturbation, state.p_bdy, lead_seconds, dt_s, config)
    pb = _apply_3d(_base_pressure(state), state.pb_bdy, lead_seconds, dt_s, config)
    mu_perturbation = _apply_3d(state.mu_perturbation[None, :, :], state.mu_bdy, lead_seconds, dt_s, config)[0]
    mub = _apply_3d(_base_mu(state)[None, :, :], state.mub_bdy, lead_seconds, dt_s, config)[0]
    if config.force_geopotential:
        ph_perturbation = _apply_3d(state.ph_perturbation, state.ph_bdy, lead_seconds, dt_s, config)
        phb = _apply_3d(_base_geopotential(state), state.phb_bdy, lead_seconds, dt_s, config)
        ph_total = phb + ph_perturbation
    else:
        # NESTED-child geopotential handling (force_geopotential=False).
        #
        # The d03 +1.5 K T2 warm bias was traced (2026-06-01 Opus bisection +
        # GPT energy-budget RCAs) to a near-uniform +2.6 kPa diagnostic
        # surface-pressure offset that inflates T2 through the Exner conversion;
        # the offset arises because this branch leaves the perturbation
        # geopotential ph' UNFORCED at the nested boundary while mu'/theta'/qv'
        # ARE forced, so the interior equilibrates to a wrong geopotential
        # reference (ph' "bowed low" mid-column).
        #
        # ATTEMPTED FIX (NOT viable here): re-derive a hydrostatically consistent
        # ph' in the boundary ring from the forced mu'/theta'/qv' (the helpers
        # ``_hydrostatic_ph_perturbation`` / ``_relax_field_to_target`` below) and
        # force it via either a hard ring overwrite or the WRF spec+relax nudge.
        # BOTH variants drive the acoustic w-ph small-step solve unstable (w -> 1e83,
        # p' -> 1e16 by forecast hour 1; proofs/v010_validation/pipeline_run_d03_
        # phfix_short6h{,_v4}.json -> PIPELINE_BLOCKED / NONFINITE_STATE).  Root
        # cause: ph' is an ACOUSTIC-COUPLED prognostic (advanced with w in the
        # small step); injecting an externally-derived ph' value at the END of the
        # step is inconsistent with the carried w and excites the w-ph acoustic
        # resonance.  WRF instead recomputes the child geopotential hydrostatically
        # at vertical-interpolation/init time (med_interp_domain) and, at the
        # lateral boundary, forces the MASS-COUPLED variables through the in-loop
        # relaxation tendency (relax_bdy_dry), NOT via a decoupled end-of-step ph'
        # value.  A faithful fix therefore needs the ph' boundary forcing folded
        # INTO the acoustic small-step loop coupled with w (like the existing
        # ``apply_normal_bdy_work`` does for the normal momentum), which is a dycore
        # change beyond this module.  The stable alternative the bisection flagged
        # (re-reference the DIAGNOSTIC surface pressure feeding T2's Exner, in
        # runtime/surface diagnostics) lives outside this file.  Until one of those
        # lands, leave ph' dynamically/hydrostatically consistent with the child
        # column (the validated-stable free-drift behaviour).
        ph_perturbation = state.ph_perturbation
        ph_total = state.ph_total
    return state.replace(
        u=u,
        v=v,
        w=w,
        theta=theta,
        qv=qv,
        qc=qc,
        qr=qr,
        qi=qi,
        qs=qs,
        qg=qg,
        Ni=Ni,
        Nr=Nr,
        p_total=pb + p_perturbation,
        p_perturbation=p_perturbation,
        ph_total=ph_total,
        ph_perturbation=ph_perturbation,
        mu_total=mub + mu_perturbation,
        mu_perturbation=mu_perturbation,
    )


def interpolate_boundary_leaf(boundary, lead_seconds, cadence_s: float = 3600.0):
    """Linearly interpolate one boundary leaf along its leading time axis.

    ``boundary`` is ``(time, side, bdy_width, z, side_len)``; the result drops
    the time axis: ``(side, bdy_width, z, side_len)``. Equivalent to WRF's
    ``field_bdy + dtbc*field_bdy_tend`` linear-in-time forcing.
    """

    max_index = int(boundary.shape[0]) - 1
    lead_index = jnp.asarray(lead_seconds, dtype=jnp.float64) / float(cadence_s)
    lower = jnp.clip(jnp.floor(lead_index).astype(jnp.int32), 0, max_index)
    upper = jnp.clip(lower + 1, 0, max_index)
    alpha = jnp.clip(lead_index - lower.astype(jnp.float64), 0.0, 1.0)
    lower_values = jnp.take(boundary, lower, axis=0)
    upper_values = jnp.take(boundary, upper, axis=0)
    return (lower_values * (1.0 - alpha) + upper_values * alpha).astype(boundary.dtype)


def _base_pressure(state: State):
    return state.p_total - state.p_perturbation


def _base_geopotential(state: State):
    return state.ph_total - state.ph_perturbation


def _base_mu(state: State):
    return state.mu_total - state.mu_perturbation


# ---------------------------------------------------------------------------
# Nested-child hydrostatic geopotential rebuild (P0-6 / d03 T2 Exner fix).
# ---------------------------------------------------------------------------


def _hydrostatic_ph_perturbation(theta_full, qv, mu_perturbation, mub, pb, metrics: DycoreMetrics):
    """Re-derive a hydrostatically self-consistent perturbation geopotential.

    Reproduces WRF ``calc_p_rho_phi`` (``module_big_step_utilities_em.F``) for
    ``non_hydrostatic=.false., hypsometric_opt==1`` -- the canonical hydrostatic
    pressure / inverse-density / geopotential solve -- and is the EXACT inverse
    of the dycore's nonhydrostatic ``al`` diagnostic (line 1029, used by
    :func:`gpuwrf.dynamics.acoustic_wrf.diagnose_pressure_al_alt`).  Given the
    FORCED column thermodynamics (full ``theta``, ``qv``, perturbation dry mass
    ``mu'``, base dry mass ``mub``, base pressure ``pb``) it returns the
    perturbation geopotential ``ph'`` (faces, ``(nz+1, ny, nx)``) anchored at the
    surface (``ph'[0]=0``) so the total geopotential at the ground equals
    ``phb[0]``.

    Steps (with the pinned-run hybrid coefficients ``c2h==0`` so ``c1h*muts``):
      1. dry hydrostatic pressure ``p`` by downward integration from the lid
         (``:1099-1145``): ``p[top] = -0.5*(c1*mu' + qtot*(c1*muts))/rdnw[top]``,
         ``p[k]   = p[k+1] - (c1*mu' + qtot*(c1*muts))/rdn[k+1]`` with
         ``qtot`` the face-averaged total water (here ``qv`` only, matching the
         single-moisture-species hydrostatic init).
      2. perturbation inverse density from the EOS (``:1115-1142``):
         ``al = (R_d/p0)*theta_full*qvf*((p+pb)/p0)^cvpm - alb``,
         ``alb = (R_d/p0)*t0          *((pb)/p0)^cvpm`` (dry base column),
         ``qvf = 1 + 0.608*qv``.
      3. upward geopotential integration (``:1183-1193``, opt==1):
         ``ph'[k+1] = ph'[k] - dnw[k]*( (c1*muts+c2)*al[k] + c1*mu'*alb[k] )``,
         ``ph'[0]=0``.

    ``dnw`` is the SIGNED eta-face spacing (negative for the normal decreasing
    eta ordering), ``rdnw=1/dnw``, ``rdn`` the mass-level reciprocal spacing,
    matching the dycore metric convention exactly.
    """

    dtype = theta_full.dtype
    c1h = metrics.c1h.astype(dtype)[:, None, None]      # (nz,1,1)
    c2h = metrics.c2h.astype(dtype)[:, None, None]
    rdnw = metrics.rdnw.astype(dtype)[:, None, None]
    rdn = metrics.rdn.astype(dtype)[:, None, None]
    dnw = metrics.dnw.astype(dtype)[:, None, None]

    mu_p = mu_perturbation.astype(dtype)[None, :, :]    # (1,ny,nx)
    muts = (mub.astype(dtype) + mu_perturbation.astype(dtype))[None, :, :]
    pb = pb.astype(dtype)
    qv = qv.astype(dtype)

    nz = theta_full.shape[0]

    # --- 1. dry hydrostatic perturbation pressure p (downward from the lid) ---
    # WRF qtot face-average: top layer uses qv[top]; interior k uses
    # 0.5*(qv[k]+qv[k+1]).  c1*mu' + qf1*(c1*muts+c2) is the layer dry-mass weight.
    # The downward recurrence p[k] = p[k+1] - g[k], with
    #   g[k] = (c1h[k]*mu' + qtot[k]*(c1h[k]*muts+c2h[k])) / rdn[k+1],   k < nz-1
    #   p[nz-1] = -0.5*(c1h*mu' + qtot[nz-1]*(c1h*muts+c2h)) * dnw[nz-1]
    # is a REVERSE cumulative sum (p[k] = p_top - sum_{j=k}^{nz-2} g[j]), expressed
    # below as a vectorised cumsum so the JIT graph stays a single fused op rather
    # than an nz-deep unrolled dependency chain (which compiled pathologically slow).
    mass_full = c1h * muts + c2h                        # (c1h*muts + c2h) per (k,y,x)
    mu_weight = c1h * mu_p                               # c1h * mu'

    qtot_top = qv[nz - 1]
    p_top = -0.5 * (mu_weight[nz - 1] + qtot_top * mass_full[nz - 1]) * dnw[nz - 1]
    #   note: -X / rdnw[k] == -X * dnw[k]  (rdnw = 1/dnw)

    # Per-layer downward decrement g[k] for k = 0 .. nz-2 (face-averaged qtot).
    qtot_face = 0.5 * (qv[:-1] + qv[1:])                 # (nz-1,ny,nx)
    g = (mu_weight[:-1] + qtot_face * mass_full[:-1]) / rdn[1:nz]  # (nz-1,ny,nx)
    # p[k] = p_top - sum_{j=k}^{nz-2} g[j].  Build the suffix-sum from the top down.
    suffix = jnp.cumsum(g[::-1], axis=0)[::-1]           # suffix[k] = sum_{j=k}^{nz-2} g[j]
    p_interior = p_top[None, :, :] - suffix              # (nz-1,ny,nx) for k=0..nz-2
    p_pert = jnp.concatenate((p_interior, p_top[None, :, :]), axis=0)  # (nz,ny,nx)

    # --- 2. perturbation inverse density al via the dry EOS ---
    base_pressure = jnp.maximum(pb, 1.0)
    total_pressure = jnp.maximum(p_pert + pb, 1.0)
    qvf = 1.0 + 0.608 * qv
    alb = (_R_D / _P0_PA) * _THETA_BASE_OFFSET * ((base_pressure / _P0_PA) ** _CVPM)
    al = (_R_D / _P0_PA) * theta_full * qvf * ((total_pressure / _P0_PA) ** _CVPM) - alb

    # --- 3. upward perturbation-geopotential integration (opt==1) ---
    # ph'[k+1] = ph'[k] - dnw[k]*( (c1*muts+c2)*al[k] + c1*mu'*alb[k] ), ph'[0]=0.
    layer_incr = -dnw * (mass_full * al + mu_weight * alb)   # (nz,ny,nx) per-layer ph' delta
    ny = theta_full.shape[1]
    nx = theta_full.shape[2]
    ph0 = jnp.zeros((1, ny, nx), dtype=dtype)
    ph_pert = jnp.concatenate((ph0, jnp.cumsum(layer_incr, axis=0)), axis=0)  # (nz+1,ny,nx)
    return ph_pert


def _forcing_leaf_from_field(field, config: BoundaryConfig):
    """Build a ``(side, bdy_width, z, side_len)`` forcing leaf from a full field.

    Extracts the outer ``spec_zone + relax_zone`` rows/columns of ``field``
    ``(z, ny, nx)`` for each side in the exact layout :func:`_strip` expects, so
    the standard WRF spec + relaxation scatter (:func:`_apply_side_spec` /
    :func:`_apply_side_relax`) can nudge the interior toward ``field`` smoothly
    instead of hard-overwriting it (a hard ring overwrite shocks the acoustic
    w-ph solve and blows up the dycore).  ``side_len`` is padded to the field's
    own tangential extent (this leaf is only ever applied to the SAME field).
    """

    z_len, y_len, x_len = field.shape
    nwidth = int(config.spec_zone + config.relax_zone)
    side_len = max(y_len, x_len)

    def _pad_tan(strip):  # (bdy_width, z, tan) -> (bdy_width, z, side_len)
        tan = strip.shape[-1]
        if tan < side_len:
            strip = jnp.pad(strip, ((0, 0), (0, 0), (0, side_len - tan)), mode="edge")
        return strip

    # W: columns b_dist from the west edge -> strip (bdy_width, z, y).
    w = jnp.stack([field[:, :, b] for b in range(nwidth)], axis=0).transpose(0, 1, 2)
    # transpose to (bdy_width, z, y): field[:, :, b] is (z, y) already.
    e = jnp.stack([field[:, :, x_len - 1 - b] for b in range(nwidth)], axis=0)
    s = jnp.stack([field[:, b, :] for b in range(nwidth)], axis=0)      # (bdy_width, z, x)
    n = jnp.stack([field[:, y_len - 1 - b, :] for b in range(nwidth)], axis=0)
    leaf = jnp.stack([_pad_tan(w), _pad_tan(e), _pad_tan(s), _pad_tan(n)], axis=0)
    return leaf  # (side, bdy_width, z, side_len)


def _relax_field_to_target(field, target, dt_s: float, config: BoundaryConfig):
    """Nudge the boundary ring of ``field`` toward ``target`` via WRF spec+relax.

    Reuses the exact spec-zone hard-set + relaxation-zone Laplacian nudge that
    every other forced field uses, so the geopotential ring transitions smoothly
    from the (hydrostatically consistent) boundary value into the dycore-advanced
    interior.  ``target`` is the full hydrostatic ``ph'`` field; the forcing leaf
    is sampled from it.
    """

    forcing = _forcing_leaf_from_field(target, config)
    out = field
    for side in SIDES:
        out = _apply_side_relax(out, field, forcing, side, dt_s, config)
    for side in SIDES:
        out = _apply_side_spec(out, forcing, side, config)
    return out


def _apply_3d(field, boundary, lead_seconds, dt_s: float, config: BoundaryConfig):
    """WRF spec-zone + relax-zone update for one ``(z, y, x)`` field.

    ``forcing`` is the time-interpolated strip ``(side, bdy_width, z, side_len)``
    whose ``bdy_width`` axis runs outer (index 0, the domain edge) to inner.
    """

    forcing = interpolate_boundary_leaf(boundary, lead_seconds, config.update_cadence_s)
    # WRF applies spec_bdytend (outer zone) and relax_bdytend (relaxation zone)
    # in one pass.  WRF computes every side's relaxation tendency from the SAME
    # input field within an RK substep (the field is not mutated mid-pass), so
    # we evaluate all four relaxation slices against the original ``field`` and
    # only then scatter them.  With WRF corner trimming each relaxation cell is
    # owned by exactly one side, so the scatter order is immaterial.
    out = field
    for side in SIDES:
        out = _apply_side_relax(out, field, forcing, side, dt_s, config)
    # Overwrite the spec zone last so the outer specified rows are exactly the
    # boundary value (WRF spec zone is a hard set; relaxation never touches
    # b_dist < spec_zone).
    for side in SIDES:
        out = _apply_side_spec(out, forcing, side, config)
    return out


def _strip(forcing, side: str, width_index: int, z_len: int, side_len: int):
    """Return the boundary strip ``(z, side_len)`` for one side and bdy width.

    Two leaf layouts are accepted (per the frozen State docstring):
    * Current ``(side, bdy_width, z, side_len)`` -> per-side ``(bdy_width, z,
      side_len)``; we index the requested ``bdy_width``.
    * Legacy ``(side, z, side_len)`` (no ``bdy_width`` axis) -> per-side ``(z,
      side_len)``; every width maps to the same single strip (the relaxation
      normal-neighbour terms then collapse, which is the documented legacy
      behaviour -- only the spec zone is exact).

    ``side_len`` is the tangential extent (``y`` for W/E, ``x`` for S/N), padded
    to ``max(nx+1, ny+1)`` so we slice to the field's actual tangential length.
    """

    side_values = forcing[SIDE_INDEX[side]]
    if side_values.ndim == 3:  # (bdy_width, z, side_len)
        max_width = int(side_values.shape[0]) - 1
        width_index = min(max(int(width_index), 0), max_width)
        return side_values[width_index, :z_len, :side_len]
    # legacy (z, side_len): no bdy_width axis
    return side_values[:z_len, :side_len]


# ---------------------------------------------------------------------------
# Specified (outer) zone: hard-set the outermost spec_zone rows/cols.
# ---------------------------------------------------------------------------


def _apply_side_spec(field, forcing, side: str, config: BoundaryConfig):
    z_len, y_len, x_len = field.shape
    out = field
    for b_dist in range(int(config.spec_zone)):
        if side == "W":
            out = out.at[:, :, b_dist].set(_strip(forcing, side, b_dist, z_len, y_len))
        elif side == "E":
            out = out.at[:, :, x_len - 1 - b_dist].set(_strip(forcing, side, b_dist, z_len, y_len))
        elif side == "S":
            out = out.at[:, b_dist, :].set(_strip(forcing, side, b_dist, z_len, x_len))
        else:  # N
            out = out.at[:, y_len - 1 - b_dist, :].set(_strip(forcing, side, b_dist, z_len, x_len))
    return out


# ---------------------------------------------------------------------------
# Relaxation zone: WRF relax_bdytend stencil on the residual (bdy - field).
# ---------------------------------------------------------------------------


def _shift_clamp(values, shift: int, axis: int):
    """Shift ``values`` by ``shift`` along ``axis``, clamping (reflecting) the
    edge element to match WRF's ``max(i-1,ibs)`` / ``min(i+1,ibe)``."""

    n = values.shape[axis]
    if shift == 1:  # tangential s-1 neighbour: index max(i-1, 0)
        idx = jnp.concatenate((jnp.array([0]), jnp.arange(n - 1)))
    else:  # tangential s+1 neighbour: index min(i+1, n-1)
        idx = jnp.concatenate((jnp.arange(1, n), jnp.array([n - 1])))
    return jnp.take(values, idx, axis=axis)


def _relax_row(field, forcing, side, b_dist, z_len, side_len, *, weight_f, weight_g):
    """Compute the WRF-relaxed row ``b_dist`` for one side.

    Returns the full ``(z, side_len)`` relaxed slice; the caller masks it to the
    WRF tangential extent (corner trimming) before scattering.

    fls0 = bdy[b_dist]   - field(row b_dist)
    fls1 = bdy[b_dist]   - field(row b_dist), tangential +s neighbour
    fls2 = bdy[b_dist]   - field(row b_dist), tangential -s neighbour
    fls3 = bdy[b_dist-1] - field(row b_dist-1)            (normal, toward edge)
    fls4 = bdy[b_dist+1] - field(row b_dist+1)            (normal, toward interior)
    """

    if side == "W":
        cur = field[:, :, b_dist]
        toward_edge = field[:, :, b_dist - 1]
        toward_interior = field[:, :, b_dist + 1]
    elif side == "E":
        x = field.shape[2] - 1 - b_dist
        cur = field[:, :, x]
        toward_edge = field[:, :, x + 1]
        toward_interior = field[:, :, x - 1]
    elif side == "S":
        cur = field[:, b_dist, :]
        toward_edge = field[:, b_dist - 1, :]
        toward_interior = field[:, b_dist + 1, :]
    else:  # N
        y = field.shape[1] - 1 - b_dist
        cur = field[:, y, :]
        toward_edge = field[:, y + 1, :]
        toward_interior = field[:, y - 1, :]

    bdy0 = _strip(forcing, side, b_dist, z_len, side_len)
    bdy_edge = _strip(forcing, side, b_dist - 1, z_len, side_len)
    bdy_interior = _strip(forcing, side, b_dist + 1, z_len, side_len)

    fls0 = bdy0 - cur
    # tangential axis of the (z, side_len) slice is axis 1
    fls1 = _shift_clamp(fls0, +1, axis=1)
    fls2 = _shift_clamp(fls0, -1, axis=1)
    fls3 = bdy_edge - toward_edge
    fls4 = bdy_interior - toward_interior
    laplacian = fls1 + fls2 + fls3 + fls4 - 4.0 * fls0
    return cur + weight_f * fls0 - weight_g * laplacian


def _apply_side_relax(out, field, forcing, side: str, dt_s: float, config: BoundaryConfig):
    """Scatter one side's relaxation rows into ``out``; stencil reads ``field``.

    ``field`` is the unmodified input (WRF reads the same field for all sides in
    one RK substep); ``out`` is the accumulating result that we write into.
    """

    z_len, y_len, x_len = field.shape
    spec_zone = int(config.spec_zone)
    relax_zone = int(config.relax_zone)
    for b_dist in range(spec_zone, relax_zone):
        weight_f, weight_g = _wrf_relax_weights(b_dist, dt_s, config)
        if side in ("W", "E"):
            side_len = y_len
            relaxed = _relax_row(field, forcing, side, b_dist, z_len, side_len, weight_f=weight_f, weight_g=weight_g)
            # WRF X-boundary relaxation tangential j in [b_dist+jbs+1, jbe-b_dist-1]
            start, end = b_dist + 1, y_len - b_dist - 1
            col = b_dist if side == "W" else x_len - 1 - b_dist
            out = out.at[:, start:end, col].set(relaxed[:, start:end])
        else:  # S, N
            side_len = x_len
            relaxed = _relax_row(field, forcing, side, b_dist, z_len, side_len, weight_f=weight_f, weight_g=weight_g)
            # WRF Y-boundary relaxation tangential i in [b_limit+ibs, ibe-b_limit]
            # with b_limit=b_dist -> corners owned by W/E so the diagonal is
            # updated exactly once.
            start, end = b_dist, x_len - b_dist
            row = b_dist if side == "S" else y_len - 1 - b_dist
            out = out.at[:, row, start:end].set(relaxed[:, start:end])
    return out


def _wrf_relax_weights(b_dist: int, dt_s: float, config: BoundaryConfig) -> tuple[float, float]:
    """Return ``dt*fcx`` and ``dt*gcx`` for relaxation row ``b_dist``.

    Reproduces ``lbc_fcx_gcx`` (``dyn_em/module_bc_em.F``). WRF indexes the
    weight by Fortran ``loop = b_dist + 1`` (1-based). With ``spec_zone=1``,
    ``relax_zone=4`` the active rows are ``b_dist = 1,2,3`` -> ``loop = 2,3,4``
    -> linear taper ``1, 2/3, 1/3``. The optional ``spec_exp`` sponge is the
    ``exp(-(loop-(spec_zone+1))*spec_exp)`` multiplier (0 for the pinned run).

    The per-step value update folds WRF's RK ``dt`` factor in: WRF accumulates a
    tendency ``fcx*fls0 - ...`` integrated over ``dt``; one decoupled value step
    is ``dt*fcx*fls0 - ...``.
    """

    loop_1based = int(b_dist) + 1
    numerator = float(config.spec_zone + config.relax_zone - loop_1based)
    denominator = float(config.relax_zone - 1)
    linear = max(0.0, numerator / denominator) if denominator > 0.0 else 0.0
    sponge = math.exp(-(loop_1based - (config.spec_zone + 1)) * config.spec_exp)
    fcx = 0.1 / float(dt_s) * linear * sponge
    gcx = 1.0 / float(dt_s) / 50.0 * linear * sponge
    return float(dt_s) * fcx, float(dt_s) * gcx


# ---------------------------------------------------------------------------
# In-loop NORMAL-momentum boundary protection (WRF advance_uv spec_zone +
# relax_bdy_dry on the COUPLED small-step work array).
# ---------------------------------------------------------------------------
#
# WRF treats the normal momentum (u at W/E, v at S/N) inside the acoustic
# small-step loop, NOT as an end-of-step value nudge:
#
#   * advance_uv (module_small_step_em.F:734-942) restricts its u/v PGF/tendency
#     loops to ``i_start = max(its, ids+spec_zone)`` .. ``i_endu = min(ite,
#     ide-spec_zone)`` (and analogously for v in j), so the OUTERMOST
#     ``spec_zone`` normal face (the domain-edge face, b_dist=0) is NEVER
#     advanced by the acoustic solver.  It is instead driven by
#     ``spec_bdyupdate(u_2, ru_tend, dts_rk, spec_zone)`` (solve_em.F:1346-1364,
#     INSIDE the small_steps DO loop) toward the boundary tendency that
#     ``spec_bdy_dry`` (module_bc_em.F:479-494) wrote into ``ru_tend``/``rv_tend``.
#   * The relaxation zone (b_dist = spec_zone .. relax_zone-1) gets a relaxation
#     tendency folded into ``ru_tend``/``rv_tend`` ONCE per full step at rk_step==1
#     (relax_bdy_dry -> relax_bdytend_core, module_bc.F:1221-1427) computed from
#     the COUPLED momentum residual ``(bdy - ru)``; that tendency is then applied
#     every acoustic substep via ``u_2 += dts*ru_tend`` inside advance_uv.
#
# Our acoustic core advances the COUPLED perturbation work array ``u_work``
# (= ``(mass_ref*u_1 - mass_cur*u_2)/msf``, small_step_prep_wrf), exactly WRF's
# ``grid%u_2``.  So we reproduce both effects directly on the work array:
#   * spec face   : DRIVE ``u_work`` to the work-array boundary target each substep
#                   (equivalent to WRF not advancing it + spec_bdyupdate pinning it
#                   to the boundary value);
#   * relax faces : NUDGE ``u_work`` toward the work-array boundary target with the
#                   relax_bdytend Laplacian stencil, scaled by ``dts`` per substep
#                   (so the per-full-step total matches the WRF tendency once-set,
#                   applied-every-substep behaviour).
#
# The work-array target is the value whose small_step_finish reconstruction equals
# the interpolated decoupled boundary velocity ``u_bdy``:
#     u = (msf*u_work + u_save*mass_cur)/mass_stage   (small_step_finish_wrf)
#  => u_work_bdy = (u_bdy*mass_stage - u_save*mass_cur)/msf .


def normal_bdy_work_target_u(
    u_bdy_strip, u_save, mass_u_cur, mass_u_stage, msfuy, *, config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG
):
    """Coupled work-array target for the W/E normal face (whole ``(z, ny, nx+1)``).

    Only the ``spec_zone + relax_zone`` outer columns at W and E are meaningful;
    interior columns stay zero (the caller only reads the boundary columns).
    ``u_bdy_strip`` is the time-interpolated decoupled boundary leaf for u
    (``(side, bdy_width, z, side_len)``).
    """

    z_len, y_len, x_len = u_save.shape  # x_len == nx+1
    target = jnp.zeros_like(u_save)
    nzone = int(config.spec_zone + config.relax_zone)
    for b_dist in range(nzone):
        w_strip = _strip(u_bdy_strip, "W", b_dist, z_len, y_len)
        e_strip = _strip(u_bdy_strip, "E", b_dist, z_len, y_len)
        cw = b_dist
        ce = x_len - 1 - b_dist
        # msfuy is (ny, nx+1); the W/E column ``c`` map factor is msfuy[:, c] (ny,),
        # broadcast against the (nz, ny) column slice.
        tw = (w_strip * mass_u_stage[:, :, cw] - u_save[:, :, cw] * mass_u_cur[:, :, cw]) / msfuy[:, cw][None, :]
        te = (e_strip * mass_u_stage[:, :, ce] - u_save[:, :, ce] * mass_u_cur[:, :, ce]) / msfuy[:, ce][None, :]
        target = target.at[:, :, cw].set(tw)
        target = target.at[:, :, ce].set(te)
    return target


def normal_bdy_work_target_v(
    v_bdy_strip, v_save, mass_v_cur, mass_v_stage, msfvx, *, config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG
):
    """Coupled work-array target for the S/N normal face (whole ``(z, ny+1, nx)``)."""

    z_len, y_len, x_len = v_save.shape  # y_len == ny+1
    target = jnp.zeros_like(v_save)
    nzone = int(config.spec_zone + config.relax_zone)
    for b_dist in range(nzone):
        s_strip = _strip(v_bdy_strip, "S", b_dist, z_len, x_len)
        n_strip = _strip(v_bdy_strip, "N", b_dist, z_len, x_len)
        rs = b_dist
        rn = y_len - 1 - b_dist
        # msfvx is (ny+1, nx); the S/N row ``r`` map factor is msfvx[r, :] (nx,),
        # broadcast against the (nz, nx) row slice.
        ts = (s_strip * mass_v_stage[:, rs, :] - v_save[:, rs, :] * mass_v_cur[:, rs, :]) / msfvx[rs, :][None, :]
        tn = (n_strip * mass_v_stage[:, rn, :] - v_save[:, rn, :] * mass_v_cur[:, rn, :]) / msfvx[rn, :][None, :]
        target = target.at[:, rs, :].set(ts)
        target = target.at[:, rn, :].set(tn)
    return target


# WIND-FIX relaxation strength multiplier on the WRF per-step relax weight.
# WRF's nominal ``fcx=0.1/dt_model`` (relax_strength=1) is calibrated for WRF's
# fully mass-coupled, consistent boundary pressure.  This side-history replay
# forces the boundary with DECOUPLED wrfout leaves (boundary_apply module
# docstring: an O(mu') approximation), so the nominal weight is far too weak to
# hold the normal momentum against the per-substep acoustic PGF (at strength 1
# the relax zone still spiked v row1 -17->-14.5; proofs/wind/
# fixed_probe_05h_v1_persubstep.log).  The spike-elimination sweep
# (proofs/wind/fixed_probe_vec_s20.log) found strength 20 removes it cleanly:
# v row1 -17->-6, u col1 -10->+0.9, both tracking the boundary smoothly, run
# finite, v|max| 27->18.  This is a single decoupled-replay calibration constant
# (a per-substep convex-blend weight ~0.86/step at the spec-adjacent relax cell),
# NOT a per-cell masking clamp.
#
# v0.14 bdy-auditor (2026-06-11): env-overridable for the venting-driver strength
# sweep (GPUWRF_NORMAL_BDY_RELAX_STRENGTH); default unchanged so production stays
# byte-identical when the env is unset.
NORMAL_BDY_RELAX_STRENGTH = float(os.environ.get("GPUWRF_NORMAL_BDY_RELAX_STRENGTH", "20.0"))


def _normal_relax_weights_u(z_len, y_len, x_len, sub_ratio, config: BoundaryConfig, dtype):
    """Static per-substep convex-blend weight mask for the W/E NORMAL u face.

    Built with numpy at trace time (shapes + config are static), so the whole
    in-loop boundary protection is ONE vectorised ``u += w*(target-u)`` op rather
    than dozens of per-(b_dist, side) scatters -- the loop form produced a
    pathologically slow XLA compile (proofs/wind/fixed_probe_s20 compile alarm).

    Spec columns (b_dist < spec_zone) get weight 1 (full drive to the boundary
    work target == WRF advance_uv excluding them + spec_bdyupdate pinning them).
    Relax columns get ``clip(sub_ratio*0.1*linear(b_dist), 0, 1)`` with WRF's
    X-boundary tangential corner trim ``j in [b_dist+1, ny-b_dist-2]``.
    """

    import numpy as _np

    spec_zone = int(config.spec_zone)
    relax_zone = int(config.relax_zone)
    w = _np.zeros((y_len, x_len), dtype=_np.float64)
    for b_dist in range(spec_zone):
        w[:, b_dist] = 1.0
        w[:, x_len - 1 - b_dist] = 1.0
    for b_dist in range(spec_zone, relax_zone):
        loop_1based = b_dist + 1
        linear = max(0.0, (spec_zone + relax_zone - loop_1based) / float(relax_zone - 1)) if relax_zone > 1 else 0.0
        weight = min(1.0, max(0.0, sub_ratio * 0.1 * linear))
        start, end = b_dist + 1, y_len - b_dist - 1  # WRF tangential corner trim
        w[start:end, b_dist] = weight
        w[start:end, x_len - 1 - b_dist] = weight
    return jnp.asarray(w[None, :, :], dtype=dtype)


def _normal_relax_weights_v(z_len, y_len, x_len, sub_ratio, config: BoundaryConfig, dtype):
    """Static per-substep convex-blend weight mask for the S/N NORMAL v face.

    Relax rows use WRF's Y-boundary tangential corner trim ``i in [b_dist,
    nx-b_dist-1]`` so the diagonal corners are owned by the W/E (u) boundaries.
    """

    import numpy as _np

    spec_zone = int(config.spec_zone)
    relax_zone = int(config.relax_zone)
    w = _np.zeros((y_len, x_len), dtype=_np.float64)
    for b_dist in range(spec_zone):
        w[b_dist, :] = 1.0
        w[y_len - 1 - b_dist, :] = 1.0
    for b_dist in range(spec_zone, relax_zone):
        loop_1based = b_dist + 1
        linear = max(0.0, (spec_zone + relax_zone - loop_1based) / float(relax_zone - 1)) if relax_zone > 1 else 0.0
        weight = min(1.0, max(0.0, sub_ratio * 0.1 * linear))
        start, end = b_dist, x_len - b_dist  # WRF tangential corner trim
        w[b_dist, start:end] = weight
        w[y_len - 1 - b_dist, start:end] = weight
    return jnp.asarray(w[None, :, :], dtype=dtype)


def apply_normal_bdy_work(
    u_work,
    v_work,
    u_target,
    v_target,
    dts: float,
    dt_full: float,
    *,
    config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
    relax_strength: float | None = None,
):
    """Apply WRF spec-freeze + relaxation to the NORMAL momentum work arrays.

    ``u_work``/``v_work`` are the post-``advance_uv`` coupled small-step work
    arrays (WRF ``grid%u_2``/``grid%v_2``); ``u_target``/``v_target`` are the
    work-array boundary targets from :func:`normal_bdy_work_target_u`/``_v``.

    Implementation: a per-substep CONVEX BLEND toward the boundary work target,
    ``u = u + w*(target - u)``.  The weight ``w`` is built from the WRF
    relaxation taper -- spec column weight 1 (WRF advance_uv excludes the spec
    face + spec_bdyupdate pins it), relax columns
    ``clip(dts/dt_full * strength * 0.1 * linear(b_dist), 0, 1)`` with WRF's
    corner trimming.  WRF folds ``fcx*fls0`` into ``ru_tend`` once and applies it
    every substep via ``u_2 += dts*ru_tend``; the convex blend is the
    contraction-stable equivalent on the recomputed residual (the WRF additive
    form on a decoupled-replay boundary was too weak to hold the normal momentum
    against the per-substep acoustic PGF, so ``strength`` raises the pull -- a
    single calibration constant, NOT a per-cell clamp).  ``dts`` is the acoustic
    substep, ``dt_full`` the model timestep (WRF ``grid%dt``).

    Tangential components (v at W/E, u at S/N) are LEFT UNTOUCHED -- WRF's
    relax/spec for those is the existing end-of-step ``apply_lateral_boundaries``
    path, and the diagnosis localised the blow-up to the normal component only.
    """

    strength = float(NORMAL_BDY_RELAX_STRENGTH if relax_strength is None else relax_strength)
    # dts/dt_full converts the WRF per-step relax weight to a per-substep one.
    sub_ratio = strength * (float(dts) / float(dt_full) if float(dt_full) != 0.0 else 0.0)

    zu, yu, xu = u_work.shape
    wu = _normal_relax_weights_u(zu, yu, xu, sub_ratio, config, u_work.dtype)
    u = u_work + wu * (u_target - u_work)

    zv, yv, xv = v_work.shape
    wv = _normal_relax_weights_v(zv, yv, xv, sub_ratio, config, v_work.dtype)
    v = v_work + wv * (v_target - v_work)

    return u, v


# ---------------------------------------------------------------------------
# In-acoustic-loop NESTED ph'/w boundary forcing (P0-6 / d03 T2 Exner fix).
# ---------------------------------------------------------------------------
#
# WRF cadence for the perturbation geopotential ph_2 and (for nests) w_2 at a
# nested lateral boundary, established from pristine WRF + the GPT-5.5 review
# (.agent/reviews/2026-06-01-gpt-nest-ph-boundary-wrf-review.md):
#
#   * relax zone (b_dist = spec_zone .. relax_zone-1): WRF builds a relaxation
#     tendency ONCE per RK stage in ``relax_bdy_dry`` (module_bc_em.F:274-344)
#     from the MASS-WEIGHTED full-level geopotential ``rfield = mass_weight(ph,
#     mut, c1f, c2f)`` (and, for nests, ``mass_weight(w)``) relaxed toward the
#     parent boundary leaf via the ``relax_bdytend`` stencil
#     (fcx*fls0 - gcx*laplacian, module_bc.F:1293-1427).  ``rk_addtend_dry``
#     (module_em.F:107-110) folds it into the carried ``ph_tend``/``rw_tend`` as
#     ``ph_tend += ph_tendf/msfty``; those total tendencies are then carried
#     UNCHANGED through every acoustic substep and applied INSIDE ``advance_w``
#     (rhs of the phi equation gets ``dts*ph_tend``; the buoyancy/PGF source gets
#     ``rw_tend``).  So the relaxation forcing flows THROUGH the implicit (w,ph)
#     solve -- intrinsically coupled with w, never an after-the-fact overwrite.
#     THIS is the piece the failed end-of-step attempt skipped.
#
#   * specified zone (the outermost spec_zone row, b_dist < spec_zone): WRF
#     updates ph_2 every acoustic substep AFTER advance_w via the special
#     mass-coupled ``spec_bdyupdate_ph`` (module_bc_em.F:17-157, called
#     solve_em.F:1587), then (for nests) ``spec_bdyupdate(w_2)``, then
#     ``calc_p_rho``.
#
# The TARGET is the parent boundary leaf ``ph_bdy`` (the d02->d03 interpolated
# PARENT perturbation geopotential), time-interpolated ONCE per step -- a frozen
# boundary-cadence attractor, NOT a moving target re-derived from the live
# acoustic state each substep (the GPT review's "moving attractor" pitfall).  An
# offline check (scripts/diag/d03_phfix_target_check.py) confirmed the re-derived
# hydrostatic ``_hydrostatic_ph_perturbation`` diverges from the real (corpus-
# matching) ph' by hundreds-to-thousands of m^2/s^2 aloft, while the state ph'
# already equals the parent leaf to within a few m^2/s^2 at every ring level at
# the IC -- so the parent leaf is the correct WRF-faithful target.


SPEC_PLUS_RELAX_WIDTH = int(DEFAULT_BOUNDARY_CONFIG.spec_zone + DEFAULT_BOUNDARY_CONFIG.relax_zone)


def _full_ring_target_from_leaf(leaf, z_len, y_len, x_len, dtype):
    """Scatter the time-interpolated boundary strip ``leaf`` into a full ring field.

    ``leaf`` is ``(side, bdy_width, z, side_len)``.  Returns a ``(z, ny, nx)`` array
    whose ``spec_zone + relax_zone`` outer rows/columns hold the boundary value at
    each ``b_dist`` (interior left zero -- the relaxation stencil only reads the
    ring out to b_dist=relax_zone, and the innermost relax row's interior-pointing
    neighbour ``fls4`` reads against the live field, which is exactly WRF's
    ``field(i+1)`` interior cell, so the zero fill there is never used).
    """

    target = jnp.zeros((z_len, y_len, x_len), dtype=dtype)
    for b_dist in range(SPEC_PLUS_RELAX_WIDTH):
        w_strip = _strip(leaf, "W", b_dist, z_len, y_len).astype(dtype)  # (z, y)
        e_strip = _strip(leaf, "E", b_dist, z_len, y_len).astype(dtype)
        s_strip = _strip(leaf, "S", b_dist, z_len, x_len).astype(dtype)  # (z, x)
        n_strip = _strip(leaf, "N", b_dist, z_len, x_len).astype(dtype)
        target = target.at[:, :, b_dist].set(w_strip)
        target = target.at[:, :, x_len - 1 - b_dist].set(e_strip)
        target = target.at[:, b_dist, :].set(s_strip)
        target = target.at[:, y_len - 1 - b_dist, :].set(n_strip)
    return target


def _relax_tendency_row(field, target, side, b_dist, *, fcx, gcx):
    """WRF ``relax_bdytend`` tendency increment for one row (full-field target).

    Mirrors :func:`_relax_row` but (a) returns the TENDENCY contribution
    ``fcx*fls0 - gcx*(fls1+fls2+fls3+fls4-4*fls0)`` (NOT the value update), and
    (b) reads the boundary value from a FULL-FIELD ring target ``target``
    ``(z, ny, nx)`` at the SAME grid locations as ``field``, so the residual
    ``bdy - field`` and its tangential/normal neighbours align cell-for-cell.
    The five WRF residual stencil points (fls0..fls4) use the tangential clamp
    (``_shift_clamp``) and the normal toward-edge / toward-interior neighbours.
    """

    if side == "W":
        cur, t_edge, t_int = field[:, :, b_dist], field[:, :, b_dist - 1], field[:, :, b_dist + 1]
        b0, b_edge, b_int = target[:, :, b_dist], target[:, :, b_dist - 1], target[:, :, b_dist + 1]
    elif side == "E":
        x = field.shape[2] - 1 - b_dist
        cur, t_edge, t_int = field[:, :, x], field[:, :, x + 1], field[:, :, x - 1]
        b0, b_edge, b_int = target[:, :, x], target[:, :, x + 1], target[:, :, x - 1]
    elif side == "S":
        cur, t_edge, t_int = field[:, b_dist, :], field[:, b_dist - 1, :], field[:, b_dist + 1, :]
        b0, b_edge, b_int = target[:, b_dist, :], target[:, b_dist - 1, :], target[:, b_dist + 1, :]
    else:  # N
        y = field.shape[1] - 1 - b_dist
        cur, t_edge, t_int = field[:, y, :], field[:, y + 1, :], field[:, y - 1, :]
        b0, b_edge, b_int = target[:, y, :], target[:, y + 1, :], target[:, y - 1, :]

    fls0 = b0 - cur
    # tangential axis of the (z, side_len) slice is axis 1
    fls1 = _shift_clamp(fls0, +1, axis=1)
    fls2 = _shift_clamp(fls0, -1, axis=1)
    fls3 = b_edge - t_edge
    fls4 = b_int - t_int
    laplacian = fls1 + fls2 + fls3 + fls4 - 4.0 * fls0
    return float(fcx) * fls0 - float(gcx) * laplacian


def _scatter_relax_tendency(field_coupled, target_coupled, dt_full: float, config: BoundaryConfig):
    """Build the relaxation-zone tendency (per-second) for a mass-coupled field.

    Reproduces WRF ``relax_bdy_dry`` (the ``relax_bdytend_tile`` half) for one
    ``(z, ny, nx)`` mass-coupled field ``(c1f*mut+c2f)*ph`` relaxed toward the
    mass-coupled full-ring boundary target ``target_coupled``.  Returns a full
    ``(z, ny, nx)`` tendency with the WRF corner trim, NONZERO only in the
    relaxation zone (b_dist = spec_zone .. relax_zone-1).  Units: same as
    ``field_coupled`` per second (the ``fcx=0.1/dt`` taper integrates to a
    0.1*residual full-step nudge at the spec-adjacent cell), so the caller adds
    ``tend/msfty`` to ``ph_tend``.
    """

    z_len, y_len, x_len = field_coupled.shape
    tend = jnp.zeros_like(field_coupled)
    spec_zone = int(config.spec_zone)
    relax_zone = int(config.relax_zone)
    for b_dist in range(spec_zone, relax_zone):
        fcx, gcx = _wrf_relax_weights(b_dist, dt_full, config)
        # _wrf_relax_weights returns dt*fcx, dt*gcx; the per-second tendency is the
        # WRF fcx/gcx, i.e. (dt*fcx)/dt.  Divide by dt_full to recover fcx, gcx.
        fcx = fcx / float(dt_full)
        gcx = gcx / float(dt_full)
        for side in SIDES:
            row_t = _relax_tendency_row(field_coupled, target_coupled, side, b_dist, fcx=fcx, gcx=gcx)
            if side in ("W", "E"):
                start, end = b_dist + 1, y_len - b_dist - 1  # WRF X-bdy tangential trim
                col = b_dist if side == "W" else x_len - 1 - b_dist
                tend = tend.at[:, start:end, col].add(row_t[:, start:end])
            else:  # S, N
                start, end = b_dist, x_len - b_dist  # WRF Y-bdy tangential trim
                row = b_dist if side == "S" else y_len - 1 - b_dist
                tend = tend.at[:, row, start:end].add(row_t[:, start:end])
    return tend


def nested_ph_relax_tendency(ph_perturbation, ph_bdy_leaf, mut, msfty, c1f, c2f, dt_full: float, config: BoundaryConfig):
    """Relaxation-zone ``ph_tend`` contribution for the nested boundary (WRF-faithful).

    Builds the WRF ``relax_bdy_dry`` ph relaxation: scatter the parent boundary
    STRIP leaf into a full ring target, mass-weight both the full-level
    perturbation geopotential and the target by ``(c1f*mut+c2f)`` (WRF
    ``mass_weight(ph, mut, c1f, c2f)``), relax via the ``relax_bdytend`` stencil,
    then divide by ``msfty`` (WRF ``rk_addtend_dry`` ``ph_tend += ph_tendf/msfty``).
    The result is ADDED to the stage ``ph_tend`` so it flows through ``advance_w``
    every acoustic substep, coupled with w.

    ``ph_perturbation`` is perturbation geopotential on faces ``(nz+1, ny, nx)``;
    ``ph_bdy_leaf`` is the time-interpolated parent STRIP leaf
    ``(side, bdy_width, z, side_len)``.
    """

    dtype = ph_perturbation.dtype
    z_len, y_len, x_len = ph_perturbation.shape
    target = _full_ring_target_from_leaf(ph_bdy_leaf, z_len, y_len, x_len, dtype)
    mass_f = c1f.astype(dtype)[:, None, None] * mut.astype(dtype)[None, :, :] + c2f.astype(dtype)[:, None, None]
    field_coupled = mass_f * ph_perturbation
    target_coupled = mass_f * target
    tend_coupled = _scatter_relax_tendency(field_coupled, target_coupled, float(dt_full), config)
    return tend_coupled / msfty.astype(dtype)[None, :, :]


def nested_w_relax_tendency(w, w_bdy_leaf, mut, msfty, c1f, c2f, dt_full: float, config: BoundaryConfig):
    """Relaxation-zone ``rw_tend`` contribution for the nested boundary (WRF nests only).

    WRF ``relax_bdy_dry`` (module_bc_em.F:320-344) relaxes the MASS-WEIGHTED w for
    nested domains; ``rk_addtend_dry`` folds it as ``rw_tend += rw_tendf/msfty``.
    ``w`` is vertical velocity on faces ``(nz+1, ny, nx)``; ``w_bdy_leaf`` is the
    parent STRIP leaf.
    """

    dtype = w.dtype
    z_len, y_len, x_len = w.shape
    target = _full_ring_target_from_leaf(w_bdy_leaf, z_len, y_len, x_len, dtype)
    mass_f = c1f.astype(dtype)[:, None, None] * mut.astype(dtype)[None, :, :] + c2f.astype(dtype)[:, None, None]
    field_coupled = mass_f * w
    target_coupled = mass_f * target
    tend_coupled = _scatter_relax_tendency(field_coupled, target_coupled, float(dt_full), config)
    return tend_coupled / msfty.astype(dtype)[None, :, :]


# ---------------------------------------------------------------------------
# SPECIFIED-domain WRF boundary cadence (v0.14 stage3/wrapper-cadence sprint).
# ---------------------------------------------------------------------------
#
# WRF cadence for a SPECIFIED real domain (d01, wrfbdy-driven), from pristine
# solve_em.F / module_bc_em.F / module_bc.F:
#
#   * relax zone (b_dist = spec_zone .. relax_zone-1): ``relax_bdy_dry`` runs
#     ONCE per step at rk_step==1 (solve_em.F:938-965) on the STEP-START fields,
#     adding the Davies tendency (fcx*fls0 - gcx*laplacian, relax_bdytend_core)
#     for the COUPLED ru/rv (couple_momentum), the MASS-WEIGHTED t (c1h/c2h) and
#     ph (c1f/c2f), and the plain 2-D mu.  ``rk_addtend_dry`` folds u/v/t/ph into
#     the step-constant ``*_tendf`` lane consumed at EVERY RK stage; the mu relax
#     goes straight into the stage-1 ``mu_tend`` and is NOT re-added at stages
#     2-3 (rk_tendency zeroes mu_tend each stage -- a WRF quirk we preserve).
#     NO w relax under specified (module_bc_em.F:320-344 is nested-only).
#
#   * spec zone (ring 0): the small-step routines EXCLUDE it; instead
#     ``spec_bdyupdate`` advances u_2/v_2/t_2/mu_2/muts by ``dts*bdy_tend`` every
#     acoustic substep (solve_em.F:1346-1490), ``spec_bdyupdate_ph`` updates the
#     coupled ph_2 (:1587), and ``zero_grad_bdy`` copies the nearest interior w
#     into the ring (:1601-1609, specified only).  Net effect: ring 0 follows the
#     linear-in-time wrfbdy trajectory within the step.  Our in-loop equivalent
#     pins the ring-0 WORK arrays to the value whose small_step_finish
#     reconstruction equals the STAGE-END interpolated leaf (the within-stage
#     linearisation difference is <= dt_stage/cadence of the interval increment,
#     ~0.1% -- second-order against the 200 Pa/stage free-dynamics drift this
#     replaces).
#
#   * end of step: WRF has NO end-of-step dry-field overwrite and NEVER forces
#     the diagnostic p; ``apply_lateral_boundaries(dry_spec_only=True)`` keeps
#     only the ring-0 spec re-sync (idempotent against the in-loop pins) and the
#     full moisture handling, dropping the once-per-step relax-zone value nudge
#     (replaced by the per-stage tendencies above) and the p'/pb forcing.
#
# Approximation kept from the existing helpers (documented, O(ring mu drift)):
# our leaves store DECOUPLED values, so relax residuals couple BOTH sides with
# the step-start reference mass instead of WRF's file-coupled bdy values.


@dataclass(frozen=True)
class SpecifiedRelaxTendencies:
    """Step-constant WRF ``relax_bdy_dry`` tendencies for the specified domain.

    ``ru``/``rv`` are COUPLED momentum tendencies (added to the coupled
    ``ru_tend``/``rv_tend`` lane at every RK stage, the rk_addtend_dry net
    effect).  ``t`` and ``ph`` are mass-coupled and ALREADY divided by ``msfty``
    (the ``t_tend += t_tendf/msfty`` convention).  ``mu`` is the plain 2-D mass
    relax tendency, applied at rk_step==1 only (the WRF quirk).
    """

    ru: jax.Array
    rv: jax.Array
    t: jax.Array
    ph: jax.Array
    mu: jax.Array


def specified_relax_dry_tendencies(
    reference,
    lead_seconds,
    metrics,
    dt_full: float,
    config: BoundaryConfig,
):
    """Build the WRF ``relax_bdy_dry`` tendency bundle from the step-start state.

    ``reference`` is the step-start (rk1 reference) :class:`State`; targets are
    the time-interpolated decoupled boundary leaves at the step-start lead
    (WRF evaluates ``bdy + dtbc*bdy_tend`` at the rk_step==1 call).  All five
    relax stencils reuse :func:`_scatter_relax_tendency` (the exact WRF
    relax_bdytend_core port with corner trims); the staggered u/v shapes flow
    through unchanged because the stencil indexes the last two axes generically,
    which reproduces WRF's ``ibe=ide`` (u) / ``jbe=jde`` (v) extensions.
    """

    cadence = float(config.update_cadence_s)
    dtype = reference.u.dtype
    c1h = metrics.c1h.astype(dtype)[:, None, None]
    c2h = metrics.c2h.astype(dtype)[:, None, None]
    c1f = metrics.c1f.astype(dtype)[:, None, None]
    c2f = metrics.c2f.astype(dtype)[:, None, None]

    mu_total = reference.mu_total.astype(dtype)
    # face-averaged full dry mass (WRF calculate_full muu/muv; edge-padded faces)
    muu = 0.5 * (
        jnp.concatenate([mu_total[:, :1], mu_total], axis=1)
        + jnp.concatenate([mu_total, mu_total[:, -1:]], axis=1)
    )
    muv = 0.5 * (
        jnp.concatenate([mu_total[:1, :], mu_total], axis=0)
        + jnp.concatenate([mu_total, mu_total[-1:, :]], axis=0)
    )

    z_u = int(reference.u.shape[0])
    y_u = int(reference.u.shape[1])
    x_u = int(reference.u.shape[2])
    z_v, y_v, x_v = (int(s) for s in reference.v.shape)
    nz = int(reference.theta.shape[0])
    ny = int(reference.theta.shape[1])
    nx = int(reference.theta.shape[2])
    nzp1 = int(reference.ph_perturbation.shape[0])

    u_leaf = interpolate_boundary_leaf(reference.u_bdy, lead_seconds, cadence)
    v_leaf = interpolate_boundary_leaf(reference.v_bdy, lead_seconds, cadence)
    t_leaf = interpolate_boundary_leaf(reference.theta_bdy, lead_seconds, cadence)
    ph_leaf = interpolate_boundary_leaf(reference.ph_bdy, lead_seconds, cadence)
    mu_leaf = interpolate_boundary_leaf(reference.mu_bdy, lead_seconds, cadence)

    u_target = _full_ring_target_from_leaf(u_leaf, z_u, y_u, x_u, dtype)
    v_target = _full_ring_target_from_leaf(v_leaf, z_v, y_v, x_v, dtype)
    t_target = _full_ring_target_from_leaf(t_leaf, nz, ny, nx, dtype)
    ph_target = _full_ring_target_from_leaf(ph_leaf, nzp1, ny, nx, dtype)
    mu_target = _full_ring_target_from_leaf(mu_leaf, 1, ny, nx, dtype)[0]

    # COUPLED residual space (couple_momentum / mass_weight); both sides use the
    # step-start reference mass (decoupled-leaf approximation, see block comment).
    mass_u = c1h * muu[None, :, :] + c2h
    mass_v = c1h * muv[None, :, :] + c2h
    mass_h = c1h * mu_total[None, :, :] + c2h
    mass_f = c1f * mu_total[None, :, :] + c2f
    msfuy = metrics.msfuy.astype(dtype)[None, :, :]
    msfvx = metrics.msfvx.astype(dtype)[None, :, :]
    msfty = metrics.msfty.astype(dtype)[None, :, :]

    ru_relax = _scatter_relax_tendency(
        mass_u * reference.u.astype(dtype) / msfuy,
        mass_u * u_target / msfuy,
        float(dt_full),
        config,
    )
    rv_relax = _scatter_relax_tendency(
        mass_v * reference.v.astype(dtype) / msfvx,
        mass_v * v_target / msfvx,
        float(dt_full),
        config,
    )
    # theta: same mass couples both sides, so the WRF t0=300 offset cancels in
    # the residual and the full-theta leaf convention can be used directly.
    t_relax = _scatter_relax_tendency(
        mass_h * reference.theta.astype(dtype),
        mass_h * t_target,
        float(dt_full),
        config,
    ) / msfty
    ph_relax = _scatter_relax_tendency(
        mass_f * reference.ph_perturbation.astype(dtype),
        mass_f * ph_target,
        float(dt_full),
        config,
    ) / msfty
    mu_relax = _scatter_relax_tendency(
        reference.mu_perturbation.astype(dtype)[None, :, :],
        mu_target[None, :, :],
        float(dt_full),
        config,
    )[0]

    return SpecifiedRelaxTendencies(ru=ru_relax, rv=rv_relax, t=t_relax, ph=ph_relax, mu=mu_relax)


def tangential_bdy_work_target_u(
    u_bdy_strip, u_save, mass_u_cur, mass_u_stage, msfuy, *, config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG
):
    """Coupled work-array ring-0 target for the TANGENTIAL u rows (S/N edges).

    Same reconstruction algebra as :func:`normal_bdy_work_target_u`, on the
    spec-zone ROWS of the u faces (WRF ``spec_bdyupdate(u, 'u')`` covers the
    y-side spec rows with the full face range -- y-sides own the corners).
    Only the ``spec_zone`` outer rows are meaningful; the rest stays zero.
    """

    z_len, y_len, x_len = u_save.shape  # x_len == nx+1
    target = jnp.zeros_like(u_save)
    for b_dist in range(int(config.spec_zone)):
        s_strip = _strip(u_bdy_strip, "S", b_dist, z_len, x_len)
        n_strip = _strip(u_bdy_strip, "N", b_dist, z_len, x_len)
        rs = b_dist
        rn = y_len - 1 - b_dist
        ts = (s_strip * mass_u_stage[:, rs, :] - u_save[:, rs, :] * mass_u_cur[:, rs, :]) / msfuy[rs, :][None, :]
        tn = (n_strip * mass_u_stage[:, rn, :] - u_save[:, rn, :] * mass_u_cur[:, rn, :]) / msfuy[rn, :][None, :]
        target = target.at[:, rs, :].set(ts)
        target = target.at[:, rn, :].set(tn)
    return target


def tangential_bdy_work_target_v(
    v_bdy_strip, v_save, mass_v_cur, mass_v_stage, msfvx, *, config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG
):
    """Coupled work-array ring-0 target for the TANGENTIAL v columns (W/E edges).

    WRF ``spec_bdyupdate(v, 'v')`` x-side rows trim the corners (j in
    [b_dist+1, jbe-b_dist-1]); the caller applies that trim when scattering.
    """

    z_len, y_len, x_len = v_save.shape  # y_len == ny+1
    target = jnp.zeros_like(v_save)
    for b_dist in range(int(config.spec_zone)):
        w_strip = _strip(v_bdy_strip, "W", b_dist, z_len, y_len)
        e_strip = _strip(v_bdy_strip, "E", b_dist, z_len, y_len)
        cw = b_dist
        ce = x_len - 1 - b_dist
        tw = (w_strip * mass_v_stage[:, :, cw] - v_save[:, :, cw] * mass_v_cur[:, :, cw]) / msfvx[:, cw][None, :]
        te = (e_strip * mass_v_stage[:, :, ce] - v_save[:, :, ce] * mass_v_cur[:, :, ce]) / msfvx[:, ce][None, :]
        target = target.at[:, :, cw].set(tw)
        target = target.at[:, :, ce].set(te)
    return target


def spec_bdyupdate_ph_inloop(
    ph_work, ph_bdy_leaf, ph_save, mu_tend, muts, c1f, c2f, dts: float, config: BoundaryConfig
):
    """WRF ``spec_bdyupdate_ph`` (spec-zone, in-acoustic-loop) on the ph WORK delta.

    Source: WRF ``module_bc_em.F:17-157`` (called solve_em.F:1587 every acoustic
    substep AFTER ``advance_w``).  WRF updates the COUPLED ``ph_2`` in the outer
    ``spec_zone`` row:

      field = field*(c1*mu_old+c2)/(c1*muts+c2)
            + dts*field_tend/(c1*muts+c2)
            + ph_save*((c1*mu_old+c2)/(c1*muts+c2) - 1)

    with ``mu_old = muts - dts*mu_tend``.  Our small-step ``ph`` array is the
    UNCOUPLED perturbation-delta ``ph_work = ph'_ref - ph'`` (small_step_finish
    reconstructs ``ph' = ph_work + ph_save``).  Translating WRF's coupled formula
    to our uncoupled-delta representation: WRF's coupled ``ph_2 = (c1*muts+c2)*ph'``
    so the spec-zone full perturbation geopotential is driven to the boundary leaf
    value (its mass-reweight + tendency form), i.e. ``ph'(spec) -> ph_bdy_leaf``.
    Therefore in the uncoupled-delta space the spec-zone work delta becomes
    ``ph_work(spec) = ph_bdy_leaf - ph_save`` (so that ph_work + ph_save = leaf).

    This is a SINGLE-row hard pin (spec_zone=1), applied to the outermost row only;
    the relaxation zone is owned by the ``ph_tend`` path inside ``advance_w``.  The
    mass-reweight + mu_tend correction terms are O(mu') and ~0 for this fixed-mass
    replay (muts==mut, mu_tend~0), so the leaf-pin is the dominant, WRF-consistent
    effect; we apply the leaf value directly (the exact WRF reweight reduces to it
    when mu_old==muts).  Full-level extent (WRF ``ktf=kte`` for 'h').
    """

    del mu_tend, muts, c1f, c2f, dts  # O(mu') reweight terms ~0 for fixed-mass replay
    z_len, y_len, x_len = ph_work.shape
    target_delta = ph_bdy_leaf.astype(ph_work.dtype) - ph_save.astype(ph_work.dtype)
    out = ph_work
    for b_dist in range(int(config.spec_zone)):
        # W / E columns
        out = out.at[:, :, b_dist].set(target_delta[:, :, b_dist])
        out = out.at[:, :, x_len - 1 - b_dist].set(target_delta[:, :, x_len - 1 - b_dist])
        # S / N rows
        out = out.at[:, b_dist, :].set(target_delta[:, b_dist, :])
        out = out.at[:, y_len - 1 - b_dist, :].set(target_delta[:, y_len - 1 - b_dist, :])
    return out


__all__ = [
    "BoundaryConfig",
    "DEFAULT_BOUNDARY_CONFIG",
    "SIDES",
    "SIDE_INDEX",
    "apply_lateral_boundaries",
    "interpolate_boundary_leaf",
    "normal_bdy_work_target_u",
    "normal_bdy_work_target_v",
    "apply_normal_bdy_work",
    "nested_ph_relax_tendency",
    "nested_w_relax_tendency",
    "spec_bdyupdate_ph_inloop",
    "SpecifiedRelaxTendencies",
    "specified_relax_dry_tendencies",
    "tangential_bdy_work_target_u",
    "tangential_bdy_work_target_v",
]
