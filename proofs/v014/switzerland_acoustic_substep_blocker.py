#!/usr/bin/env python
"""V0.14 Switzerland h36 acoustic-substep blocker localization.

Compares the JAX operational RK-stage-boundary states for steps 7201/7202
against the WRF-native stage-boundary truth already captured by the HPG
native-face sprint (hpg_dumps calls 21601-21606: the live mu/p/ph/al/alt/
php/cqu/cqv/muu/muv arrays seen by ``horizontal_pressure_gradient`` at
RK1/RK2/RK3 of steps 7201-7202 of the bit-exact 36h30m truth re-run).

The JAX side replicates ``operational_mode._physics_boundary_step`` EAGERLY
(no jax.jit) with captures after every RK stage, starting from the same h36
state proven bit-identical to WRF call 21601.  The replica is validated
against the real jitted ``_physics_boundary_step`` (same inputs -> same final
state) so the captures describe the production path, not a lookalike.

Mapping (ncall = (step-1)*3 + rk; JAX re-init step 1 == WRF step 7201):
  step 1 stage-1 out -> call 21602 (RK2 of 7201)
  step 1 stage-2 out -> call 21603 (RK3 of 7201)
  step 1 final (post moisture/physics/boundary) -> call 21604 (RK1 of 7202)
  step 2 stage-1 out -> call 21605, stage-2 out -> call 21606.

Modes:
  --stage-compare  run the eager replica + WRF comparison (GPU)
  --analyze        collate the proof JSON verdict
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_HPG_SPEC = importlib.util.spec_from_file_location(
    "hpg_native_face_proof", Path(__file__).with_name("switzerland_hpg_native_face_fix.py")
)
hpg = importlib.util.module_from_spec(_HPG_SPEC)
_HPG_SPEC.loader.exec_module(hpg)  # type: ignore[union-attr]

NATIVE_ROOT = hpg.NATIVE_ROOT
PROBE_ROOT = hpg.PROBE_ROOT
FIX_GPU = PROBE_ROOT / "gpu_output_acoustic_substep_fix"
OUT_JSON = ROOT / "proofs/v014/switzerland_acoustic_substep_blocker.json"

# Lateral spec(1)+relax(4) zone is 5 wide; interior verdict region starts at 8
# (same depth as the venting budget control surface).
INTERIOR_DEPTH = 8


def _stats(arr: np.ndarray) -> dict[str, float]:
    values = np.asarray(arr, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.nanmean(values)),
        "rmse": float(np.sqrt(np.nanmean(values * values))),
        "max_abs": float(np.nanmax(np.abs(values))),
    }


def _region_masks(ny: int, nx: int, depth: int = INTERIOR_DEPTH) -> tuple[np.ndarray, np.ndarray]:
    jj, ii = np.mgrid[0:ny, 0:nx]
    interior = (ii >= depth) & (ii < nx - depth) & (jj >= depth) & (jj < ny - depth)
    return interior, ~interior


def _split_stats(diff: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Stats over full/interior/boundary-band horizontal regions of [k,j,i] or [j,i]."""

    diff = np.asarray(diff, dtype=np.float64)
    if diff.ndim == 2:
        ny, nx = diff.shape
        interior2, band2 = _region_masks(ny, nx)
        return {
            "full": _stats(diff[valid]),
            "interior": _stats(diff[valid & interior2]),
            "band": _stats(diff[valid & band2]),
        }
    nz, ny, nx = diff.shape
    interior2, band2 = _region_masks(ny, nx)
    interior = np.broadcast_to(interior2[None], diff.shape)
    band = np.broadcast_to(band2[None], diff.shape)
    return {
        "full": _stats(diff[valid]),
        "interior": _stats(diff[valid & interior]),
        "band": _stats(diff[valid & band]),
    }


# --------------------------------------------------------------------------
# JAX-side capture of the WRF-comparable observables from a State.
# --------------------------------------------------------------------------

def observe_state(state: Any, namelist: Any) -> dict[str, np.ndarray]:
    """Extract the fields the WRF HPG dump records, as the next stage sees them."""

    from gpuwrf.dynamics.core import rk_addtend_dry as rk
    from gpuwrf.dynamics.acoustic_wrf import moisture_coupling_factors

    metrics = namelist.metrics
    ph_abs, p_abs, al, alt, php = rk._absolute_diagnostics(
        state, metrics, hypsometric_opt=int(namelist.hypsometric_opt)
    )
    mu_total = np.asarray(state.mu_total, dtype=np.float64)
    muu = 0.5 * sum(np.asarray(p, dtype=np.float64) for p in rk._x_face_pair_2d(state.mu_total))
    muv = 0.5 * sum(np.asarray(p, dtype=np.float64) for p in rk._y_face_pair_2d(state.mu_total))
    cqu, cqv = moisture_coupling_factors(state)
    return {
        "mu": np.asarray(state.mu_perturbation, dtype=np.float64),
        "p": np.asarray(state.p_perturbation, dtype=np.float64),
        "ph": np.asarray(state.ph_perturbation, dtype=np.float64),
        "al": np.asarray(al, dtype=np.float64),
        "alt": np.asarray(alt, dtype=np.float64),
        "php": np.asarray(php, dtype=np.float64),
        "muu": muu,
        "muv": muv,
        "cqu": np.asarray(cqu, dtype=np.float64),
        "cqv": np.asarray(cqv, dtype=np.float64),
        "mu_total_mean": np.asarray([mu_total.mean()]),
        "u": np.asarray(state.u, dtype=np.float64),
        "v": np.asarray(state.v, dtype=np.float64),
        "w": np.asarray(state.w, dtype=np.float64),
        "theta": np.asarray(state.theta, dtype=np.float64),
    }


def _ring_profile(diff: np.ndarray, valid: np.ndarray, max_ring: int = 12) -> dict[str, float]:
    """Increment-error rmse per lateral ring (ring = min distance to any edge)."""

    if diff.ndim == 2:
        ny, nx = diff.shape
    else:
        _, ny, nx = diff.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    ring2 = np.minimum(np.minimum(ii, nx - 1 - ii), np.minimum(jj, ny - 1 - jj))
    ring = ring2 if diff.ndim == 2 else np.broadcast_to(ring2[None], diff.shape)
    out: dict[str, float] = {}
    for r in range(max_ring):
        sel = valid & (ring == r)
        if sel.any():
            vals = diff[sel]
            out[str(r)] = float(np.sqrt(np.nanmean(vals * vals)))
    return out


def compare_capture_to_wrf(
    capture: Mapping[str, np.ndarray],
    base_capture: Mapping[str, np.ndarray],
    wrf_call: Mapping[str, Any],
    wrf_base: Mapping[str, Any],
) -> dict[str, Any]:
    """Per-field state + increment-vs-base diffs (JAX capture vs WRF call)."""

    nz = wrf_call["dims"]["nz"]
    ny = wrf_call["dims"]["ny"]
    nx = wrf_call["dims"]["nx"]
    w3, w2 = wrf_call["fields3"], wrf_call["fields2"]
    b3, b2 = wrf_base["fields3"], wrf_base["fields2"]
    out: dict[str, Any] = {}

    def cmp3(name: str, jax_arr: np.ndarray, kz: int, jy: int, ix: int) -> None:
        wrf_arr = w3[name][:kz, :jy, :ix]
        wrf_b = b3[name][:kz, :jy, :ix]
        jax_arr = jax_arr[:kz, :jy, :ix]
        jax_b = np.asarray(base_capture[name])[:kz, :jy, :ix]
        valid = np.isfinite(wrf_arr) & (wrf_arr > -9.0e33) & np.isfinite(wrf_b) & (wrf_b > -9.0e33)
        incr = (jax_arr - jax_b) - (wrf_arr - wrf_b)
        out[name] = {
            "state_err": _split_stats(jax_arr - wrf_arr, valid),
            "incr_err": _split_stats(incr, valid),
            "wrf_incr": _split_stats(wrf_arr - wrf_b, valid),
            "incr_err_ring_rmse": _ring_profile(incr, valid),
        }

    def cmp2(name: str, jax_arr: np.ndarray, jy: int, ix: int) -> None:
        wrf_arr = w2[name][:jy, :ix]
        wrf_b = b2[name][:jy, :ix]
        jax_arr = np.asarray(jax_arr)[:jy, :ix]
        jax_b = np.asarray(base_capture[name])[:jy, :ix]
        valid = np.isfinite(wrf_arr) & np.isfinite(wrf_b)
        incr = (jax_arr - jax_b) - (wrf_arr - wrf_b)
        out[name] = {
            "state_err": _split_stats(jax_arr - wrf_arr, valid),
            "incr_err": _split_stats(incr, valid),
            "wrf_incr": _split_stats(wrf_arr - wrf_b, valid),
            "incr_err_ring_rmse": _ring_profile(incr, valid),
        }

    cmp2("mu", capture["mu"], ny, nx)
    cmp3("p", capture["p"], nz, ny, nx)
    cmp3("ph", capture["ph"], nz + 1, ny, nx)
    cmp3("al", capture["al"], nz, ny, nx)
    cmp3("alt", capture["alt"], nz, ny, nx)
    cmp3("php", capture["php"], nz, ny, nx)
    cmp2("muu", capture["muu"], ny, nx + 1)
    cmp2("muv", capture["muv"], ny + 1, nx)
    cmp3("cqu", capture["cqu"], nz, ny, nx + 1)
    cmp3("cqv", capture["cqv"], nz, ny + 1, nx)
    return out


# --------------------------------------------------------------------------
# Eager replica of operational_mode._physics_boundary_step with captures.
# Mirrors operational_mode.py:2300-2430 (_rk_scan_step / advance_stage) and
# :3304-3370 (_physics_boundary_step_with_limiter_diagnostics) verbatim.
# --------------------------------------------------------------------------

def _advance_stage_replica(
    om: Any,
    stage_carry: Any,
    stage: Any,
    namelist: Any,
    physics_tendencies: Any,
    rk1_reference: Any,
    lead_seconds: Any,
    intra: dict[str, Any] | None = None,
) -> Any:
    import jax.numpy as jnp
    from gpuwrf.contracts.halo import apply_halo
    from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
    from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
    from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf

    haloed = apply_halo(stage_carry.state, halo_spec(namelist.grid))
    tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
    stage_velocities = (
        om._stage_transport_velocities(haloed, namelist)
        if bool(namelist.use_flux_advection)
        else None
    )
    # v0.14 stage3/wrapper cadence: mirror production's step-constant
    # relax_bdy_dry bundle (None unless namelist.specified_bdy_cadence).
    bdy_relax = om._specified_bdy_relax(rk1_reference, namelist, lead_seconds)
    tendencies = om._augment_large_step_tendencies(
        haloed,
        tendencies,
        namelist,
        rk_step=int(stage.rk_step),
        physics_tendencies=physics_tendencies,
        step_origin=rk1_reference,
        transport_velocities=stage_velocities,
        bdy_relax=bdy_relax,
    )
    moisture_advected = (
        bool(namelist.use_flux_advection) and int(namelist.moist_adv_opt) != 0
    )
    q_tendencies = (
        om._moisture_coupled_tendencies(
            haloed,
            namelist,
            rk_step=int(stage.rk_step),
            step_origin=rk1_reference,
            transport_velocities=stage_velocities,
        )
        if moisture_advected
        else None
    )
    candidate = apply_halo(stage_carry.state, halo_spec(namelist.grid))
    # Mirrors operational_mode.advance_stage: WRF re-diagnoses the stage omega
    # (rk_step_prep calc_ww_cp) every stage; for specified/nested real domains
    # the edge-faithful construction replaces the periodic-wrap rom.
    from gpuwrf.dynamics.flux_advection import stage_omega_specified

    _per_x, _spec, _nest = om._acoustic_lateral_bc_flags(namelist)
    if stage_velocities is not None and (_spec or _nest):
        ww_stage = stage_omega_specified(
            haloed.u,
            haloed.v,
            haloed.mu_total,
            c1h=namelist.metrics.c1h,
            c2h=namelist.metrics.c2h,
            dnw=namelist.metrics.dnw,
            rdx=1.0 / float(namelist.grid.projection.dx_m),
            rdy=1.0 / float(namelist.grid.projection.dy_m),
            msfuy=namelist.metrics.msfuy,
            msfvx=namelist.metrics.msfvx,
            msftx=namelist.metrics.msftx,
        )
    elif stage_velocities is not None:
        ww_stage = stage_velocities.rom
    else:
        ww_stage = stage_carry.ww
    prep = small_step_prep_wrf(
        candidate,
        int(stage.rk_step),
        float(stage.dt_rk),
        metrics=namelist.metrics,
        reference_state=rk1_reference,
        ww=ww_stage,
    )
    pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
    if intra is not None:
        intra["tend"] = {
            name: np.asarray(getattr(tendencies, name), dtype=np.float64)
            for name in ("u", "v", "w", "theta", "mu")
        }
        intra["prep"] = {
            "alt": np.asarray(prep.alt, dtype=np.float64),
            "al_full": np.asarray(prep.al, dtype=np.float64),
            "c2a": np.asarray(prep.c2a, dtype=np.float64),
            "mu_work": np.asarray(prep.mu_work, dtype=np.float64),
            "muts": np.asarray(prep.muts, dtype=np.float64),
            "php": np.asarray(prep.php, dtype=np.float64),
        }
    stage_carry = om._acoustic_scan(
        stage_carry.replace(state=candidate),
        namelist,
        stage=stage,
        prep=prep,
        pressure=pressure,
        tendencies=tendencies,
        lead_seconds=lead_seconds,
        bdy_relax=bdy_relax,
    )
    if moisture_advected:
        stage_carry = stage_carry.replace(
            state=om._apply_moisture_large_step(
                stage_carry.state,
                rk1_reference,
                q_tendencies=q_tendencies,
                dt_rk=float(stage.dt_rk),
                metrics=namelist.metrics,
            )
        )
    stage_carry = stage_carry.replace(state=apply_halo(stage_carry.state, halo_spec(namelist.grid)))
    return stage_carry


def _physics_boundary_step_replica(
    om: Any,
    carry: Any,
    namelist: Any,
    step_index: int,
    *,
    run_radiation: bool,
    capture: dict[str, Any],
    capture_intra: bool = False,
) -> Any:
    import jax.numpy as jnp
    from gpuwrf.contracts.halo import apply_halo
    from gpuwrf.dynamics.advection import halo_spec

    physical_origin = carry.state
    lead_seconds = jnp.asarray(step_index, dtype=jnp.int32).astype(jnp.float64) * float(namelist.dt_s)
    physics_forcing = om._physics_step_forcing(
        carry,
        namelist,
        lead_seconds,
        run_radiation=run_radiation,
        first_timestep=jnp.equal(jnp.asarray(step_index, dtype=jnp.int32), 1),
    )
    carry = physics_forcing.carry

    # _rk_scan_step body with per-stage capture
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    rk1_reference = origin
    dt = float(namelist.dt_s)
    ns = int(namelist.acoustic_substeps)
    stages = (
        om._RKStageDescriptor(1, dt / 3.0, dt / 3.0, 1),
        om._RKStageDescriptor(2, 0.5 * dt, dt / float(ns), max(1, ns // 2)),
        om._RKStageDescriptor(3, dt, dt / float(ns), ns),
    )
    carry = carry.replace(state=origin)
    for stage in stages:
        intra = {} if capture_intra else None
        carry = _advance_stage_replica(
            om, carry, stage, namelist, physics_forcing.dry_tendencies, rk1_reference, lead_seconds, intra=intra
        )
        capture[f"stage{stage.rk_step}"] = observe_state(carry.state, namelist)
        if intra is not None:
            capture[f"stage{stage.rk_step}_intra"] = intra

    # _physics_boundary_step wrapper (post-RK)
    next_state = carry.state
    if bool(physics_forcing.enabled):
        next_state = om._apply_physics_non_dry_updates(next_state, physical_origin, physics_forcing.state)
        carry = carry.replace(state=next_state)
    if not bool(namelist.disable_guards):
        next_state, _diag = om._limit_guarded_dynamics_state_with_diagnostics(next_state, physical_origin)
        next_state = next_state.replace(
            qv=om._valid_mixing_ratio(next_state.qv, physical_origin.qv),
            qc=om._valid_mixing_ratio(next_state.qc, physical_origin.qc),
            qr=om._valid_mixing_ratio(next_state.qr, physical_origin.qr),
            qi=om._valid_mixing_ratio(next_state.qi, physical_origin.qi),
            qs=om._valid_mixing_ratio(next_state.qs, physical_origin.qs),
            qg=om._valid_mixing_ratio(next_state.qg, physical_origin.qg),
        )
    if bool(namelist.run_boundary):
        bounded = om.apply_lateral_boundaries(
            next_state, lead_seconds, float(namelist.dt_s), namelist.boundary_config, namelist.metrics,
            dry_spec_only=om._specified_bdy_cadence_active(namelist),
        )
        if bool(namelist.disable_guards):
            next_state = bounded
        else:
            next_state = bounded.replace(
                u=om._finite_or_origin(bounded.u, physical_origin.u),
                v=om._finite_or_origin(bounded.v, physical_origin.v),
                w=om._finite_or_origin(bounded.w, physical_origin.w),
                theta=om._finite_or_origin(bounded.theta, physical_origin.theta),
                qv=om._valid_mixing_ratio(bounded.qv, physical_origin.qv),
                p=om._finite_or_origin(bounded.p, physical_origin.p),
                ph=om._finite_or_origin(bounded.ph, physical_origin.ph),
                p_total=om._finite_or_origin(bounded.p_total, physical_origin.p_total),
                ph_total=om._finite_or_origin(bounded.ph_total, physical_origin.ph_total),
                p_perturbation=om._finite_or_origin(bounded.p_perturbation, physical_origin.p_perturbation),
                ph_perturbation=om._finite_or_origin(bounded.ph_perturbation, physical_origin.ph_perturbation),
            )
            next_state = om._limit_guarded_mass_state(next_state, physical_origin)
    next_state = om._enforce_operational_precision(next_state, force_fp64=bool(namelist.force_fp64))
    final_carry = om._maybe_exchange_sharded_carry_halos(carry.replace(state=next_state))
    capture["final"] = observe_state(final_carry.state, namelist)
    return final_carry


def _state_max_diffs(left: Any, right: Any) -> dict[str, float]:
    import jax.numpy as jnp

    out = {}
    for name in ("u", "v", "w", "theta", "mu_perturbation", "p_perturbation", "ph_perturbation", "qv"):
        a = getattr(left, name)
        b = getattr(right, name)
        out[name] = float(jnp.max(jnp.abs(a - b)))
    return out


def run_stage_compare(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import gpuwrf.runtime.operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry

    t_start = time.perf_counter()
    case, state0, run_dir = hpg._build_state(NATIVE_ROOT)
    namelist = case.namelist
    if int(args.substeps) > 0:
        namelist = dataclasses.replace(namelist, acoustic_substeps=int(args.substeps))
    if float(args.dt) > 0:
        # WRF truth ran dt=18 s; the JAX pipeline default for this case is 10 s.
        # Window-matched stage comparison vs calls 21602-21606 requires dt=18.
        namelist = dataclasses.replace(namelist, dt_s=float(args.dt))
    if args.no_physics:
        namelist = dataclasses.replace(namelist, run_physics=False)
    run_radiation = not bool(args.no_radiation) and bool(namelist.run_physics)

    print(
        f"[acoustic] tag={args.tag} substeps={namelist.acoustic_substeps} "
        f"run_physics={namelist.run_physics} run_radiation={run_radiation} "
        f"hypsometric_opt={namelist.hypsometric_opt} backend={jax.default_backend()}",
        flush=True,
    )

    wrf_calls = {ncall: hpg.assemble_wrf_call(ncall) for ncall in (21601, 21602, 21603, 21604, 21605, 21606)}
    wrf_base = wrf_calls[21601]

    carry0 = initial_operational_carry(state0)
    base_capture = observe_state(state0, namelist)

    # base sanity: state vs call 21601 (must be ~bit-zero on mu/p/ph)
    base_cmp = compare_capture_to_wrf(base_capture, base_capture, wrf_base, wrf_base)
    print(
        "[acoustic] base identity (vs 21601): "
        + " ".join(f"{k}={base_cmp[k]['state_err']['full'].get('max_abs', float('nan')):.3e}" for k in ("mu", "p", "ph")),
        flush=True,
    )

    captures: dict[int, dict[str, Any]] = {}
    carry = carry0
    carry_after_step1 = None
    for step in (1, 2):
        cap: dict[str, Any] = {}
        carry = _physics_boundary_step_replica(
            om, carry, namelist, step, run_radiation=run_radiation,
            capture=cap, capture_intra=bool(args.capture_intra),
        )
        jax.block_until_ready(carry.state.u)
        captures[step] = cap
        if step == 1:
            carry_after_step1 = carry
        print(f"[acoustic] step {step} replica done ({time.perf_counter() - t_start:.1f}s)", flush=True)
        if step >= int(args.steps):
            break

    # replica validity: jitted production step on the same inputs
    validity = None
    if not args.skip_validity:
        def _one(carry_in, namelist_in, idx):
            return om._physics_boundary_step(carry_in, namelist_in, idx, run_radiation=run_radiation, debug=False)

        jit_step = jax.jit(_one)
        jit_carry = jit_step(carry0, namelist, jnp.asarray(1, dtype=jnp.int32))
        jax.block_until_ready(jit_carry.state.u)
        validity = _state_max_diffs(jit_carry.state, carry_after_step1.state)
        print(f"[acoustic] replica-vs-jit max diffs: {validity}", flush=True)

    # comparisons
    mapping = {
        "step1_stage1_vs_21602": (captures.get(1, {}).get("stage1"), 21602),
        "step1_stage2_vs_21603": (captures.get(1, {}).get("stage2"), 21603),
        "step1_stage3_raw": (captures.get(1, {}).get("stage3"), None),
        "step1_final_vs_21604": (captures.get(1, {}).get("final"), 21604),
        "step2_stage1_vs_21605": (captures.get(2, {}).get("stage1"), 21605),
        "step2_stage2_vs_21606": (captures.get(2, {}).get("stage2"), 21606),
    }
    comparisons: dict[str, Any] = {}
    for label, (cap, ncall) in mapping.items():
        if cap is None:
            continue
        if ncall is None:
            # stage-3 capture has no separate WRF dump (21604 is post end-of-step);
            # compare against 21604 anyway, labeled as pre-wrapper.
            comparisons[label + "_vs_21604_prewrapper"] = compare_capture_to_wrf(
                cap, base_capture, wrf_calls[21604], wrf_base
            )
            continue
        wrf_base_for_step = wrf_calls[21604] if label.startswith("step2") else wrf_base
        jax_base_for_step = captures[1]["final"] if label.startswith("step2") else base_capture
        comparisons[label] = compare_capture_to_wrf(cap, jax_base_for_step, wrf_calls[ncall], wrf_base_for_step)

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "tag": str(args.tag),
        "config": {
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "run_physics": bool(namelist.run_physics),
            "run_radiation": bool(run_radiation),
            "hypsometric_opt": int(namelist.hypsometric_opt),
            "epssm": float(namelist.epssm),
            "dt_s": float(namelist.dt_s),
            "steps": int(args.steps),
        },
        "base_identity_vs_21601": {
            k: base_cmp[k]["state_err"]["full"] for k in ("mu", "p", "ph", "al", "alt", "muu", "muv", "cqu", "cqv")
        },
        "replica_vs_jit_max_diffs": validity,
        "comparisons": comparisons,
        "wall_s": float(time.perf_counter() - t_start),
    }

    existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    existing.setdefault("stage_compare", {})[str(args.tag)] = result
    hpg.write_json(OUT_JSON, existing)
    print(f"wrote stage_compare[{args.tag}] to {OUT_JSON}", flush=True)
    return result


def run_forecast_variant(args: argparse.Namespace) -> None:
    """1-3h GPU forecast from the h36 re-init with the acoustic-substep fixes.

    Identical config to the prior sprints' hourly-gate runs (PROBE_ROOT run_h36,
    production pipeline defaults: dt/substeps from the case), so the budget
    collapse is comparable against ec4d6769 (gpu_output) and 3d0b439c
    (gpu_output_hpg_native_face_fix).
    """

    from gpuwrf.integration import daily_pipeline as dp

    out_dir = PROBE_ROOT / args.outdir if args.outdir else FIX_GPU
    extra: dict[str, Any] = {}
    if float(args.forecast_dt) > 0:
        extra["dt_s"] = float(args.forecast_dt)
    if int(args.forecast_substeps) > 0:
        extra["acoustic_substeps"] = int(args.forecast_substeps)
    config = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=int(args.hours),
        output_dir=out_dir,
        proof_dir=PROBE_ROOT / "proofs_acoustic_substep_fix",
        run_root=PROBE_ROOT,
        domain="d01",
        **extra,
    )
    result = dp._run_forecast_sequence(config, output_dir=config.output_dir)
    print(f"forecast result: status={result.status} hours={result.hours} output_dir={result.output_dir}", flush=True)


def hourly_gate() -> dict[str, Any]:
    CPU = hpg.CPU
    baselines = {
        "old_ec4d6769": hpg.BASELINE_GPU,
        "hypso_3d0b439c": hpg.FIX_GPU,
        # fixes 1-5 (constants/g/surface-w BC/w_damp/stage omega) without the
        # diff_opt=1/km_opt=4 smag2d threading:
        "acoustic_fix_nokm": PROBE_ROOT / "gpu_output_acoustic_substep_fix_nokm",
        "acoustic_fix": FIX_GPU,
        # all fixes + WRF-cadence-matched dt=18 s / 4 sound steps (the CPU truth
        # cadence; the pipeline default is dt=10 / 10 substeps):
        "acoustic_fix_dt18": PROBE_ROOT / "gpu_output_acoustic_substep_fix_dt18",
    }
    out: dict[str, Any] = {"available": hpg.fn(FIX_GPU, 37).exists(), "fixed_output": str(FIX_GPU)}
    if not out["available"]:
        return out
    cpu_budget = hpg.budget_between(CPU, 36, CPU, 37, depth=8)
    out["cpu_budget_h36_h37_depth8"] = cpu_budget
    excesses: dict[str, float] = {}
    for name, base in baselines.items():
        if not hpg.fn(base, 37).exists():
            out[f"{name}_budget"] = {"available": False, "path": str(base)}
            continue
        budget = hpg.budget_between(CPU, 36, base, 37, depth=8)
        excess = budget["net_influx_pa_per_cell_h"] - cpu_budget["net_influx_pa_per_cell_h"]
        out[f"{name}_budget"] = budget
        excesses[name] = float(excess)
    out["excess_outflux_pa_per_cell_h"] = excesses
    if "old_ec4d6769" in excesses and "acoustic_fix" in excesses:
        out["collapse_fraction_vs_old"] = float(
            1.0 - abs(excesses["acoustic_fix"]) / max(abs(excesses["old_ec4d6769"]), 1.0e-12)
        )
    if "hypso_3d0b439c" in excesses and "acoustic_fix" in excesses:
        out["collapse_fraction_vs_hypso"] = float(
            1.0 - abs(excesses["acoustic_fix"]) / max(abs(excesses["hypso_3d0b439c"]), 1.0e-12)
        )
    out["metrics_h37"] = hpg.field_metrics(FIX_GPU, 37)
    out["baseline_metrics_h37"] = hpg.field_metrics(hpg.BASELINE_GPU, 37)
    if hpg.fn(FIX_GPU, 38).exists():
        cpu_b38 = hpg.budget_between(CPU, 36, CPU, 38, depth=8)
        fix_b38 = hpg.budget_between(CPU, 36, FIX_GPU, 38, depth=8)
        out["h36_h38"] = {
            "cpu": cpu_b38,
            "acoustic_fix": fix_b38,
            "fixed_excess": float(fix_b38["net_influx_pa_per_cell_h"] - cpu_b38["net_influx_pa_per_cell_h"]),
        }
        out["metrics_h38"] = hpg.field_metrics(FIX_GPU, 38)
    return out


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    existing["hourly_gate"] = hourly_gate()
    hpg.write_json(OUT_JSON, existing)
    print(json.dumps({
        "hourly_gate": {
            k: existing["hourly_gate"].get(k)
            for k in ("excess_outflux_pa_per_cell_h", "collapse_fraction_vs_old", "collapse_fraction_vs_hypso", "available")
        },
        "stage_compare_tags": sorted((existing.get("stage_compare") or {}).keys()),
    }, indent=2, default=str))
    return existing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-compare", action="store_true")
    parser.add_argument("--forecast-variant", action="store_true")
    parser.add_argument("--hours", type=int, default=2)
    parser.add_argument("--outdir", default="", help="output dir name under PROBE_ROOT; default canonical")
    parser.add_argument("--forecast-dt", type=float, default=0.0, help="override pipeline dt_s; 0 keeps default")
    parser.add_argument("--forecast-substeps", type=int, default=0, help="override pipeline acoustic_substeps; 0 keeps default")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--substeps", type=int, default=4, help="acoustic substeps; 0 keeps pipeline default")
    parser.add_argument("--dt", type=float, default=18.0, help="model dt seconds; 0 keeps pipeline default")
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--no-radiation", action="store_true")
    parser.add_argument("--capture-intra", action="store_true")
    parser.add_argument("--skip-validity", action="store_true")
    args = parser.parse_args()

    if args.stage_compare:
        run_stage_compare(args)
    elif args.forecast_variant:
        run_forecast_variant(args)
    else:
        analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
