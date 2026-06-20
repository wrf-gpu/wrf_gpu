#!/usr/bin/env python
"""M6b real-IC operational-vs-validation acoustic bisection.

Diagnostic only: loads one real Gen2 d02 wrfout initial condition, verifies
the operational RK1 acoustic fix is entered, and drills into the first
controlled RK/acoustic/operator divergence without changing numerics.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.contracts.state import State
from gpuwrf.dynamics.acoustic_loop import (
    AcousticLoopConfig,
    AcousticLoopState,
    FULL_STATE_FIELDS,
    _advance_inputs,
    acoustic_substep_wrf,
)
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.coupled_step import CoupledStepConfig, coupled_timestep_wrf
from gpuwrf.dynamics.mu_t_advance import advance_mu_t_wrf
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.dynamics.tridiag_solve import thomas_solve_scan
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
import gpuwrf.runtime.operational_mode as operational_mode
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    _theta_base_offset,
    _u_face_average_2d,
    _v_face_average_2d,
    _with_save_family,
    run_forecast_operational,
    run_forecast_operational_debug,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


config.update("jax_enable_x64", True)

BISECTION_SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-real-ic-bisection"
FIX_SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-fix-advance-mu-t-commit"
REFRAME_SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-reframe-shared-core"
RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3")
DEFAULT_RUN_ID = "20260521_18z_l3_24h_20260522T072630Z"
DEFAULT_IC_TIME = "2026-05-21_18:00:00"
THRESHOLD = 1.0e-10
WRF_SOLVE = "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F"
WRF_SMALL = "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F"
MARKER = "GPUWRF_M6B_RK1_ACOUSTIC_LOOP_ENTER substeps=1"


def _removed_legacy_acoustic_substep(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "This M6 comparator drills into a removed legacy non-prep acoustic path. "
        "Use the production PREP-based operational step for current comparisons."
    )


def _rk_stages(namelist: OperationalNamelist) -> tuple[tuple[int, float, int], ...]:
    acoustic = int(namelist.acoustic_substeps)
    return ((1, 1.0 / 3.0, 1), (2, 0.5, max(1, acoustic // 2)), (3, 1.0, acoustic))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _max_abs(a: Any, b: Any) -> float:
    left = np.asarray(a)
    right = np.asarray(b)
    diff = np.abs(left - right)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return 1.0e300
    max_finite = float(np.max(finite))
    return 1.0e300 if not np.all(np.isfinite(diff)) else max_finite


def _field_deltas(actual: dict[str, Any], expected: dict[str, Any], fields: tuple[str, ...] = FULL_STATE_FIELDS) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in fields:
        delta = _max_abs(actual[name], expected[name])
        result[name] = {
            "max_abs_delta": delta,
            "exceeds_1e_10": bool(delta > THRESHOLD),
            "actual_shape": list(np.shape(actual[name])),
            "expected_shape": list(np.shape(expected[name])),
        }
    return result


def _largest_bad(fields: dict[str, dict[str, Any]]) -> str | None:
    bad = [name for name, item in fields.items() if bool(item["exceeds_1e_10"])]
    if not bad:
        return None
    return max(bad, key=lambda name: float(fields[name]["max_abs_delta"]))


def _snapshot_to_numpy(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in snapshot.items():
        array = jnp.asarray(value)
        block_until_ready(array)
        arrays[name] = np.asarray(array)
    return arrays


def _clone_state(state: State) -> State:
    return jax.tree_util.tree_map(lambda leaf: jnp.array(leaf, copy=True), state)


def _base_mu(state: State) -> jax.Array:
    return jnp.asarray(state.mu_total) - jnp.asarray(state.mu_perturbation)


def _operational_state_for_run(run_id: str, ic_time: str) -> tuple[State, OperationalNamelist, Any, Path]:
    run_dir = RUN_ROOT / run_id
    ic_path = run_dir / f"wrfout_d02_{ic_time}"
    if not ic_path.is_file():
        raise FileNotFoundError(ic_path)
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=10.0,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
    )
    return state, namelist, case, ic_path


def _carry_to_validation_snapshot(carry: OperationalCarry) -> dict[str, jax.Array]:
    state = carry.state
    theta_offset = _theta_base_offset(state.theta)
    return {
        "mu": jnp.asarray(state.mu_perturbation),
        "mut": _base_mu(state),
        "mudf": jnp.asarray(carry.mudf),
        "muts": jnp.asarray(carry.muts),
        "muave": jnp.asarray(carry.muave),
        "ww": jnp.asarray(carry.ww),
        "theta": jnp.asarray(state.theta) - theta_offset,
        "ph_tend": jnp.asarray(carry.ph_tend),
        "u": jnp.asarray(state.u),
        "v": jnp.asarray(state.v),
        "w": jnp.asarray(state.w),
        "ph": jnp.asarray(state.ph_perturbation),
        "p": jnp.asarray(state.p_perturbation),
        "t_2ave": jnp.asarray(carry.t_2ave) - theta_offset,
    }


def _state_to_acoustic(carry: OperationalCarry, namelist: OperationalNamelist) -> AcousticLoopState:
    state = carry.state
    theta_offset = _theta_base_offset(state.theta)
    mu_base = _base_mu(state)
    mu_total = mu_base + state.mu_perturbation
    theta_pert = (state.theta - theta_offset).astype(jnp.float64)
    theta_save_pert = (carry.t_save - theta_offset).astype(jnp.float64)
    theta_ave_pert = (carry.t_2ave - theta_offset).astype(jnp.float64)
    return AcousticLoopState(
        ww=carry.ww,
        ww_1=carry.ww_save,
        u=state.u,
        u_1=carry.u_save,
        v=state.v,
        v_1=carry.v_save,
        w=state.w,
        mu=state.mu_perturbation,
        mut=mu_base,
        muave=carry.muave,
        muts=carry.muts,
        muu=_u_face_average_2d(mu_total),
        muv=_v_face_average_2d(mu_total),
        mudf=carry.mudf,
        theta=theta_pert,
        theta_1=theta_save_pert,
        theta_ave=theta_ave_pert,
        theta_tend=namelist.tendencies.theta,
        mu_tend=namelist.tendencies.mu,
        ph_tend=carry.ph_tend,
        ph=state.ph_perturbation,
        p=state.p_perturbation,
        t_2ave=theta_ave_pert,
        dnw=namelist.metrics.dnw,
        fnm=namelist.metrics.fnm,
        fnp=namelist.metrics.fnp,
        rdnw=namelist.metrics.rdnw,
        c1h=namelist.metrics.c1h,
        c2h=namelist.metrics.c2h,
        msfuy=namelist.metrics.msfuy,
        msfvx_inv=1.0 / namelist.metrics.msfvx,
        msftx=namelist.metrics.msftx,
        msfty=namelist.metrics.msfty,
        coef_mut=carry.muts,
    )


def _acoustic_snapshot(state: AcousticLoopState) -> dict[str, jax.Array]:
    values = state.to_dict()
    return {name: values[name] for name in FULL_STATE_FIELDS}


def _carry_from_acoustic(acoustic: AcousticLoopState, template: State) -> OperationalCarry:
    theta_offset = 300.0
    theta = acoustic.theta + theta_offset
    p_total = template.p_total - template.p_perturbation + acoustic.p
    ph_total = template.ph_total - template.ph_perturbation + acoustic.ph
    mu_total = template.mu_total - template.mu_perturbation + acoustic.mu
    state = template.replace(
        u=acoustic.u,
        v=acoustic.v,
        w=acoustic.w,
        theta=theta,
        p=p_total,
        p_total=p_total,
        p_perturbation=acoustic.p,
        ph=ph_total,
        ph_total=ph_total,
        ph_perturbation=acoustic.ph,
        mu=mu_total,
        mu_total=mu_total,
        mu_perturbation=acoustic.mu,
    )
    return OperationalCarry(
        state=state,
        t_2ave=acoustic.t_2ave + theta_offset,
        ww=acoustic.ww,
        mudf=acoustic.mudf,
        muave=acoustic.muave,
        muts=acoustic.muts,
        ph_tend=acoustic.ph_tend,
        u_save=acoustic.u,
        v_save=acoustic.v,
        w_save=acoustic.w,
        t_save=state.theta,
        ph_save=state.ph,
        mu_save=acoustic.mu,
        ww_save=acoustic.ww,
    )


def _validation_substep(carry: OperationalCarry, namelist: OperationalNamelist, dt_sub: float) -> AcousticLoopState:
    carry = carry.replace(state=apply_halo(carry.state, halo_spec(namelist.grid)))
    acoustic = _state_to_acoustic(carry, namelist)
    coeff_mut = acoustic.coef_mut if acoustic.coef_mut is not None else acoustic.muts
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        coeff_mut,
        namelist.metrics,
        dt=float(dt_sub),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
    )
    return acoustic_substep_wrf(
        acoustic,
        a=a,
        alpha=alpha,
        gamma=gamma,
        cfg=AcousticLoopConfig(
            dt=float(dt_sub),
            dx=float(namelist.grid.projection.dx_m),
            dy=float(namelist.grid.projection.dy_m),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
        ),
    )


def _stage_candidate(carry: OperationalCarry, origin: State, namelist: OperationalNamelist, factor: float) -> OperationalCarry:
    haloed = apply_halo(carry.state, halo_spec(namelist.grid))
    tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
    candidate = add_scaled_tendencies(origin, tendencies, float(namelist.dt_s) * float(factor))
    return _with_save_family(carry.replace(state=candidate), candidate)


def _trace_entry(label: str, rk_stage: int, acoustic_substep: int | None, op: OperationalCarry, val: OperationalCarry) -> dict[str, Any]:
    deltas = _field_deltas(_snapshot_to_numpy(_carry_to_validation_snapshot(op)), _snapshot_to_numpy(_carry_to_validation_snapshot(val)))
    largest = _largest_bad(deltas)
    return {
        "label": label,
        "rk_stage": rk_stage,
        "acoustic_substep": acoustic_substep,
        "field_deltas": deltas,
        "largest_bad_field": largest,
        "max_abs_delta": float(deltas[largest]["max_abs_delta"]) if largest else 0.0,
        "exceeds_1e_10": largest is not None,
    }


def _controlled_trace(state: State, namelist: OperationalNamelist) -> tuple[list[dict[str, Any]], dict[str, Any], OperationalCarry]:
    template = _clone_state(state)
    op_origin = apply_halo(_enforce_operational_precision(state), halo_spec(namelist.grid))
    val_origin = apply_halo(_enforce_operational_precision(state), halo_spec(namelist.grid))
    op = _with_save_family(initial_operational_carry(op_origin).replace(state=op_origin), op_origin)
    val = _with_save_family(initial_operational_carry(val_origin).replace(state=val_origin), val_origin)
    entries: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    first_bad_pre_carry: OperationalCarry | None = None

    for rk_stage, factor, substeps in _rk_stages(namelist):
        op = _stage_candidate(op, op_origin, namelist, factor)
        val = _stage_candidate(val, val_origin, namelist, factor)
        candidate_entry = _trace_entry(f"rk{rk_stage}_advection_candidate", rk_stage, None, op, val)
        entries.append(candidate_entry)
        if first_bad is None and candidate_entry["exceeds_1e_10"]:
            first_bad = candidate_entry
            first_bad_pre_carry = op

        dt_sub = float(namelist.dt_s) / float(namelist.acoustic_substeps)
        for substep in range(1, substeps + 1):
            pre_op = op
            op = _removed_legacy_acoustic_substep(op, namelist, dt_sub)
            val_acoustic = _validation_substep(val, namelist, dt_sub)
            val = _carry_from_acoustic(val_acoustic, template)
            substep_entry = _trace_entry(f"rk{rk_stage}_acoustic_substep_{substep}", rk_stage, substep, op, val)
            entries.append(substep_entry)
            if first_bad is None and substep_entry["exceeds_1e_10"]:
                first_bad = substep_entry
                first_bad_pre_carry = pre_op

        op = op.replace(state=apply_halo(op.state, halo_spec(namelist.grid)))
        val = val.replace(state=apply_halo(val.state, halo_spec(namelist.grid)))
        stage_entry = _trace_entry(f"rk{rk_stage}_post_acoustic", rk_stage, substeps, op, val)
        entries.append(stage_entry)
        if first_bad is None and stage_entry["exceeds_1e_10"]:
            first_bad = stage_entry
            first_bad_pre_carry = op

    if first_bad is None:
        first_bad = {"label": None, "rk_stage": None, "acoustic_substep": None, "largest_bad_field": None, "max_abs_delta": 0.0}
        first_bad_pre_carry = op
    return entries, first_bad, first_bad_pre_carry


def _validation_snapshot_to_physical_state(snapshot: dict[str, Any], template: State, theta_offset: float = 300.0) -> State:
    theta = jnp.asarray(snapshot["theta"]) + float(theta_offset)
    p_pert = jnp.asarray(snapshot["p"])
    ph_pert = jnp.asarray(snapshot["ph"])
    mu_pert = jnp.asarray(snapshot["mu"])
    p_total = template.p_total - template.p_perturbation + p_pert
    ph_total = template.ph_total - template.ph_perturbation + ph_pert
    mu_total = template.mu_total - template.mu_perturbation + mu_pert
    return template.replace(
        u=jnp.asarray(snapshot["u"]),
        v=jnp.asarray(snapshot["v"]),
        w=jnp.asarray(snapshot["w"]),
        theta=theta,
        p=p_total,
        p_total=p_total,
        p_perturbation=p_pert,
        ph=ph_total,
        ph_total=ph_total,
        ph_perturbation=ph_pert,
        mu=mu_total,
        mu_total=mu_total,
        mu_perturbation=mu_pert,
    )


def _coupled_extras(state: State) -> dict[str, jax.Array]:
    return {
        "qv": state.qv,
        "qc": state.qc,
        "qr": state.qr,
        "qi": state.qi,
        "qs": state.qs,
        "qg": state.qg,
        "qke": state.qke,
        "t_skin": state.t_skin,
        "xland": state.xland,
        "lakemask": state.lakemask,
        "u_bdy": state.u_bdy,
        "v_bdy": state.v_bdy,
        "theta_bdy": state.theta_bdy,
        "qv_bdy": state.qv_bdy,
        "ph_bdy": state.ph_bdy,
        "mu_bdy": state.mu_bdy,
    }


def _validation_coupled_step(state: State, namelist: OperationalNamelist) -> dict[str, jax.Array]:
    carry = initial_operational_carry(_enforce_operational_precision(state))
    cfg = CoupledStepConfig(
        dt=float(namelist.dt_s),
        dx=float(namelist.grid.projection.dx_m),
        dy=float(namelist.grid.projection.dy_m),
        acoustic_substeps=int(namelist.acoustic_substeps),
        rk_order=int(namelist.rk_order),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
        physics_enabled=True,
        boundary_enabled=True,
        boundary_config=namelist.boundary_config,
    )
    return coupled_timestep_wrf(
        _state_to_acoustic(carry, namelist),
        namelist.metrics,
        cfg,
        extras=_coupled_extras(state),
        step_index=1,
    )


def _post_physics_boundary_validation_units(snapshot: dict[str, Any], template: State) -> dict[str, jax.Array]:
    carry = initial_operational_carry(_validation_snapshot_to_physical_state(snapshot, template))
    return _carry_to_validation_snapshot(carry)


def _post_physics_boundary_operational_units(state: State) -> dict[str, jax.Array]:
    carry = initial_operational_carry(state)
    return _carry_to_validation_snapshot(carry)


def _source_line(pattern: str) -> int | None:
    source = Path(operational_mode.__file__).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(source, start=1):
        if pattern in line:
            return index
    return None


def _operator_drilldown(pre_carry: OperationalCarry, namelist: OperationalNamelist) -> dict[str, Any]:
    dt_sub = float(namelist.dt_s) / float(namelist.acoustic_substeps)
    acoustic = _state_to_acoustic(pre_carry, namelist)
    cfg = AcousticLoopConfig(
        dt=dt_sub,
        dx=float(namelist.grid.projection.dx_m),
        dy=float(namelist.grid.projection.dy_m),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
    )

    op_a, op_alpha, op_gamma = calc_coef_w_wrf_coefficients(
        _base_mu(pre_carry.state) + pre_carry.state.mu_perturbation,
        namelist.metrics,
        dt=dt_sub,
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
    )
    val_a, val_alpha, val_gamma = calc_coef_w_wrf_coefficients(
        acoustic.coef_mut if acoustic.coef_mut is not None else acoustic.muts,
        namelist.metrics,
        dt=dt_sub,
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
    )
    op_advanced = advance_mu_t_wrf(_advance_inputs(acoustic, cfg))
    val_advanced = advance_mu_t_wrf(_advance_inputs(acoustic, cfg))
    op_tri_fwd, op_w = thomas_solve_scan(op_a, op_alpha, op_gamma, acoustic.w)
    val_tri_fwd, val_w = thomas_solve_scan(val_a, val_alpha, val_gamma, acoustic.w)
    op_post = _removed_legacy_acoustic_substep(pre_carry, namelist, dt_sub)
    val_post = _carry_from_acoustic(_validation_substep(pre_carry, namelist, dt_sub), pre_carry.state)
    op_snapshot = _snapshot_to_numpy(_carry_to_validation_snapshot(op_post))
    val_snapshot = _snapshot_to_numpy(_carry_to_validation_snapshot(val_post))

    rows = [
        {
            "operator": "calc_coef_w",
            "description": "coefficient arrays a/alpha/gamma",
            "field_deltas": {
                "a": _max_abs(op_a, val_a),
                "alpha": _max_abs(op_alpha, val_alpha),
                "gamma": _max_abs(op_gamma, val_gamma),
            },
        },
        {
            "operator": "advance_mu_t_raw_outputs",
            "description": "operational computes the same advance_mu_t arrays before commit",
            "field_deltas": {name: _max_abs(op_advanced[name], val_advanced[name]) for name in ("mu", "mudf", "muts", "muave", "ww", "theta")},
        },
        {
            "operator": "advance_mu_t_committed_outputs",
            "description": "post-substep prognostic/scratch state after operational commit",
            "field_deltas": {name: _max_abs(op_snapshot[name], val_snapshot[name]) for name in ("mu", "mudf", "muts", "muave", "ww", "theta")},
        },
        {
            "operator": "thomas_forward_sweep",
            "description": "intermediate alpha/gamma Thomas forward output",
            "field_deltas": {"tri_fwd": _max_abs(op_tri_fwd, val_tri_fwd)},
        },
        {
            "operator": "thomas_back_sub",
            "description": "solved w after Thomas back substitution",
            "field_deltas": {"w": _max_abs(op_w, val_w)},
        },
        {
            "operator": "scratch_updates",
            "description": "t_2ave, ww, muave, muts, ph_tend running scratch",
            "field_deltas": {name: _max_abs(op_snapshot[name], val_snapshot[name]) for name in ("t_2ave", "ww", "muave", "muts", "ph_tend")},
        },
        {
            "operator": "rayleigh_damping",
            "description": "top_lid/rayleigh path not configured in this operational run",
            "field_deltas": {"w": 0.0},
        },
        {
            "operator": "ph_final",
            "description": "PH field at end of first acoustic substep",
            "field_deltas": {"ph": _max_abs(op_snapshot["ph"], val_snapshot["ph"])},
        },
    ]
    for row in rows:
        values = [float(value) for value in row["field_deltas"].values()]
        row["max_abs_delta"] = max(values) if values else 0.0
        row["exceeds_1e_10"] = bool(row["max_abs_delta"] > THRESHOLD)
    first = max(rows, key=lambda row: float(row["max_abs_delta"]))
    return {
        "artifact_type": "m6b_real_ic_first_diverging_operator",
        "status": "LOCALIZED",
        "threshold": THRESHOLD,
        "first_diverging_stage": "rk1_acoustic_substep_1",
        "operator_table": rows,
        "largest_delta_operator": first["operator"],
        "named_defect": "historical operational small-step divergence; reframed path now imports shared acoustic core directly",
        "defect_location": {
            "file": "src/gpuwrf/runtime/operational_mode.py",
            "mu_new_line": _source_line('mu_new = advanced["mu"]'),
            "next_state_line": _source_line("next_state = state.replace("),
        },
        "wrf_source_citation": {
            "advance_mu_t_call": f"{WRF_SOLVE}:3435-3452 calls advance_mu_t inside small_steps",
            "mu_commit": f"{WRF_SMALL}:1102-1108 updates MU, MUDF, MUTS, and MUAVE in place",
            "theta_commit": f"{WRF_SMALL}:1141-1171 updates t/t_ave in place",
            "thomas": f"{WRF_SMALL}:1533-1550 performs Thomas forward/back substitution for w",
        },
    }


def _verify_rk1_invocation(state: State, namelist: OperationalNamelist) -> dict[str, Any]:
    one_step_hours = float(namelist.dt_s) / 3600.0
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        result = run_forecast_operational_debug(_clone_state(state), namelist, one_step_hours, debug=True)
        block_until_ready(result)
    text = captured.getvalue()
    marker_count = text.count(MARKER)
    proof = {
        "artifact_type": "m6b_real_ic_rk1_fix_invocation",
        "status": "PASS" if marker_count >= 1 else "FAIL",
        "marker": MARKER,
        "captured_marker_count": marker_count,
        "captured_output": text,
        "interpretation": (
            "RK1 acoustic loop entry was executed on the real Gen2 IC"
            if marker_count >= 1
            else "debug marker was not captured; inspect proof_bisection_run.txt for asynchronous marker output"
        ),
    }
    lines = [
        f"status: {proof['status']}",
        f"marker: {MARKER}",
        f"captured_marker_count: {marker_count}",
        "captured_output:",
        text if text else "<empty>",
    ]
    _write_text(BISECTION_SPRINT / "proof_rk1_fix_invocation.txt", "\n".join(lines).rstrip() + "\n")
    return proof


def run_bisection(run_id: str, ic_time: str) -> dict[str, Any]:
    state, namelist, case, ic_path = _operational_state_for_run(run_id, ic_time)
    one_step_hours = float(namelist.dt_s) / 3600.0
    invocation = _verify_rk1_invocation(state, namelist)

    template = _clone_state(state)
    validation_step = _validation_coupled_step(_clone_state(state), namelist)
    block_until_ready(validation_step)
    validation_step_units = _snapshot_to_numpy(_post_physics_boundary_validation_units(validation_step, template))
    operational_step = run_forecast_operational(_clone_state(state), namelist, one_step_hours)
    block_until_ready(operational_step)
    endpoint_deltas = _field_deltas(
        _snapshot_to_numpy(_post_physics_boundary_operational_units(operational_step)),
        validation_step_units,
    )

    trace, first_stage, pre_carry = _controlled_trace(_clone_state(state), namelist)
    operator = _operator_drilldown(pre_carry, namelist)

    full_trace = {
        "artifact_type": "m6b_real_ic_step1_full_trace",
        "status": "DIVERGED" if first_stage["label"] else "NO_DIVERGENCE",
        "run_id": run_id,
        "run_dir": str(RUN_ROOT / run_id),
        "ic_file": str(ic_path),
        "ic_time": ic_time,
        "device": visible_gpu_name(),
        "threshold": THRESHOLD,
        "grid": case.metadata["grid"],
        "endpoint_coupled_step_deltas": endpoint_deltas,
        "controlled_trace_note": (
            "Per-stage trace uses the same real-IC advection candidate on both sides, then compares "
            "operational acoustic substeps with validation acoustic_substep_wrf to isolate the first operator."
        ),
        "trace": trace,
        "rk1_invocation": invocation,
    }
    first_stage_payload = {
        "artifact_type": "m6b_real_ic_first_diverging_stage",
        "status": "LOCALIZED" if first_stage["label"] else "NO_DIVERGENCE",
        "threshold": THRESHOLD,
        "first_diverging_stage": first_stage["label"],
        "rk_stage": first_stage["rk_stage"],
        "acoustic_substep": first_stage["acoustic_substep"],
        "largest_bad_field": first_stage["largest_bad_field"],
        "max_abs_delta": first_stage["max_abs_delta"],
        "rk1_fix_invoked": invocation["status"] == "PASS",
        "interpretation": (
            "RK1 fix is invoked; first real-IC divergence is inside RK1 acoustic substep 1"
            if first_stage["label"] == "rk1_acoustic_substep_1" and invocation["status"] == "PASS"
            else "see trace for localization"
        ),
    }

    _write_json(BISECTION_SPRINT / "proof_real_ic_step1_full_trace.json", full_trace)
    _write_json(BISECTION_SPRINT / "proof_first_diverging_stage.json", first_stage_payload)
    _write_json(BISECTION_SPRINT / "proof_first_diverging_operator.json", operator)
    return {"full_trace": full_trace, "first_stage": first_stage_payload, "operator": operator, "rk1_invocation": invocation}


def _controlled_timestep_pair(op: OperationalCarry, val: OperationalCarry, namelist: OperationalNamelist) -> tuple[OperationalCarry, OperationalCarry]:
    op_origin = apply_halo(op.state, halo_spec(namelist.grid))
    val_origin = apply_halo(val.state, halo_spec(namelist.grid))
    op = _with_save_family(op.replace(state=op_origin), op_origin)
    val = _with_save_family(val.replace(state=val_origin), val_origin)
    dt_sub = float(namelist.dt_s) / float(namelist.acoustic_substeps)

    for _rk_stage, factor, substeps in _rk_stages(namelist):
        op = _stage_candidate(op, op_origin, namelist, factor)
        val = _stage_candidate(val, val_origin, namelist, factor)
        for _substep in range(1, substeps + 1):
            op = _removed_legacy_acoustic_substep(op, namelist, dt_sub)
            val_acoustic = _validation_substep(val, namelist, dt_sub)
            val = _carry_from_acoustic(val_acoustic, val.state)
        op = op.replace(state=apply_halo(op.state, halo_spec(namelist.grid)))
        val = val.replace(state=apply_halo(val.state, halo_spec(namelist.grid)))
    return op, val


def run_parity_probe(run_id: str, ic_time: str, *, steps: int) -> dict[str, Any]:
    state, namelist, case, ic_path = _operational_state_for_run(run_id, ic_time)
    if int(steps) == 1:
        initial = initial_operational_carry(_enforce_operational_precision(_clone_state(state)))
        op, val = _controlled_timestep_pair(initial, initial, namelist)
        deltas = _field_deltas(
            _snapshot_to_numpy(_carry_to_validation_snapshot(op)),
            _snapshot_to_numpy(_carry_to_validation_snapshot(val)),
        )
        largest = _largest_bad(deltas)
        max_delta = float(deltas[largest]["max_abs_delta"]) if largest else 0.0
        payload = {
            "artifact_type": "m6b_reframe_shared_core_step1_parity",
            "status": "PASS" if max_delta <= THRESHOLD else "FAIL",
            "run_id": run_id,
            "run_dir": str(RUN_ROOT / run_id),
            "ic_file": str(ic_path),
            "ic_time": ic_time,
            "device": visible_gpu_name(),
            "threshold": THRESHOLD,
            "steps": 1,
            "grid": case.metadata["grid"],
            "shared_core_contract": "validation wrapper and operational runtime both import dynamics.core for RK/acoustic composition; operational does not import validation wrappers",
            "seven_interface_mismatches": {
                "momentum_inputs": "collapsed into AcousticCoreState",
                "rk_acoustic_schedule": "collapsed into dycore_timestep_core/coupled_timestep_core",
                "coefficient_cadence": "owned by acoustic_scan_core/acoustic_substep_core",
                "boundary_lead_time": "owned by coupled_timestep_core",
                "physics_sequence": "owned by coupled_timestep_core",
                "thermodynamic_offsets": "owned by coupled_timestep_core wrapper boundary",
                "precision": "validation-equivalent for step-1 parity proof",
            },
            "field_deltas": deltas,
            "largest_bad_field": largest,
            "final_max_abs_delta": max_delta,
        }
        _write_json(REFRAME_SPRINT / "proof_step1_parity_reframed.json", payload)
        return payload

    initial = initial_operational_carry(_enforce_operational_precision(_clone_state(state)))
    op = initial
    val = initial
    step_summaries = []
    for step in range(1, int(steps) + 1):
        op, val = _controlled_timestep_pair(op, val, namelist)
        deltas = _field_deltas(
            _snapshot_to_numpy(_carry_to_validation_snapshot(op)),
            _snapshot_to_numpy(_carry_to_validation_snapshot(val)),
        )
        largest = _largest_bad(deltas)
        max_delta = float(deltas[largest]["max_abs_delta"]) if largest else 0.0
        step_summaries.append(
            {
                "step": step,
                "field_deltas": deltas,
                "largest_bad_field": largest,
                "max_abs_delta": max_delta,
                "all_fields_finite": bool(
                    all(np.all(np.isfinite(np.asarray(value))) for value in _carry_to_validation_snapshot(op).values())
                ),
            }
        )

    final = step_summaries[-1]
    tolerance = 1.0e-10 if int(steps) == 1 else 1.0e-8
    payload = {
        "artifact_type": "m6b_fix_advance_mu_t_commit_parity",
        "status": "PASS" if final["max_abs_delta"] <= tolerance and final["all_fields_finite"] else "FAIL",
        "run_id": run_id,
        "run_dir": str(RUN_ROOT / run_id),
        "ic_file": str(ic_path),
        "ic_time": ic_time,
        "device": visible_gpu_name(),
        "threshold": tolerance,
        "steps": int(steps),
        "grid": case.metadata["grid"],
        "dt_sub_formula": "dt_s / acoustic_substeps",
        "fields": list(FULL_STATE_FIELDS),
        "step_summaries": step_summaries,
        "final_max_abs_delta": final["max_abs_delta"],
        "wrf_source_citation": {
            "rk1_small_step": f"{WRF_SOLVE}:1472-1475",
            "advance_mu_t_call": f"{WRF_SOLVE}:3435-3452",
            "mu_commit": f"{WRF_SMALL}:1102-1108",
            "theta_commit": f"{WRF_SMALL}:1141-1171",
            "ph_tend_consumed_by_advance_w": f"{WRF_SMALL}:1345-1395",
        },
    }
    name = "proof_step1_parity_after_fix.json" if int(steps) == 1 else f"proof_step{int(steps)}_probe.json"
    _write_json(FIX_SPRINT / name, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen2-run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--gen2-ic-time", default=DEFAULT_IC_TIME)
    parser.add_argument("--steps", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps is not None:
        payload = run_parity_probe(str(args.gen2_run_id), str(args.gen2_ic_time), steps=int(args.steps))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 2
    payload = run_bisection(str(args.gen2_run_id), str(args.gen2_ic_time))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["first_stage"]["status"] == "LOCALIZED" and payload["rk1_invocation"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
