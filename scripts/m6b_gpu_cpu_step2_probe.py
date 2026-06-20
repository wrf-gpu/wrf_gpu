#!/usr/bin/env python
"""M6b GPU/CPU step-2 divergence probe.

Diagnostic only.  This script compares the validation wrapper composition and
operational composition for five 10 s steps on the pinned 20260521 Gen2 d02
initial condition.  It writes one per-path proof and, once all four contracted
outputs exist, the aggregate 4x5 matrix plus the divergence memos.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
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

from gpuwrf.dynamics.validation_wrappers import DycoreStepConfig, dycore_timestep_wrf
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _acoustic_core_state,
    _enforce_operational_precision,
    _theta_base_offset,
    run_forecast_operational,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


config.update("jax_enable_x64", True)

SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-gpu-cpu-step2-divergence"
V3_SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-honest-1h-canary-V3"
CPU_BISECT_REPORT = (
    ROOT
    / ".agent"
    / "sprints"
    / "2026-05-25-m6b-standalone-vs-comparator-bisect"
    / "worker-report.md"
)
RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3")
DEFAULT_RUN_ID = "20260521_18z_l3_24h_20260522T072630Z"
DEFAULT_IC_TIME = "2026-05-21_18:00:00"
DT_S = 10.0
SUMMARY_FIELDS = ("theta", "mu", "u", "v", "w")
EXPECTED_OUTPUTS = {
    "cpu_validation": SPRINT / "cpu_validation.json",
    "cpu_operational": SPRINT / "cpu_operational.json",
    "gpu_validation": SPRINT / "gpu_validation.json",
    "gpu_operational": SPRINT / "gpu_operational.json",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _all_leaves_finite(state: Any) -> bool:
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]
    return bool(np.asarray(jnp.all(jnp.asarray(checks))))


def _first_nonfinite_field(state: Any) -> str | None:
    for name in getattr(state, "__slots__", ()):
        value = np.asarray(getattr(state, name))
        if not np.all(np.isfinite(value)):
            return str(name)
    return None


def _field_summary(state: Any, name: str) -> dict[str, Any]:
    array = np.asarray(getattr(state, name))
    finite = array[np.isfinite(array)]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        f"min_{name}": _json_float(np.min(finite)) if finite.size else None,
        f"max_{name}": _json_float(np.max(finite)) if finite.size else None,
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "all_finite": bool(finite.size == array.size),
    }


def _state_step_summary(
    *,
    state: Any,
    path_name: str,
    step: int,
    wall_s: float,
    run_id: str,
) -> dict[str, Any]:
    field_stats = {name: _field_summary(state, name) for name in SUMMARY_FIELDS}
    return {
        "path": path_name,
        "step": int(step),
        "lead_seconds": float(step) * DT_S,
        "all_state_leaves_finite": _all_leaves_finite(state),
        "largest_bad_field": _first_nonfinite_field(state),
        "wall_time_s_including_compile": float(wall_s),
        "fields": field_stats,
        "max_theta": field_stats["theta"]["max_theta"],
        "min_theta": field_stats["theta"]["min_theta"],
        "max_mu": field_stats["mu"]["max_mu"],
        "min_mu": field_stats["mu"]["min_mu"],
        "max_u": field_stats["u"]["max_u"],
        "min_u": field_stats["u"]["min_u"],
        "max_v": field_stats["v"]["max_v"],
        "min_v": field_stats["v"]["min_v"],
        "max_w": field_stats["w"]["max_w"],
        "min_w": field_stats["w"]["min_w"],
        "wrf_reference": {
            "available": False,
            "max_abs_delta": None,
            "reason": (
                f"{int(step)} contracted steps cover {int(step * DT_S)} s; "
                "the Gen2 d02 WRF reference available to this harness is hourly wrfout history."
            ),
        },
        "run_id": run_id,
    }


def _case_state_and_namelist(run_id: str, ic_time: str) -> tuple[Any, OperationalNamelist, Any, Path]:
    run_dir = RUN_ROOT / run_id
    ic_path = run_dir / f"wrfout_d02_{ic_time}"
    if not ic_path.is_file():
        raise FileNotFoundError(ic_path)
    if _platform_label() == "cpu":
        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=DT_S,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
    )
    return state, namelist, case, ic_path


def _coupled_extras(state: Any) -> dict[str, jax.Array]:
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


def _state_from_coupled_snapshot(snapshot: dict[str, Any], template: Any, theta_offset: Any, dt_s: float) -> Any:
    theta = jnp.asarray(snapshot["theta"]) + theta_offset
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
        qv=template.qv + jnp.asarray(snapshot["qv_phys_tend"]) * float(dt_s),
        qc=template.qc + jnp.asarray(snapshot["qc_phys_tend"]) * float(dt_s),
        qr=template.qr + jnp.asarray(snapshot["qr_phys_tend"]) * float(dt_s),
        qi=template.qi + jnp.asarray(snapshot["qi_phys_tend"]) * float(dt_s),
        qs=template.qs + jnp.asarray(snapshot["qs_phys_tend"]) * float(dt_s),
        qg=template.qg + jnp.asarray(snapshot["qg_phys_tend"]) * float(dt_s),
        qke=template.qke + jnp.asarray(snapshot["qke_phys_tend"]) * float(dt_s),
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


def _carry_from_coupled_snapshot(snapshot: dict[str, Any], template: Any, theta_offset: Any, dt_s: float) -> OperationalCarry:
    next_state = _state_from_coupled_snapshot(snapshot, template, theta_offset, dt_s)
    return OperationalCarry(
        state=next_state,
        t_2ave=jnp.asarray(snapshot["t_2ave"]) + theta_offset,
        ww=jnp.asarray(snapshot["ww"]),
        mudf=jnp.asarray(snapshot["mudf"]),
        muave=jnp.asarray(snapshot["muave"]),
        muts=jnp.asarray(snapshot["muts"]),
        ph_tend=jnp.asarray(snapshot["ph_tend"]),
        u_save=next_state.u,
        v_save=next_state.v,
        w_save=next_state.w,
        t_save=next_state.theta,
        ph_save=next_state.ph,
        mu_save=jnp.asarray(snapshot["mu"]),
        ww_save=jnp.asarray(snapshot["ww"]),
    )


def _carry_from_dycore_snapshot(snapshot: dict[str, Any], template: Any, theta_offset: Any) -> OperationalCarry:
    theta = jnp.asarray(snapshot["theta"]) + theta_offset
    p_pert = jnp.asarray(snapshot["p"])
    ph_pert = jnp.asarray(snapshot["ph"])
    mu_pert = jnp.asarray(snapshot["mu"])
    p_total = template.p_total - template.p_perturbation + p_pert
    ph_total = template.ph_total - template.ph_perturbation + ph_pert
    mu_total = template.mu_total - template.mu_perturbation + mu_pert
    next_state = template.replace(
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
    return OperationalCarry(
        state=next_state,
        t_2ave=jnp.asarray(snapshot["t_2ave"]) + theta_offset,
        ww=jnp.asarray(snapshot["ww"]),
        mudf=jnp.asarray(snapshot["mudf"]),
        muave=jnp.asarray(snapshot["muave"]),
        muts=jnp.asarray(snapshot["muts"]),
        ph_tend=jnp.asarray(snapshot["ph_tend"]),
        u_save=next_state.u,
        v_save=next_state.v,
        w_save=next_state.w,
        t_save=next_state.theta,
        ph_save=next_state.ph,
        mu_save=jnp.asarray(snapshot["mu"]),
        ww_save=jnp.asarray(snapshot["ww"]),
    )


def _run_validation_path(state: Any, namelist: OperationalNamelist, steps: int, run_id: str) -> list[dict[str, Any]]:
    carry = initial_operational_carry(_enforce_operational_precision(state))
    records: list[dict[str, Any]] = []
    for step in range(1, int(steps) + 1):
        start = time.perf_counter()
        acoustic = _acoustic_core_state(carry, namelist)
        theta_offset = _theta_base_offset(carry.state.theta)
        _rk_snapshots, snapshot = dycore_timestep_wrf(
            acoustic,
            namelist.metrics,
            DycoreStepConfig(
                dt=float(namelist.dt_s),
                dx=float(namelist.grid.projection.dx_m),
                dy=float(namelist.grid.projection.dy_m),
                acoustic_substeps=int(namelist.acoustic_substeps),
                rk_order=int(namelist.rk_order),
                epssm=float(namelist.epssm),
                top_lid=bool(namelist.top_lid),
                physics_enabled=False,
                boundary_enabled=False,
            ),
        )
        block_until_ready(snapshot)
        carry = _carry_from_dycore_snapshot(snapshot, carry.state, theta_offset)
        block_until_ready(carry.state)
        wall_s = time.perf_counter() - start
        records.append(_state_step_summary(state=carry.state, path_name="validation", step=step, wall_s=wall_s, run_id=run_id))
    return records


def _run_operational_path(state: Any, namelist: OperationalNamelist, steps: int, run_id: str) -> list[dict[str, Any]]:
    current = state
    step_hours = float(namelist.dt_s) / 3600.0
    records: list[dict[str, Any]] = []
    for step in range(1, int(steps) + 1):
        start = time.perf_counter()
        if _platform_label() == "cpu":
            with jax.disable_jit():
                current = run_forecast_operational(current, namelist, step_hours)
        else:
            current = run_forecast_operational(current, namelist, step_hours)
        block_until_ready(current)
        wall_s = time.perf_counter() - start
        records.append(_state_step_summary(state=current, path_name="operational", step=step, wall_s=wall_s, run_id=run_id))
    return records


def _platform_label() -> str:
    requested = os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME") or "default"
    backend = jax.default_backend()
    if requested == "cpu" or backend == "cpu":
        return "cpu"
    return "gpu"


def _run_probe(path_name: str, steps: int, run_id: str, ic_time: str) -> dict[str, Any]:
    state, namelist, case, ic_path = _case_state_and_namelist(run_id, ic_time)
    if path_name == "validation":
        records = _run_validation_path(state, namelist, steps, run_id)
    elif path_name == "operational":
        records = _run_operational_path(state, namelist, steps, run_id)
    else:
        raise ValueError(f"unknown path {path_name!r}")
    first_nonfinite = next((record for record in records if not record["all_state_leaves_finite"]), None)
    payload = {
        "artifact_type": "m6b_gpu_cpu_step2_single_path_probe",
        "status": "PASS" if first_nonfinite is None else "NONFINITE",
        "platform": _platform_label(),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "device": visible_gpu_name(),
        "path": path_name,
        "steps": int(steps),
        "dt_s": DT_S,
        "run_id": run_id,
        "run_dir": str(RUN_ROOT / run_id),
        "ic_file": str(ic_path),
        "grid": case.metadata["grid"],
        "first_nonfinite_step": None if first_nonfinite is None else int(first_nonfinite["step"]),
        "first_nonfinite_field": None if first_nonfinite is None else first_nonfinite["largest_bad_field"],
        "matrix_rows": records,
    }
    return payload


def _step2_bad(record: dict[str, Any]) -> bool:
    rows = record.get("matrix_rows", [])
    if len(rows) < 2:
        return False
    return not bool(rows[1].get("all_state_leaves_finite"))


def _step2_finite(record: dict[str, Any]) -> bool:
    rows = record.get("matrix_rows", [])
    return len(rows) >= 2 and bool(rows[1].get("all_state_leaves_finite"))


def _derive_verdict(records: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    cv = records.get("cpu_validation")
    co = records.get("cpu_operational")
    gv = records.get("gpu_validation")
    go = records.get("gpu_operational")
    if not all((cv, co, gv, go)):
        return "INSUFFICIENT-EVIDENCE", "blocker", "Complete all four contracted probes."
    cpu_step2_finite = _step2_finite(cv) and _step2_finite(co)
    if _step2_finite(gv) and _step2_finite(go) and cpu_step2_finite:
        return (
            "(A)-SENTINEL-COINCIDENCE",
            "minor",
            "2026-05-25-m6b-comparator-nan-sentinel-audit: audit comparator max_abs_delta arithmetic, NaN sentinel handling, and field-order reporting.",
        )
    if _step2_bad(gv) and cpu_step2_finite:
        return (
            "(B)-GPU-SHARED-BUG",
            "blocker",
            "2026-05-25-m6b-gpu-shared-core-step2-localization: localize GPU-only divergence inside dynamics/core acoustic scan, tridiagonal solve, and JAX lax.scan precision behavior.",
        )
    if _step2_bad(gv) and _step2_bad(go) and cpu_step2_finite:
        return (
            "(B)-GPU-SHARED-BUG",
            "blocker",
            "2026-05-25-m6b-gpu-shared-core-step2-localization: localize shared GPU core divergence before touching operational composition.",
        )
    if _step2_finite(gv) and _step2_bad(go):
        return (
            "(C)-OPERATIONAL-ONLY-GPU",
            "blocker",
            "2026-05-25-m6b-operational-gpu-step2-localization: isolate operational-only GPU divergence against the validation wrapper path.",
        )
    return (
        "INSUFFICIENT-EVIDENCE",
        "blocker",
        "2026-05-25-m6b-step2-probe-rerun: rerun the four-path probe with complete logs and first-bad-field snapshots.",
    )


def _read_expected_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for key, path in EXPECTED_OUTPUTS.items():
        if path.is_file():
            try:
                records[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return records


def _load_v3_bounds_summary() -> dict[str, Any]:
    path = V3_SPRINT / "proof_bounds.json"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for run in payload.get("runs", []):
        bounds = run.get("bounds_audit", {})
        first_bad = bounds.get("first_bad_step") or {}
        rows.append(
            {
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "blocker": run.get("blocker"),
                "steps_checked": bounds.get("steps_checked"),
                "first_bad_step": first_bad.get("step"),
                "first_bad_lead_seconds": first_bad.get("lead_seconds"),
                "first_bad_all_leaves_finite": first_bad.get("all_leaves_finite"),
            }
        )
    return {"available": True, "path": str(path), "status": payload.get("status"), "runs": rows}


def _write_v3_diff(records: dict[str, dict[str, Any]], verdict: str) -> None:
    v3_summary = _load_v3_bounds_summary()
    source_path = ROOT / "scripts" / "m6b_canary_1h_honest_v3.py"
    source = source_path.read_text(encoding="utf-8")
    v3_uses_operational = "run_forecast_operational" in source
    v3_uses_validation = "validation_wrappers" in source
    cpu_report_text = CPU_BISECT_REPORT.read_text(encoding="utf-8") if CPU_BISECT_REPORT.is_file() else ""
    cpu_report_anchor = "2, 5, 10 steps" if "2, 5, 10" in cpu_report_text else "not found"
    complete = set(records) == set(EXPECTED_OUTPUTS)
    lines = [
        "# V3 vs Comparator Diff",
        "",
        f"- Aggregate matrix complete: {complete}.",
        f"- Current verdict: {verdict}.",
        f"- V3 script path: `scripts/m6b_canary_1h_honest_v3.py`.",
        f"- V3 uses `run_forecast_operational`: {v3_uses_operational}.",
        f"- V3 imports validation wrappers: {v3_uses_validation}.",
        f"- Prior CPU bisect anchor in worker report: {cpu_report_anchor}; report path `{CPU_BISECT_REPORT.relative_to(ROOT)}`.",
        "- V3 bounds path was a stepwise operational-mode bounds audit, not a validation-wrapper comparator.",
        "- Comparator/bisect path checked bitwise harness agreement; V3 checked physical bounds and fail-fast validity.",
        "- Tolerance/field-order difference: comparator proofs use max-abs field deltas over broad state snapshots; V3 records theta/u/v/w physical bounds and does not compute `max_abs_delta`.",
        "- Warm/cold cache difference: V3 compiled during the stepwise audit and did not record a JIT-cache warm-state discriminator. This sprint records wall time per step but does not claim performance.",
        "",
        "## V3 Bounds Summary",
        "",
        "```json",
        json.dumps(v3_summary, indent=2, sort_keys=True),
        "```",
    ]
    _write_text(SPRINT / "proof_v3_vs_comparator_diff.md", "\n".join(lines))


def _write_divergence_memo(records: dict[str, dict[str, Any]], verdict: str, severity: str, next_sprint: str) -> None:
    step2_rows = {}
    for key, record in records.items():
        rows = record.get("matrix_rows", [])
        if len(rows) >= 2:
            row = rows[1]
            step2_rows[key] = {
                "all_state_leaves_finite": row.get("all_state_leaves_finite"),
                "largest_bad_field": row.get("largest_bad_field"),
                "max_theta": row.get("max_theta"),
                "min_theta": row.get("min_theta"),
                "max_mu": row.get("max_mu"),
                "min_mu": row.get("min_mu"),
            }
    lines = [
        "# Divergence Memo",
        "",
        f"Verdict: {verdict}",
        "",
        "Evidence:",
        f"- 4x5 matrix: `{(SPRINT / 'proof_4path_step2_matrix.json').relative_to(ROOT)}`.",
        f"- V3-vs-comparator diff: `{(SPRINT / 'proof_v3_vs_comparator_diff.md').relative_to(ROOT)}`.",
        f"- Step-2 rows: `{json.dumps(step2_rows, sort_keys=True)}`.",
        "",
        f"Recommended next sprint: {next_sprint}",
        "",
        f"Severity for M6 close: {severity}",
    ]
    if verdict == "(A)-SENTINEL-COINCIDENCE":
        lines.append("Note: this resolves only the step-2 contradiction; the V3 physical-bounds blocker remains separate evidence against closing M6.")
    _write_text(SPRINT / "divergence_memo.md", "\n".join(lines))


def _maybe_write_aggregate() -> None:
    records = _read_expected_records()
    verdict, severity, next_sprint = _derive_verdict(records)
    matrix = {
        "artifact_type": "m6b_gpu_cpu_step2_4path_matrix",
        "status": "COMPLETE" if set(records) == set(EXPECTED_OUTPUTS) else "PARTIAL",
        "expected_outputs": {key: str(path.relative_to(ROOT)) for key, path in EXPECTED_OUTPUTS.items()},
        "present_outputs": sorted(records),
        "verdict": verdict,
        "severity_for_m6_close": severity,
        "recommended_next_sprint": next_sprint,
        "matrix": {key: record.get("matrix_rows", []) for key, record in sorted(records.items())},
    }
    _write_json(SPRINT / "proof_4path_step2_matrix.json", matrix)
    _write_v3_diff(records, verdict)
    _write_divergence_memo(records, verdict, severity, next_sprint)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("validation", "operational"), required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gen2-run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--gen2-ic-time", default=DEFAULT_IC_TIME)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = _run_probe(str(args.path), int(args.steps), str(args.gen2_run_id), str(args.gen2_ic_time))
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    _write_json(output, payload)
    _maybe_write_aggregate()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
