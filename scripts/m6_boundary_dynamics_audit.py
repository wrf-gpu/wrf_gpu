#!/usr/bin/env python
"""M6 boundary/dynamics audit — read-only investigation driver.

Per the sprint contract at
``.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/sprint-contract.md``,
we run the operational forecast on each of the 3 V3 ICs for 1h (360 steps at
dt=10s) and capture per-step extrema (min, max, abs_max) for the fields the
microphysics-feedback worker flagged as physically suspect:

    p_perturbation, p_total, u, v, w

We then:
  - Compare against Gen2 wrfout per-hour extrema (best truth available).
  - Classify excursions vs physical-reason envelopes:
      pressure perturbation: |p'| ~ <= 5000 Pa anywhere on the column
      total pressure:        ~ 1 Pa (top) .. 1.1e5 Pa (surface)
      horizontal wind:       |u|,|v| <= 80 m/s jet-stream cap
      vertical motion:       |w| <= 10 m/s convective cap
  - Localize boundary-ring vs interior max for the worst field.
  - Cross-check the boundary five-cell ring vs the interior.

The pass bands per sprint contract:
  PASS=A : all fields physically reasonable (within 1x envelope).
  PASS=D : exceedances are <= 2x envelope (absurd-looking but bounded).
  FAIL=B/C : exceedances > 10x envelope.
  Intermediate (2x–10x): CONDITIONAL — bounded but suspect; flag for closer
  inspection in localization.

No code under ``src/`` is modified. No fix is attempted. This is purely an
audit: read state, compute extrema, write proof JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational


config.update("jax_enable_x64", True)


RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3")

IC_RUN_IDS = {
    "20260429": "20260429_18z_l3_24h_20260524T204451Z",
    "20260509": "20260509_18z_l3_24h_20260511T190519Z",
    "20260521": "20260521_18z_l3_24h_20260522T072630Z",
}

DT_S = 10.0
STEPS_1H = 360
BOUNDARY_RING_CELLS = 5

# Physical-reason envelope (1x band).
ENV = {
    "p_perturbation": 5.0e3,    # |p'| <= 5 kPa anywhere — strong storms maybe push 1 kPa
    "p_total":        2.0e5,    # total pressure (Pa); top ~50 Pa to surface ~1.1e5 Pa
    "u":              80.0,     # m/s, jet stream cap
    "v":              80.0,
    "w":              10.0,     # m/s, convective cap (stratiform << 1 m/s)
}

# Soft floor for p_total (Pa) — below this value at top is implausible but
# may reflect coarse top boundary conditions; values < 0 are unphysical.
P_TOTAL_FLOOR_PA = 1.0

PHYS_REASON_VERDICT = {
    "PASS_A":      "within 1x physical-reason envelope",
    "PASS_D":      "bounded within 2x envelope (absurd-looking but acceptance-bounded)",
    "COND_2_10X":  "exceeds 2x but <= 10x envelope — bounded but physically suspect",
    "FAIL_B_OR_C": "exceeds 10x envelope — dynamics or boundary defect likely",
}


# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        if np.isfinite(scalar):
            return scalar
        return str(scalar)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_minmax(arr: Any) -> dict[str, Any]:
    """Return min, max, abs_max, nonfinite_count for a JAX/Numpy array."""
    a = np.asarray(jax.device_get(arr), dtype=np.float64)
    finite_mask = np.isfinite(a)
    n_nonfinite = int(a.size - finite_mask.sum())
    if not finite_mask.any():
        return {
            "min": None,
            "max": None,
            "abs_max": None,
            "nonfinite_count": n_nonfinite,
            "size": int(a.size),
            "all_nonfinite": True,
        }
    f = a[finite_mask]
    return {
        "min": float(f.min()),
        "max": float(f.max()),
        "abs_max": float(np.abs(f).max()),
        "nonfinite_count": n_nonfinite,
        "size": int(a.size),
        "all_nonfinite": False,
    }


def _ring_vs_interior(arr3d: Any, ring: int = BOUNDARY_RING_CELLS) -> dict[str, Any]:
    """Split a (k,j,i) field into boundary-ring (first `ring` cells from any
    horizontal edge) and interior. Return per-region abs_max."""
    a = np.asarray(jax.device_get(arr3d), dtype=np.float64)
    if a.ndim != 3:
        return {"ndim": int(a.ndim), "skipped": True}
    _, ny, nx = a.shape
    if ny <= 2 * ring or nx <= 2 * ring:
        return {"ny": int(ny), "nx": int(nx), "skipped_too_small": True}
    interior_mask = np.zeros((ny, nx), dtype=bool)
    interior_mask[ring:ny - ring, ring:nx - ring] = True
    # Broadcast to (k, ny, nx) for masking on horizontal only.
    ring_slab = a[:, ~interior_mask]
    interior_slab = a[:, interior_mask]

    def _stat(s: np.ndarray) -> dict[str, Any]:
        f = s[np.isfinite(s)]
        if f.size == 0:
            return {"abs_max": None, "size": int(s.size)}
        return {"abs_max": float(np.abs(f).max()), "size": int(s.size)}

    return {
        "ring_cells": int(ring),
        "ring_horiz_cells": int((~interior_mask).sum()),
        "interior_horiz_cells": int(interior_mask.sum()),
        "ring": _stat(ring_slab),
        "interior": _stat(interior_slab),
    }


def _state_extrema(state: Any) -> dict[str, Any]:
    """Per-step extrema for the suspect fields."""
    return {
        "p_perturbation": _safe_minmax(state.p_perturbation),
        "p_total":        _safe_minmax(state.p_total),
        "p":              _safe_minmax(state.p),
        "u":              _safe_minmax(state.u),
        "v":              _safe_minmax(state.v),
        "w":              _safe_minmax(state.w),
        "theta":          _safe_minmax(state.theta),
        "qv":             _safe_minmax(state.qv),
    }


def _case_state_and_namelist(run_id: str) -> tuple[Any, OperationalNamelist, Any, dict[str, Any]]:
    run_dir = RUN_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Gen2 run dir not found: {run_dir}")
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
    meta = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "grid": case.metadata["grid"],
        "namelist": {
            "dt_s": namelist.dt_s,
            "acoustic_substeps": namelist.acoustic_substeps,
            "run_physics": namelist.run_physics,
            "run_boundary": namelist.run_boundary,
            "use_vertical_solver": namelist.use_vertical_solver,
            "radiation_cadence_steps": namelist.radiation_cadence_steps,
        },
    }
    return state, namelist, case, meta


def _wrf_hourly_extrema(run_id: str, lead_hours: int) -> dict[str, Any]:
    """Sample Gen2 wrfout at lead_hours for the 4 fields we can extract directly:
    U (u-staggered), V (v-staggered), W (w-staggered), P_HYD or P (perturbation)
    plus PB (base), giving p_total = P + PB approximately, and p_perturbation = P.
    """
    run = Gen2Run(RUN_ROOT / run_id)
    history = run.history_files("d02")
    idx = min(int(lead_hours), len(history) - 1)
    out: dict[str, Any] = {"time_index": idx, "lead_hours_requested": int(lead_hours), "path": str(history[idx])}

    def _try(varname: str) -> Any | None:
        try:
            return np.asarray(run.load("d02", varname, time=idx, lazy=False), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - hourly truth file may not have all vars
            return f"unavailable:{exc.__class__.__name__}"

    u = _try("U")
    v = _try("V")
    w = _try("W")
    p_pert = _try("P")
    pb = _try("PB")

    def _stat(x: Any) -> Any:
        if isinstance(x, str) or x is None:
            return {"unavailable": True, "note": x}
        finite = np.isfinite(x)
        if not finite.any():
            return {"all_nonfinite": True, "shape": list(x.shape)}
        return {
            "min": float(np.nanmin(x[finite])),
            "max": float(np.nanmax(x[finite])),
            "abs_max": float(np.abs(x[finite]).max()),
            "shape": list(x.shape),
        }

    out["u"] = _stat(u)
    out["v"] = _stat(v)
    out["w"] = _stat(w)
    out["p_perturbation"] = _stat(p_pert)
    if isinstance(p_pert, np.ndarray) and isinstance(pb, np.ndarray):
        out["p_total"] = _stat(p_pert + pb)
    else:
        out["p_total"] = {"unavailable": True}
    return out


# ---------------------------------------------------------------------------
# Stage 1 — catalog


def stage1_catalog(run_id: str, output_dir: Path, *, steps: int = STEPS_1H) -> dict[str, Any]:
    state, namelist, _case, meta = _case_state_and_namelist(run_id)
    step_hours = float(namelist.dt_s) / 3600.0

    timeline: list[dict[str, Any]] = []
    current = state
    timeline.append({"step": 0, "lead_s": 0.0, "extrema": _state_extrema(current), "ring": _ring_vs_interior(current.p_total)})

    wall_start = time.perf_counter()
    aborted_at: int | None = None
    for step in range(1, int(steps) + 1):
        current = run_forecast_operational(current, namelist, step_hours)
        block_until_ready(current)
        ex = _state_extrema(current)
        # Capture ring-vs-interior for p_total each step — that's the field
        # most strongly suspected to carry boundary-coupling pollution.
        ring = _ring_vs_interior(current.p_total)
        timeline.append({"step": step, "lead_s": step * float(namelist.dt_s), "extrema": ex, "ring": ring})
        any_nonfinite = any(
            (ex[name]["nonfinite_count"] > 0) if (isinstance(ex[name], dict) and "nonfinite_count" in ex[name]) else False
            for name in ("p_perturbation", "p_total", "u", "v", "w", "theta")
        )
        if any_nonfinite:
            aborted_at = step
            break
    wall_s = time.perf_counter() - wall_start

    wrf_ref = _wrf_hourly_extrema(run_id, lead_hours=1)

    payload = {
        "artifact_type": "m6_boundary_dynamics_audit_stage1_catalog",
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": visible_gpu_name(),
        "input": meta,
        "steps_requested": int(steps),
        "steps_completed": int(steps if aborted_at is None else aborted_at),
        "aborted_on_nonfinite_step": aborted_at,
        "wall_time_s": wall_s,
        "wrf_reference_at_lead_1h": wrf_ref,
        "envelope_used_for_classification": ENV,
        "timeline": timeline,
    }
    _write_json(output_dir / f"proof_excursion_catalog_{run_id.split('_')[0]}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage 2 — classify


def _classify_field(field: str, run_max_abs: float | None, run_min: float | None, run_max: float | None) -> dict[str, Any]:
    if run_max_abs is None:
        return {"verdict": "FAIL_NONFINITE", "note": "all-nonfinite or unavailable"}

    env = ENV[field]
    ratio = float(run_max_abs) / float(env)
    if ratio <= 1.0:
        verdict = "PASS_A"
    elif ratio <= 2.0:
        verdict = "PASS_D"
    elif ratio <= 10.0:
        verdict = "COND_2_10X"
    else:
        verdict = "FAIL_B_OR_C"

    # Special-case p_total: negative values are unphysical regardless of ratio.
    extra = {}
    if field == "p_total" and run_min is not None and run_min < P_TOTAL_FLOOR_PA:
        extra["p_total_below_floor"] = True
        extra["p_total_min_seen_pa"] = float(run_min)
        if verdict in ("PASS_A", "PASS_D"):
            verdict = "COND_2_10X"
        if run_min < -1.0:
            verdict = "FAIL_B_OR_C"

    return {
        "envelope_1x": env,
        "run_abs_max": float(run_max_abs),
        "ratio_to_envelope": ratio,
        "verdict": verdict,
        "note": PHYS_REASON_VERDICT.get(verdict, ""),
        **extra,
    }


def stage2_classify(catalogs: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    per_ic: dict[str, Any] = {}
    worst = {"ic": None, "field": None, "ratio": -1.0, "step": None, "verdict": "PASS_A"}

    def _verdict_rank(v: str) -> int:
        return {"PASS_A": 0, "PASS_D": 1, "COND_2_10X": 2, "FAIL_B_OR_C": 3, "FAIL_NONFINITE": 4}.get(v, 0)

    for ic_label, cat in catalogs.items():
        timeline = cat["timeline"]
        per_field: dict[str, Any] = {}
        for field in ("p_perturbation", "p_total", "u", "v", "w"):
            run_abs_max = None
            run_min = None
            run_max = None
            argmax_step = None
            for entry in timeline:
                ex = entry["extrema"].get(field)
                if not isinstance(ex, dict):
                    continue
                am = ex.get("abs_max")
                if am is None:
                    continue
                if (run_abs_max is None) or (am > run_abs_max):
                    run_abs_max = float(am)
                    argmax_step = int(entry["step"])
                mn = ex.get("min")
                if mn is not None and (run_min is None or mn < run_min):
                    run_min = float(mn)
                mx = ex.get("max")
                if mx is not None and (run_max is None or mx > run_max):
                    run_max = float(mx)
            cls = _classify_field(field, run_abs_max, run_min, run_max)
            cls["step_of_run_abs_max"] = argmax_step
            cls["run_min"] = run_min
            cls["run_max"] = run_max
            per_field[field] = cls

            if _verdict_rank(cls["verdict"]) > _verdict_rank(worst["verdict"]) or (
                _verdict_rank(cls["verdict"]) == _verdict_rank(worst["verdict"]) and cls.get("ratio_to_envelope", 0) > worst["ratio"]
            ):
                worst = {
                    "ic": ic_label,
                    "field": field,
                    "ratio": float(cls.get("ratio_to_envelope", 0.0)),
                    "step": argmax_step,
                    "verdict": cls["verdict"],
                }
        per_ic[ic_label] = {
            "wrf_reference_at_lead_1h": cat.get("wrf_reference_at_lead_1h"),
            "fields": per_field,
        }

    aggregate_verdict = worst["verdict"]
    payload = {
        "artifact_type": "m6_boundary_dynamics_audit_stage2_classification",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "envelope_used_for_classification": ENV,
        "verdict_meaning": PHYS_REASON_VERDICT,
        "per_ic": per_ic,
        "worst_excursion": worst,
        "aggregate_verdict": aggregate_verdict,
    }
    _write_json(output_dir / "proof_excursion_classification.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage 3 — localize


def stage3_localize(catalogs: dict[str, dict[str, Any]], classification: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    worst = classification["worst_excursion"]
    if worst["ic"] is None:
        payload = {
            "artifact_type": "m6_boundary_dynamics_audit_stage3_source_localization",
            "status": "SKIPPED_NO_WORST",
            "note": "no worst excursion identified — Stage 2 likely all PASS_A.",
        }
        _write_json(output_dir / "proof_source_localization.json", payload)
        return payload

    ic_label = worst["ic"]
    field = worst["field"]
    run_id = IC_RUN_IDS[ic_label]
    step_target = int(worst["step"])
    cat = catalogs[ic_label]

    # Replay step-by-step until step_target, snapshot the full 3D field there,
    # then compute boundary-ring vs interior + argmax location.
    state, namelist, _case, _meta = _case_state_and_namelist(run_id)
    step_hours = float(namelist.dt_s) / 3600.0
    current = state
    for step in range(1, max(1, step_target) + 1):
        current = run_forecast_operational(current, namelist, step_hours)
        block_until_ready(current)
        if step == step_target:
            break

    arr = np.asarray(jax.device_get(getattr(current, field)), dtype=np.float64)

    finite = np.isfinite(arr)
    if finite.any():
        abs_arr = np.where(finite, np.abs(arr), -np.inf)
        flat_idx = int(np.argmax(abs_arr))
        loc = np.unravel_index(flat_idx, arr.shape)
    else:
        loc = (0, 0, 0)

    if arr.ndim == 3:
        k, j, i = (int(loc[0]), int(loc[1]), int(loc[2]))
    elif arr.ndim == 2:
        j, i, k = int(loc[0]), int(loc[1]), 0
    else:
        k, j, i = 0, 0, 0

    ring = _ring_vs_interior(arr)
    # Classify boundary vs interior:
    src = "indeterminate"
    rm = ring.get("ring", {}).get("abs_max")
    im = ring.get("interior", {}).get("abs_max")
    if isinstance(rm, (int, float)) and isinstance(im, (int, float)):
        if rm > 2.0 * im:
            src = "A_boundary_dominant"     # boundary forcing
        elif im > 2.0 * rm:
            src = "B_or_C_interior_dominant"  # dynamics core or operational composition
        else:
            src = "mixed"

    if arr.ndim == 3:
        ny, nx = arr.shape[1], arr.shape[2]
    elif arr.ndim == 2:
        ny, nx = arr.shape[0], arr.shape[1]
    else:
        ny, nx = 0, 0
    in_ring = (j < BOUNDARY_RING_CELLS or i < BOUNDARY_RING_CELLS or
               j >= ny - BOUNDARY_RING_CELLS or i >= nx - BOUNDARY_RING_CELLS)

    snapshot = {
        "argmax_abs_cell": {"k": k, "j": j, "i": i},
        "argmax_value": float(arr[k, j, i]) if arr.ndim == 3 and np.isfinite(arr[k, j, i]) else str(arr[k, j, i] if arr.ndim == 3 else arr[j, i]),
        "argmax_in_boundary_ring": bool(in_ring),
        "ring_vs_interior_abs_max": ring,
        "ring_boundary_dominance_verdict": src,
    }

    # WRF reference at this step's nearest hour for context:
    nearest_hour = max(0, int(round(step_target * float(namelist.dt_s) / 3600.0)))
    wrf_ref = _wrf_hourly_extrema(run_id, lead_hours=nearest_hour)

    payload = {
        "artifact_type": "m6_boundary_dynamics_audit_stage3_source_localization",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "worst_excursion_from_stage2": worst,
        "replay_step": step_target,
        "field": field,
        "snapshot": snapshot,
        "wrf_reference_at_nearest_hour": wrf_ref,
        "note": (
            "argmax_in_boundary_ring=True + ring >> interior → (A) boundary forcing "
            "(wrfbdy_d01→wrfinput_d02 interpolation). interior >> ring → (B/C) dycore "
            "or operational composition. mixed → no decisive separation, look at "
            "earlier step where excursion first crosses physical-reason."
        ),
    }
    _write_json(output_dir / "proof_source_localization.json", payload)
    return payload


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Sprint output directory.")
    parser.add_argument(
        "--ics",
        default=",".join(IC_RUN_IDS.keys()),
        help="Comma-separated subset of IC labels (default: all 3).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS_1H,
        help="How many dt steps to run per IC (default 360 = 1h).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_labels = [s.strip() for s in args.ics.split(",") if s.strip()]

    catalogs: dict[str, dict[str, Any]] = {}
    for ic in ic_labels:
        if ic not in IC_RUN_IDS:
            raise SystemExit(f"unknown IC label: {ic}; valid: {list(IC_RUN_IDS)}")
        run_id = IC_RUN_IDS[ic]
        print(f"[stage1] cataloging {ic} run_id={run_id} steps={args.steps}", flush=True)
        catalogs[ic] = stage1_catalog(run_id, output_dir, steps=int(args.steps))
        last = catalogs[ic]["timeline"][-1]
        ex = last["extrema"]
        print(
            f"[stage1] {ic} step={last['step']}/{args.steps} "
            f"p_total abs_max={ex['p_total']['abs_max']} "
            f"u abs_max={ex['u']['abs_max']} v abs_max={ex['v']['abs_max']} "
            f"w abs_max={ex['w']['abs_max']}",
            flush=True,
        )

    print("[stage2] classifying excursions", flush=True)
    classification = stage2_classify(catalogs, output_dir)
    print(f"[stage2] aggregate verdict = {classification['aggregate_verdict']}", flush=True)
    print(f"[stage2] worst excursion = {classification['worst_excursion']}", flush=True)

    print("[stage3] localizing worst-excursion source", flush=True)
    localization = stage3_localize(catalogs, classification, output_dir)
    print(f"[stage3] verdict = {localization.get('snapshot', {}).get('ring_boundary_dominance_verdict')}", flush=True)

    summary = {
        "artifact_type": "m6_boundary_dynamics_audit_summary",
        "ics_audited": ic_labels,
        "steps_per_ic": int(args.steps),
        "aggregate_verdict": classification["aggregate_verdict"],
        "worst_excursion": classification["worst_excursion"],
        "ring_dominance_verdict": localization.get("snapshot", {}).get("ring_boundary_dominance_verdict"),
        "proofs": [
            *(f"proof_excursion_catalog_{ic}.json" for ic in ic_labels),
            "proof_excursion_classification.json",
            "proof_source_localization.json",
        ],
    }
    _write_json(output_dir / "proof_audit_summary.json", summary)
    print(f"[done] summary written to {output_dir / 'proof_audit_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
