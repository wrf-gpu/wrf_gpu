#!/usr/bin/env python
"""v0.1.0 d03 1km Tenerife replay-nesting forecast + validation orchestrator.

This is a focused one-level-down extension of ``scripts/m7_l2_d02_replay.py``.
The d02 path drives a 3km GPU forecast from replayed d01 boundaries; this path
drives a **1km Tenerife (d03)** GPU forecast from replayed **d02** boundaries,
then validates it against the corpus L3 d03 (1km TF) truth.

Reuse (NOT duplication):
  * Boundary forcing -- ``gpuwrf.integration.d02_replay.build_replay_case`` with
    ``boundary_domain="d02"`` invokes the SAME generic nested-parent machinery
    (``load_nested_parent_boundary_leaves`` / ``_nested_axis_coords`` /
    ``_interp_parent_horizontal``) that the d02 path uses for d01->d02.  The
    child's own grid metadata (parent_grid_ratio=3, i_parent_start=52,
    j_parent_start=20 relative to d02) selects the right parent sub-window
    automatically -- no d03-specific interpolation code is added.
  * Forecast engine -- ``gpuwrf.integration.daily_pipeline.execute_daily_pipeline``
    runs the per-forecast-hour segmented operational scan, writes wrfout files,
    and performs the finite / land-refresh guards.  We only supply a d03
    ``case_builder`` and a d03 ``DailyPipelineConfig``.
  * Operational numerics -- the case builder uses the exact Sprint-U operational
    namelist of ``daily_pipeline._build_real_case`` (force_fp64, top_lid=True,
    epssm=0.5, use_flux_advection, diff_6th_opt=2/0.12, w_damping=1, damp_opt=3,
    zdamp=5000, dampcoef=0.2), with a CFL-appropriate 1km timestep.
  * Validation -- ``gpuwrf.validation.data_quality.compute_rmse_against_gen2``
    (generic; the d02 ``write_tier4_rmse`` calls the same function) plus a
    persistence (t=0 hold) baseline so the forecast skill is reported against a
    do-nothing reference, not just an absolute RMSE.

The d03 1km timestep: the L3 namelist runs d01=18s, d02=6s, d03=2s (WRF native,
parent_time_step_ratio=3).  The d02 GPU pipeline uses dt=10s at 3km; scaling by
the 3x finer grid gives ~3.3s.  We use dt=3.0s (acoustic_substeps=10) for d03 to
keep horizontal advective+acoustic CFL safely below the d02 operating point.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from netCDF4 import Dataset

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.coupling.boundary_apply import BoundaryConfig  # noqa: E402
from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402
from gpuwrf.integration.daily_pipeline import (  # noqa: E402
    DailyCase,
    DailyPipelineConfig,
    execute_daily_pipeline,
    resolve_run_dir,
    write_json,
)
from gpuwrf.io.data_inventory import parse_run_id, parse_wrfout_valid_time  # noqa: E402
from gpuwrf.runtime.operational_mode import OperationalNamelist  # noqa: E402
from gpuwrf.validation.data_quality import compute_rmse_against_gen2  # noqa: E402


L3_RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
OUTPUT_ROOT = Path("/tmp/v010_d03_runs")
PROOF_DIR = ROOT / "proofs" / "v010_validation"
DEFAULT_RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"

# Tenerife 1km mass grid (93x75x44) -- guards the i/o domain wiring.
EXPECTED_D03_MASS_SHAPE_YX = (75, 93)
D03_DT_S = 3.0
D03_ACOUSTIC_SUBSTEPS = 10
D03_RADIATION_CADENCE_STEPS = 600  # ~30 min radiation cadence at 3s dt.

# Operational acceptance thresholds (informational gate; the d02 lane uses the
# same numbers).  We additionally require the GPU forecast to BEAT persistence.
RMSE_THRESHOLDS = {"T2": 3.0, "U10": 7.5, "V10": 7.5}
PRECIP_FIELD = "RAINNC"


def _pin_orchestration_cpus() -> list[int] | None:
    if not hasattr(os, "sched_setaffinity"):
        return None
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass
    return sorted(os.sched_getaffinity(0))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _coerce_run_start(value: str) -> datetime:
    text = value.strip().replace("Z", "")
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def build_l3_d03_daily_case(config: DailyPipelineConfig) -> tuple[DailyCase, Path]:
    """Build the d03 1km Tenerife daily case: d02-nested boundaries + Sprint-U numerics."""

    run_dir = resolve_run_dir(config.run_id, config.run_root)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing run directory: {run_dir}")
    # Nested-parent (d02 -> d03) boundary forcing via the SAME generic machinery
    # the d02 path uses for d01 -> d02.  No d03-specific interpolation code.
    replay = build_replay_case(run_dir, domain=config.domain, boundary_domain="d02")
    actual_yx = (int(replay.grid.ny), int(replay.grid.nx))
    if actual_yx != EXPECTED_D03_MASS_SHAPE_YX:
        raise ValueError(
            f"d03 mass grid must be {EXPECTED_D03_MASS_SHAPE_YX} (1km Tenerife); "
            f"got {actual_yx} from {run_dir}"
        )
    state = replay.state.replace(
        p=replay.state.p_total, ph=replay.state.ph_total, mu=replay.state.mu_total
    )
    # IDENTICAL Sprint-U operational numerics to daily_pipeline._build_real_case
    # (grid-agnostic: epssm/diff_6th/damp_opt/non_hydrostatic are per-domain
    # constants in the real L3 namelist).  dt is the only knob scaled for 1km.
    # NESTED-boundary geopotential fix: the d02->d03 boundary leaves are the
    # PARENT (3 km) perturbation geopotential bilinearly interpolated to the 1 km
    # child grid, which is NOT hydrostatically consistent with the child column.
    # Overwriting the prognostic ``ph`` in the boundary ring from that strip every
    # acoustic-coupled step pumps spurious vertical motion that warms the interior
    # (proven root cause of the d03 +6.8 K T2 / +7 K theta[0] bias: the ph forcing
    # alone reproduces +2.84 K/10 min, while forcing u/v/w/theta/qv/mu/p each stays
    # within +/-0.13 K).  Disable geopotential overwrite for the nested boundary;
    # ``ph`` then stays dynamically/hydrostatically consistent with the forced
    # mu/theta.  The validated d02 SELF-REPLAY path keeps force_geopotential=True
    # (its strips ARE self-consistent), so this does not touch d02.
    # P0-6 in-loop nested ph'/w boundary forcing toggles (env override for the
    # short-run isolation sweep; defaults match BoundaryConfig).
    def _envflag(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")

    # Defaults match BoundaryConfig (all OFF -> validated free-drift baseline; the
    # 2026-06-01 short-d03 sweep showed the in-loop ph' forcing toward the decoupled
    # parent leaf pumps interior w, see ...-opus-d03-phfix-INLOOP-findings.md).  The
    # env vars are the isolation-sweep override only.
    _DEF = BoundaryConfig()
    nested_boundary_config = BoundaryConfig(
        force_geopotential=False,
        nested_ph_relax=_envflag("D03_NESTED_PH_RELAX", _DEF.nested_ph_relax),
        nested_w_relax=_envflag("D03_NESTED_W_RELAX", _DEF.nested_w_relax),
        nested_ph_spec=_envflag("D03_NESTED_PH_SPEC", _DEF.nested_ph_spec),
    )
    namelist = OperationalNamelist.from_grid(
        replay.grid,
        tendencies=replay.tendencies,
        metrics=replay.metrics,
        dt_s=float(config.dt_s),
        acoustic_substeps=int(config.acoustic_substeps),
        radiation_cadence_steps=int(config.radiation_cadence_steps),
        boundary_config=nested_boundary_config,
        use_vertical_solver=True,
        use_flux_advection=True,
        force_fp64=True,
        diff_6th_opt=2,
        diff_6th_factor=0.12,
        w_damping=1,
        damp_opt=3,
        zdamp=5000.0,
        dampcoef=0.2,
        epssm=0.5,
        top_lid=True,
    )
    run_start = _coerce_run_start(str(replay.metadata["run_start_label"]))
    # v0.9.0 d02-replay stability fix carries to d03 (qke-fix follow-up):
    #   (1) STABILITY NAMELIST -- the OperationalNamelist above already routes the
    #       d03 forecast through the SAME validated operational Gen2 stability set
    #       as daily_pipeline._build_real_case / m7_l2_d02_replay.build_l2_daily_case
    #       (top_lid=True, epssm=0.5, w_damping=1, damp_opt=3, zdamp=5000,
    #       dampcoef=0.2, diff_6th_opt=2/0.12, use_flux_advection, force_fp64).  This
    #       builder has NEVER used the weak dataclass defaults; the d02-replay
    #       hour-1 blow-up cure (proofs/v090/d02replay_qke_fix_verify.json) is that
    #       same set, so d03 cannot hit the weak-namelist blow-up.
    #   (2) MYNN qke COLD-START SEED -- ``build_replay_case`` (called above) applies
    #       ``_wrf_mynn_coldstart_qke`` to the loaded d03 IC: when the parent carries
    #       no TKE (MAXVAL(qke)<0.0002, the corpus d03 t=0 case) it seeds WRF's
    #       ``mym_initialize`` background TKE profile, exactly as for d02.  The seed
    #       result is surfaced in ``qke_coldstart`` below so d03 proofs record it.
    qke_coldstart = replay.metadata.get("qke_coldstart", {})
    metadata = {
        "run_id": replay.metadata.get("run_id"),
        "run_dir": str(run_dir),
        "domain": config.domain,
        "grid": replay.metadata.get("grid", {}),
        "boundary": replay.metadata.get("boundary", {}),
        "qke_coldstart": qke_coldstart,
        "namelist": {
            "dt_s": float(namelist.dt_s),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "rk_order": int(namelist.rk_order),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "use_vertical_solver": bool(namelist.use_vertical_solver),
            "use_flux_advection": bool(namelist.use_flux_advection),
            "force_fp64": bool(namelist.force_fp64),
            "diff_6th_opt": int(namelist.diff_6th_opt),
            "diff_6th_factor": float(namelist.diff_6th_factor),
            "w_damping": int(namelist.w_damping),
            "damp_opt": int(namelist.damp_opt),
            "zdamp": float(namelist.zdamp),
            "dampcoef": float(namelist.dampcoef),
            "epssm": float(namelist.epssm),
            "top_lid": bool(namelist.top_lid),
            "disable_guards": bool(namelist.disable_guards),
        },
        # build_replay_case marks the source as the d02-replay loader so the
        # daily-pipeline hourly land-state refresh stays active for d03 too.
        "source": "gpuwrf.integration.d02_replay.build_replay_case",
        "d03_replay_adapter": {
            "domain": config.domain,
            "parent_domain": "d02",
            "expected_mass_shape_yx": list(EXPECTED_D03_MASS_SHAPE_YX),
            "actual_mass_shape_yx": list(actual_yx),
            "ic_source": "corpus L3 wrfout_d03 snapshot at t=0 plus wrfinput metrics/land state",
            "boundary_source": "parent d02 hourly wrfout interpolated to child d03 side strips",
            "operational_numerics": "Sprint-U _build_real_case namelist (1km CFL dt)",
        },
    }
    return (
        DailyCase(
            state=state,
            grid=replay.grid,
            namelist=namelist,
            run_start=run_start,
            metadata=metadata,
        ),
        run_dir,
    )


def _read_surface(path: str | Path, fields: Iterable[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with Dataset(path, "r") as dataset:
        for name in fields:
            if name not in dataset.variables:
                continue
            variable = dataset.variables[name]
            data = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
            out[name] = np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.nanmean(diff * diff)))


def write_d03_validation(
    *,
    wrfout_files: list[Path],
    reference_run_dir: Path,
    proof_path: str | Path,
) -> dict[str, Any]:
    """Validate the GPU d03 forecast vs corpus 1km TF truth: RMSE + persistence skill.

    Persistence baseline = hold the t=0 corpus d03 surface state for every lead
    (a do-nothing forecast).  The GPU forecast must BEAT persistence to claim
    real forecast skill, not just a small absolute RMSE.
    """

    run = parse_run_id(reference_run_dir.name)
    reference_dir = Path(reference_run_dir)
    # t=0 corpus d03 surface state for the persistence baseline.
    init_files = sorted(reference_dir.glob("wrfout_d03_*"))
    if not init_files:
        raise FileNotFoundError(f"no wrfout_d03 files in {reference_dir}")
    surface_fields = ("T2", "U10", "V10", PRECIP_FIELD)
    persistence_state = _read_surface(init_files[0], surface_fields)

    per_lead: list[dict[str, Any]] = []
    failures: list[str] = []
    beats_persistence_all: dict[str, list[bool]] = {name: [] for name in surface_fields}

    for wrfout in wrfout_files:
        valid_time = parse_wrfout_valid_time(wrfout).isoformat()
        # Match the corpus d03 truth file for this valid time.
        truth_name = wrfout.name  # GPU writer uses wrfout_d03_<valid> naming.
        truth_path = reference_dir / truth_name
        if not truth_path.is_file():
            failures.append(f"missing corpus d03 truth for {truth_name}")
            continue
        gpu = _read_surface(wrfout, surface_fields)
        truth = _read_surface(truth_path, surface_fields)
        lead_record: dict[str, Any] = {"valid_time_utc": valid_time, "fields": {}}
        for name in surface_fields:
            if name not in gpu or name not in truth:
                continue
            gpu_field = gpu[name]
            truth_field = truth[name]
            if gpu_field.shape != truth_field.shape:
                failures.append(f"{name} shape mismatch {gpu_field.shape} vs {truth_field.shape} at {valid_time}")
                continue
            rmse = _rmse(gpu_field, truth_field)
            pers = persistence_state.get(name)
            pers_rmse = _rmse(pers, truth_field) if pers is not None and pers.shape == truth_field.shape else None
            beats = bool(pers_rmse is not None and rmse <= pers_rmse)
            beats_persistence_all[name].append(beats)
            threshold = RMSE_THRESHOLDS.get(name)
            passed = bool(np.isfinite(rmse) and (threshold is None or rmse <= threshold))
            lead_record["fields"][name] = {
                "rmse": rmse,
                "persistence_rmse": pers_rmse,
                "beats_persistence": beats,
                "skill_score": (float(1.0 - rmse / pers_rmse) if pers_rmse not in (None, 0.0) else None),
                "threshold": threshold,
                "within_threshold": passed,
                "units": "K" if name == "T2" else ("mm" if name == PRECIP_FIELD else "m s-1"),
                "mean_error": float(np.nanmean(gpu_field - truth_field)),
                "max_abs_error": float(np.nanmax(np.abs(gpu_field - truth_field))),
            }
        per_lead.append(lead_record)

    if not per_lead:
        failures.append("no leads validated")

    # Final-lead summary (the operational gate) + over-run aggregates.
    final = per_lead[-1] if per_lead else {"fields": {}}
    final_fields = final.get("fields", {})
    threshold_pass = all(
        rec.get("within_threshold", False)
        for name, rec in final_fields.items()
        if name in RMSE_THRESHOLDS
    ) and bool(final_fields)
    persistence_pass = {
        name: (bool(values) and all(values)) for name, values in beats_persistence_all.items()
    }
    if not threshold_pass:
        failures.append("final-lead RMSE exceeded threshold on T2/U10/V10")

    status = "PASS" if not failures and threshold_pass else "FAIL"
    payload = {
        "schema": "V010D03ReplayValidation",
        "schema_version": 1,
        "status": status,
        "reference_run_dir": str(reference_dir),
        "run_id": reference_dir.name,
        "cycle": {"start_date": run.get("start_date"), "cycle_hour_utc": run.get("cycle_hour_utc")},
        "lead_count": len(per_lead),
        "final_lead": final.get("valid_time_utc"),
        "final_lead_fields": final_fields,
        "per_lead": per_lead,
        "persistence_beat_all_leads": persistence_pass,
        "thresholds": RMSE_THRESHOLDS,
        "failures": failures,
        "notes": [
            "RMSE is gridded GPU-d03 vs corpus-d03 (1km Tenerife) truth.",
            "Persistence baseline holds the t=0 corpus d03 surface state at every lead.",
            "beats_persistence requires GPU RMSE <= persistence RMSE for that field/lead.",
        ],
    }
    _write_json(proof_path, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=L3_RUN_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--dt-s", type=float, default=D03_DT_S)
    parser.add_argument("--acoustic-substeps", type=int, default=D03_ACOUSTIC_SUBSTEPS)
    parser.add_argument("--radiation-cadence-steps", type=int, default=D03_RADIATION_CADENCE_STEPS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--proof-dir", type=Path, default=PROOF_DIR)
    parser.add_argument("--tag", default=None, help="optional suffix for output/proof file names")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    affinity = _pin_orchestration_cpus()
    proof_dir = Path(args.proof_dir)
    proof_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    run_dir = resolve_run_dir(args.run_id, args.run_root)
    output_dir = Path(args.output_root) / f"d03_{args.run_id}{tag}"
    config = DailyPipelineConfig(
        run_id=args.run_id,
        hours=int(args.hours),
        output_dir=output_dir,
        proof_dir=proof_dir,
        run_root=Path(args.run_root),
        score=False,
        domain="d03",
        dt_s=float(args.dt_s),
        acoustic_substeps=int(args.acoustic_substeps),
        radiation_cadence_steps=int(args.radiation_cadence_steps),
    )

    pipeline_payload = execute_daily_pipeline(config, case_builder=build_l3_d03_daily_case)
    if affinity is not None:
        pipeline_payload["orchestration_cpu_affinity"] = affinity
    write_json(proof_dir / f"pipeline_run_d03{tag}.json", pipeline_payload)

    wrfout_files = [Path(p) for p in pipeline_payload.get("wrfout_files", [])]
    if pipeline_payload.get("verdict") == "PIPELINE_BLOCKED" or not wrfout_files:
        reason = pipeline_payload.get("reason", "pipeline did not produce wrfouts")
        validation = {
            "schema": "V010D03ReplayValidation",
            "schema_version": 1,
            "status": "BLOCKED",
            "reason": str(reason),
            "detail": pipeline_payload,
        }
        _write_json(proof_dir / f"d03_validation{tag}.json", validation)
        verdict = "D03_1KM_BLOCKED"
    else:
        try:
            validation = write_d03_validation(
                wrfout_files=wrfout_files,
                reference_run_dir=run_dir,
                proof_path=proof_dir / f"d03_validation{tag}.json",
            )
            verdict = "D03_1KM_VALIDATED" if validation.get("status") == "PASS" else "D03_1KM_BOUNDED_FAIL"
        except Exception as exc:  # noqa: BLE001 -- honest failure capture for the proof
            validation = {
                "schema": "V010D03ReplayValidation",
                "schema_version": 1,
                "status": "BLOCKED",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            _write_json(proof_dir / f"d03_validation{tag}.json", validation)
            verdict = "D03_1KM_BLOCKED"

    summary = {
        "schema": "V010D03ReplayValidationSummary",
        "schema_version": 1,
        "verdict": verdict,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "hours": int(args.hours),
        "dt_s": float(args.dt_s),
        "device": pipeline_payload.get("device"),
        "wall_clock_total_s": pipeline_payload.get("wall_clock_total_s"),
        "wall_clock_per_forecast_hour_s": pipeline_payload.get("wall_clock_per_forecast_hour_s"),
        "pipeline_verdict": pipeline_payload.get("verdict"),
        "all_finite": (pipeline_payload.get("all_finite_check") or {}).get("all_finite"),
        "validation_status": validation.get("status"),
        "final_lead_fields": validation.get("final_lead_fields"),
        "persistence_beat_all_leads": validation.get("persistence_beat_all_leads"),
        "proofs": {
            "pipeline": str(proof_dir / f"pipeline_run_d03{tag}.json"),
            "validation": str(proof_dir / f"d03_validation{tag}.json"),
        },
    }
    _write_json(proof_dir / f"d03_summary{tag}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verdict == "D03_1KM_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
