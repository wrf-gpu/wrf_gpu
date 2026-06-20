#!/usr/bin/env python
"""M7 L2 d02 replay-validation orchestrator."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping
import sys

import numpy as np
from netCDF4 import Dataset

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.integration.d02_replay import build_l2_d02_replay_case  # noqa: E402
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


SPRINT_DIR = ROOT / ".agent" / "sprints" / "2026-05-27-m7-l2-d02-replay-validation"
L2_RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2")
OUTPUT_ROOT = Path("/tmp/m7_pipeline_runs")
RMSE_THRESHOLDS = {"T2": 3.0, "U10": 7.5, "V10": 7.5}

# v0.9.0 validation-burst: precision mode for the forecast namelist.
# True  -> full fp64 (the prior gate mode).
# False -> ADR-007 gated-fp32 (theta/u/v/qv fp32; mu/p/ph/w + acoustic/pressure
#          accumulators fp64) -- the OPERATIONAL SHIP mode.  Selected via
#          --gated-fp32 on the CLI; read by build_l2_daily_case (whose signature
#          is fixed by execute_daily_pipeline's case_builder contract).
_FORCE_FP64 = True


def _pin_orchestration_cpus() -> list[int] | None:
    if not hasattr(os, "sched_setaffinity"):
        return None
    cpus = {0, 1, 2, 3}
    try:
        os.sched_setaffinity(0, cpus)
    except OSError:
        pass
    return sorted(os.sched_getaffinity(0))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _time_range(files: list[Path]) -> dict[str, Any]:
    times: list[datetime] = []
    for path in files:
        try:
            times.append(parse_wrfout_valid_time(path))
        except ValueError:
            pass
    times = sorted(times)
    if not times:
        return {"start": None, "end": None, "observed_hours": 0, "valid_times_utc": []}
    return {
        "start": times[0].isoformat(),
        "end": times[-1].isoformat(),
        "observed_hours": int((times[-1] - times[0]).total_seconds() // 3600),
        "valid_times_utc": [time.isoformat() for time in times],
    }


def _expected_times(start: datetime | None, expected_hours: int | None) -> list[datetime]:
    if start is None or expected_hours is None:
        return []
    return [start + timedelta(hours=hour) for hour in range(int(expected_hours) + 1)]


def _parse_start_from_run_id(run_id: str) -> datetime | None:
    parsed = parse_run_id(run_id)
    if parsed["start_date"] is None or parsed["cycle_hour_utc"] is None:
        return None
    return datetime.fromisoformat(f"{parsed['start_date']}T{int(parsed['cycle_hour_utc']):02d}:00:00+00:00")


def _grid_shape(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with Dataset(path, "r") as dataset:
        dims = dataset.dimensions
        shape = {
            "mass_shape": [
                int(len(dims["bottom_top"])) if "bottom_top" in dims else None,
                int(len(dims["south_north"])) if "south_north" in dims else None,
                int(len(dims["west_east"])) if "west_east" in dims else None,
            ],
            "staggered_shape": [
                int(len(dims["bottom_top_stag"])) if "bottom_top_stag" in dims else None,
                int(len(dims["south_north_stag"])) if "south_north_stag" in dims else None,
                int(len(dims["west_east_stag"])) if "west_east_stag" in dims else None,
            ],
        }
        attrs = {}
        for name in ("DX", "DY", "MAP_PROJ", "CEN_LAT", "CEN_LON"):
            if hasattr(dataset, name):
                value = getattr(dataset, name)
                attrs[name] = value.item() if hasattr(value, "item") else value
        shape["global_attributes"] = attrs
        return shape


def inventory_l2_run(run_dir: str | Path, *, requested_hours: int = 24) -> dict[str, Any]:
    path = Path(run_dir)
    d01_files = sorted(path.glob("wrfout_d01_*"), key=lambda item: item.name)
    d02_files = sorted(path.glob("wrfout_d02_*"), key=lambda item: item.name)
    parsed = parse_run_id(path.name)
    expected_hours = parsed.get("forecast_hours_advertised")
    d01_range = _time_range(d01_files)
    d02_range = _time_range(d02_files)
    start = None
    if d02_range["start"]:
        start = datetime.fromisoformat(str(d02_range["start"]))
    elif d01_range["start"]:
        start = datetime.fromisoformat(str(d01_range["start"]))
    else:
        start = _parse_start_from_run_id(path.name)
    if expected_hours is None:
        expected_hours = max(int(d01_range["observed_hours"]), int(d02_range["observed_hours"])) if start else None

    expected = _expected_times(start, expected_hours)
    d01_observed = {parse_wrfout_valid_time(file) for file in d01_files if file.name.startswith("wrfout_d01_")}
    d02_observed = {parse_wrfout_valid_time(file) for file in d02_files if file.name.startswith("wrfout_d02_")}
    missing_d01 = [time for time in expected if time not in d01_observed]
    missing_d02 = [time for time in expected if time not in d02_observed]
    requested_count = int(requested_hours) + 1
    complete_for_requested_hours = (
        len(d01_files) >= requested_count
        and len(d02_files) >= requested_count
        and int(d01_range["observed_hours"]) >= int(requested_hours)
        and int(d02_range["observed_hours"]) >= int(requested_hours)
    )
    complete_full = bool(expected_hours is not None and not missing_d01 and not missing_d02 and d01_files and d02_files)
    first_d01 = d01_files[0] if d01_files else path / "wrfinput_d01"
    first_d02 = d02_files[0] if d02_files else path / "wrfinput_d02"
    return {
        "run_id": path.name,
        "run_path": str(path),
        "d01_file_count": int(len(d01_files)),
        "d02_file_count": int(len(d02_files)),
        "expected_hours": int(expected_hours) if expected_hours is not None else None,
        "expected_file_count": int(expected_hours + 1) if expected_hours is not None else None,
        "requested_hours": int(requested_hours),
        "complete_for_requested_hours": bool(complete_for_requested_hours),
        "complete_full": bool(complete_full),
        "grid_shapes": {
            "d01": _grid_shape(first_d01),
            "d02": _grid_shape(first_d02),
        },
        "time_coverage": {
            "d01": d01_range,
            "d02": d02_range,
        },
        "missing": {
            "d01_count": int(len(missing_d01)),
            "d02_count": int(len(missing_d02)),
            "d01_valid_times_utc": [time.isoformat() for time in missing_d01],
            "d02_valid_times_utc": [time.isoformat() for time in missing_d02],
        },
        "first_files": {
            "d01": d01_files[0].name if d01_files else None,
            "d02": d02_files[0].name if d02_files else None,
        },
        "last_files": {
            "d01": d01_files[-1].name if d01_files else None,
            "d02": d02_files[-1].name if d02_files else None,
        },
    }


def build_l2_inventory(root: str | Path = L2_RUN_ROOT, *, requested_hours: int = 24) -> dict[str, Any]:
    base = Path(root)
    runs = [
        inventory_l2_run(path, requested_hours=requested_hours)
        for path in sorted(base.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ] if base.exists() else []
    return {
        "schema": "M7L2D02ReplayInventory",
        "schema_version": 1,
        "root": str(base),
        "requested_hours": int(requested_hours),
        "run_count": int(len(runs)),
        "complete_full_run_count": int(sum(1 for run in runs if run["complete_full"])),
        "complete_for_requested_hours_count": int(sum(1 for run in runs if run["complete_for_requested_hours"])),
        "complete_run_ids": [run["run_id"] for run in runs if run["complete_full"]],
        "requested_hours_run_ids": [run["run_id"] for run in runs if run["complete_for_requested_hours"]],
        "runs": runs,
    }


def select_l2_run(inventory: Mapping[str, Any], *, requested_hours: int = 24) -> str:
    runs = list(inventory.get("runs", []))
    full = [run for run in runs if run.get("complete_full") and run.get("complete_for_requested_hours")]
    if full:
        return sorted(full, key=lambda run: run["run_id"])[-1]["run_id"]
    requested = [run for run in runs if run.get("complete_for_requested_hours")]
    if requested:
        return sorted(requested, key=lambda run: run["run_id"])[-1]["run_id"]
    raise RuntimeError(f"no L2 run has both d01 and d02 wrfout coverage for {requested_hours}h")


def _coerce_run_start(value: str) -> datetime:
    text = value.strip().replace("Z", "")
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def build_l2_daily_case(config: DailyPipelineConfig) -> tuple[DailyCase, Path]:
    run_dir = resolve_run_dir(config.run_id, config.run_root)
    replay = build_l2_d02_replay_case(run_dir, domain=config.domain, parent_domain="d01")
    state = replay.state.replace(p=replay.state.p_total, ph=replay.state.ph_total, mu=replay.state.mu_total)
    # v0.9.0 d02-replay hour-1 stability hardening (fix-B): route this harness to
    # the SAME validated operational Gen2-d02 stability namelist that the
    # v0.1.0-validated path daily_pipeline._build_real_case carries.  This harness
    # previously built the forecast namelist with EVERY stability flag at its
    # dataclass default -- top_lid=False (OPEN TOP), epssm=0.1, no w/Rayleigh
    # damping, no 6th-order filter, legacy primitive advection, fp32 -- i.e.
    # STRICTLY WEAKER than the documented-unstable open-top real-init case
    # (proofs/dycore_realinit/step5_opentop_bndy.json).  See _build_real_case for
    # the per-flag root-cause annotations.  Independent of (and complementary to)
    # the MYNN qke cold-start seed in d02_replay.build_replay_case.
    namelist = OperationalNamelist.from_grid(
        replay.grid,
        tendencies=replay.tendencies,
        metrics=replay.metrics,
        dt_s=float(config.dt_s),
        acoustic_substeps=int(config.acoustic_substeps),
        radiation_cadence_steps=int(config.radiation_cadence_steps),
        use_vertical_solver=True,
        use_flux_advection=True,
        force_fp64=bool(_FORCE_FP64),
        diff_6th_opt=2,
        diff_6th_factor=0.12,
        w_damping=1,
        damp_opt=3,
        zdamp=5000.0,
        dampcoef=0.2,
        epssm=0.5,
        top_lid=True,
    )
    metadata = {
        "run_id": replay.metadata.get("run_id"),
        "run_dir": str(run_dir),
        "domain": config.domain,
        "grid": replay.metadata.get("grid", {}),
        "boundary": replay.metadata.get("boundary", {}),
        "l2_replay_adapter": replay.metadata.get("l2_replay_adapter", {}),
        "qke_coldstart": replay.metadata.get("qke_coldstart", {}),
        "namelist": {
            "dt_s": float(namelist.dt_s),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "rk_order": int(namelist.rk_order),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "use_vertical_solver": bool(namelist.use_vertical_solver),
            # v0.9.0 stability-namelist hardening (matches _build_real_case):
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
        },
        "source": "gpuwrf.integration.d02_replay.build_l2_d02_replay_case",
    }
    return (
        DailyCase(
            state=state,
            grid=replay.grid,
            namelist=namelist,
            run_start=_coerce_run_start(str(replay.metadata["run_start_label"])),
            metadata=metadata,
        ),
        run_dir,
    )


def _read_time_squeezed(dataset: Dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    data = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
    return np.asarray(np.ma.filled(data, np.nan))


def _surface_namespace(path: str | Path) -> SimpleNamespace:
    with Dataset(path, "r") as dataset:
        return SimpleNamespace(
            T2=_read_time_squeezed(dataset, "T2"),
            U10=_read_time_squeezed(dataset, "U10"),
            V10=_read_time_squeezed(dataset, "V10"),
        )


def write_tier4_rmse(
    *,
    final_wrfout: str | Path,
    reference_run_dir: str | Path,
    proof_path: str | Path,
) -> dict[str, Any]:
    valid_time = parse_wrfout_valid_time(final_wrfout).isoformat()
    state = _surface_namespace(final_wrfout)
    raw = compute_rmse_against_gen2(state, reference_run_dir, valid_time, fields=("T2", "U10", "V10"))
    fields: dict[str, Any] = {}
    failures: list[str] = []
    for name, result in raw.items():
        error_map = np.asarray(result["error_map"], dtype=np.float64)
        rmse = float(result["rmse"])
        threshold = float(RMSE_THRESHOLDS[name])
        passed = bool(np.isfinite(rmse) and rmse <= threshold)
        if not passed:
            failures.append(f"{name} rmse={rmse:g} > {threshold:g}")
        fields[name] = {
            "rmse": rmse,
            "threshold": threshold,
            "units": "K" if name == "T2" else "m s-1",
            "pass": passed,
            "max_abs_error": float(np.nanmax(np.abs(error_map))),
            "mean_error": float(np.nanmean(error_map)),
            "shape": list(error_map.shape),
            "valid_time_utc": result["valid_time_utc"],
            "gen2_source_file": result["gen2_source_file"],
        }
    payload = {
        "schema": "M7L2D02Tier4RMSE",
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "final_wrfout": str(final_wrfout),
        "reference_run_dir": str(reference_run_dir),
        "valid_time_utc": valid_time,
        "fields": fields,
        "failures": failures,
    }
    _write_json(proof_path, payload)
    return payload


def _iter_numeric_variables(dataset: Dataset) -> Iterable[tuple[str, np.ndarray]]:
    for name, variable in dataset.variables.items():
        if np.dtype(variable.dtype).kind not in {"b", "i", "u", "f", "c"}:
            continue
        yield name, _read_time_squeezed(dataset, name)


def write_bounds_check(*, wrfout_files: list[str | Path], proof_path: str | Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    failures: list[str] = []
    aggregate = {
        "theta_min_k": None,
        "theta_max_k": None,
        "theta_lower_30_min_k": None,
        "theta_lower_30_max_k": None,
        "u_abs_max_m_s": None,
        "v_abs_max_m_s": None,
        "w_abs_max_m_s": None,
    }

    def update_agg(key: str, value: float, op) -> None:
        current = aggregate[key]
        aggregate[key] = value if current is None else op(float(current), value)

    for item in wrfout_files:
        path = Path(item)
        with Dataset(path, "r") as dataset:
            nonfinite: dict[str, int] = {}
            numeric_count = 0
            for name, values in _iter_numeric_variables(dataset):
                numeric_count += 1
                finite = np.isfinite(values)
                if not bool(finite.all()):
                    nonfinite[name] = int(values.size - int(finite.sum()))
            theta = _read_time_squeezed(dataset, "T").astype(np.float64) + 300.0
            lower_levels = theta[: min(30, theta.shape[0]), :, :]
            u = _read_time_squeezed(dataset, "U").astype(np.float64)
            v = _read_time_squeezed(dataset, "V").astype(np.float64)
            w = _read_time_squeezed(dataset, "W").astype(np.float64)
        theta_min = float(np.nanmin(theta))
        theta_max = float(np.nanmax(theta))
        lower_min = float(np.nanmin(lower_levels))
        lower_max = float(np.nanmax(lower_levels))
        u_abs = float(np.nanmax(np.abs(u)))
        v_abs = float(np.nanmax(np.abs(v)))
        w_abs = float(np.nanmax(np.abs(w)))
        update_agg("theta_min_k", theta_min, min)
        update_agg("theta_max_k", theta_max, max)
        update_agg("theta_lower_30_min_k", lower_min, min)
        update_agg("theta_lower_30_max_k", lower_max, max)
        update_agg("u_abs_max_m_s", u_abs, max)
        update_agg("v_abs_max_m_s", v_abs, max)
        update_agg("w_abs_max_m_s", w_abs, max)
        record = {
            "path": str(path),
            "valid_time_utc": parse_wrfout_valid_time(path).isoformat(),
            "numeric_variable_count": int(numeric_count),
            "all_numeric_fields_finite": not nonfinite,
            "nonfinite_counts": nonfinite,
            "theta_min_k": theta_min,
            "theta_max_k": theta_max,
            "theta_lower_30_min_k": lower_min,
            "theta_lower_30_max_k": lower_max,
            "u_abs_max_m_s": u_abs,
            "v_abs_max_m_s": v_abs,
            "w_abs_max_m_s": w_abs,
        }
        if nonfinite:
            failures.append(f"{path.name} contains nonfinite numeric fields")
        files.append(record)

    theta_broad = aggregate["theta_min_k"] is not None and 150.0 <= float(aggregate["theta_min_k"]) <= float(aggregate["theta_max_k"]) <= 550.0
    theta_lower = (
        aggregate["theta_lower_30_min_k"] is not None
        and 200.0 <= float(aggregate["theta_lower_30_min_k"])
        and float(aggregate["theta_lower_30_max_k"]) <= 400.0
    )
    wind = all(
        aggregate[key] is not None and float(aggregate[key]) <= 150.0
        for key in ("u_abs_max_m_s", "v_abs_max_m_s", "w_abs_max_m_s")
    )
    if not theta_broad:
        failures.append("theta broad bounds 150..550 K failed")
    if not theta_lower:
        failures.append("theta lower-30-level bounds 200..400 K failed")
    if not wind:
        failures.append("wind absolute bound <=150 m s-1 failed")
    payload = {
        "schema": "M7L2D02BoundsCheck",
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "bounds_policy": {
            "theta_broad_k": [150.0, 550.0],
            "theta_lower_30_levels_k": [200.0, 400.0],
            "wind_abs_max_m_s": 150.0,
        },
        "aggregate": aggregate,
        "file_count": int(len(files)),
        "files": files,
        "failures": failures,
    }
    _write_json(proof_path, payload)
    return payload


def write_wall_clock(
    *,
    pipeline_payload: Mapping[str, Any],
    proof_path: str | Path,
    run_dir: str | Path,
    affinity: list[int] | None,
) -> dict[str, Any]:
    payload = {
        "schema": "M7L2D02WallClock",
        "schema_version": 1,
        "status": "PASS" if pipeline_payload.get("wall_clock_total_s") is not None else "BLOCKED",
        "run_id": pipeline_payload.get("run_id"),
        "run_dir": str(run_dir),
        "hours": pipeline_payload.get("hours"),
        "device": pipeline_payload.get("device"),
        "orchestration_cpu_affinity": affinity,
        "wall_clock_total_s": pipeline_payload.get("wall_clock_total_s"),
        "wall_clock_forecast_only_s": pipeline_payload.get("wall_clock_forecast_only_s"),
        "wall_clock_per_hour_s": pipeline_payload.get("wall_clock_per_hour_s"),
        "wall_clock_per_forecast_hour_s": pipeline_payload.get("wall_clock_per_forecast_hour_s"),
        "speedup_status": pipeline_payload.get("speedup_status"),
        "pipeline_verdict": pipeline_payload.get("verdict"),
        "output_dir": pipeline_payload.get("output_dir"),
    }
    _write_json(proof_path, payload)
    return payload


def _write_blocked_proofs(proof_dir: Path, *, reason: str, detail: Mapping[str, Any]) -> None:
    for name, schema in (
        ("tier4_rmse_l2_d02.json", "M7L2D02Tier4RMSE"),
        ("bounds_check_l2_d02.json", "M7L2D02BoundsCheck"),
        ("wall_clock_l2_d02.json", "M7L2D02WallClock"),
    ):
        _write_json(
            proof_dir / name,
            {
                "schema": schema,
                "schema_version": 1,
                "status": "BLOCKED",
                "reason": reason,
                "detail": dict(detail),
            },
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=L2_RUN_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--proof-dir", type=Path, default=SPRINT_DIR)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--gated-fp32",
        action="store_true",
        help="Run the forecast in ADR-007 gated-fp32 (theta/u/v/qv fp32; mu/p/ph/w + "
        "acoustic/pressure accumulators fp64) -- the OPERATIONAL SHIP mode. Default is full fp64.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global _FORCE_FP64
    args = parse_args(argv)
    if getattr(args, "gated_fp32", False):
        _FORCE_FP64 = False
    affinity = _pin_orchestration_cpus()
    proof_dir = Path(args.proof_dir)
    proof_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_l2_inventory(args.run_root, requested_hours=int(args.hours))
    inventory_path = proof_dir / "l2_inventory.json"
    _write_json(inventory_path, inventory)
    if args.inventory_only:
        print(json.dumps({"inventory": str(inventory_path), "run_count": inventory["run_count"]}, indent=2))
        return 0

    run_id = args.run_id or select_l2_run(inventory, requested_hours=int(args.hours))
    run_dir = resolve_run_dir(run_id, args.run_root)
    output_dir = Path(args.output_root) / f"l2_d02_{run_id}"
    config = DailyPipelineConfig(
        run_id=run_id,
        hours=int(args.hours),
        output_dir=output_dir,
        proof_dir=proof_dir,
        run_root=Path(args.run_root),
        score=False,
        domain="d02",
    )

    pipeline_payload = execute_daily_pipeline(config, case_builder=build_l2_daily_case)
    if affinity is not None:
        pipeline_payload["orchestration_cpu_affinity"] = affinity
        write_json(proof_dir / "pipeline_run_l2_d02.json", pipeline_payload)

    try:
        wrfout_files = [Path(path) for path in pipeline_payload.get("wrfout_files", [])]
        if pipeline_payload.get("verdict") == "PIPELINE_BLOCKED" or not wrfout_files:
            reason = pipeline_payload.get("reason", "pipeline did not produce wrfouts")
            _write_blocked_proofs(proof_dir, reason=str(reason), detail=pipeline_payload)
            verdict = "L2_D02_BLOCKED"
            rmse = {"status": "BLOCKED"}
            bounds = {"status": "BLOCKED"}
            wall = {"status": "BLOCKED"}
        else:
            final_wrfout = wrfout_files[-1]
            rmse = write_tier4_rmse(
                final_wrfout=final_wrfout,
                reference_run_dir=run_dir,
                proof_path=proof_dir / "tier4_rmse_l2_d02.json",
            )
            bounds = write_bounds_check(
                wrfout_files=wrfout_files,
                proof_path=proof_dir / "bounds_check_l2_d02.json",
            )
            wall = write_wall_clock(
                pipeline_payload=pipeline_payload,
                proof_path=proof_dir / "wall_clock_l2_d02.json",
                run_dir=run_dir,
                affinity=affinity,
            )
            verdict = "L2_D02_GREEN" if rmse["status"] == "PASS" and bounds["status"] == "PASS" else "L2_D02_BOUNDED_FAIL"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        detail = {"pipeline_payload": dict(pipeline_payload), "run_dir": str(run_dir)}
        _write_blocked_proofs(proof_dir, reason=reason, detail=detail)
        verdict = "L2_D02_BLOCKED"
        rmse = {"status": "BLOCKED", "reason": reason}
        bounds = {"status": "BLOCKED", "reason": reason}
        wall = {"status": "BLOCKED", "reason": reason}

    summary = {
        "schema": "M7L2D02ReplayValidationSummary",
        "schema_version": 1,
        "verdict": verdict,
        "publish_readiness": (
            "nested-grid backfills are publish-ready for this L2 d02 path"
            if verdict == "L2_D02_GREEN"
            else "nested-grid backfills are not publish-ready from this sprint evidence"
        ),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "proofs": {
            "inventory": str(inventory_path),
            "tier4_rmse": str(proof_dir / "tier4_rmse_l2_d02.json"),
            "bounds": str(proof_dir / "bounds_check_l2_d02.json"),
            "wall_clock": str(proof_dir / "wall_clock_l2_d02.json"),
        },
        "statuses": {
            "pipeline": pipeline_payload.get("verdict"),
            "rmse": rmse.get("status"),
            "bounds": bounds.get("status"),
            "wall_clock": wall.get("status"),
        },
    }
    _write_json(proof_dir / "l2_d02_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verdict == "L2_D02_GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
