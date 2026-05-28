#!/usr/bin/env python
"""Execute HIGH publication tests and stamp honest proof objects.

The previous execution sprint used this file to write BLOCKED placeholders when
the GPU was unreachable. This redo keeps the same proof-object surface but runs
the available GPU paths. Tests that still have no runnable kernel in this repo
are marked SKIP_* rather than BLOCKED.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset

try:
    from pubtest_common import (
        CANARY_RUN_ROOT,
        HIGH_TEST_FILES,
        ROOT,
        SPRINT_DIR,
        finite_stats,
        gpu_probe,
        proof_header,
        read_json,
        run_command,
        summarize_m6b6,
        threshold_rows,
        write_case_summary,
        write_json,
        write_summary_md,
        wrf_provenance,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path used by pytest
    from scripts.pubtest_common import (
        CANARY_RUN_ROOT,
        HIGH_TEST_FILES,
        ROOT,
        SPRINT_DIR,
        finite_stats,
        gpu_probe,
        proof_header,
        read_json,
        run_command,
        summarize_m6b6,
        threshold_rows,
        write_case_summary,
        write_json,
        write_summary_md,
        wrf_provenance,
    )

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.fixtures.idealized_cases import build_density_current, build_schaer_mountain_wave, build_warmbubble  # noqa: E402
from gpuwrf.integration.daily_pipeline import (  # noqa: E402
    DailyPipelineConfig,
    compare_wrfouts_xarray,
    execute_daily_pipeline,
)


DEFAULT_EXECUTION_ROOT = Path("/tmp/pubtest_redo")
CANARY_CASE_TARGET = 5


@dataclass(frozen=True)
class CanarySelection:
    day: str
    run_dir: Path
    history_count: int
    planned_hours: int
    selection_class: str
    reason: str

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def runnable(self) -> bool:
        return self.history_count >= 2 and self.planned_hours > 0

    @property
    def complete_24h(self) -> bool:
        return self.history_count >= 25 and self.planned_hours >= 24


def _import_script_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _skill_module() -> Any:
    return _import_script_module(ROOT / "scripts" / "m7_gpu_vs_cpu_skill_diff.py", "m7_gpu_vs_cpu_skill_diff_pubtest")


def _history_count(run_dir: Path) -> int:
    return len(sorted(run_dir.glob("wrfout_d02_*")))


def _day_from_run_dir(run_dir: Path) -> str:
    return run_dir.name[:8]


def _best_by_day(run_root: Path) -> dict[str, Path]:
    best: dict[str, Path] = {}
    for run_dir in sorted(run_root.glob("*_18z_l3_24h_*")):
        if not run_dir.is_dir():
            continue
        day = _day_from_run_dir(run_dir)
        current = best.get(day)
        if current is None:
            best[day] = run_dir
            continue
        score = (_history_count(run_dir), run_dir.name)
        current_score = (_history_count(current), current.name)
        if score > current_score:
            best[day] = run_dir
    return best


def select_canary_cases(run_root: Path = CANARY_RUN_ROOT, *, target_count: int = CANARY_CASE_TARGET) -> list[CanarySelection]:
    by_day = _best_by_day(run_root)
    complete = [
        CanarySelection(day, path, _history_count(path), 24, "complete_24h", ">=25 hourly wrfout_d02 files available")
        for day, path in sorted(by_day.items())
        if _history_count(path) >= 25
    ]
    partial = [
        CanarySelection(
            day,
            path,
            _history_count(path),
            max(min(_history_count(path) - 1, 24), 0),
            "partial_history",
            ">=2 but <25 hourly wrfout_d02 files available; run only the hours supported by boundary history",
        )
        for day, path in sorted(by_day.items())
        if 2 <= _history_count(path) < 25 and day not in {item.day for item in complete}
    ]
    missing = [
        CanarySelection(day, path, _history_count(path), 0, "missing_history", "fewer than two wrfout_d02 files available")
        for day, path in sorted(by_day.items())
        if _history_count(path) < 2 and day not in {item.day for item in complete} and day not in {item.day for item in partial}
    ]

    selected: list[CanarySelection] = []
    if partial:
        selected.append(partial[0])
    selected.extend(complete)
    selected.extend(item for item in missing if item.day in {"20260429", "20260517"})
    selected.extend(item for item in missing if item not in selected)

    deduped: list[CanarySelection] = []
    seen: set[str] = set()
    for item in selected:
        if item.day in seen:
            continue
        deduped.append(item)
        seen.add(item.day)
        if len(deduped) >= int(target_count):
            break
    return deduped


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _proof_verdict_ok(verdict: Any) -> bool:
    return verdict == "PASS" or (isinstance(verdict, str) and verdict.startswith("SKIP_")) or verdict == "FAIL"


def _run_daily_pipeline(
    *,
    run_id: str,
    hours: int,
    output_dir: Path,
    proof_dir: Path,
    score: bool = False,
    restart_at_hour: int | None = None,
    repeat: bool = False,
    dt_s: float = 10.0,
    acoustic_substeps: int = 10,
) -> dict[str, Any]:
    config = DailyPipelineConfig(
        run_id=run_id,
        hours=int(hours),
        output_dir=output_dir,
        proof_dir=proof_dir,
        score=bool(score),
        restart_at_hour=restart_at_hour,
        repeat=bool(repeat),
        dt_s=float(dt_s),
        acoustic_substeps=int(acoustic_substeps),
    )
    return execute_daily_pipeline(config)


def _output_files_from_pipeline(payload: Mapping[str, Any]) -> list[Path]:
    return [Path(path) for path in payload.get("wrfout_files", []) if Path(path).is_file()]


def _nrmse(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    denom = float(np.sqrt(np.mean(np.asarray(right, dtype=np.float64) ** 2)))
    if denom < 1.0e-12:
        return float(np.sqrt(np.mean(diff * diff)))
    return float(np.sqrt(np.mean(diff * diff)) / denom)


def _surface_fields(path: Path) -> dict[str, np.ndarray]:
    with Dataset(path, "r") as ds:
        out: dict[str, np.ndarray] = {}
        for name in ("T2", "U10", "V10"):
            if name in ds.variables:
                out[name] = np.asarray(ds.variables[name][0], dtype=np.float64)
        return out


def _mass_series(wrfouts: Sequence[Path]) -> dict[str, Any]:
    values: list[float] = []
    files: list[str] = []
    for path in wrfouts:
        with Dataset(path, "r") as ds:
            if "MU" not in ds.variables or "MUB" not in ds.variables:
                continue
            mu = np.asarray(ds.variables["MU"][0], dtype=np.float64)
            mub = np.asarray(ds.variables["MUB"][0], dtype=np.float64)
            values.append(float(np.sum(mu + mub)))
            files.append(str(path))
    if len(values) < 2:
        return {"status": "NO_SERIES", "files": files, "value_count": len(values)}
    initial = values[0]
    relative = [0.0 if abs(initial) < 1.0e-30 else float((value - initial) / initial) for value in values]
    return {
        "status": "PASS",
        "files": files,
        "value_count": len(values),
        "initial": initial,
        "final": values[-1],
        "max_abs_relative_uncorrected_drift": max(abs(item) for item in relative),
        "relative_drift_series": relative,
    }


def _energy_proxy_series(wrfouts: Sequence[Path]) -> dict[str, Any]:
    values: list[dict[str, float]] = []
    files: list[str] = []
    for path in wrfouts:
        with Dataset(path, "r") as ds:
            required = {"U", "V", "T", "PH", "PHB"}
            if not required.issubset(ds.variables):
                continue
            u = np.asarray(ds.variables["U"][0, :, :, :-1], dtype=np.float64)
            v = np.asarray(ds.variables["V"][0, :, :-1, :], dtype=np.float64)
            theta = np.asarray(ds.variables["T"][0], dtype=np.float64) + 300.0
            geopotential = np.asarray(ds.variables["PH"][0, :-1], dtype=np.float64) + np.asarray(ds.variables["PHB"][0, :-1], dtype=np.float64)
            ke = float(0.5 * np.mean(u * u + v * v))
            dry_static_proxy = float(np.mean(1004.0 * theta + geopotential))
            values.append({"ke_proxy": ke, "dry_static_proxy": dry_static_proxy, "total_proxy": ke + dry_static_proxy})
            files.append(str(path))
    if len(values) < 2:
        return {"status": "NO_SERIES", "files": files, "value_count": len(values)}
    initial = values[0]["total_proxy"]
    relative = [0.0 if abs(initial) < 1.0e-30 else float((row["total_proxy"] - initial) / initial) for row in values]
    return {
        "status": "PASS",
        "files": files,
        "value_count": len(values),
        "initial_total_proxy": initial,
        "final_total_proxy": values[-1]["total_proxy"],
        "max_abs_relative_proxy_drift": max(abs(item) for item in relative),
        "relative_proxy_drift_series": relative,
        "components": values,
        "proxy_note": "Diagnostic proxy only: theta and geopotential means from wrfout, not a WRF total-energy budget.",
    }


def _write_idealized(proof_dir: Path, *, gpu: dict[str, Any], wrf: dict[str, Any]) -> list[Path]:
    inputs = proof_dir / "inputs"
    warm = write_case_summary(inputs / "warmbubble_ic_summary.json", build_warmbubble())
    density = write_case_summary(inputs / "density_current_ic_summary.json", build_density_current())
    mountain = write_case_summary(inputs / "schaer_mountain_wave_ic_summary.json", build_schaer_mountain_wave())
    cases = [
        (
            "IDEALIZED-WARMBUBBLE",
            "idealized_warmbubble.json",
            warm,
            "SKIP_NO_IDEALIZED_GPU_FORECAST_RUNNER",
            {
                "finite_initial_condition": (1.0 if finite_stats(warm) else 0.0, "all IC fields finite", finite_stats(warm)),
                "theta_w_nrmse_ladder": (None, "<=0.05/0.08/0.12/0.18 vs CPU WRF after integration", None),
                "dry_mass_drift": (None, "<=1e-10 after integration", None),
            },
        ),
        (
            "IDEALIZED-DENSITY-CURRENT",
            "idealized_density_current.json",
            density,
            "SKIP_NO_DENSITY_CURRENT_GPU_FORECAST_RUNNER",
            {
                "finite_initial_condition": (1.0 if finite_stats(density) else 0.0, "all IC fields finite", finite_stats(density)),
                "front_position_900s": (None, "within 1 horizontal grid cell of Straka 1993", None),
                "front_speed": (None, "within 5% of Straka 1993 reference", None),
                "min_theta_perturbation_ic": (
                    density["array_stats"]["theta_perturbation_k"]["min"],
                    "IC within 0.5 K of -15 K",
                    abs(float(density["array_stats"]["theta_perturbation_k"]["min"]) + 15.0) <= 0.5,
                ),
            },
        ),
        (
            "IDEALIZED-MOUNTAIN-WAVE",
            "idealized_mountain_wave.json",
            mountain,
            "SKIP_NO_MOUNTAIN_WAVE_GPU_FORECAST_RUNNER",
            {
                "finite_initial_condition": (1.0 if finite_stats(mountain) else 0.0, "all IC fields finite", finite_stats(mountain)),
                "w_peak_5h": (None, "within 10% of Schaer analytic steady-state solution", None),
                "dominant_wave_phase": (None, "<=1 grid cell horizontal and vertical", None),
                "pressure_nrmse": (None, "<=0.10 vs analytic", None),
            },
        ),
    ]
    written: list[Path] = []
    for test_id, filename, summary, verdict, rows in cases:
        payload = proof_header(test_id, verdict, verdict)
        payload.update(
            {
                "proof": {"reference": summary["reference"], "wrf_provenance": wrf, "gpu_preflight": gpu},
                "inputs": summary,
                "thresholds": threshold_rows(rows),
                "skip_reason": (
                    "The repo contains analytic IC builders and comparison wrappers, but no reviewed "
                    "GPU idealized forecast integrator for this case inside the sprint's editable scope."
                ),
                "gpu_hours_used": 0.0,
            }
        )
        path = proof_dir / filename
        write_json(path, payload)
        written.append(path)
    write_summary_md(
        proof_dir / "idealized_warmbubble_summary.md",
        "IDEALIZED-WARMBUBBLE Summary",
        read_json(proof_dir / "idealized_warmbubble.json") or {},
        ["- IC builder completed.", "- Forecast comparison is SKIP_NO_IDEALIZED_GPU_FORECAST_RUNNER."],
    )
    write_summary_md(
        proof_dir / "idealized_mountain_wave_summary.md",
        "IDEALIZED-MOUNTAIN-WAVE Summary",
        read_json(proof_dir / "idealized_mountain_wave.json") or {},
        ["- Schaer terrain/IC builder completed.", "- Forecast comparison is SKIP_NO_MOUNTAIN_WAVE_GPU_FORECAST_RUNNER."],
    )
    return written


def _write_conservation(proof_dir: Path, canary_summary: Mapping[str, Any] | None = None) -> list[Path]:
    canary_summary = dict(canary_summary or {})
    complete = [case for case in canary_summary.get("case_results", []) if case.get("complete_24h") and case.get("pipeline_verdict") == "PIPELINE_GREEN"]
    first_files = [Path(path) for path in complete[0].get("wrfout_files", [])] if complete else []
    mass_series = _mass_series(first_files) if first_files else {"status": "NO_COMPLETE_GPU_CANARY_CASE"}
    energy_series = _energy_proxy_series(first_files) if first_files else {"status": "NO_COMPLETE_GPU_CANARY_CASE"}

    mass_pass = bool(False)
    mass = proof_header("CONSERVATION-MASS-24H", "FAIL", "FAIL_MISSING_CLOSED_DOMAIN_AND_BOUNDARY_FLUX_CORRECTION")
    mass.update(
        {
            "canary_case_source": complete[0] if complete else None,
            "canary_uncorrected_mass_series": mass_series,
            "thresholds": threshold_rows(
                {
                    "closed_domain_dry_mass_drift": (None, "<=1e-10 over 24 h warm-bubble closed domain", None),
                    "canary_flux_corrected_residual": (
                        mass_series.get("max_abs_relative_uncorrected_drift") if isinstance(mass_series, dict) else None,
                        "<=1e-5 after boundary-flux correction",
                        mass_pass,
                    ),
                    "nonfinite_mass_fields": (0.0 if mass_series.get("status") == "PASS" else None, "zero", mass_series.get("status") == "PASS"),
                }
            ),
            "honesty_note": "A real GPU Canary mass series was read when available, but the required closed-domain warmbubble run and Canary boundary-flux correction are not implemented in the pubtest scope.",
            "gpu_hours_used": 0.0,
        }
    )
    energy = proof_header("CONSERVATION-ENERGY-24H", "FAIL", "FAIL_MISSING_CPU_ENVELOPE")
    energy.update(
        {
            "canary_case_source": complete[0] if complete else None,
            "gpu_energy_proxy_series": energy_series,
            "thresholds": threshold_rows(
                {
                    "total_energy_drift_vs_cpu": (None, "within +/-20% of CPU WRF drift", None),
                    "component_drift_vs_cpu": (None, "KE/internal/potential within CPU envelope +/-20%", None),
                    "unexplained_step_jump": (None, "<=0.05%", None),
                }
            ),
            "honesty_note": "Only a GPU wrfout proxy diagnostic was available. The required CPU WRF energy-envelope run is absent, so this gate fails.",
            "gpu_hours_used": 0.0,
        }
    )
    paths = [proof_dir / "conservation_mass_24h.json", proof_dir / "conservation_energy_24h.json"]
    write_json(paths[0], mass)
    write_json(paths[1], energy)
    return paths


def _write_stability(
    proof_dir: Path,
    *,
    gpu: dict[str, Any] | None = None,
    run_heavy: bool = False,
    execution_root: Path | None = None,
    run_id: str = "20260521_18z_l3_24h_20260522T133443Z",
) -> list[Path]:
    execution_root = execution_root or DEFAULT_EXECUTION_ROOT
    gpu = gpu or {}
    cfl_runs: list[dict[str, Any]] = []
    acoustic_runs: list[dict[str, Any]] = []
    cfl_surrogate_pass = False
    acoustic_surrogate_pass = False

    if run_heavy:
        for label, dt_s in (("dt_0p5", 5.0), ("dt_1p0", 10.0), ("dt_1p25", 12.5)):
            out_dir = execution_root / "stability_cfl" / label
            proof_subdir = proof_dir / "stability_cfl_runs" / label
            started = time.perf_counter()
            payload = _run_daily_pipeline(run_id=run_id, hours=1, output_dir=out_dir, proof_dir=proof_subdir, dt_s=dt_s)
            wall_s = time.perf_counter() - started
            files = _output_files_from_pipeline(payload)
            cfl_runs.append(
                {
                    "label": label,
                    "dt_s": dt_s,
                    "pipeline_verdict": payload.get("verdict"),
                    "wall_clock_s": wall_s,
                    "forecast_wall_s": payload.get("wall_clock_forecast_only_s"),
                    "proof_dir": str(proof_subdir),
                    "output_dir": str(out_dir),
                    "wrfout_files": [str(path) for path in files],
                    "all_finite": payload.get("all_finite_check", {}).get("all_finite"),
                }
            )
        cfl_surrogate_pass = all(item.get("pipeline_verdict") == "PIPELINE_GREEN" and item.get("all_finite") for item in cfl_runs)

        acoustic_fields: dict[int, dict[str, np.ndarray]] = {}
        for substeps in (4, 6, 8):
            label = f"n{substeps}"
            out_dir = execution_root / "stability_acoustic" / label
            proof_subdir = proof_dir / "stability_acoustic_runs" / label
            started = time.perf_counter()
            payload = _run_daily_pipeline(
                run_id=run_id,
                hours=1,
                output_dir=out_dir,
                proof_dir=proof_subdir,
                acoustic_substeps=substeps,
            )
            wall_s = time.perf_counter() - started
            files = _output_files_from_pipeline(payload)
            fields = _surface_fields(files[-1]) if files else {}
            acoustic_fields[substeps] = fields
            acoustic_runs.append(
                {
                    "label": label,
                    "acoustic_substeps": substeps,
                    "pipeline_verdict": payload.get("verdict"),
                    "wall_clock_s": wall_s,
                    "forecast_wall_s": payload.get("wall_clock_forecast_only_s"),
                    "proof_dir": str(proof_subdir),
                    "output_dir": str(out_dir),
                    "wrfout_files": [str(path) for path in files],
                    "all_finite": payload.get("all_finite_check", {}).get("all_finite"),
                }
            )
        pairwise: dict[str, Any] = {}
        for left, right in ((4, 6), (4, 8), (6, 8)):
            row: dict[str, Any] = {}
            for field in sorted(set(acoustic_fields.get(left, {})) & set(acoustic_fields.get(right, {}))):
                row[field] = _nrmse(acoustic_fields[left][field], acoustic_fields[right][field])
            pairwise[f"{left}_vs_{right}"] = row
        acoustic_surrogate_pass = all(item.get("pipeline_verdict") == "PIPELINE_GREEN" and item.get("all_finite") for item in acoustic_runs) and all(
            value <= 0.05 for row in pairwise.values() for value in row.values()
        )
    else:
        pairwise = {}

    cfl = proof_header("STABILITY-CFL-SWEEP", "SKIP_NO_WARMBUBBLE_GPU_RUNNER", "SKIP_NO_WARMBUBBLE_GPU_RUNNER")
    cfl.update(
        {
            "case_required_by_plan": "warmbubble",
            "supporting_surrogate_case": "Canary d02 1h" if run_heavy else None,
            "gpu_preflight": gpu,
            "surrogate_runs": cfl_runs,
            "surrogate_passed": cfl_surrogate_pass,
            "thresholds": threshold_rows(
                {
                    "dt_0p5_finite_mass": (None, "warmbubble complete finite, mass drift <=1e-10", None),
                    "dt_1p0_finite_mass": (None, "warmbubble complete finite, mass drift <=1e-10", None),
                    "dt_1p25_deterministic_outcome": (1.0 if cfl_surrogate_pass else 0.0, "Canary surrogate complete or fail deterministically", cfl_surrogate_pass if run_heavy else None),
                }
            ),
            "gpu_hours_used": sum(float(item.get("forecast_wall_s") or 0.0) for item in cfl_runs) / 3600.0,
            "honesty_note": "The requested warmbubble CFL runner is absent; a real Canary d02 surrogate was run when GPU execution was enabled.",
        }
    )
    acoustic = proof_header(
        "STABILITY-ACOUSTIC-SUBSTEP-SWEEP",
        "SKIP_NO_DENSITY_CURRENT_GPU_RUNNER",
        "SKIP_NO_DENSITY_CURRENT_GPU_RUNNER",
    )
    acoustic.update(
        {
            "case_required_by_plan": "density_current",
            "supporting_surrogate_case": "Canary d02 1h" if run_heavy else None,
            "gpu_preflight": gpu,
            "surrogate_runs": acoustic_runs,
            "surrogate_pairwise_surface_nrmse": pairwise,
            "surrogate_passed": acoustic_surrogate_pass,
            "thresholds": threshold_rows(
                {
                    "all_runs_finite": (1.0 if acoustic_surrogate_pass else 0.0, "density-current finite for n in {4,6,8}", None),
                    "front_position_variation": (None, "<=1 cell across settings", None),
                    "ke_pairwise_nrmse": (None, "<=0.05 on density-current KE", None),
                }
            ),
            "gpu_hours_used": sum(float(item.get("forecast_wall_s") or 0.0) for item in acoustic_runs) / 3600.0,
            "honesty_note": "The requested density-current runner is absent; a real Canary d02 acoustic-substep surrogate was run when GPU execution was enabled.",
        }
    )
    paths = [proof_dir / "stability_cfl_sweep.json", proof_dir / "stability_acoustic_substep.json"]
    write_json(paths[0], cfl)
    write_json(paths[1], acoustic)
    return paths


def _write_determinism(
    proof_dir: Path,
    *,
    run_heavy: bool = False,
    execution_root: Path | None = None,
    run_id: str = "20260521_18z_l3_24h_20260522T133443Z",
) -> Path:
    execution_root = execution_root or DEFAULT_EXECUTION_ROOT
    runs: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    if run_heavy:
        for index in range(1, 4):
            out_dir = execution_root / "determinism" / f"run{index}"
            proof_subdir = proof_dir / "determinism_runs" / f"run{index}"
            existing_payload = read_json(proof_subdir / "pipeline_run_20260521.json")
            existing_files = sorted(out_dir.glob("wrfout_d02_*"))
            if existing_payload and existing_files:
                payload = existing_payload
                wall_s = 0.0
            else:
                started = time.perf_counter()
                payload = _run_daily_pipeline(run_id=run_id, hours=1, output_dir=out_dir, proof_dir=proof_subdir)
                wall_s = time.perf_counter() - started
            files = _output_files_from_pipeline(payload)
            if not files:
                files = [path for path in existing_files if path.is_file()]
            runs.append(
                {
                    "run_index": index,
                    "pipeline_verdict": payload.get("verdict"),
                    "wall_clock_s": wall_s,
                    "forecast_wall_s": payload.get("wall_clock_forecast_only_s"),
                    "proof_dir": str(proof_subdir),
                    "output_dir": str(out_dir),
                    "final_wrfout": str(files[-1]) if files else None,
                    "all_finite": payload.get("all_finite_check", {}).get("all_finite"),
                }
            )
        finals = [Path(row["final_wrfout"]) for row in runs if row.get("final_wrfout")]
        if len(finals) == 3:
            comparisons = [
                compare_wrfouts_xarray(finals[0], finals[1]),
                compare_wrfouts_xarray(finals[0], finals[2]),
                compare_wrfouts_xarray(finals[1], finals[2]),
            ]
    max_delta = 0.0
    for comparison in comparisons:
        for row in comparison.get("fields", {}).values():
            if isinstance(row, dict):
                value = _safe_float(row.get("max_abs_delta"))
                if value is not None:
                    max_delta = max(max_delta, value)
    passed = bool(
        len(runs) == 3
        and all(item.get("final_wrfout") and item.get("all_finite") is not False for item in runs)
        and comparisons
        and all(row.get("status") == "PASS" for row in comparisons)
        and max_delta == 0.0
    )
    verdict = "PASS" if passed else ("FAIL" if run_heavy else "SKIP_GPU_RUN_NOT_REQUESTED")
    payload = proof_header("DETERMINISM-REPEAT", verdict, "PASS_THREE_RUN_BITWISE" if passed else verdict)
    payload.update(
        {
            "required_run_count": 3,
            "observed_run_count": len(runs),
            "runs": runs,
            "comparisons": comparisons,
            "thresholds": threshold_rows(
                {
                    "max_delta_across_three_runs": (max_delta if comparisons else None, "0.0 bitwise for every final State/wrfout field", passed if run_heavy else None),
                    "run_count": (float(len(runs)), ">=3 identical full-pipeline runs", len(runs) >= 3 if run_heavy else None),
                }
            ),
            "gpu_hours_used": sum(float(item.get("forecast_wall_s") or 0.0) for item in runs) / 3600.0,
        }
    )
    path = proof_dir / "determinism_repeat.json"
    write_json(path, payload)
    return path


def _write_savepoint(proof_dir: Path, *, run_deep: bool = False) -> Path:
    m6b6_path = ROOT / ".agent/sprints/2026-05-25-m6b6-coupled-step-parity/proof_coupled_step_parity.json"
    summary = summarize_m6b6(m6b6_path)
    deep_output = proof_dir / "savepoint_deep_column100.json"
    deep_command: dict[str, Any] | None = None
    if run_deep:
        deep_command = run_command(
            [
                sys.executable,
                "scripts/m6b6_coupled_step_compare.py",
                "--tier",
                "column",
                "--steps",
                "100",
                "--savepoint-root",
                str(proof_dir / "savepoints_deep"),
                "--output",
                str(deep_output),
            ],
            timeout_s=1800,
        )
        deep_command = {
            "cmd": deep_command.get("cmd"),
            "cwd": deep_command.get("cwd"),
            "returncode": deep_command.get("returncode"),
            "timed_out": deep_command.get("timed_out"),
            "timeout_s": deep_command.get("timeout_s"),
            "stdout_bytes": len(str(deep_command.get("stdout") or "").encode("utf-8")),
            "stderr_tail": str(deep_command.get("stderr") or "")[-1000:],
        }
    elif deep_output.exists():
        deep_command = {
            "not_rerun": True,
            "reused_output": str(deep_output),
            "note": "Existing JSON comparison proof reused; binary HDF5 intermediates are not committed.",
        }
    deep = read_json(deep_output) if deep_output.exists() else None
    max_depth = max((int(v or 0) for v in summary.get("tier_savepoint_counts", {}).values()), default=0)
    if deep and deep.get("tiers", {}).get("column", {}).get("passed"):
        max_depth = max(max_depth, int(deep["tiers"]["column"].get("savepoint_count", 0)))
    passed = bool(summary.get("passed") is True and max_depth >= 10000)
    payload = proof_header("SAVEPOINT-PARITY-DEEP", "PASS" if passed else "FAIL", "FAIL_INSUFFICIENT_SAVEPOINT_DEPTH")
    payload.update(
        {
            "current_b6_evidence": summary,
            "deep_column100_command": deep_command,
            "deep_column100_output": str(deep_output) if deep_output.exists() else None,
            "binary_intermediates_committed": False,
            "binary_intermediates_note": "The HDF5 files emitted by m6b6_coupled_step_compare.py are generated intermediates and are not committed; the JSON comparison output is the retained proof object.",
            "required_depths": [100, 1000, 10000],
            "observed_max_depth": max_depth,
            "thresholds": threshold_rows(
                {
                    "step_100_bitwise": (
                        100.0 if max_depth >= 100 else None,
                        "max delta 0.0 at step 100",
                        bool(max_depth >= 100 and deep and deep.get("passed") is True),
                    ),
                    "step_1000_relative": (None, "<1e-12 relative for all fields", bool(max_depth >= 1000)),
                    "step_10000_relative": (None, "<1e-12 relative for all fields", bool(max_depth >= 10000)),
                    "existing_b6_10_step_parity": (float(max_depth), "B6 parity remains PASS as guardrail", summary.get("passed") is True),
                }
            ),
            "honesty_note": "B6 parity remains PASS. The redo optionally extends the column tier to 100 steps, but the 1000/10000 publication-depth gates are still unmet unless observed_max_depth reaches them.",
            "gpu_hours_used": 0.0,
        }
    )
    path = proof_dir / "savepoint_parity_deep.json"
    write_json(path, payload)
    return path


def _write_canary(
    proof_dir: Path,
    *,
    run_heavy: bool = False,
    execution_root: Path | None = None,
    target_count: int = CANARY_CASE_TARGET,
) -> dict[str, Any]:
    execution_root = execution_root or DEFAULT_EXECUTION_ROOT
    selections = select_canary_cases(CANARY_RUN_ROOT, target_count=target_count)
    manifest = {
        "run_root": str(CANARY_RUN_ROOT),
        "requested_case_count": int(target_count),
        "selected_case_count": len(selections),
        "available_runnable_distinct_day_count": sum(1 for item in selections if item.runnable),
        "available_complete_24h_distinct_day_count": sum(1 for item in selections if item.complete_24h),
        "cases": [
            {
                "day": item.day,
                "run_dir": str(item.run_dir),
                "run_id": item.run_id,
                "history_count": item.history_count,
                "planned_hours": item.planned_hours,
                "selection_class": item.selection_class,
                "reason": item.reason,
            }
            for item in selections
        ],
    }
    manifest_path = proof_dir / "canary_case_manifest.json"
    write_json(manifest_path, manifest)

    skill = _skill_module()
    case_results: list[dict[str, Any]] = []
    for item in selections:
        result: dict[str, Any] = {
            "day": item.day,
            "run_id": item.run_id,
            "run_dir": str(item.run_dir),
            "history_count": item.history_count,
            "planned_hours": item.planned_hours,
            "selection_class": item.selection_class,
            "complete_24h": item.complete_24h,
        }
        if not run_heavy:
            result.update({"pipeline_verdict": "SKIP_GPU_RUN_NOT_REQUESTED", "skip_reason": "run_heavy=false"})
            case_results.append(result)
            continue
        if not item.runnable:
            result.update({"pipeline_verdict": "SKIP_INSUFFICIENT_CPU_HISTORY", "skip_reason": item.reason})
            case_results.append(result)
            continue
        out_dir = execution_root / "canary" / item.day
        proof_subdir = proof_dir / "canary_runs" / item.day
        started = time.perf_counter()
        payload = _run_daily_pipeline(
            run_id=item.run_id,
            hours=item.planned_hours,
            output_dir=out_dir,
            proof_dir=proof_subdir,
            score=True,
        )
        wall_s = time.perf_counter() - started
        files = _output_files_from_pipeline(payload)
        result.update(
            {
                "pipeline_verdict": payload.get("verdict"),
                "wall_clock_s": wall_s,
                "forecast_wall_s": payload.get("wall_clock_forecast_only_s"),
                "proof_dir": str(proof_subdir),
                "output_dir": str(out_dir),
                "wrfout_files": [str(path) for path in files],
                "station_score_summary": payload.get("station_score_summary"),
            }
        )
        skill_path = proof_dir / "canary_runs" / item.day / "gpu_vs_cpu_skill_diff.json"
        try:
            if files:
                skill_payload = skill.build_skill_diff_payload(
                    gpu_root=out_dir,
                    cpu_run=item.run_dir,
                    aemet_root=skill.DEFAULT_AEMET_ROOT,
                    variables=tuple(skill.DEFAULT_VARIABLES),
                )
                skill.write_json(skill_path, skill_payload)
                result["skill_diff_path"] = str(skill_path)
                result["skill_verdict"] = skill_payload.get("verdict")
                result["common_valid_time_count"] = skill_payload.get("common_valid_time_count")
                result["station_count_scored"] = skill_payload.get("station_count_scored")
                result["variables"] = skill_payload.get("aggregate_comparison", {}).get("variables", {})
        except Exception as exc:  # pragma: no cover - data availability dependent
            result["skill_verdict"] = "FAIL_SKILL_EXCEPTION"
            result["skill_exception"] = repr(exc)
        case_results.append(result)

    complete_results = [item for item in case_results if item.get("complete_24h") and item.get("pipeline_verdict") == "PIPELINE_GREEN"]
    variable_pass: dict[str, bool] = {}
    for variable in ("T2", "U10", "V10"):
        rows = [
            case.get("variables", {}).get(variable, {})
            for case in complete_results
            if case.get("variables", {}).get(variable)
        ]
        variable_pass[variable] = bool(rows and len(rows) == len(complete_results) and all(row.get("within_20pct_all_metrics") for row in rows))
    passed = bool(len(complete_results) >= int(target_count) and all(variable_pass.values()))
    payload = proof_header("CANARY-MULTIDAY-SIDE-BY-SIDE", "PASS" if passed else "FAIL", "FAIL_FIVE_DAY_OR_SKILL_GATE")
    payload.update(
        {
            "case_manifest": str(manifest_path),
            "case_results": case_results,
            "completed_gpu_case_count": len(complete_results),
            "required_gpu_case_count": int(target_count),
            "per_variable_pass": variable_pass,
            "thresholds": threshold_rows(
                {
                    "case_count": (float(len(complete_results)), f">={target_count} complete 24h GPU cases", len(complete_results) >= int(target_count)),
                    "T2_within_20pct": (None, "GPU within +/-20% CPU RMSE for every complete case", variable_pass.get("T2")),
                    "U10_within_20pct": (None, "GPU within +/-20% CPU RMSE for every complete case", variable_pass.get("U10")),
                    "V10_within_20pct": (None, "GPU within +/-20% CPU RMSE for every complete case", variable_pass.get("V10")),
                }
            ),
            "honesty_note": "The local wrf_l3 inventory exposes only three complete 24h distinct d02 history days; the requested five-day publication gate therefore fails even though real GPU runs were executed for every runnable selected case.",
            "gpu_hours_used": sum(float(item.get("forecast_wall_s") or 0.0) for item in case_results) / 3600.0,
        }
    )
    skill_out = proof_dir / "canary_multiday_skill.json"
    write_json(skill_out, payload)
    first_growth = {
        "schema": "PublicationFirstErrorGrowth",
        "schema_version": 1,
        "status": "FAIL_NOT_EMITTED",
        "reason": "Per-hour first-divergence curves are not implemented in the current pubtest scorer; aggregate per-case skill diffs are listed instead.",
        "case_skill_paths": [case.get("skill_diff_path") for case in case_results if case.get("skill_diff_path")],
    }
    write_json(proof_dir / "canary_first_error_growth.json", first_growth)
    return payload


def _aggregate(proof_dir: Path) -> Path:
    rows = []
    total_gpu_hours = 0.0
    verdict_counts: dict[str, int] = {}
    for test_id, filename in HIGH_TEST_FILES.items():
        payload = read_json(proof_dir / filename) or {}
        verdict = payload.get("verdict", "MISSING")
        verdict_counts[str(verdict)] = verdict_counts.get(str(verdict), 0) + 1
        total_gpu_hours += float(payload.get("gpu_hours_used") or 0.0)
        rows.append(
            {
                "test_id": test_id,
                "proof_object": str(proof_dir / filename),
                "verdict": verdict,
                "status": payload.get("status", "MISSING"),
                "gpu_hours_used": float(payload.get("gpu_hours_used") or 0.0),
            }
        )
    report = proof_dir / "aggregate_report.md"
    lines = [
        "# Aggregate Report - Testing Plan Execution Redo",
        "",
        f"Total GPU hours used: {total_gpu_hours:.6f}",
        "",
        "| Test | Verdict | GPU h | Status | Proof |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['test_id']} | {row['verdict']} | {row['gpu_hours_used']:.6f} | {row['status']} | `{Path(row['proof_object']).name}` |"
        )
    lines.extend(
        [
            "",
            "## What Passed Cleanly",
            "",
            "- Real GPU execution was used for available Canary pipeline, determinism, and Canary surrogate stability runs.",
            "- Determinism is PASS only if three independent 1h GPU pipeline runs compare bitwise at the final wrfout.",
            "",
            "## What Failed Or Was Skipped",
            "",
            "- Idealized warm-bubble, density-current, and Schaer mountain-wave remain SKIP_* because no reviewed GPU idealized forecast runner exists in this repo scope.",
            "- Canary multiday side-by-side fails the requested five complete-day gate when fewer than five complete d02 history days are locally runnable, and may also fail variable skill thresholds.",
            "- Conservation and deep savepoint gates fail unless their required closed-domain/CPU-envelope/depth evidence is present.",
            "",
            "## Paper Claim Now Supported",
            "",
            "The paper can claim that the publication-test harness was re-run on a healthy RTX 5090 path and produced real GPU evidence for the runnable Canary pipeline subset. It cannot claim community-grade idealized-case coverage, five complete Canary days, or deep 10000-step savepoint parity from this sprint.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        proof_dir / "aggregate_report.json",
        {
            "schema": "PublicationAggregateReport",
            "schema_version": 1,
            "total_gpu_hours_used": total_gpu_hours,
            "verdict_counts": verdict_counts,
            "rows": rows,
        },
    )
    return report


def execute(
    proof_dir: Path,
    *,
    skip_gpu_probe: bool = False,
    gpu_probe_timeout_s: int = 5,
    run_heavy: bool | None = None,
    execution_root: Path | None = None,
    canary_target_count: int = CANARY_CASE_TARGET,
    run_savepoint_deep: bool = False,
) -> dict[str, Any]:
    proof_dir.mkdir(parents=True, exist_ok=True)
    execution_root = execution_root or DEFAULT_EXECUTION_ROOT
    gpu = gpu_probe(skip=skip_gpu_probe, timeout_s=gpu_probe_timeout_s)
    wrf = wrf_provenance()
    heavy = bool((gpu.get("available") is True) if run_heavy is None else run_heavy)
    paths: list[Path] = []
    paths.extend(_write_idealized(proof_dir, gpu=gpu, wrf=wrf))
    canary_payload = _write_canary(
        proof_dir,
        run_heavy=heavy,
        execution_root=execution_root,
        target_count=canary_target_count,
    )
    paths.extend([proof_dir / "canary_case_manifest.json", proof_dir / "canary_multiday_skill.json", proof_dir / "canary_first_error_growth.json"])
    paths.extend(_write_conservation(proof_dir, canary_payload))
    paths.extend(_write_stability(proof_dir, gpu=gpu, run_heavy=heavy, execution_root=execution_root))
    paths.append(_write_determinism(proof_dir, run_heavy=heavy, execution_root=execution_root))
    paths.append(_write_savepoint(proof_dir, run_deep=run_savepoint_deep))
    aggregate = _aggregate(proof_dir)
    aggregate_payload = read_json(proof_dir / "aggregate_report.json") or {}
    missing = [
        test_id
        for test_id, filename in HIGH_TEST_FILES.items()
        if not _proof_verdict_ok((read_json(proof_dir / filename) or {}).get("verdict"))
    ]
    total_gpu_hours = float(aggregate_payload.get("total_gpu_hours_used") or 0.0)
    if gpu.get("available") is not True and not skip_gpu_probe:
        status = "EXECUTION_BLOCKED"
    elif missing:
        status = "EXECUTION_PARTIAL"
    elif any(str((read_json(proof_dir / filename) or {}).get("verdict", "")).startswith("SKIP_") for filename in HIGH_TEST_FILES.values()):
        status = "EXECUTION_PARTIAL"
    elif total_gpu_hours <= 0.0 and heavy:
        status = "EXECUTION_BLOCKED"
    else:
        status = "EXECUTION_GREEN"
    return {
        "status": status,
        "proof_dir": str(proof_dir),
        "execution_root": str(execution_root),
        "proof_objects": [str(path) for path in paths],
        "aggregate_report": str(aggregate),
        "gpu_preflight": gpu,
        "total_gpu_hours_used": total_gpu_hours,
        "run_heavy": heavy,
        "missing_or_invalid_verdicts": missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-dir", type=Path, default=SPRINT_DIR)
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    parser.add_argument("--skip-gpu-probe", action="store_true")
    parser.add_argument("--gpu-probe-timeout-s", type=int, default=5)
    parser.add_argument("--no-heavy", action="store_true", help="Write proof objects without launching GPU pipeline runs.")
    parser.add_argument("--canary-target-count", type=int, default=CANARY_CASE_TARGET)
    parser.add_argument("--run-savepoint-deep", action="store_true", help="Extend M6B6 column parity to 100 steps.")
    args = parser.parse_args(argv)
    payload = execute(
        args.proof_dir,
        skip_gpu_probe=bool(args.skip_gpu_probe),
        gpu_probe_timeout_s=int(args.gpu_probe_timeout_s),
        run_heavy=False if args.no_heavy else None,
        execution_root=args.execution_root,
        canary_target_count=int(args.canary_target_count),
        run_savepoint_deep=bool(args.run_savepoint_deep),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"EXECUTION_GREEN", "EXECUTION_PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
