#!/usr/bin/env python3
"""Self-serve GPU-vs-CPU-WRF equivalence demo for the wrf_gpu port.

WHAT THIS PROVES (and what it does NOT)
=======================================
This script lets a skeptic *run and check* that, given the SAME initial and
lateral-boundary conditions, the JAX GPU port reproduces a retained CPU-WRF
forecast field-by-field, grid-point-by-grid-point, hour-by-hour, within
PREDECLARED per-field tolerances.

It is an HONEST, REPRODUCIBLE, NON-self-compare test:

  * The GPU port is run through the validated REPLAY path
    (``python -m gpuwrf.cli run --input-dir <case> ... --domain d02``). In that
    path the GPU's initial state and lateral boundaries are taken from the SAME
    CPU-WRF ``wrfout`` history that we then compare against. So this is a
    genuine "same ICs/LBCs -> does the independent GPU integrator reproduce
    CPU-WRF?" experiment, NOT a JAX-vs-JAX self-compare and NOT a comparison of
    the model against its own output.

  * The reference is a *retained* CPU-WRF run on disk (Fortran WRF v4, built and
    run on CPU). No fresh CPU run is needed.

HONEST FRAMING (printed in the output and stamped into the proof JSON):

  * This is NUMERICAL / OPERATIONAL equivalence within a predeclared tolerance,
    NOT bitwise identity against Fortran WRF. Two independent integrators (a
    Fortran CPU code and a JAX GPU port) of the same PDE will differ at the
    bit/round-off level and will diverge slowly under chaotic dynamics; the
    question is whether they stay within an operationally meaningful tolerance.
  * The GPU port is separately *bitwise self-deterministic* (same inputs ->
    identical outputs run-to-run); that is asserted elsewhere in the suite and
    is NOT what this script measures.
  * Boundary-forced replay keeps the two solutions tied together at the domain
    edges, which is exactly the operational use case (short-range, LBC-driven
    regional NWP) and is the regime in which equivalence is claimed.

PREDECLARED PER-FIELD TOLERANCES (stated up front; see ``FIELD_TOLERANCES``)
===========================================================================
A field PASSES if its POOLED (all hours, all grid points) RMSE is at or below
the declared ``rmse`` tolerance. The tolerances below are *operational* limits
chosen to be comparable to (or tighter than) the run-to-run / IC-uncertainty
spread of CPU-WRF itself for a short boundary-forced regional forecast, NOT
round-off limits. They are deliberately fixed in this header so the verdict
cannot be moved to fit the data.

  Field    | RMSE tol            | rationale
  ---------+---------------------+-------------------------------------------
  T2       | 1.5 K               | 2 m temperature; sub-station-error
  U10/V10  | 1.5 m s-1           | 10 m wind components
  PSFC     | 120 Pa              | surface pressure (~0.1% of ~100 kPa)
  RAINNC   | 1.0 mm              | accumulated grid-scale precip
  T        | 1.5 K               | 3D perturbation potential temperature (theta-300)
  U / V    | 1.8 m s-1           | 3D horizontal wind components
  W        | 0.30 m s-1          | 3D vertical velocity (small magnitude, noisy)
  QVAPOR   | 1.0e-3 kg kg-1      | 3D water-vapour mixing ratio

The overall VERDICT is EQUIVALENT iff every compared field is at or below its
tolerance. Any exceedance is reported per-field with its numbers (a documented
tolerance with one honest exceedance is worth more than a hidden pass).

USAGE
=====
    python scripts/equivalence_demo.py \
        --case-dir /mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_<...> \
        --domain   d02 \
        --hours    24 \
        --out      proofs/v0120/equivalence_demo_20260509_d02.json

``--case-dir`` is a retained CPU-WRF run directory holding both the
``namelist.input`` and the hourly ``wrfout_<domain>_*`` history (the default
reference is the retained ``20260509_18z`` d02 case). The script runs the GPU
forecast into a temporary output dir, then compares against the CPU history in
the SAME case dir.

Field ownership: this file + ``docs/equivalence-demo.md`` only. It shells out to
the public CLI and reads NetCDF; it does not import or modify the pipeline,
io, integration, or perf code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# PREDECLARED tolerances (operational; pooled-RMSE gate). DO NOT tune to data. #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldTol:
    """Predeclared tolerance + metadata for one comparison field."""

    name: str
    rmse: float
    units: str
    kind: str  # "surface" or "column"


FIELD_TOLERANCES: tuple[FieldTol, ...] = (
    FieldTol("T2", 1.5, "K", "surface"),
    FieldTol("U10", 1.5, "m s-1", "surface"),
    FieldTol("V10", 1.5, "m s-1", "surface"),
    FieldTol("PSFC", 120.0, "Pa", "surface"),
    FieldTol("RAINNC", 1.0, "mm", "surface"),
    FieldTol("T", 1.5, "K", "column"),
    FieldTol("U", 1.8, "m s-1", "column"),
    FieldTol("V", 1.8, "m s-1", "column"),
    FieldTol("W", 0.30, "m s-1", "column"),
    FieldTol("QVAPOR", 1.0e-3, "kg kg-1", "column"),
)
FIELD_TOL_BY_NAME = {t.name: t for t in FIELD_TOLERANCES}

# WRF per-step CPU timing line for the d02 (domain 2) main solver.
_CPU_TIMING_RE = re.compile(
    r"Timing for main: time \S+ on domain\s+(?P<domain>\d+):\s+(?P<elapsed>[\d.]+) elapsed seconds"
)


# --------------------------------------------------------------------------- #
# Field comparison                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class FieldAccum:
    """Streaming accumulator for one field's pooled stats across all timesteps."""

    name: str
    sum_sq: float = 0.0      # sum of squared error (for pooled RMSE)
    sum_err: float = 0.0     # sum of error (for pooled bias)
    n: int = 0               # number of compared points
    max_abs: float = 0.0     # pooled max absolute difference
    per_step: list[dict[str, Any]] = field(default_factory=list)

    def add_step(self, lead_hours: float, gpu: np.ndarray, cpu: np.ndarray) -> None:
        gpu = np.asarray(gpu, dtype=np.float64)
        cpu = np.asarray(cpu, dtype=np.float64)
        if gpu.shape != cpu.shape:
            raise ValueError(
                f"{self.name}: shape mismatch GPU {gpu.shape} vs CPU {cpu.shape}"
            )
        err = gpu - cpu
        finite = np.isfinite(err)
        if not finite.all():
            err = err[finite]
        npts = int(err.size)
        if npts == 0:
            self.per_step.append(
                {"lead_hours": lead_hours, "n": 0, "rmse": None, "bias": None, "max_abs_diff": None}
            )
            return
        step_sq = float(np.sum(err * err))
        step_sum = float(np.sum(err))
        step_max = float(np.max(np.abs(err)))
        self.sum_sq += step_sq
        self.sum_err += step_sum
        self.n += npts
        self.max_abs = max(self.max_abs, step_max)
        self.per_step.append(
            {
                "lead_hours": lead_hours,
                "n": npts,
                "rmse": math.sqrt(step_sq / npts),
                "bias": step_sum / npts,
                "max_abs_diff": step_max,
            }
        )

    def summary(self) -> dict[str, Any]:
        tol = FIELD_TOL_BY_NAME[self.name]
        if self.n == 0:
            return {
                "field": self.name,
                "units": tol.units,
                "kind": tol.kind,
                "n_points": 0,
                "pooled_rmse": None,
                "pooled_bias": None,
                "pooled_max_abs_diff": None,
                "rmse_tol": tol.rmse,
                "verdict": "NO_DATA",
                "per_step": self.per_step,
            }
        pooled_rmse = math.sqrt(self.sum_sq / self.n)
        verdict = "PASS" if pooled_rmse <= tol.rmse else "EXCEEDS_TOL"
        return {
            "field": self.name,
            "units": tol.units,
            "kind": tol.kind,
            "n_points": self.n,
            "pooled_rmse": pooled_rmse,
            "pooled_bias": self.sum_err / self.n,
            "pooled_max_abs_diff": self.max_abs,
            "rmse_tol": tol.rmse,
            "verdict": verdict,
            "per_step": self.per_step,
        }


# --------------------------------------------------------------------------- #
# wrfout time parsing / matching                                              #
# --------------------------------------------------------------------------- #
def _wrfout_time_label(path: Path) -> str:
    """Return the ``YYYY-MM-DD_HH:MM:SS`` time portion of a wrfout filename."""
    name = path.name
    # wrfout_d02_2026-05-09_18:00:00
    m = re.search(r"wrfout_d\d+_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})", name)
    if not m:
        raise ValueError(f"cannot parse time from wrfout name: {name}")
    return m.group(1)


def _parse_label_to_dt(label: str) -> datetime:
    return datetime.strptime(label, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def _read_field(ds: Any, name: str) -> np.ndarray | None:
    """Read a wrfout field (drop the leading singleton Time axis). None if absent."""
    if name not in ds.variables:
        return None
    arr = np.asarray(ds.variables[name][:], dtype=np.float64)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


# --------------------------------------------------------------------------- #
# CPU-WRF speedup baseline (from retained RSL timing logs; d02 main solver)    #
# --------------------------------------------------------------------------- #
def cpu_d02_wall_for_hours(case_dir: Path, hours: float) -> dict[str, Any]:
    """Estimate the CPU-WRF wall time to integrate ``hours`` of the d02 solver.

    Uses the retained ``rsl.error.0000`` / ``rsl.out.0000`` per-step "Timing for
    main ... on domain 2" records: mean per-step wall * (number of d02 steps in
    ``hours``). This isolates the d02 main-solver cost (the apples-to-apples
    comparand for the GPU d02 forecast), independent of the parent d01 nest.
    """
    times: list[datetime] = []
    elapsed: list[float] = []
    src_files: list[str] = []
    for fname in ("rsl.error.0000", "rsl.out.0000", "namelist.output"):
        fpath = case_dir / fname
        if not fpath.is_file():
            continue
        text = fpath.read_text(encoding="utf-8", errors="replace")
        found = False
        for m in _CPU_TIMING_RE.finditer(text):
            if int(m.group("domain")) != 2:
                continue
            elapsed.append(float(m.group("elapsed")))
            found = True
        # also collect d02 timestamps for the model step inference
        for tm in re.finditer(
            r"Timing for main: time (\S+) on domain\s+2:", text
        ):
            try:
                times.append(_parse_label_to_dt(tm.group(1)))
            except ValueError:
                pass
        if found:
            src_files.append(str(fpath))
        if elapsed:
            break  # rsl.error and rsl.out duplicate; one source is enough

    if not elapsed:
        return {
            "status": "NO_BASELINE",
            "reason": "no d02 'Timing for main' records in rsl logs",
            "cpu_wall_s": None,
        }

    # Infer the d02 model step (seconds) from consecutive distinct timestamps.
    dt_s = None
    uniq = sorted(set(times))
    deltas = [
        (b - a).total_seconds() for a, b in zip(uniq[:-1], uniq[1:]) if (b - a).total_seconds() > 0
    ]
    if deltas:
        dt_s = float(np.median(deltas))
    mean_step = float(np.mean(elapsed))
    if dt_s and dt_s > 0:
        n_steps = hours * 3600.0 / dt_s
        cpu_wall = mean_step * n_steps
        method = "mean d02 per-step wall * step-count for the lead"
    else:
        n_steps = float(len(elapsed))
        cpu_wall = mean_step * n_steps
        method = "mean d02 per-step wall * recorded step count (dt unknown)"
    return {
        "status": "OK",
        "method": method,
        "source_files": src_files,
        "d02_model_step_s": dt_s,
        "mean_step_wall_s": mean_step,
        "median_step_wall_s": float(np.median(elapsed)),
        "n_steps_recorded": int(len(elapsed)),
        "n_steps_for_lead": float(n_steps),
        "hours": float(hours),
        "cpu_wall_s": float(cpu_wall),
    }


# --------------------------------------------------------------------------- #
# Run the GPU forecast through the public CLI (replay path)                    #
# --------------------------------------------------------------------------- #
def run_gpu_forecast(
    case_dir: Path,
    domain: str,
    hours: int,
    output_dir: Path,
    scratch_dir: Path,
) -> dict[str, Any]:
    """Invoke ``python -m gpuwrf.cli run`` (replay path) and return its payload + wall."""
    cmd = [
        sys.executable,
        "-m",
        "gpuwrf.cli",
        "run",
        "--input-dir",
        str(case_dir),
        "--output-dir",
        str(output_dir),
        "--scratch-dir",
        str(scratch_dir),
        "--domain",
        domain,
        "--hours",
        str(hours),
    ]
    print(f"[equivalence-demo] GPU run: {' '.join(cmd)}", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.stderr:
        # Surface the CLI's mode/diagnostic lines for transparency.
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    if proc.returncode != 0:
        raise RuntimeError(
            f"GPU CLI run failed (exit {proc.returncode}). "
            f"stdout tail:\n{proc.stdout[-2000:]}"
        )
    # The CLI prints exactly one JSON payload as the last stdout block.
    payload = _parse_last_json(proc.stdout)
    payload["_subprocess_wall_s"] = wall
    return payload


def _parse_last_json(text: str) -> dict[str, Any]:
    """Extract the final top-level JSON object printed by the CLI."""
    # CLI uses json.dumps(indent=2); the payload starts at the last line that is
    # exactly "{" at column 0 and runs to the final "}".
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.rstrip() == "{":
            start = i
    if start is None:
        # fall back: try to find the last '{' ... '}' span
        last_open = text.rfind("\n{")
        if last_open == -1:
            raise ValueError(f"no JSON payload found in CLI stdout:\n{text[-2000:]}")
        return json.loads(text[last_open + 1 :])
    return json.loads("\n".join(lines[start:]))


# --------------------------------------------------------------------------- #
# Compare GPU vs CPU wrfout files                                             #
# --------------------------------------------------------------------------- #
def compare_outputs(
    gpu_files: Sequence[Path],
    case_dir: Path,
    init_label: str,
) -> dict[str, Any]:
    """Compare each GPU wrfout against the same-basename CPU wrfout in case_dir."""
    from netCDF4 import Dataset  # deferred

    accums = {t.name: FieldAccum(t.name) for t in FIELD_TOLERANCES}
    init_dt = _parse_label_to_dt(init_label)
    compared_files: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for gpu_path in gpu_files:
        try:
            label = _wrfout_time_label(gpu_path)
        except ValueError as exc:
            skipped.append({"file": str(gpu_path), "reason": str(exc)})
            continue
        cpu_path = case_dir / gpu_path.name
        if not cpu_path.is_file():
            skipped.append({"file": gpu_path.name, "reason": "no CPU reference of same basename"})
            continue
        lead_hours = (_parse_label_to_dt(label) - init_dt).total_seconds() / 3600.0
        file_fields: list[str] = []
        with Dataset(gpu_path) as gds, Dataset(cpu_path) as cds:
            for tol in FIELD_TOLERANCES:
                gpu_arr = _read_field(gds, tol.name)
                cpu_arr = _read_field(cds, tol.name)
                if gpu_arr is None or cpu_arr is None:
                    continue
                accums[tol.name].add_step(lead_hours, gpu_arr, cpu_arr)
                file_fields.append(tol.name)
        compared_files.append(
            {
                "file": gpu_path.name,
                "lead_hours": lead_hours,
                "fields_compared": file_fields,
            }
        )

    field_summaries = [accums[t.name].summary() for t in FIELD_TOLERANCES]
    compared = [f for f in field_summaries if f["verdict"] != "NO_DATA"]
    exceedances = [f for f in compared if f["verdict"] == "EXCEEDS_TOL"]
    overall = "EQUIVALENT" if compared and not exceedances else (
        "NOT_EQUIVALENT" if exceedances else "NO_DATA"
    )
    return {
        "overall_verdict": overall,
        "fields": field_summaries,
        "n_fields_compared": len(compared),
        "n_fields_exceeding_tol": len(exceedances),
        "exceeding_fields": [f["field"] for f in exceedances],
        "compared_files": compared_files,
        "skipped_files": skipped,
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _fmt(v: Any, prec: int = 4) -> str:
    if v is None:
        return "    n/a"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.3e}"
        return f"{v:.{prec}f}"
    return str(v)


def print_table(comparison: dict[str, Any], speedup: dict[str, Any], hours: int) -> None:
    print("\n" + "=" * 78)
    print("  GPU vs CPU-WRF EQUIVALENCE DEMO  --  numerical/operational equivalence")
    print("  (NOT bitwise vs Fortran; same ICs/LBCs replay; not a self-compare)")
    print("=" * 78)
    header = f"  {'field':9s} {'units':10s} {'pooled_RMSE':>12s} {'tol':>10s} "
    header += f"{'bias':>12s} {'max|diff|':>12s}  verdict"
    print(header)
    print("  " + "-" * 74)
    for f in comparison["fields"]:
        if f["verdict"] == "NO_DATA":
            print(f"  {f['field']:9s} {f['units']:10s} {'(not in output)':>12s}")
            continue
        line = (
            f"  {f['field']:9s} {f['units']:10s} "
            f"{_fmt(f['pooled_rmse']):>12s} {_fmt(f['rmse_tol']):>10s} "
            f"{_fmt(f['pooled_bias']):>12s} {_fmt(f['pooled_max_abs_diff']):>12s}  "
            f"{f['verdict']}"
        )
        print(line)
    print("  " + "-" * 74)
    print(f"  OVERALL VERDICT: {comparison['overall_verdict']}  "
          f"({comparison['n_fields_compared']} fields compared, "
          f"{comparison['n_fields_exceeding_tol']} exceed tol)")
    if comparison["exceeding_fields"]:
        print(f"  EXCEEDING FIELDS: {', '.join(comparison['exceeding_fields'])}")
    print()
    if speedup.get("status") == "OK" and speedup.get("gpu_wall_s"):
        print(f"  SPEEDUP (d02, {hours} h forecast, GPU integrate vs CPU-WRF d02 solver):")
        print(f"    GPU forecast wall : {_fmt(speedup['gpu_wall_s'], 1)} s")
        print(f"    CPU-WRF d02 wall  : {_fmt(speedup['cpu_wall_s'], 1)} s "
              f"(from retained RSL timing)")
        print(f"    speedup           : {_fmt(speedup['speedup'], 2)} x")
    else:
        print(f"  SPEEDUP: unavailable ({speedup.get('reason', speedup.get('status'))})")
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="equivalence_demo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--case-dir",
        type=Path,
        default=Path("/mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z"),
        help="Retained CPU-WRF run dir (namelist + hourly wrfout history). "
        "Default = the retained 20260509_18z d02 reference case.",
    )
    p.add_argument("--domain", default="d02", help="Domain id to run/compare (e.g. d02).")
    p.add_argument("--hours", type=int, default=24, help="Forecast lead hours to run and compare.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("proofs/v0120/equivalence_demo_20260509_d02.json"),
        help="Path for the verdict + stats proof JSON.",
    )
    p.add_argument(
        "--gpu-output-dir",
        type=Path,
        default=None,
        help="Where to write the GPU wrfout (default: a temp dir, cleaned up).",
    )
    p.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Disk-backed scratch for the GPU run (default: a temp dir).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    case_dir: Path = args.case_dir
    domain: str = args.domain
    hours: int = args.hours

    if not case_dir.is_dir():
        print(f"equivalence_demo: --case-dir not found: {case_dir}", file=sys.stderr)
        return 2
    cpu_hist = sorted(case_dir.glob(f"wrfout_{domain}_*"))
    if len(cpu_hist) < 2:
        print(
            f"equivalence_demo: need >=2 retained CPU wrfout_{domain}_* in {case_dir} "
            f"(found {len(cpu_hist)})",
            file=sys.stderr,
        )
        return 2
    init_label = _wrfout_time_label(cpu_hist[0])

    # Temp dirs (cleaned on exit) unless the user pins them.
    tmp_ctx = None
    if args.gpu_output_dir is None or args.scratch_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="eqdemo_")
        tmp_root = Path(tmp_ctx.name)
    gpu_out = args.gpu_output_dir or (tmp_root / "gpu_out")
    scratch = args.scratch_dir or (tmp_root / "scratch")
    gpu_out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Run the GPU forecast (replay path) through the public CLI.
        payload = run_gpu_forecast(case_dir, domain, hours, gpu_out, scratch)
        gpu_files = [Path(p) for p in payload.get("wrfout_files", [])]
        gpu_files = [p for p in gpu_files if p.is_file()]
        if not gpu_files:
            print("equivalence_demo: GPU run produced no wrfout files", file=sys.stderr)
            return 1

        # GPU forecast-only wall (excludes JIT warm/IO setup where the pipeline
        # reports it; fall back to the subprocess wall if not present).
        gpu_wall = payload.get("wall_clock_forecast_only_s")
        if gpu_wall is None:
            gpu_wall = payload.get("wall_clock_total_s")
        if gpu_wall is None:
            gpu_wall = payload.get("_subprocess_wall_s")
        gpu_wall = float(gpu_wall) if gpu_wall is not None else None

        # 2. Compare GPU vs CPU field-by-field, all timesteps, all grid points.
        comparison = compare_outputs(gpu_files, case_dir, init_label)

        # 3. Speedup vs the retained CPU-WRF d02 solver baseline.
        lead_compared = max(
            (f["lead_hours"] for f in comparison["compared_files"]), default=float(hours)
        )
        speedup = cpu_d02_wall_for_hours(case_dir, lead_compared)
        if speedup.get("status") == "OK" and gpu_wall and gpu_wall > 0:
            speedup["gpu_wall_s"] = gpu_wall
            speedup["speedup"] = float(speedup["cpu_wall_s"]) / gpu_wall
        else:
            speedup["gpu_wall_s"] = gpu_wall
            speedup["speedup"] = None

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    # 4. Build, write, and print the proof.
    proof = {
        "schema": "GpuwrfEquivalenceDemo",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "framing": (
            "Numerical/operational equivalence of the JAX GPU port vs a retained CPU-WRF "
            "forecast under the SAME ICs/LBCs (validated replay path). NOT bitwise vs Fortran; "
            "NOT a JAX-vs-JAX self-compare. The GPU port is separately bitwise self-deterministic "
            "(asserted elsewhere, not measured here). Verdict is against PREDECLARED per-field "
            "pooled-RMSE tolerances stated in the script header."
        ),
        "case_dir": str(case_dir),
        "domain": domain,
        "hours_requested": hours,
        "init_time": init_label,
        "device": payload.get("device"),
        "init_mode": payload.get("init_mode"),
        "gpu_pipeline_verdict": payload.get("verdict"),
        "gpu_wall_clock_forecast_only_s": payload.get("wall_clock_forecast_only_s"),
        "gpu_wall_clock_total_s": payload.get("wall_clock_total_s"),
        "predeclared_tolerances": [
            {"field": t.name, "rmse_tol": t.rmse, "units": t.units, "kind": t.kind}
            for t in FIELD_TOLERANCES
        ],
        "comparison": comparison,
        "speedup": speedup,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proof, indent=2, sort_keys=True, default=str) + "\n")

    print_table(comparison, speedup, hours)
    print(f"[equivalence-demo] proof written to {args.out}", file=sys.stderr)

    # Exit non-zero only if a field exceeded tolerance OR nothing was comparable;
    # an honest EQUIVALENT verdict exits 0.
    if comparison["overall_verdict"] == "EQUIVALENT":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
