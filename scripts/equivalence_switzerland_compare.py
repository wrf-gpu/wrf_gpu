#!/usr/bin/env python3
"""Compare a GPU-port wrfout directory against a CPU-WRF reference directory.

Switzerland (Gotthard) NATIVE-INIT equivalence comparator. Unlike the in-place
replay demo (``equivalence_demo.py``), here the GPU and the CPU reference are
two SEPARATE forecast runs that started from the *same* ``wrfinput_d01`` +
``wrfbdy_d01`` (produced by ``real.exe`` from GFS — see
``scripts/build_switzerland_case.sh``). This is the honest experiment:

    same real.exe ICs/LBCs  ->  does the independent JAX GPU integrator
    reproduce CPU-WRF (Fortran) field-by-field within predeclared tolerance?

It is NOT a self-compare and NOT bitwise-vs-Fortran (two independent
integrators of the same PDE differ at round-off and diverge slowly under
chaotic dynamics; the question is whether they stay within operational tol).

The PREDECLARED per-field tolerances and the pooled-RMSE comparison engine are
REUSED verbatim from ``equivalence_demo.py`` (``FIELD_TOLERANCES``,
``FieldAccum``, ``_read_field``, ``_wrfout_time_label``) so the verdict logic is
identical to the validated Canary demo — only the I/O wiring (two separate dirs,
externally-measured wall clocks) differs.

USAGE
=====
    python scripts/equivalence_switzerland_compare.py \
        --gpu-dir   runs/switzerland/run_gpu \
        --cpu-dir   runs/switzerland/run_cpu \
        --gpu-wall-s 600 --cpu-wall-s 2520 \
        --out       proofs/v0120/equivalence_switzerland.json

``--gpu-wall-s`` / ``--cpu-wall-s`` are the measured forecast wall-clocks (the
orchestrator ``equivalence_switzerland.sh`` captures and passes them); if
omitted the speedup is reported as unavailable rather than fabricated.

Field ownership: this file + ``scripts/equivalence_switzerland.sh`` +
``scripts/build_switzerland_case.sh`` + ``docs/equivalence-switzerland.md``.
It imports the read-only comparison engine from ``equivalence_demo.py``; it does
not modify the pipeline, io, integration, or perf code.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Reuse the validated comparison engine + tolerances from the Canary demo.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from equivalence_demo import (  # noqa: E402
    FIELD_TOLERANCES,
    FieldAccum,
    _read_field,
    _wrfout_time_label,
    print_table,
)


def _parse_label_to_dt(label: str) -> datetime:
    return datetime.strptime(label, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def compare_dirs(gpu_dir: Path, cpu_dir: Path, domain: str) -> dict[str, Any]:
    """Compare each GPU wrfout against the same-timestamp CPU wrfout."""
    from netCDF4 import Dataset  # deferred (heavy import)

    gpu_files = sorted(gpu_dir.glob(f"wrfout_{domain}_*"))
    cpu_files = sorted(cpu_dir.glob(f"wrfout_{domain}_*"))
    if not gpu_files:
        raise SystemExit(f"no GPU wrfout_{domain}_* in {gpu_dir}")
    if not cpu_files:
        raise SystemExit(f"no CPU wrfout_{domain}_* in {cpu_dir}")

    cpu_by_label = {_wrfout_time_label(p): p for p in cpu_files}
    init_label = _wrfout_time_label(gpu_files[0])
    init_dt = _parse_label_to_dt(init_label)

    accums = {t.name: FieldAccum(t.name) for t in FIELD_TOLERANCES}
    compared_files: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for gpu_path in gpu_files:
        label = _wrfout_time_label(gpu_path)
        cpu_path = cpu_by_label.get(label)
        if cpu_path is None:
            skipped.append({"file": gpu_path.name, "reason": "no CPU reference of same timestamp"})
            continue
        lead_hours = (_parse_label_to_dt(label) - init_dt).total_seconds() / 3600.0
        file_fields: list[str] = []
        with Dataset(gpu_path) as gds, Dataset(cpu_path) as cds:
            for tol in FIELD_TOLERANCES:
                g = _read_field(gds, tol.name)
                c = _read_field(cds, tol.name)
                if g is None or c is None:
                    continue
                accums[tol.name].add_step(lead_hours, g, c)
                file_fields.append(tol.name)
        compared_files.append(
            {"file": gpu_path.name, "lead_hours": lead_hours, "fields_compared": file_fields}
        )

    field_summaries = [accums[t.name].summary() for t in FIELD_TOLERANCES]
    compared = [f for f in field_summaries if f["verdict"] != "NO_DATA"]
    exceedances = [f for f in compared if f["verdict"] == "EXCEEDS_TOL"]
    overall = (
        "EQUIVALENT"
        if compared and not exceedances
        else ("NOT_EQUIVALENT" if exceedances else "NO_DATA")
    )
    return {
        "overall_verdict": overall,
        "fields": field_summaries,
        "n_fields_compared": len(compared),
        "n_fields_exceeding_tol": len(exceedances),
        "exceeding_fields": [f["field"] for f in exceedances],
        "compared_files": compared_files,
        "skipped_files": skipped,
        "init_label": init_label,
    }


def build_speedup(
    gpu_wall_s: float | None,
    cpu_wall_s: float | None,
    cpu_build_label: str | None = None,
    cpu_ranks: int | None = None,
    hours: int | None = None,
) -> dict[str, Any]:
    """Honest speedup from externally-measured forecast wall clocks.

    The CPU build is whatever the orchestrator labels (``--cpu-build-label`` /
    ``--cpu-ranks``). For the v0.12.0 BIG-grid benchmark this is a **28-rank
    dmpar MPI** CPU-WRF (the project's honest denominator), NOT 1-core serial.
    The orchestrator measures the real wall-clock of each run directly and
    passes both; we label the basis plainly and never fabricate a ratio.
    """
    if gpu_wall_s is None or cpu_wall_s is None or gpu_wall_s <= 0:
        return {
            "status": "UNAVAILABLE",
            "reason": "gpu/cpu wall-clock not provided (pass --gpu-wall-s/--cpu-wall-s)",
            "gpu_wall_s": gpu_wall_s,
            "cpu_wall_s": cpu_wall_s,
        }
    label = cpu_build_label or (
        f"{cpu_ranks}-rank dmpar MPI gfortran" if cpu_ranks
        else "serial gfortran (single core)"
    )
    out: dict[str, Any] = {
        "status": "OK",
        "method": "measured end-to-end forecast wall clock (GPU JAX vs CPU-WRF)",
        "cpu_build": label,
        "cpu_ranks": cpu_ranks,
        "gpu_wall_s": float(gpu_wall_s),
        "cpu_wall_s": float(cpu_wall_s),
        "speedup": float(cpu_wall_s) / float(gpu_wall_s),
    }
    if hours:
        out["gpu_wall_per_fcst_hour_s"] = round(float(gpu_wall_s) / hours, 2)
        out["cpu_wall_per_fcst_hour_s"] = round(float(cpu_wall_s) / hours, 2)
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="equivalence_switzerland_compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gpu-dir", type=Path, required=True, help="Dir with GPU-port wrfout_<domain>_*")
    p.add_argument("--cpu-dir", type=Path, required=True, help="Dir with CPU-WRF wrfout_<domain>_*")
    p.add_argument("--domain", default="d01", help="Domain id (default d01; single-domain case).")
    p.add_argument("--hours", type=int, default=24, help="Forecast lead hours (for the report header).")
    p.add_argument("--gpu-wall-s", type=float, default=None, help="Measured GPU forecast wall (s).")
    p.add_argument("--cpu-wall-s", type=float, default=None, help="Measured CPU-WRF forecast wall (s).")
    p.add_argument(
        "--cpu-build-label",
        default=None,
        help="Honest label for the CPU build (e.g. '28-rank dmpar MPI gfortran').",
    )
    p.add_argument(
        "--cpu-ranks",
        type=int,
        default=None,
        help="MPI rank count of the CPU reference (28 for the big-grid benchmark).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("proofs/v0120/equivalence_switzerland.json"),
        help="Path for the verdict + stats proof JSON.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.gpu_dir.is_dir():
        print(f"error: --gpu-dir not found: {args.gpu_dir}", file=sys.stderr)
        return 2
    if not args.cpu_dir.is_dir():
        print(f"error: --cpu-dir not found: {args.cpu_dir}", file=sys.stderr)
        return 2

    comparison = compare_dirs(args.gpu_dir, args.cpu_dir, args.domain)
    speedup = build_speedup(
        args.gpu_wall_s,
        args.cpu_wall_s,
        cpu_build_label=args.cpu_build_label,
        cpu_ranks=args.cpu_ranks,
        hours=args.hours,
    )

    # Reuse the demo's table renderer (same look as the Canary demo).
    print_table(comparison, speedup, args.hours)

    payload = {
        "schema": "v0.12.0-equivalence-switzerland-2026-06-07",
        "case": "Gotthard / Central Switzerland (NON-Canary generalization)",
        "domain": args.domain,
        "hours": args.hours,
        "framing": (
            "Numerical/operational equivalence within predeclared per-field tolerances. "
            "GPU (JAX) and CPU-WRF both integrate the SAME real.exe wrfinput_d01 + wrfbdy_d01 "
            "(GFS-forced). NOT bitwise-vs-Fortran; NOT a self-compare. The speedup denominator "
            "is whatever build the orchestrator labels (for the v0.12.0 big-grid benchmark: "
            "28-rank dmpar MPI CPU-WRF, the HONEST denominator, same grid, per forecast hour). "
            "Like the Canary case, late-lead winds may exceed tol (chaotic divergence) -> "
            "NOT_EQUIVALENT is an honest, expected possibility; the value is a real, "
            "reproducible cross-code comparison on a new region."
        ),
        "gpu_dir": str(args.gpu_dir),
        "cpu_dir": str(args.cpu_dir),
        "tolerances": {t.name: {"rmse": t.rmse, "units": t.units, "kind": t.kind} for t in FIELD_TOLERANCES},
        "comparison": comparison,
        "speedup": speedup,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"  proof written: {args.out}")

    return 0 if comparison["overall_verdict"] in ("EQUIVALENT", "NOT_EQUIVALENT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
