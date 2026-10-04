#!/usr/bin/env python3
"""Run ONE fresh 12-rank CPU-WRF arm of FAST-v025 and report every timing class.

Contract §5.3 requires cold compile, cached load, warm integration, CPU
integration, I/O, profiler perturbation, and lock wait to be reported
*separately* — "hiding one inside another invalidates the gate". For the CPU arm
the separable classes are staging, WRF start-up (input read, table load, first
step), integration, and I/O, so each is measured and emitted on its own.

Every invocation stages a brand-new run directory. Nothing is reused between
arms, which is what makes the contract's fresh-CPU rule mechanically enforceable
rather than a promise: a stale result cannot be produced by this script because
there is no directory for it to come from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_case import (  # noqa: E402
    FAST_ACTUAL_ADVANCE_SECONDS,
    FAST_DT_SECONDS,
    FAST_RANKS,
    FAST_SCHEDULED_WINDOW_SECONDS,
    FAST_START,
    MPIRUN,
    WRF_EXE,
    case_descriptor,
    sha256_file,
    stage_fast_case,
)
from wrf_timing import WRF_TIME_FMT, summarize  # noqa: E402

# The quiet 12-core envelope, chosen to avoid the cores carrying the active
# production/downscale jobs (contract §13). Recorded in every result so a
# repeat can reproduce the exact resource envelope.
DEFAULT_CPU_LIST = "16-27"

# Production launch flags, carried over verbatim from the ALISIOS launcher so the
# measurement method matches the runs the legacy scalars came from.
PRODUCTION_MPI_FLAGS = ["--use-hwthread-cpus", "--bind-to", "none"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cpu_jiffies() -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and len(line) > 3 and line[3].isdigit():
            fields = line.split()
            out[int(fields[0][3:])] = [int(x) for x in fields[1:]]
    return out


def _busy_percent(before: dict[int, list[int]], after: dict[int, list[int]]) -> dict[int, float]:
    busy: dict[int, float] = {}
    for cpu, first in before.items():
        delta = [b - a for a, b in zip(first, after[cpu])]
        total = sum(delta)
        idle = delta[3] + delta[4]
        busy[cpu] = 100.0 * (total - idle) / total if total else 0.0
    return busy


def _parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def contention_snapshot(
    before: dict[int, list[int]], after: dict[int, list[int]], cpu_list: str
) -> dict:
    """How busy the box was *during* this arm.

    A CPU timing repeat is only comparable to another if the machine state was
    comparable. Recording contention per arm is what lets a noisy repeat be
    discarded with evidence instead of silently widening the spread.
    """
    busy = _busy_percent(before, after)
    mine = _parse_cpu_list(cpu_list)
    others = [c for c in busy if c not in mine]
    load1, load5, load15 = (
        float(x) for x in Path("/proc/loadavg").read_text().split()[:3]
    )
    return {
        "loadavg": {"1m": load1, "5m": load5, "15m": load15},
        "own_cores": cpu_list,
        "own_cores_busy_mean_percent": (
            sum(busy[c] for c in mine) / len(mine) if mine else 0.0
        ),
        "foreign_cores_busy_mean_percent": (
            sum(busy[c] for c in others) / len(others) if others else 0.0
        ),
        "foreign_cores_busy_max_percent": max((busy[c] for c in others), default=0.0),
        "foreign_cores_over_50pct": sorted(c for c in others if busy[c] > 50.0),
        "per_core_busy_percent": {str(c): round(busy[c], 2) for c in sorted(busy)},
    }


def wrfout_digest(path: Path) -> dict:
    """Order-stable digest of a wrfout's numeric content.

    Hashing the file bytes would fold in netCDF layout and any header the
    library stamps; hashing variable data in sorted name order gives a digest
    that is stable across repeats iff the *physics* is deterministic, which is
    the property §5.4 actually asks for.
    """
    import netCDF4
    import numpy as np

    digest = hashlib.sha256()
    per_var: dict[str, str] = {}
    with netCDF4.Dataset(path) as ds:
        for name in sorted(ds.variables):
            var = ds.variables[name]
            data = np.asarray(var[:])
            if data.dtype.kind == "f":
                payload = np.ascontiguousarray(data, dtype=">f8").tobytes()
            elif data.dtype.kind in "iu":
                payload = np.ascontiguousarray(data, dtype=">i8").tobytes()
            else:
                payload = np.ascontiguousarray(data).tobytes()
            var_hash = hashlib.sha256(name.encode() + payload).hexdigest()
            per_var[name] = var_hash
            digest.update(var_hash.encode())
    return {"digest": digest.hexdigest(), "variables": len(per_var), "per_variable": per_var}


def finiteness_audit(path: Path) -> dict:
    """Zero non-finite values is a hard FAST-v025 gate (contract §5.3)."""
    import netCDF4
    import numpy as np

    offenders: dict[str, dict] = {}
    checked = 0
    with netCDF4.Dataset(path) as ds:
        for name in sorted(ds.variables):
            data = np.asarray(ds.variables[name][:])
            if data.dtype.kind != "f":
                continue
            checked += 1
            bad = ~np.isfinite(data)
            n_bad = int(bad.sum())
            if n_bad:
                offenders[name] = {"non_finite": n_bad, "size": int(data.size)}
    return {
        "float_variables_checked": checked,
        "non_finite_total": sum(v["non_finite"] for v in offenders.values()),
        "offenders": offenders,
        "pass": not offenders,
    }


_RSL_START_RE = re.compile(r"Timing for processing wrfinput file .*?:\s+([0-9.]+) elapsed")


def run_arm(
    *,
    run_root: Path,
    label: str,
    cpu_list: str = DEFAULT_CPU_LIST,
    ranks: int = FAST_RANKS,
    keep_outputs: bool = True,
    mpi_flags: list[str] | None = None,
    max_dom: int = 1,
    hours: int = 1,
) -> dict:
    """Stage, run, and reduce one fresh CPU-WRF arm.

    ``max_dom=2`` runs the legacy-probe variant (production 2-domain grid, same
    one-hour window) used to reconstruct the 86.45 s/model-hour scalar.
    """
    mpi_flags = PRODUCTION_MPI_FLAGS if mpi_flags is None else mpi_flags
    run_dir = run_root / label
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; every arm must be fresh")

    t_stage0 = time.perf_counter()
    staged = stage_fast_case(run_dir, max_dom=max_dom, hours=hours)
    stage_seconds = time.perf_counter() - t_stage0

    env = dict(os.environ)
    # Explicit thread counts: WRF dmpar is MPI-only, but an implicitly threaded
    # BLAS/OpenMP underneath would silently oversubscribe the 12-core envelope.
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    cmd = [
        "taskset",
        "-c",
        cpu_list,
        str(MPIRUN),
        *mpi_flags,
        "-np",
        str(ranks),
        str(WRF_EXE),
    ]

    started_at = _now()
    jiffies_before = _cpu_jiffies()
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=run_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wallclock_seconds = time.perf_counter() - t0
    jiffies_after = _cpu_jiffies()
    finished_at = _now()
    contention = contention_snapshot(jiffies_before, jiffies_after, cpu_list)

    (run_dir / "launcher_stdout.txt").write_text(proc.stdout)

    result: dict = {
        "label": label,
        "max_dom": max_dom,
        "hours": hours,
        "variant": "FAST-v025" if max_dom == 1 else "legacy-probe-2domain",
        "run_dir": str(run_dir),
        "command": cmd,
        "mpi_flags": mpi_flags,
        "cpu_list": cpu_list,
        "ranks": ranks,
        "returncode": proc.returncode,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "stage_seconds": stage_seconds,
        "launcher_wallclock_seconds": wallclock_seconds,
        "contention": contention,
        "namelist_sha256": staged.namelist_sha256,
        "source_namelist_sha256": staged.source_namelist_sha256,
        "host": platform.node(),
        "mpirun_realpath": str(MPIRUN.resolve()),
        "mpirun_sha256": sha256_file(MPIRUN.resolve()),
        "wrf_exe_realpath": str(WRF_EXE.resolve()),
        "wrf_exe_sha256": sha256_file(WRF_EXE.resolve()),
    }

    rsl = run_dir / "rsl.out.0000"
    if proc.returncode != 0 or not rsl.exists():
        result["status"] = "FAILED"
        result["tail"] = proc.stdout[-4000:]
        return result

    startup = _RSL_START_RE.search(rsl.read_text(errors="replace"))
    result["wrfinput_read_seconds"] = float(startup.group(1)) if startup else None

    scheduled = FAST_SCHEDULED_WINDOW_SECONDS * hours
    expected_steps = -(-scheduled // FAST_DT_SECONDS)
    expected_advance = expected_steps * FAST_DT_SECONDS
    timing = summarize(
        rsl,
        dt_seconds=FAST_DT_SECONDS,
        scheduled_window_seconds=scheduled,
        start_time=FAST_START,
        warmup_steps=1,
        wallclock_seconds=wallclock_seconds,
    )
    result["timing"] = timing

    d1 = timing["domain1"]
    window = d1["window"]
    if window["steps"] != expected_steps:
        result["status"] = "FAILED"
        result["error"] = (
            f"expected {expected_steps} domain-1 steps, got {window['steps']}"
        )
        return result
    if window["actual_model_advance_seconds"] != expected_advance:
        result["status"] = "FAILED"
        result["error"] = "model advance does not match the frozen FAST-v025 window"
        return result

    # Every requested domain must actually have integrated. A namelist that
    # gives d02 no end time leaves it with no steps, and the run still exits 0 --
    # so the domain count is checked against the timing stream, not the namelist.
    integrated = set(timing["domains_present"])
    expected_domains = set(range(1, max_dom + 1))
    if integrated != expected_domains:
        result["status"] = "FAILED"
        result["error"] = (
            f"max_dom={max_dom} requested domains {sorted(expected_domains)} but only "
            f"{sorted(integrated)} produced timing lines; a domain silently did not run"
        )
        return result
    if max_dom == 2:
        # d02 runs parent_time_step_ratio=3 substeps per d01 step.
        expected_d02 = window["steps"] * 3
        actual_d02 = timing.get("domain2_steps")
        if actual_d02 != expected_d02:
            result["status"] = "FAILED"
            result["error"] = (
                f"expected {expected_d02} domain-2 steps (3 per d01 step), got {actual_d02}"
            )
            return result

    outputs = sorted(run_dir.glob("wrfout_d01_*"))
    result["wrfout_files"] = [p.name for p in outputs]
    end_stamp = (FAST_START + timedelta(seconds=expected_advance)).strftime(
        "%Y-%m-%d_%H:%M:%S"
    )
    final = [p for p in outputs if p.name.endswith(end_stamp)]
    if not final:
        result["status"] = "FAILED"
        result["error"] = f"no end-of-window wrfout; got {[p.name for p in outputs]}"
        return result

    final_path = final[0]
    result["final_wrfout"] = final_path.name
    result["final_wrfout_sha256"] = sha256_file(final_path)
    result["finiteness"] = finiteness_audit(final_path)
    digest = wrfout_digest(final_path)
    result["content_digest"] = digest["digest"]
    result["content_digest_variables"] = digest["variables"]

    # The headline numbers, named so nothing is hidden inside anything else.
    result["timing_classes"] = {
        "stage_seconds": stage_seconds,
        "launcher_wallclock_seconds": wallclock_seconds,
        "wrfinput_read_seconds": result["wrfinput_read_seconds"],
        "integration_step_sum_seconds": d1["step_sum_seconds"],
        "first_step_seconds": d1["first_step_seconds"],
        "io_write_seconds": timing["io_seconds_total"],
        "launcher_minus_step_sum_seconds": wallclock_seconds - d1["step_sum_seconds"],
        "step_sum_per_fc_hour": d1["step_sum_per_fc_hour"],
        "step_sum_warm_per_fc_hour": d1["step_sum_warm_per_fc_hour"],
        "wallclock_per_fc_hour": timing["wallclock_per_fc_hour"],
        "step_sum_per_fc_hour_scheduled_window": d1[
            "step_sum_per_fc_hour_scheduled_window"
        ],
    }
    result["status"] = "OK"

    if not keep_outputs:
        for path in outputs:
            path.unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cpu-list", default=DEFAULT_CPU_LIST)
    parser.add_argument("--ranks", type=int, default=FAST_RANKS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--max-dom",
        type=int,
        default=1,
        choices=(1, 2),
        help="1 = FAST-v025; 2 = legacy 2-domain probe (86.45 reconstruction)",
    )
    parser.add_argument("--hours", type=int, default=1, help="forecast hours in the window")
    parser.add_argument(
        "--bind-to-core",
        action="store_true",
        help="documented low-variance variant instead of production --bind-to none",
    )
    args = parser.parse_args()

    flags = (
        ["--use-hwthread-cpus", "--bind-to", "core"]
        if args.bind_to_core
        else PRODUCTION_MPI_FLAGS
    )
    result = run_arm(
        run_root=args.run_root,
        label=args.label,
        cpu_list=args.cpu_list,
        ranks=args.ranks,
        mpi_flags=flags,
        max_dom=args.max_dom,
        hours=args.hours,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text if not args.out else json.dumps(result.get("timing_classes", result), indent=2, sort_keys=True))
    return 0 if result.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
